"""
accumulation_signals.py — 吸筹 / 出货 detection.

The question these answer is not "is money flowing in" but "is money flowing in
WHILE PRICE REFUSES TO MOVE" — absorption — versus "is price still rising WHILE
MONEY LEAVES" — distribution. Neither price nor flow alone separates the two;
the divergence between them does, conditioned on where price sits in its range.

    price low in range  + flow in,  price flat   →  吸筹
    price high in range + flow out, price firm   →  出货

WHEN, not just WHETHER
----------------------
Most of these are conditions over a window, not single-day events, so a bare
"fired / not fired" hides the thing you actually need to know. Every detector
here is therefore evaluated as a BOOLEAN TIME SERIES over the whole history and
summarised by run_info(): whether it is true today, how many consecutive
sessions it has been true, the date that run began, and how many times it fired
in the last 60 sessions. Point events (a failed breakout, a heavy-inflow down
day) come back as a list of dates instead.

Caveats worth carrying into any reading of the output
-----------------------------------------------------
`main_net` is Tushare's large + extra-large order flow, classified by ORDER
SIZE, not by who placed it. Institutions increasingly slice orders with
VWAP/TWAP algos precisely to avoid leaving this footprint, and a single large
retail order counts the same as a fund. It is a proxy with real leakage.

These patterns are descriptive. A stock can show every accumulation signature
and then drift sideways for a year. Nothing here has been tested for
expectancy — run candidates through backtest_signal_expectancy before treating
any of it as tradeable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A signal must hold this many sessions before it is worth showing, so a single
# noisy day doesn't light up the panel.
MIN_RUN = 2
RECENT_WINDOW = 5        # "fired recently" tolerance
DEFAULT_WINDOW = 20      # the standard evaluation window, ~1 trading month


# ── Timing summary ────────────────────────────────────────────────────────────

def run_info(flags: pd.Series, recent: int = RECENT_WINDOW) -> dict:
    """
    Turn a boolean series into an answer to "when did this happen?".

    Returns active (true on the latest bar), run (consecutive sessions true up
    to now), since (date that run started), last (most recent true date), and
    count_60 (times true in the last 60 sessions).
    """
    if flags is None or len(flags) == 0:
        return {"active": False, "recent": False, "run": 0,
                "since": None, "last": None, "count_60": 0, "dates": []}

    f = flags.fillna(False).astype(bool)
    vals = f.to_numpy()

    run = 0
    for v in vals[::-1]:
        if v:
            run += 1
        else:
            break

    true_idx = f.index[f]
    since = f.index[len(f) - run] if run else None

    return {
        "active": bool(vals[-1]),
        "recent": bool(f.tail(recent).any()),
        "run": int(run),
        "since": None if since is None else pd.Timestamp(since).strftime("%Y-%m-%d"),
        "last": None if not len(true_idx) else pd.Timestamp(true_idx[-1]).strftime("%Y-%m-%d"),
        "count_60": int(f.tail(60).sum()),
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in true_idx[-6:]],
    }


def _sig(key, label, cn, kind, flags, detail, value=None, available=True):
    """Package one detector's boolean series with its timing summary."""
    info = run_info(flags) if available else {
        "active": False, "recent": False, "run": 0,
        "since": None, "last": None, "count_60": 0, "dates": []}
    return {"key": key, "label": label, "cn": cn, "kind": kind,
            "detail": detail, "value": value, "available": available,
            "flags": flags if available else None, **info}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slope(s: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope, normalised by the series' own level so it compares."""
    def _f(a):
        if np.isnan(a).any():
            return np.nan
        x = np.arange(len(a))
        return np.polyfit(x, a, 1)[0]
    raw = s.rolling(window).apply(_f, raw=True)
    scale = s.abs().rolling(window).mean().replace(0, np.nan)
    return raw / scale


def _range_position(df: pd.DataFrame, lookback: int = 120) -> pd.Series:
    """Where price sits in its own N-day Donchian channel, 0 (low) to 1 (high)."""
    hi = df["High"].rolling(lookback, min_periods=20).max()
    lo = df["Low"].rolling(lookback, min_periods=20).min()
    return ((df["Close"] - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1)


def _zscore(s: pd.Series, window: int = 60) -> pd.Series:
    m = s.rolling(window, min_periods=20).mean()
    sd = s.rolling(window, min_periods=20).std()
    return (s - m) / sd.replace(0, np.nan)


# ── Detection ─────────────────────────────────────────────────────────────────

def detect(df: pd.DataFrame, main_net: "pd.Series | None" = None,
           window: int = DEFAULT_WINDOW) -> dict:
    """
    Evaluate every 吸筹 / 出货 detector.

    `df`       — analysis_df from run_single_stock_analysis (OHLCV + indicators)
    `main_net` — 主力净流入 series (万元), aligned on date. None disables the
                 four flow-based detectors, which are marked unavailable rather
                 than silently reported as not firing.

    Returns {"ok", "accumulation": [...], "distribution": [...], "window", ...}
    """
    if df is None or df.empty or len(df) < 40:
        return {"ok": False, "reason": "not enough history",
                "accumulation": [], "distribution": []}

    d = df.sort_index()
    ret = d["Close"].pct_change()
    pos = _range_position(d)
    up, down = ret > 0, ret < 0

    has_flow = main_net is not None and not main_net.empty
    mf = main_net.reindex(d.index) if has_flow else pd.Series(np.nan, index=d.index)
    mf_z = _zscore(mf) if has_flow else mf
    cum_flow = mf.fillna(0).cumsum() if has_flow else mf

    price_slope = _slope(d["Close"], window)
    flat_price = price_slope.abs() < 0.0015          # ~flat over the window

    acc: list[dict] = []
    dist: list[dict] = []

    # ── 吸筹 ──────────────────────────────────────────────────────────────────

    # The strongest single tell: price falls, large orders keep buying.
    down_flow = mf.where(down)
    n_down = down.rolling(window).sum()
    acc.append(_sig(
        "down_absorption", "Down-day absorption", "下跌日承接", "window",
        (down_flow.rolling(window, min_periods=5).mean() > 0) & (n_down >= 4),
        "On down days over the window, 主力 is still a net buyer — supply is "
        "being absorbed rather than chased.",
        value=None if not has_flow else
        f"{down_flow.rolling(window, min_periods=5).mean().iloc[-1]:,.0f} 万",
        available=has_flow))

    # Flow stops tracking price: money goes in regardless of the day's move.
    corr = ret.rolling(window).corr(mf) if has_flow else mf
    acc.append(_sig(
        "flow_decoupled", "Flow decoupled from price", "资金与价格脱钩", "window",
        (corr.abs() < 0.2) & (cum_flow.diff(window) > 0),
        "Normally flow and price move together. Near-zero correlation with "
        "cumulative flow still rising means buying is indifferent to price.",
        value=None if not has_flow else f"r={corr.iloc[-1]:+.2f}",
        available=has_flow))

    acc.append(_sig(
        "obv_divergence_bull", "OBV rising, price flat", "OBV背离(多)", "window",
        (_slope(d["OBV"], window) > 0.002) & flat_price,
        "Volume is accumulating on the buy side while price goes nowhere — the "
        "same idea as flow absorption, from volume alone.",
        value=None, available="OBV" in d.columns))

    acc.append(_sig(
        "vol_compression", "Volatility compression", "缩量横盘", "window",
        d.get("BB_Width_Percentile", pd.Series(np.nan, index=d.index)) < 0.25,
        "Bollinger width in the bottom quartile of its own post-anchor history "
        "— the range is tightening.",
        value=None if "BB_Width_Percentile" not in d.columns else
        f"{d['BB_Width_Percentile'].iloc[-1]:.0%} pct",
        available="BB_Width_Percentile" in d.columns))

    rising_floor = (d["Low"].rolling(10).min()
                    > d["Low"].rolling(10).min().shift(window)) & flat_price
    acc.append(_sig(
        "rising_floor", "Rising floor in a flat range", "重心上移", "window",
        rising_floor,
        "The 10-day low keeps stepping up while the range stays flat — sellers "
        "are being priced out.",
        value=None))

    acc.append(_sig(
        "low_in_range", "Low in its own range", "处于区间低位", "state",
        pos < 0.40,
        "Price sits in the bottom 40% of its 120-day channel. Accumulation "
        "means little at the top of a range.",
        value=f"{pos.iloc[-1]:.0%}" if pos.notna().any() else None))

    acc.append(_sig(
        "heavy_inflow_down_day", "Heavy inflow on a down day", "逆势大额流入", "event",
        (mf_z > 1.5) & down,
        "Individual sessions where price fell but 主力 inflow was unusually "
        "large. These are the footprints, dated.",
        value=None, available=has_flow))

    # ── 出货 ──────────────────────────────────────────────────────────────────

    dist.append(_sig(
        "flow_divergence_bear", "Price up, flow leaving", "量价背离(空)", "window",
        (price_slope > 0.0015) & (cum_flow.diff(window) < 0) & (pos > 0.5),
        "Price is still climbing while cumulative 主力 flow falls — someone is "
        "selling into the strength.",
        value=None if not has_flow else f"{cum_flow.diff(window).iloc[-1]:,.0f} 万",
        available=has_flow))

    dist.append(_sig(
        "outflow_on_up_days", "Outflow on up days", "上涨日净流出", "window",
        (mf.where(up).rolling(window, min_periods=4).mean() < 0)
        & (up.rolling(window).sum() >= 4) & (pos > 0.5),
        "On the days price rises, 主力 is a net seller — rallies are being "
        "used as exit liquidity.",
        value=None if not has_flow else
        f"{mf.where(up).rolling(window, min_periods=4).mean().iloc[-1]:,.0f} 万",
        available=has_flow))

    body = (d["Close"] - d["Open"]).abs()
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    dist.append(_sig(
        "volume_stall", "Heavy volume, no progress", "放量滞涨", "window",
        (d.get("Volume_ZScore", pd.Series(np.nan, index=d.index)) > 1.0)
        & ((body / rng).rolling(5).mean() < 0.35) & (pos > 0.5),
        "Volume well above normal while candles close near their open — a lot "
        "of trade producing no move is a transfer of ownership.",
        value=None, available="Volume_ZScore" in d.columns))

    upper = (d["High"] - d[["Open", "Close"]].max(axis=1)) / rng
    dist.append(_sig(
        "upper_shadows", "Repeated upper shadows", "长上影线", "window",
        (upper.rolling(5).mean() > 0.45) & (pos > 0.55),
        "Intraday pushes higher keep getting sold back before the close.",
        value=f"{upper.rolling(5).mean().iloc[-1]:.0%}" if upper.notna().any() else None))

    # Bearish RSI divergence: price makes a higher high, RSI does not.
    if "RSI_14" in d.columns:
        ph = d["Close"].rolling(40).max()
        rsi_at_hi = d["RSI_14"].rolling(40).max()
        bear_div = ((d["Close"] >= ph * 0.995)
                    & (d["RSI_14"] < rsi_at_hi - 5) & (pos > 0.6))
    else:
        bear_div = pd.Series(False, index=d.index)
    dist.append(_sig(
        "rsi_divergence", "RSI bearish divergence", "RSI顶背离", "window",
        bear_div,
        "Price at/near a 40-day high while RSI sits well below its own high "
        "for the period — momentum is not confirming.",
        value=None, available="RSI_14" in d.columns))

    prior_hi = d["High"].rolling(20).max().shift(1)
    dist.append(_sig(
        "failed_breakout", "Failed breakout", "冲高回落", "event",
        (d["High"] > prior_hi) & (d["Close"] < prior_hi) & (pos > 0.5),
        "Sessions that traded above the prior 20-day high but closed back "
        "below it. Dated, individually.",
        value=None))

    dist.append(_sig(
        "high_in_range", "High in its own range", "处于区间高位", "state",
        pos > 0.75,
        "Price sits in the top 25% of its 120-day channel — the only place "
        "distribution can happen.",
        value=f"{pos.iloc[-1]:.0%}" if pos.notna().any() else None))

    return {
        "ok": True,
        "accumulation": acc,
        "distribution": dist,
        "window": window,
        "has_flow": has_flow,
        "as_of": pd.Timestamp(d.index[-1]).strftime("%Y-%m-%d"),
        "range_position": float(pos.iloc[-1]) if pos.notna().any() else None,
    }


def summarise(result: dict, min_run: int = MIN_RUN) -> dict:
    """
    Count what is currently firing on each side, ignoring one-day flickers.

    A signal counts as live if it is active now with a run of at least
    `min_run`, or — for point events, which never run consecutively — if it
    fired within RECENT_WINDOW sessions.
    """
    def _live(s):
        if not s.get("available"):
            return False
        if s["kind"] == "event":
            return s["recent"]
        return s["active"] and s["run"] >= min_run

    acc = [s for s in result.get("accumulation", []) if _live(s)]
    dis = [s for s in result.get("distribution", []) if _live(s)]
    n_a, n_d = len(acc), len(dis)

    # Tone stays neutral for every verdict and the coloured square carries the
    # direction. Streamlit's semantic boxes are green-good / red-bad, which is
    # backwards from the A-share convention used everywhere else in this app
    # (red = up) — a green "accumulation" banner beside a red 吸筹 heading would
    # contradict itself on the same screen.
    if n_a and n_a >= n_d + 2:
        verdict, tone = "🟥 偏吸筹 Accumulation-leaning", "info"
    elif n_d and n_d >= n_a + 2:
        verdict, tone = "🟩 偏出货 Distribution-leaning", "info"
    elif n_a or n_d:
        verdict, tone = "⬜ 信号混杂 Mixed / transitional", "info"
    else:
        verdict, tone = "⬜ 无明显信号 Nothing firing", "info"

    return {"acc_live": acc, "dist_live": dis, "n_acc": n_a, "n_dist": n_d,
            "verdict": verdict, "tone": tone}
