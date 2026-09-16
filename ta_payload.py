"""
ta_payload.py — the Technical Analysis page as data, for the SPA frontend.

The Streamlit page computes its chart inside create_single_stock_chart_analysis
and hands Plotly a figure. The new frontend draws with lightweight-charts in
the browser, so it needs the same numbers as plain arrays instead. This module
produces them.

Parity is the contract. Every series and marker rule here mirrors a specific
block of the Streamlit chart function — same window (chart_window), same
rolling windows, same ddof, same thresholds, same ±DI gates on the ADX markers
— so a marker on the new chart means exactly what it means on the old one.
test_ta_payload_parity.py builds the real Plotly figure and compares the
arrays trace by trace; if either side changes, that test is what notices.

Arrays are aligned to `dates`, one entry per bar, with None for bars where a
value does not exist yet (warm-up). Markers are sent as bar INDICES into those
arrays rather than dates, which keeps the payload small and makes "which bar"
unambiguous on the client.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

INITIAL_VISIBLE_BARS = 250          # ≈ one trading year in the opening view
Z_WINDOW = 20
OBV_MOM_WINDOW = 20                 # page default (mf_cum_days = 20)
MF_ROLLING_DAYS = 20
MF_LOOKBACK_DAYS = 250
TREND_SHADE_MIN_DAYS = 5

# Continuous ADX lifecycle states drawn as the ribbon at the foot of the ADX
# pane, with the Streamlit chart's colours.
ADX_RIBBON = {
    'Accelerating Up': '#ef4444',
    'Strong Trend': '#991b1b',
    'Losing Steam': '#f59e0b',
    'Accelerating Down': '#22c55e',
    'Slowing Down': '#62C7F7',
}

REGIME_COLORS = {
    'Low Volatility': 'rgba(34, 197, 94, 0.08)',
    'Normal Volatility': 'rgba(59, 130, 246, 0.05)',
    'High Volatility': 'rgba(255, 110, 0, 0.11)',
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _num(v, nd: int = 4):
    """JSON-safe float: None for NaN/inf/missing, rounded to keep payload small."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _arr(s, nd: int = 4) -> list:
    return [_num(v, nd) for v in (s.tolist() if hasattr(s, "tolist") else s)]


def _idx(mask) -> list[int]:
    """Positions where a boolean Series/array is True (NaN counts as False)."""
    m = pd.Series(mask).fillna(False).astype(bool).to_numpy()
    return [int(i) for i in np.flatnonzero(m)]


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def chart_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    The exact slice the Streamlit chart draws: from the regime anchor when that
    leaves at least 60 bars, else the last INITIAL_VISIBLE_BARS.
    """
    from analysis_engine import REGIME_ANCHOR
    if df is None or getattr(df, "empty", True):
        return df
    d = df.sort_index()
    anchored = d[d.index >= pd.Timestamp(REGIME_ANCHOR)]
    if len(anchored) >= 60:
        return anchored
    return d.tail(INITIAL_VISIBLE_BARS)


def _segments(mask: pd.Series, min_len: int = 1) -> list[dict]:
    """Runs of consecutive True as {from, to} bar indices (inclusive)."""
    m = mask.fillna(False).astype(bool).to_numpy()
    out, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if (not v or i == len(m) - 1) and start is not None:
            end = i if v else i - 1
            if end - start + 1 >= min_len:
                out.append({"from": start, "to": end})
            start = None
    return out


def _regime_segments(df: pd.DataFrame) -> list[dict]:
    if 'Market_Regime' not in df.columns:
        return []
    r = df['Market_Regime'].to_numpy()
    out, start = [], 0
    for i in range(1, len(r) + 1):
        if i == len(r) or r[i] != r[start]:
            if isinstance(r[start], str) and r[start] in REGIME_COLORS:
                out.append({"from": start, "to": i - 1, "regime": r[start],
                            "color": REGIME_COLORS[r[start]]})
            start = i
    return out


# ── data loaders ─────────────────────────────────────────────────────────────
def load_moneyflow(ticker: str, lookback_days: int = MF_LOOKBACK_DAYS):
    """
    主力净流入 per day = (大单买 + 特大单买) − (大单卖 + 特大单卖), in 万元.
    Same Tushare `moneyflow` call and definition as the Streamlit page's
    loader. Returns a DataFrame with a `main_net` column, or None.
    """
    import data_manager
    try:
        data_manager.init_tushare()
        api = data_manager.TUSHARE_API
        if api is None:
            return None
        ts_code = data_manager.get_tushare_ticker(ticker)
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=int(lookback_days * 1.6) + 20)).strftime("%Y%m%d")
        mf = api.moneyflow(ts_code=ts_code, start_date=start, end_date=end)
        if mf is None or mf.empty:
            return None
        mf = mf.copy()
        for c in ("buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"):
            mf[c] = pd.to_numeric(mf.get(c), errors="coerce").fillna(0)
        mf["main_net"] = (mf["buy_lg_amount"] + mf["buy_elg_amount"]
                          - mf["sell_lg_amount"] - mf["sell_elg_amount"])
        mf["date"] = pd.to_datetime(mf["trade_date"], format="%Y%m%d", errors="coerce")
        mf = mf.dropna(subset=["date"]).set_index("date").sort_index()
        return mf[["main_net"]]
    except Exception as exc:
        print(f"[ta_payload] moneyflow {ticker} failed: {exc}")
        return None


# ── the payload ──────────────────────────────────────────────────────────────
def build_series(analysis_df: pd.DataFrame,
                 fundamentals_df: pd.DataFrame | None = None,
                 moneyflow_df: pd.DataFrame | None = None) -> dict:
    """
    Every chart array and marker for the windowed bars. Pure: takes frames,
    returns a dict. Split from build_payload so the parity test can feed it
    the exact frames it hands the Streamlit chart function.
    """
    df = chart_window(analysis_df)
    n = len(df)

    # Volume pane — OBV动能: net signed volume over N sessions / avg daily volume.
    avgvol = df['Volume'].rolling(20, min_periods=1).mean().replace(0, np.nan)
    obv_mom = ((df['OBV'] - df['OBV'].shift(OBV_MOM_WINDOW)) / avgvol
               if 'OBV' in df.columns else pd.Series(np.nan, index=df.index))

    # Z-score pane — 20-day, population std, returns for price.
    rets = df['Close'].pct_change()
    vol = df['Volume']
    price_z = (rets - rets.rolling(Z_WINDOW).mean()) / rets.rolling(Z_WINDOW).std(ddof=0)
    vol_z = (vol - vol.rolling(Z_WINDOW).mean()) / vol.rolling(Z_WINDOW).std(ddof=0)

    # P/E pane.
    pe = pd.Series(np.nan, index=df.index)
    if fundamentals_df is not None and not fundamentals_df.empty and 'PE_TTM' in fundamentals_df.columns:
        pe = fundamentals_df.reindex(df.index, method='ffill')['PE_TTM']

    # 主力净流入 pane — daily bars plus rolling N-day sum.
    mf_daily = pd.Series(np.nan, index=df.index)
    mf_roll = pd.Series(np.nan, index=df.index)
    has_mf = False
    if moneyflow_df is not None and not moneyflow_df.empty and 'main_net' in moneyflow_df.columns:
        mf_daily = moneyflow_df['main_net'].reindex(df.index).fillna(0)
        mf_roll = mf_daily.rolling(MF_ROLLING_DAYS, min_periods=1).sum()
        has_mf = True

    di_plus, di_minus = _col(df, 'DI_Plus'), _col(df, 'DI_Minus')
    di_up = di_plus >= di_minus            # Bottoming / Reversing Up gate
    di_up_sell = di_plus > di_minus        # Peaking / Reversing Down gate
    pattern = _col(df, 'ADX_Pattern', '').astype(str)

    series = {
        "MA5": _arr(_col(df, 'MA5'), 3), "MA10": _arr(_col(df, 'MA10'), 3),
        "MA20": _arr(_col(df, 'MA20'), 3), "MA50": _arr(_col(df, 'MA50'), 3),
        "MA60": _arr(_col(df, 'MA60'), 3), "MA200": _arr(_col(df, 'MA200'), 3),
        "EMA5": _arr(_col(df, 'EMA5'), 3),
        "BB_Upper": _arr(_col(df, 'BB_Upper'), 3), "BB_Lower": _arr(_col(df, 'BB_Lower'), 3),
        "Vol_Scaled_OBV": _arr(_col(df, 'Volume_Scaled_OBV'), 3),
        "OBV_Mom": _arr(obv_mom, 3),
        "MACD": _arr(_col(df, 'MACD')), "MACD_Signal": _arr(_col(df, 'MACD_Signal')),
        "MACD_Hist": _arr(_col(df, 'MACD_Hist') * 2.5),   # page scales 2.5x
        "RSI": _arr(_col(df, 'RSI_14'), 2),
        "RSI_P10": _arr(_col(df, 'RSI_P10'), 2), "RSI_P90": _arr(_col(df, 'RSI_P90'), 2),
        "ADX": _arr(_col(df, 'ADX'), 2), "ADX_LOWESS": _arr(_col(df, 'ADX_LOWESS'), 2),
        "ADX_BB_Upper": _arr(_col(df, 'ADX_BB_Upper'), 2),
        "ADX_BB_Lower": _arr(_col(df, 'ADX_BB_Lower'), 2),
        "DI_Plus": _arr(di_plus, 2), "DI_Minus": _arr(di_minus, 2),
        "Price_Z": _arr(price_z, 3), "Volume_Z": _arr(vol_z, 3),
        "PE_TTM": _arr(pe, 2),
        "MF_Daily": _arr(mf_daily, 1) if has_mf else [None] * n,
        "MF_Rolling": _arr(mf_roll, 1) if has_mf else [None] * n,
    }

    markers = {
        "price": {
            "accumulation": _idx(_col(df, 'Signal_Accumulation', False)),
            "squeeze": _idx(_col(df, 'Signal_Squeeze', False)),
            "squeeze_bull": _idx(_col(df, 'Squeeze_Fired_Bullish', False)),
            "squeeze_bear": _idx(_col(df, 'Squeeze_Fired_Bearish', False)),
            "downtrend_reversal": _idx(_col(df, 'Downtrend_Reversal', False)),
            "uptrend_reversal": _idx(_col(df, 'Uptrend_Reversal', False)),
            "exit_macd": _idx(_col(df, 'Exit_MACD_Lead', False)),
            "screaming_buy": _idx(_col(df, 'Entry_Candidate', '').astype(str) == 'Screaming Buy'),
            "screaming_sell": _idx(_col(df, 'Exit_Candidate', '').astype(str) == 'Screaming Sell'),
        },
        "macd": {
            "trigger": _idx(_col(df, 'MACD_Trigger', False) == True),     # noqa: E712
            "peaking": _idx(_col(df, 'MACD_Peaking', False) == True),     # noqa: E712
            "bearish_cross": _idx(_col(df, 'MACD_BearishCrossover', False) == True),  # noqa: E712
        },
        "rsi": {
            "bottoming": _idx(_col(df, 'RSI_Bottoming', False) == True),  # noqa: E712
            "peaking": _idx(_col(df, 'RSI_Peaking', False) == True),      # noqa: E712
        },
        "adx": {
            "di_screaming_buy": _idx(_col(df, 'DI_Screaming_Buy', False) == True),   # noqa: E712
            "di_screaming_sell": _idx(_col(df, 'DI_Screaming_Sell', False) == True),  # noqa: E712
            "bottoming": _idx((pattern == 'Bottoming') & di_up),
            "reversing_up": _idx((pattern == 'Reversing Up') & di_up),
            "peaking": _idx((pattern == 'Peaking') & di_up_sell),
            "reversing_down": _idx((pattern == 'Reversing Down') & di_up_sell),
        },
        "z": {
            "oversold": _idx(price_z <= -2.5),
            "overbought": _idx(price_z >= 2.0),
        },
    }

    ribbon = [{"i": int(i), "state": s, "color": ADX_RIBBON[s]}
              for i, s in enumerate(pattern.tolist()) if s in ADX_RIBBON]

    bands = {
        "regime": _regime_segments(df),
        "macd_uptrend": _segments(_col(df, 'Large_Uptrend', False), TREND_SHADE_MIN_DAYS),
        "macd_downtrend": _segments(_col(df, 'Large_Downtrend', False), TREND_SHADE_MIN_DAYS),
    }

    return {
        "dates": [d.strftime('%Y-%m-%d') for d in df.index],
        "ohlcv": {
            "o": _arr(df['Open'], 3), "h": _arr(df['High'], 3),
            "l": _arr(df['Low'], 3), "c": _arr(df['Close'], 3),
            "v": _arr(df['Volume'], 0),
        },
        "series": series,
        "markers": markers,
        "adx_ribbon": ribbon,
        "bands": bands,
        "has_moneyflow": has_mf,
        "initial_visible": min(n, INITIAL_VISIBLE_BARS),
    }


def build_header(analysis_df: pd.DataFrame, fundamentals_df: pd.DataFrame | None) -> dict:
    last = analysis_df.iloc[-1]
    prev_close = float(analysis_df['Close'].iloc[-2]) if len(analysis_df) > 1 else float(last['Close'])
    close = float(last['Close'])
    out = {
        "date": analysis_df.index[-1].strftime('%Y-%m-%d'),
        "close": _num(close, 3),
        "prev_close": _num(prev_close, 3),
        "change_pct": _num((close / prev_close - 1) * 100 if prev_close else None, 2),
        "total_mv_yi": None, "circ_mv_yi": None,
        "pe_ttm": None, "pb": None, "turnover_rate": None,
    }
    if fundamentals_df is not None and not fundamentals_df.empty:
        f = fundamentals_df.iloc[-1]
        out.update({
            "total_mv_yi": _num(f.get('Total_MV_Yi'), 1),
            "circ_mv_yi": _num(f.get('Circ_MV_Yi'), 1),
            "pe_ttm": _num(f.get('PE_TTM'), 2),
            "pb": _num(f.get('PB'), 2),
            "turnover_rate": _num(f.get('Turnover_Rate'), 2),
        })
    return out


def build_signals(analysis_df: pd.DataFrame, boxes: list[dict]) -> dict:
    """
    The status strip. Same flags as the Streamlit cards, with one correction:
    the Streamlit header reads 'MACD_Classic_Crossover' / 'MACD_Bearish_Crossover',
    which the engine never produces (it writes MACD_ClassicCrossover /
    MACD_BearishCrossover), so MACD金叉 and MACD死叉 could never appear there.
    This uses the columns that exist.
    """
    last = analysis_df.iloc[-1]

    def flag(col):
        return bool(last.get(col, False)) if col in analysis_df.columns else False

    bull = [lbl for col, lbl in (("MACD_Bottoming", "MACD底"),
                                 ("MACD_ClassicCrossover", "MACD金叉"),
                                 ("RSI_Bottoming", "RSI底")) if flag(col)]
    bear = [lbl for col, lbl in (("MACD_Peaking", "MACD顶"),
                                 ("MACD_BearishCrossover", "MACD死叉"),
                                 ("RSI_Peaking", "RSI顶")) if flag(col)]

    active = next((b for b in boxes if b.get("is_active")), None)
    box = None
    if active is not None:
        box = {"kind": active.get("kind"), "status": active.get("status"),
               "status_cn": active.get("status_cn")}
        if active.get("kind") == "BOX":
            box.update({
                "top": _num(active["top"], 3), "bot": _num(active["bot"], 3),
                "height_pct": _num(active["height_pct"], 1),
                "position": _num(active.get("position"), 3),
                "touches_top": int(active["touches_top"]),
                "touches_bot": int(active["touches_bot"]),
                "sessions": int(active["n_sessions"]),
                "quality": _num(active["quality"], 2),
            })
        else:
            box.update({"drift_pct": _num(active.get("drift_pct"), 1),
                        "r2": _num(active.get("r2"), 2)})

    return {
        "squeeze": flag("Signal_Squeeze"),
        "accumulation": flag("Signal_Accumulation"),
        "bull": bull, "bear": bear,
        "box": box,
        "regime": str(last.get("Market_Regime") or "Normal Volatility"),
        "adx": _num(last.get("ADX"), 1),
        "adx_pattern": str(last.get("ADX_Pattern") or ""),
    }


def build_boxes(boxes: list[dict], dates: list[str]) -> list[dict]:
    """Boxes clipped to the chart window, as bar indices the client can draw."""
    pos = {d: i for i, d in enumerate(dates)}
    first = dates[0] if dates else None
    out = []
    for b in boxes[-3:]:
        s, e = b["start"].strftime('%Y-%m-%d'), b["end"].strftime('%Y-%m-%d')
        if first is None or e < first:
            continue
        s = max(s, first)
        # nearest bar at/after start, at/before end
        si = next((pos[d] for d in dates if d >= s), None)
        ei = max((pos[d] for d in dates if d <= e), default=None)
        if si is None or ei is None or ei < si:
            continue
        item = {"from": si, "to": ei, "kind": b.get("kind"),
                "status_cn": b.get("status_cn"), "is_active": bool(b.get("is_active"))}
        if b.get("kind") == "BOX":
            item.update({"top": _num(b["top"], 3), "bot": _num(b["bot"], 3),
                         "zone": _num(b.get("zone"), 3),
                         "touches_top": int(b["touches_top"]),
                         "touches_bot": int(b["touches_bot"]),
                         "height_pct": _num(b["height_pct"], 1),
                         "sessions": int(b["n_sessions"]),
                         "quality": _num(b["quality"], 2)})
        else:
            item.update({"drift_pct": _num(b.get("drift_pct"), 1),
                         "r2": _num(b.get("r2"), 2)})
        out.append(item)
    return out


def build_chips(analysis_df: pd.DataFrame,
                fundamentals_df: pd.DataFrame | None,
                decay: float = 1.0) -> dict | None:
    """
    筹码分布 for the side panel, computed with chip_distribution.py — the same
    module and the same inputs the Streamlit page uses (前复权 prices plus the
    turnover rate that already came with the fundamentals, so no extra API
    call).

    The histogram is trimmed to the bins that actually hold chips: the grid
    pads 3% past the traded range at both ends, and shipping those empty bins
    would be a third of the array for nothing.
    """
    if fundamentals_df is None or fundamentals_df.empty:
        return None
    if "Turnover_Rate" not in fundamentals_df.columns:
        return None
    turn = fundamentals_df["Turnover_Rate"].reindex(analysis_df.index)
    if turn.notna().sum() < 30:
        return None

    import chip_distribution as cdist
    try:
        m = cdist.analyse(analysis_df, turn, decay=decay)
    except Exception as exc:
        print(f"[ta_payload] chips failed: {exc}")
        return None
    if not m.get("ok"):
        return None

    grid, chips = m["grid"], m["chips"]
    keep = chips > chips.max() * 0.002
    idx = np.flatnonzero(keep)
    lo, hi = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, len(grid))

    return {
        "prices": [_num(v, 3) for v in grid[lo:hi]],
        # Percent of float at each price, so the client never divides.
        "weights": [_num(v * 100, 4) for v in chips[lo:hi]],
        "winner_rate": _num(m["winner_rate"] * 100, 1),
        "trapped_rate": _num(m["trapped_rate"] * 100, 1),
        "weight_avg": _num(m["weight_avg"], 2),
        "concentration": _num(m["concentration"], 3),
        "cost_5pct": _num(m["cost_5pct"], 2),
        "cost_15pct": _num(m["cost_15pct"], 2),
        "cost_50pct": _num(m["cost_50pct"], 2),
        "cost_85pct": _num(m["cost_85pct"], 2),
        "cost_95pct": _num(m["cost_95pct"], 2),
        "peak_price": _num(m.get("peak_price"), 2),
        "n_peaks": int(m.get("n_peaks", 0)),
        "peaks": [{"price": _num(p["price"], 2), "share": _num(p["share"] * 100, 1)}
                  for p in m.get("peaks", [])],
        "setup_score": _num(m.get("setup_score"), 2),
        "setup_label": m.get("setup_label"),
        "converged": bool(m["converged"]),
        "seed_remaining": _num(m["seed_remaining"] * 100, 1),
        "cum_turnover_pct": _num(m["cum_turnover_pct"], 0),
        "sessions": int(m["sessions"]),
        "decay": float(decay),
    }


def build_payload(ticker: str, analysis_df: pd.DataFrame | None = None,
                  fundamentals_df: pd.DataFrame | None = None) -> dict:
    """
    Fetch, analyse and package one stock for the frontend.

    `analysis_df` / `fundamentals_df` let a caller that already holds the
    frames skip the fetch and the HMM — the API keeps them cached so the
    What-If simulator and the comparison overlay reuse this stock's analysis
    instead of paying ~20s for it again.
    """
    import data_manager
    import box_detection as bxd
    import watchlist_scan
    from analysis_engine import run_single_stock_analysis

    # Retried, and the two failure modes kept apart: ts.pro_bar returns None
    # when the CALL fails (timeout, rate-limit hiccup) while a genuinely young
    # stock returns a short frame. Folding them together told the user
    # "not enough price history" for what was really a network blip — which is
    # exactly what it said the first time this page hit a flaky fetch.
    if analysis_df is None:
        stock_df = watchlist_scan.fetch_frame(ticker)
        if stock_df is None:
            raise RuntimeError(
                f"price data for {ticker} could not be fetched "
                f"(after {watchlist_scan.FETCH_ATTEMPTS} attempts) — try again")
        if len(stock_df) < 60:
            raise LookupError(f"not enough price history for {ticker}")
        analysis_df = run_single_stock_analysis(stock_df)
        fundamentals_df = data_manager.get_stock_fundamentals_live(
            ticker, stock_df.index.min().strftime('%Y%m%d'),
            stock_df.index.max().strftime('%Y%m%d'))
    moneyflow_df = load_moneyflow(ticker)
    boxes = bxd.detect_boxes(analysis_df)

    body = build_series(analysis_df, fundamentals_df, moneyflow_df)
    return {
        "ticker": ticker,
        "name": data_manager.get_stock_name_from_db(ticker) or ticker,
        "header": build_header(analysis_df, fundamentals_df),
        "signals": build_signals(analysis_df, boxes),
        "boxes": build_boxes(boxes, body["dates"]),
        "chips": build_chips(analysis_df, fundamentals_df),
        **body,
    }
