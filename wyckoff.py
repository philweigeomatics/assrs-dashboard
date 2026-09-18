"""
wyckoff.py — Wyckoff phases for an index, defined statistically.

Wyckoff's four phases (accumulation → markup → distribution → markdown) are
normally read off a chart by eye, which means two people reading the same chart
disagree and neither can be shown to be wrong. This defines them arithmetically
instead, from three measurements that need no moving-average lag and no
discretion:

    range position  where the close sits inside the 120-day Donchian channel,
                    0.0 at the low, 1.0 at the high
    volatility      20-day return σ against its own 120-day baseline
    volume z        against a 60-day mean and σ

and then:

    markup          top quartile of the range, and above the 20-day mean
    markdown        bottom quartile, and below it
    distribution    upper half, but volatility ABOVE baseline — high churn at a
                    high price is the statistical signature of supply meeting
                    demand
    accumulation    lower half, volatility BELOW baseline — quiet absorption
    transition      none of the above; said out loud rather than forced into
                    the nearest phase

The order matters and is not cosmetic: markup/markdown are tested first, so a
quiet drift to new highs is markup, and only a *churning* upper half is called
distribution. Read the conditions as a chain, not a set.

This is the same arithmetic the Streamlit dashboard has always used, lifted out
of the page so the API can serve it and so it can be tested. What is new is
`edge`: what each phase has actually been followed by in this index's own
history — mean forward return over the next month, per phase, with sample
sizes. A regime label that has never been checked against what came next is a
vocabulary, not a signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 120          # ≈ six months — the channel and the volatility baseline
VOL_WINDOW = 60         # volume z-score window
TREND_WINDOW = 20       # the mean that separates markup from a high-range drift
HORIZON = 20            # ≈ one month, for the forward-return edge

#: Consecutive bars a new phase must hold before it is reported as the phase.
#:
#: Without this the raw rules flip on live data roughly every four sessions —
#: 49 changes in 180 bars of 沪深300 — which is a barcode, not a regime. The
#: confirmation is strictly BACKWARD-looking: at bar t it asks whether the raw
#: label has been the same for the last CONFIRM bars, never whether it will
#: persist. A "smoothing" that absorbed short runs into their neighbours would
#: need bar t+2 to label bar t, which would quietly put look-ahead into the
#: forward-return table underneath it.
CONFIRM = 3

#: Bars drawn by default. The channel needs LOOKBACK bars behind the first one,
#: so `analyse` wants roughly LOOKBACK + BARS of history to fill the chart.
BARS = 180

#: Below this many observations a phase's forward return is noise.
MEANINGFUL_N = 30

PHASES = {
    "accumulation": {"label": "吸筹", "en": "Accumulation", "color": "#0ea5e9",
                     "means": "价格压在6个月区间下半部，波动低于基准 — 安静的吸纳。"},
    "markup":       {"label": "拉升", "en": "Markup", "color": "#d70015",
                     "means": "价格进入6个月区间前25%，且站上20日均线 — 上行趋势成立。"},
    "distribution": {"label": "派发", "en": "Distribution", "color": "#f59e0b",
                     "means": "价格仍在上半部，但波动高于基准 — 高位换手，供给在见到需求。"},
    "markdown":     {"label": "下跌", "en": "Markdown", "color": "#1f7a35",
                     "means": "价格跌入区间后25%，且在20日均线之下 — 下行主导。"},
    "transition":   {"label": "过渡", "en": "Transition", "color": "#8e8e93",
                     "means": "四个统计条件都不成立 — 市场夹在两个状态之间。"},
}

ORDER = ("markup", "markdown", "distribution", "accumulation")


def classify(df: pd.DataFrame, lookback: int = LOOKBACK,
             confirm: int = CONFIRM) -> pd.DataFrame:
    """
    The measurements plus a `phase` column, indexed like `df`.

    Rows before the longest window can be filled are dropped — a phase computed
    from a half-formed baseline is not a weaker signal, it is a different
    statistic wearing the same name.
    """
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise LookupError(f"缺少 {col} 列，无法计算 Wyckoff 阶段")
    if len(df) < lookback + TREND_WINDOW:
        raise LookupError(f"至少需要 {lookback + TREND_WINDOW} 根K线，只有 {len(df)} 根")

    out = pd.DataFrame(index=df.index)
    hi = df["High"].rolling(lookback).max()
    lo = df["Low"].rolling(lookback).min()
    span = (hi - lo).replace(0.0, np.nan)

    out["high"] = hi
    out["low"] = lo
    out["position"] = (df["Close"] - lo) / span

    ret = df["Close"].pct_change()
    out["volatility"] = ret.rolling(TREND_WINDOW).std()
    out["vol_baseline"] = out["volatility"].rolling(lookback).mean()

    vmean = df["Volume"].rolling(VOL_WINDOW).mean()
    vstd = df["Volume"].rolling(VOL_WINDOW).std().replace(0.0, np.nan)
    out["volume_z"] = (df["Volume"] - vmean) / vstd

    trend = df["Close"].rolling(TREND_WINDOW).mean()
    out["trend"] = trend

    conditions = [
        (out["position"] > 0.75) & (df["Close"] > trend),
        (out["position"] < 0.25) & (df["Close"] < trend),
        (out["position"] >= 0.50) & (out["volatility"] > out["vol_baseline"]),
        (out["position"] < 0.50) & (out["volatility"] <= out["vol_baseline"]),
    ]
    out["raw_phase"] = np.select(conditions, list(ORDER), default="transition")
    # np.select happily labels rows whose inputs are all NaN (every condition
    # is False → "transition"), which would put a fabricated phase at the very
    # start of every chart. Dropped BEFORE confirmation so the streak does not
    # start counting inside the warm-up.
    out = out.dropna(subset=["vol_baseline", "position"])
    out["phase"] = _confirm(out["raw_phase"], confirm)
    return out


def _confirm(raw: pd.Series, n: int) -> pd.Series:
    """
    Report a new phase only once the raw rules have agreed for `n` bars.

    Causal by construction: bar t's label depends on bars t-n+1…t and nothing
    later, so the series can be read as what you would have seen in real time —
    which is the only version it is honest to measure forward returns against.
    """
    if n <= 1 or raw.empty:
        return raw.copy()

    values = raw.to_numpy()
    out = np.empty_like(values)
    current = values[0]
    streak = 1
    for i, v in enumerate(values):
        if i:
            streak = streak + 1 if v == values[i - 1] else 1
            if v != current and streak >= n:
                current = v
        out[i] = current
    return pd.Series(out, index=raw.index, name="phase")


def _spans(phase: pd.Series) -> list[dict]:
    """Run-length encode the phase column into [from, to] bar spans."""
    if phase.empty:
        return []
    runs, start, prev = [], 0, phase.iloc[0]
    for i, value in enumerate(phase):
        if value != prev:
            runs.append({"from": start, "to": i - 1, "phase": prev})
            start, prev = i, value
    runs.append({"from": start, "to": len(phase) - 1, "phase": prev})
    return runs


def _edge(close: pd.Series, phase: pd.Series, horizon: int) -> dict:
    """
    What followed each phase, in this index's own history.

    Forward `horizon`-bar return, grouped by the phase in force at the time,
    reported as a deviation from the all-phase mean. The absolute numbers are
    dominated by whatever the index did overall; the deviation is the part that
    is about the phase.
    """
    fwd = (close.shift(-horizon) / close - 1.0)
    joined = pd.DataFrame({"p": phase, "f": fwd}).dropna()
    base = float(joined["f"].mean()) if not joined.empty else 0.0

    rows = []
    for key in list(ORDER) + ["transition"]:
        vals = joined.loc[joined["p"] == key, "f"].to_numpy(dtype=float)
        if vals.size == 0:
            rows.append({"phase": key, "n": 0, "mean_pct": None,
                         "edge_pp": None, "win_pct": None, "thin": True})
            continue
        rows.append({
            "phase": key,
            "n": int(vals.size),
            "mean_pct": round(float(vals.mean()) * 100, 2),
            "edge_pp": round(float(vals.mean() - base) * 100, 2),
            "win_pct": round(float((vals > 0).mean()) * 100, 1),
            "thin": bool(vals.size < MEANINGFUL_N),
        })
    rows.sort(key=lambda r: (r["edge_pp"] is None, -(r["edge_pp"] or 0)))
    return {"horizon": horizon, "baseline_pct": round(base * 100, 2),
            "meaningful_at": MEANINGFUL_N, "rows": rows,
            # Overlapping windows: 20-bar forward returns from adjacent days
            # share 19 of their 20 days, so `n` is nowhere near that many
            # independent observations. Said here so the client can say it too.
            "overlapping": True}


def analyse(df: pd.DataFrame, *, lookback: int = LOOKBACK, bars: int = BARS,
            horizon: int = HORIZON, confirm: int = CONFIRM,
            name: str = "指数") -> dict:
    """Everything the panel draws, for one index."""
    m = classify(df, lookback=lookback, confirm=confirm)
    if m.empty:
        raise LookupError("滚动窗口填满之前没有可用的K线")

    edge = _edge(df["Close"].reindex(m.index), m["phase"], horizon)

    view = m.tail(bars)
    px = df.loc[view.index]
    last = view.iloc[-1]

    spans = _spans(view["phase"])
    # How long the CURRENT phase has run — on the full history, not the window
    # drawn, so a phase that began before the chart does is not truncated to
    # "started on the first bar you can see".
    full = m["phase"]
    changed = full != full.shift(1)
    since_pos = int(np.flatnonzero(changed.to_numpy())[-1])
    since = full.index[since_pos]

    return {
        "name": name,
        "phase": str(last["phase"]),
        "phases": PHASES,
        "asof": view.index[-1].strftime("%Y-%m-%d"),
        "since": since.strftime("%Y-%m-%d"),
        "days_in_phase": int(len(full) - since_pos),
        "position_pct": round(float(last["position"]) * 100, 1),
        "volume_z": round(float(last["volume_z"]), 2) if pd.notna(last["volume_z"]) else None,
        "volatility": round(float(last["volatility"]) * 100, 2),
        "vol_baseline": round(float(last["vol_baseline"]) * 100, 2),
        "lookback": lookback,
        "confirm": confirm,
        "dates": [d.strftime("%Y-%m-%d") for d in view.index],
        "bars": [{"o": round(float(o), 2), "h": round(float(h), 2),
                  "l": round(float(lo), 2), "c": round(float(c), 2)}
                 for o, h, lo, c in zip(px["Open"], px["High"], px["Low"], px["Close"])],
        "channel": {
            "high": [round(float(v), 2) for v in view["high"]],
            "low": [round(float(v), 2) for v in view["low"]],
        },
        "spans": spans,
        "edge": edge,
    }
