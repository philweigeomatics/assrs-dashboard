"""
strategies/t_trading.py — 做T候选: which of your holdings suit intraday T+0
round-trips.

Extracted verbatim from pages/t_trading_scanner.py so the Streamlit page and
the new app score identically. The weights, the gates and the verdict cuts are
the page's, unchanged; what is new is only that the scoring is a pure function
of a price frame and two numbers, so it can be tested without a network.

WHAT IT IS LOOKING FOR

A-shares are T+1: shares bought today cannot be sold today. 做T is the way
around that — you already hold the stock, so you sell into a morning spike and
buy the same quantity back lower, or buy a dip and sell the pre-existing shares
into the bounce. Your position ends the day where it started and the swing is
harvested. That only works on a stock with a particular shape:

  * It has to MOVE intraday. No range, no trade — 30% of the score.
  * You have to be able to get in and out. 20-day average turnover, 25%.
  * It has to come BACK. |Close − Open| ÷ (High − Low) says how much of the
    day's range the close gives back: low means the close lands mid-range, an
    oscillator rather than a one-way run. 25%, and lower is better.
  * It must not be trending hard. ADX inside [15, 35] — under 15 nothing
    happens, over 35 a one-way trend means the shares you sold at 10:00 are
    gone at 14:00. 10%.
  * It must not sit at a 60-day extreme, where the next move is a breakout
    rather than a swing. 10%.

Three hard gates reject regardless of score: a 涨停/跌停 in the last five
sessions (you cannot trade a locked limit), turnover under 2%, or an average
intraday range under 1%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Component weights. They sum to 1.0, and a component with no data is dropped
#: with the remainder renormalised — so a missing turnover reading lowers
#: confidence rather than silently scoring zero.
WEIGHTS = {"range": 0.30, "turnover": 0.25, "meanrev": 0.25, "adx": 0.10, "extreme": 0.10}

STRONG_AT = 75.0
OK_AT = 55.0

#: Hard gates. Below these the stock is unsuitable whatever else it scores.
MIN_TURNOVER_PCT = 2.0
MIN_RANGE_PCT = 1.0

MIN_BARS = 60


@dataclass(frozen=True)
class Params:
    """The page's defaults, which are the spec's defaults."""
    range_target: float = 4.0
    turnover_target: float = 8.0
    adx_lo: float = 15.0
    adx_hi: float = 35.0
    extreme_pct: float = 5.0


DEFAULTS = Params()


# ── components ───────────────────────────────────────────────────────────────
def intraday_range_pct(df: pd.DataFrame, window: int = 20) -> float | None:
    """Average (High − Low) ÷ Open over the window, in percent."""
    if len(df) < window:
        return None
    r = ((df["High"] - df["Low"]) / df["Open"]).tail(window)
    return float(r.mean() * 100)


def mean_reversion_bias(df: pd.DataFrame, window: int = 20) -> float | None:
    """
    Average |Close − Open| ÷ (High − Low). Near 0 the close lands in the middle
    of the day's range — the stock gives back what it makes, which is the whole
    premise. Near 1 it closes at an extreme and the range is a one-way move.
    """
    if len(df) < window:
        return None
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    bias = ((df["Close"] - df["Open"]).abs() / rng).dropna().tail(window)
    return float(bias.mean()) if not bias.empty else None


def adx_14(df: pd.DataFrame, window: int = 14) -> float | None:
    """Wilder ADX. Computed here rather than imported so this module stays
    free of the analysis engine and its HMM."""
    if len(df) < window * 3:
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    up, down = high.diff(), -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up
    minus_dm = ((down > up) & (down > 0)).astype(float) * down
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    val = dx.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    return None if pd.isna(val) else float(val)


def distance_from_extreme(df: pd.DataFrame, lookback: int = 60,
                          extreme_pct: float = 5.0) -> float | None:
    """
    0 at a 60-day high or low, 1 in the middle. At an extreme the next move is
    a breakout or a breakdown, and 做T sells into a bounce that does not come.
    """
    if len(df) < lookback:
        return None
    w = df.tail(lookback)
    hi, lo, px = float(w["High"].max()), float(w["Low"].min()), float(df["Close"].iloc[-1])
    if hi <= lo:
        return None
    nearest = min((hi - px) / px * 100, (px - lo) / px * 100)
    if nearest <= 0:
        return 0.0
    if nearest >= extreme_pct * 3:
        return 1.0
    return float(min(max((nearest - extreme_pct) / (2 * extreme_pct), 0.0), 1.0))


def adx_band_score(adx: float | None, lo: float, hi: float) -> float | None:
    """1.0 inside the band, decaying by 0.1 per point outside it."""
    if adx is None:
        return None
    if lo <= adx <= hi:
        return 1.0
    if adx < lo:
        return float(max(0.0, 1.0 - (lo - adx) / 10.0))
    return float(max(0.0, 1.0 - (adx - hi) / 10.0))


def _saturating(value: float | None, target: float) -> float | None:
    """value/target, capped at 1 — more than the target earns no extra credit."""
    return None if value is None else float(min(max(value / target, 0.0), 1.0))


def _r(v, nd):
    return None if v is None else round(v, nd)


# ── scoring ──────────────────────────────────────────────────────────────────
def score(df: pd.DataFrame | None, *, turnover_pct: float | None,
          limit_event: bool, params: Params = DEFAULTS) -> dict:
    """
    The composite, from a price frame and the two facts the frame cannot give.

    Pure: no fetching, no database, no clock. `turnover_pct` is the 20-day
    average from daily_basic and `limit_event` says whether the stock hit a
    limit in the last five sessions — both come from the caller.
    """
    if df is None or len(df) < MIN_BARS:
        return {"score": None, "verdict": "no_data", "rank": 5,
                "why": "价格历史不足", "parts": {},
                "range_pct": None, "turnover_pct": _r(turnover_pct, 2),
                "meanrev_bias": None, "adx": None, "range_pos": None,
                "limit_event": bool(limit_event)}

    range_pct = intraday_range_pct(df, 20)
    bias = mean_reversion_bias(df, 20)
    adx = adx_14(df, 14)
    range_pos = distance_from_extreme(df, 60, params.extreme_pct)

    metrics = {
        "range_pct": _r(range_pct, 2),
        "turnover_pct": _r(turnover_pct, 2),
        "meanrev_bias": _r(bias, 3),
        "adx": _r(adx, 1),
        "range_pos": _r(range_pos, 2),
        "limit_event": bool(limit_event),
    }

    # Gates, in the page's order of precedence.
    fail = None
    if limit_event:
        fail = "近5日有涨停或跌停 — 封板时无法做T"
    elif turnover_pct is not None and turnover_pct < MIN_TURNOVER_PCT:
        fail = f"流动性不足（换手 {turnover_pct:.1f}% < {MIN_TURNOVER_PCT}%）"
    elif range_pct is not None and range_pct < MIN_RANGE_PCT:
        fail = f"日内波动太小（{range_pct:.2f}% < {MIN_RANGE_PCT}%）"
    if fail:
        return {"score": 0.0, "verdict": "skip", "rank": 4, "why": fail,
                "parts": {}, **metrics}

    parts = {
        "range": _saturating(range_pct, params.range_target),
        "turnover": _saturating(turnover_pct, params.turnover_target),
        "meanrev": None if bias is None else float(max(0.0, 1.0 - bias)),
        "adx": adx_band_score(adx, params.adx_lo, params.adx_hi),
        "extreme": range_pos,
    }
    used = sum(WEIGHTS[k] for k, v in parts.items() if v is not None)
    pct = (round(sum(WEIGHTS[k] * v for k, v in parts.items() if v is not None)
                 / used * 100, 1) if used > 0 else 0.0)

    if pct >= STRONG_AT:
        verdict, rank = "strong", 0
    elif pct >= OK_AT:
        verdict, rank = "ok", 1
    else:
        verdict, rank = "not_now", 2

    return {
        "score": pct, "verdict": verdict, "rank": rank, "why": "",
        "parts": {k: _r(v, 3) for k, v in parts.items()},
        **metrics,
    }


VERDICT_CN = {
    "strong": "🟢 适合做T",
    "ok": "🟡 勉强可以",
    "not_now": "⚪ 暂不适合",
    "skip": "⛔ 排除",
    "no_data": "⚠️ 无数据",
}
