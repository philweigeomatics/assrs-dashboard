"""
Choosing the stretch of history a portfolio is judged on. A-shares only.

A "last N days" lookback can only ever describe the most recent stretch, and
that stretch may be one regime or the tail of two. This reads ten years of a
major index, splits it into up and down legs, and prices each stock over any
window of it, so the stocks can be compared inside one regime before any
weights exist.

The legs are hindsight and are labelled as such. A turning point is only known
once price has already reversed by the threshold, which on 沪深300 has taken
anywhere from 9 to 233 sessions. That is fine for slicing history and useless
as a live signal.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import portfolio_build as pb

#: The indices a window can be read against. An allow-list because an open
#: parameter is an open Tushare call on someone else's quota.
INDICES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000300.SH": "沪深300",
    "000688.SH": "科创50",
}

HISTORY_YEARS = 10

#: A stock whose first price is more than this far past the start of a span
#: did not trade through it.
LATE_START_DAYS = 20

#: Fewer overlapping days than this and a beta is mostly noise.
MIN_BETA_DAYS = 20

History = Callable[[str], pd.Series]


def _history_start() -> pd.Timestamp:
    return pd.Timestamp.today().normalize() - pd.DateOffset(years=HISTORY_YEARS)


def _clean(close: pd.Series) -> pd.Series:
    s = close.dropna().copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def index_history(code: str) -> pd.Series:
    """Ten years of one major index's closes."""
    if code not in INDICES:
        raise LookupError(f"没有这个指数：{code}")
    import data_manager

    df = data_manager.get_index_data_live(
        code, start_date=_history_start().strftime("%Y%m%d"))
    if df is None or df.empty or "Close" not in df:
        raise RuntimeError(f"{INDICES[code]} 行情读取失败")
    return _clean(df["Close"])


def fetch_history(sym: str) -> pd.Series:
    """Ten years of one stock's adjusted closes. Raises rather than returns nothing."""
    import markets

    market, code = markets.parse(sym)
    close = market.fetch_closes_since(code, _history_start().date())
    if close is None or close.dropna().empty:
        raise RuntimeError(f"{sym}: 没有返回行情")
    return _clean(close)


def _load(syms: list[str], history: History) -> tuple[dict, list[str], list[str]]:
    def fetch(sym: str):
        try:
            return sym, history(sym), None
        except Exception as exc:                                   # noqa: BLE001
            return sym, None, f"{sym}: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(fetch, syms))
    frames = {s: c for s, c, _ in got if c is not None}
    return frames, [s for s, c, _ in got if c is None], [e for _, _, e in got if e]


# ── regimes ──────────────────────────────────────────────────────────────────
def turning_points(close: pd.Series, pct: float) -> list[tuple]:
    """
    Confirmed swing highs and lows, oldest first, as (date, kind, confirmed).

    A high becomes a top on the day price has fallen `pct` from it, and a low
    becomes a bottom on the day price has risen `pct` from it. `confirmed` is
    that day, which is the only day the turn could actually have been known.
    """
    hi = lo = close.index[0]
    trend = 0
    out: list[tuple] = []
    for t, v in close.items():
        if v > close[hi]:
            hi = t
        if v < close[lo]:
            lo = t
        if trend >= 0 and v <= close[hi] * (1 - pct):
            out.append((hi, "top", t))
            trend, lo = -1, t
        elif trend <= 0 and v >= close[lo] * (1 + pct):
            out.append((lo, "bottom", t))
            trend, hi = 1, t
    return out


def _leg(s: pd.Series, a, b, kind: str, confirmed) -> dict:
    return {
        "from": str(a.date()), "to": str(b.date()), "kind": kind,
        "change_pct": round(float(s[b] / s[a] - 1) * 100, 1),
        "sessions": int(len(s.loc[a:b]) - 1),
        "confirmed": str(confirmed.date()) if confirmed is not None else None,
        # The last leg has no end yet: it is whatever has happened so far.
        "open": confirmed is None,
    }


def regimes(close: pd.Series, pct: float = 0.20) -> list[dict]:
    """The index cut into alternating up and down legs, oldest first."""
    s = close.dropna()
    if len(s) < 2:
        return []
    points = turning_points(s, pct)
    legs = []
    prev = s.index[0]
    for when, kind, confirmed in points:
        if when > prev:
            legs.append(_leg(s, prev, when, "up" if kind == "top" else "down", confirmed))
        prev = when
    if s.index[-1] > prev:
        if points:
            kind = "down" if points[-1][1] == "top" else "up"
        else:
            kind = "up" if s.iloc[-1] >= s.iloc[0] else "down"
        legs.append(_leg(s, prev, s.index[-1], kind, None))
    return legs


def index_view(close: pd.Series, code: str, pct: float) -> dict:
    return {
        "code": code,
        "label": INDICES[code],
        "dates": [str(d.date()) for d in close.index],
        "closes": [round(float(v), 2) for v in close],
        "regimes": regimes(close, pct),
    }


# ── one stock, one window ────────────────────────────────────────────────────
EMPTY_STATS = {"from": None, "days": 0, "partial": False, "change_pct": None,
               "ann_return_pct": None, "ann_vol_pct": None, "sharpe": None,
               "max_drawdown_pct": None, "beta": None}


def beta_of(daily: pd.Series, bench: pd.Series) -> float | None:
    both = pd.concat([daily, bench], axis=1, join="inner").dropna()
    if len(both) < MIN_BETA_DAYS:
        return None
    var = float(both.iloc[:, 1].var())
    if var <= 0:
        return None
    return round(float(both.iloc[:, 0].cov(both.iloc[:, 1])) / var, 2)


def window_stats(close: pd.Series, start, end, rf: float, bench: pd.Series | None,
                 calendar: pd.DatetimeIndex | None = None) -> dict:
    """
    One name over the window, on the market's sessions.

    The same arithmetic as the 实测 row of the optimiser, so a stock reads the
    same in this table as it does beside its weight. That includes a suspended
    day counting as a flat day: the position is still held, and dropping those
    days would annualise over fewer sessions and read higher.
    """
    start_ts = pd.Timestamp(start)
    if calendar is not None:
        close = close.reindex(close.index.union(calendar)).ffill().reindex(calendar)
    s = close.loc[start_ts:pd.Timestamp(end)].dropna()
    if len(s) < 2:
        return dict(EMPTY_STATS)
    daily = s.pct_change(fill_method=None).dropna()
    realised = pb.stats_of(daily.to_frame("x"), pd.Series({"x": 1.0}), rf)
    return {
        "from": str(s.index[0].date()),
        "days": int(len(daily)),
        "partial": bool(s.index[0] > start_ts + pd.Timedelta(days=LATE_START_DAYS)),
        "change_pct": round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 2),
        **realised,
        "beta": None if bench is None else beta_of(daily, bench),
    }


def leg_return(close: pd.Series, leg: dict) -> float | None:
    """A stock's move over one leg, or None if it did not trade through it."""
    a = pd.Timestamp(leg["from"])
    s = close.loc[a:pd.Timestamp(leg["to"])].dropna()
    if len(s) < 2 or s.index[0] > a + pd.Timedelta(days=LATE_START_DAYS):
        return None
    return round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 1)


def stats(symbols: list[str], start: str, end: str, *, index_close: pd.Series, code: str,
          pct: float = 0.20, rf: float = 0.03, history: History = fetch_history) -> dict:
    """Every picked stock over the window, and over every leg of the index."""
    import markets

    syms = list(dict.fromkeys(markets.canonical(s) for s in symbols))
    if not syms:
        raise LookupError("没有可用的标的")
    if pb.market_of(syms) != "CN":
        raise LookupError("时间窗口分析目前只支持 A 股")

    frames, missing, why = _load(syms, history)
    if not frames:
        raise RuntimeError("行情读取失败 — " + "；".join(why[:3]))

    span = index_close.loc[pd.Timestamp(start):pd.Timestamp(end)]
    bench = span.pct_change(fill_method=None).dropna()
    legs = list(reversed(regimes(index_close, pct)))       # newest first
    return {
        "from": str(span.index[0].date()) if len(span) else start,
        "to": str(span.index[-1].date()) if len(span) else end,
        "sessions": int(max(0, len(span) - 1)),
        "missing": missing,
        "index": {"code": code, "label": INDICES[code],
                  **window_stats(index_close, start, end, rf, bench)},
        "rows": [{"t": t, **window_stats(c, start, end, rf, bench, span.index)}
                 for t, c in frames.items()],
        "regimes": legs,
        "cells": {t: [leg_return(c, leg) for leg in legs] for t, c in frames.items()},
    }


# ── the optimiser's closes, for a chosen window ──────────────────────────────
def window_closes(symbols: list[str], start: str, end: str | None,
                  history: History = fetch_history) -> pd.DataFrame:
    """
    Aligned closes over the window, for the optimiser.

    Every name has to be priced on every date, so a stock listed partway
    through moves the start. Which one did is kept in `attrs`, since a window
    shorter than the one drawn has to say why.
    """
    frames, missing, why = _load(symbols, history)
    if not frames:
        raise RuntimeError("行情读取失败 — " + "；".join(why[:3]))

    start_ts = pd.Timestamp(start)
    # Forward-filled before the cut, so a stock suspended on the first day of
    # the window still has the price it was suspended at.
    px = pd.concat([c.rename(t) for t, c in frames.items()], axis=1).sort_index()
    px = px.dropna(axis=1, how="all")
    first = {c: px[c].first_valid_index() for c in px.columns}
    px = px.ffill().loc[start_ts:(pd.Timestamp(end) if end else None)].dropna(how="any")
    if len(px) < pb.MIN_SESSIONS:
        raise LookupError(
            f"这个窗口里共同交易日只有 {len(px)} 天，少于 {pb.MIN_SESSIONS} 天，"
            f"协方差矩阵没有意义")

    youngest = max(first, key=lambda c: first[c])
    late = first[youngest] > start_ts + pd.Timedelta(days=LATE_START_DAYS)
    px.attrs.update(missing=missing,
                    limited_by=(youngest, str(first[youngest].date())) if late else None)
    return px


__all__ = ["INDICES", "fetch_history", "index_history", "index_view", "regimes",
           "stats", "turning_points", "window_closes", "window_stats"]
