"""
stock_bond.py — 股债跷跷板: is money leaning towards stocks or bonds, and
since when?

The measure
-----------
Money that leaves stocks for safety mostly goes to government bonds, and
comes back when the fear passes. Flows are not published daily, but their
result is: the price of stocks against the price of bonds. So the measure is

    股债比 = (stock index / its first value) / (bond index / its first value)

on TOTAL-RETURN indexes on both sides — dividends and coupons included —
because a price index would have stocks lose ground to bonds by their
dividend yield every year for no reason:

    A 股  沪深300全收益 (H00300.CSI)  against  中证全债 (H11001.CSI)
    美股  SPY                        against  IEF (7–10 year Treasuries)
          (Yahoo's adjusted closes: dividends and coupons reinvested)

Rising: stocks are beating bonds — risk appetite, money leaning to stocks.
Falling: bonds are winning — risk aversion, money leaning to bonds.

The cycles
----------
A cycle starts or ends where the ratio turns, and a turn is only a turn once
it has reversed by `pct` from its extreme (portfolio_window.regimes, the same
rule the 组合优化 time window cuts the index with). So every leg carries the
day its START was confirmed, which is later than the start itself: the
honest timing for anyone who would have acted on it. The last leg is open —
and the payload says how far the ratio would have to move from here for the
next turn to be confirmed.

The seesaw itself
-----------------
Whether stocks and bonds are moving AGAINST each other right now is the
rolling correlation of their daily returns. Negative: the classic seesaw —
money moving from one to the other. Positive: both moved by the same thing,
usually interest rates (2022 in the US: stocks and bonds fell together), and
then a falling ratio is not money hiding in bonds.

Pure: the closes are passed in; the API fetches them.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import portfolio_window as pw

PAIRS = {
    "CN": {"equity": ("H00300.CSI", "沪深300全收益"), "bond": ("H11001.CSI", "中证全债"),
           "label": "A 股"},
    "US": {"equity": ("SPY", "标普500 (SPY)"), "bond": ("IEF", "7–10年美债 (IEF)"),
           "label": "美股"},
}
PCTS = (0.05, 0.08, 0.10, 0.15)
DEFAULT_PCT = 0.10
ROLL = 60                 # sessions in the rolling correlation
RECENT = 20               # sessions in "lately"
CHART_EVERY = 5           # sessions per chart point (weekly)


def _num(v, nd: int = 4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)


def ratio(equity: pd.Series, bond: pd.Series) -> pd.DataFrame:
    """Both rebased to 100 on their first shared day, and their ratio (×100)."""
    df = pd.concat([pd.to_numeric(equity, errors="coerce").rename("e"),
                    pd.to_numeric(bond, errors="coerce").rename("b")], axis=1, join="inner")
    df = df[(df["e"] > 0) & (df["b"] > 0)].dropna().sort_index()
    if len(df) < ROLL + 2:
        raise LookupError("股票和债券的共同交易日太少")
    e = df["e"] / df["e"].iloc[0] * 100
    b = df["b"] / df["b"].iloc[0] * 100
    return pd.DataFrame({"e": e, "b": b, "r": e / b * 100})


def _leg_stats(d: pd.DataFrame, leg: dict) -> dict:
    a, b = pd.Timestamp(leg["from"]), pd.Timestamp(leg["to"])
    s = d.loc[a:b]
    pct = lambda col: _num((s[col].iloc[-1] / s[col].iloc[0] - 1) * 100, 1)  # noqa: E731
    return {
        "from": leg["from"], "to": leg["to"], "sessions": leg["sessions"],
        # Up legs of the ratio are stocks winning; down legs, bonds.
        "side": "equity" if leg["kind"] == "up" else "bond",
        "ratio_pct": leg["change_pct"], "equity_pct": pct("e"), "bond_pct": pct("b"),
        "confirmed": leg["confirmed"], "open": leg["open"],
    }


def to_turn(r: pd.Series, leg: dict, pct: float) -> dict:
    """
    For the open leg: the extreme it has reached, and how far the ratio has
    to move from today for the opposite turn to be confirmed.
    """
    s = r.loc[pd.Timestamp(leg["from"]):]
    now = float(s.iloc[-1])
    if leg["side"] == "equity":
        peak = float(s.max())
        line = peak * (1 - pct)
        return {"extreme": _num(peak, 3), "extreme_date": str(s.idxmax().date()),
                "from_extreme": _num(now / peak - 1), "line": _num(line, 3),
                "need": _num(line / now - 1)}           # negative: a fall to confirm
    trough = float(s.min())
    line = trough * (1 + pct)
    return {"extreme": _num(trough, 3), "extreme_date": str(s.idxmin().date()),
            "from_extreme": _num(now / trough - 1), "line": _num(line, 3),
            "need": _num(line / now - 1)}               # positive: a rise to confirm


def build(equity: pd.Series, bond: pd.Series, *, pct: float = DEFAULT_PCT) -> dict:
    d = ratio(equity, bond)
    r = d["r"]
    legs = [_leg_stats(d, leg) for leg in pw.regimes(r, pct)]
    if not legs:
        raise LookupError("股债比没有可划分的阶段")
    # regimes() dates each leg's END confirmation; a leg's START was known
    # when the leg before it was confirmed over.
    ends = [leg.pop("confirmed") for leg in legs]
    for i, leg in enumerate(legs):
        leg["start_confirmed"] = ends[i - 1] if i else None
        leg["end_confirmed"] = ends[i]
    cur = legs[-1]

    rets = d[["e", "b"]].pct_change(fill_method=None)
    corr = rets["e"].rolling(ROLL).corr(rets["b"])
    corr_now = _num(corr.iloc[-1], 3)
    hist = corr.dropna()

    closed = [x for x in legs if not x["open"]]

    def med(side):
        v = [x["sessions"] for x in closed if x["side"] == side]
        return int(np.median(v)) if v else None

    keep = list(range(0, len(d), CHART_EVERY))
    if keep[-1] != len(d) - 1:
        keep.append(len(d) - 1)
    return {
        "pct": pct,
        "from": str(d.index[0].date()), "to": str(d.index[-1].date()),
        "now": {
            "ratio": _num(r.iloc[-1], 3), "side": cur["side"], "since": cur["from"],
            "sessions": cur["sessions"], "ratio_pct": cur["ratio_pct"],
            "equity_pct": cur["equity_pct"], "bond_pct": cur["bond_pct"],
            "confirmed": cur["start_confirmed"],
            "recent_pct": _num((r.iloc[-1] / r.iloc[-1 - RECENT] - 1) * 100, 2)
                          if len(r) > RECENT else None,
            "corr": corr_now,
            "corr_pct": _num((hist <= corr_now).mean(), 3) if corr_now is not None and len(hist) else None,
            **to_turn(r, cur, pct),
        },
        "typical": {"equity": med("equity"), "bond": med("bond")},
        "legs": legs,
        "chart": {
            "dates": [str(d.index[i].date()) for i in keep],
            "ratio": [_num(r.iloc[i], 3) for i in keep],
            "equity": [_num(d["e"].iloc[i], 2) for i in keep],
            "bond": [_num(d["b"].iloc[i], 2) for i in keep],
            "corr": [_num(corr.iloc[i], 3) for i in keep],
        },
    }


__all__ = ["DEFAULT_PCT", "PAIRS", "PCTS", "ROLL", "build", "ratio", "to_turn"]
