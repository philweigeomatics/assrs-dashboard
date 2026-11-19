"""
fair_value.py — 估值: what the market paid for this stock's growth in normal
times, applied to today.

The idea
--------
A stock's price is its earnings times a multiple, and the multiple is mostly
two things: how fast the earnings are expected to grow, and the market's
mood. The mood swings the multiple of everything at once — the 2015 bubble,
early 2021, the weeks after 24 September 2024 on the way up; 2018 and early
2024 on the way down — and a fair value read off those days inherits it.

So the mood is taken out first, at the market level:

  1. 沪深300's PE_TTM over the last ten years stands for the market. Days in
     its top 15% or bottom 15% are dropped (percentile of the ten-year
     sample, not a distance from the mean: the cut is the same share of days
     however skewed the series is). What is left are NORMAL days.

  2. On the normal days, the stock's PE divided by the earnings growth the
     market could see that day (PEG — growth counted from the day after each
     report was announced, see price_zones.profit_growth) is the price the
     market paid per point of growth when it was neither euphoric nor
     panicking. Its middle half (25th–75th percentile) is the stock's normal
     price of growth. Faster growth in a period already raises the PE it
     justifies, so that period is not "expensive" for growing faster.

  3. Today:
       fair PE     = normal PEG × today's growth (percent)
       fair price  = today's price × fair PE / today's PE
                     (price / PE is earnings per share, so this is EPS × fair PE)
       implied g   = today's PE / normal PEG
                     — the growth today's price is only fair at, which needs no
                     current growth at all.
     Verdict: below the band 低估, inside 合理, above 高估.

Where it does not hold, the payload says so instead of printing a number:
growth under 5% (PEG is arithmetic there, not a valuation), a loss (no PE),
too few normal days with a PEG, and cyclical earnings — when profit growth
keeps changing sign, the PE is lowest at the peak of the cycle and a PEG
"bargain" is the top.

"Fair" is what the market used to pay for this growth, not intrinsic value.
If the company has changed, its own history misleads.

Pure: the series are passed in; the API fetches them.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

YEARS = 10
MARKET_CODE = "000300.SH"
MARKET_LABEL = "沪深300"
CUT = 0.15                 # dropped from each end of the market's PE
MIN_GROWTH = 0.05          # PEG only where earnings grow at least 5% a year
MIN_NORMAL = 120           # normal sessions with a value, for a band at all
SOLID_NORMAL = 250         # ... and for one worth calling reliable
QUANTS = (0.25, 0.5, 0.75)
CYCLE_FLIPS = 3            # growth sign changes over the reports = cyclical
CHART_EVERY = 5            # sessions per chart point (weekly)
MIN_RUN = 5                # sessions: a shorter crossing of a cut-off is noise, not a regime
BRIDGE = 10                # sessions: a shorter return to normal between two spells is one spell
SLOWED_BY = 0.05           # growth this much below the normal days' = the normal PE overstates


def start_date(today: date, years: int = YEARS) -> date:
    return today - timedelta(days=round(365.25 * years))


def _num(v, nd: int = 4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _series(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


# ── 1. the market's normal days ──────────────────────────────────────────────
def market_band(market_pe: pd.Series, cut: float = CUT) -> dict:
    """
    The market's PE cut-offs and, per day, whether it was normal.

    Percentiles of the whole sample: the question is "was this one of the
    market's most expensive (or cheapest) days of the decade", which is a
    rank, not a distance from the average.
    """
    m = _series(market_pe).dropna()
    m = m[m > 0]
    if len(m) < 250:
        raise LookupError(f"{MARKET_LABEL} 市盈率历史不足一年，无法划分正常时期")
    lo, hi = float(m.quantile(cut)), float(m.quantile(1 - cut))
    state = smooth_state(pd.Series(np.where(m > hi, 1, np.where(m < lo, -1, 0)), index=m.index))
    now = float(m.iloc[-1])
    return {
        "lo": lo, "hi": hi, "series": m, "state": state,
        "now": now, "now_pct": float((m <= now).mean()),
        "now_state": int(state.iloc[-1]),
    }


def _runs(v: np.ndarray) -> list[tuple[int, int, int]]:
    """(value, first, last) for each run of equal values."""
    out, a = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[a]:
            out.append((int(v[a]), a, i - 1))
            a = i
    return out


def smooth_state(state: pd.Series, min_run: int = MIN_RUN, bridge: int = BRIDGE) -> pd.Series:
    """
    Regimes, not days. The market's PE crosses a percentile line back and
    forth for a day or two at a time; a crossing shorter than `min_run`
    sessions is not a regime and stays normal, and a return to normal shorter
    than `bridge` between two spells of the same kind joins them.
    """
    v = state.to_numpy().astype(int).copy()
    for s, a, b in _runs(v):
        if s != 0 and b - a + 1 < min_run:
            v[a:b + 1] = 0
    runs = _runs(v)
    for k, (s, a, b) in enumerate(runs):
        if s == 0 and 0 < k < len(runs) - 1 and b - a + 1 < bridge                 and runs[k - 1][0] == runs[k + 1][0] != 0:
            v[a:b + 1] = runs[k - 1][0]
    return pd.Series(v, index=state.index)


def excluded_runs(state: pd.Series) -> list[dict]:
    """Stretches the market spent above (high) or below (low) its normal band."""
    out, cur, a, prev = [], 0, None, None
    for d, s in state.items():
        if s != cur:
            if cur != 0:
                out.append({"from": a.strftime("%Y-%m-%d"), "to": prev.strftime("%Y-%m-%d"),
                            "kind": "high" if cur > 0 else "low"})
            cur, a = s, d
        prev = d
    if cur != 0 and a is not None:
        out.append({"from": a.strftime("%Y-%m-%d"), "to": prev.strftime("%Y-%m-%d"),
                    "kind": "high" if cur > 0 else "low"})
    return out


# ── 2. the stock's price of growth on normal days ────────────────────────────
def _peg(pe: np.ndarray, g: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pe / (g * 100)
    out[~((pe > 0) & (g >= MIN_GROWTH))] = np.nan
    return out


def _band(vals: np.ndarray) -> dict | None:
    v = vals[~np.isnan(vals)]
    if len(v) < MIN_NORMAL:
        return None
    q = np.quantile(v, QUANTS)
    return {"lo": float(q[0]), "mid": float(q[1]), "hi": float(q[2]), "n": int(len(v))}


def cyclical(growth: pd.DataFrame | None) -> dict:
    """
    Earnings that keep turning: TTM growth changing sign CYCLE_FLIPS times or
    more over the reports, or a TTM loss among them. Steel, chemicals,
    shipping, hogs — their PE is lowest at the top of the cycle.
    """
    if growth is None or growth.empty:
        return {"flag": False, "flips": 0, "losses": 0}
    g = pd.to_numeric(growth["g1"], errors="coerce").dropna()
    signs = np.sign(g.to_numpy())
    signs = signs[signs != 0]
    flips = int((signs[1:] != signs[:-1]).sum()) if len(signs) > 1 else 0
    losses = int((pd.to_numeric(growth["ttm"], errors="coerce") <= 0).sum())
    return {"flag": bool(flips >= CYCLE_FLIPS or losses > 0), "flips": flips, "losses": losses}


def build(*, pe: pd.Series, close: pd.Series, growth: pd.DataFrame | None,
          market_pe: pd.Series, forecast: dict | None = None) -> dict:
    """
    The 估值 payload. `pe` and `close` are the stock's daily PE_TTM and
    unadjusted close (the price that PE is computed on), `growth` is
    price_zones.profit_growth over the reports, `market_pe` is 沪深300's
    daily PE_TTM. `forecast` is the company's latest 业绩预告 for a period
    not yet reported: {"period", "growth"} or None.
    """
    mk = market_band(market_pe)

    p = _series(pe)
    c = _series(close).reindex(p.index)
    if p.dropna().empty or c.dropna().empty:
        raise LookupError("没有市盈率数据")
    days = p.index
    # The market's state on each of the stock's sessions (its own calendar;
    # a day the index has no reading for keeps the last one).
    state = mk["state"].reindex(mk["state"].index.union(days)).ffill().reindex(days)
    normal = (state == 0).to_numpy()

    pe_arr = p.to_numpy(dtype=float)
    pe_arr = np.where(pe_arr > 0, pe_arr, np.nan)          # a loss has no PE
    g1 = g3 = np.full(len(days), np.nan)
    g_now = {"g1": None, "g3": None, "period": None}
    if growth is not None and not growth.empty:
        al = growth.reindex(growth.index.union(days)).ffill().reindex(days)
        g1 = al["g1"].to_numpy(dtype=float)
        g3 = al["g3"].to_numpy(dtype=float)
        last = al.iloc[-1]
        g_now = {"g1": _num(last["g1"]), "g3": _num(last["g3"]),
                 "period": None if pd.isna(last["end_date"]) else str(last["end_date"])}

    # "Now" only if the last PE is from the last few sessions: a company
    # that has gone loss-making has no current PE, and a stale one misleads.
    ok = np.flatnonzero(~np.isnan(pe_arr))
    pe_now = float(pe_arr[ok[-1]]) if len(ok) and ok[-1] >= len(pe_arr) - 5 else None
    px_now = float(c.dropna().iloc[-1])

    def price(fair_pe):
        return px_now * fair_pe / pe_now if pe_now and fair_pe else None

    methods: dict[str, dict] = {}
    # Plain PE: no growth in it, so it answers for a loss-free company that
    # grows too slowly for PEG — and is the baseline the PEG reads beat.
    pe_band = _band(np.where(normal, pe_arr, np.nan))
    methods["pe"] = _method_pe(pe_band, pe_arr, normal, price)
    for key, g in (("peg1", g1), ("peg3", g3)):
        gn = g_now["g1" if key == "peg1" else "g3"]
        methods[key] = _method_peg(_band(np.where(normal, _peg(pe_arr, g), np.nan)),
                                   g, normal, gn, pe_now, price)

    # The three-year rate first: one exceptional year does not swing it.
    primary = next((k for k in ("peg3", "peg1") if methods[k]["fair"]), None) \
        or ("pe" if methods["pe"]["fair"] else None)

    # A second reading that needs no current growth above 5%: the growth
    # today's price implies at the normal PEG, against what was delivered.
    implied = None
    ik = next((k for k in ("peg3", "peg1") if methods[k]["implied"]), None)
    if ik:
        im = methods[ik]["implied"]
        actual = g_now["g3" if ik == "peg3" else "g1"]
        implied = {"method": ik, **im, "actual": actual,
                   "call": None if actual is None else
                           "over" if actual < im["lo"] else "under" if actual > im["hi"] else "fair"}

    # Plain PE carries the growth of the days it was paid on. When the
    # company grows far slower now, the normal PE overstates its worth.
    then = methods["peg3"]["growth_then"] or methods["peg1"]["growth_then"]
    now_g = g_now["g3"] if g_now["g3"] is not None else g_now["g1"]
    methods["pe"]["slowed"] = bool(then is not None and now_g is not None
                                   and now_g < then - SLOWED_BY)
    methods["pe"]["growth_then"] = then

    verdict = None
    if primary:
        f = methods[primary]["fair"]
        call = "under" if px_now < f["lo"] else "over" if px_now > f["hi"] else "fair"
        # PE alone on a company that has slowed down, against a growth
        # reading that says the opposite: no verdict, both readings shown.
        if primary == "pe" and methods["pe"]["slowed"] and implied                 and implied["call"] and implied["call"] != call:
            call = "mixed"
        verdict = {
            "method": primary,
            "call": call,
            "gap_mid": _num(px_now / f["mid"] - 1),
            "to_lo": _num(f["lo"] / px_now - 1), "to_hi": _num(f["hi"] / px_now - 1),
        }

    # Scenarios: the same normal price of growth on each growth assumption.
    peg_key = next((k for k in ("peg3", "peg1") if methods[k]["band"]), None)
    scenarios = []
    if peg_key and pe_now:
        b = methods[peg_key]["band"]
        cands = [("g3", "三年复合增速", g_now["g3"], None),
                 ("g1", "近一年增速", g_now["g1"], None)]
        if forecast and forecast.get("growth") is not None:
            cands.append(("forecast", "公司业绩预告", forecast["growth"], forecast.get("period")))
        for key, label, g, period in cands:
            row = {"key": key, "label": label, "growth": _num(g), "period": period,
                   "lo": None, "mid": None, "hi": None, "upside": None}
            if g is not None and g >= MIN_GROWTH:
                row.update({k: _num(price(b[k] * g * 100), 3) for k in ("lo", "mid", "hi")})
                row["upside"] = _num(row["mid"] / px_now - 1) if row["mid"] else None
            scenarios.append(row)

    n_peg = max((methods[k]["band"] or {}).get("n", 0) for k in ("peg1", "peg3"))
    every = slice(None, None, CHART_EVERY)
    keep = list(range(len(days)))[every]
    if keep and keep[-1] != len(days) - 1:
        keep.append(len(days) - 1)
    mser = mk["series"]
    return {
        "market": {
            "code": MARKET_CODE, "label": MARKET_LABEL, "cut": CUT,
            "lo": _num(mk["lo"], 2), "hi": _num(mk["hi"], 2),
            "now": _num(mk["now"], 2), "now_pct": _num(mk["now_pct"], 3),
            "now_state": mk["now_state"],
            "from": mser.index[0].strftime("%Y-%m-%d"), "to": mser.index[-1].strftime("%Y-%m-%d"),
            "excluded": excluded_runs(mk["state"]),
        },
        "start": days[0].strftime("%Y-%m-%d"), "end": days[-1].strftime("%Y-%m-%d"),
        "sessions": int(len(days)), "normal_sessions": int(normal.sum()),
        "now": {"price": _num(px_now, 3), "pe": _num(pe_now, 2), **g_now},
        "methods": methods, "primary": primary, "verdict": verdict, "implied": implied,
        "scenarios": scenarios, "scenario_basis": peg_key,
        "cyclical": cyclical(growth),
        "confidence": _confidence(primary, methods),
        "chart": {
            "dates": [days[i].strftime("%Y-%m-%d") for i in keep],
            "market_pe": [_num(v, 2) for v in mser.reindex(mser.index.union(days)).ffill()
                          .reindex(days).to_numpy()[keep]],
            "pe": [_num(pe_arr[i], 2) for i in keep],
            "peg1": [_num(v, 3) for v in _peg(pe_arr, g1)[keep]],
            "peg3": [_num(v, 3) for v in _peg(pe_arr, g3)[keep]],
            "normal": [bool(normal[i]) for i in keep],
        },
    }


def _confidence(primary, methods) -> str:
    if not primary:
        return "none"
    return "ok" if (methods[primary]["band"] or {}).get("n", 0) >= SOLID_NORMAL else "low"


def _pack(band):
    return None if band is None else {k: _num(band[k], 3) for k in ("lo", "mid", "hi")} | {"n": band["n"]}


def _method_pe(band, pe_arr, normal, price) -> dict:
    fair = None
    if band:
        prices = {k: price(band[k]) for k in ("lo", "mid", "hi")}
        if all(prices.values()):
            fair = {k: _num(v, 3) for k, v in prices.items()}
    return {"band": _pack(band), "fair": fair, "implied": None,
            "why": None if band else f"正常时期有市盈率的交易日不足 {MIN_NORMAL} 天"}


def _method_peg(band, g, normal, g_now, pe_now, price) -> dict:
    out = {"band": _pack(band), "fair": None, "implied": None, "why": None,
           # The growth the market was paying for on those days — the
           # context a PEG needs: 1.0 on 10% growth is not 1.0 on 40%.
           "growth_then": _num(np.nanmedian(np.where(normal & (g >= MIN_GROWTH), g, np.nan)))
                          if np.any(normal & (g >= MIN_GROWTH)) else None}
    if band is None:
        out["why"] = f"正常时期增速 ≥ {MIN_GROWTH:.0%} 且盈利的交易日不足 {MIN_NORMAL} 天"
        return out
    if pe_now:
        # Low PEG ↔ high implied growth: the cheaper the market used to price
        # growth, the more growth today's PE has to be carrying.
        out["implied"] = {"lo": _num(pe_now / (band["hi"] * 100)),
                          "mid": _num(pe_now / (band["mid"] * 100)),
                          "hi": _num(pe_now / (band["lo"] * 100))}
    else:
        out["why"] = "现在亏损，没有市盈率"
        return out
    if g_now is None or g_now < MIN_GROWTH:
        out["why"] = (f"现在的增速 {g_now:+.1%} 低于 {MIN_GROWTH:.0%}，PEG 不适用"
                      if g_now is not None else "没有现在的增速")
        return out
    out["fair"] = {k: _num(price(band[k] * g_now * 100), 3) for k in ("lo", "mid", "hi")}
    return out


def forecast_growth(rows: pd.DataFrame | None, last_reported: str | None) -> dict | None:
    """
    The company's own 业绩预告 for a period after the latest report, as a
    growth figure: the midpoint of its stated profit change. It is the
    year-to-date change the notice gives, not a twelve-month rate, and the
    page labels it so.
    """
    if rows is None or rows.empty:
        return None
    r = rows.copy()
    r["end_date"] = r["end_date"].astype(str).str[:8]
    if last_reported:
        r = r[r["end_date"] > str(last_reported)]
    r = r.dropna(subset=["p_change_min", "p_change_max"], how="all")
    if r.empty:
        return None
    r = r.sort_values(["end_date", "ann_date"]).iloc[-1]
    lo, hi = pd.to_numeric(r["p_change_min"], errors="coerce"), pd.to_numeric(r["p_change_max"], errors="coerce")
    vals = [v for v in (lo, hi) if pd.notna(v)]
    if not vals:
        return None
    return {"period": r["end_date"], "growth": _num(float(np.mean(vals)) / 100),
            "ann_date": str(r["ann_date"])[:8], "type": str(r.get("type") or "")}


__all__ = ["CUT", "MARKET_CODE", "MARKET_LABEL", "YEARS", "build", "cyclical",
           "excluded_runs", "forecast_growth", "market_band", "start_date"]
