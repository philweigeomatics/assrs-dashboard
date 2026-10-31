"""
custom_index.py — 自建指数: a rule you watch, not a book you hold.

An index here is a list of members with dates, a weighting method and a
rebalance schedule. Its level is computed the way an index provider computes
one: as a portfolio of units that only changes at rebalances and membership
changes, so the level is continuous across every one of them by construction
— the divisor bookkeeping of a real index, without the divisor.

Everything in this file is pure: closes, market caps and member rows in, the
level and its explanation out. Fetching, persistence and the decision that a
stock has been delisted live in api/index_api.py.

THE RULES OF THE LEDGER

  * Changes take effect at a close. A member added on day d is bought at d's
    close and moves the index from d+1; one removed on d rides d's move and
    is sold at d's close. The same holds for a scheduled rebalance, which
    happens on the first session of each new period.
  * A halted stock (停牌) is carried at its last price and cannot be traded:
    a rebalance leaves its units alone and reweights the rest around it, and
    a stock added while halted joins on the first day it trades again.
  * A member removed for any reason — by you, or because it stopped trading
    after an acquisition — leaves at its last price, and the proceeds go to
    the remaining members. Under a rebalancing method that is a full
    reweight; under buy-and-hold it is pro-rata, so what you chose not to
    rebalance is not rebalanced behind your back.
  * Prices are adjusted for splits and dividends, so the level is a total
    return index (dividends reinvested), as most ETFs are benchmarked.

WHY UNITS, NOT WEIGHTED RETURNS

Multiplying daily returns by fixed weights is a portfolio rebalanced every
day — a different index from the one described, and for volatile members a
noticeably different number (the 板块接力 work found a daily-rebalanced
basket at +731% against +603% held). Units drift with prices between
rebalances, which is what an equal-weight index actually does.

The same units make contribution exact: a member's share of a move is its
units times its price change, and those shares sum to the index's move with
nothing left over.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

METHODS: dict[str, tuple[str, str]] = {
    "equal": ("等权", "每次调仓把每只股票重设为相同权重；两次调仓之间随涨跌漂移"),
    "equal_hold": ("等资金买入持有", "起始日每只投入相同资金，之后不调仓，涨得多的自然权重变大"),
    "cap": ("市值加权", "按市值分配，大公司权重大（A 股用流通市值）"),
    "cap_capped": ("市值加权·单只上限", "按市值分配，但单只不超过上限，超出部分按比例分给其余"),
    "price": ("价格加权", "按股价分配，股价高的权重大（道琼斯的做法）"),
    "custom": ("自定权重", "按你设定的权重，每次调仓恢复"),
}

REBALANCE: dict[str, str] = {"M": "每月", "Q": "每季", "H": "每半年", "Y": "每年", "never": "不调仓"}

#: Methods whose weights drift and are never reset.
HOLD_METHODS = {"equal_hold"}

DEFAULT_CAP = 0.10
BASE_VALUE = 1000.0

WINDOWS = {"1d": 1, "5d": 5, "1m": 21, "3m": 63, "1y": 252}


def _f(v, nd=2):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


# ── weights ──────────────────────────────────────────────────────────────────
def capped(raw: dict[str, float], cap: float) -> dict[str, float]:
    """
    Proportional weights with no name above `cap`: the excess of every name
    over the cap goes to the uncapped ones in proportion, repeated until
    nothing is over — the way capped indices (Nasdaq-100, sector SPDRs) do it.
    A cap that cannot hold (five names at 10% sum to 50%) is equal weight.
    """
    names = [t for t, v in raw.items() if v > 0]
    if not names:
        return {}
    if cap * len(names) <= 1 + 1e-9:
        return {t: 1 / len(names) for t in names}
    fixed: dict[str, float] = {}
    while True:
        free = [t for t in names if t not in fixed]
        left = 1 - cap * len(fixed)
        tot = sum(raw[t] for t in free)
        w = {t: left * raw[t] / tot for t in free}
        over = [t for t in free if w[t] > cap + 1e-12]
        if not over:
            return {**{t: cap for t in fixed}, **w}
        for t in over:
            fixed[t] = cap


def targets(method: str, names: list[str], *, price: pd.Series, mcap: pd.Series | None,
            custom: dict[str, float], cap: float) -> tuple[dict[str, float], list[str]]:
    """Target weights for `names` on one day, and the names whose cap had to be guessed."""
    if not names:
        return {}, []
    guessed: list[str] = []
    if method in ("equal", "equal_hold"):
        raw = {t: 1.0 for t in names}
    elif method == "price":
        raw = {t: float(price[t]) for t in names}
    elif method == "custom":
        raw = {t: float(custom.get(t) or 0) for t in names}
        if sum(raw.values()) <= 0:
            raw = {t: 1.0 for t in names}
    else:
        known = {t: float(mcap[t]) for t in names
                 if mcap is not None and t in mcap.index and pd.notna(mcap[t]) and mcap[t] > 0}
        fill = float(np.mean(list(known.values()))) if known else 1.0
        guessed = [t for t in names if t not in known]
        raw = {t: known.get(t, fill) for t in names}
    if method == "cap_capped":
        return capped(raw, cap), guessed
    tot = sum(raw.values())
    return ({t: v / tot for t, v in raw.items()} if tot > 0 else {t: 1 / len(names) for t in names}), guessed


def period_key(d: pd.Timestamp, freq: str):
    if freq == "M":
        return (d.year, d.month)
    if freq == "Q":
        return (d.year, (d.month - 1) // 3)
    if freq == "H":
        return (d.year, (d.month - 1) // 6)
    if freq == "Y":
        return d.year
    return None


def _member_on(rows: list[dict], d: pd.Timestamp) -> bool:
    for r in rows:
        start = pd.Timestamp(r["added_on"])
        end = pd.Timestamp(r["removed_on"]) if r.get("removed_on") else None
        if start <= d and (end is None or d < end):
            return True
    return False


# ── the ledger ───────────────────────────────────────────────────────────────
def build(closes: pd.DataFrame, members: list[dict], *, method: str, rebalance: str,
          base_date, base_value: float = BASE_VALUE, cap: float = DEFAULT_CAP,
          caps: pd.DataFrame | None = None) -> dict:
    """
    Run the index from `base_date` over the dates in `closes`.

    `closes` — adjusted closes on the market's calendar, NaN where a stock did
    not trade (before listing, halted, after delisting). `members` — rows of
    {ticker, added_on, removed_on, weight_pct}; a ticker may have several rows
    if it left and came back.

    Returns the level, the end-of-day weights, every member's daily value
    change (for contribution), and the change log.
    """
    if method not in METHODS:
        raise ValueError(f"未知的加权方式：{method}")
    if rebalance not in REBALANCE:
        raise ValueError(f"未知的调仓频率：{rebalance}")
    base = pd.Timestamp(base_date)
    closes = closes.sort_index()
    closes = closes.loc[closes.index >= base]
    if closes.empty:
        raise LookupError("起始日之后没有行情")

    by_t: dict[str, list[dict]] = {}
    for m in members:
        by_t.setdefault(m["ticker"], []).append(m)
    tickers = [t for t in by_t if t in closes.columns]
    traded = closes[tickers].notna()
    px = closes[tickers].ffill()
    custom = {t: max(float(r.get("weight_pct") or 0) for r in rows) for t, rows in by_t.items()}
    mcap = caps.reindex(closes.index).ffill() if caps is not None else None
    hold = method in HOLD_METHODS or rebalance == "never"

    dates = closes.index
    n = len(dates)
    level = np.full(n, np.nan)
    weights = np.zeros((n, len(tickers)))
    dv = np.zeros((n, len(tickers)))
    col = {t: i for i, t in enumerate(tickers)}
    units: dict[str, float] = {}
    cash = 0.0
    events: list[dict] = []
    guessed_caps: set[str] = set()
    started = False

    for k, d in enumerate(dates):
        row = px.loc[d]
        if k > 0:
            prev = px.loc[dates[k - 1]]
            for t, u in units.items():
                dv[k, col[t]] = u * (row[t] - prev[t])
        value = cash + sum(u * row[t] for t, u in units.items())
        if not started:
            value = base_value

        # Who belongs today: a member on this date, with a price — and a
        # newcomer only once it actually trades.
        active = {t for t in tickers if _member_on(by_t[t], d) and pd.notna(row[t])
                  and (t in units or traded.at[d, t])}
        leaving = set(units) - active
        joining = active - set(units)
        scheduled = (started and not hold and k > 0
                     and period_key(d, rebalance) != period_key(dates[k - 1], rebalance))

        if not started and not active:
            continue                                   # nothing to hold yet
        if not started or leaving or joining or scheduled:
            before = {t: u * row[t] / value for t, u in units.items()} if value > 0 else {}
            for t in sorted(leaving):
                cash += units.pop(t) * row[t]
                why = "delisted" if any(r.get("removal") == "delisted" and r.get("removed_on")
                                        and pd.Timestamp(r["removed_on"]) <= d for r in by_t[t]) else "user"
                events.append({"date": str(d.date()), "kind": "remove", "ticker": t, "reason": why})
            frozen = {t for t in units if not traded.at[d, t]}
            frozen_value = sum(units[t] * row[t] for t in frozen)
            names = sorted(t for t in active if t not in frozen)
            price_row = row
            cap_row = mcap.loc[d] if mcap is not None else None

            if not started or not hold:
                w, g = targets(method, names, price=price_row, mcap=cap_row, custom=custom, cap=cap)
                guessed_caps.update(g)
                investable = value - frozen_value
                for t in list(units):
                    if t not in frozen:
                        units.pop(t)
                for t in names:
                    units[t] = investable * w[t] / row[t]
                cash = 0.0 if names else investable
            else:
                # Buy-and-hold: newcomers get their target share of the whole;
                # everyone already held is scaled together to fund them (or to
                # absorb a leaver's proceeds). Nobody else is reset.
                w, g = targets(method, sorted(active), price=price_row, mcap=cap_row,
                               custom=custom, cap=cap)
                guessed_caps.update(g)
                buy = {t: value * w[t] for t in joining if t not in frozen}
                stay = [t for t in units if t not in frozen]
                stay_value = sum(units[t] * row[t] for t in stay)
                room = value - frozen_value - sum(buy.values())
                if stay and stay_value > 0:
                    f = room / stay_value
                    for t in stay:
                        units[t] *= f
                    cash = 0.0
                else:
                    cash = room
                for t, v in buy.items():
                    units[t] = v / row[t]

            if not started:
                events.append({"date": str(d.date()), "kind": "start", "n": len(names)})
            else:
                for t in sorted(joining):
                    events.append({"date": str(d.date()), "kind": "add", "ticker": t})
                if scheduled and not hold:
                    after = {t: u * row[t] / value for t, u in units.items()}
                    turn = sum(abs(after.get(t, 0) - before.get(t, 0)) for t in set(after) | set(before)) / 2
                    events.append({"date": str(d.date()), "kind": "rebalance", "turnover": _f(turn * 100, 1)})
            started = True

        level[k] = value
        if value > 0:
            for t, u in units.items():
                weights[k, col[t]] = u * row[t] / value

    keep = ~np.isnan(level)
    if not keep.any():
        raise LookupError("起始日之后没有任何成分股有行情")
    idx = dates[keep]
    return {
        "level": pd.Series(level[keep], idx),
        "weights": pd.DataFrame(weights[keep], idx, tickers),
        "dv": pd.DataFrame(dv[keep], idx, tickers),
        "traded": traded.loc[idx],
        "held": sorted(units),
        "events": events,
        "guessed_caps": sorted(guessed_caps),
        "hold": hold,
    }


# ── reading it ───────────────────────────────────────────────────────────────
def contribution(level: pd.Series, dv: pd.DataFrame, days: int | None) -> dict[str, float]:
    """
    Each member's share of the index move over the last `days` sessions, in
    percentage points of the starting level. Sums to the index's return.
    """
    if days is None or days >= len(level):
        start = 0
    else:
        start = len(level) - 1 - days
    v0 = float(level.iloc[start])
    part = dv.iloc[start + 1:].sum()
    return {t: float(v) / v0 * 100 for t, v in part.items()} if v0 > 0 else {}


def _ret(s: pd.Series, k: int) -> float | None:
    s = s.dropna()
    return (s.iloc[-1] / s.iloc[-1 - k] - 1) * 100 if len(s) > k else None


def _ytd(s: pd.Series) -> float | None:
    s = s.dropna()
    prev = s[s.index.year < s.index[-1].year]
    if prev.empty:
        return None
    return (s.iloc[-1] / prev.iloc[-1] - 1) * 100


def stats(level: pd.Series) -> dict:
    r = level.pct_change().dropna()
    peak = level.cummax()
    dd = level / peak - 1
    ma200 = level.rolling(200).mean()
    return {
        "level": _f(level.iloc[-1], 2),
        "ret": {**{k: _f(_ret(level, v), 2) for k, v in WINDOWS.items()},
                "ytd": _f(_ytd(level), 2), "all": _f((level.iloc[-1] / level.iloc[0] - 1) * 100, 2)},
        "vol": _f(r.tail(252).std() * math.sqrt(252) * 100, 1) if len(r) > 20 else None,
        "max_dd": _f(dd.min() * 100, 1),
        "max_dd_date": str(dd.idxmin().date()) if len(dd) else None,
        "dd_now": _f(dd.iloc[-1] * 100, 1),
        "d200": _f((level.iloc[-1] / ma200.iloc[-1] - 1) * 100, 1) if pd.notna(ma200.iloc[-1]) else None,
    }


def next_rebalance(last: pd.Timestamp, freq: str) -> str | None:
    if freq == "never":
        return None
    step = {"M": 1, "Q": 3, "H": 6, "Y": 12}[freq]
    month0 = (last.month - 1) // step * step          # first month of this period, 0-based
    m = month0 + step
    y = last.year + m // 12
    return str(date(y, m % 12 + 1, 1))


def summarise(run: dict, *, members: list[dict], names: dict[str, str], method: str,
              rebalance: str, cap: float, bench: pd.Series | None, live_from,
              closes: pd.DataFrame, caps: pd.DataFrame | None) -> dict:
    level: pd.Series = run["level"]
    last = level.index[-1]
    w_now = run["weights"].iloc[-1]
    traded_now = run["traded"].iloc[-1]
    held = set(run["held"])

    # Where the weights would be reset to at the next rebalance — the drift
    # is the gap between that and today. Meaningless for buy-and-hold.
    tgt: dict[str, float] = {}
    if not run["hold"]:
        names_now = sorted(t for t in held if traded_now.get(t, False))
        px_now = closes.ffill().loc[last]
        cap_now = caps.reindex(closes.index).ffill().loc[last] if caps is not None else None
        custom = {m["ticker"]: float(m.get("weight_pct") or 0) for m in members}
        tgt, _ = targets(method, names_now, price=px_now, mcap=cap_now, custom=custom, cap=cap)

    contrib = {k: contribution(level, run["dv"], v) for k, v in
               {"1d": 1, "5d": 5, "1m": 21, "3m": 63, "all": None}.items()}
    rows = []
    for m in members:
        t = m["ticker"]
        s = closes[t].dropna() if t in closes.columns else pd.Series(dtype=float)
        removed = m.get("removed_on")
        if t in held:
            status = "active" if traded_now.get(t, False) else "halted"
        elif removed and pd.Timestamp(removed) <= last:
            status = "delisted" if m.get("removal") == "delisted" else "removed"
        elif s.empty:
            status = "nodata"
        else:
            status = "pending"
        rows.append({
            "id": m.get("id"), "ticker": t, "name": names.get(t) or m.get("name") or t,
            "status": status,
            "weight": _f(w_now.get(t, 0) * 100, 2) if t in held else None,
            "target": _f(tgt[t] * 100, 2) if t in tgt else None,
            "custom": _f(m.get("weight_pct"), 2),
            "price": _f(s.iloc[-1], 2) if len(s) else None,
            "ret_1d": _f(_ret(s, 1), 2) if t in held else None,
            "contrib": {k: _f(c.get(t), 2) for k, c in contrib.items()},
            "added_on": str(m["added_on"])[:10], "removed_on": str(removed)[:10] if removed else None,
            "exit_price": _f(m.get("exit_price"), 2),
        })
    order = {"active": 0, "halted": 1, "pending": 2, "nodata": 3, "removed": 4, "delisted": 5}
    rows.sort(key=lambda r: (order[r["status"]], -(r["weight"] or 0), r["ticker"]))

    b = None
    if bench is not None and not bench.dropna().empty:
        bb = bench.reindex(level.index).ffill().dropna()
        if len(bb):
            reb = bb / bb.iloc[0] * float(level.loc[bb.index[0]])
            b = {"series": [_f(v, 2) for v in reb.reindex(level.index).to_numpy()],
                 "ret": {**{k: _f(_ret(bb, v), 2) for k, v in WINDOWS.items()},
                         "ytd": _f(_ytd(bb), 2), "all": _f((bb.iloc[-1] / bb.iloc[0] - 1) * 100, 2)}}

    ma = level.rolling(200).mean()
    live = pd.Timestamp(live_from) if live_from else None
    return {
        "as_of": str(last.date()), "start": str(level.index[0].date()),
        "live_from": str(live.date()) if live is not None and live > level.index[0] else None,
        "stats": stats(level),
        "chart": {
            "dates": [str(d.date()) for d in level.index],
            "level": [_f(v, 2) for v in level.to_numpy()],
            "ma200": [_f(v, 2) for v in ma.to_numpy()],
            "bench": b["series"] if b else None,
        },
        "bench_ret": b["ret"] if b else None,
        "members": rows,
        "events": run["events"][::-1],
        "guessed_caps": run["guessed_caps"],
        "next_rebalance": None if run["hold"] else next_rebalance(last, rebalance),
        "hold": run["hold"],
    }


def detect_exits(closes: pd.DataFrame, gap: int, confirmed: set[str] | None = None) -> dict[str, pd.Timestamp]:
    """
    Members whose prices stopped at least `gap` sessions before the calendar
    ends — acquired, merged or delisted — with their last trading day.

    `confirmed`, when given, is the exchange's own list of delisted codes and
    the gap alone is not enough: an A-share can be halted for months and come
    back.
    """
    out = {}
    end = len(closes)
    for t in closes.columns:
        s = closes[t]
        lv = s.last_valid_index()
        if lv is None:
            continue
        pos = closes.index.get_loc(lv)
        if end - 1 - pos >= gap and (confirmed is None or t in confirmed):
            out[t] = lv
    return out
