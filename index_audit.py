"""
index_audit.py — 指数体检: is every stored day of a 自建指数 right?

A day that fails to store announces itself. A day stored WRONG does not:
on 2026-11-19 Yahoo's download lacked nine of AI Infrastructure's members,
they were stored as halted, and every later day carried the error quietly
(0.4–0.5% by 11-25) until a split broke something loud. So the stored record
is checked against itself and against fresh prices, day by day:

  missing   a session the market traded that has no stored day
  behind    finished sessions after the last stored day
  halted    a member stored as halted (held flat) on a day it has a price
  mismatch  the stored level against the level rebuilt from the day before:
            yesterday's cash + each holding's value × its fresh return. A
            rebalance or a membership change at the close moves value between
            holdings without changing the total, so this holds on every day.

A day whose holdings the fresh download cannot price at all (a member Yahoo
has since deleted after an acquisition) is counted as unverifiable rather
than flagged.

Pure: the stored days, fresh closes and the market's sessions are passed in.
"""

from __future__ import annotations

import math

import pandas as pd

#: Stored level against the rebuilt one; beyond this a day is flagged. The
#: repaired 11-19..25 days matched to the cent, so this is room for a data
#: source revising a price, not for an error.
TOL = 0.003


def _f(v, nd=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def audit(days: list[dict], closes: pd.DataFrame, sessions: pd.DatetimeIndex,
          *, since: str | None = None, tol: float = TOL) -> dict:
    """
    `days`: stored rows oldest first, each {d, close, state}; `closes`: fresh
    adjusted closes, one column per ticker; `sessions`: the market's finished
    sessions. Checks the days from `since` (the day before it is the base).
    """
    issues: list[dict] = []
    if not days:
        return {"issues": [], "checked": 0, "unverifiable": 0, "behind": 0, "last": None,
                "latest": str(sessions[-1].date()) if len(sessions) else None}
    stored = {d["d"]: d for d in days}
    order = [d["d"] for d in days]
    start = since or order[0]
    first_i = next((i for i, d in enumerate(order) if d >= start), len(order))

    # Sessions with no stored day, between the first checked day and the last stored.
    last = order[-1]
    for s in sessions:
        k = str(s.date())
        if start <= k <= last and k not in stored and k >= order[0]:
            issues.append({"d": k, "kind": "missing", "detail": "这个交易日没有存下来"})
    behind = [str(s.date()) for s in sessions if str(s.date()) > last]

    px = closes.sort_index()
    # Returns run from each stock's last traded price: a stock resuming after
    # a suspension (中微公司, 2025-12-24 → 2026-01-05, +14%) earns its whole
    # gap on the day it trades again, and that day is stored correctly.
    last_px = px.ffill()
    checked = unverifiable = 0
    for i in range(max(first_i, 1), len(order)):
        prev, cur = stored[order[i - 1]], stored[order[i]]
        sp, sc = prev.get("state") or {}, cur.get("state") or {}
        dp, dc = pd.Timestamp(order[i - 1]), pd.Timestamp(order[i])

        # Halted on a day it traded.
        traded = [t for t in (sc.get("h") or [])
                  if t in px.columns and dc in px.index and pd.notna(px.at[dc, t])]
        if traded:
            issues.append({"d": order[i], "kind": "halted", "tickers": traded,
                           "detail": f"{len(traded)} 只记成停牌（按原价不动），但当天有成交价"})

        # The level rebuilt from yesterday's holdings and fresh returns.
        held = sp.get("v") or {}
        if any(t not in px.columns for t in held) or dp not in px.index or dc not in px.index:
            unverifiable += 1
            continue
        est = float(sp.get("cash") or 0.0)
        for t, v in held.items():
            a, b = last_px.at[dp, t], last_px.at[dc, t]
            est += float(v) * (float(b) / float(a) if pd.notna(a) and pd.notna(b) and a else 1.0)
        checked += 1
        got = _f(cur.get("close"), 6)
        if got is None or est <= 0:
            continue
        diff = got / est - 1
        if abs(diff) > tol:
            issues.append({"d": order[i], "kind": "mismatch", "stored": _f(got, 4), "expected": _f(est, 4),
                           "diff": _f(diff, 5),
                           "detail": f"存的 {got:.2f}，用前一天的持仓和最新行情重算是 {est:.2f}（差 {diff * 100:+.2f}%）"})

    issues.sort(key=lambda x: (x["d"], x["kind"]))
    return {"issues": issues, "checked": checked, "unverifiable": unverifiable,
            "behind": len(behind), "behind_from": behind[0] if behind else None,
            "last": last, "latest": str(sessions[-1].date()) if len(sessions) else None,
            # Where a repair would start: the first wrong or missing day.
            "repair_from": issues[0]["d"] if issues else None}


__all__ = ["TOL", "audit"]
