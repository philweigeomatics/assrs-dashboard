"""
scan_indexes.py — carry every user's 自建指数 forward one session (GitHub Actions).

Each index is stored day by day (custom_index_day). This continues each from
its last stored close with a few months of fresh prices — one download per
market for all indexes together — and stores the sessions since. An index
with no record yet is computed from its start date. Nothing already stored
is recomputed.

It is also what writes an acquired member's exit down: Yahoo deletes an
acquired company's history some weeks after the deal closes, so an index
nobody opens still has to be run.

Runs at 20:00 Beijing: after the A-share close and Tushare's publishing,
before the US open, so both markets' latest sessions are final. Exits non-zero
when any index fails; tables not yet created is a skip.

Then the last AUDIT_SESSIONS stored days of every index are checked
(index_audit): a day stored WRONG does not fail anything by itself — 2026-11-19
sat in two indexes for a week — so a wrong, missing or falsely halted day also
exits non-zero, and GitHub's failed-run email is the alert. The admin tab's
指数体检 shows the same, over a year, with a button to repair.
"""

from __future__ import annotations

import sys

#: Trading days back the daily job re-checks; the admin tab looks further.
AUDIT_SESSIONS = 20


def main() -> int:
    from api import index_api

    try:
        report = index_api.refresh_all()
    except Exception as exc:                                       # noqa: BLE001
        if "custom_index" in str(exc):
            print(f"[indexes] tables not created yet — skipping ({type(exc).__name__}: {exc})"[:300])
            return 0
        raise
    if not report:
        print("[indexes] no indexes yet")
        return 0
    bad = 0
    for r in report:
        if not r["ok"]:
            bad += 1
            print(f"[indexes] ✗ {r['name']} ({r['id']}): {r['error']}")
        elif r.get("rebuilt"):
            print(f"[indexes] ✓ {r['name']} ({r['id']}) computed from its start: {r['added']} days stored")
        else:
            note = f" · 新退市: {', '.join(r['new_exits'])}" if r.get("new_exits") else ""
            print(f"[indexes] ✓ {r['name']} ({r['id']}) +{r['added']} day(s) · {r['level']} @ {r['as_of']}{note}")
    print(f"[indexes] {len(report) - bad}/{len(report)} up to date")
    return 1 if (audit(index_api) or bad) else 0


def audit(index_api) -> int:
    """Re-check the recent stored days; the number of indexes with a problem."""
    try:
        checks = index_api.health(AUDIT_SESSIONS)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[audit] could not run: {type(exc).__name__}: {exc}"[:200])
        return 0
    wrong = 0
    for h in checks:
        if h.get("error"):
            print(f"[audit] ? {h['name']} ({h['id']}): {h['error']}")
            continue
        if not h["issues"]:
            continue
        wrong += 1
        days = sorted({i["d"] for i in h["issues"]})
        print(f"[audit] ✗ {h['name']} ({h['id']}): {len(days)} 天有问题 —— {', '.join(days)}"
              f" · 管理 → 指数体检 可从 {h['repair_from']} 起重算")
        for i in h["issues"]:
            print(f"[audit]     {i['d']} {i['kind']}: {i['detail']}")
    print(f"[audit] {len(checks) - wrong}/{len(checks)} 最近 {AUDIT_SESSIONS} 个交易日核对无误")
    return wrong


if __name__ == "__main__":
    sys.exit(main())
