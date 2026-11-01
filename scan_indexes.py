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
"""

from __future__ import annotations

import sys


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
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
