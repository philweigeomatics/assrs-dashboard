"""
scan_indexes.py — compute every user's 自建指数 once a day (GitHub Actions).

Nothing about an index's level needs a schedule: it is replayed from its
definition whenever it is opened, rebalances included. What does need one is
the exit record. Yahoo deletes an acquired company's history some weeks after
the deal closes, and the exit is only written down when the index is
computed — so an index nobody opens for a month could lose a member's past.
Running every index daily closes that gap.

One bulk price download per market for all indexes together. Exits non-zero
only when an index fails to compute; tables not yet created is a skip.
"""

from __future__ import annotations

import sys


def main() -> int:
    from api import index_api

    try:
        report = index_api.refresh_all()
    except Exception as exc:                                       # noqa: BLE001
        if "custom_index" in str(exc):
            print(f"[indexes] tables not created yet — skipping ({type(exc).__name__})")
            return 0
        raise
    if not report:
        print("[indexes] no indexes yet")
        return 0
    bad = 0
    for r in report:
        if r["ok"]:
            note = f" · 新退市: {', '.join(r['new_exits'])}" if r["new_exits"] else ""
            print(f"[indexes] ✓ {r['name']} ({r['id']}) {r['level']} @ {r['as_of']}{note}")
        else:
            bad += 1
            print(f"[indexes] ✗ {r['name']} ({r['id']}): {r['error']}")
    print(f"[indexes] {len(report) - bad}/{len(report)} computed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
