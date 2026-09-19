"""
whatif_cache.py — remembering a 尾盘推演 until the bar it describes is stale.

The AI read of the last real session is a pure function of that session: the
same stock, the same closing bar, the same 吸筹/出货 window produce the same
brief and therefore the same answer. Re-running it costs a DeepSeek call and
twenty to sixty seconds every time the page is opened, to arrive back where it
started. So it is cached on the bar, not on a clock.

Cached on the BAR DATE specifically, and that is the whole design:

  * A time-to-live is the wrong instrument. Six hours would regenerate the
    read halfway through a session in which nothing changed, and would still
    be serving yesterday's read an hour after a new close. The read is valid
    for exactly as long as it is the latest bar — no longer, and no less.
  * One row per ticker. Only the newest bar's read is ever wanted; a row for
    a bar three weeks gone can never be served again, so it is overwritten
    rather than accumulated.

The simulated ("ghost") read is NOT cached. Its inputs are a continuous
slider, so the odds of the same hypothetical being asked for twice are close
to zero, and caching it would fill the table with reads nobody returns to.

A missing table means no caching, not an error. This is an optimisation; an
AI read that works and costs a call is strictly better than a failure about a
table that was never created. The one place that is NOT true is a write, where
silence would make every load look like a cache miss forever — so a failed
write says so once, in the log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

TABLE = "whatif_cache"

#: Set after the first failed read so a missing table logs once rather than
#: once per page load.
_warned = False


def _db():
    import data_manager
    return data_manager.db


def load(ticker: str, bar_date: str, window: int, *, db=None) -> dict | None:
    """
    The saved read for this exact bar, or None.

    None covers every "no" there is — no row, a row for an older bar, a row
    for a different 吸筹 window, a table that does not exist, unreadable JSON.
    The caller's next step is identical in all of them, so they do not need
    telling apart.
    """
    global _warned
    db = db or _db()
    try:
        df = db.read_table(TABLE, filters={"ticker": ticker}, limit=1)
    except Exception as exc:                                       # noqa: BLE001
        if not _warned:
            _warned = True
            print(f"[whatif_cache] not caching — {type(exc).__name__}: {exc}. "
                  f"Run supabase/migrations/20260919_whatif_cache.sql to enable.")
        return None
    if df is None or df.empty:
        return None

    row = df.iloc[0].to_dict()
    if str(row.get("bar_date") or "") != str(bar_date):
        return None
    try:
        if int(row.get("ad_window") or 0) != int(window):
            return None
        payload = json.loads(row["payload"])
    except Exception:                                              # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None

    payload["cached"] = True
    payload["generated_at"] = row.get("generated_at")
    return payload


def save(ticker: str, bar_date: str, window: int, payload: dict, *, db=None) -> bool:
    """Store this bar's read, replacing whatever was there for this ticker."""
    db = db or _db()
    record = {
        "ticker": ticker,
        "bar_date": str(bar_date),
        "ad_window": int(window),
        # `cached` belongs to the response, not the cache: storing it would
        # make a freshly generated read claim to have been served from here.
        "payload": json.dumps({k: v for k, v in payload.items()
                               if k not in ("cached", "generated_at")},
                              ensure_ascii=False),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        db.insert_records(TABLE, [record], upsert=True)
        return True
    except Exception as exc:                                       # noqa: BLE001
        print(f"[whatif_cache] write failed for {ticker}: {type(exc).__name__}: {exc}")
        return False


def clear(ticker: str, *, db=None) -> None:
    """Forget this ticker's read — used by the regenerate button."""
    db = db or _db()
    try:
        db.delete_records(TABLE, {"ticker": ticker})
    except Exception:                                              # noqa: BLE001
        pass
