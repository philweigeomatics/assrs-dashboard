"""
discover_cache.py — keeping a two-minute search until something invalidates it.

The pair search walks eighty tickers one at a time before any statistics run.
Re-running it because a tab was switched is two minutes and eighty Tushare
calls spent arriving back at the same answer.

The result is a pure function of five things, and every one of them is part
of the validity check:

    the watchlist      add or remove a stock and the universe changed —
                       fingerprinted, because "80 stocks" is not the same
                       claim as "these 80 stocks"
    the kind           pair-trade and lead-lag are different searches
    lookback_days      a different window is a different history
    min_corr           a different shortlist
    within_sector      a different candidate set
    target             a different stock's peers is a different question
    the latest session a new close is new evidence

Nothing else can change the answer, so nothing else should expire it — which
is why there is no TTL here. See market_clock for how the last of those is
measured, and why "the exchange was open" is not the same question.

One row per user per kind. A search for a watchlist you no longer hold can
never be served again, so it is overwritten rather than accumulated.

A missing table means no caching, not an error: this is an optimisation, and
a search that works and costs two minutes beats a failure about a table.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

TABLE = "discover_cache"

_warned = False


def fingerprint(tickers) -> str:
    """
    A short, order-independent digest of the watchlist.

    Order-independent because the watchlist arrives sorted by date added, and
    re-adding a stock you already had would otherwise look like a change. The
    SET is what determines the answer.
    """
    codes = sorted({str(t).strip() for t in tickers if str(t).strip()})
    return hashlib.sha256("|".join(codes).encode()).hexdigest()[:16]


def _db():
    import data_manager
    return data_manager.db


def key_of(kind: str, params: dict, tickers, session: str) -> dict:
    """Everything that decides whether a stored result is still the answer."""
    return {
        "kind": str(kind),
        "lookback_days": int(params.get("lookback_days") or 0),
        # Rounded: a slider at 0.4500001 is the same search as 0.45, and
        # float equality would make every load a miss.
        "min_corr": round(float(params.get("min_corr") or 0), 3),
        "within_sector": bool(params.get("within_sector")),
        # A different target is a different search entirely. Leaving it out
        # would serve one stock's peers as another's.
        "target": str(params.get("target") or ""),
        "watchlist": fingerprint(tickers),
        "session": str(session),
    }


def load(app_user_id: int, key: dict, *, db=None) -> dict | None:
    """The stored result if every part of `key` still matches, else None."""
    global _warned
    db = db or _db()
    if not key.get("session"):
        # market_clock could not say what the latest session is. Treat that as
        # a miss: serving a stale search because the freshness check itself
        # failed is worse than repeating the search.
        return None
    try:
        rows = db.read_table(TABLE, filters={"app_user_id": int(app_user_id),
                                             "kind": key["kind"]}, limit=1)
    except Exception as exc:                                       # noqa: BLE001
        if not _warned:
            _warned = True
            print(f"[discover_cache] not caching — {type(exc).__name__}: {exc}. "
                  f"Run supabase/migrations/20260920_discover_cache.sql to enable.")
        return None
    if rows is None or rows.empty:
        return None

    row = rows.iloc[0].to_dict()
    try:
        stored = json.loads(row["cache_key"])
    except Exception:                                              # noqa: BLE001
        return None
    if stored != key:
        return None
    try:
        payload = json.loads(row["payload"])
    except Exception:                                              # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None

    payload["cached"] = True
    payload["generated_at"] = row.get("generated_at")
    return payload


def save(app_user_id: int, key: dict, payload: dict, *, db=None) -> bool:
    """Store this search, replacing whatever was there for this user and kind."""
    db = db or _db()
    record = {
        "app_user_id": int(app_user_id),
        "kind": key["kind"],
        "cache_key": json.dumps(key, sort_keys=True, ensure_ascii=False),
        # `cached` belongs to the response, not the store: keeping it would
        # make a freshly computed search claim to have come from here.
        "payload": json.dumps({k: v for k, v in payload.items()
                               if k not in ("cached", "generated_at")},
                              ensure_ascii=False),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not key.get("session"):
        # A row whose freshness basis is unknown can never be validly served —
        # and if it were stored, a LATER load that also could not determine the
        # session would match it and serve a result with nothing behind it.
        print("[discover_cache] not storing — the latest session is unknown")
        return False
    try:
        db.insert_records(TABLE, [record], upsert=True)
        return True
    except Exception as exc:                                       # noqa: BLE001
        print(f"[discover_cache] write failed for user {app_user_id}: "
              f"{type(exc).__name__}: {exc}")
        return False


def clear(app_user_id: int, kind: str | None = None, *, db=None) -> None:
    db = db or _db()
    where = {"app_user_id": int(app_user_id)}
    if kind:
        where["kind"] = kind
    try:
        db.delete_records(TABLE, where)
    except Exception:                                              # noqa: BLE001
        pass
