"""
尾盘推演 cache: valid for exactly as long as the bar it describes.

Offline — a stand-in database. The point of these tests is that the cache is
keyed on the BAR, not on a clock. A time-to-live would regenerate the read
halfway through a session in which nothing changed, and would still be serving
yesterday's read an hour after a new close. Neither is what "as long as no new
data has arrived" means.

    python -m pytest api/tests/test_whatif_cache.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import whatif_cache as wc  # noqa: E402

READ = {"mode": "actual", "bar_date": "2026-09-18",
        "read": {"headline": "缩量整理"}, "crossings": []}


class FakeDB:
    """One table, and a switch to make it not exist."""

    def __init__(self, rows=None, broken=False):
        self.rows = list(rows or [])
        self.broken = broken
        self.writes = 0
        self.deletes = 0

    def read_table(self, table, filters=None, limit=None, **kw):
        if self.broken:
            raise RuntimeError("PGRST205 relation whatif_cache does not exist")
        got = [r for r in self.rows
               if all(r.get(k) == v for k, v in (filters or {}).items())]
        return pd.DataFrame(got[:limit] if limit else got)

    def insert_records(self, table, records, upsert=False):
        if self.broken:
            raise RuntimeError("relation whatif_cache does not exist")
        self.writes += 1
        assert upsert, "one row per ticker — a plain insert would duplicate"
        for rec in records:
            self.rows = [r for r in self.rows if r["ticker"] != rec["ticker"]]
            self.rows.append(dict(rec))

    def delete_records(self, table, filters):
        self.deletes += 1
        self.rows = [r for r in self.rows
                     if not all(r.get(k) == v for k, v in filters.items())]


def stored(db, ticker="600519"):
    return next(r for r in db.rows if r["ticker"] == ticker)


# ── the bar is the key ───────────────────────────────────────────────────────
def test_the_same_bar_is_served_from_the_cache():
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, READ, db=db)

    hit = wc.load("600519", "2026-09-18", 20, db=db)
    assert hit is not None
    assert hit["read"]["headline"] == "缩量整理"
    assert hit["cached"] is True and hit["generated_at"]


def test_a_new_session_is_a_miss():
    """The whole point: a new close invalidates, nothing else does."""
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, READ, db=db)
    assert wc.load("600519", "2026-09-19", 20, db=db) is None


def test_the_cache_does_not_expire_on_its_own():
    """
    No TTL. A read generated at 09:30 is still the right read at 14:00 of the
    same session, because the bar it describes has not changed.
    """
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, READ, db=db)
    old = stored(db)
    old["generated_at"] = "2020-01-01T00:00:00+00:00"       # ancient
    assert wc.load("600519", "2026-09-18", 20, db=db) is not None


def test_a_different_accumulation_window_is_a_different_read():
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, READ, db=db)
    assert wc.load("600519", "2026-09-18", 60, db=db) is None


def test_another_stock_is_never_served():
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, READ, db=db)
    assert wc.load("000001", "2026-09-18", 20, db=db) is None


# ── one row per ticker ───────────────────────────────────────────────────────
def test_a_new_bar_replaces_the_old_row_rather_than_adding_one():
    """
    A read for a bar three weeks gone can never be served again. Accumulating
    them would grow the table by one row per stock per trading day, forever.
    """
    db = FakeDB()
    for day in ("2026-09-16", "2026-09-17", "2026-09-18"):
        wc.save("600519", day, 20, {**READ, "bar_date": day}, db=db)

    assert len(db.rows) == 1
    assert stored(db)["bar_date"] == "2026-09-18"


def test_clearing_forgets_only_that_ticker():
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, READ, db=db)
    wc.save("000001", "2026-09-18", 20, READ, db=db)

    wc.clear("600519", db=db)
    assert wc.load("600519", "2026-09-18", 20, db=db) is None
    assert wc.load("000001", "2026-09-18", 20, db=db) is not None


# ── what gets stored ─────────────────────────────────────────────────────────
def test_the_cached_flag_is_not_itself_cached():
    """
    Otherwise a freshly generated read, saved and then read back, would claim
    to have come from the cache — and every load would look like a hit.
    """
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, {**READ, "cached": False}, db=db)
    payload = json.loads(stored(db)["payload"])

    assert "cached" not in payload and "generated_at" not in payload
    assert wc.load("600519", "2026-09-18", 20, db=db)["cached"] is True


def test_the_payload_survives_a_round_trip_intact():
    db = FakeDB()
    rich = {**READ, "read": {"headline": "放量突破", "key_changes": ["MACD 金叉"],
                             "levels": {"confirm": "68.06"}}}
    wc.save("600519", "2026-09-18", 20, rich, db=db)
    hit = wc.load("600519", "2026-09-18", 20, db=db)

    assert hit["read"] == rich["read"]
    assert hit["crossings"] == []


# ── failing softly ───────────────────────────────────────────────────────────
def test_a_missing_table_means_no_cache_not_an_error(capsys):
    """
    Caching is an optimisation. An AI read that works and costs a call beats
    a failure about a table nobody created.
    """
    wc._warned = False
    db = FakeDB(broken=True)

    assert wc.load("600519", "2026-09-18", 20, db=db) is None
    assert wc.save("600519", "2026-09-18", 20, READ, db=db) is False
    assert "20260919_whatif_cache.sql" in capsys.readouterr().out


def test_the_missing_table_is_reported_once_not_once_per_page_load(capsys):
    wc._warned = False
    db = FakeDB(broken=True)
    for _ in range(5):
        wc.load("600519", "2026-09-18", 20, db=db)
    assert capsys.readouterr().out.count("Run supabase/migrations") == 1


def test_unreadable_stored_json_is_a_miss_not_a_crash():
    db = FakeDB()
    wc.save("600519", "2026-09-18", 20, READ, db=db)
    stored(db)["payload"] = "{not json"
    assert wc.load("600519", "2026-09-18", 20, db=db) is None


def test_a_row_with_a_missing_column_is_a_miss():
    db = FakeDB(rows=[{"ticker": "600519", "bar_date": "2026-09-18"}])
    assert wc.load("600519", "2026-09-18", 20, db=db) is None


def test_an_empty_table_is_simply_a_miss():
    assert wc.load("600519", "2026-09-18", 20, db=FakeDB()) is None
