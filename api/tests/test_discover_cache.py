"""
Keeping a two-minute search, and knowing exactly when to throw it away.

Offline. The search walks eighty tickers one at a time before any statistics
run, so re-running it because a tab was switched is two minutes spent
arriving at the same answer. But a cache that outlives its truth is worse
than no cache, so every input that can change the answer is in the key — and
nothing else is, which is why there is no TTL.

    python -m pytest api/tests/test_discover_cache.py -q
"""

from __future__ import annotations

import json
import os
import sys
import types

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import discover_cache as dc  # noqa: E402
import market_clock as mc  # noqa: E402

WATCH = ["600519", "000001", "600522"]
PARAMS = {"lookback_days": 504, "min_corr": 0.45, "within_sector": False,
          "target": None}
RESULT = {"kind": "pair-trade", "rows": [{"a": "600519", "b": "000001"}],
          "funnel": {"survivors": 1}}


class FakeDB:
    def __init__(self, broken=False):
        self.rows, self.broken, self.writes = [], broken, 0

    def read_table(self, table, filters=None, limit=None, **kw):
        if self.broken:
            raise RuntimeError("PGRST205 relation discover_cache does not exist")
        got = [r for r in self.rows
               if all(r.get(k) == v for k, v in (filters or {}).items())]
        return pd.DataFrame(got[:limit] if limit else got)

    def insert_records(self, table, records, upsert=False):
        if self.broken:
            raise RuntimeError("does not exist")
        self.writes += 1
        assert upsert, "one row per (user, kind) — a plain insert would duplicate"
        for rec in records:
            same = (rec["app_user_id"], rec["kind"])
            self.rows = [r for r in self.rows
                         if (r["app_user_id"], r["kind"]) != same]
            self.rows.append(dict(rec))

    def delete_records(self, table, filters):
        self.rows = [r for r in self.rows
                     if not all(r.get(k) == v for k, v in filters.items())]


def key(session="2026-09-18", kind="pair-trade", watch=None, **over):
    return dc.key_of(kind, {**PARAMS, **over}, watch or WATCH, session)


# ── what invalidates ─────────────────────────────────────────────────────────
def test_the_same_search_comes_back_without_re_running():
    db = FakeDB()
    dc.save(1, key(), RESULT, db=db)
    hit = dc.load(1, key(), db=db)

    assert hit is not None
    assert hit["rows"] == RESULT["rows"]
    assert hit["cached"] is True and hit["generated_at"]


@pytest.mark.parametrize("change", [
    {"session": "2026-09-19"},                     # a new close
    {"kind": "lead-lag"},                          # a different search
    {"lookback_days": 252},                        # a different history
    {"min_corr": 0.6},                             # a different shortlist
    {"within_sector": True},                       # a different candidate set
    {"watch": WATCH + ["300750"]},                 # a different universe
    {"target": "600519"},                          # a different stock's peers
])
def test_anything_that_changes_the_answer_is_a_miss(change):
    db = FakeDB()
    dc.save(1, key(), RESULT, db=db)
    assert dc.load(1, key(**change), db=db) is None


def test_nothing_else_expires_it():
    """
    No TTL. A search from a week ago is still the answer if no session has
    published and nothing else moved — and a clock could only ever throw away
    a correct result or serve a wrong one.
    """
    db = FakeDB()
    dc.save(1, key(), RESULT, db=db)
    db.rows[0]["generated_at"] = "2020-01-01T00:00:00+00:00"
    assert dc.load(1, key(), db=db) is not None


def test_another_user_is_never_served():
    db = FakeDB()
    dc.save(1, key(), RESULT, db=db)
    assert dc.load(2, key(), db=db) is None


def test_a_reordered_watchlist_is_the_same_watchlist():
    """
    The list arrives sorted by date added, so re-adding a stock you already
    held would otherwise read as a change. The SET decides the answer.
    """
    assert dc.fingerprint(["A", "B", "C"]) == dc.fingerprint(["C", "A", "B"])
    assert dc.fingerprint(["A", "B"]) != dc.fingerprint(["A", "B", "C"])
    assert dc.fingerprint(["A", "A", "B"]) == dc.fingerprint(["A", "B"])


def test_a_slider_that_barely_moved_is_the_same_search():
    """Float equality on a slider value would make every load a miss."""
    assert key(min_corr=0.45) == key(min_corr=0.4500001)
    assert key(min_corr=0.45) != key(min_corr=0.455)


# ── the freshness check failing is a miss, not a hit ─────────────────────────
def test_an_unknown_session_refuses_to_serve_anything():
    """
    Serving a stale search because the freshness check itself failed is the
    one outcome worse than repeating the search.
    """
    db = FakeDB()
    dc.save(1, key(), RESULT, db=db)
    assert dc.load(1, key(session=""), db=db) is None
    assert dc.load(1, key(session=None), db=db) is None


def test_a_result_with_no_freshness_basis_is_never_stored():
    """
    The hole the guard on load alone does not close: if the session was
    unknown when the result was SAVED, a later load that also cannot
    determine the session would match it exactly and serve a search with
    nothing behind it.
    """
    db = FakeDB()
    assert dc.save(1, key(session=""), RESULT, db=db) is False
    assert db.rows == []
    assert dc.load(1, key(session=""), db=db) is None


# ── storage ──────────────────────────────────────────────────────────────────
def test_a_new_search_replaces_the_old_row():
    db = FakeDB()
    for session in ("2026-09-16", "2026-09-17", "2026-09-18"):
        dc.save(1, key(session=session), RESULT, db=db)

    assert len(db.rows) == 1
    assert json.loads(db.rows[0]["cache_key"])["session"] == "2026-09-18"


def test_the_two_kinds_are_stored_separately():
    db = FakeDB()
    dc.save(1, key(kind="pair-trade"), RESULT, db=db)
    dc.save(1, key(kind="lead-lag"), {**RESULT, "kind": "lead-lag"}, db=db)

    assert len(db.rows) == 2
    assert dc.load(1, key(kind="pair-trade"), db=db)["kind"] == "pair-trade"
    assert dc.load(1, key(kind="lead-lag"), db=db)["kind"] == "lead-lag"


def test_the_cached_flag_is_not_itself_stored():
    db = FakeDB()
    dc.save(1, key(), {**RESULT, "cached": False, "generated_at": "x"}, db=db)
    payload = json.loads(db.rows[0]["payload"])
    assert "cached" not in payload and "generated_at" not in payload


def test_clearing_one_kind_leaves_the_other():
    db = FakeDB()
    dc.save(1, key(kind="pair-trade"), RESULT, db=db)
    dc.save(1, key(kind="lead-lag"), RESULT, db=db)

    dc.clear(1, "pair-trade", db=db)
    assert dc.load(1, key(kind="pair-trade"), db=db) is None
    assert dc.load(1, key(kind="lead-lag"), db=db) is not None


# ── failing softly ───────────────────────────────────────────────────────────
def test_a_missing_table_means_no_cache_not_an_error(capsys):
    dc._warned = False
    db = FakeDB(broken=True)

    assert dc.load(1, key(), db=db) is None
    assert dc.save(1, key(), RESULT, db=db) is False
    assert "20260920_discover_cache.sql" in capsys.readouterr().out


def test_unreadable_stored_json_is_a_miss():
    db = FakeDB()
    dc.save(1, key(), RESULT, db=db)
    db.rows[0]["payload"] = "[" * 3
    assert dc.load(1, key(), db=db) is None


# ── the freshness signal itself ──────────────────────────────────────────────
def frame(dates):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Close": range(len(idx))}, index=idx)


def fake_dm(monkeypatch, result, calls=None):
    mod = types.ModuleType("data_manager")

    def get_index_data_live(code, lookback_days=180, **kw):
        if calls is not None:
            calls.append(code)
        if isinstance(result, Exception):
            raise result
        return result

    mod.get_index_data_live = get_index_data_live
    monkeypatch.setitem(sys.modules, "data_manager", mod)


def test_the_latest_session_is_the_newest_published_bar(monkeypatch):
    """
    Not the calendar. Tushare will say the exchange was OPEN today hours
    before today's bar is published; a cache keyed on that invalidates itself
    into an empty fetch every afternoon.
    """
    mc.forget()
    fake_dm(monkeypatch, frame(["2026-09-16", "2026-09-17", "2026-09-18"]))
    assert mc.latest_session() == "2026-09-18"


def test_the_answer_is_memoised_rather_than_re_fetched(monkeypatch):
    """A cache-validity check must not cost more than what it is guarding."""
    mc.forget()
    calls = []
    fake_dm(monkeypatch, frame(["2026-09-18"]), calls)

    for _ in range(5):
        mc.latest_session()
    assert len(calls) == 1


def test_a_failed_check_says_unknown_rather_than_guessing(monkeypatch):
    mc.forget()
    fake_dm(monkeypatch, RuntimeError("tushare down"))
    assert mc.latest_session() is None

    mc.forget()
    fake_dm(monkeypatch, pd.DataFrame())
    assert mc.latest_session() is None


def test_an_unknown_answer_is_not_memoised(monkeypatch):
    """Otherwise one outage would freeze the cache for the whole TTL."""
    mc.forget()
    calls = []
    fake_dm(monkeypatch, RuntimeError("down"), calls)
    mc.latest_session()
    mc.latest_session()
    assert len(calls) == 2


def test_the_reference_is_an_index_not_a_stock(monkeypatch):
    """A suspended stock's last bar is not the market's last session."""
    mc.forget()
    calls = []
    fake_dm(monkeypatch, frame(["2026-09-18"]), calls)
    mc.latest_session()

    assert calls == [mc.REFERENCE_INDEX]
    assert mc.REFERENCE_INDEX.endswith(".SH")


# ── reading is separate from running ─────────────────────────────────────────
def test_the_read_route_and_the_run_route_are_different_methods():
    """
    A page load must never be able to start a two-minute job, and a finished
    search must never need a click to come back. One GET that only reads, one
    POST that only runs.
    """
    from api.main import app

    methods = {tuple(sorted(r.methods)) for r in app.routes
               if getattr(r, "path", "") == "/strategies/discover"}
    assert methods == {("GET",), ("POST",)}


def test_the_read_route_reads_and_never_runs(monkeypatch):
    from api import main as api_main

    db = FakeDB()
    stored = key(session="2026-09-18")
    dc.save(1, stored, RESULT, db=db)

    dm = types.ModuleType("data_manager")
    dm.db = db
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    monkeypatch.setattr(api_main, "_discover_key", lambda *a, **kw: (WATCH, stored))
    monkeypatch.setitem(sys.modules, "api.lead_lag_api", types.SimpleNamespace(
        discover=lambda *a, **kw: pytest.fail("the GET must not run the search")))

    got = api_main.strategies_discover_stored(user=types.SimpleNamespace(id=1))
    assert got is not None and got["cached"] is True
    assert got["rows"] == RESULT["rows"]


def test_the_read_route_returns_null_when_there_is_nothing_stored(monkeypatch):
    """Null, not an error — "nothing yet" is the normal first state."""
    from api import main as api_main

    dm = types.ModuleType("data_manager")
    dm.db = FakeDB()
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    monkeypatch.setattr(api_main, "_discover_key",
                        lambda *a, **kw: (WATCH, key(session="2026-09-18")))

    assert api_main.strategies_discover_stored(user=types.SimpleNamespace(id=1)) is None
