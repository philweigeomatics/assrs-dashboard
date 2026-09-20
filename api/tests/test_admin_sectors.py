"""
Sector membership: who may change it, and what it refuses to do.

A sector's stock list is the input to its PPI, and the PPI drives the
heatmap, the breadth grid, the rotation map and the regime score for every
user. So the gate is on the server, and the floor of two stocks is enforced
where the write happens rather than in the screen that calls it.

    python -m pytest api/tests/test_admin_sectors.py -q
"""

from __future__ import annotations

import os
import sys
import types

import pandas as pd
import pytest

_NOW = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from api import admin_api  # noqa: E402

BASIC = [{"ticker": "600519", "name": "贵州茅台"},
         {"ticker": "000001", "name": "平安银行"},
         {"ticker": "600036", "name": "招商银行"},
         {"ticker": "300750", "name": "宁德时代"},
         {"ticker": "002594", "name": "比亚迪"}]


class FakeDM:
    """Just enough data_manager to exercise the module."""

    def __init__(self, sectors=None, basic=BASIC, supabase=False, missing=()):
        self.sectors = sectors if sectors is not None else {
            "银行": ["000001", "600036"],
            "白酒": ["600519", "000001", "600036"],
        }
        self.basic = list(basic)
        self.supabase = supabase
        self.missing = set(missing)
        self.added, self.removed = [], []
        self.raw_rows = []
        self.jobs = []

    # ── reads ───────────────────────────────────────────────────────────
    def get_sector_stock_map(self):
        return {k: list(v) for k, v in self.sectors.items()}

    def get_all_stock_basic(self):
        return list(self.basic)

    def get_all_sector_stock_map_raw(self):
        return pd.DataFrame(self.raw_rows)

    def get_missing_ppi_tables(self, names):
        return [n for n in names if n in self.missing]

    def get_missing_breadth_columns(self, names):
        return [n for n in names if n in self.missing]

    # ── rebuild jobs ────────────────────────────────────────────────────
    def get_recent_rebuild_jobs(self, limit=15):
        return pd.DataFrame(self.jobs[:limit])

    def create_rebuild_job(self, job_type, sectors):
        import json as _json
        job_id = f"job{len(self.jobs) + 1}"
        self.jobs.insert(0, {
            "job_id": job_id, "job_type": job_type,
            "sectors": "__all__" if sectors == "__all__"
                       else _json.dumps(sectors, ensure_ascii=False),
            "status": "pending", "progress": 0,
            "progress_message": "Job created",
            "created_at": _NOW, "started_at": None,
            "completed_at": None, "error_message": None,
        })
        return job_id

    # ── writes ──────────────────────────────────────────────────────────
    def add_new_sector(self, sector, tickers):
        for t in tickers:
            self.add_stock_to_sector(sector, t)

    def add_stock_to_sector(self, sector, ticker):
        self.added.append((sector, ticker))
        self.sectors.setdefault(sector, []).append(ticker)

    def remove_stock_from_sector(self, sector, ticker):
        self.removed.append((sector, ticker))
        self.sectors[sector] = [t for t in self.sectors[sector] if t != ticker]


@pytest.fixture
def dm(monkeypatch):
    fake = FakeDM()
    monkeypatch.setattr(admin_api, "_dm", lambda: fake)
    monkeypatch.setitem(sys.modules, "db_config",
                        types.SimpleNamespace(USE_SUPABASE=False))
    return fake


# ── the gate ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("role", ["user", "", None, "Admin ", "administrator"])
def test_only_an_exact_admin_role_gets_through(role):
    """
    Not "contains admin", not "truthy". `administrator` is not an admin here,
    and a trailing space must not be a way in either.
    """
    from fastapi import HTTPException

    from api.main import _require_admin

    user = types.SimpleNamespace(role=role)
    if role in ("Admin ",):
        # Whitespace is not trimmed by the check, so this must be refused.
        with pytest.raises(HTTPException):
            _require_admin(user)
        return
    with pytest.raises(HTTPException) as e:
        _require_admin(user)
    assert e.value.status_code == 403


@pytest.mark.parametrize("role", ["admin", "ADMIN", "Admin"])
def test_the_admin_role_is_case_insensitive(role):
    from api.main import _require_admin
    _require_admin(types.SimpleNamespace(role=role))     # must not raise


def test_every_admin_route_checks_on_the_server():
    """
    The screen hides these controls; hiding a control does not stop a curl.
    Read from the source, so a new route added without the gate fails here.
    """
    import inspect

    from api import main as api_main

    handlers = [r.endpoint for r in api_main.app.routes
                if getattr(r, "path", "").startswith("/admin")]
    assert handlers, "no admin routes found"
    for fn in handlers:
        src = inspect.getsource(fn)
        assert "_require_admin(user)" in src, f"{fn.__name__} has no admin gate"


# ── overview ─────────────────────────────────────────────────────────────────
def test_the_overview_names_every_stock(dm):
    out = admin_api.overview()
    banks = next(s for s in out["sectors"] if s["name"] == "银行")
    assert [x["t"] for x in banks["stocks"]] == ["000001", "600036"]
    assert [x["n"] for x in banks["stocks"]] == ["平安银行", "招商银行"]


def test_removed_stocks_are_shown_not_hidden(dm):
    """
    "What did I take out of 半导体 last month" is the question asked right
    after a number moves, and a screen showing only the current state cannot
    answer it.
    """
    dm.raw_rows = [
        {"sector": "银行", "ticker": "600519", "is_active": 0,
         "added_at": "2026-01-02 09:00:00", "removed_at": "2026-03-04 11:22:33"},
        {"sector": "银行", "ticker": "000001", "is_active": 1,
         "added_at": "2026-01-02 09:00:00", "removed_at": None},
    ]
    banks = next(s for s in admin_api.overview()["sectors"] if s["name"] == "银行")
    assert banks["removed"] == [
        {"t": "600519", "n": "贵州茅台", "removed_at": "2026-03-04 11:22:33"}]


def test_the_overview_survives_an_empty_history_table(dm):
    dm.raw_rows = []
    assert all(s["removed"] == [] for s in admin_api.overview()["sectors"])


# ── adding ───────────────────────────────────────────────────────────────────
def test_adding_a_stock_writes_it(dm):
    out = admin_api.add_stock("银行", "600519")
    assert dm.added == [("银行", "600519")]
    assert out["name"] == "贵州茅台" and out["action"] == "added"


def test_a_stock_already_in_the_sector_is_refused(dm):
    with pytest.raises(ValueError, match="已经在"):
        admin_api.add_stock("银行", "000001")
    assert dm.added == []


def test_an_unknown_ticker_is_refused(dm):
    with pytest.raises(ValueError, match="不在股票列表"):
        admin_api.add_stock("银行", "999999")
    assert dm.added == []


def test_an_unreadable_stock_list_does_not_block_every_edit(monkeypatch):
    """
    stock_basic being unavailable is an outage, not a reason to refuse an
    admin a correct edit. The typo check is a convenience; it must not
    become a hard dependency.
    """
    fake = FakeDM(basic=[])
    monkeypatch.setattr(admin_api, "_dm", lambda: fake)
    admin_api.add_stock("银行", "300750")
    assert fake.added == [("银行", "300750")]


def test_adding_to_a_sector_that_does_not_exist_is_a_lookup_error(dm):
    with pytest.raises(LookupError, match="不存在"):
        admin_api.add_stock("不存在的板块", "600519")


# ── removing ─────────────────────────────────────────────────────────────────
def test_removing_a_stock_writes_it(dm):
    admin_api.remove_stock("白酒", "600519")
    assert dm.removed == [("白酒", "600519")]


def test_a_sector_cannot_be_cut_below_two_stocks(dm):
    """
    A one-stock PPI is that stock's price wearing a sector's name, and every
    screen downstream would go on calling it a sector.
    """
    with pytest.raises(ValueError, match="至少要保留"):
        admin_api.remove_stock("银行", "000001")
    assert dm.removed == []


def test_the_floor_is_the_module_constant_not_a_literal(dm):
    """Raising MIN_STOCKS must actually raise the floor."""
    dm.sectors["白酒"] = ["600519", "000001", "600036"]
    admin_api.MIN_STOCKS, keep = 3, admin_api.MIN_STOCKS
    try:
        with pytest.raises(ValueError):
            admin_api.remove_stock("白酒", "600519")
    finally:
        admin_api.MIN_STOCKS = keep


def test_removing_something_that_is_not_there_is_refused(dm):
    with pytest.raises(ValueError, match="不在"):
        admin_api.remove_stock("银行", "300750")


# ── creating ─────────────────────────────────────────────────────────────────
def test_creating_a_sector_adds_every_stock(dm):
    out = admin_api.create_sector("新能源", ["300750", "002594"])
    assert out["created"] is True
    assert dm.added == [("新能源", "300750"), ("新能源", "002594")]


def test_duplicates_in_the_list_are_collapsed(dm):
    out = admin_api.create_sector("新能源", ["300750", "300750", "002594"])
    assert out["stocks"] == ["300750", "002594"]
    assert len(dm.added) == 2


def test_a_sector_needs_two_stocks_to_be_created(dm):
    with pytest.raises(ValueError, match="至少需要"):
        admin_api.create_sector("新能源", ["300750", "300750"])
    assert dm.added == []


def test_an_existing_name_is_refused(dm):
    with pytest.raises(ValueError, match="已存在"):
        admin_api.create_sector("银行", ["300750", "002594"])


@pytest.mark.parametrize("name", ["", "   ", "a" * 41, "新 能源", 'a"b',
                                  "a;drop", "a/b", "a-b", "a.b", "a%b"])
def test_a_name_that_cannot_be_a_table_name_is_refused(name, dm):
    """
    The name becomes PPI_<name> as a table and a column in market_breadth.
    Anything needing different quoting in those two places is out.
    """
    with pytest.raises(ValueError):
        admin_api.create_sector(name, ["300750", "002594"])
    assert dm.added == []


def test_a_normal_chinese_or_english_name_is_accepted(dm):
    for name in ("储能", "Energy_Storage", "半导体设备", "AI2025"):
        admin_api.validate_name(name, {})


# ── the Supabase manual step ─────────────────────────────────────────────────
def test_on_supabase_a_missing_table_blocks_the_write(monkeypatch):
    """
    The client key cannot create a table. Writing the membership rows anyway
    leaves a sector that every rebuild fails on, so nothing is written and
    the SQL comes back instead.
    """
    fake = FakeDM(supabase=True, missing={"储能"})
    monkeypatch.setattr(admin_api, "_dm", lambda: fake)
    monkeypatch.setitem(sys.modules, "db_config",
                        types.SimpleNamespace(USE_SUPABASE=True))

    out = admin_api.create_sector("储能", ["300750", "002594"])
    assert out["created"] is False
    assert fake.added == [], "rows were written despite the missing table"
    assert any("PPI_储能" in q for q in out["sql"])
    assert any("market_breadth" in q for q in out["sql"])


def test_on_supabase_an_existing_table_creates_normally(monkeypatch):
    fake = FakeDM(supabase=True, missing=set())
    monkeypatch.setattr(admin_api, "_dm", lambda: fake)
    monkeypatch.setitem(sys.modules, "db_config",
                        types.SimpleNamespace(USE_SUPABASE=True))

    out = admin_api.create_sector("储能", ["300750", "002594"])
    assert out["created"] is True and out["sql"] == []
    assert len(fake.added) == 2


def test_sqlite_needs_no_manual_step(dm):
    assert admin_api.pending_sql("任何板块") == []


def test_a_failed_existence_check_blocks_rather_than_assumes(monkeypatch):
    """
    Not knowing whether the table exists must not read as "it does". The
    optimistic branch writes a sector whose rebuild can never succeed.
    """
    fake = FakeDM(supabase=True)

    def boom(_names):
        raise RuntimeError("supabase unreachable")

    fake.get_missing_ppi_tables = boom
    monkeypatch.setattr(admin_api, "_dm", lambda: fake)
    monkeypatch.setitem(sys.modules, "db_config",
                        types.SimpleNamespace(USE_SUPABASE=True))

    out = admin_api.create_sector("储能", ["300750", "002594"])
    assert out["created"] is False and out["sql"]
    assert fake.added == []


# ── rebuild jobs ─────────────────────────────────────────────────────────────
def _ago(minutes):
    import datetime as _dt
    return (_dt.datetime.now() - _dt.timedelta(minutes=minutes)
            ).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def started(monkeypatch, dm):
    """Records what would have been handed to the background thread."""
    calls = []
    runner = types.ModuleType("rebuild_runner")
    runner.start_rebuild_thread = lambda job_id, sectors: calls.append(
        (job_id, sectors))
    monkeypatch.setitem(sys.modules, "rebuild_runner", runner)
    return calls


def test_a_rebuild_creates_a_job_and_starts_the_worker(dm, started):
    out = admin_api.start_rebuild(["银行"])
    assert out["job_type"] == "sector_rebuild"
    assert started == [(out["job_id"], ["银行"])]


def test_rebuilding_everything_is_marked_as_such(dm, started):
    out = admin_api.start_rebuild("__all__")
    assert out["job_type"] == "full_rebuild"
    assert started[0][1] == "__all__"


def test_two_rebuilds_at_once_are_refused(dm, started):
    """
    Both wipe and rewrite the same PPI tables, and the loser silently
    corrupts the winner. The Streamlit page allowed this.
    """
    admin_api.start_rebuild(["银行"])
    dm.jobs[0]["status"] = "running"
    dm.jobs[0]["progress"] = 40

    with pytest.raises(ValueError, match="还在进行"):
        admin_api.start_rebuild(["白酒"])
    assert len(started) == 1, "a second worker was started"


def test_a_finished_job_does_not_block_the_next_one(dm, started):
    admin_api.start_rebuild(["银行"])
    dm.jobs[0]["status"] = "completed"
    admin_api.start_rebuild(["白酒"])
    assert len(started) == 2


def test_a_job_whose_thread_died_does_not_block_forever(dm, started):
    """
    The thread dies with its instance and the row says 'running' for good.
    Without this, one killed rebuild locks out every future one.
    """
    admin_api.start_rebuild(["银行"])
    dm.jobs[0].update(status="running",
                      created_at=_ago(admin_api.STALE_MINUTES + 30))

    admin_api.start_rebuild(["白酒"])
    assert len(started) == 2


def test_a_stalled_job_is_reported_as_stalled(dm, started):
    admin_api.start_rebuild(["银行"])
    dm.jobs[0].update(status="running", progress=42,
                      created_at=_ago(admin_api.STALE_MINUTES + 5))

    job = admin_api.jobs()["jobs"][0]
    assert job["status"] == "stalled"
    assert job["progress"] == 42, "progress is kept, only the status changes"
    assert job["age_minutes"] > admin_api.STALE_MINUTES


def test_a_young_running_job_is_still_running(dm, started):
    admin_api.start_rebuild(["银行"])
    dm.jobs[0].update(status="running", created_at=_ago(5))
    assert admin_api.jobs()["jobs"][0]["status"] == "running"
    assert admin_api.jobs()["running"] is True


def test_a_finished_job_is_never_called_stalled(dm, started):
    """Age alone must not condemn a job that already completed."""
    admin_api.start_rebuild(["银行"])
    dm.jobs[0].update(status="completed", progress=100,
                      created_at=_ago(60 * 24 * 30))
    assert admin_api.jobs()["jobs"][0]["status"] == "completed"
    assert admin_api.jobs()["running"] is False


def test_the_sector_list_is_readable_not_raw_json(dm, started):
    admin_api.start_rebuild(["银行", "白酒"])
    job = admin_api.jobs()["jobs"][0]
    assert job["sectors"] == ["银行", "白酒"] and job["scope"] == "some"

    dm.jobs.clear()
    admin_api.start_rebuild("__all__")
    assert admin_api.jobs()["jobs"][0]["scope"] == "all"


def test_unknown_sectors_are_dropped_and_an_empty_list_is_refused(dm, started):
    out = admin_api.start_rebuild(["银行", "不存在"])
    assert started[-1][1] == ["银行"]

    dm.jobs[0]["status"] = "completed"
    with pytest.raises(ValueError, match="没有可重建"):
        admin_api.start_rebuild(["不存在"])
    assert out["job_id"]


def test_an_unparseable_sectors_field_does_not_break_the_list(dm, started):
    admin_api.start_rebuild(["银行"])
    dm.jobs[0]["sectors"] = "not json at all"
    assert admin_api.jobs()["jobs"][0]["sectors"] == ["not json at all"]


# ── creating a sector rebuilds it ────────────────────────────────────────────
def test_creating_a_sector_starts_its_rebuild(dm, started):
    """A new sector has no PPI until one is built, so it is part of creating."""
    out = admin_api.create_sector("储能", ["300750", "002594"])
    assert out["created"] is True and out["job_id"]
    assert started == [(out["job_id"], ["储能"])]


def test_a_sector_is_still_created_when_a_rebuild_cannot_start(dm, started):
    """
    The rows are written either way. Reporting a failure would imply the
    sector does not exist, and the admin would create it twice.
    """
    admin_api.start_rebuild(["银行"])
    dm.jobs[0]["status"] = "running"

    out = admin_api.create_sector("储能", ["300750", "002594"])
    assert out["created"] is True
    assert out["job_id"] is None and "还在进行" in out["job_error"]
    assert "储能" in dm.get_sector_stock_map()


def test_a_blocked_sector_creation_writes_nothing_and_starts_nothing(monkeypatch):
    """The Supabase path: no rows, no job, just the SQL."""
    fake = FakeDM(supabase=True, missing={"储能"})
    monkeypatch.setattr(admin_api, "_dm", lambda: fake)
    monkeypatch.setitem(sys.modules, "db_config",
                        types.SimpleNamespace(USE_SUPABASE=True))
    calls = []
    runner = types.ModuleType("rebuild_runner")
    runner.start_rebuild_thread = lambda *a: calls.append(a)
    monkeypatch.setitem(sys.modules, "rebuild_runner", runner)

    out = admin_api.create_sector("储能", ["300750", "002594"])
    assert out["created"] is False and out["job_id"] is None
    assert fake.added == [] and calls == []
