"""
The index strip, and the thing that makes it easy to get quietly wrong.

These markets are not on the same clock. At ten in the morning in Shanghai
the S&P's latest number is yesterday's close and today's Shanghai bar does
not exist. Side by side under one date, a reader compares a session that has
happened with one that has not — so the tests here are mostly about dates,
not about numbers.

    python -m pytest api/tests/test_world_indices.py -q
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import world_indices as wi  # noqa: E402


def closes(values, end="2026-09-22"):
    idx = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(values, index=idx, dtype="float64")


# ── one row ──────────────────────────────────────────────────────────────────
def test_the_move_is_against_the_previous_bar():
    row = wi._row("^X", "X", closes([100.0, 110.0]))
    assert row["close"] == 110.0
    assert row["change_pct"] == 10.0


def test_a_fall_is_negative():
    assert wi._row("^X", "X", closes([100.0, 95.0]))["change_pct"] == -5.0


def test_the_date_is_the_bar_s_own_date_not_today():
    """
    The whole point. A row carrying 'today' would make Friday's S&P close
    look like it happened this morning.
    """
    row = wi._row("^X", "X", closes([1.0, 2.0], end="2026-09-18"))
    assert row["date"] == "2026-09-18"


def test_gaps_in_the_series_are_skipped_not_counted_as_a_move():
    row = wi._row("^X", "X", closes([100.0, np.nan, 102.0]))
    assert row["change_pct"] == 2.0


def test_one_bar_is_not_enough_for_a_change():
    assert wi._row("^X", "X", closes([100.0])) is None
    assert wi._row("^X", "X", closes([])) is None


def test_a_zero_previous_close_does_not_divide():
    assert wi._row("^X", "X", closes([0.0, 100.0])) is None


def test_the_sparkline_is_the_tail_and_no_longer():
    row = wi._row("^X", "X", closes([float(i) for i in range(40)]))
    assert len(row["spark"]) == wi.SPARK
    assert row["spark"][-1] == row["close"]


def test_a_short_series_gives_a_short_sparkline_rather_than_padding():
    row = wi._row("^X", "X", closes([1.0, 2.0, 3.0]))
    assert row["spark"] == [1.0, 2.0, 3.0]


# ── how far behind ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("older, newer, expect", [
    ("2026-09-22", "2026-09-22", 0),
    ("2026-09-21", "2026-09-22", 1),
    ("2026-09-18", "2026-09-22", 2),      # Fri → Tue, weekend not counted
    ("2026-09-23", "2026-09-22", 0),      # ahead is not behind
])
def test_behind_is_counted_in_sessions_not_calendar_days(older, newer, expect):
    """
    Friday to Monday is one session behind, not three. Counting calendar
    days would paint every Monday morning as an outage.
    """
    assert wi._sessions_between(older, newer) == expect


def test_an_unparseable_date_does_not_raise():
    assert wi._sessions_between("not a date", "2026-09-22") == 0


# ── the snapshot ─────────────────────────────────────────────────────────────
@pytest.fixture
def fake(monkeypatch):
    """Both sources stubbed, so the test is about dates and shaping."""
    def tushare(members):
        return [{"code": c, "name": n, "close": 100.0, "change_pct": 1.0,
                 "date": "2026-09-22", "spark": [99.0, 100.0]}
                for c, n in members]

    def yahoo(members):
        return [{"code": c, "name": n, "close": 200.0, "change_pct": -0.5,
                 "date": "2026-09-21", "spark": [201.0, 200.0]}
                for c, n in members]

    monkeypatch.setattr(wi, "_tushare", tushare)
    monkeypatch.setattr(wi, "_yahoo", yahoo)


def test_every_group_comes_back_with_its_members(fake):
    out = wi.snapshot()
    assert [g["name"] for g in out["groups"]] == [g["name"] for g in wi.GROUPS]
    for g, spec in zip(out["groups"], wi.GROUPS):
        assert len(g["indices"]) == len(spec["members"])
        assert g["missing"] == 0


def test_as_of_is_the_newest_bar_in_the_whole_set(fake):
    out = wi.snapshot()
    assert out["as_of"] == "2026-09-22"


def test_an_older_market_is_marked_behind(fake):
    out = wi.snapshot()
    asia = next(g for g in out["groups"] if g["name"] == "亚太")
    china = next(g for g in out["groups"] if g["name"] == "A 股")
    assert all(r["behind_days"] == 1 for r in asia["indices"])
    assert all(r["behind_days"] == 0 for r in china["indices"])


def test_one_source_failing_leaves_the_other_s_rows(monkeypatch, fake):
    """
    A strip that vanishes because Frankfurt was unreachable is worse than
    one that shows Shanghai and says the rest is missing.
    """
    def boom(_members):
        raise RuntimeError("yahoo down")

    monkeypatch.setattr(wi, "_yahoo", boom)
    out = wi.snapshot()

    china = next(g for g in out["groups"] if g["name"] == "A 股")
    west = next(g for g in out["groups"] if g["name"] == "欧美")
    assert len(china["indices"]) == 6
    assert west["indices"] == [] and west["missing"] == len(wi.GROUPS[2]["members"])
    assert out["as_of"] == "2026-09-22"


def test_everything_failing_is_an_empty_strip_not_a_crash(monkeypatch):
    monkeypatch.setattr(wi, "_tushare", lambda m: [])
    monkeypatch.setattr(wi, "_yahoo", lambda m: [])
    out = wi.snapshot()
    assert out["as_of"] is None and out["total"] == 0
    assert all(g["indices"] == [] for g in out["groups"])


def test_a_partial_group_reports_how_many_are_missing(monkeypatch, fake):
    monkeypatch.setattr(wi, "_yahoo", lambda members: [
        {"code": members[0][0], "name": members[0][1], "close": 1.0,
         "change_pct": 0.0, "date": "2026-09-22", "spark": [1.0, 1.0]}])
    out = wi.snapshot()
    west = next(g for g in out["groups"] if g["name"] == "欧美")
    assert len(west["indices"]) == 1
    assert west["missing"] == len(wi.GROUPS[2]["members"]) - 1


# ── still trading, or settled ────────────────────────────────────────────────
def test_a_bar_dated_today_in_its_own_timezone_is_flagged(monkeypatch):
    """
    A row dated today is not necessarily a close — the session may still be
    running, and "+0.4%" then means "so far", which is a different claim.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    today_sh = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    monkeypatch.setattr(wi, "_tushare", lambda members: [
        {"code": c, "name": n, "close": 1.0, "change_pct": 0.5,
         "date": today_sh, "spark": [1.0, 1.0]} for c, n in members])
    monkeypatch.setattr(wi, "_yahoo", lambda members: [
        {"code": c, "name": n, "close": 1.0, "change_pct": 0.5,
         "date": "2020-01-02", "spark": [1.0, 1.0]} for c, n in members])

    out = wi.snapshot()
    china = next(g for g in out["groups"] if g["name"] == "A 股")
    west = next(g for g in out["groups"] if g["name"] == "欧美")
    assert all(r["today"] for r in china["indices"])
    assert all(not r["today"] for r in west["indices"])


def test_each_group_is_judged_against_its_own_timezone(fake):
    """Not against the server's. The server could be anywhere."""
    out = wi.snapshot()
    zones = {g["name"]: g["tz"] for g in out["groups"]}
    assert zones["A 股"] == "Asia/Shanghai"
    assert zones["欧美"] == "America/New_York"


# ── the payload leaves as JSON ───────────────────────────────────────────────
def test_the_payload_has_no_nan_in_it(fake):
    import json
    json.dumps(wi.snapshot(), allow_nan=False)


def test_yahoo_shape_handles_a_single_symbol(monkeypatch):
    """
    yfinance drops the ticker level when only one symbol is requested, and
    the indexing that works for many then raises for one.
    """
    frame = pd.DataFrame({"Close": [10.0, 11.0]},
                         index=pd.bdate_range(end="2026-09-22", periods=2))
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *a, **kw: frame
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    rows = wi._yahoo([("^GSPC", "标普500")])
    assert len(rows) == 1 and rows[0]["change_pct"] == 10.0
