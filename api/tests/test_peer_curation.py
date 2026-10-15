"""
同业对比 — saving a peer list, and showing it.

Two bugs, found together:

1. An edited list kept coming back unchanged. The brief was cached in
   process memory and the save cleared it — on whichever Cloud Run instance
   took the save. Every other instance kept serving the old list for twenty
   minutes. The peer list is now read fresh on every request.

2. The editor was a textarea of "code name" lines, and the server trusted
   both halves. An audit of 382 saved peers found 30 with the wrong name:
   28 stale (闻泰科技, now *ST闻泰 — a risk warning hidden by the label), and
   2 a different company outright (300369 labelled 卓翼科技, while the row
   showed 绿盟科技's PE and margins). Codes are now the only input and names
   are looked up.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def fake_dm(monkeypatch):
    """
    Stand-ins for data_manager and ai_client, installed before equity_brief
    is imported. The real data_manager reads Supabase secrets at import time,
    so a test that imports it for real only passes on a machine that has them.
    """
    dm = types.ModuleType("data_manager")
    dm.db = types.SimpleNamespace(
        read_table=lambda *a, **k: None,
        delete_records=lambda *a, **k: None,
        insert_records=lambda *a, **k: None)
    dm.get_all_stock_basic = lambda: []
    dm.get_stock_name_from_db = lambda t: None
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    monkeypatch.setitem(sys.modules, "ai_client", types.ModuleType("ai_client"))
    # A previously imported equity_brief holds the REAL data_manager.
    monkeypatch.delitem(sys.modules, "equity_brief", raising=False)
    return dm


@pytest.fixture
def ea(monkeypatch, fake_dm):
    """equity_api with a known name map and no network behind the metrics."""
    from api import equity_api

    monkeypatch.setattr(equity_api, "official_names", lambda: {
        "002281": "光迅科技",
        "300308": "中际旭创",
        "600487": "亨通光电",
        "300369": "绿盟科技",
        "002369": "卓翼科技",
        "600745": "*ST闻泰",
    })
    monkeypatch.setattr(equity_api, "_peer_metrics", lambda t: {
        "ticker": t, "pe_ttm": None, "pb": None, "mv_yi": None, "roe": None,
        "gross_margin": None, "net_margin": None, "revenue_yoy": None,
        "profit_yoy": None, "debt_to_assets": None, "period": None})
    return equity_api


# ── the name comes from the code ─────────────────────────────────────────────
def test_the_official_name_wins_over_the_stored_one(ea):
    """The row's numbers belong to the code, so must the label."""
    rows = ea.peer_table("002281", "光迅科技",
                         [{"ticker": "300369", "name": "卓翼科技"}])["rows"]
    peer = next(r for r in rows if not r["is_target"])
    assert peer["name"] == "绿盟科技"


def test_a_disagreement_is_flagged_not_silently_corrected(ea):
    """
    Correcting quietly would hide that an AI-generated list put the wrong
    company in — which is the thing worth knowing.
    """
    rows = ea.peer_table("002281", "光迅科技",
                         [{"ticker": "300369", "name": "卓翼科技"}])["rows"]
    peer = next(r for r in rows if not r["is_target"])
    assert peer["name_mismatch"] is True
    assert peer["claimed_name"] == "卓翼科技"


def test_a_stale_name_surfaces_a_risk_warning(ea):
    """闻泰科技 → *ST闻泰: the stored label was hiding the ST."""
    rows = ea.peer_table("002281", "光迅科技",
                         [{"ticker": "600745", "name": "闻泰科技"}])["rows"]
    peer = next(r for r in rows if not r["is_target"])
    assert peer["name"] == "*ST闻泰"
    assert peer["name_mismatch"] is True


def test_a_matching_name_is_not_flagged(ea):
    rows = ea.peer_table("002281", "光迅科技",
                         [{"ticker": "300308", "name": "中际旭创"}])["rows"]
    assert not next(r for r in rows if not r["is_target"])["name_mismatch"]


def test_a_code_with_no_official_name_is_marked(ea):
    rows = ea.peer_table("002281", "光迅科技",
                         [{"ticker": "888888", "name": "不存在"}])["rows"]
    peer = next(r for r in rows if not r["is_target"])
    assert peer["unknown_code"] is True
    assert peer["name"] == "不存在"          # nothing better to show


# ── validating a save ────────────────────────────────────────────────────────
def test_real_codes_come_back_with_their_real_names(ea):
    kept, rejected = ea.validate_peers("002281", ["300308", "600487"])
    assert kept == [{"ticker": "300308", "name": "中际旭创"},
                    {"ticker": "600487", "name": "亨通光电"}]
    assert rejected == []


def test_an_unknown_code_is_rejected_not_dropped(ea):
    kept, rejected = ea.validate_peers("002281", ["300308", "999999"])
    assert [k["ticker"] for k in kept] == ["300308"]
    assert rejected == ["999999"]


def test_a_near_miss_is_not_padded_into_a_plausible_code(ea):
    """`zfill` used to turn "60051" into "060051", which looks like a code."""
    kept, rejected = ea.validate_peers("002281", ["60051"])
    assert kept == [] and rejected == ["60051"]


def test_the_target_cannot_be_its_own_peer(ea):
    kept, rejected = ea.validate_peers("002281", ["002281", "300308"])
    assert [k["ticker"] for k in kept] == ["300308"]
    assert rejected == []


def test_duplicates_collapse(ea):
    kept, _ = ea.validate_peers("002281", ["300308", "300308"])
    assert len(kept) == 1


def test_one_digit_off_is_a_different_company_not_a_typo_fix(ea):
    """
    300369 and 002369 are both real, and different. Validation can only
    confirm a code exists — which is why the editor picks from a list.
    """
    kept, _ = ea.validate_peers("002281", ["300369"])
    assert kept[0]["name"] == "绿盟科技"


# ── the write has to say whether it happened ─────────────────────────────────
def _brief(monkeypatch, fail: bool):
    import equity_brief as eb  # the fake_dm fixture has already installed stubs

    calls = []

    def insert(table, rows):
        if fail:
            raise RuntimeError("RLS")
        calls.append(rows)

    fake_db = types.SimpleNamespace(
        delete_records=lambda *a, **k: None, insert_records=insert)
    monkeypatch.setattr(eb.data_manager, "db", fake_db)
    return eb, calls


def test_a_failed_write_reports_failure(monkeypatch, fake_dm):
    """It used to print a warning, return normally, and the route said saved."""
    eb, _ = _brief(monkeypatch, fail=True)
    assert eb.save_competitors_curated("002281", [{"ticker": "300308"}]) is False


def test_a_successful_write_reports_success(monkeypatch, fake_dm):
    eb, calls = _brief(monkeypatch, fail=False)
    assert eb.save_competitors_curated("002281", [{"ticker": "300308"}]) is True
    assert len(calls) == 1


# ── fresh reads ──────────────────────────────────────────────────────────────
def test_the_ai_half_is_not_part_of_the_cached_half(ea):
    """
    The cache-coherence fix in one assertion: the slow half that the route
    caches must not contain the peers, or an edit on one Cloud Run instance
    stays invisible on the others.
    """
    import inspect

    src = inspect.getsource(ea.data_half)
    assert "peer" not in src and "cached_ai" not in src
    assert "peer_table" in inspect.getsource(ea.ai_half)


def test_cached_ai_reads_every_section_in_one_query(monkeypatch, ea):
    import json

    import pandas as pd

    queries = []
    frame = pd.DataFrame([
        {"ticker": "002281", "section": "competitors",
         "payload": json.dumps({"competitors": [{"ticker": "300308"}]}),
         "generated_at": "2026-10-15"},
        {"ticker": "002281", "section": "overview",
         "payload": json.dumps({"text": "x"}), "generated_at": "2026-10-15"},
    ])

    def read_table(table, filters=None, **kw):
        queries.append(filters)
        return frame

    import equity_brief as eb
    monkeypatch.setattr(sys.modules["data_manager"], "db",
                        types.SimpleNamespace(read_table=read_table))
    monkeypatch.setattr(eb, "ensure_equity_brief_cache_table", lambda: None)

    got = ea.cached_ai("002281")
    assert len(queries) == 1
    assert got["competitors"]["payload"]["competitors"][0]["ticker"] == "300308"
    assert got["porters"] is None
