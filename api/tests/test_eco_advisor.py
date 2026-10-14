"""
Reading a BATCH of economic releases.

The arithmetic is done here and handed to the model as fact — a model asked
to subtract two strings with percent signs in them gets some of them wrong
with complete confidence, and those differences are the whole substance.

The other thing pinned here is the refusal: asked for a theme in twelve
unrelated releases a model will invent one, so `coherent` has to survive.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import eco_advisor as ea  # noqa: E402


def _ev(event="X", country="美国", value=None, forecast=None, prev=None,
        date="2026-10-13", time="20:30"):
    return {"date": date, "time": time, "country": country, "currency": "USD",
            "event": event, "value": value, "prev": prev, "forecast": forecast}


# ── parsing the strings Tushare sends ────────────────────────────────────────
@pytest.mark.parametrize("text, want", [
    ("5.54%", 5.54),
    ("-0.3%", -0.3),
    ("4.834%", 4.834),
    ("1,234.5", 1234.5),
    ("97.9", 97.9),
    (None, None),
    ("", None),
    ("—", None),
])
def test_the_number_comes_out_of_the_string(text, want):
    got = ea._num(text)
    assert (got is None) if want is None else got == pytest.approx(want)


def test_the_unit_is_kept_so_a_delta_reads_in_the_same_unit():
    assert ea._unit("5.54%") == "%"
    assert ea._unit("97.9") == ""


# ── the differences ──────────────────────────────────────────────────────────
def test_surprise_is_actual_minus_forecast():
    m = ea.measure(_ev(value="5.54%", forecast="5.40%", prev="4.83%"))
    assert m["surprise"] == pytest.approx(0.14)
    assert m["change"] == pytest.approx(0.71)
    assert m["released"] is True


def test_a_miss_is_negative():
    """日本PPI 6.8 against 7.6 is a downside surprise, not an upside one."""
    m = ea.measure(_ev(value="6.8%", forecast="7.6%", prev="7.6%"))
    assert m["surprise"] == pytest.approx(-0.8)
    assert m["change"] == pytest.approx(-0.8)


def test_an_unreleased_event_has_an_expected_change_and_no_surprise():
    m = ea.measure(_ev(value=None, forecast="0.5%", prev="0.2%"))
    assert m["released"] is False
    assert m["surprise"] is None
    assert m["expected_change"] == pytest.approx(0.3)


def test_a_missing_forecast_leaves_the_surprise_absent_not_zero():
    """Zero surprise means "landed on forecast", which is a real reading."""
    m = ea.measure(_ev(value="3.5%", forecast=None, prev="2.7%"))
    assert m["surprise"] is None
    assert m["change"] == pytest.approx(0.8)


def test_landing_exactly_on_forecast_is_a_zero_not_a_none():
    m = ea.measure(_ev(value="3.3%", forecast="3.3%", prev="2.9%"))
    assert m["surprise"] == 0
    assert m["change"] == pytest.approx(0.4)


# ── the brief ────────────────────────────────────────────────────────────────
def test_the_brief_counts_what_can_actually_be_compared():
    got = ea.build_brief([
        _ev("A", value="1%", forecast="0.5%", prev="0%"),
        _ev("B", value="2%", prev="1%"),
        _ev("C", forecast="3%", prev="2%"),
    ])
    assert got["ok"] and got["count"] == 3
    assert got["released"] == 2 and got["upcoming"] == 1
    assert got["with_surprise"] == 1


def test_events_are_ordered_by_when_they_happen():
    got = ea.build_brief([
        _ev("later", date="2026-10-14", time="08:00"),
        _ev("earlier", date="2026-10-13", time="20:30"),
        _ev("same day earlier", date="2026-10-14", time="01:00"),
    ])
    assert [r["event"] for r in got["rows"]] == [
        "earlier", "same day earlier", "later"]


def test_an_empty_selection_is_refused():
    assert ea.build_brief([])["ok"] is False


def test_too_many_events_is_refused_with_the_count():
    got = ea.build_brief([_ev(f"E{i}") for i in range(ea.MAX_EVENTS + 1)])
    assert got["ok"] is False
    assert str(ea.MAX_EVENTS) in got["reason"]


# ── the lines the model sees ─────────────────────────────────────────────────
def test_the_line_carries_the_computed_difference_not_just_the_values():
    """
    The point of the whole module: the model reads "+0.14%", it does not
    derive it.
    """
    brief = ea.build_brief([_ev("印度CPI", country="印度",
                                value="5.54%", forecast="5.40%", prev="4.83%")])
    line = ea._lines(brief)
    assert "意外+0.14%" in line
    assert "环比+0.71%" in line
    assert "[印度]" in line


def test_an_unreleased_event_is_marked_as_such_in_the_line():
    """So the model cannot say a forecast "showed" anything."""
    brief = ea.build_brief([_ev("美国CPI", forecast="0.3%", prev="0.2%")])
    line = ea._lines(brief)
    assert "未公布" in line
    assert "意外" not in line


# ── the call ─────────────────────────────────────────────────────────────────
def test_explain_sends_one_call_for_the_whole_selection(monkeypatch):
    """Batched on purpose — cheaper to ask and a better question."""
    calls = []
    fake = types.ModuleType("ai_client")

    def call_json(system, user, **kw):
        calls.append((system, user, kw))
        return {"headline": "ok", "coherent": True}

    fake.call_json = call_json
    monkeypatch.setitem(sys.modules, "ai_client", fake)

    brief = ea.build_brief([_ev(f"E{i}", value="1%", forecast="0.5%")
                            for i in range(8)])
    out = ea.explain(brief)

    assert len(calls) == 1
    assert out["count"] == 8
    # Every event reached the one prompt.
    for i in range(8):
        assert f"E{i}" in calls[0][1]


def test_the_prompt_tells_the_model_it_may_refuse_a_theme(monkeypatch):
    """The guard against a manufactured narrative lives in the prompt."""
    assert "凑不成" in ea._SYSTEM
    assert "coherent" in ea._SYSTEM


def test_the_prompt_forbids_recomputing_the_differences():
    assert "不要自己重算" in ea._SYSTEM


def test_explain_refuses_a_brief_that_is_not_ok():
    with pytest.raises(LookupError):
        ea.explain({"ok": False, "reason": "没有选中任何事件"})
