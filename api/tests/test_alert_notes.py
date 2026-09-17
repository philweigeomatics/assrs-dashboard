"""
alert_notes: resolving a prediction, and scoring it honestly.

Offline — no database. Two properties carry the whole feature:

  * A note must NOT resolve until a session later than it exists. Resolving a
    Friday note on Saturday would mark it against a bar that never traded.
  * The scorecard must compare against a baseline. A 60% hit rate on a stock
    that closes green 58% of the time is not skill, and a scorecard that omits
    that is flattery.

    python -m pytest api/tests/test_alert_notes.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import alert_notes as an  # noqa: E402


def frame(closes, opens=None, volumes=None, start="2026-09-01") -> pd.DataFrame:
    n = len(closes)
    idx = pd.bdate_range(start, periods=n)
    opens = opens if opens is not None else [c * 0.99 for c in closes]
    volumes = volumes if volumes is not None else [1_000_000.0] * n
    return pd.DataFrame({
        "Open": opens,
        "High": [max(o, c) * 1.01 for o, c in zip(opens, closes)],
        "Low": [min(o, c) * 0.99 for o, c in zip(opens, closes)],
        "Close": closes,
        "Volume": volumes,
    }, index=idx)


def note(scan_date, preds, ticker="603650"):
    return {"ticker": ticker, "scan_date": scan_date,
            "predictions": [an.normalise(p) for p in preds]}


# ── the timing rule ──────────────────────────────────────────────────────────
def test_a_note_does_not_resolve_without_a_later_session():
    """Friday's note is not wrong on Saturday — it is simply not settled."""
    df = frame([10, 11, 12])                      # last bar 2026-09-03
    n = note("2026-09-03", [{"kind": "direction_up"}])
    assert an.resolve_note(n, df) is None


def test_it_resolves_on_the_next_session_that_exists():
    """
    A gap of any length is handled by there being no row — a weekend and a
    week-long holiday look identical, and neither needs calendar logic.
    """
    df = frame([10, 11, 12])
    df = pd.concat([df, frame([13], opens=[12.5], start="2026-09-14")])  # 9 days later

    out = an.resolve_note(note("2026-09-03", [{"kind": "direction_up"}]), df)
    assert out is not None
    assert out["resolved_date"] == "2026-09-14"
    assert out["outcome"]["bar"]["close"] == 13


def test_it_resolves_against_the_very_next_bar_not_the_latest():
    """A note settles on the day after it, even once a week has passed."""
    df = frame([10, 11, 12, 9, 20])
    out = an.resolve_note(note("2026-09-02", [{"kind": "direction_up"}]), df)
    assert out["resolved_date"] == "2026-09-03"
    assert out["outcome"]["bar"]["close"] == 12


# ── the claims ───────────────────────────────────────────────────────────────
def test_direction_reads_close_against_open():
    df = frame([10, 11], opens=[10, 10.5])         # second bar 10.5 → 11, green
    up = an.resolve_note(note("2026-09-01", [{"kind": "direction_up"}]), df)
    down = an.resolve_note(note("2026-09-01", [{"kind": "direction_down"}]), df)

    assert up["outcome"]["claims"][0]["hit"] is True
    assert down["outcome"]["claims"][0]["hit"] is False


@pytest.mark.parametrize("kind, value, close, expected", [
    ("close_above", 68.06, 68.50, True),
    ("close_above", 68.06, 67.90, False),
    ("close_below", 68.06, 67.90, True),
    ("close_below", 68.06, 68.50, False),
])
def test_level_claims(kind, value, close, expected):
    df = frame([68.0, close])
    out = an.resolve_note(note("2026-09-01", [{"kind": kind, "value": value}]), df)
    assert out["outcome"]["claims"][0]["hit"] is expected


def test_a_range_accepts_its_bounds_either_way_round():
    """Typing the high number first is a typo, not a different prediction."""
    a = an.normalise({"kind": "close_between", "value": 70, "value2": 68})
    assert (a["value"], a["value2"]) == (68.0, 70.0)

    df = frame([68.0, 69.0])
    out = an.resolve_note({"ticker": "x", "scan_date": "2026-09-01", "predictions": [a]}, df)
    assert out["outcome"]["claims"][0]["hit"] is True


def test_volume_is_compared_with_the_average_of_the_previous_n_days():
    # Previous three days average 1,000; the new bar trades 600.
    df = frame([10, 10, 10, 10], volumes=[1000, 1000, 1000, 600])
    out = an.resolve_note(
        note("2026-09-03", [{"kind": "volume_below_avg", "lookback": 3}]), df)
    claim = out["outcome"]["claims"][0]

    assert claim["hit"] is True
    assert "0.60×" in claim["actual"]

    above = an.resolve_note(
        note("2026-09-03", [{"kind": "volume_above_avg", "lookback": 3}]), df)
    assert above["outcome"]["claims"][0]["hit"] is False


def test_change_between_uses_the_previous_close():
    df = frame([100.0, 103.0])
    out = an.resolve_note(
        note("2026-09-01", [{"kind": "change_between", "value": 2, "value2": 5}]), df)
    assert out["outcome"]["claims"][0]["hit"] is True
    assert "+3.00%" in out["outcome"]["claims"][0]["actual"]


def test_claims_are_scored_one_by_one():
    """The useful finding is usually that one claim held and another did not."""
    df = frame([68.0, 69.0], opens=[68.0, 70.0], volumes=[1000, 1000])
    out = an.resolve_note(note("2026-09-01", [
        {"kind": "close_above", "value": 68.06},   # 69.0 > 68.06 → hit
        {"kind": "direction_up"},                  # 70.0 → 69.0 is red → miss
    ]), df)

    hits = [c["hit"] for c in out["outcome"]["claims"]]
    assert hits == [True, False]
    assert out["outcome"]["hits"] == 1
    assert out["outcome"]["decided"] == 2
    assert out["outcome"]["score_pct"] == 50.0


def test_a_claim_nothing_can_decide_is_left_undecided():
    """An undecidable claim must not silently count as a miss."""
    df = frame([10.0, 11.0], volumes=[0.0, 5.0])
    out = an.resolve_note(note("2026-09-01", [{"kind": "volume_below_avg"}]), df)
    claim = out["outcome"]["claims"][0]
    assert claim["hit"] is None
    assert out["outcome"]["decided"] == 0
    assert out["outcome"]["score_pct"] is None


def test_unwritable_claims_are_refused_at_writing_time():
    with pytest.raises(LookupError):
        an.normalise({"kind": "close_above"})            # no level
    with pytest.raises(LookupError):
        an.normalise({"kind": "moon_phase"})             # not a thing
    with pytest.raises(LookupError):
        an.normalise({"kind": "close_above", "value": "abc"})


# ── the baseline, which is the point ─────────────────────────────────────────
def test_each_claim_carries_the_baseline_it_has_to_beat():
    """A stock that is nearly always green makes 'green' a worthless call."""
    closes = [10 + i for i in range(40)] + [55.0]      # every prior bar green
    df = frame(closes, opens=[c - 0.5 for c in closes])

    out = an.resolve_note(note(str(df.index[-2].date()), [{"kind": "direction_up"}]), df)
    claim = out["outcome"]["claims"][0]
    assert claim["hit"] is True
    assert claim["baseline"] is not None and claim["baseline"] > 90


def test_the_scorecard_reports_edge_over_the_baseline_not_the_raw_rate():
    """
    Sixty percent right against a fifty-eight percent baseline is two points of
    edge, not sixty points of skill.
    """
    notes = [{"outcome": {"claims": [
        {"kind": "direction_up", "hit": True, "baseline": 58.0},
        {"kind": "direction_up", "hit": True, "baseline": 58.0},
        {"kind": "direction_up", "hit": True, "baseline": 58.0},
        {"kind": "direction_up", "hit": False, "baseline": 58.0},
        {"kind": "direction_up", "hit": False, "baseline": 58.0},
    ]}}]
    card = an.scorecard(notes)
    row = card["rows"][0]

    assert row["rate_pct"] == 60.0
    assert row["baseline_pct"] == 58.0
    assert row["edge_pp"] == 2.0


def test_a_negative_edge_is_reported_as_negative():
    """Being worse than the naive rule is the most useful thing to learn."""
    notes = [{"outcome": {"claims": [
        {"kind": "close_above", "hit": False, "baseline": 70.0},
        {"kind": "close_above", "hit": False, "baseline": 70.0},
        {"kind": "close_above", "hit": True, "baseline": 70.0},
    ]}}]
    row = an.scorecard(notes)["rows"][0]
    assert row["edge_pp"] < 0


def test_the_scorecard_splits_by_claim_type():
    notes = [{"outcome": {"claims": [
        {"kind": "direction_up", "hit": False, "baseline": 50.0},
        {"kind": "volume_below_avg", "hit": True, "baseline": 50.0},
        {"kind": "volume_below_avg", "hit": True, "baseline": 50.0},
    ]}}]
    card = an.scorecard(notes)
    by = {r["kind"]: r for r in card["rows"]}

    assert by["volume_below_avg"]["rate_pct"] == 100.0
    assert by["direction_up"]["rate_pct"] == 0.0
    assert card["total"] == 3 and card["hits"] == 2


def test_undecided_claims_never_reach_the_scorecard():
    notes = [{"outcome": {"claims": [
        {"kind": "direction_up", "hit": None, "baseline": None},
        {"kind": "direction_up", "hit": True, "baseline": 50.0},
    ]}}]
    assert an.scorecard(notes)["total"] == 1


def test_an_empty_history_scores_nothing_rather_than_zero():
    card = an.scorecard([])
    assert card["total"] == 0
    assert card["rate_pct"] is None
    assert card["rows"] == []
