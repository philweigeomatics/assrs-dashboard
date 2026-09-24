"""
宏观与大宗 — the macro series and the commodity term structure.

The two things worth pinning: Tushare's inconsistent column case, which made
the Streamlit PMI card permanently blank, and the sign conventions on a
forward curve, where getting backwardation and contango the wrong way round
inverts the reading of every commodity on the page.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import macro  # noqa: E402


# ── column case ──────────────────────────────────────────────────────────────
def test_series_finds_a_column_whatever_case_tushare_chose():
    """
    cn_cpi returns `month`; cn_pmi returns `MONTH` and `PMI010000`. The
    Streamlit page asked for lowercase everywhere, so its PMI card has shown
    "—" since it was written while the series was there the whole time.
    """
    df = pd.DataFrame({"MONTH": ["202606", "202607", "202608"],
                       "PMI010000": [49.0, 49.2, 49.8]})
    got = macro.series(df, "month", "pmi010000")
    assert got is not None
    assert got["value"] == pytest.approx(49.8)
    assert got["change"] == pytest.approx(0.6)
    assert got["period"] == "2026-08"


def test_series_is_none_when_the_column_really_is_absent():
    df = pd.DataFrame({"month": ["202608"], "something_else": [1.0]})
    assert macro.series(df, "month", "nt_yoy") is None


def test_series_survives_a_dead_endpoint():
    assert macro.series(None, "month", "nt_yoy") is None
    assert macro.series(pd.DataFrame(), "month", "nt_yoy") is None


def test_series_sorts_by_period_before_taking_the_latest():
    """Tushare returns macro newest-first; taking iloc[-1] raw reads the oldest."""
    df = pd.DataFrame({"month": ["202608", "202607", "202606"],
                       "nt_yoy": [0.8, 0.5, 0.3]})
    got = macro.series(df, "month", "nt_yoy")
    assert got["value"] == pytest.approx(0.8)
    assert got["history"] == [0.3, 0.5, 0.8]


def test_series_ignores_periods_with_no_reading():
    df = pd.DataFrame({"month": ["202606", "202607", "202608"],
                       "nt_yoy": [0.3, None, 0.8]})
    got = macro.series(df, "month", "nt_yoy")
    assert got["value"] == pytest.approx(0.8)
    assert got["prev"] == pytest.approx(0.3)      # not None


# ── term structure ───────────────────────────────────────────────────────────
def _curve(prices, maturities, oi=None):
    return pd.DataFrame({
        "symbol": [f"X{i}" for i in range(len(prices))],
        "maturity": maturities,
        "price": prices,
        "oi": oi if oi is not None else [1000] * len(prices),
    })


def test_near_richer_than_far_is_backwardation():
    """Spot tight. Rolling the front forward pays, so the roll yield is positive."""
    t = macro.term_structure(_curve([110.0, 100.0], ["20260115", "20260214"]))
    assert t["state"] == "backwardation"
    assert t["tone"] == "up"
    assert t["spread"] == pytest.approx(10.0)
    assert t["roll_ann_pct"] > 0


def test_far_richer_than_near_is_contango():
    """Carry priced in. Rolling the front forward costs."""
    t = macro.term_structure(_curve([100.0, 110.0], ["20260115", "20260214"]))
    assert t["state"] == "contango"
    assert t["tone"] == "down"
    assert t["spread"] == pytest.approx(-10.0)
    assert t["roll_ann_pct"] < 0


def test_a_curve_within_the_flat_band_is_not_called_a_signal():
    t = macro.term_structure(_curve([100.0, 100.02], ["20260115", "20260214"]))
    assert t["state"] == "flat"


def test_roll_yield_is_annualised_over_the_real_gap_not_a_nominal_month():
    """
    Two contracts a fortnight apart carry the same spread over half the time,
    so the annualised figure has to be twice as large.
    """
    near = macro.term_structure(_curve([102.0, 100.0], ["20260101", "20260115"]))
    far = macro.term_structure(_curve([102.0, 100.0], ["20260101", "20260131"]))
    assert near["days_between"] == 14
    assert far["days_between"] == 30
    assert near["roll_ann_pct"] > far["roll_ann_pct"] * 1.9


def test_term_structure_needs_two_contracts():
    assert macro.term_structure(_curve([100.0], ["20260115"])) is None
    assert macro.term_structure(None) is None


# ── liquidity filter ─────────────────────────────────────────────────────────
def test_liquid_drops_the_dead_months():
    """
    Real shape: rebar's 2709 contract had 67 lots open against 2701's
    1,607,164. Its settlement is administrative, and leaving it in puts a
    kink in the curve that nobody could have traded.
    """
    curve = _curve([100.0, 101.0, 102.0, 103.0],
                   ["20260115", "20260215", "20260315", "20260415"],
                   oi=[200_000, 1_600_000, 4_000, 67])
    kept = macro.liquid(curve)
    assert list(kept["oi"]) == [200_000, 1_600_000]


def test_liquid_keeps_everything_when_open_interest_is_missing():
    curve = _curve([100.0, 101.0], ["20260115", "20260215"]).drop(columns=["oi"])
    assert len(macro.liquid(curve)) == 2


def test_liquid_never_returns_fewer_than_two_contracts():
    """A filter that leaves one contract has destroyed the term structure."""
    curve = _curve([100.0, 101.0, 102.0],
                   ["20260115", "20260215", "20260315"],
                   oi=[1_000_000, 10, 10])
    kept = macro.liquid(curve)
    assert len(kept) == 3          # falls back rather than yielding one row


def test_liquid_keeps_everything_when_all_open_interest_is_zero():
    curve = _curve([100.0, 101.0], ["20260115", "20260215"], oi=[0, 0])
    assert len(macro.liquid(curve)) == 2


# ── the catalogue ────────────────────────────────────────────────────────────
def test_every_product_has_a_unique_code_and_a_known_exchange():
    codes = [f["code"] for f in macro.FUTURES]
    assert len(codes) == len(set(codes))
    assert {f["exchange"] for f in macro.FUTURES} <= {"SHFE", "DCE", "CZCE", "INE"}
    assert set(macro.BY_CODE) == set(codes)


def test_every_macro_card_names_a_group_a_unit_and_an_explanation():
    for label, group, _unit, endpoint, pc, vc, _thr, note in macro.MACRO:
        assert label and group and endpoint and pc and vc
        assert note, f"{label} has no explanation"


def test_pmi_cards_carry_the_fifty_line():
    """The threshold is what makes 49.8 readable as contraction at a glance."""
    pmi = [m for m in macro.MACRO if "PMI" in m[0]]
    assert pmi, "no PMI card"
    assert all(m[6] == 50.0 for m in pmi)
