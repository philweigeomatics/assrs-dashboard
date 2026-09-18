"""
exposure: where the book really is, once you look through the ETFs.

Offline. The interesting property is the look-through: a book that is 60% VFV
and 40% one technology name is not "60% ETF, 40% technology" — it is most of
the way to a technology bet, and only the look-through says so.

    python -m pytest api/tests/test_exposure.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import exposure as ex  # noqa: E402


def hold(symbol, value, group="stock"):
    return {"symbol": symbol, "market_value_base": value, "group": group}


def pct(res, key):
    return next((r["pct"] for r in res["rows"] if r["sector"] == key), 0.0)


# ── the spelling problem ─────────────────────────────────────────────────────
@pytest.mark.parametrize("raw", ["Real Estate", "realestate", "real_estate", "REAL ESTATE"])
def test_the_two_yahoo_spellings_land_on_one_sector(raw):
    """
    A stock's sector is Title Case, an ETF's weighting keys are snake_case with
    one irregular member. Left alone, real-estate exposure splits across two
    rows that never add up to anything.
    """
    assert ex.canon(raw) == "real_estate"


@pytest.mark.parametrize("raw, want", [
    ("Technology", "technology"),
    ("Financial Services", "financial_services"),
    ("financial_services", "financial_services"),
    ("Communication Services", "communication_services"),
    ("Basic Materials", "basic_materials"),
    ("Healthcare", "healthcare"),
])
def test_every_sector_name_normalises(raw, want):
    assert ex.canon(raw) == want


def test_something_that_is_not_a_sector_is_not_invented():
    assert ex.canon("Blockchain") is None
    assert ex.canon("") is None and ex.canon(None) is None


# ── look-through ─────────────────────────────────────────────────────────────
def test_an_etfs_weight_is_spread_across_the_sectors_it_actually_holds():
    res = ex.analyse(
        [hold("VFV.TO", 6000, "etf"), hold("AAPL", 4000)],
        {"VFV.TO": {"weights": {"technology": 0.5, "healthcare": 0.3,
                                "energy": 0.2}},
         "AAPL": {"sector": "Technology", "industry": "Consumer Electronics"}})

    # 60% of the book × 50% tech, plus the whole 40% Apple position.
    assert pct(res, "technology") == pytest.approx(30 + 40, abs=0.01)
    assert pct(res, "healthcare") == pytest.approx(18, abs=0.01)
    assert pct(res, "energy") == pytest.approx(12, abs=0.01)
    assert sum(r["pct"] for r in res["rows"]) == pytest.approx(100, abs=0.05)


def test_the_two_sources_of_a_sector_are_reported_separately():
    """"I chose this" and "my index fund chose this" are different facts."""
    res = ex.analyse(
        [hold("VFV.TO", 6000, "etf"), hold("AAPL", 4000)],
        {"VFV.TO": {"weights": {"technology": 1.0}},
         "AAPL": {"sector": "Technology"}})
    tech = next(r for r in res["rows"] if r["sector"] == "technology")

    assert tech["from_etfs_pct"] == pytest.approx(60, abs=0.01)
    assert tech["from_stocks_pct"] == pytest.approx(40, abs=0.01)
    assert sorted(tech["holdings"]) == ["AAPL", "VFV.TO"]


def test_an_etf_that_cannot_be_looked_through_is_unknown_not_guessed():
    res = ex.analyse([hold("MYSTERY.TO", 1000, "etf")],
                     {"MYSTERY.TO": {"sector": "Financial Services"}})
    # Its own listing's sector is not the sector of what it holds.
    assert res["unknown_pct"] == pytest.approx(100, abs=0.01)
    assert pct(res, "financial_services") == 0.0


def test_weightings_that_do_not_quite_sum_to_one_are_rescaled():
    res = ex.analyse([hold("E", 1000, "etf")],
                     {"E": {"weights": {"technology": 0.49, "energy": 0.49}}})
    assert pct(res, "technology") == pytest.approx(50, abs=0.1)
    assert res["unknown_pct"] == 0.0


def test_badly_incomplete_weightings_are_not_rescaled_into_invention():
    """Stretching a 40% total up to 100% would fabricate exposure."""
    res = ex.analyse([hold("E", 1000, "etf")],
                     {"E": {"weights": {"technology": 0.4}}})
    assert pct(res, "technology") == pytest.approx(40, abs=0.1)
    assert res["unknown_pct"] == pytest.approx(60, abs=0.1)


# ── stocks ───────────────────────────────────────────────────────────────────
def test_a_stock_contributes_its_whole_weight_to_one_sector():
    res = ex.analyse([hold("AAPL", 2500), hold("ENB.TO", 7500)],
                     {"AAPL": {"sector": "Technology"},
                      "ENB.TO": {"sector": "Energy"}})
    assert pct(res, "technology") == pytest.approx(25, abs=0.01)
    assert pct(res, "energy") == pytest.approx(75, abs=0.01)


def test_an_unknown_sector_is_counted_not_dropped():
    """
    Dropping it would quietly inflate every other sector — the one error
    nobody notices, because the remaining numbers still add to 100.
    """
    res = ex.analyse([hold("AAPL", 5000), hold("???", 5000)],
                     {"AAPL": {"sector": "Technology"}})
    assert pct(res, "technology") == pytest.approx(50, abs=0.01)
    assert res["unknown_pct"] == pytest.approx(50, abs=0.01)


def test_industries_are_stocks_only_and_say_so():
    """
    An ETF publishes sector weights, never industry weights. A portfolio-wide
    industry table cannot exist, so this one is explicitly based on the stock
    sleeve alone.
    """
    res = ex.analyse(
        [hold("VFV.TO", 6000, "etf"), hold("AAPL", 3000), hold("NVDA", 1000)],
        {"VFV.TO": {"weights": {"technology": 1.0}},
         "AAPL": {"sector": "Technology", "industry": "Consumer Electronics"},
         "NVDA": {"sector": "Technology", "industry": "Semiconductors"}})

    assert res["industry_basis"] == 4000
    assert {r["name"]: r["pct"] for r in res["industries"]} == {
        "Consumer Electronics": 75.0, "Semiconductors": 25.0}
    assert res["split"] == {"stock_pct": 40.0, "etf_pct": 60.0}


def test_concentration_names_the_largest_sector():
    res = ex.analyse([hold("A", 5000), hold("B", 3000), hold("C", 2000)],
                     {"A": {"sector": "Technology"}, "B": {"sector": "Energy"},
                      "C": {"sector": "Utilities"}})
    c = res["concentration"]
    assert c["sectors"] == 3
    assert c["top_sector"] == ex.SECTORS["technology"]
    assert c["top_pct"] == pytest.approx(50, abs=0.01)
    assert c["top3_pct"] == pytest.approx(100, abs=0.05)


def test_an_empty_book_is_refused_rather_than_divided_by_zero():
    with pytest.raises(LookupError):
        ex.analyse([], {})
    with pytest.raises(LookupError):
        ex.analyse([hold("A", 0)], {})


def test_rows_come_back_largest_first():
    res = ex.analyse([hold("A", 1000), hold("B", 9000)],
                     {"A": {"sector": "Technology"}, "B": {"sector": "Energy"}})
    assert [r["sector"] for r in res["rows"]] == ["energy", "technology"]
