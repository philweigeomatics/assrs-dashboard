"""
价值区间 — the price ranges a stock kept returning to.

Pinned: a stock that sat at two levels gets exactly those two zones, with
the stays where they happened; a trend gets no zone rather than an invented
one; short excursions do not split a stay; a young listing starts at its
listing with the IPO weeks left out; and the PE translation is price ×
PE_then / PE_now.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import price_zones as Z  # noqa: E402

ASKED = date(2021, 11, 11)


def frame(levels, start="2021-11-15", noise=0.012, seed=3, band=0.01):
    rng = np.random.default_rng(seed)
    c = np.asarray(levels, float) * np.exp(rng.normal(0, noise, len(levels)))
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"Close": c, "High": c * (1 + band), "Low": c * (1 - band),
                         "Volume": 1e6}, index=idx)


def two_levels():
    return np.concatenate([np.full(300, 10.0), np.linspace(10, 20, 100), np.full(300, 20.0),
                           np.linspace(20, 10, 60), np.full(200, 10.0)])


def test_two_plateaus_give_two_zones_with_their_stays():
    out = Z.build(frame(two_levels()), asked_from=ASKED)
    zs = out["zones"]
    assert len(zs) == 2
    top, bottom = zs                              # highest price first
    assert top["lo"] < 20 < top["hi"] and bottom["lo"] < 10 < bottom["hi"]
    assert bottom["rank"] == 1 and top["rank"] == 2
    assert bottom["visits"] == 2 and top["visits"] == 1
    assert bottom["periods"][0]["from"] == "2021-11-15"
    assert out["now"]["zone"] == bottom["id"]
    # The bands are tight around the levels, not the whole range.
    assert top["hi"] / top["lo"] < 1.15


def test_a_steady_trend_settles_nowhere():
    out = Z.build(frame(np.geomspace(10, 60, 1000), noise=0.004), asked_from=ASKED)
    assert out["settled"] < 0.15
    assert all(z["share"] < 0.15 for z in out["zones"])


def test_short_excursions_do_not_split_a_stay():
    inside = np.ones(60, bool)
    inside[20:23] = False                          # three sessions out
    assert Z.stays(inside) == [(0, 59)]
    inside[30:40] = False                          # ten sessions out — a new stay
    assert Z.stays(inside) == [(0, 29), (40, 59)]


def test_passing_through_is_not_a_stay():
    inside = np.zeros(100, bool)
    inside[10:15] = True
    assert Z.stays(inside) == []


def test_a_young_listing_starts_at_listing_and_skips_the_ipo_weeks():
    lv = np.concatenate([np.linspace(30, 60, Z.IPO_SKIP), np.full(300, 25.0)])
    out = Z.build(frame(lv, start="2024-03-01"), asked_from=ASKED)
    assert out["listed_late"] and out["ipo_skipped"] == Z.IPO_SKIP
    assert out["start"] == "2024-03-01"
    assert out["profile_start"] > out["start"]
    # The IPO pop is drawn but carries no zone.
    assert all(z["hi"] < 30 for z in out["zones"])
    assert len(out["dates"]) == len(lv)


def test_a_full_window_skips_nothing():
    out = Z.build(frame(two_levels()), asked_from=ASKED)
    assert not out["listed_late"] and out["ipo_skipped"] == 0
    assert out["sessions"] == len(two_levels())


def test_too_little_history_is_refused():
    with pytest.raises(LookupError):
        Z.build(frame(np.full(Z.MIN_BARS - 1, 10.0), start="2026-05-01"), asked_from=ASKED)


def test_pe_translates_a_zone_into_todays_earnings():
    df = frame(two_levels())
    # PE 40 while at 20, 20 while at 10 the first time, 10 now: earnings rose.
    pe = pd.Series(np.nan, index=df.index)
    pe.iloc[:300] = 20.0
    pe.iloc[400:700] = 40.0
    pe.iloc[760:] = 10.0
    out = Z.build(df, asked_from=ASKED, pe=pe)
    top = out["zones"][0]
    assert top["pe"] == pytest.approx(40.0)
    assert out["now"]["pe"] == pytest.approx(10.0)
    assert top["pe_price"] == pytest.approx(out["now"]["close"] * 4, rel=1e-3)


def test_loss_making_pe_is_ignored():
    df = frame(two_levels())
    pe = pd.Series(-5.0, index=df.index)
    out = Z.build(df, asked_from=ASKED, pe=pe)
    assert out["now"]["pe"] is None
    assert all(z["pe"] is None and z["pe_price"] is None for z in out["zones"])


def test_profile_spreads_each_session_over_its_range():
    df = frame(np.full(200, 10.0), noise=0.0, band=0.05)
    p = Z.profile(df)
    assert p["time"].sum() == pytest.approx(200)
    assert p["volume"].sum() == pytest.approx(200 * 1e6)


def test_value_area_holds_seventy_percent_around_the_most_visited_price():
    out = Z.build(frame(two_levels()), asked_from=ASKED)
    va = out["value_area"]
    assert va["share"] >= Z.VALUE_AREA
    assert va["lo"] <= va["poc"] <= va["hi"]
    assert 9 < va["poc"] < 11
