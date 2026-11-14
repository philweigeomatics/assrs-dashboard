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


# ── PEG: growth as the market could see it ────────────────────────────────────
def reports(profits: dict[str, float], ann: dict[str, str] | None = None) -> pd.DataFrame:
    """Cumulative year-to-date profits by period end, announced a month after by default."""
    rows = []
    for e, v in profits.items():
        a = (ann or {}).get(e) or (pd.Timestamp(e) + pd.Timedelta(days=30)).strftime("%Y%m%d")
        rows.append({"ann_date": a, "end_date": e, "profit": v})
    return pd.DataFrame(rows)


def test_ttm_from_cumulative_quarters():
    # 2023: 25 a quarter (FY 100). 2024: 30 a quarter.
    r = reports({"20231231": 100, "20240331": 30, "20240630": 60, "20240930": 90, "20241231": 120,
                 "20230331": 25, "20230630": 50, "20230930": 75})
    t = Z._ttm_profits(r).set_index("end_date")["ttm"]
    assert t["20240331"] == pytest.approx(105)          # 30 + 100 - 25
    assert t["20240930"] == pytest.approx(115)          # 90 + 100 - 75
    assert t["20241231"] == pytest.approx(120)
    assert "20230331" not in t.index                    # no 2022 to complete it


def test_growth_counts_from_the_day_after_announcement():
    yrs = {f"{y}1231": 100 * 1.2 ** (y - 2019) for y in range(2019, 2025)}
    g = Z.profit_growth(reports(yrs))
    last = g.iloc[-1]
    assert last["end_date"] == "20241231"
    assert last["g1"] == pytest.approx(0.2)
    assert last["g3"] == pytest.approx(0.2)
    assert g.index[-1] == pd.Timestamp("20241231") + pd.Timedelta(days=31)


def test_the_first_announcement_counts_not_the_restatement():
    r = reports({"20221231": 100, "20231231": 120})
    r = pd.concat([r, pd.DataFrame([{"ann_date": "20240901", "end_date": "20231231", "profit": 150}])])
    g = Z.profit_growth(r)
    assert g.iloc[-1]["g1"] == pytest.approx(0.2)


def test_a_late_report_does_not_step_growth_back():
    r = reports({"20221231": 100, "20231231": 120, "20230331": 30, "20240331": 33},
                ann={"20231231": "20240420", "20240331": "20240410"})
    g = Z.profit_growth(r)
    # Q1 2024 needs FY2023 for its TTM, so it only exists once FY2023 is out;
    # FY2023 announced after Q1 2024 must not replace the later period.
    assert list(g["end_date"])[-1] >= "20231231"
    assert g["end_date"].is_monotonic_increasing


def test_peg_is_pe_over_growth_and_skips_slow_growers():
    df = frame(two_levels())
    pe = pd.Series(20.0, index=df.index)
    growth = pd.DataFrame({"end_date": ["20201231"], "ttm": [1.0], "g1": [0.25], "g3": [0.02]},
                          index=[pd.Timestamp("2020-01-01")])
    out = Z.build(df, asked_from=ASKED, pe=pe, growth=growth)
    assert out["has_growth"]
    assert out["now"]["peg1"] == pytest.approx(0.8)       # 20 / 25
    assert out["now"]["peg3"] is None                     # 2% growth: PEG does not apply
    z = out["zones"][0]
    assert z["peg1"] == pytest.approx(0.8) and z["g1"] == pytest.approx(0.25)
    assert z["peg1_price"] == pytest.approx(out["now"]["close"], rel=1e-3)
    assert z["peg3"] is None and z["peg3_price"] is None
    # Today's PE of 20 at the zone's PEG of 0.8 is a bet on 25% growth.
    assert z["peg1_implied"] == pytest.approx(0.25)
    assert z["peg3_implied"] is None


def test_implied_growth_needs_no_current_growth():
    df = frame(two_levels())
    pe = pd.Series(20.0, index=df.index)
    # 25% growth while the stock sat in its first zone, a loss of growth now.
    growth = pd.DataFrame({"end_date": ["20201231", "20250101"], "ttm": [1.0, 1.0],
                           "g1": [0.25, -0.10], "g3": [np.nan, np.nan]},
                          index=[pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01")])
    out = Z.build(df, asked_from=ASKED, pe=pe, growth=growth)
    first = next(z for z in out["zones"] if z["first"] == "2021-11-15")
    assert out["now"]["peg1"] is None                       # shrinking: PEG does not apply
    assert first["peg1"] == pytest.approx(0.8)
    assert first["peg1_implied"] == pytest.approx(0.25)     # yet the bet is still readable


def test_no_growth_data_leaves_peg_out():
    out = Z.build(frame(two_levels()), asked_from=ASKED, pe=pd.Series(20.0, index=frame(two_levels()).index))
    assert not out["has_growth"]
    assert out["now"]["peg1"] is None and all(z["peg1"] is None for z in out["zones"])
