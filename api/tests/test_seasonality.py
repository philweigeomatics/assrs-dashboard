"""
季节性 — laying years over each other, and not fooling yourself about it.

The arithmetic tests pin the calendar (same month-day, same slot; weeks from
Jan 1; each year from the previous close). The statistics tests pin the part
that matters most: on pure noise the significance test must NOT find
seasonality any more often than it promises, and on a planted effect with
enough years it must.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import seasonality as S  # noqa: E402

TODAY = date(2026, 10, 18)


def walk(seed=1, plant=0.0, month=11, start="2015-12-01", end="2026-10-16",
         vol=0.02, drift=0.0003):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    r = rng.normal(drift, vol, len(idx))
    r[idx.month == month] += plant
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


# ── the calendar ─────────────────────────────────────────────────────────────
def test_the_same_month_day_is_the_same_slot_in_every_year():
    assert S.doy(date(2023, 3, 1)) == S.doy(date(2024, 3, 1)) == 60
    assert S.doy(date(2024, 2, 29)) == 59
    assert S.doy(date(2025, 12, 31)) == 365


def test_weeks_are_fixed_blocks_from_jan_1():
    assert S.bucket(S.doy(date(2025, 1, 7))) == 0
    assert S.bucket(S.doy(date(2025, 1, 8))) == 1
    assert S.bucket(S.doy(date(2025, 12, 31))) == 51
    labels = S.week_labels()
    assert len(labels) == 52 and labels[0] == "01-01~01-07" and labels[-1].endswith("12-31")


def test_a_year_is_measured_from_the_previous_close():
    """The first session's move belongs to the year it happens in."""
    idx = pd.bdate_range("2024-12-02", "2025-12-31")
    idx = idx[idx != "2025-01-01"]                           # New Year: closed
    s = pd.Series(100.0, index=idx)
    s[s.index >= "2025-01-02"] = 110.0                       # +10% on day one
    w, _ = S._weekly(s, 2025)
    assert np.expm1(w[0]) == pytest.approx(0.10)
    assert np.allclose(w[1:], 0)


def test_a_closed_week_is_flat_and_marked_untraded():
    idx = pd.bdate_range("2024-12-02", "2025-12-31")
    # 国庆, widened to cover the whole block: slot 273-279 is Sep 30 - Oct 6.
    idx = idx[~((idx >= "2025-09-29") & (idx <= "2025-10-08"))]
    s = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    w, traded = S._weekly(s, 2025)
    k = S.bucket(S.doy(date(2025, 10, 1)))
    assert w[k] == 0 and not traded[k]
    assert traded[k - 1] and traded[k + 1]


def test_this_year_stops_at_this_week():
    s = walk(end="2026-10-16")
    out = S.build(s, years=3, today=TODAY, market="US", shuffles=20)
    wk = out["abs"]["weekly"]["2026"]
    now = S.bucket(S.doy(date(2026, 10, 16)))
    assert wk[now] is not None and all(v is None for v in wk[now + 1:])


# ── finding windows ──────────────────────────────────────────────────────────
def test_with_few_years_every_year_must_agree():
    assert S.allowed_misses(3) == 0 and S.allowed_misses(5) == 0
    assert S.allowed_misses(10) == 2


def test_a_window_where_one_of_three_years_disagrees_does_not_qualify():
    W = np.zeros((3, S.WEEKS))
    W[:, 10] = [0.05, 0.05, -0.05]
    W[:, 11] = [0.05, 0.05, -0.05]
    assert not any(w["start"] in (9, 10, 11) and w["dir"] == "up" for w in S.find_windows(W))


def test_a_window_every_year_agreed_on_is_found_where_it_is():
    rng = np.random.default_rng(3)
    W = rng.normal(0, 0.01, (5, S.WEEKS))
    W[:, 20:24] += 0.03
    top = S.find_windows(W)[0]
    assert top["dir"] == "up"
    assert 18 <= top["start"] <= 22 and top["start"] + top["weeks"] >= 23


def test_the_weakest_year_is_reported_not_the_average():
    W = np.zeros((3, S.WEEKS))
    W[:, 30:32] = np.array([[0.02], [0.05], [0.10]])
    w = next(x for x in S.find_windows(W) if x["dir"] == "up")
    assert np.expm1(w["weak"]) == pytest.approx(np.expm1(0.04), rel=1e-6)


def test_the_shuffle_keeps_each_year_and_only_moves_the_dates():
    """A rotation must not change what a year did — only when."""
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, (4, S.WEEKS))
    n = W.shape[0]
    off = rng.integers(0, S.WEEKS, n)
    idx = (np.arange(S.WEEKS)[None, :] - off[:, None]) % S.WEEKS
    R = W[np.arange(n)[:, None], idx]
    assert np.allclose(np.sort(R, axis=1), np.sort(W, axis=1))


def test_the_test_is_deterministic():
    W = np.random.default_rng(5).normal(0, 0.02, (3, S.WEEKS))
    assert np.array_equal(S.null_best(W, 50), S.null_best(W, 50))


# ── the statistics ───────────────────────────────────────────────────────────
def test_noise_is_not_called_seasonal_more_often_than_promised():
    """
    On random walks, p < 0.05 should come up about 5% of the time. Allow
    generous slack for 40 draws, but a broken null — say one that ignores
    each year's trend — calls far more than that significant.
    """
    ps = [S.build(walk(seed), years=5, today=TODAY, market="US", shuffles=100)
          ["abs"]["significance"]["p_best"] for seed in range(40)]
    assert np.mean(np.array(ps) < 0.05) <= 0.15


def test_a_strong_planted_effect_is_found_with_ten_years():
    hits = 0
    for seed in range(10):
        a = S.build(walk(seed, plant=0.006), years=10, today=TODAY, market="US",
                    shuffles=100)["abs"]
        top = a["windows"][0]
        if a["significance"]["p_best"] < 0.05 and top["dir"] == "up" and "10-" <= top["from"] <= "11-30":
            hits += 1
    assert hits >= 6


def test_three_years_of_noise_says_it_cannot_tell():
    a = S.build(walk(2), years=3, today=TODAY, market="US", shuffles=100)["abs"]
    if a["significance"]["verdict"] != "strong":
        assert any("很难与运气区分" in c for c in a["conclusions"])


def test_a_stock_that_is_its_index_has_no_excess_seasonality():
    s = walk(7)
    out = S.build(s, years=5, today=TODAY, market="US", bench=s * 3, shuffles=50)
    assert out["excess"]["windows"] == []
    assert out["abs"]["windows"]


# ── what comes next, and this year ───────────────────────────────────────────
def test_ahead_uses_the_same_date_each_year_and_skips_years_without_the_data():
    s = walk(end="2026-10-16")
    h = S.ahead(s, [2023, 2024, 2025], TODAY, 30)
    assert set(h["per_year"]) == {"2023", "2024", "2025"}
    exp = s[:"2023-11-17"].iloc[-1] / s[:"2023-10-18"].iloc[-1] - 1
    assert h["per_year"]["2023"] == pytest.approx(exp * 100, abs=0.01)
    late = S.ahead(s, [2025], date(2026, 12, 20), 30)      # 2026-01-19 exists
    assert late["n"] == 1
    gone = S.ahead(s[:"2025-12-31"], [2025], date(2026, 12, 20), 30)
    assert gone["n"] == 0


def test_feb_29_falls_back_to_feb_28():
    assert S._same_day(2025, date(2024, 2, 29)) == date(2025, 2, 28)


def test_ytd_ranks_this_year_among_the_same_date_in_past_years():
    idx = pd.bdate_range("2022-12-01", "2026-10-16")
    s = pd.Series(100.0, index=idx)
    for y, g in [(2023, 1.05), (2024, 1.20), (2025, 0.90), (2026, 1.10)]:
        s[(s.index >= f"{y}-03-01")] *= g
    t = S.ytd(s, [2023, 2024, 2025], 2026)
    # Each year rebases on the previous close, so its YTD is exactly its own g.
    assert t["now"] == pytest.approx(10.0)
    assert t["rank"] == 2 and t["of"] == 4


# ── A-shares ─────────────────────────────────────────────────────────────────
def test_spring_festival_is_measured_ten_sessions_either_side():
    idx = pd.bdate_range("2024-12-01", "2025-03-31")
    idx = idx[~((idx >= "2025-01-28") & (idx <= "2025-02-04"))]
    s = pd.Series(100.0, index=idx)
    pre = s.index[s.index < "2025-01-29"]
    s[s.index >= pre[-10]] = 110.0                           # +10% into it, then flat
    out = S.spring(s, [2025])[0]
    assert out["before"] == pytest.approx(10.0) and out["after"] == pytest.approx(0.0)


def test_a_share_years_are_tagged_by_the_924_pivot():
    out = S.build(walk(), years=5, today=TODAY, market="CN", shuffles=20)
    tags = {y["year"]: y["regime"] for y in out["years"]}
    assert tags == {2021: "pre", 2022: "pre", 2023: "pre", 2024: "pivot", 2025: "post"}
    assert out["spring_dates"]["2026"] == "02-17"


def test_other_markets_get_no_924_tag_and_no_spring_festival():
    out = S.build(walk(), years=3, today=TODAY, market="US", shuffles=20)
    assert all(y["regime"] is None for y in out["years"])
    assert out["spring_dates"] == {} and out["abs"]["spring"] == []


# ── the edges ────────────────────────────────────────────────────────────────
def test_a_year_listed_part_way_through_is_skipped_and_said_so():
    s = walk(start="2023-06-01")
    out = S.build(s, years=3, today=TODAY, market="US", shuffles=20)
    assert out["skipped"] == [2023]
    assert [y["year"] for y in out["years"]] == [2024, 2025]


def test_fewer_than_two_full_years_is_an_error_not_a_chart():
    with pytest.raises(LookupError):
        S.build(walk(start="2024-08-01"), years=3, today=TODAY, market="US")


def test_only_the_offered_year_counts_are_accepted():
    with pytest.raises(ValueError):
        S.build(walk(), years=4, today=TODAY, market="US")


def test_the_fetch_starts_the_december_before_the_first_year():
    assert S.start_date(3, TODAY) == date(2022, 12, 1)


# ── 劳动节 / 国庆 / 春节 ─────────────────────────────────────────────────────
def _cn_days(start="2019-12-01", end="2026-10-16", closed=()):
    idx = pd.bdate_range(start, end)
    for a, b in closed:
        idx = idx[~((idx >= a) & (idx <= b))]
    return idx


def test_the_closure_is_found_from_the_bars_not_a_table():
    """2023's 国庆 began Sep 29 with 中秋: the last session was Sep 28."""
    idx = _cn_days(closed=[("2023-09-29", "2023-10-06")])
    s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    r = next(x for x in S.holiday(s, "national", [2023], 5) if x["year"] == 2023)
    assert r["last_before"] == "09-28" and r["reopen"] == "10-09"


def test_after_counts_the_gap_on_reopening():
    idx = _cn_days(closed=[("2025-10-01", "2025-10-08")])
    s = pd.Series(100.0, index=idx)
    s[s.index >= "2025-10-09"] = 105.0                       # gaps up on reopening
    r = S.holiday(s, "national", [2025], 5)[0]
    assert r["before"] == pytest.approx(0.0) and r["after"] == pytest.approx(5.0)


def test_before_is_the_last_n_sessions_into_the_closure():
    idx = _cn_days(closed=[("2025-05-01", "2025-05-05")])
    s = pd.Series(100.0, index=idx)
    pre = s.index[s.index < "2025-05-01"]
    s[s.index >= pre[-5]] = 103.0                            # rises over the last 5 only
    assert S.holiday(s, "labour", [2025], 5)[0]["before"] == pytest.approx(3.0)
    assert S.holiday(s, "labour", [2025], 10)[0]["before"] == pytest.approx(3.0)
    s2 = s.copy()
    s2[s2.index >= pre[-4]] = 103.0
    s2[pre[-5]] = 100.0
    assert S.holiday(s2, "labour", [2025], 5)[0]["before"] == pytest.approx(3.0)


def test_one_huge_year_does_not_make_a_holiday_look_reliable():
    """
    茅台 before 国庆 2024: +38.6% from the 924 rally; the typical year +1%.
    Planted into random walks, that one year must be named as the outlier
    and must not, on its own, get the holiday called significant — beyond the
    5% of the time noise alone would.
    """
    flagged, named = 0, 0
    for seed in range(30):
        rng = np.random.default_rng(seed)
        idx = _cn_days()
        s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx)))), index=idx)
        pre = s.index[s.index < "2024-10-01"]
        s[s.index >= pre[-5]] *= 1.40
        rows = S.holiday(s, "national", [2020, 2021, 2022, 2023, 2024, 2025], 5)
        side = S._side(s, rows, "before", 5, None)
        flagged += side["p"] < 0.05
        named += bool(side["outlier"]) and side["outlier"]["year"] == 2024
    # Measured: 6/30 with the plant, 3/30 without it. One year can still nudge
    # a six-year median; on the mean, 6 of 6 years at +9% average flags nearly
    # every run.
    assert flagged <= 9
    assert named >= 25


def test_a_move_every_year_shares_is_called_out():
    rng = np.random.default_rng(9)
    idx = _cn_days()
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx)))), index=idx)
    for y in range(2020, 2026):
        pre = s.index[s.index < f"{y}-10-01"]
        s[s.index >= pre[-5]] *= 1.06
    rows = S.holiday(s, "national", list(range(2020, 2026)), 5)
    side = S._side(s, rows, "before", 5, None)
    assert side["ups"] == 6 and side["p"] < 0.05 and side["outlier"] is None


def test_holidays_are_a_shares_only():
    us = S.build(walk(), years=3, today=TODAY, market="US", shuffles=20)
    cn = S.build(walk(), years=3, today=TODAY, market="CN", shuffles=20)
    assert us["abs"]["holidays"] == {}
    assert set(cn["abs"]["holidays"]) == {"spring", "labour", "national"}
    assert set(cn["abs"]["holidays"]["national"]) == {"5", "10"}


def test_the_conclusions_name_the_year_doing_the_pulling():
    idx = _cn_days()
    rng = np.random.default_rng(4)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))), index=idx)
    pre = s.index[s.index < "2024-10-01"]
    s[s.index >= pre[-5]] *= 1.40
    out = S.build(s, years=5, today=TODAY, market="CN", shuffles=20)
    line = next(c for c in out["abs"]["conclusions"] if c.startswith("国庆"))
    assert "2024" in line and "中位" in line


def test_the_random_comparison_uses_the_same_statistic_as_the_holiday():
    """
    A stock that sits still and then jumps 10% one session in five. Its
    typical day is 0%, so a median of +1% is unusual — but an average of +1%
    is what it does almost every time. Comparing a holiday's median against
    random averages would call ordinary behaviour remarkable, or the reverse.
    """
    idx = _cn_days("2019-12-01", "2025-12-31")
    jumps = np.where(np.arange(len(idx)) % 5 == 0, np.log(1.10), 0.0)
    s = pd.Series(100 * np.exp(np.cumsum(jumps)), index=idx)
    p = S.placebo_p(s, [2020, 2021, 2022, 2023, 2024, 2025], 1, 1.0)
    assert p < 0.3
