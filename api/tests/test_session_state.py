"""
Is the newest bar a finished session?

The bug behind this module: 尾盘推演 reads the last daily bar as a completed
session, which is true of Tushare (it publishes after the close) and false of
Yahoo (it serves a running bar). At 11:52 in New York, AAPL's "daily" bar held
13.3M shares against a 20-day average of 42.3M — read as a finished day that
is 0.32x normal, a dead tape. 36% of the session had happened, so the real
pace was 0.87x: an ordinary morning.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import session_state as ss  # noqa: E402

NY = ZoneInfo("America/New_York")
SH = ZoneInfo("Asia/Shanghai")
THU = date(2026, 10, 1)          # a Thursday
CN_THU = date(2026, 10, 8)       # a Thursday after the holiday


# ── session length ───────────────────────────────────────────────────────────
def test_session_lengths_are_the_regular_sessions():
    assert ss.session_minutes("US") == 390        # 09:30–16:00
    assert ss.session_minutes("CA") == 390
    assert ss.session_minutes("CN") == 240        # 4 hours, lunch excluded


def test_an_unknown_market_does_not_crash():
    assert ss.session_minutes("XX") == ss.session_minutes("US")


# ── elapsed fraction ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("hh, mm, want", [
    (8, 0, 0.0),           # pre-market is not the session
    (9, 30, 0.0),
    (12, 45, 0.5),         # exactly half of 09:30–16:00
    (16, 0, 1.0),
    (20, 0, 1.0),          # after hours do not extend it past 1
])
def test_us_elapsed(hh, mm, want):
    got = ss.elapsed_fraction("US", datetime(2026, 10, 1, hh, mm, tzinfo=NY))
    assert got == pytest.approx(want, abs=0.005)


@pytest.mark.parametrize("hh, mm, want", [
    (10, 30, 0.25),
    (11, 30, 0.50),        # morning session ends
    (12, 0, 0.50),         # …and lunch adds nothing
    (12, 59, 0.50),
    (13, 30, 0.625),
    (15, 0, 1.0),
])
def test_the_a_share_lunch_break_is_not_trading_time(hh, mm, want):
    """
    Counting 11:30–13:00 would put a noon bar at 62.5% rather than 50% and
    overstate how formed it is by a quarter.
    """
    got = ss.elapsed_fraction("CN", datetime(2026, 10, 8, hh, mm, tzinfo=SH))
    assert got == pytest.approx(want, abs=0.005)


def test_a_weekend_is_a_closed_session_whatever_the_clock_says():
    sat = datetime(2026, 10, 3, 12, 0, tzinfo=NY)
    assert ss.elapsed_fraction("US", sat) == 1.0


# ── bar state ────────────────────────────────────────────────────────────────
def test_a_bar_dated_before_today_is_finished():
    now = datetime(2026, 10, 1, 11, 52, tzinfo=NY)
    assert ss.state("US", date(2026, 9, 30), now).complete is True


def test_todays_bar_is_unfinished_while_the_session_runs():
    now = datetime(2026, 10, 1, 11, 52, tzinfo=NY)
    bar = ss.state("US", THU, now)
    assert bar.complete is False
    assert bar.elapsed_pct == pytest.approx(36.4, abs=0.2)


def test_todays_bar_is_finished_once_the_bell_goes():
    now = datetime(2026, 10, 1, 16, 0, tzinfo=NY)
    assert ss.state("US", THU, now).complete is True


def test_an_a_share_bar_read_in_the_evening_is_finished():
    """
    The common case and the one that must not regress: Tushare publishes
    after the close, so every A-share bar reaching this code is whole.
    """
    now = datetime(2026, 10, 8, 23, 50, tzinfo=SH)
    assert ss.state("CN", CN_THU, now).complete is True


def test_a_bar_dated_in_the_future_is_treated_as_finished():
    """A clock or data problem, not a live session — do not invent a fraction."""
    now = datetime(2026, 10, 1, 11, 52, tzinfo=NY)
    bar = ss.state("US", date(2026, 10, 5), now)
    assert bar.complete is True and bar.elapsed == 1.0


def test_an_unparseable_date_does_not_raise():
    assert ss.state("US", "not-a-date").complete in (True, False)


# ── projection ───────────────────────────────────────────────────────────────
def test_the_aapl_case_end_to_end():
    """The measurement this module exists for."""
    now = datetime(2026, 10, 1, 11, 52, tzinfo=NY)
    bar = ss.state("US", THU, now)
    raw, avg = 13_346_489, 42_303_870

    assert raw / avg == pytest.approx(0.32, abs=0.01)        # what it said
    assert bar.project(raw) / avg == pytest.approx(0.87, abs=0.02)   # what it is


def test_a_finished_bar_projects_to_itself():
    """No scaling on a whole day, or every A-share volume would move."""
    now = datetime(2026, 10, 8, 23, 50, tzinfo=SH)
    bar = ss.state("CN", CN_THU, now)
    assert bar.project(38_331) == 38_331


def test_the_first_minutes_are_not_projected():
    """
    The opening auction scales to nonsense: 2% of a session carrying the
    overnight order book becomes 50x a normal day.
    """
    now = datetime(2026, 10, 1, 9, 40, tzinfo=NY)
    bar = ss.state("US", THU, now)
    assert bar.projectable is False
    assert bar.project(1_000_000) is None


def test_projection_grows_as_the_session_shrinks():
    raw = 10_000_000
    quarter = ss.state("US", THU, datetime(2026, 10, 1, 11, 7, tzinfo=NY))
    half = ss.state("US", THU, datetime(2026, 10, 1, 12, 45, tzinfo=NY))
    assert quarter.project(raw) > half.project(raw)
    assert half.project(raw) == pytest.approx(raw * 2, rel=0.01)


def test_project_passes_none_through():
    bar = ss.state("US", THU, datetime(2026, 10, 1, 12, 45, tzinfo=NY))
    assert bar.project(None) is None


# ── the line that reaches a human ────────────────────────────────────────────
def test_describe_says_which_situation_it_is():
    live = ss.state("US", THU, datetime(2026, 10, 1, 11, 52, tzinfo=NY))
    done = ss.state("CN", CN_THU, datetime(2026, 10, 8, 23, 50, tzinfo=SH))
    assert "仍在交易中" in ss.describe(live) and "36" in ss.describe(live)
    assert "已收盘" in ss.describe(done)


# ── what the brief actually hands the model ──────────────────────────────────
def _frame(n=160, vol=40_000_000.0):
    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2026-01-02", periods=n)
    rng = np.random.default_rng(5)
    close = 300 + np.cumsum(rng.normal(0, 2, n))
    # Volume varies: a constant series has zero standard deviation, so a
    # Z-score does not exist and the test would be asserting against the
    # fixture rather than the code. The mean stays at `vol`.
    noise = rng.normal(0, vol * 0.1, n)
    return pd.DataFrame({
        "Open": close, "High": close + 2, "Low": close - 2, "Close": close,
        "Volume": np.clip(vol + noise - noise.mean(), vol * 0.5, None),
    }, index=idx)


def _brief(bar, last_volume):
    import whatif_advisor as wadv
    from analysis_engine import run_single_stock_analysis

    df = _frame()
    df.iloc[-1, df.columns.get_loc("Volume")] = last_volume
    adf = run_single_stock_analysis(df.copy())
    sim = wadv.sim_from_real_bar(adf)
    return wadv.build_brief(adf.iloc[:-1], sim, ticker="X", name="X",
                            mode="actual", bar=bar,
                            bar_date=str(adf.index[-1].date()))


def test_a_live_bar_is_compared_on_its_projection_not_its_raw_volume():
    """
    The fix, at the level the model sees it. A third of a day's volume set
    against an average of whole days is what read 0.32x; the projection is
    what makes the ratio mean anything.
    """
    live = ss.state("US", THU, datetime(2026, 10, 1, 12, 45, tzinfo=NY))  # 50%
    got = _brief(live, 20_000_000.0)["simulated_bar"]

    assert got["volume"] == pytest.approx(20_000_000)          # what traded
    assert got["volume_projected"] == pytest.approx(40_000_000)  # the day
    assert got["volume_partial"] is True
    # Against a 40M history: raw would read 0.5x, the projection reads 1.0x.
    assert got["volume_vs_10d"] == pytest.approx(1.0, abs=0.08)


def test_a_finished_bar_is_untouched():
    """The A-share path must produce exactly what it always did."""
    done = ss.state("CN", CN_THU, datetime(2026, 10, 8, 23, 50, tzinfo=SH))
    got = _brief(done, 20_000_000.0)["simulated_bar"]

    assert got["volume_projected"] is None
    assert got["volume_partial"] is False
    assert got["volume_vs_10d"] == pytest.approx(0.5, abs=0.05)


def test_no_bar_given_behaves_like_a_finished_one():
    """A caller that knows nothing about sessions keeps the old behaviour."""
    got = _brief(None, 20_000_000.0)["simulated_bar"]
    assert got["volume_projected"] is None
    assert got["volume_vs_10d"] == pytest.approx(0.5, abs=0.05)


def test_the_brief_carries_the_session_so_the_prompt_can_reframe():
    live = ss.state("US", THU, datetime(2026, 10, 1, 12, 45, tzinfo=NY))
    sess = _brief(live, 20_000_000.0)["session"]
    assert sess["complete"] is False
    assert sess["elapsed_pct"] == pytest.approx(50.0, abs=0.5)
    assert sess["market"] == "US"


def test_a_simulated_bar_has_no_session_block():
    """A hypothetical bar is complete by construction; nothing to project."""
    assert _brief(None, 20_000_000.0)["session"] is None


def test_the_us_volume_z_is_no_longer_blank():
    """
    Separate gap found on the way: the Yahoo path never wrote Vol_Mean_100d,
    so a US name had no volume Z in the brief at all.
    """
    live = ss.state("US", THU, datetime(2026, 10, 1, 12, 45, tzinfo=NY))
    assert _brief(live, 20_000_000.0)["simulated_bar"]["volume_z"] is not None
