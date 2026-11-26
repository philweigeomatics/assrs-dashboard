"""Holes in Yahoo's daily bars (a split) filled from its hourly bars, and only the holes."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api import index_api as A  # noqa: E402

CAL = pd.DatetimeIndex(pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-20",
                                       "2026-11-23", "2026-11-24", "2026-11-25"]))


def daily():
    """Sandisk around its split: 11-20 no prices, 11-23/24 absent, 11-25 no close."""
    close = pd.DataFrame({"SNDK": [97.03, 94.39, np.nan, np.nan, np.nan, np.nan],
                          "AAPL": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0]}, index=CAL)
    opens = close.copy()
    opens.loc["2026-11-25", "SNDK"] = 71.315
    vol = close.notna().astype(float) * 1e6
    return {"Close": close, "Open": opens, "Volume": vol}


def hourly(scale=1.0):
    rows = []
    for d, o, c in [("2026-11-18", 100.7, 97.0), ("2026-11-19", 101.4, 94.37), ("2026-11-20", 90.0, 79.30),
                    ("2026-11-23", 80.0, 87.38), ("2026-11-24", 86.0, 77.89), ("2026-11-25", 71.3, 73.51)]:
        rows.append((pd.Timestamp(f"{d} 09:30"), o * scale, (o + c) / 2 * scale, 1e6))
        rows.append((pd.Timestamp(f"{d} 15:30"), (o + c) / 2 * scale, c * scale, 2e6))
    h = pd.DataFrame(rows, columns=["t", "Open", "Close", "Volume"]).set_index("t")
    return {"SNDK": h}


def test_the_gaps_are_the_sessions_without_a_close_after_trading_began():
    got = daily()
    gaps = A.daily_gaps(got["Close"], CAL)
    assert list(gaps) == ["SNDK"]
    assert [str(d.date()) for d in gaps["SNDK"]] == ["2026-11-20", "2026-11-23", "2026-11-24", "2026-11-25"]


def test_only_the_gaps_are_filled_from_the_hourly_closes():
    got = daily()
    done = A.fill_from_hourly(got, hourly(), A.daily_gaps(got["Close"], CAL))
    assert done == {"SNDK": ["2026-11-20", "2026-11-23", "2026-11-24", "2026-11-25"]}
    c = got["Close"]["SNDK"]
    assert c["2026-11-19"] == 94.39                       # a daily close is never overwritten
    assert abs(c["2026-11-24"] - 77.89 * 94.39 / 94.37) < 0.05
    # The open the daily bar did have is kept; the missing ones come from the first hour.
    assert got["Open"].at[pd.Timestamp("2026-11-25"), "SNDK"] == 71.315
    assert not np.isnan(got["Open"].at[pd.Timestamp("2026-11-23"), "SNDK"])
    assert got["Volume"].at[pd.Timestamp("2026-11-23"), "SNDK"] == 3e6


def test_hourly_bars_on_another_basis_are_not_used():
    """Hourly prices 8x the daily ones (an unadjusted feed) would store a crash."""
    got = daily()
    done = A.fill_from_hourly(got, hourly(scale=8.0), A.daily_gaps(got["Close"], CAL))
    assert done == {} and got["Close"]["SNDK"].isna().sum() == 4


def test_a_halted_stock_without_hourly_bars_stays_a_gap():
    got = daily()
    assert A.fill_from_hourly(got, {}, A.daily_gaps(got["Close"], CAL)) == {}


# ── the guard before a day is stored ────────────────────────────────────────
def _state():
    last = {"d": "2026-11-18", "state": {"v": {"US:A": 50.0, "US:B": 50.0}, "h": []}}
    members = [{"ticker": t, "added_on": "2026-01-02", "removed_on": None} for t in ("US:A", "US:B")]
    return last, members


def test_a_us_holding_missing_one_new_session_holds_the_day_back():
    """2026-11-19: nine members' bars missing; stored as halted, the move shifted a day."""
    import pytest
    last, members = _state()
    closes = pd.DataFrame({"US:A": [10.0, 10.5, 10.4], "US:B": [20.0, np.nan, 21.0]},
                          index=pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-20"]))
    with pytest.raises(RuntimeError, match="US:B.*2026-11-19"):
        A._check_fresh(last, closes, members, "US")


def test_an_a_share_holding_may_be_suspended_for_a_session():
    last, members = _state()
    closes = pd.DataFrame({"US:A": [10.0, 10.5, 10.4], "US:B": [20.0, np.nan, 21.0]},
                          index=pd.to_datetime(["2026-11-18", "2026-11-19", "2026-11-20"]))
    A._check_fresh(last, closes, members, "CN")             # 停牌: held flat, stored


def test_complete_us_sessions_pass():
    last, members = _state()
    closes = pd.DataFrame({"US:A": [10.0, 10.5], "US:B": [20.0, 20.5]},
                          index=pd.to_datetime(["2026-11-18", "2026-11-19"]))
    A._check_fresh(last, closes, members, "US")
