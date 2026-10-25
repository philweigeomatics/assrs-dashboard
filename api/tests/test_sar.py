"""
Parabolic SAR on the main chart — Wilder's trailing stop.

Held to the `ta` library bar for bar, except where `ta` gets Wilder's
two-previous-lows clamp wrong; those bars are checked to be exactly that
mistake and nothing else.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ta_payload as T  # noqa: E402


def bars(seed=0, n=600, vol=0.02):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, vol, n)))
    h = c * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = c * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(h, idx), pd.Series(lo, idx), pd.Series(c, idx)


def _ta(h, lo, c):
    ta = pytest.importorskip("ta")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ind = ta.trend.PSARIndicator(h, lo, c, step=0.02, max_step=0.2)
        return ind.psar(), ind.psar_up().notna()


# ── against the library ──────────────────────────────────────────────────────
def test_it_matches_ta_almost_everywhere():
    for seed in range(4):
        h, lo, c = bars(seed)
        sar, trend = T.parabolic_sar(h, lo, c)
        ref, ref_up = _ta(h, lo, c)
        same = np.isclose(sar.iloc[2:], ref.iloc[2:], rtol=1e-9)
        assert same.mean() >= 0.99
        assert ((trend.iloc[2:] > 0) == ref_up.iloc[2:]).mean() >= 0.99


def test_where_it_differs_from_ta_ta_breaks_the_two_lows_rule():
    """Every disagreement starts on a bar where ta's long stop sits above a prior low."""
    for seed in range(6):
        h, lo, c = bars(seed)
        sar, trend = T.parabolic_sar(h, lo, c)
        ref, ref_up = _ta(h, lo, c)
        diff = np.flatnonzero(~np.isclose(sar.to_numpy(), ref.to_numpy(), rtol=1e-9))
        diff = diff[diff >= 2]
        if not len(diff):
            continue
        i = diff[0]                       # later bars inherit the first difference
        up_ref = bool(ref_up.iloc[i])
        if up_ref:
            assert ref.iloc[i] > min(lo.iloc[i - 1], lo.iloc[i - 2]) + 1e-12
        else:
            assert ref.iloc[i] < max(h.iloc[i - 1], h.iloc[i - 2]) - 1e-12


# ── Wilder's rules ───────────────────────────────────────────────────────────
def test_a_long_stop_never_sits_above_the_two_previous_lows():
    """
    Over several histories — the case ta gets wrong is rare, a few bars in a
    few hundred, and on the long side rarer still: seeds 12, 14 and 27 are
    histories where it bites a long stop.
    """
    for seed in (*range(6), 12, 14, 27):
        h, lo, c = bars(seed)
        sar, trend = T.parabolic_sar(h, lo, c)
        for i in range(3, len(sar)):
            if trend.iloc[i] > 0 and trend.iloc[i - 1] > 0:
                assert sar.iloc[i] <= min(lo.iloc[i - 1], lo.iloc[i - 2]) + 1e-9
            if trend.iloc[i] < 0 and trend.iloc[i - 1] < 0:
                assert sar.iloc[i] >= max(h.iloc[i - 1], h.iloc[i - 2]) - 1e-9


def test_a_flip_starts_the_new_stop_at_the_old_trends_extreme():
    h, lo, c = bars(2)
    sar, trend = T.parabolic_sar(h, lo, c)
    t = trend.to_numpy()
    flips = np.flatnonzero(t[3:] != t[2:-1]) + 3
    assert len(flips) > 5
    for k, f in enumerate(flips[1:], start=1):
        start = flips[k - 1]
        if t[f] < 0:                       # long → short: stop jumps to the rally's high
            assert sar.iloc[f] == pytest.approx(h.iloc[start:f].max())
        else:
            assert sar.iloc[f] == pytest.approx(lo.iloc[start:f].min())


def test_a_steady_rally_never_stops_out_and_the_stop_closes_in():
    idx = pd.bdate_range("2024-01-01", periods=80)
    c = pd.Series(np.linspace(100, 180, 80), idx)
    sar, trend = T.parabolic_sar(c * 1.005, c * 0.995, c)
    assert (trend.iloc[2:] > 0).all()
    gap = (c - sar).iloc[2:]
    assert gap.iloc[-1] < gap.iloc[5]


def test_a_crash_flips_it_short_on_the_bar_the_low_crosses_the_stop():
    idx = pd.bdate_range("2024-01-01", periods=60)
    c = pd.Series(np.r_[np.linspace(100, 150, 40), np.linspace(150, 120, 20)], idx)
    h, lo = c * 1.004, c * 0.996
    sar, trend = T.parabolic_sar(h, lo, c)
    f = int(np.flatnonzero(trend.to_numpy() < 0)[0])
    assert trend.iloc[f - 1] > 0 and 39 <= f <= 45
    # The flip bar is the one whose low went through where the long stop was heading.
    assert lo.iloc[f] < sar.iloc[f - 1] + 0.2 * (h.iloc[:f].max() - sar.iloc[f - 1])


# ── on the page ──────────────────────────────────────────────────────────────
def _frame(n=500, seed=0):
    h, lo, c = bars(seed, n)
    return pd.DataFrame({"Open": c.shift(1).fillna(c.iloc[0]), "High": h, "Low": lo,
                         "Close": c, "Volume": 1e6}, index=c.index)


def test_the_chart_gets_sar_and_its_direction_already_settled_on_the_first_bar():
    df = _frame()
    out = T.build_series(df)
    sar, tr = out["series"]["SAR"], out["series"]["SAR_Trend"]
    assert len(sar) == len(tr) == len(out["dates"])
    assert sar[0] is not None and tr[0] in (1, -1)
    full, _ = T.parabolic_sar(df["High"], df["Low"], df["Close"])
    assert sar[-1] == pytest.approx(full.iloc[-1], abs=1e-3)


def test_the_header_says_where_the_stop_is_and_since_when():
    df = _frame(seed=3)
    s = T.sar_summary(df)
    sar, trend = T.parabolic_sar(df["High"], df["Low"], df["Close"])
    assert s["trend"] == ("long" if trend.iloc[-1] > 0 else "short")
    assert s["value"] == pytest.approx(sar.iloc[-1], abs=1e-3)
    assert s["distance_pct"] > 0
    t = trend.to_numpy()
    assert all(t[-s["bars"]:] == t[-1]) and t[-s["bars"] - 1] != t[-1]


def test_too_short_a_history_has_no_sar_rather_than_a_wrong_one():
    idx = pd.bdate_range("2024-01-01", periods=2)
    sar, trend = T.parabolic_sar(pd.Series([1.0, 2.0], idx), pd.Series([0.5, 1.5], idx),
                                 pd.Series([0.8, 1.8], idx))
    assert sar.isna().all() and trend.isna().all()


def test_a_short_stop_is_reported_as_the_rise_that_would_hit_it():
    """Short: the stop is ABOVE price, so the distance is how far price must rise."""
    idx = pd.bdate_range("2024-01-01", periods=80)
    c = pd.Series(np.r_[np.linspace(100, 150, 40), np.linspace(150, 110, 40)], idx)
    df = pd.DataFrame({"High": c * 1.004, "Low": c * 0.996, "Close": c}, index=idx)
    s = T.sar_summary(df)
    assert s["trend"] == "short"
    assert s["value"] > c.iloc[-1]
    assert s["distance_pct"] == pytest.approx((s["value"] - c.iloc[-1]) / c.iloc[-1] * 100, abs=0.01)


def test_off_a_flip_the_dot_is_always_outside_the_bar():
    """
    A long stop under the low, a short stop over the high — except on the
    flip bar itself, where Wilder restarts the stop at the old extreme and a
    wide-range day can contain it (seen on 茅台 2026-05-27, AAPL 2024-12-18).
    """
    for seed in range(4):
        h, lo, c = bars(seed)
        sar, trend = T.parabolic_sar(h, lo, c)
        for i in range(3, len(sar)):
            if trend.iloc[i] != trend.iloc[i - 1]:
                continue
            if trend.iloc[i] > 0:
                assert sar.iloc[i] <= lo.iloc[i] + 1e-9
            else:
                assert sar.iloc[i] >= h.iloc[i] - 1e-9
