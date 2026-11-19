"""fair_value — normal days from the market's PE, the stock's price of growth on them."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import fair_value as fv  # noqa: E402

DAYS = pd.bdate_range("2017-01-02", periods=1000)


def market(high=(), low=(), base=12.0):
    """A flat market PE with chosen stretches made expensive or cheap."""
    v = np.full(len(DAYS), base) + np.linspace(-0.3, 0.3, len(DAYS))
    for a, b in high:
        v[a:b] = 20.0
    for a, b in low:
        v[a:b] = 8.0
    return pd.Series(v, index=DAYS)


def growth_at(g1, g3=None, ttm=100.0):
    """One report effective from the first day, carried all the way."""
    return pd.DataFrame({"end_date": ["20161231"], "ttm": [ttm], "g1": [g1],
                         "g3": [g1 if g3 is None else g3]},
                        index=pd.DatetimeIndex([DAYS[0]]))


def test_cut_offs_are_percentiles_of_the_sample():
    m = pd.Series(np.arange(1, 1001, dtype=float), index=DAYS)
    b = fv.market_band(m)
    assert b["lo"] == pytest.approx(m.quantile(0.15))
    assert b["hi"] == pytest.approx(m.quantile(0.85))
    assert (b["state"] == 0).mean() == pytest.approx(0.70, abs=0.01)


def test_a_crossing_of_a_few_days_is_not_a_regime_and_short_gaps_are_bridged():
    s = pd.Series(0, index=DAYS[:60])
    s.iloc[5:8] = 1          # 3 days: noise
    s.iloc[20:30] = 1        # a spell...
    s.iloc[30:35] = 0        # ...a 5-day return to normal...
    s.iloc[35:45] = 1        # ...and the same spell again
    got = fv.smooth_state(s)
    assert got.iloc[5:8].eq(0).all()
    assert got.iloc[20:45].eq(1).all()
    runs = fv.excluded_runs(got)
    assert len(runs) == 1 and runs[0]["kind"] == "high"


def test_the_band_comes_from_normal_days_only():
    # PEG 1.0 (PE 20 on 20% growth) on normal days, 3.0 while the market is
    # euphoric — the euphoric days must not lift the band.
    mk = market(high=[(200, 400)])
    pe = pd.Series(20.0, index=DAYS)
    pe.iloc[200:400] = 60.0
    pe.iloc[-1] = 30.0                                        # today: PE 30
    close = pd.Series(30.0, index=DAYS)
    d = fv.build(pe=pe, close=close, growth=growth_at(0.20), market_pe=mk)
    peg = d["methods"]["peg3"]
    assert peg["band"]["mid"] == pytest.approx(1.0)
    # Fair PE = 1.0 × 20 = 20 against today's 30: price 30 → fair 20.
    assert peg["fair"]["mid"] == pytest.approx(20.0)
    assert d["verdict"]["call"] == "over" and d["primary"] == "peg3"
    # At PEG 1.0 a PE of 30 implies 30% growth; 20% was delivered.
    assert peg["implied"]["mid"] == pytest.approx(0.30)
    assert d["implied"]["call"] == "over"
    json.dumps(d, allow_nan=False)


def test_slow_growth_falls_back_to_pe_and_says_why():
    mk = market()
    pe = pd.Series(15.0, index=DAYS)
    close = pd.Series(10.0, index=DAYS)
    close.iloc[-1] = 8.0
    pe.iloc[-1] = 12.0
    d = fv.build(pe=pe, close=close, growth=growth_at(0.02), market_pe=mk)
    assert d["primary"] == "pe"
    # Never 5% growth on any normal day: no band at all, and a reason.
    assert d["methods"]["peg3"]["fair"] is None and d["methods"]["peg3"]["why"]
    assert d["methods"]["pe"]["fair"]["mid"] == pytest.approx(8.0 * 15 / 12)
    assert d["verdict"]["call"] == "under"


def test_a_loss_today_gives_no_price_but_keeps_the_band():
    mk = market()
    pe = pd.Series(15.0, index=DAYS)
    pe.iloc[-20:] = -5.0
    d = fv.build(pe=pe, close=pd.Series(10.0, index=DAYS), growth=growth_at(0.2), market_pe=mk)
    assert d["now"]["pe"] is None and d["verdict"] is None
    assert d["methods"]["pe"]["band"] is not None


def test_pe_on_a_company_that_slowed_and_disagrees_with_growth_is_mixed():
    # Normal days: PE 30 on 30% growth. Today growth has slowed to 3% and the
    # PE has fallen to 20 — "cheap" on PE, dear on the growth it implies.
    mk = market()
    pe = pd.Series(30.0, index=DAYS)
    pe.iloc[-1] = 20.0
    g = pd.DataFrame({"end_date": ["20161231", "20191231"], "ttm": [100.0, 103.0],
                      "g1": [0.30, 0.03], "g3": [0.30, 0.03]},
                     index=pd.DatetimeIndex([DAYS[0], DAYS[-1]]))
    d = fv.build(pe=pe, close=pd.Series(20.0, index=DAYS), growth=g, market_pe=mk)
    assert d["methods"]["pe"]["slowed"] is True
    assert d["implied"]["call"] == "over"
    assert d["verdict"]["call"] == "mixed"


def test_cyclical_earnings_are_flagged():
    g = pd.DataFrame({"g1": [0.3, -0.2, 0.4, -0.1, 0.2], "ttm": [1, 1, 1, 1, 1]})
    assert fv.cyclical(g)["flag"] is True
    steady = pd.DataFrame({"g1": [0.1, 0.12, 0.08], "ttm": [1, 1, 1]})
    assert fv.cyclical(steady)["flag"] is False


def test_forecast_takes_the_latest_unreported_period_midpoint():
    rows = pd.DataFrame({
        "ann_date": ["20260110", "20251015"], "end_date": ["20251231", "20250930"],
        "p_change_min": [20.0, 5.0], "p_change_max": [40.0, 15.0], "type": ["预增", "略增"],
    })
    got = fv.forecast_growth(rows, "20250930")
    assert got["period"] == "20251231" and got["growth"] == pytest.approx(0.30)
    assert fv.forecast_growth(rows, "20251231") is None
