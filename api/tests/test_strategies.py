"""
strategies.t_trading and strategies.mean_reversion, on frames built to trip
each rule on purpose.

Offline. These are ports of Streamlit pages, and the risk in a port is that a
threshold or a weight quietly changes meaning — so the gates, the weights and
the verdict cuts are asserted against the values the pages use, not just
"something plausible came out".

    python -m pytest api/tests/test_strategies.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from strategies import mean_reversion as mr, t_trading as tt  # noqa: E402

N = 200
DATES = pd.bdate_range("2025-01-01", periods=N)


def bars(*, range_pct=4.0, bias=0.2, drift=0.0, base=100.0, n=N) -> pd.DataFrame:
    """
    A price frame with a chosen average intraday range and close-vs-open bias.

    `bias` is |Close − Open| ÷ (High − Low): 0 closes mid-range (an oscillator),
    1 closes at the extreme (a trender).
    """
    idx = DATES[:n]
    rng = np.random.default_rng(7)
    opens, highs, lows, closes = [], [], [], []
    px = base
    for i in range(n):
        o = px
        span = o * range_pct / 100
        body = span * bias * (1 if i % 2 else -1)
        c = o + body + o * drift
        h = max(o, c) + (span - abs(body)) / 2
        lo = min(o, c) - (span - abs(body)) / 2
        opens.append(o); highs.append(h); lows.append(lo); closes.append(c)
        px = c
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": rng.integers(1e6, 2e6, n).astype(float)}, index=idx)


# ── 做T ──────────────────────────────────────────────────────────────────────
def test_components_measure_what_they_are_built_from():
    df = bars(range_pct=4.0, bias=0.2)
    assert tt.intraday_range_pct(df, 20) == pytest.approx(4.0, abs=0.3)
    assert tt.mean_reversion_bias(df, 20) == pytest.approx(0.2, abs=0.05)


def test_a_wide_ranging_oscillator_scores_well():
    df = bars(range_pct=5.0, bias=0.15)
    out = tt.score(df, turnover_pct=10.0, limit_event=False)

    assert out["verdict"] in ("strong", "ok")
    assert out["score"] >= tt.OK_AT
    # Range and turnover both past target, so both components saturate at 1.
    assert out["parts"]["range"] == 1.0
    assert out["parts"]["turnover"] == 1.0


def test_a_one_way_trender_scores_worse_than_an_oscillator():
    """Same range, but the close lands at the extreme instead of mid-range."""
    osc = tt.score(bars(range_pct=4.0, bias=0.1), turnover_pct=10.0, limit_event=False)
    trend = tt.score(bars(range_pct=4.0, bias=0.9), turnover_pct=10.0, limit_event=False)
    assert osc["score"] > trend["score"]


@pytest.mark.parametrize("kwargs, expect", [
    (dict(turnover_pct=10.0, limit_event=True), "近5日"),
    (dict(turnover_pct=1.5, limit_event=False), "流动性"),
])
def test_hard_gates_reject_whatever_else_is_good(kwargs, expect):
    """A perfect oscillator is still unusable if it is locked or illiquid."""
    out = tt.score(bars(range_pct=6.0, bias=0.1), **kwargs)
    assert out["verdict"] == "skip"
    assert out["score"] == 0.0
    assert expect in out["why"]


def test_a_motionless_stock_is_gated_on_range():
    out = tt.score(bars(range_pct=0.5, bias=0.2), turnover_pct=10.0, limit_event=False)
    assert out["verdict"] == "skip"
    assert "日内波动" in out["why"]


def test_the_limit_gate_outranks_the_others():
    """Locked at the limit is the reason, even when it is also illiquid."""
    out = tt.score(bars(range_pct=6.0, bias=0.1), turnover_pct=0.5, limit_event=True)
    assert "近5日" in out["why"]


def test_adx_band_is_flat_inside_and_decays_outside():
    p = tt.DEFAULTS
    assert tt.adx_band_score(15, p.adx_lo, p.adx_hi) == 1.0
    assert tt.adx_band_score(25, p.adx_lo, p.adx_hi) == 1.0
    assert tt.adx_band_score(35, p.adx_lo, p.adx_hi) == 1.0
    # 0.1 per point outside the band, floored at zero.
    assert tt.adx_band_score(10, p.adx_lo, p.adx_hi) == pytest.approx(0.5)
    assert tt.adx_band_score(45, p.adx_lo, p.adx_hi) == pytest.approx(0.0)
    assert tt.adx_band_score(60, p.adx_lo, p.adx_hi) == 0.0


def test_a_missing_component_renormalises_rather_than_scoring_zero():
    """No turnover reading must not be treated as zero turnover."""
    have = tt.score(bars(), turnover_pct=8.0, limit_event=False)
    missing = tt.score(bars(), turnover_pct=None, limit_event=False)
    assert missing["score"] is not None
    # Dropping a component the stock was MAXING should not raise its score.
    assert missing["score"] <= have["score"] + 1e-9
    # And it must be far above what scoring it as zero would give.
    assert missing["score"] > have["score"] * (1 - tt.WEIGHTS["turnover"]) + 1


def test_weights_sum_to_one_and_cuts_match_the_page():
    assert sum(tt.WEIGHTS.values()) == pytest.approx(1.0)
    assert (tt.STRONG_AT, tt.OK_AT) == (75.0, 55.0)
    assert (tt.MIN_TURNOVER_PCT, tt.MIN_RANGE_PCT) == (2.0, 1.0)


def test_too_little_history_is_no_data_not_a_low_score():
    out = tt.score(bars(n=30), turnover_pct=10.0, limit_event=False)
    assert out["verdict"] == "no_data"
    assert out["score"] is None


# ── 反转 ─────────────────────────────────────────────────────────────────────
def _panic(down_days=5, crash=-0.09, fading_volume=True):
    """
    A quiet series, then a slide, then one violent capitulation day.

    Not a run of identical crash days: the z-score measures the latest bar
    against the PRIOR twenty, so four −9% days in that window inflate sigma
    and flatten the fifth to about −2. The rule is built to fire on the day
    the selling breaks, and this is what that shape looks like.
    """
    rng = np.random.default_rng(3)
    rets = list(rng.normal(0.0002, 0.006, N - down_days))
    rets += [-0.02] * (down_days - 1) + [crash]
    close = pd.Series(100 * np.cumprod(1 + np.array(rets)), index=DATES)

    vol = list(rng.integers(1_000_000, 1_200_000, N - down_days).astype(float))
    tail = ([700_000.0 - 50_000 * i for i in range(down_days)] if fading_volume
            else [1_500_000.0 + 200_000 * i for i in range(down_days)])
    return close, pd.Series(vol + tail, index=DATES)


def test_a_textbook_panic_passes_every_rule():
    close, vol = _panic()
    out = mr.evaluate(close, vol, name="某某股份", vs_sector_pp=-8.0)

    assert out["verdict"] == "strong"
    assert out["passed"] == 5
    assert all(out["rules"].values())
    assert out["z"] <= mr.DEFAULTS.z_max
    assert out["rsi"] < mr.DEFAULTS.rsi_max
    assert out["down_days"] >= mr.DEFAULTS.down_days_min


def test_falling_on_rising_volume_is_not_exhaustion():
    """Distribution, not capitulation — the volume rule must fail."""
    close, vol = _panic(fading_volume=False)
    out = mr.evaluate(close, vol, name="某某股份", vs_sector_pp=-8.0)

    assert out["rules"]["vol"] is False
    assert out["verdict"] != "strong"


def test_a_stock_falling_with_its_whole_sector_is_not_an_isolated_panic():
    close, vol = _panic()
    out = mr.evaluate(close, vol, name="某某股份", vs_sector_pp=-0.5)

    assert out["rules"]["sector"] is False
    assert out["verdict"] != "strong"


def test_an_untracked_sector_scores_half_credit_not_a_failure():
    """Missing data is not evidence against the stock."""
    close, vol = _panic()
    out = mr.evaluate(close, vol, name="某某股份", vs_sector_pp=None)

    assert out["rules"]["sector"] is None
    # Four hard rules pass and the fifth is unknown: still the top verdict,
    # because failing it would punish a stock for being outside a PPI index.
    assert out["verdict"] == "strong"
    assert out["passed"] == 4


def test_st_names_are_rejected_before_any_rule_runs():
    close, vol = _panic()
    for name in ("ST某某", "*ST某某", "st abc"):
        out = mr.evaluate(close, vol, name=name, vs_sector_pp=-8.0)
        assert out["verdict"] == "skip"
        assert out["rules"] == {}


def test_a_calm_stock_is_not_a_candidate():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, N)), index=DATES)
    vol = pd.Series(rng.integers(1e6, 1.1e6, N).astype(float), index=DATES)

    out = mr.evaluate(close, vol, name="某某股份", vs_sector_pp=1.0)
    assert out["verdict"] == "not_now"
    assert out["passed"] <= 1


def test_the_z_score_excludes_the_bar_it_is_judging():
    """
    Including today in its own baseline would inflate the deviation and
    understate exactly the day that matters.
    """
    rng = np.random.default_rng(4)
    calm = list(rng.normal(0, 0.004, 60))
    close = pd.Series(100 * np.cumprod(1 + np.array(calm + [-0.08])),
                      index=DATES[:61])
    z = mr.zscore_of_latest_return(close, 20)

    # A −8% day against a 0.4% daily sigma is an enormous z; had the bar been
    # in its own sample the deviation would swell and the z would shrink.
    assert z < -10


def test_thresholds_match_the_page():
    p = mr.DEFAULTS
    assert (p.z_max, p.down_days_min, p.rsi_max, p.sector_div_pp) == (-2.5, 4, 25.0, 5.0)
