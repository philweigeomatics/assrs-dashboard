"""
pair_compare on series whose answers are known by construction.

Offline. Every stock here is built FROM the benchmark with a stated beta and a
stated daily alpha, so beta, alpha, R², capture ratios and the attribution
split all have arithmetic right answers to check against — not "looks about
right". A statistics panel that is quietly wrong is worse than no panel, since
its whole purpose is to be trusted over eyeballing two lines.

    python -m pytest api/tests/test_pair_compare.py -q
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pair_compare as pc  # noqa: E402

N = 400
DATES = pd.bdate_range("2024-01-02", periods=N)


def _prices(logrets) -> pd.Series:
    """
    Price path from LOG returns — the space the model regresses in.

    Starts AT 100 rather than at the first return, so the series' own log
    returns are exactly the ones passed in. Without the leading 0 the first
    day is silently swallowed, and a deliberately flat market ends up with a
    small drift that has no business being there.
    """
    lr = np.asarray(logrets, dtype=float)[:len(DATES) - 1]
    return pd.Series(100 * np.exp(np.concatenate([[0.0], np.cumsum(lr)])), index=DATES)


@pytest.fixture()
def market():
    """The benchmark's daily LOG returns."""
    rng = np.random.default_rng(3)
    return rng.normal(0.0004, 0.011, N)


def _levered(market, beta, daily_alpha=0.0):
    """
    A stock that is EXACTLY beta x market + alpha in LOG space, with no
    idiosyncratic noise — so beta and alpha have exact right answers.

    Building it in simple-return space instead would embed volatility drag:
    a constant 1.8x of simple returns compounds to LESS than 1.8x of log
    returns, which the model would correctly report as negative alpha. Real,
    but not what these tests are trying to pin down.
    """
    return _prices(beta * np.asarray(market) + daily_alpha)


def test_beta_and_alpha_recover_what_was_built_in(market):
    a = _levered(market, 1.5, 0.0004)
    b = _levered(market, 0.8, 0.0)
    out = pc.compare(a, b, bench_close=_prices(market), window="all")

    assert out["a"]["beta"] == pytest.approx(1.5, abs=0.01)
    assert out["b"]["beta"] == pytest.approx(0.8, abs=0.01)
    # Alpha is reported as what a daily log alpha COMPOUNDS to over a year.
    assert out["a"]["alpha_annual_pct"] == pytest.approx(
        (math.exp(0.0004 * 252) - 1) * 100, abs=0.5)
    assert out["b"]["alpha_annual_pct"] == pytest.approx(0.0, abs=0.5)
    # No idiosyncratic noise: the market explains everything.
    assert out["a"]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert out["b"]["r2"] == pytest.approx(1.0, abs=1e-6)


def test_capture_ratios_equal_beta_when_there_is_no_noise(market):
    """A pure 1.4x stock takes 140% of up moves AND 140% of down moves."""
    a = _levered(market, 1.4)
    out = pc.compare(a, _levered(market, 1.0), bench_close=_prices(market), window="all")

    # Capture runs on SIMPLE returns while the stock is levered in log space,
    # so convexity lifts both a little above 140 — a real effect, not slack.
    assert out["a"]["up_capture_pct"] == pytest.approx(140, abs=2.0)
    assert out["a"]["down_capture_pct"] == pytest.approx(140, abs=2.0)


def test_volatility_scales_with_beta(market):
    a = _levered(market, 2.0)
    b = _levered(market, 1.0)
    out = pc.compare(a, b, bench_close=_prices(market), window="all")
    assert out["a"]["vol_annual_pct"] == pytest.approx(2 * out["b"]["vol_annual_pct"], rel=0.03)


def test_pair_stats_describe_the_relationship_between_the_two(market):
    a = _levered(market, 1.5, 0.0004)
    b = _levered(market, 0.8)
    out = pc.compare(a, b, bench_close=_prices(market), window="all")
    pair = out["pair"]

    # Both are pure market plays, so they are perfectly correlated with
    # each other, and A moves 1.5/0.8 for each 1% of B.
    assert pair["correlation"] == pytest.approx(1.0, abs=1e-3)
    assert pair["beta_a_on_b"] == pytest.approx(1.5 / 0.8, abs=0.02)
    # The gap is computed before rounding; the two totals are each rounded to
    # 2dp first, so subtracting the DISPLAYED numbers can differ by 0.01.
    assert pair["return_gap_pct"] == pytest.approx(
        out["a"]["total_return_pct"] - out["b"]["total_return_pct"], abs=0.02)
    assert pair["tracking_error_pct"] > 0
    assert len(pair["ratio"]) == len(pair["dates"]) == out["bars"]
    assert pair["ratio"][0] == pytest.approx(100.0, abs=1e-6)


def test_attribution_splits_the_gap_into_beta_and_alpha(market):
    """
    The whole point of the panel: A beat B — how much was taking more market
    risk, and how much was A itself?
    """
    a = _levered(market, 1.6, 0.0005)   # more beta AND real alpha
    b = _levered(market, 0.9, 0.0)
    out = pc.compare(a, b, bench_close=_prices(market), window="all")
    att = out["attribution"]

    assert att["beta_a"] == pytest.approx(1.6, abs=0.01)
    assert att["beta_b"] == pytest.approx(0.9, abs=0.01)
    assert att["alpha_a_pct"] == pytest.approx((math.exp(0.0005 * 252) - 1) * 100, abs=0.5)
    assert att["alpha_b_pct"] == pytest.approx(0.0, abs=0.5)
    assert att["alpha_factor_pct"] > 0


def test_the_two_factors_multiply_back_to_the_gap_exactly(market):
    """
    The claim the panel rests on: ratio = beta_factor x alpha_factor, with
    nothing left over. If this drifts, the attribution is decoration.
    """
    for beta_a, beta_b, alpha in ((1.6, 0.9, 0.0005), (0.7, 1.4, -0.0003), (2.2, 1.0, 0.0)):
        out = pc.compare(_levered(market, beta_a, alpha), _levered(market, beta_b),
                         bench_close=_prices(market), window="all")
        att = out["attribution"]
        product = (1 + att["beta_factor_pct"] / 100) * (1 + att["alpha_factor_pct"] / 100)
        assert product == pytest.approx(att["gap_ratio"], rel=1e-3)
        assert abs(att["residual_pct"]) < 0.01, "residual must be zero by construction"


def test_a_pure_beta_difference_is_all_beta(market):
    """Same alpha, different leverage: nothing may be attributed to alpha."""
    out = pc.compare(_levered(market, 1.8), _levered(market, 0.9),
                     bench_close=_prices(market), window="all")
    att = out["attribution"]
    assert att["alpha_factor_pct"] == pytest.approx(0.0, abs=0.5)
    assert att["beta_factor_pct"] == pytest.approx((att["gap_ratio"] - 1) * 100, rel=1e-3)


def test_a_flat_market_attributes_everything_to_alpha(market):
    """
    With the market going nowhere, beta cannot explain a gap however large the
    difference in beta is — which is exactly the 长飞/中天 case.
    """
    rng = np.random.default_rng(21)
    # Moves every day, but ends exactly where it started: beta has something
    # to regress on, and no cumulative market move to hand anyone.
    flat = rng.normal(0, 0.011, N)
    flat[:N - 1] -= flat[:N - 1].mean()     # the bars _prices actually uses
    a = _levered(flat, 1.8, 0.004)
    b = _levered(flat, 0.6, 0.001)
    att = pc.compare(a, b, bench_close=_prices(flat), window="all")["attribution"]

    assert att["market_return_pct"] == pytest.approx(0.0, abs=0.01)
    assert att["beta_factor_pct"] == pytest.approx(0.0, abs=0.01)
    assert att["alpha_factor_pct"] == pytest.approx((att["gap_ratio"] - 1) * 100, rel=1e-3)


def test_max_drawdown_is_the_worst_peak_to_trough(market):
    # 100 → 120 → 60 → 90: the worst fall is 120 → 60, i.e. −50%.
    path = pd.Series([100, 120, 60, 90.0], index=pd.bdate_range("2024-01-02", periods=4))
    assert pc._max_drawdown(path) == pytest.approx(-50.0)


def test_valuation_splits_the_move_into_rerating_and_earnings(market):
    """
    Price = EPS × PE, so (1+total) must equal (1+rerating)(1+earnings). This
    is the check that the decomposition is an identity and not an estimate.
    """
    close = _prices(market)
    # PE doubles across the window; whatever is left is earnings.
    pe = pd.Series(np.linspace(20, 40, N), index=DATES)
    fund = pd.DataFrame({"PE_TTM": pe, "PB": pe / 10,
                         "Total_MV_Yi": close * 10, "Turnover_Rate": 1.5}, index=DATES)

    out = pc.compare(close, _prices(market * 0.9), bench_close=_prices(market),
                     a_fund=fund, window="all")
    val = out["a"]["valuation"]

    assert val["rerating_pct"] == pytest.approx(100.0, abs=0.5)
    total = out["a"]["total_return_pct"] / 100
    identity = (1 + val["rerating_pct"] / 100) * (1 + val["earnings_pct"] / 100) - 1
    assert identity == pytest.approx(total, abs=1e-3)


def test_loss_making_companies_get_no_valuation_rather_than_a_fake_one(market):
    close = _prices(market)
    fund = pd.DataFrame({"PE_TTM": pd.Series(-15.0, index=DATES)}, index=DATES)
    out = pc.compare(close, _prices(market * 0.9), a_fund=fund, window="all")
    assert out["a"]["valuation"] is None
    assert out["b"]["valuation"] is None   # none supplied at all


def test_windows_cut_to_the_right_length(market):
    a, b = _levered(market, 1.2), _levered(market, 1.0)
    for name, bars in (("60", 60), ("120", 120), ("252", 252)):
        out = pc.compare(a, b, bench_close=_prices(market), window=name)
        assert out["bars"] == bars + 1, "N returns need N+1 closes"
        assert out["a"]["bars"] == bars
    assert pc.compare(a, b, window="all")["bars"] == N


def test_short_windows_do_not_get_an_annualised_headline(market):
    """Annualising 60 days turns noise into a confident-looking CAGR."""
    a, b = _levered(market, 1.2), _levered(market, 1.0)
    assert pc.compare(a, b, window="60")["a"]["cagr_pct"] is None
    assert pc.compare(a, b, window="252")["a"]["cagr_pct"] is not None


def test_only_overlapping_sessions_are_compared(market):
    """A stock suspended for a stretch must not score on days it did not trade."""
    a, b = _levered(market, 1.2), _levered(market, 1.0)
    b = b.drop(b.index[100:160])

    out = pc.compare(a, b, bench_close=_prices(market), window="all")
    assert out["bars"] == N - 60
    assert len(out["pair"]["ratio"]) == N - 60


def test_refuses_to_report_statistics_it_cannot_support(market):
    a, b = _levered(market, 1.2), _levered(market, 1.0)
    with pytest.raises(LookupError):
        pc.compare(a.head(20), b.head(20), window="all")
    with pytest.raises(LookupError):
        pc.compare(a, b, window="7")


def test_monthly_relative_compounds_to_the_total_gap(market):
    a, b = _levered(market, 1.5, 0.0004), _levered(market, 0.8)
    out = pc.compare(a, b, window="all")

    compounded = math.prod(1 + m["rel_pct"] / 100 for m in out["pair"]["monthly"])
    ratio_total = out["pair"]["ratio"][-1] / 100
    # Each month is rounded to 2dp for display, and ~20 of those compound, so
    # the identity holds to about 1e-3 — far tighter than the several percent
    # that adding the months instead of compounding them would cost.
    assert compounded == pytest.approx(ratio_total, rel=1e-3)


def test_every_number_is_json_safe(market):
    """NaN/inf anywhere in the payload breaks the response, not just the cell."""
    import json
    a, b = _levered(market, 1.2), _levered(market, 1.0)
    out = pc.compare(a, b, bench_close=_prices(market), window="all")
    text = json.dumps(out, allow_nan=False)   # raises if any NaN/inf survived
    assert "NaN" not in text and "Infinity" not in text


def test_price_level_does_not_affect_any_statistic(market):
    """
    Real tickers trade at wildly different prices — ¥300 against ¥30. Every
    number here describes RETURNS, so multiplying one stock's price by a
    constant must change nothing at all. A gap computed from raw price levels
    instead of from each stock's own starting point would report that ¥30
    stock as though it had lost 90%.
    """
    a = _levered(market, 1.5, 0.0004)
    b = _levered(market, 0.8)
    base = pc.compare(a, b, bench_close=_prices(market), window="all")
    scaled = pc.compare(a * 37.5, b * 0.21, bench_close=_prices(market), window="all")

    assert scaled["attribution"]["gap_ratio"] == pytest.approx(
        base["attribution"]["gap_ratio"], rel=1e-9)
    assert scaled["pair"]["return_gap_pct"] == pytest.approx(
        base["pair"]["return_gap_pct"], rel=1e-9)
    assert scaled["pair"]["ratio"][-1] == pytest.approx(base["pair"]["ratio"][-1], rel=1e-9)
    assert scaled["a"]["total_return_pct"] == pytest.approx(base["a"]["total_return_pct"])
    assert scaled["b"]["beta"] == pytest.approx(base["b"]["beta"])
