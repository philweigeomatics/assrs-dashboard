"""
strategies.pair_trade on pairs built to be cointegrated, and pairs built not
to be.

Offline. The property that matters most is the walk-forward one: the hedge
ratio used on any day must not have seen that day. It is the difference
between a strategy and a description of the past, and it is invisible in the
output — a lookahead spread looks BETTER, not broken.

    python -m pytest api/tests/test_pair_trade.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from strategies import pair_trade as pt  # noqa: E402

N = 700
DATES = pd.bdate_range("2023-01-02", periods=N)


def cointegrated(beta=1.3, noise=0.01, seed=1) -> pd.DataFrame:
    """
    B is a fixed multiple of A plus a stationary wobble, so log B − beta·log A
    is mean-reverting by construction.
    """
    rng = np.random.default_rng(seed)
    a = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, N)))
    # An AR(1) wobble: mean-reverting, which is what makes the pair tradeable.
    w = np.zeros(N)
    for i in range(1, N):
        w[i] = 0.93 * w[i - 1] + rng.normal(0, noise)
    b = np.exp(np.log(a) * beta + w + 1.0)
    return pd.DataFrame({"AAA": a, "BBB": b}, index=DATES)


def independent(seed=2) -> pd.DataFrame:
    """Two unrelated random walks — no long-run relationship to find."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, N))),
        "BBB": 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, N))),
    }, index=DATES)


def _analyse(prices):
    return pt.analyse_pair("AAA", "BBB", np.log(prices))


# ── the walk-forward property ────────────────────────────────────────────────
def test_the_hedge_ratio_never_sees_the_day_it_is_used_on():
    """
    Truncating the series must not change the spread on the days that remain.

    If any part of the estimate looked ahead, cutting off the future would move
    earlier values. This is the check that the shift(1) is real.
    """
    prices = cointegrated()
    full = _analyse(prices)
    cut = _analyse(prices.iloc[:-60])

    shared = cut["dates"]
    a = pd.Series(full["spread"], index=full["dates"]).loc[shared]
    b = pd.Series(cut["spread"], index=shared)
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-9)


def test_the_warm_up_is_dropped_not_filled():
    """The first ols_window days have no ratio yet and must not be invented."""
    prices = cointegrated()
    out = pt.analyse_pair("AAA", "BBB", np.log(prices), ols_window=252)
    assert len(out["dates"]) <= N - 252
    assert not np.isnan(out["spread"]).any()


# ── telling a real pair from a fake one ──────────────────────────────────────
def test_a_built_in_relationship_is_found():
    out = _analyse(cointegrated())
    assert out["coint_ok"] is True
    assert out["adf_ok"] is True
    assert out["corr"] > 0.5
    # analyse_pair regresses A ON B, so it recovers d(logA)/d(logB) = 1/beta,
    # not the beta the fixture wrote into B. Getting this backwards is easy and
    # would look like a broken hedge ratio rather than a misread direction.
    assert out["beta_now"] == pytest.approx(1 / 1.3, abs=0.2)


def test_two_random_walks_are_not_sold_as_a_pair():
    """
    Asserted across several seeds, not one. Engle-Granger is a p-value test at
    p < 0.10, so roughly one unrelated pair in ten passes it by chance — a
    single-seed assertion here would be testing the seed, not the engine.
    """
    scores = [_analyse(independent(seed=s))["score"] for s in (2, 11, 23, 37, 41)]
    assert sum(1 for s in scores if s >= pt.GOOD_SCORE) <= 1
    assert sum(scores) / len(scores) < pt.GOOD_SCORE


def test_a_real_pair_outscores_an_unrelated_one():
    assert _analyse(cointegrated())["score"] > _analyse(independent())["score"]


# ── the individual statistics ────────────────────────────────────────────────
def test_hurst_separates_a_mean_reverting_spread_from_a_trending_one():
    """
    The ORDERING is what the score depends on, and the ordering is what this
    asserts. The absolute level is not trustworthy: this is single-scale R/S
    (log(R/S)/log n over the whole series), which reads high on short samples
    — a φ=0.5 AR(1) comes out near 0.61, well above the 0.45 gate. Ported
    as-is from the Streamlit page on purpose, since changing the estimator
    would move every historical score.
    """
    rng = np.random.default_rng(5)
    walk = np.cumsum(rng.normal(0, 1, 500))
    reverting = np.zeros(500)
    for i in range(1, 500):
        reverting[i] = 0.5 * reverting[i - 1] + rng.normal(0, 1)

    assert pt.hurst_rs(reverting) < pt.hurst_rs(walk)


def test_half_life_recovers_the_decay_it_was_given():
    """An AR(1) with φ=0.9 has a half-life of ln2 / −ln(0.9) ≈ 6.6 days."""
    rng = np.random.default_rng(6)
    s = np.zeros(4000)
    for i in range(1, 4000):
        s[i] = 0.9 * s[i - 1] + rng.normal(0, 1)
    assert pt.half_life(s) == pytest.approx(6.58, abs=1.0)


def test_an_explosive_oscillation_is_rejected_outright():
    """
    Alternating sign every step is an AR(1) outside the unit circle. It fits a
    steeply negative slope and would otherwise produce a SHORT, attractive
    half-life — the guard exists precisely because the number looks good.
    """
    assert pt.half_life(np.array([(-1.0) ** i * 5 for i in range(500)])) == 999.0


def test_a_random_walk_never_looks_tradeable():
    """
    It may come back finite — a walk's slope is slightly negative by chance —
    but never inside the 5-30 day band the score rewards, which is the
    property that matters.
    """
    for seed in (7, 13, 29):
        hl = pt.half_life(np.cumsum(np.random.default_rng(seed).normal(0, 1, 500)))
        assert not (5 <= hl <= 30), f"seed {seed} gave a tradeable-looking {hl}"


# ── the signal, which is buy-only ────────────────────────────────────────────
@pytest.mark.parametrize("z, signal, buys", [
    (-2.5, "BUY_A", "AAA"),
    (2.5, "BUY_B", "BBB"),
    (-1.7, "WATCH", "AAA"),
    (0.3, "NEUTRAL", "AAA"),
])
def test_the_signal_always_names_a_leg_to_buy(z, signal, buys):
    """A-shares cannot be shorted, so no reading may require a short."""
    out = {"z_now": z, "code_a": "AAA", "code_b": "BBB"}
    sig, buy, reduce_ = pt.signal_for_pair(out)
    assert (sig, buy) == (signal, buys)
    assert reduce_ != buy


def test_a_stretched_spread_buys_the_cheap_leg():
    """Negative z means A is cheap against B — so A is the one to buy."""
    assert pt.signal_for_pair({"z_now": -3.0, "code_a": "X", "code_b": "Y"})[1] == "X"
    assert pt.signal_for_pair({"z_now": 3.0, "code_a": "X", "code_b": "Y"})[1] == "Y"


# ── trades ───────────────────────────────────────────────────────────────────
def test_trades_open_at_the_threshold_and_close_at_zero():
    prices = cointegrated()
    out = _analyse(prices)
    trades = pt.detect_trades(out["z_series"], out["dates"], prices, "AAA", "BBB")

    assert trades, "a mean-reverting spread should cross ±2 at some point"
    for t in trades:
        assert abs(t["entry_z"]) >= pt.ENTRY_Z - 1e-9
        assert t["buy_code"] == ("AAA" if t["direction"] == "BUY_A" else "BBB")
        assert t["entry"] <= t["exit"]
        if not t["open"]:
            # Closed trades exit on the far side of zero from where they opened.
            assert (t["exit_z"] >= 0) if t["direction"] == "BUY_A" else (t["exit_z"] <= 0)

    # At most one trade may still be open, and only the last one.
    assert sum(1 for t in trades if t["open"]) <= 1
    if trades[-1]["open"]:
        assert all(not t["open"] for t in trades[:-1])


def test_pnl_is_measured_on_the_bought_leg():
    prices = cointegrated()
    out = _analyse(prices)
    trades = [t for t in pt.detect_trades(out["z_series"], out["dates"], prices, "AAA", "BBB")
              if not t["open"]]
    assert trades
    t = trades[0]
    expected = (t["exit_price"] / t["entry_price"] - 1) * 100
    assert t["pnl_pct"] == pytest.approx(expected, abs=0.02)


# ── ranking a set ────────────────────────────────────────────────────────────
def test_every_unique_pair_is_tested_once():
    prices = cointegrated()
    prices["CCC"] = prices["AAA"] * 0.7
    prices["DDD"] = independent(seed=9)["BBB"].values

    results, skipped = pt.rank_pairs(prices)
    pairs = {tuple(sorted((r["code_a"], r["code_b"]))) for r in results}
    # 4 codes → 6 unordered pairs, none repeated and none reversed.
    assert len(results) + len(skipped) == 6
    assert len(pairs) == len(results)


def test_results_come_back_best_first():
    prices = cointegrated()
    prices["DDD"] = independent(seed=9)["BBB"].values
    results, _ = pt.rank_pairs(prices)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_a_pair_that_cannot_be_assessed_is_reported_not_dropped():
    """Too little history is an answer — silently omitting the pair is not."""
    prices = cointegrated().iloc[:300]
    results, skipped = pt.rank_pairs(prices, ols_window=252)
    assert len(results) + len(skipped) == 1
    if skipped:
        assert "不足" in skipped[0]["why"]


def test_refuses_sets_it_cannot_handle():
    prices = cointegrated()
    with pytest.raises(LookupError):
        pt.rank_pairs(prices[["AAA"]])
    wide = prices.copy()
    for i in range(pt.MAX_LEGS):
        wide[f"X{i}"] = prices["AAA"] * (1 + i)
    with pytest.raises(LookupError):
        pt.rank_pairs(wide)
