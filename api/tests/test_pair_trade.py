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
    (1.7, "WATCH", "BBB"),          # a watch that leans the other way
    (0.3, "NEUTRAL", "AAA"),
])
def test_the_signal_always_names_a_leg_to_buy(z, signal, buys):
    """A-shares cannot be shorted, so no reading may require a short."""
    out = {"z_now": z, "code_a": "AAA", "code_b": "BBB"}
    sig, buy, reduce_ = pt.signal_for_pair(out)
    assert (sig, buy) == (signal, buys)
    assert reduce_ != buy


def test_a_watch_names_the_leg_it_is_leaning_towards():
    """
    Which way it is leaning is the entire content of a watch. Returning the
    pair in declaration order tells the reader to go and work out the sign of
    z for themselves.
    """
    assert pt.signal_for_pair({"z_now": -1.7, "code_a": "X", "code_b": "Y"})[1] == "X"
    assert pt.signal_for_pair({"z_now": 1.7, "code_a": "X", "code_b": "Y"})[1] == "Y"


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


# ── what the two legs were doing ─────────────────────────────────────────────
def test_both_legs_come_back_on_the_spread_s_own_dates():
    """
    The chart draws the legs under the z-score against a shared x-axis. One
    row of misalignment and a trade marker sits over the wrong day.
    """
    prices = cointegrated()
    out = _analyse(prices)

    assert len(out["px_a"]) == len(out["px_b"]) == len(out["dates"])
    # The same closes, not a second read of them.
    assert np.allclose(out["px_a"], prices["AAA"].loc[out["dates"]].values)
    assert np.allclose(out["px_b"], prices["BBB"].loc[out["dates"]].values)


def test_a_trade_says_what_each_leg_did_not_only_the_one_we_held():
    prices = cointegrated()
    out = _analyse(prices)
    trades = [t for t in pt.detect_trades(out["z_series"], out["dates"], prices,
                                          "AAA", "BBB") if not t["open"]]
    assert trades

    for t in trades:
        lo = prices.index.get_indexer([pd.Timestamp(t["entry"])], method="nearest")[0]
        hi = prices.index.get_indexer([pd.Timestamp(t["exit"])], method="nearest")[0]
        for code, field in (("AAA", "a_ret_pct"), ("BBB", "b_ret_pct")):
            expected = (prices[code].iloc[hi] / prices[code].iloc[lo] - 1) * 100
            assert t[field] == pytest.approx(expected, abs=0.02)
        assert t["pattern"] in pt.PATTERNS


def test_the_bought_leg_s_return_is_the_pnl_not_a_second_opinion():
    """
    Two lookups for the same number can disagree — on a suspension, a
    nearest-date match, any reindex. Then the table says +4% and the
    explanation beside it says +3.7% for the same leg.
    """
    prices = cointegrated()
    out = _analyse(prices)
    for t in pt.detect_trades(out["z_series"], out["dates"], prices, "AAA", "BBB"):
        mine = t["a_ret_pct"] if t["buy_code"] == "AAA" else t["b_ret_pct"]
        assert t["pnl_pct"] == mine


@pytest.mark.parametrize("a, b, pattern", [
    (8.0, 3.0, "BOTH_UP"),          # both rose — the bought leg simply rose more
    (-2.0, -9.0, "BOTH_DOWN"),      # both fell — it converged by falling less
    (6.0, -4.0, "A_UP_B_DOWN"),     # the textbook picture, and the rarest
    (-4.0, 6.0, "B_UP_A_DOWN"),
    (0.2, -0.1, "FLAT"),            # neither leg moved; the z did
    (9.0, 0.05, "BOTH_UP"),         # B is flat, but nothing fell
    (0.05, -9.0, "BOTH_DOWN"),      # and the same on the way down
    (-9.0, 0.05, "BOTH_DOWN"),
])
def test_the_same_convergence_is_four_different_things(a, b, pattern):
    """
    Every one of these is a spread closing back to zero, and the z-chart
    draws them identically. They are not the same trade to have held.
    """
    assert pt.leg_pattern(a, b) == pattern


def test_a_leg_that_barely_moved_is_not_called_a_rise():
    """
    Without a dead zone, "A rose 9%, B went nowhere" reads as "both rose",
    which is the exact confusion this is here to remove.
    """
    assert pt.leg_pattern(9.0, 0.05) == "BOTH_UP"
    assert pt.leg_pattern(9.0, 0.05, flat=0.0) == "BOTH_UP"
    assert pt.leg_pattern(0.05, 0.02) == "FLAT"
    # With no dead zone at all, the same numbers become a divergence.
    assert pt.leg_pattern(0.05, -0.02, flat=0.0) == "A_UP_B_DOWN"


def test_an_unpriceable_leg_is_unknown_rather_than_flat():
    """nan is "we could not read it", which is not the claim "it did not move"."""
    assert pt.leg_pattern(float("nan"), 3.0) is None
    assert pt.leg_pattern(3.0, None) is None


def test_an_unpriceable_trade_still_produces_a_row():
    """A gap in one leg must not take the whole trade list down with it."""
    prices = cointegrated()
    broken = prices.copy()
    broken["BBB"] = np.nan
    t = pt.make_trade(prices.index[300], prices.index[310], -2.1, 0.1,
                      "BUY_A", False, broken, "AAA", "BBB")

    assert t["a_ret_pct"] is not None and t["b_ret_pct"] is None
    assert t["pattern"] is None
    assert t["pnl_pct"] is not None      # we held AAA, and AAA is readable


# ── the thresholds the screen explains ───────────────────────────────────────
def test_the_published_gates_are_the_ones_actually_applied():
    """
    The help text on the page is generated from GATES. If GATES said 0.10
    while the code compared against 0.05, the screen would confidently
    explain a rule that is not the rule.
    """
    prices = cointegrated()
    out = _analyse(prices)
    g = pt.GATES

    assert out["coint_ok"] == (out["eg_p"] < g["coint_p"])
    assert out["adf_ok"] == (out["adf_p"] < g["adf_p"])
    assert out["hurst_ok"] == (out["hurst"] < g["hurst_max"])
    assert out["hl_ok"] == (g["hl_min"] <= out["half_life"] <= g["hl_max"])


def test_every_gate_a_row_displays_has_a_published_threshold():
    """A column with a pass/fail colour the reader cannot look up is worse
    than no colour."""
    out = _analyse(cointegrated())
    shown = {k for k in out if k.endswith("_ok")}
    assert shown == {"coint_ok", "adf_ok", "hurst_ok", "hl_ok"}
    for need in ("coint_p", "adf_p", "hurst_max", "hl_min", "hl_max",
                 "entry_z", "watch_z", "good_score", "z_window", "ols_window"):
        assert need in pt.GATES, need


@pytest.mark.parametrize("attr, value, field, expect", [
    ("COINT_P", 0.0, "coint_ok", False), ("COINT_P", 1.0, "coint_ok", True),
    ("ADF_P", 0.0, "adf_ok", False), ("ADF_P", 1.0, "adf_ok", True),
    ("HURST_MAX", 0.0, "hurst_ok", False), ("HURST_MAX", 9.0, "hurst_ok", True),
    ("HL_MIN", 999.0, "hl_ok", False),
])
def test_a_gate_moving_moves_the_verdict_with_it(attr, value, field, expect):
    """
    Binding each constant into the comparison is the whole point: move the
    threshold and the flag has to follow. Set to an impossible value rather
    than a plausible one, so the test does not depend on where any fixture's
    p value happens to land.
    """
    prices = cointegrated()
    keep = getattr(pt, attr)
    try:
        setattr(pt, attr, value)
        assert _analyse(prices)[field] is expect
    finally:
        setattr(pt, attr, keep)


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
