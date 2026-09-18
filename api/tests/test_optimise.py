"""
optimise: allocations, and whether changing to them is worth anything.

Offline, on series whose right answer is known by construction — one asset
with half the volatility of another must get more than half the book under
minimum variance, risk parity must equalise risk contributions, and every
method must respect the cap and stay long-only.

The last group of tests is about the part that is easy to get wrong in a way
nobody notices: the walk-forward must FIT on the training half only. An
optimiser that has seen the test set always wins, and a page that reports that
win as evidence is worse than a page with no evidence at all.

    python -m pytest api/tests/test_optimise.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import optimise as opt  # noqa: E402


def rets(n=600, seed=0, vols=(0.008, 0.016, 0.024), drifts=None) -> pd.DataFrame:
    """Independent assets with known, different volatilities."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    drifts = drifts or [0.0003] * len(vols)
    return pd.DataFrame(
        {f"A{i}": rng.normal(d, v, n) for i, (v, d) in enumerate(zip(vols, drifts))},
        index=idx)


def correlated(n=600, seed=1) -> pd.DataFrame:
    """
    Two assets that move together, one twice as volatile.

    With a correlation near 1, the minimum-variance portfolio wants to SHORT
    the noisy one and hold more than 100% of the quiet one — which is why this
    is the fixture that proves the long-only constraint is doing something.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.010, n)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"QUIET": common + rng.normal(0, 0.0006, n),
                         "LOUD": 2 * common + rng.normal(0, 0.0012, n)}, index=idx)


def w(frame, method, **kw):
    return opt.weights(frame, method, **kw)


# ── the shape of any answer ──────────────────────────────────────────────────
@pytest.mark.parametrize("method", sorted(opt.METHODS))
def test_every_method_is_long_only_and_fully_invested(method):
    got = w(rets(), method)
    assert got.sum() == pytest.approx(1.0, abs=1e-6)
    assert (got >= -1e-9).all()


@pytest.mark.parametrize("method", sorted(opt.METHODS))
def test_correlated_assets_do_not_produce_a_short(method):
    got = w(correlated(), method, cap=1.0)
    assert (got >= -1e-9).all()
    assert got.sum() == pytest.approx(1.0, abs=1e-6)


def test_the_solver_refuses_a_short_even_when_the_objective_asks_for_one():
    """
    Tested against the solver directly, with an objective whose minimum IS a
    short position. Real objectives almost never ask for one here — the
    shrunk covariance pulls correlations away from the extremes that make
    shorting attractive — so an artificial objective is the only way to show
    the constraint is actually enforced rather than merely never triggered.
    """
    got = opt._solve(lambda x: float(x[1]), 4, cap=0.5)    # minimised at x[1] = -1

    assert got[1] >= -1e-9
    assert got.max() <= 0.5 + 1e-9
    assert got.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("method", sorted(opt.METHODS))
def test_every_method_respects_the_cap(method):
    """
    An "optimal" portfolio that is 90% one name is a statement about the
    sample, not advice. Checked against the EFFECTIVE cap, which is the one
    the payload reports — see effective_cap for why it is not always the one
    asked for.
    """
    frame = rets(vols=(0.004, 0.02, 0.03, 0.04, 0.05))
    cap = opt.effective_cap(0.3, frame.shape[1])
    assert w(frame, method, cap=0.3).max() <= cap + 1e-6


@pytest.mark.parametrize("method", sorted(opt.METHODS))
def test_a_generous_cap_still_binds_on_a_wide_book(method):
    """With twelve names, 30% is above 2/n and applies as asked."""
    frame = rets(vols=(0.003,) + (0.04,) * 11)
    assert opt.effective_cap(0.3, 12) == pytest.approx(0.3)
    assert w(frame, method, cap=0.3).max() <= 0.3 + 1e-6


def test_a_cap_below_equal_weight_is_raised_rather_than_failing_silently():
    """10% each across five assets cannot sum to 1; the solve would just fail."""
    got = w(rets(vols=(0.01,) * 5), "min_var", cap=0.10)
    assert got.sum() == pytest.approx(1.0, abs=1e-6)


def test_a_cap_at_equal_weight_would_leave_only_one_feasible_portfolio():
    """
    25% across four names admits exactly one answer, so every method returns
    equal weighting and the optimiser looks broken rather than constrained.
    The floor is twice equal weight, which always leaves room to differ.
    """
    assert opt.effective_cap(0.25, 4) == pytest.approx(0.5)
    assert opt.effective_cap(0.25, 20) == pytest.approx(0.25)

    frame = rets(vols=(0.005, 0.030, 0.030, 0.030))
    assert w(frame, "min_var", cap=0.25).max() > 0.25 + 1e-6


def test_trivial_weights_are_zeroed_not_left_as_dust():
    got = w(rets(vols=(0.004, 0.05, 0.06, 0.07)), "min_var", cap=0.9)
    assert all(v == 0 or v >= opt.MIN_WEIGHT for v in got)


def test_a_single_asset_gets_everything():
    assert w(rets(vols=(0.01,)), "min_var").to_dict() == {"A0": 1.0}


def test_an_unknown_method_is_refused():
    with pytest.raises(LookupError):
        w(rets(), "magic")


# ── what each method is for ──────────────────────────────────────────────────
def test_minimum_variance_leans_on_the_quiet_asset():
    got = w(rets(vols=(0.006, 0.030)), "min_var", cap=1.0)
    assert got["A0"] > got["A1"]
    # And it beats equal weighting at the thing it optimises.
    frame = rets(vols=(0.006, 0.030))
    assert (opt.stats(frame, got)["ann_vol_pct"]
            < opt.stats(frame, w(frame, "equal"))["ann_vol_pct"])


def test_risk_parity_equalises_risk_rather_than_money():
    frame = rets(vols=(0.006, 0.012, 0.024))
    got = w(frame, "risk_parity", cap=1.0)
    contrib = opt._risk_contrib(frame, got)

    assert contrib.max() - contrib.min() < 0.05
    # Which for independent assets means weights inverse to volatility.
    assert got["A0"] > got["A1"] > got["A2"]


def test_equal_weight_is_exactly_that():
    got = w(rets(vols=(0.01, 0.02, 0.03, 0.04)), "equal")
    assert set(got.round(6)) == {0.25}


def test_max_sharpe_chases_the_asset_that_did_best():
    """
    Which is precisely why it is flagged: that is a fact about the sample.
    """
    frame = rets(vols=(0.012, 0.012), drifts=(0.0002, 0.0012))
    got = w(frame, "max_sharpe", cap=1.0)
    assert got["A1"] > got["A0"]
    assert opt.METHODS["max_sharpe"]["needs_returns"] is True


def test_only_max_sharpe_is_flagged_as_needing_return_forecasts():
    needs = {k for k, v in opt.METHODS.items() if v["needs_returns"]}
    assert needs == {"max_sharpe"}


# ── the suggestion ───────────────────────────────────────────────────────────
def current(frame, weights):
    return pd.Series(weights, index=frame.columns, dtype=float)


def test_a_suggestion_says_what_to_change_not_just_what_to_hold():
    frame = rets(vols=(0.006, 0.030))
    out = opt.suggest(frame, current(frame, [0.5, 0.5]), "min_var", cap=1.0)
    rows = {r["symbol"]: r for r in out["rows"]}

    assert rows["A0"]["current_pct"] == pytest.approx(50, abs=0.01)
    assert rows["A0"]["delta_pct"] == pytest.approx(
        rows["A0"]["target_pct"] - 50, abs=0.01)
    assert rows["A0"]["delta_pct"] > 0 and rows["A1"]["delta_pct"] < 0


def test_turnover_is_the_share_of_the_book_that_would_move():
    frame = rets(vols=(0.01, 0.01))
    out = opt.suggest(frame, current(frame, [1.0, 0.0]), "equal")
    # 100% → 50/50 means moving half the book.
    assert out["turnover_pct"] == pytest.approx(50, abs=0.01)


def test_the_suggestion_carries_its_own_caveat():
    frame = rets()
    assert "样本外" in opt.suggest(frame, current(frame, [1, 0, 0]), "min_var")["basis"]
    assert opt.suggest(frame, current(frame, [1, 0, 0]), "max_sharpe")["overfit_risk"]
    assert not opt.suggest(frame, current(frame, [1, 0, 0]), "min_var")["overfit_risk"]


def test_one_asset_cannot_be_optimised():
    frame = rets(vols=(0.01,))
    with pytest.raises(LookupError, match="两只"):
        opt.suggest(frame, current(frame, [1.0]), "min_var")


# ── the evidence ─────────────────────────────────────────────────────────────
def test_the_walk_forward_fits_only_on_the_training_half():
    """
    The whole point, and the easiest thing to get silently wrong: fitting on
    the full series and then "measuring" on part of it. Every method wins, and
    the table becomes worse than no table.

    Pinned by reproducing the arithmetic: the reported out-of-sample figure
    must equal what TRAIN-fitted weights score on the test half, and must NOT
    equal what all-data-fitted weights score there.
    """
    frame = rets(n=600, vols=(0.006, 0.030))
    frame.iloc[300:, 0] *= 5.0          # the regime flips at the split
    frame.iloc[300:, 1] /= 5.0

    wf = opt.walk_forward(frame, current(frame, [0.5, 0.5]), ["min_var"])
    reported = wf["rows"][0]["out_of_sample"]["ann_vol_pct"]

    cut = len(frame) // 2
    honest = opt.stats(frame.iloc[cut:], opt.weights(frame.iloc[:cut], "min_var"))
    cheating = opt.stats(frame.iloc[cut:], opt.weights(frame, "min_var"))

    assert reported == pytest.approx(honest["ann_vol_pct"], abs=0.01)
    assert reported != pytest.approx(cheating["ann_vol_pct"], abs=0.01)
    assert wf["rows"][0]["in_sample"]["ann_vol_pct"] < reported


def test_the_walk_forward_measures_the_current_book_on_the_same_data():
    """Without a baseline, "optimised vol 12%" is not a comparison."""
    frame = rets(n=600)
    wf = opt.walk_forward(frame, current(frame, [0.7, 0.2, 0.1]), ["min_var", "equal"])
    labels = {r["method"] for r in wf["rows"]}

    assert "current" in labels and {"min_var", "equal"} <= labels
    for row in wf["rows"]:
        assert row["out_of_sample"]["ann_vol_pct"] > 0


def test_the_two_halves_do_not_overlap():
    frame = rets(n=600)
    wf = opt.walk_forward(frame, current(frame, [1, 0, 0]), ["equal"])
    assert wf["train"]["to"] < wf["test"]["from"]
    assert wf["train"]["sessions"] + wf["test"]["sessions"] == 600


def test_too_little_history_for_a_split_is_refused():
    with pytest.raises(LookupError, match="样本外"):
        opt.walk_forward(rets(n=100), current(rets(n=100), [1, 0, 0]), ["equal"])


def test_minimum_variance_actually_reduces_volatility_out_of_sample():
    """
    The claim the panel makes. It has to hold on data the fit never saw, on a
    series whose covariance is stable — if it fails here, the plumbing is wrong
    rather than the world being uncooperative.
    """
    frame = rets(n=800, vols=(0.005, 0.010, 0.030), seed=7)
    wf = opt.walk_forward(frame, current(frame, [1 / 3, 1 / 3, 1 / 3]),
                          ["min_var", "equal"], cap=0.8)
    got = {r["method"]: r["out_of_sample"]["ann_vol_pct"] for r in wf["rows"]}
    assert got["min_var"] < got["equal"]
