"""
pair_scan: finding pairs without manufacturing them.

Offline, on synthetic series whose truth is known by construction. The tests
that matter most are the two negative ones: a watchlist of independent random
walks must produce NO confirmed pairs however many are tested, and a pair that
is related only in the first half must not survive.

Eighty stocks make 3,160 pairs and 6,320 lead-lag tests. At a 5% threshold
that is ~316 "significant" results from pure noise. Anything here that lets
those through is worse than not having the feature.

    python -m pytest api/tests/test_pair_scan.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pair_scan as ps  # noqa: E402

N = 600


def frame(cols: dict) -> pd.DataFrame:
    return pd.DataFrame(cols, index=pd.bdate_range("2023-01-02", periods=N))


def noise(seed, n=N, vol=0.012):
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, vol, n)))


def independent(k=12) -> pd.DataFrame:
    """A watchlist of unrelated random walks. The null, in full."""
    return frame({f"S{i:02d}": noise(i) for i in range(k)})


def cointegrated(seed=0, n=N, kappa=0.08):
    """Two prices tied by a mean-reverting spread, all the way through."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.011, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - kappa) + rng.normal(0, 0.012)
    return 100 * np.exp(common + spread), 100 * np.exp(common)


# ── the null: nothing must come back ─────────────────────────────────────────
def test_a_watchlist_of_random_walks_confirms_nothing():
    """
    The test this module exists to pass. Sixty-six pairs of unrelated series;
    a raw 5% screen would call several of them relationships.
    """
    for kind in ps.KINDS:
        out = ps.scan(independent(12), kind, min_corr=0.0)
        assert out["funnel"]["survivors"] == 0, f"{kind}: {out['rows'][:2]}"


def test_the_funnel_shows_the_screen_firing_and_the_confirmation_catching_it():
    """
    In-sample hits on pure noise are expected and are not the finding — the
    gap between `screened` and `survivors` is.
    """
    out = ps.scan(independent(16), "lead-lag", min_corr=0.0)
    f = out["funnel"]

    assert f["pairs_possible"] == 16 * 15 // 2
    assert f["screened"] >= 1            # the raw screen does fire on noise…
    assert f["survivors"] == 0           # …and nothing survives the holdout


def test_a_relationship_present_only_in_the_first_half_does_not_survive():
    """
    The exact shape of an overfit discovery: real where it was found, absent
    where it was checked.
    """
    a, b = cointegrated(seed=3)
    half = N // 2
    broken = b.copy()
    broken[half:] = noise(99, N - half) * (b[half - 1] / noise(99, N - half)[0])

    out = ps.scan(frame({"A": a, "B": broken, "C": noise(7), "D": noise(8)}),
                  "pair-trade", min_corr=0.0)
    got = {(r["a"], r["b"]): r for r in out["rows"]}
    assert not any(r["survives"] for r in got.values())


# ── the signal: a real pair must come back ───────────────────────────────────
def test_a_genuinely_cointegrated_pair_is_found_among_decoys():
    # kappa 0.15: a spread that closes in about five sessions, which is what
    # "strong evidence" looks like. Weaker ties are found but do not survive
    # the correction — see the next test, which is the honest half of this.
    a, b = cointegrated(seed=11, kappa=0.15)
    px = frame({"REAL_A": a, "REAL_B": b,
                **{f"N{i}": noise(100 + i) for i in range(8)}})

    out = ps.scan(px, "pair-trade", min_corr=0.0)
    winners = {(r["a"], r["b"]) for r in out["rows"] if r["survives"]}

    assert ("REAL_A", "REAL_B") in winners
    assert out["funnel"]["survivors"] >= 1


def test_a_found_pair_carries_what_is_needed_to_trade_it():
    a, b = cointegrated(seed=11, kappa=0.15)
    out = ps.scan(frame({"A": a, "B": b, "C": noise(5)}), "pair-trade", min_corr=0.0)
    row = next(r for r in out["rows"] if r["survives"])

    assert row["beta"] is not None
    assert row["half_life"] is not None and row["half_life"] > 0
    assert row["tradeable"] is True
    assert row["p_train"] < ps.ALPHA and row["p_test"] < ps.ALPHA


def test_a_marginal_pair_is_reported_but_does_not_survive_the_correction():
    """
    kappa 0.08 clears both halves on its own (p ≈ 0.03 then 0.04) and still
    fails once the other retested pairs are accounted for. That is the
    correction doing its job, not a bug: evidence that thin, found by
    searching, is not evidence.

    It still appears in the rows — with `survives` false — because "we looked
    and this was the closest thing" is worth seeing.
    """
    a, b = cointegrated(seed=11, kappa=0.08)
    out = ps.scan(frame({"A": a, "B": b, **{f"N{i}": noise(50 + i) for i in range(6)}}),
                  "pair-trade", min_corr=0.0)
    row = next((r for r in out["rows"] if {r["a"], r["b"]} == {"A", "B"}), None)

    assert row is not None
    assert row["p_train"] < ps.ALPHA and row["p_test"] < ps.ALPHA
    assert row["q"] > row["p_test"]          # corrected upward…
    assert row["survives"] is False          # …past the bar


@pytest.mark.parametrize("half_life, tradeable", [
    (5.0, True), (29.9, True), (30.1, False), (170.0, False),
    (float("nan"), False), (-4.0, False),
])
def test_a_slow_spread_is_flagged_untradeable_rather_than_hidden(half_life, tradeable):
    """
    Cointegrated and untradeable is a real answer — a spread that takes six
    months to close is not a trade — so it is flagged, not filtered.

    Tested on the flag directly rather than through a series, because in this
    model slow mean reversion and weak cointegration evidence are the SAME
    parameter: any fixture slow enough to be untradeable is also too weak to
    be found. That is a property of the world, not of the fixture.
    """
    row = ps._row({"a": "A", "b": "B", "corr": 0.8,
                   "train": {"p": 0.01, "beta": 1.0, "half_life": half_life, "n": 300},
                   "test": {"p": 0.01, "beta": 1.0, "half_life": half_life, "n": 300}},
                  0.01, "pair-trade")
    assert row["tradeable"] is tradeable


# ── the split ────────────────────────────────────────────────────────────────
def test_the_halves_do_not_overlap_and_cover_everything():
    out = ps.scan(independent(4), "pair-trade", min_corr=0.0)
    assert out["train"]["to"] < out["test"]["from"]
    assert out["train"]["sessions"] + out["test"]["sessions"] == out["sessions"]


def test_the_shortlist_is_built_on_the_training_half_only(monkeypatch):
    """
    Letting the confirming half influence which pairs were chosen is exactly
    the leak the split exists to prevent.
    """
    seen = {}

    real = ps.shortlist
    def spy(rets, **kw):
        seen["rows"] = len(rets)
        return real(rets, **kw)
    monkeypatch.setattr(ps, "shortlist", spy)

    px = independent(4)
    out = ps.scan(px, "pair-trade", min_corr=0.0)
    # One row lost to pct_change; anything near the full length means the
    # whole history was used.
    assert seen["rows"] <= out["train"]["sessions"]


def test_too_little_history_to_split_is_refused():
    short = pd.DataFrame({"A": noise(1, 150), "B": noise(2, 150)},
                         index=pd.bdate_range("2024-01-01", periods=150))
    with pytest.raises(LookupError, match="切成两半|足够长"):
        ps.scan(short, "pair-trade", min_corr=0.0)


def test_a_stock_with_too_little_history_is_left_out_of_the_universe():
    px = independent(3)
    px["NEW"] = np.nan
    px.iloc[-40:, px.columns.get_loc("NEW")] = noise(42, 40)

    out = ps.scan(px, "pair-trade", min_corr=0.0)
    assert out["funnel"]["universe"] == 3
    assert all("NEW" not in (r["a"], r["b"]) for r in out["rows"])


# ── the shortlist ────────────────────────────────────────────────────────────
def test_the_shortlist_narrows_without_testing_anything():
    """
    A descriptive statistic, not a hypothesis test — which is what makes it
    free of multiple-testing cost.
    """
    rets = independent(10).pct_change().dropna()
    pairs, funnel = ps.shortlist(rets, min_corr=0.9)

    assert funnel["pairs_possible"] == 45
    assert pairs == [] and funnel["shortlisted"] == 0


def test_the_shortlist_is_capped_however_correlated_everything_is():
    a, b = cointegrated(seed=4)
    px = frame({f"C{i}": a * (1 + 0.001 * i) for i in range(12)})
    pairs, funnel = ps.shortlist(px.pct_change().dropna(), min_corr=0.0, cap=7)

    assert len(pairs) == 7 and funnel["shortlisted"] == 7
    assert funnel["pairs_correlated"] == 66


def test_the_shortlist_can_be_restricted_to_one_sector():
    """
    Two stocks from unrelated industries correlating 0.6 over a year are
    usually telling you about the market, not about each other.
    """
    rets = independent(6).pct_change().dropna()
    groups = {"S00": "半导体", "S01": "半导体", "S02": "银行",
              "S03": "银行", "S04": "银行", "S05": "白酒"}
    pairs, _ = ps.shortlist(rets, min_corr=0.0, within=groups)

    assert all(groups[a] == groups[b] for a, b, _ in pairs)
    assert len(pairs) == 1 + 3          # 1 semi pair + 3 bank pairs, 白酒 alone


# ── lead-lag specifics ───────────────────────────────────────────────────────
@pytest.mark.parametrize("train_leads, test_leads, agree", [
    ("a", "a", True), ("b", "b", True), ("a", "b", False), ("b", "a", False),
])
def test_a_pair_that_swaps_direction_between_halves_is_flagged(
        train_leads, test_leads, agree):
    """
    Not a lead — noise that happened to be significant twice, pointing a
    different way each time. Tested on the flag directly so both outcomes are
    covered; a run on random data gives whichever the seed happens to produce.
    """
    row = ps._row({"a": "A", "b": "B", "corr": 0.7,
                   "train": {"p": 0.01, "leads": train_leads, "lag": 1, "n": 300},
                   "test": {"p": 0.01, "leads": test_leads, "lag": 2, "n": 300}},
                  0.01, "lead-lag")
    assert row["same_direction"] is agree
    assert row["leads"] == test_leads          # reported from the CONFIRMING half


def test_a_lead_lag_row_says_whether_both_halves_agree_on_direction():
    out = ps.scan(independent(10), "lead-lag", min_corr=0.0)
    for row in out["rows"]:
        assert isinstance(row["same_direction"], bool)
        assert row["leads"] in ("a", "b")


def test_an_engineered_lead_is_found_and_pointed_the_right_way():
    rng = np.random.default_rng(5)
    driver = rng.normal(0, 0.012, N)
    follower = np.r_[0, 0, driver[:-2]] * 0.9 + rng.normal(0, 0.004, N)
    px = frame({"LEADER": 100 * np.exp(np.cumsum(driver)),
                "FOLLOWER": 100 * np.exp(np.cumsum(follower)),
                "N1": noise(31), "N2": noise(32)})

    out = ps.scan(px, "lead-lag", min_corr=0.0)
    row = next((r for r in out["rows"]
                if {r["a"], r["b"]} == {"LEADER", "FOLLOWER"}), None)

    assert row is not None and row["survives"]
    leader = row["a"] if row["leads"] == "a" else row["b"]
    assert leader == "LEADER"
    assert row["same_direction"] is True


# ── the arithmetic of the report ─────────────────────────────────────────────
def test_the_expected_by_chance_count_matches_what_was_retested():
    out = ps.scan(independent(14), "lead-lag", min_corr=0.0)
    f = out["funnel"]
    assert f["expected_by_chance"] == pytest.approx(f["retested"] * ps.ALPHA, abs=0.01)


def test_the_raw_holdout_count_is_reported_beside_what_noise_would_give():
    """
    `survivors` is corrected and `expected_by_chance` is not, so comparing
    those two directly flatters the screen. The raw holdout count is the
    number that belongs next to the expectation.
    """
    out = ps.scan(independent(14), "lead-lag", min_corr=0.0)
    f = out["funnel"]

    assert f["retest_hits"] == sum(1 for r in out["rows"]
                                   if (r["p_test"] or 1) < ps.ALPHA)
    assert f["retest_hits"] >= f["survivors"]     # correction only ever removes


def test_the_funnel_only_ever_narrows():
    out = ps.scan(independent(14), "lead-lag", min_corr=0.0)
    f = out["funnel"]
    assert (f["pairs_possible"] >= f["pairs_correlated"] >= f["shortlisted"]
            >= f["screened"] >= f["survivors"])


def test_an_unknown_kind_is_refused():
    with pytest.raises(LookupError):
        ps.scan(independent(3), "vibes")
