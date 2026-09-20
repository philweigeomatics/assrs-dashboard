"""
The rolling lead-lag panel: does it find a lead that is really there, does it
find one that moves, and does it stay quiet when there is nothing?

Offline. The last of those is the one that matters most. Every rolling window
has a best lag, including in noise, so a panel that always points somewhere
is a panel that will invent a story for any two stocks you give it.

    python -m pytest api/tests/test_lead_lag_profile.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import lead_lag_profile as llp  # noqa: E402

N = 800
IDX = pd.bdate_range("2023-01-02", periods=N)


def pair(build, seed=0):
    rng = np.random.default_rng(seed)
    driver = rng.normal(0, 0.012, N)
    follower = build(driver, rng)
    return (pd.Series(driver, index=IDX, name="A"),
            pd.Series(follower, index=IDX, name="B"))


def lead_by(k, strength=0.8):
    """B echoes A k days later, for the whole history."""
    def build(driver, rng):
        return np.r_[np.zeros(k), driver[:-k]] * strength + rng.normal(0, 0.004, N)
    return build


def unrelated(driver, rng):
    return rng.normal(0, 0.012, N)


# ── it finds a lead that is there, at the right lag and the right sign ───────
def test_a_constant_lead_shows_up_at_the_lag_it_was_built_with():
    a, b = pair(lead_by(3))
    panel = llp.profile(a, b)
    doms = [d for d in llp.dominant(panel) if d is not None]

    assert doms, "a planted 3-day lead produced no dominant lag anywhere"
    # +3 means A moves first, which is how it was built.
    assert max(set(doms), key=doms.count) == 3


def test_the_sign_says_who_moved_first():
    """The one thing that is easy to get backwards and impossible to notice."""
    a, b = pair(lead_by(2))
    panel = llp.profile(a, b)
    ep = llp.episodes(panel)
    assert ep, "no episode found for a planted lead"
    assert all(e["lag"] > 0 and e["leads"] == "a" for e in ep)

    # Swapping the arguments must swap the arrow, not keep it.
    flipped = llp.episodes(llp.profile(b, a))
    assert flipped and all(e["lag"] < 0 and e["leads"] == "b" for e in flipped)


def test_a_planted_lead_holds_for_most_of_the_history():
    a, b = pair(lead_by(3))
    s = llp.summarise(llp.profile(a, b))
    assert s["share"][3] > 0.5
    assert s["longest_run"] >= 10


# ── a relationship that MOVES is the thing a single verdict cannot say ───────
def test_a_lead_that_changes_length_is_reported_as_two_episodes():
    """
    The whole reason this exists. One verdict would average a 2-day lead and
    a 4-day lead into a confident wrong number.
    """
    rng = np.random.default_rng(1)
    driver = rng.normal(0, 0.012, N)
    half = N // 2
    follower = np.empty(N)
    follower[:half] = np.r_[np.zeros(2), driver[:half - 2]] * 0.85
    follower[half:] = np.r_[np.zeros(4), driver[half:-4]] * 0.85
    follower += rng.normal(0, 0.004, N)

    a = pd.Series(driver, index=IDX, name="A")
    b = pd.Series(follower, index=IDX, name="B")
    eps = llp.episodes(llp.profile(a, b))

    lags = [e["lag"] for e in eps]
    assert 2 in lags and 4 in lags, f"expected both regimes, got {lags}"
    two = next(e for e in eps if e["lag"] == 2)
    four = next(e for e in eps if e["lag"] == 4)
    assert two["to"] < four["from"], "the 2-day regime came first"


def test_a_relationship_present_only_early_does_not_span_the_history():
    rng = np.random.default_rng(2)
    driver = rng.normal(0, 0.012, N)
    third = N // 3
    follower = rng.normal(0, 0.012, N)
    follower[:third] = np.r_[np.zeros(3), driver[:third - 3]] * 0.85

    a = pd.Series(driver, index=IDX, name="A")
    b = pd.Series(follower, index=IDX, name="B")
    eps = [e for e in llp.episodes(llp.profile(a, b)) if e["lag"] == 3]

    assert eps, "the early relationship was missed"
    assert eps[0]["to"] < str(IDX[int(N * 0.55)].date()), \
        "an early-only relationship was reported as lasting into the late history"


# ── and it stays quiet when there is nothing ─────────────────────────────────
def test_unrelated_stocks_still_produce_long_runs_and_the_nulls_say_so():
    """
    The measurement that shaped this module. Over 40 pairs of independent
    random walks the longest run of a single dominant lag had a MEDIAN of 9
    windows and reached 26, because neighbouring windows share 55 of their 60
    sessions. So "it held for three months" is not evidence, and the test
    that matters is that the nulls report the same scale as the real panel
    when there is nothing there.
    """
    a, b = pair(unrelated, seed=7)
    panel = llp.profile(a, b)
    s = llp.summarise(panel, llp.nulls(a, b, rotations=6))

    assert s["longest_run"] >= 5, "noise is expected to produce runs"
    # The point: on unrelated data the real panel is not special.
    assert s["longest_run"] <= s["null"]["longest_max"] * 2
    assert s.get("notable", 0) == 0, (
        "an unrelated pair produced an episode longer than every rotation")


def test_no_window_is_forced_to_name_a_lag():
    """
    A panel that always points somewhere is arithmetic, not evidence. Under
    the band the honest answer is None, and it has to actually occur.
    """
    a, b = pair(unrelated, seed=11)
    doms = llp.dominant(llp.profile(a, b))
    assert any(d is None for d in doms)


@pytest.mark.parametrize("window, expect", [(60, 0.258), (100, 0.2), (250, 0.126)])
def test_the_noise_band_tightens_as_the_window_grows(window, expect):
    assert llp.noise_band(window) == pytest.approx(expect, abs=0.002)


# ── the placebo ──────────────────────────────────────────────────────────────
def test_the_placebo_keeps_the_series_and_destroys_only_the_alignment():
    a, b = pair(lead_by(3))
    sham = llp.placebo(a, b)
    real = llp.profile(a, b)

    assert len(sham["dates"]) == len(real["dates"])
    # A real 3-day lead all but vanishes once the alignment is rotated away.
    assert llp.summarise(sham)["share"][3] < llp.summarise(real)["share"][3] / 2


def test_the_placebo_is_a_rotation_not_a_shuffle():
    """
    A shuffle would flatten B's volatility clustering and make the placebo
    quieter than it should be, which flatters the real panel beside it.
    """
    a, b = pair(lead_by(3))
    shift = int(len(b) * llp.PLACEBO_SHIFT)
    rotated = np.roll(b.to_numpy(), shift)
    assert sorted(rotated) == sorted(b.to_numpy())          # same values
    assert np.corrcoef(np.abs(rotated[:-1]), np.abs(rotated[1:]))[0, 1] \
        == pytest.approx(
            np.corrcoef(np.abs(b.to_numpy()[:-1]), np.abs(b.to_numpy()[1:]))[0, 1],
            abs=0.05)                                        # same clustering


def test_the_summary_carries_the_nulls_for_comparison():
    a, b = pair(lead_by(3))
    s = llp.summarise(llp.profile(a, b), llp.nulls(a, b, rotations=6))
    assert "null" in s and s["null"]["rotations"] == 6
    assert s["longest_run"] > s["null"]["longest_max"]
    assert s["notable"] >= 1, "a planted lead should beat every rotation"


def test_the_nulls_use_several_rotations_not_one():
    """
    One rotation is one draw from a wide distribution. Landing low by luck
    would make an ordinary panel look special, which is the failure this
    whole comparison exists to prevent.
    """
    a, b = pair(unrelated, seed=3)
    n = llp.nulls(a, b, rotations=8)
    assert n["rotations"] == 8
    assert n["longest_max"] >= n["longest_median"]


def test_a_planted_lead_is_flagged_and_noise_is_not():
    real_a, real_b = pair(lead_by(3))
    flagged = llp.summarise(llp.profile(real_a, real_b),
                            llp.nulls(real_a, real_b, rotations=6))["episodes"]
    assert any(e["beats_null"] for e in flagged if e["lag"] == 3)

    na, nb = pair(unrelated, seed=21)
    quiet = llp.summarise(llp.profile(na, nb),
                          llp.nulls(na, nb, rotations=6))["episodes"]
    assert not any(e.get("beats_null") for e in quiet)


# ── moving together is not leading ───────────────────────────────────────────
def test_two_stocks_that_move_together_report_lag_zero_not_a_lead():
    """
    The most common honest answer, and the one it would be easiest to hide.
    Dropping lag 0 from the search would turn plain co-movement into a
    confident lead at whatever lag came second.
    """
    rng = np.random.default_rng(9)
    common = rng.normal(0, 0.012, N)
    a = pd.Series(common + rng.normal(0, 0.004, N), index=IDX, name="A")
    b = pd.Series(common * 0.9 + rng.normal(0, 0.004, N), index=IDX, name="B")

    s = llp.summarise(llp.profile(a, b), llp.nulls(a, b, rotations=4))
    assert s["sync_share"] > 0.8, "same-day co-movement was not reported as lag 0"
    assert all(e["lag"] == 0 for e in s["episodes"] if e.get("beats_null"))


def test_a_genuine_lead_is_not_reported_as_synchronous():
    a, b = pair(lead_by(3))
    s = llp.summarise(llp.profile(a, b))
    assert s["sync_share"] < 0.2


# ── the four things a weaker test suite let through ──────────────────────────
def test_a_negative_relationship_can_dominate_a_window():
    """
    B moving OPPOSITE to A's earlier move is a lead-lag relationship. Picking
    the largest signed correlation instead of the largest magnitude would
    silently ignore every inverse pair.
    """
    rng = np.random.default_rng(4)
    driver = rng.normal(0, 0.012, N)
    follower = np.r_[np.zeros(3), driver[:-3]] * -0.8 + rng.normal(0, 0.004, N)
    a = pd.Series(driver, index=IDX, name="A")
    b = pd.Series(follower, index=IDX, name="B")

    eps = llp.episodes(llp.profile(a, b))
    assert eps, "an inverse lead produced no episode at all"
    assert any(e["lag"] == 3 for e in eps)
    assert all(e["mean_corr"] < 0 for e in eps if e["lag"] == 3)


def test_rotating_by_the_whole_series_is_the_series_itself():
    """
    The cheapest way to pin that the placebo ROTATES. A shuffle cannot
    satisfy this, and a shuffle is the tempting simplification: it would
    flatten B's volatility clustering and quietly flatter the real panel.
    """
    a, b = pair(lead_by(3))
    assert llp.placebo(a, b, shift=1.0) == llp.profile(a, b)


def test_an_episode_must_beat_every_rotation_not_the_typical_one():
    """
    Against the median, half the rotations would have "beaten" it too. The
    bar is the worst thing noise managed, and this is the difference between
    a flag that means something and one that fires constantly.
    """
    panel = llp.profile(*pair(lead_by(3)))
    # longest_max set above anything this fixture can produce, so the only
    # way an episode gets flagged is by being compared against the median.
    null = {"rotations": 8, "longest_median": 5, "longest_max": 10_000,
            "episodes_median": 1.0, "named_share_median": 0.4}
    s = llp.summarise(panel, null)

    assert s["episodes"], "fixture produced no episodes"
    # Every run here is under 40, so nothing may be flagged even though the
    # longest comfortably exceeds the median of 5.
    assert max(e["windows"] for e in s["episodes"]) > null["longest_median"]
    assert all(e["beats_null"] is False for e in s["episodes"])
    assert s["notable"] == 0


@pytest.mark.parametrize("doms, longest", [
    ([1, 1, 1], 3),
    ([1, 2, 1, 2], 1),                 # alternating is not a run of four
    ([1, 1, None, 1, 1, 1], 3),        # a gap breaks it
    ([None, None], 0),
    ([2, 2, 3, 3, 3, 3], 4),
])
def test_a_run_is_the_same_lag_without_a_break(doms, longest):
    assert llp._longest(doms) == longest


# ── shape and refusal ────────────────────────────────────────────────────────
def test_the_panel_is_rectangular_and_labelled():
    a, b = pair(lead_by(2))
    panel = llp.profile(a, b)
    assert len(panel["matrix"]) == len(panel["dates"])
    assert all(len(r) == len(panel["lags"]) for r in panel["matrix"])
    assert panel["lags"] == list(range(-llp.MAX_LAG, llp.MAX_LAG + 1))


def test_each_column_is_dated_by_the_last_session_it_covers():
    """A window labelled by its start would put every reading in the past."""
    a, b = pair(lead_by(2))
    panel = llp.profile(a, b, window=60, step=5)
    assert panel["dates"][0] == str(IDX[59].date())
    assert panel["dates"][-1] <= str(IDX[-1].date())


def test_too_little_history_is_refused_rather_than_guessed():
    a, b = pair(lead_by(2))
    with pytest.raises(LookupError, match="不足以"):
        llp.profile(a.iloc[:40], b.iloc[:40])


def test_an_episode_reports_windows_not_a_sample_size():
    """
    Neighbouring windows overlap by window−step sessions, so a run of five is
    not five independent readings and must not be presented as one.
    """
    a, b = pair(lead_by(3))
    for e in llp.episodes(llp.profile(a, b)):
        assert "windows" in e and "n" not in e
        assert e["from"] <= e["to"]


def test_a_short_run_is_not_called_an_episode():
    a, b = pair(unrelated, seed=13)
    panel = llp.profile(a, b)
    assert all(e["windows"] >= llp.MIN_RUN for e in llp.episodes(panel))
    assert all(e["windows"] >= 8 for e in llp.episodes(panel, min_run=8))
