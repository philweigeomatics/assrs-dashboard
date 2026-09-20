"""
Follow-through: same direction only, and measured in the follower's own units.

The two things the old correlation reading got wrong, and which these tests
exist to keep fixed:

  * an inverse relationship must NOT register as a lead
  * a 1% move must not count as a response from a stock whose ordinary day
    is 3%, and must count as a large one from a stock whose ordinary day is
    0.4%

    python -m pytest api/tests/test_rotation.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import rotation as rot  # noqa: E402

N = 900
IDX = pd.bdate_range("2022-06-01", periods=N)


def build(follower, seed=0, lead_vol=0.02):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, lead_vol, N)
    b = follower(a, rng)
    return (pd.Series(a, index=IDX, name="A"), pd.Series(b, index=IDX, name="B"))


def echo(k, gain, noise=0.004):
    """B repeats gain x A's move, k days later."""
    def f(a, rng):
        return np.r_[np.zeros(k), a[:-k]] * gain + rng.normal(0, noise, N)
    return f


def inverse(k, gain, noise=0.004):
    def f(a, rng):
        return np.r_[np.zeros(k), a[:-k]] * -gain + rng.normal(0, noise, N)
    return f


def independent(a, rng):
    return rng.normal(0, 0.02, N)


def run(a, b, **kw):
    fired = rot.follow_through(a, b, **kw)
    rows = rot.summarise(fired)
    return fired, rows, rot.nulls(a, b, rotations=4, **kw)


# ── direction: only the same way counts ──────────────────────────────────────
def test_a_follower_that_moves_the_same_way_scores_positive():
    a, b = build(echo(3, 0.8))
    _f, rows, null = run(a, b)
    at3 = next(r for r in rows if r["lag"] == 3)

    assert at3["mean"] > 0.5, "a strong same-direction echo did not register"
    assert at3["hit"] > 0.5
    assert rot.verdict(rows, null)["best_lag"] == 3


def test_a_follower_that_moves_the_opposite_way_is_not_a_lead():
    """
    The change this module exists for. Correlation called this a
    relationship of equal standing; it is not a lead, and it must not be
    reported as one.
    """
    a, b = build(inverse(3, 0.8))
    _f, rows, null = run(a, b)
    at3 = next(r for r in rows if r["lag"] == 3)

    assert at3["mean"] < 0, "an inverse pair scored as following"
    assert at3["hit"] < 0.2
    assert rot.verdict(rows, null)["best_lag"] is None


def test_falling_together_counts_exactly_like_rising_together():
    """
    Sign-aligned, so a joint sell-off is the same evidence as a joint rally.
    Both are the money moving.
    """
    a, b = build(echo(2, 0.8))
    fired = rot.follow_through(a, b)
    ups = [r for r in fired["events"] if r["dir"] == "up"]
    downs = [r for r in fired["events"] if r["dir"] == "down"]
    assert ups and downs

    mu = np.mean([r["resp"][2] for r in ups])
    md = np.mean([r["resp"][2] for r in downs])
    assert mu > 0.5 and md > 0.5
    assert abs(mu - md) < 0.35, "up days and down days scored differently"


# ── scale: measured in the follower's own units ──────────────────────────────
def test_a_small_move_from_a_volatile_follower_is_not_a_response():
    """
    The example that prompted this: leader −5%, follower −1% three days on,
    but the follower's ordinary day is ±3%. That is the follower doing
    nothing, and correlation would have called it a relationship.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.02, N)
    # B is noisy on its own and takes only a sliver of A's move.
    b = np.r_[np.zeros(3), a[:-3]] * 0.2 + rng.normal(0, 0.03, N)
    _f, rows, null = run(pd.Series(a, index=IDX, name="A"),
                         pd.Series(b, index=IDX, name="B"))
    at3 = next(r for r in rows if r["lag"] == 3)

    assert at3["mean"] < rot.FOLLOW_Z, (
        f"a 0.2x response inside 3% noise scored {at3['mean']} sigma")
    assert rot.verdict(rows, null)["best_lag"] is None


def test_the_same_percentage_counts_more_for_a_calmer_follower():
    """
    One percent from a 0.4% stock is a large move; one percent from a 3%
    stock is a Tuesday. The measure has to say so.
    """
    rng = np.random.default_rng(4)
    a = rng.normal(0, 0.02, N)
    shared = np.r_[np.zeros(2), a[:-2]] * 0.25

    calm = pd.Series(shared + rng.normal(0, 0.004, N), index=IDX, name="CALM")
    wild = pd.Series(shared + rng.normal(0, 0.030, N), index=IDX, name="WILD")
    lead = pd.Series(a, index=IDX, name="A")

    m_calm = next(r for r in rot.summarise(rot.follow_through(lead, calm))
                  if r["lag"] == 2)["mean"]
    m_wild = next(r for r in rot.summarise(rot.follow_through(lead, wild))
                  if r["lag"] == 2)["mean"]
    assert m_calm > m_wild * 2, (
        f"identical absolute response scored {m_calm} vs {m_wild}")


def test_volatility_is_trailing_not_full_sample():
    """
    A full-sample sigma lets later behaviour decide whether an earlier move
    was large. Doubling only the TAIL of the series must not change how the
    first half's days were scored.
    """
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.02, N), index=IDX, name="X")
    louder = r.copy()
    louder.iloc[N // 2:] *= 4

    z_before = rot.zscores(r).iloc[100:N // 2 - 5]
    z_after = rot.zscores(louder).iloc[100:N // 2 - 5]
    pd.testing.assert_series_equal(z_before, z_after)


def test_a_day_is_never_scored_against_itself():
    """The shift. Without it, an enormous day inflates the sigma it is
    measured against and scores as ordinary."""
    rng = np.random.default_rng(99)
    vals = rng.normal(0, 0.004, 401)
    vals[200] = 0.5                       # one enormous day
    r = pd.Series(vals, index=pd.bdate_range("2023-01-02", periods=401), name="X")
    z = rot.zscores(r, window=60)
    assert z.iloc[200] > 20, "the spike was measured against its own volatility"
    # And the days after it are still scored against a sigma that now
    # includes the spike, which is correct — they genuinely follow a shock.
    assert abs(z.iloc[201]) < 1


# ── events ───────────────────────────────────────────────────────────────────
def test_only_days_that_were_big_for_the_leader_become_events():
    a, b = build(echo(2, 0.6))
    fired = rot.follow_through(a, b, threshold=2.0)
    za = rot.zscores(pd.concat([a, b], axis=1).dropna()["A"])
    for r in fired["events"]:
        assert abs(za.loc[pd.Timestamp(r["date"])]) >= 2.0


def test_a_higher_bar_leaves_fewer_events():
    a, b = build(echo(2, 0.6))
    loose = len(rot.follow_through(a, b, threshold=1.2)["events"])
    tight = len(rot.follow_through(a, b, threshold=2.5)["events"])
    assert tight < loose


def test_every_event_is_dated_and_carries_its_own_numbers():
    """The instances are the output. An average over two years cannot say
    whether the behaviour is still there."""
    a, b = build(echo(2, 0.8))
    for r in rot.follow_through(a, b)["events"][:20]:
        assert r["date"] and r["dir"] in ("up", "down")
        assert r["a_z"] is not None and abs(r["a_z"]) >= rot.EVENT_Z - 1e-9
        assert len(r["resp"]) == rot.MAX_LAG + 1


def test_events_too_close_to_the_end_are_dropped():
    """An event with no room to respond would count as a failure to respond."""
    a, b = build(echo(2, 0.8))
    fired = rot.follow_through(a, b, maxlag=5)
    last = pd.Timestamp(fired["events"][-1]["date"])
    assert last <= pd.Timestamp(fired["to"]) - pd.tseries.offsets.BDay(5)


# ── the null, and refusing to conclude ───────────────────────────────────────
def test_an_unrelated_pair_names_no_lag():
    a, b = build(independent, seed=11)
    _f, rows, null = run(a, b)
    v = rot.verdict(rows, null)
    assert v["best_lag"] is None
    assert all(abs(r["mean"]) < 0.4 for r in rows if r["mean"] is not None)


def test_the_null_sits_near_zero_for_a_real_pair_too():
    """
    Rotation destroys only the correspondence. If the null came back high,
    the measure would be picking up something about the stocks themselves
    rather than the relationship.
    """
    a, b = build(echo(3, 0.8))
    null = rot.nulls(a, b, rotations=4)
    assert all(abs(n["mean"]) < 0.3 for n in null)


def test_a_verdict_needs_enough_events_to_average():
    a, b = build(echo(3, 0.8))
    _f, rows, null = run(a, b)
    assert rot.verdict(rows, null, min_events=10_000)["best_lag"] is None
    assert rot.verdict(rows, null, min_events=10_000)["enough"] is False


def test_lag_zero_is_reported_but_never_chosen():
    """
    Same-day co-movement is usually the whole story, and it is not a lead —
    there is no gap to act in. It is reported so the reader sees it, and
    excluded from the verdict so it cannot be sold as one.
    """
    rng = np.random.default_rng(7)
    common = rng.normal(0, 0.02, N)
    a = pd.Series(common + rng.normal(0, 0.004, N), index=IDX, name="A")
    b = pd.Series(common * 0.9 + rng.normal(0, 0.004, N), index=IDX, name="B")

    _f, rows, null = run(a, b)
    v = rot.verdict(rows, null)
    assert next(r for r in rows if r["lag"] == 0)["mean"] > 1.0
    assert v["same_day"] > 1.0
    assert v["best_lag"] is None, "same-day co-movement was sold as a lead"


def test_too_little_history_is_refused():
    a, b = build(echo(2, 0.8))
    with pytest.raises(LookupError, match="不足以"):
        rot.follow_through(a.iloc[:40], b.iloc[:40])


# ── each condition in the verdict, pinned on its own ─────────────────────────
def rows_at(lag, mean, hit, n=40):
    """A per-lag summary with one lag set to whatever is being tested."""
    return [{"lag": k, "n": n,
             "mean": mean if k == lag else 0.0,
             "median": 0.0,
             "hit": hit if k == lag else 0.1,
             "cum": 0.0} for k in range(0, 6)]


def null_at(mean_hi, hit_hi):
    return [{"lag": k, "mean": 0.0, "mean_hi": mean_hi,
             "hit": 0.2, "hit_hi": hit_hi, "rotations": 8} for k in range(0, 6)]


def test_all_three_conditions_together_name_a_lag():
    v = rot.verdict(rows_at(3, mean=0.9, hit=0.6), null_at(0.2, 0.3))
    assert v["best_lag"] == 3


def test_a_response_below_half_a_sigma_is_not_a_lead():
    """
    Condition 1. Measured on random pairs the averages wander up to about
    +0.25 sigma, so without this a rotation that happened to land low is
    enough to crown a lag. 0.3 beats the null here and still must not win.
    """
    v = rot.verdict(rows_at(3, mean=0.30, hit=0.6), null_at(0.05, 0.3))
    assert v["best_lag"] is None


def test_a_response_that_rotation_also_produces_is_not_a_lead():
    """Condition 2. Big and frequent, but the rotations reached it too."""
    v = rot.verdict(rows_at(3, mean=0.9, hit=0.6), null_at(1.2, 0.3))
    assert v["best_lag"] is None


def test_a_response_carried_by_a_few_days_is_not_a_lead():
    """
    Condition 3. A large average with a low hit rate is one or two enormous
    days, not a behaviour — and it is the shape that looks best in backtest.
    """
    v = rot.verdict(rows_at(3, mean=0.9, hit=0.25), null_at(0.2, 0.2))
    assert v["best_lag"] is None

    # Nor may it merely beat a low rotation hit rate; 40% is a floor.
    assert rot.verdict(rows_at(3, mean=0.9, hit=0.35),
                       null_at(0.2, 0.05))["best_lag"] is None


def test_the_strongest_qualifying_lag_wins():
    rows = rows_at(2, mean=0.8, hit=0.6)
    rows[4] = {**rows[4], "mean": 1.4, "hit": 0.7}
    assert rot.verdict(rows, null_at(0.2, 0.3))["best_lag"] == 4


def test_the_verdict_does_not_collide_with_the_event_list():
    """
    The payload spreads verdict() next to the events themselves. A key named
    `events` in both replaced the list of dated rows with an integer, and
    every caller then saw a count where it expected a list.
    """
    a, b = build(echo(3, 0.8))
    fired = rot.follow_through(a, b)
    rows = rot.summarise(fired)
    v = rot.verdict(rows, rot.nulls(a, b, rotations=3))

    assert "events" not in v, "verdict would overwrite the event list"
    assert v["n_events"] == len(fired["events"])

    merged = {"events": fired["events"], **v}
    assert isinstance(merged["events"], list)


# ── the cumulative view ──────────────────────────────────────────────────────
def test_the_cumulative_response_adds_up_the_days_after_the_event():
    a, b = build(echo(2, 0.8))
    fired = rot.follow_through(a, b)
    r = fired["events"][0]
    assert r["cum"][0] == 0.0
    expected = sum(v for v in r["resp"][1:4] if v is not None)
    assert r["cum"][3] == pytest.approx(expected, abs=0.02)


def test_a_drift_spread_over_days_shows_up_cumulatively():
    """A response smeared across three days is invisible on any single one."""
    rng = np.random.default_rng(8)
    a = rng.normal(0, 0.02, N)
    b = (np.r_[np.zeros(1), a[:-1]] * 0.25
         + np.r_[np.zeros(2), a[:-2]] * 0.25
         + np.r_[np.zeros(3), a[:-3]] * 0.25
         + rng.normal(0, 0.006, N))
    rows = rot.summarise(rot.follow_through(
        pd.Series(a, index=IDX, name="A"), pd.Series(b, index=IDX, name="B")))

    day3 = next(r for r in rows if r["lag"] == 3)
    assert day3["cum"] > day3["mean"], "the cumulative view added nothing"
    assert day3["cum"] > 1.0
