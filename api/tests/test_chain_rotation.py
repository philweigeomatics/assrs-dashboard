"""
板块接力 · 背离 — relays and divergences along a chain of groups.

The calibration tests matter most: on chains with nothing planted the page
must rarely claim a relay, and a planted one must be found and pointed the
right way. The divergence tests pin the no-hindsight rule: a past day's flag
cannot depend on prices after it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import chain_rotation as R  # noqa: E402


def chain(seed=0, n=700, relay=0.0, big=0.045, inverse=False):
    """Four legs of three stocks. C's big days are followed by D moving `relay` σ a day for two days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-06-01", periods=n)
    M = rng.normal(0.0003, 0.012, n)
    f = {L: rng.normal(0, 0.012, n) for L in "ABCD"}
    days = rng.choice(np.arange(80, n - 10), 25, replace=False)
    f["C"][days] += rng.choice([1, -1], 25) * big
    for t in days:
        s = np.sign(f["C"][t]) * (-1 if inverse else 1)
        f["D"][t + 1] += s * relay * 0.012
        f["D"][t + 2] += s * relay * 0.012
    closes = {f"{L}{j}": 100 * np.exp(np.cumsum(M + f[L] + rng.normal(0, 0.01, n)))
              for L in "ABCD" for j in range(3)}
    legs = [{"name": L, "tickers": [f"{L}{j}" for j in range(3)]} for L in "ABCD"]
    return pd.DataFrame(closes, index=idx), pd.Series(100 * np.exp(np.cumsum(M)), index=idx), legs


def hits(out, q=0.1):
    return [c for cells in out["relay"].values() for c in cells if c["q"] is not None and c["q"] <= q]


# ── the yardstick ────────────────────────────────────────────────────────────
def test_a_leg_is_the_equal_weight_average_of_its_members():
    idx = pd.bdate_range("2025-01-01", periods=3)
    c = pd.DataFrame({"a": [100, 110, 110], "b": [100, 100, 90]}, index=idx, dtype=float)
    lr = R.leg_returns(c, [{"name": "L", "tickers": ["a", "b"]}])
    assert lr["L"].iloc[1] == pytest.approx(0.05)
    assert lr["L"].iloc[2] == pytest.approx(-0.05)


def test_a_big_day_does_not_inflate_its_own_yardstick():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=200)
    m = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    r = pd.DataFrame({"L": rng.normal(0, 0.01, 200)}, index=idx)
    r.iloc[150, 0] = 0.08
    _, sigma, z, _ = R.abnormal(r, m)
    # An 8% day against a 1% yardstick is ~8σ. Let the day into its own
    # yardstick and the yardstick swells to ~1.4% — it reads as ~5.6σ.
    assert z["L"].iloc[150] > 7


def test_a_cluster_of_big_days_is_one_event():
    z = pd.Series([0, 3, 3.1, 2.9, 0, 0, 0, 0, 2.5, 0], dtype=float)
    pos, sgn = R.signed_events(z, 2.0)
    assert list(pos) == [1, 8]


# ── relay ────────────────────────────────────────────────────────────────────
def test_a_planted_relay_is_found_and_points_the_right_way():
    found = 0
    for seed in range(8):
        out = R.analyse(*chain(seed=seed, relay=1.2), k=2.5, lookback=504)
        found += any(c["leader"] == "C" and c["follower"] == "D" and c["excess"] > 0 for c in hits(out))
    assert found >= 5


def test_an_inverse_relay_is_reported_as_inverse():
    for seed in range(8):
        out = R.analyse(*chain(seed=seed, relay=1.2, inverse=True), k=2.5, lookback=504)
        cd = [c for c in hits(out) if c["leader"] == "C" and c["follower"] == "D"]
        if cd:
            assert all(c["excess"] < 0 for c in cd)
            return
    pytest.fail("inverse relay never found")


def test_nothing_planted_is_rarely_called_a_relay():
    flagged = sum(bool(hits(R.analyse(*chain(seed=s), k=2.0, lookback=504))) for s in range(12))
    assert flagged <= 3


def test_every_cell_on_the_page_is_corrected_together():
    out = R.analyse(*chain(seed=1), k=2.0, lookback=504)
    every = [c for cells in out["relay"].values() for c in cells if c["p"] is not None]
    assert len(every) == 3 * 12
    assert all(c["q"] >= c["p"] for c in every)


def test_bh_q_values():
    # Sorted 0.01, 0.03, 0.04, 0.5 → ×4/rank = 0.04, 0.06, 0.0533, 0.5 →
    # running minimum from the top = 0.04, 0.0533, 0.0533, 0.5.
    q = R.bh_q([0.01, 0.04, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])


def test_a_big_day_today_is_announced_with_what_followed_before():
    c, b, legs = chain(seed=2, relay=1.5)
    c = c.copy()
    c.loc[c.index[-1], ["C0", "C1", "C2"]] *= 1.08
    out = R.analyse(c, b, legs, k=2.5, lookback=504)
    assert any(x["leg"] == "C" and x["direction"] == "up" for x in out["live"])


# ── divergence ───────────────────────────────────────────────────────────────
def test_a_past_flag_never_depends_on_later_prices():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-01", periods=300)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))), index=idx)
    w1, st1 = R.flags(s)
    s2 = s.copy()
    s2.iloc[200:] *= 3
    w2, st2 = R.flags(s2)
    assert w1.iloc[:200].equals(w2.iloc[:200]) and st1.iloc[:200].equals(st2.iloc[:200])


def test_upstream_rolling_over_while_downstream_holds_its_high_is_an_episode():
    idx = pd.bdate_range("2024-01-01", periods=260)
    up = np.r_[np.linspace(100, 160, 180), np.linspace(160, 120, 80)]
    down = np.r_[np.linspace(100, 150, 200), np.linspace(150, 110, 60)]
    ix = pd.DataFrame({"U": up, "D": down}, index=idx)
    out = R.divergence(ix, ["U", "D"], idx)
    eps = out[0]["episodes"]
    assert eps, "no episode found"
    e = eps[0]
    assert pd.Timestamp(e["start"]) >= idx[180]
    assert e["days_to_peak"] <= 25


def test_episode_runs_bridge_short_gaps():
    mask = np.zeros(30, dtype=bool)
    mask[[5, 6, 7, 10, 11, 25]] = True
    assert R.runs(mask) == [(5, 11)]


def test_breadth_is_a_share():
    idx = pd.bdate_range("2024-01-01", periods=40)
    c = pd.DataFrame({"a": np.linspace(1, 2, 40), "b": np.linspace(2, 1, 40)}, index=idx)
    b = R.breadth(c).dropna()
    assert (b == 50).all()


# ── inputs ───────────────────────────────────────────────────────────────────
def test_it_refuses_what_it_cannot_answer():
    c, b, legs = chain()
    with pytest.raises(ValueError):
        R.analyse(c, b, legs[:1])
    with pytest.raises(ValueError):
        R.analyse(c, b, [legs[0], {**legs[1], "name": legs[0]["name"]}])
    with pytest.raises(ValueError):
        R.analyse(c, b, legs, k=3.0)
    with pytest.raises(ValueError):
        R.analyse(c, b, legs, lookback=100)
    with pytest.raises(LookupError):
        R.analyse(c, b, legs + [{"name": "X", "tickers": ["nope"]}])


def test_the_payload_is_json():
    import json
    out = R.analyse(*chain(seed=4, relay=1.0), k=2.0, lookback=504)
    json.dumps(out)
    assert len(out["dates"]) == len(out["bench"]) == len(out["breadth"])
    assert all(len(d["mask"]) == len(out["dates"]) for d in out["divergence"])


def test_a_leg_on_the_chart_is_bought_and_held_not_rebalanced_daily():
    """Daily rebalancing between two zig-zagging members manufactures a gain."""
    idx = pd.bdate_range("2025-01-01", periods=5)
    c = pd.DataFrame({"a": [100, 200, 100, 200, 100], "b": [100, 50, 100, 50, 100]}, index=idx, dtype=float)
    ix = R.leg_index(c, [{"name": "L", "tickers": ["a", "b"]}])
    assert ix["L"].iloc[-1] == pytest.approx(1.0)
    rebalanced = (1 + R.leg_returns(c, [{"name": "L", "tickers": ["a", "b"]}])["L"].fillna(0)).cumprod()
    assert rebalanced.iloc[-1] > 2
