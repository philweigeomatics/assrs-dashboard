"""
跷跷板 — do stocks take turns once the market is taken out?

The tests that matter most are the honesty ones: a rotation hidden under a
shared market move must be found, and noise must not be dressed up as one —
neither by picking the most negative of many pairs, nor by searching for the
most opposed split, nor by subtracting the basket's own average.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import seesaw as S  # noqa: E402

GROUPS = [(1, 1.2), (1, 1.0), (1, 1.4), (-1, 0.9), (-1, 1.1), (-1, 1.3)]


def basket(rot=0.012, seed=0, n=500, market_vol=0.015, groups=GROUPS):
    """Six stocks on one market factor; `rot` pushes the first three up when the rest go down."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-10-01", periods=n)
    M = rng.normal(0.0005, market_vol, n)
    F = rng.normal(0, rot, n) if rot else np.zeros(n)
    cols = {f"S{i}": 100 * np.exp(np.cumsum(b * M + sg * F + rng.normal(0, 0.012, n)))
            for i, (sg, b) in enumerate(groups)}
    return pd.DataFrame(cols, index=idx), pd.Series(100 * np.exp(np.cumsum(M)), index=idx)


# ── the market hides the rotation ────────────────────────────────────────────
def test_a_rotation_under_a_shared_market_looks_positive_raw_and_negative_resid():
    p, m = basket()
    out = S.analyse(p, m, labels={}, bench_label="X")
    d = out["daily"]
    cross = [d["raw_corr"][i][j] for i in range(3) for j in range(3, 6)]
    cross_res = [d["resid_corr"][i][j] for i in range(3) for j in range(3, 6)]
    assert np.mean(cross) > 0.2
    assert np.mean(cross_res) < -0.2


def test_beta_is_measured_per_stock():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.01, 400))
    r = pd.DataFrame({"a": 1.5 * m + rng.normal(0, 0.002, 400), "b": 0.5 * m + rng.normal(0, 0.002, 400)})
    resid, betas = S.residuals(r, m)
    assert betas["a"] == pytest.approx(1.5, abs=0.05) and betas["b"] == pytest.approx(0.5, abs=0.05)
    assert abs(resid["a"].corr(m)) < 0.05


def test_residuals_are_against_the_index_not_the_basket_average():
    """
    Two unrelated stocks minus their own average are exactly opposite (-1).
    Against an outside index they stay unrelated. This is the trap that would
    manufacture a seesaw from nothing.
    """
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2024-01-01", periods=400)
    p = pd.DataFrame({"a": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
                      "b": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))}, index=idx)
    m = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
    d = S.analyse(p, m, labels={}, bench_label="X")["daily"]
    assert abs(d["resid_corr"][0][1]) < 0.15


# ── finding the camps ────────────────────────────────────────────────────────
def test_the_two_camps_are_found_and_hold_out_of_sample():
    p, m = basket()
    c = S.analyse(p, m, labels={}, bench_label="X")["camps"]
    assert {frozenset(c["a"]), frozenset(c["b"])} == {frozenset(["S0", "S1", "S2"]), frozenset(["S3", "S4", "S5"])}
    assert c["confirmed"] and c["out_of_sample"]["same_split"]
    assert c["daily"]["between"] < -0.3


def _late_rotation(seed):
    p1, m1 = basket(rot=0.0, seed=seed, n=250)
    p2, m2 = basket(rot=0.015, seed=seed + 100, n=250)
    p2 = p2 / p2.iloc[0] * p1.iloc[-1]
    m2 = m2 / m2.iloc[0] * m1.iloc[-1]
    p2.index = m2.index = pd.bdate_range(p1.index[-1] + pd.Timedelta(days=1), periods=250)
    return S.analyse(pd.concat([p1, p2]), pd.concat([m1, m2]), labels={}, bench_label="X")


def test_a_rotation_that_only_started_halfway_is_not_confirmed_but_is_called_new():
    """
    The split is chosen on the first half. When the first half is noise it
    picks some other grouping; that must not count as confirming these camps,
    however well it happens to do afterwards. It is reported as recent instead.
    """
    outs = [_late_rotation(s)["camps"] for s in range(8)]
    assert not any(c and c["confirmed"] for c in outs)
    assert sum(bool(c and c["emerging"]) for c in outs) >= 6


def test_no_split_when_everything_moves_together():
    C = np.full((4, 4), 0.6)
    np.fill_diagonal(C, 1)
    assert S.best_split(C) is None


def test_two_stocks_have_no_camps_to_find():
    p, m = basket(groups=GROUPS[:1] + GROUPS[3:4])
    assert S.analyse(p, m, labels={}, bench_label="X")["camps"] is None


# ── not dressing up noise ────────────────────────────────────────────────────
def test_the_most_negative_of_many_pairs_needs_a_higher_bar():
    assert S.search_threshold(500, 15) > 2 / np.sqrt(500)
    assert S.search_threshold(500, 28) > S.search_threshold(500, 15)


def test_noise_is_rarely_called_a_seesaw():
    pair = camp = 0
    for seed in range(30):
        p, m = basket(rot=0.0, seed=seed)
        out = S.analyse(p, m, labels={}, bench_label="X")
        pair += any(c.startswith("最明显") for c in out["conclusions"])
        camp += bool(out["camps"] and out["camps"]["confirmed"])
    assert pair <= 4 and camp <= 4


def test_a_weak_rotation_is_still_found():
    hits = sum(bool((c := S.analyse(*basket(rot=0.005, seed=s), labels={}, bench_label="X")["camps"])
                    and c["confirmed"]) for s in range(10))
    assert hits >= 8


# ── the plumbing ─────────────────────────────────────────────────────────────
def test_a_week_with_no_session_drops_out():
    idx = pd.bdate_range("2025-09-22", "2025-10-24")
    idx = idx[~((idx >= "2025-10-01") & (idx <= "2025-10-03"))]
    idx = idx[~((idx >= "2025-09-29") & (idx <= "2025-09-30"))]          # whole week shut
    p = pd.DataFrame({"a": np.arange(len(idx), dtype=float) + 1}, index=idx)
    w = S.weekly_closes(p)
    assert pd.Timestamp("2025-10-03") not in w.index
    assert len(w) == 4


def test_opposite_share_ignores_flat_periods():
    a = pd.Series([1.0, -1.0, 0.0, 2.0])
    b = pd.Series([-1.0, -1.0, 5.0, -3.0])
    assert S.opposite_share(a, b) == pytest.approx(2 / 3)


def test_without_the_index_it_says_so_instead_of_guessing():
    p, _ = basket()
    out = S.analyse(p, None, labels={}, bench_label=None)
    assert out["daily"]["resid"] is None and out["camps"] is None
    assert any("无法扣除大盘" in c for c in out["conclusions"])


def test_a_short_window_warns_about_the_weekly_numbers():
    p, m = basket(n=61)
    out = S.analyse(p, m, labels={}, bench_label="X")
    assert any("周线只有" in c for c in out["conclusions"])


def test_labels_are_used_in_the_words():
    p, m = basket()
    labels = {f"S{i}": n for i, n in enumerate(["中际旭创", "新易盛", "工业富联", "金山办公", "用友网络", "科大讯飞"])}
    out = S.analyse(p, m, labels=labels, bench_label="沪深300")
    text = "".join(out["conclusions"])
    assert "沪深300" in text and "金山办公" in text
