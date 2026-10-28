"""
板块周期 — each 板块's phase in its own trend cycle, and which pairs see-saw.

The tests that matter most are about not fooling the reader: independent
sectors must rarely be called a see-saw, a real rotation must be found and
pointed the right way, and "relative to 沪深300" must not manufacture
see-saws out of β.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sector_cycle as C  # noqa: E402

IDX = pd.bdate_range("2024-01-02", periods=700)


def walk(seed, vol=0.015, drift=0.0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, len(IDX)))), IDX)


def flow(n, rng, phi=0.97, scale=0.0015):
    f = np.zeros(n)
    e = rng.normal(0, scale, n)
    for t in range(1, n):
        f[t] = phi * f[t - 1] + e[t]
    return f


def rotation(seed=0, strength=1.0, lag=4, others=3):
    """A rides +F, B rides −F `lag` days later; the rest have their own flows."""
    rng = np.random.default_rng(seed)
    n = len(IDX)
    M = rng.normal(0.0003, 0.01, n)
    F = flow(n + lag, rng) * strength
    raw = {"A": M + F[lag:] + rng.normal(0, 0.012, n), "B": M - F[:n] + rng.normal(0, 0.012, n)}
    for k in range(others):
        raw[f"N{k}"] = M + flow(n, rng) + rng.normal(0, 0.012, n)
    closes = {k: pd.Series(100 * np.exp(np.cumsum(v)), IDX) for k, v in raw.items()}
    return closes, pd.Series(100 * np.exp(np.cumsum(M)), IDX)


# ── the oscillator ───────────────────────────────────────────────────────────
def test_the_oscillator_reads_in_each_sectors_own_units():
    """A sector ten times as volatile, moving the same way, reads the same."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.005, len(IDX))
    calm = pd.Series(100 * np.exp(np.cumsum(r)), IDX)
    wild = pd.Series(100 * np.exp(np.cumsum(r * 6)), IDX)
    a, b = C.oscillator(calm).dropna(), C.oscillator(wild).dropna()
    assert np.corrcoef(a, b)[0, 1] > 0.95
    assert abs(a.std() - b.std()) / a.std() < 0.25


def test_a_spike_does_not_shrink_its_own_reading():
    """Measured against the spread up to YESTERDAY — the jump is not in its own yardstick."""
    s = walk(1, vol=0.01)
    s.iloc[400:] *= 1.15
    o = C.oscillator(s)
    dev = s / s.rolling(C.MA_DAYS).mean() - 1
    expected = dev.iloc[400] / dev.iloc[400 - C.SPREAD_DAYS:400].std()
    assert o.iloc[400] == pytest.approx(expected, rel=1e-9)


# ── phases ───────────────────────────────────────────────────────────────────
def test_a_cycle_passes_through_the_four_phases_in_order():
    t = np.arange(len(IDX))
    s = pd.Series(100 * np.exp(0.15 * np.sin(2 * np.pi * t / 80)), IDX)
    ph = C.phases(C.oscillator(s)).dropna()
    seq = [p for p, _, _ in C.runs(ph[ph.index >= "2024-10-14"])]
    order = ["up", "topping", "down", "bottoming"]
    for x, y in zip(seq, seq[1:]):
        assert order.index(y) == (order.index(x) + 1) % 4


def test_a_wobble_around_the_line_does_not_flip_the_phase():
    below = np.r_[np.full(20, -1.0), 0.1, -0.1, 0.15, -0.05, np.full(20, -1.0)]
    assert not any(C._sticky(below, C.ZERO_BAND))
    above = -below
    assert all(C._sticky(above, C.ZERO_BAND))


def test_turns_and_phases_never_disagree():
    closes, bench = rotation(seed=3)
    out = C.analyse(closes, bench, "abs")
    for r in out["board"]:
        if r["last_cross"] and r["phase"]:
            above = r["phase"] in ("up", "topping")
            assert above == (r["last_cross"]["dir"] == "up"), r


def test_a_turn_is_exactly_a_change_of_side_in_the_phase():
    o = C.oscillator(walk(11)).to_numpy()
    ph = C.phases(pd.Series(o)).to_numpy()
    side = [None if p is None else p in ("up", "topping") for p in ph]
    ups = [i for i in range(1, len(side)) if side[i] and side[i - 1] is False]
    downs = [i for i in range(1, len(side)) if side[i] is False and side[i - 1]]
    assert list(np.flatnonzero(C.crosses(o, "up"))) == ups
    assert list(np.flatnonzero(C.crosses(o, "down"))) == downs


def test_every_pair_is_corrected_for_all_the_others():
    closes, bench = rotation(seed=6)
    out = C.analyse(closes, bench, "abs")
    ps = [p["p"] for p in out["pairs"]]
    assert [p["q"] for p in out["pairs"]] == pytest.approx(C.bh_q(ps), abs=1e-4)
    assert all(p["q"] >= p["p"] for p in out["pairs"])


def test_the_typical_stay_needs_a_few_examples():
    closes, bench = rotation(seed=4)
    out = C.analyse(closes, bench, "abs")
    for r in out["board"]:
        for v in r["typical_all"].values():
            assert v is None or v >= 1


# ── surrogates and the tests on them ─────────────────────────────────────────
def test_surrogates_keep_the_rhythm_and_lose_the_timing():
    x = C.oscillator(walk(5)).dropna().to_numpy()
    s = C.surrogates(x, draws=20)
    assert s.shape == (20, len(x))
    px = np.abs(np.fft.rfft(C._z(x))) ** 2
    for row in s[:5]:
        pr = np.abs(np.fft.rfft(row)) ** 2
        assert np.allclose(pr[1:], px[1:], rtol=1e-6)
    assert abs(np.corrcoef(x, s[0])[0, 1]) < 0.5


def test_independent_sectors_are_rarely_called_a_see_saw():
    """The rotation null this replaced called 9% of these p<0.01; surrogates ~1%."""
    ps = []
    for k in range(60):
        a, b = C.oscillator(walk(100 + k)), C.oscillator(walk(500 + k))
        both = pd.concat([a, b], axis=1)[IDX >= "2024-10-14"].dropna()
        ps.append(C.lag_test(both[0].to_numpy(), both[1].to_numpy(),
                             C.surrogates(both[1].to_numpy(), draws=400))["p"])
    ps = np.asarray(ps)
    assert np.mean(ps < 0.05) <= 0.12
    assert np.mean(ps < 0.01) <= 0.05


def test_a_real_rotation_is_found_with_the_right_order():
    found = 0
    for seed in range(4):
        closes, bench = rotation(seed=seed, strength=1.2)
        out = C.analyse(closes, bench, "rel")
        ab = next(p for p in out["pairs"] if {p["a"], p["b"]} == {"A", "B"})
        lag = ab["best_lag"] if ab["a"] == "A" else -ab["best_lag"]
        if ab["q"] <= 0.1 and ab["best_corr"] < -0.4 and lag >= 0:
            found += 1
    assert found >= 3


def test_the_opposite_share_is_one_matrix_product_and_the_same_number():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=300), rng.normal(size=300) + 0.4
    surr = C.surrogates(y, draws=50)
    fast = C.opposite_test(x, y, C.prepared(y, surr))
    level = y.mean() + surr * y.std()
    slow = np.mean(np.sign(x)[None, :] != np.sign(level), axis=1)
    assert fast["expected"] == pytest.approx(float(np.median(slow)))


def test_bh():
    assert C.bh_q([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])


# ── relative to 沪深300 ──────────────────────────────────────────────────────
def test_relative_means_minus_beta_not_divided_by_the_index():
    """
    A β≈0 sector and a β≈2 sector with nothing else in common. Divided by the
    index they see-saw on every market move; with β removed they do not.
    """
    rng = np.random.default_rng(7)
    n = len(IDX)
    m = rng.normal(0, 0.012, n)
    low = pd.Series(100 * np.exp(np.cumsum(0.0 * m + rng.normal(0, 0.008, n))), IDX)
    high = pd.Series(100 * np.exp(np.cumsum(2.0 * m + rng.normal(0, 0.008, n))), IDX)
    bench = pd.Series(100 * np.exp(np.cumsum(m)), IDX)
    ratio = pd.DataFrame({"lo": low / bench, "hi": high / bench}).apply(C.oscillator)
    ratio = ratio[IDX >= "2024-10-14"].dropna()
    assert np.corrcoef(ratio["lo"], ratio["hi"])[0, 1] < -0.4
    resid = C.build_frame({"lo": low, "hi": high}, bench, "rel").apply(C.oscillator)
    resid = resid[IDX >= "2024-10-14"].dropna()
    assert abs(np.corrcoef(resid["lo"], resid["hi"])[0, 1]) < 0.2


def test_the_betas_are_reported():
    closes, bench = rotation(seed=1)
    frame = C.build_frame(closes, bench, "rel")
    assert set(frame.attrs["beta"]) == set(closes)
    assert all(0.3 < b < 1.7 for b in frame.attrs["beta"].values())


# ── the payload and its edges ────────────────────────────────────────────────
def test_the_window_starts_after_the_924_pivot():
    closes, bench = rotation(seed=2)
    out = C.analyse(closes, bench, "abs")
    assert out["start"] >= "2024-10-14"
    assert len(out["pairs"]) == 10


def test_the_pair_view_carries_its_own_evidence_and_the_present():
    closes, bench = rotation(seed=2, strength=1.2)
    d = C.pair_detail(closes, "A", "B", bench, "rel")
    assert len(d["osc_a"]) == len(d["osc_b"]) == len(d["dates"]) == len(d["opposite_mask"])
    assert len(d["lag"]["curve"]) == 2 * C.MAX_LAG + 1
    assert d["a_up_b_down"]["events"] > 0
    assert d["conclusions"][-1].startswith("现在")


def test_it_refuses_what_it_cannot_answer():
    closes, bench = rotation(seed=0)
    with pytest.raises(ValueError):
        C.pair_detail(closes, "A", "A", bench)
    with pytest.raises(LookupError):
        C.pair_detail(closes, "A", "nope", bench)
    with pytest.raises(LookupError):
        C.analyse(closes, None, "rel")
    with pytest.raises(LookupError):
        C.analyse({k: v.iloc[-200:] for k, v in closes.items()}, bench, "abs", start="2026-06-01")
