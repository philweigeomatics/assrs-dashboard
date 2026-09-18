"""
wyckoff: the four phases have to mean what they say.

Offline. Each test builds a price path with exactly one phase's signature and
checks it gets that label — a rising path into new highs is markup, a quiet
path along the lows is accumulation, a churning path near the highs is
distribution. Then the boring but load-bearing properties: no phase before the
windows are full, and the forward-return table measured against a baseline.

    python -m pytest api/tests/test_wyckoff.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import wyckoff as wy  # noqa: E402

N = 400


def frame(closes, volumes=None, wiggle=0.004) -> pd.DataFrame:
    """OHLCV around a close path, with a fixed intrabar range."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.bdate_range("2023-01-02", periods=n)
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, closes) * (1 + wiggle),
        "Low": np.minimum(opens, closes) * (1 - wiggle),
        "Close": closes,
        "Volume": np.full(n, 1e6) if volumes is None else np.asarray(volumes, float),
    }, index=idx)


def quiet(n=N, level=100.0, noise=0.002, seed=1):
    """A flat, low-volatility path — the accumulation signature."""
    rng = np.random.default_rng(seed)
    return level * np.exp(np.cumsum(rng.normal(0, noise, n)) * 0.2)


# ── the phases ───────────────────────────────────────────────────────────────
def test_a_steady_climb_to_new_highs_is_markup():
    closes = 100 * np.exp(np.linspace(0, 0.6, N))
    res = wy.analyse(frame(closes))
    assert res["phase"] == "markup"
    assert res["position_pct"] > 75


def test_a_steady_slide_to_new_lows_is_markdown():
    closes = 100 * np.exp(np.linspace(0, -0.6, N))
    res = wy.analyse(frame(closes))
    assert res["phase"] == "markdown"
    assert res["position_pct"] < 25


def test_a_quiet_drift_along_the_lows_is_accumulation():
    """
    Lower half of the range, volatility under its own baseline. The calm leg
    has to be SHORTER than the 120-day channel, or the channel ends up entirely
    inside it and a flat line reads as the top of its own range.
    """
    loud = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0008, 0.018, 300)))
    calm = np.linspace(loud[-1] * 0.80, loud[-1] * 0.808, 100)   # barely rising
    res = wy.analyse(frame(np.r_[loud, calm]))

    assert res["phase"] == "accumulation"
    assert res["position_pct"] < 50
    assert res["volatility"] < res["vol_baseline"]


def test_churn_high_in_the_range_is_distribution():
    """
    Upper half of the range with volatility ABOVE baseline. Not markup: the
    close keeps failing back under its 20-day mean, so the trend test is what
    separates the two, and that is the whole Wyckoff claim.
    """
    calm = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.002, 0.004, 300)))
    # ±6% swings on a 40-bar cycle, ending just past a peak: high in the range,
    # already back under the 20-day mean.
    churn = calm[-1] * (1 + 0.06 * np.sin(2 * np.pi * np.arange(100) / 40))
    res = wy.analyse(frame(np.r_[calm, churn]))

    assert res["phase"] == "distribution"
    assert res["position_pct"] >= 50
    assert res["volatility"] > res["vol_baseline"]


def test_markup_wins_over_distribution_when_both_conditions_hold():
    """
    A volatile climb into new highs satisfies BOTH rules. It is markup — the
    order of the conditions is the definition, not an implementation detail.
    """
    rng = np.random.default_rng(9)
    closes = 100 * np.exp(np.linspace(0, 0.8, N) + np.cumsum(rng.normal(0, 0.012, N)) * 0.1)
    m = wy.classify(frame(closes))
    last = m.iloc[-1]

    assert last["position"] > 0.75 and last["volatility"] > last["vol_baseline"]
    assert last["phase"] == "markup"


# ── the windows ──────────────────────────────────────────────────────────────
def test_no_phase_is_reported_before_its_baseline_exists():
    """
    np.select labels an all-NaN row "transition" because every condition is
    False. Those rows must be dropped, not drawn — a fabricated phase at the
    left edge of every chart is worse than a shorter chart.
    """
    m = wy.classify(frame(quiet()))
    assert m["vol_baseline"].notna().all()
    assert len(m) < N
    # The first labelled bar is the first one where every window is full.
    assert m.index[0] >= frame(quiet()).index[wy.LOOKBACK]


def test_not_enough_history_is_refused():
    with pytest.raises(LookupError, match="至少需要"):
        wy.classify(frame(quiet(n=60)))


def test_a_missing_column_is_refused_by_name():
    df = frame(quiet()).drop(columns=["Volume"])
    with pytest.raises(LookupError, match="Volume"):
        wy.classify(df)


# ── the phase run ────────────────────────────────────────────────────────────
def test_the_current_phase_is_dated_from_the_full_history_not_the_window():
    """
    A phase that began before the drawn window did not begin on the chart's
    first bar. Charting 60 of 400 bars must not shorten a 200-day markup to 60.
    """
    closes = 100 * np.exp(np.linspace(0, 0.6, N))
    wide = wy.analyse(frame(closes), bars=200)
    narrow = wy.analyse(frame(closes), bars=40)
    assert narrow["since"] == wide["since"]
    assert narrow["days_in_phase"] == wide["days_in_phase"] > 40


def test_spans_cover_every_drawn_bar_exactly_once():
    rng = np.random.default_rng(11)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.011, N)))
    res = wy.analyse(frame(closes))

    covered = []
    for s in res["spans"]:
        assert s["to"] >= s["from"]
        covered.extend(range(s["from"], s["to"] + 1))
    assert covered == list(range(len(res["dates"])))
    assert len(res["bars"]) == len(res["dates"]) == len(res["channel"]["high"])


def test_neighbouring_spans_never_share_a_phase():
    rng = np.random.default_rng(12)
    res = wy.analyse(frame(100 * np.exp(np.cumsum(rng.normal(0, 0.011, N)))))
    phases = [s["phase"] for s in res["spans"]]
    assert all(a != b for a, b in zip(phases, phases[1:]))


# ── the edge ─────────────────────────────────────────────────────────────────
def test_each_phase_reports_what_followed_it():
    rng = np.random.default_rng(13)
    res = wy.analyse(frame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, N)))))
    rows = {r["phase"]: r for r in res["edge"]["rows"]}

    assert set(rows) == set(wy.ORDER) | {"transition"}
    assert any(r["n"] > 0 for r in rows.values())


def test_the_edge_is_a_deviation_from_the_baseline_not_the_raw_return():
    """
    In an index that rose 20% over the sample, every phase "made money". The
    edges must be deviations from the pooled mean, so they cancel out.
    """
    closes = 100 * np.exp(np.linspace(0, 0.2, N)
                          + np.cumsum(np.random.default_rng(14).normal(0, 0.01, N)))
    res = wy.analyse(frame(closes))
    rows = [r for r in res["edge"]["rows"] if r["n"]]

    assert res["edge"]["baseline_pct"] > 0
    weighted = sum(r["edge_pp"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
    assert weighted == pytest.approx(0.0, abs=0.05)


def test_a_thin_phase_is_flagged_rather_than_quietly_reported():
    rng = np.random.default_rng(15)
    res = wy.analyse(frame(100 * np.exp(np.cumsum(rng.normal(0, 0.011, N)))))
    for row in res["edge"]["rows"]:
        if row["n"]:
            assert row["thin"] == (row["n"] < wy.MEANINGFUL_N)
    assert res["edge"]["overlapping"] is True


def test_the_forward_return_looks_forward():
    """
    A path that only rises must show a positive mean forward return; a shifted
    sign or a backward shift would flip it.
    """
    res = wy.analyse(frame(100 * np.exp(np.linspace(0, 0.6, N))))
    rows = [r for r in res["edge"]["rows"] if r["n"]]
    assert all(r["mean_pct"] > 0 for r in rows)
    assert res["edge"]["baseline_pct"] > 0


# ── confirmation ─────────────────────────────────────────────────────────────
def flicker(pattern: str) -> pd.Series:
    """'aaabaaa' → a phase series, one letter per bar."""
    return pd.Series(list(pattern), index=pd.bdate_range("2024-01-01", periods=len(pattern)))


def test_a_one_bar_flicker_does_not_count_as_a_regime_change():
    out = wy._confirm(flicker("aaaaabaaaaa"), 3)
    assert "".join(out) == "aaaaaaaaaaa"


def test_a_change_that_holds_is_adopted_on_the_bar_that_confirms_it():
    """
    Three bars of "b" with confirm=3: the label turns on the THIRD one, not
    retroactively on the first. Backdating it is exactly the look-ahead that
    would poison the forward-return table.
    """
    out = wy._confirm(flicker("aaaaabbbbb"), 3)
    assert "".join(out) == "aaaaaaabbb"


def test_confirmation_reads_only_the_past():
    """
    The label on any bar must not change when later bars arrive. Truncate the
    series anywhere and every surviving label has to be identical.
    """
    raw = flicker("aabaabbbaabbbbaaabab")
    full = wy._confirm(raw, 3)
    for cut in range(1, len(raw) + 1):
        assert list(wy._confirm(raw.iloc[:cut], 3)) == list(full.iloc[:cut]), f"cut={cut}"


def test_confirm_one_is_the_raw_rules_unchanged():
    raw = flicker("aabaabbbaab")
    assert list(wy._confirm(raw, 1)) == list(raw)


def test_confirmation_cuts_the_number_of_regime_changes():
    rng = np.random.default_rng(21)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.013, 600)))
    raw = wy.analyse(frame(closes), confirm=1)
    smooth = wy.analyse(frame(closes))

    assert len(smooth["spans"]) < len(raw["spans"])
    assert smooth["confirm"] == wy.CONFIRM


def test_the_raw_label_is_kept_alongside_the_confirmed_one():
    """Confirmation is a presentation choice; the rules' own answer stays."""
    m = wy.classify(frame(quiet()))
    assert {"raw_phase", "phase"} <= set(m.columns)


@pytest.mark.parametrize("seed", [5, 6, 13, 14, 16])
def test_the_warm_up_does_not_seed_the_confirmation(seed):
    """
    Rows before the windows fill read "transition" — every condition is False
    on NaN. Confirming ACROSS them means the first bars of every chart inherit
    that and spend three sessions reporting a phase the market was never in.

    Several seeds because it only shows when the first labelled bar disagrees
    with the warm-up, which a trending fixture happens to hide.
    """
    rng = np.random.default_rng(seed)
    m = wy.classify(frame(100 * np.exp(np.cumsum(rng.normal(0, 0.013, N)))))
    assert m["phase"].iloc[0] == m["raw_phase"].iloc[0]
