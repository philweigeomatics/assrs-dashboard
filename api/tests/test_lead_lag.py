"""
领先滞后: the shaping, and the correction the original screen was missing.

Offline. The statistics themselves live in lead_lag_stats.py and are not
re-tested here — they are unchanged, and the Streamlit page has been reading
them for a long time. What is new is the multiple-testing correction, and it
is the part most worth pinning: running Granger in both directions across ten
peers is twenty tests, and under the null twenty tests at a 5% threshold
produce one pass from noise ON AVERAGE — which is enough to put a spurious
"relationship" on the screen most runs.

    python -m pytest api/tests/test_lead_lag.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from api import lead_lag_api as ll  # noqa: E402


def row(ticker, p_t=None, p_s=None):
    return {"ticker": ticker, "p_t_leads_s": p_t, "p_s_leads_t": p_s}


# ── Benjamini-Hochberg ───────────────────────────────────────────────────────
def test_the_q_values_match_the_textbook():
    """
    q_i = min over k>=i of (p_k * n / k). Worked by hand on the classic
    example so the implementation is pinned to arithmetic, not to itself.
    """
    got = ll._bh([0.001, 0.008, 0.039, 0.041, 0.9])
    assert [round(q, 4) for q in got] == [0.005, 0.02, 0.0513, 0.0513, 0.9]


def test_q_values_are_monotone_in_p():
    """A smaller p can never come back with a larger q."""
    ps = [0.9, 0.001, 0.04, 0.2, 0.0001, 0.05]
    qs = ll._bh(ps)
    pairs = sorted(zip(ps, qs))
    assert all(a[1] <= b[1] + 1e-12 for a, b in zip(pairs, pairs[1:]))


def test_a_q_value_is_never_smaller_than_its_p():
    ps = [0.001, 0.01, 0.03, 0.2, 0.7]
    assert all(q >= p - 1e-12 for p, q in zip(ps, ll._bh(ps)))


def test_a_single_test_needs_no_correction():
    assert ll._bh([0.03]) == [0.03]


def test_q_values_are_capped_at_one():
    assert all(q <= 1.0 for q in ll._bh([0.6, 0.7, 0.8, 0.99]))


# ── what the correction is for ───────────────────────────────────────────────
def test_pure_noise_produces_about_one_raw_pass_per_run_and_no_survivors():
    """
    The reason this module exists, stated as the rate it actually is.

    Under the null, a p-value is uniform. Twenty of them therefore contain
    ONE pass at 5% on average — not in every run, which is why this measures
    the rate over many runs rather than trusting a seed. The original screen
    would have labelled each of those passes a relationship; after correction
    essentially none survive.
    """
    import random
    rng = random.Random(4)
    runs = 300
    raw = survived = 0
    for _ in range(runs):
        rows = [row(f"P{i}", rng.random(), rng.random()) for i in range(10)]
        _, tests = ll._fdr(rows)
        assert tests["n"] == 20 and tests["expected_false"] == pytest.approx(1.0)
        raw += tests["raw_hits"]
        survived += tests["survivors"]

    assert raw / runs == pytest.approx(1.0, abs=0.25)   # ≈ n × alpha
    assert survived / runs < 0.1                        # the correction works


def test_a_genuinely_strong_result_survives_the_correction():
    rows = [row("STRONG", 0.00001, 0.9)] + [row(f"P{i}", 0.4, 0.6) for i in range(9)]
    per_row, tests = ll._fdr(rows)

    assert per_row["STRONG"]["survives_fdr"] is True
    assert tests["survivors"] == 1
    assert per_row["P0"]["survives_fdr"] is False


def test_the_same_p_value_survives_alone_and_fails_in_a_crowd():
    """
    The whole point of the correction, in one test: significance is a property
    of the RUN, not of the pair.
    """
    alone = ll._fdr([row("A", 0.03, None)])[0]["A"]
    crowd = ll._fdr([row("A", 0.03, None)]
                    + [row(f"P{i}", 0.5, 0.5) for i in range(12)])[0]["A"]

    assert alone["survives_fdr"] is True
    assert crowd["survives_fdr"] is False
    assert crowd["q_best"] > alone["q_best"]


def test_both_directions_are_corrected_as_one_family():
    """
    They were run in the same sweep looking for whichever came out
    significant, which is exactly the situation the correction is for.
    """
    rows = [row("A", 0.01, 0.02), row("B", 0.03, 0.04)]
    _, tests = ll._fdr(rows)
    assert tests["n"] == 4


def test_each_row_reports_its_best_direction():
    per_row, _ = ll._fdr([row("A", 0.001, 0.5), row("B", 0.5, 0.002)])
    assert per_row["A"]["q_best"] == per_row["A"]["q_t_leads_s"]
    assert per_row["B"]["q_best"] == per_row["B"]["q_s_leads_t"]


def test_a_pair_with_no_usable_test_is_not_counted():
    """Granger returns nan when there is too little overlap; that is not a test."""
    per_row, tests = ll._fdr([row("A", 0.01, 0.02), row("DEAD", None, None)])

    assert tests["n"] == 2
    assert per_row["DEAD"] == {"q_t_leads_s": None, "q_s_leads_t": None,
                               "q_best": None, "survives_fdr": False}


def test_no_tests_at_all_is_reported_not_divided_by():
    per_row, tests = ll._fdr([row("DEAD", None, None)])
    assert tests["n"] == 0 and tests["survivors"] == 0
    assert per_row["DEAD"]["survives_fdr"] is False


# ── payload shaping ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "abc"])
def test_unusable_numbers_never_reach_the_client(value):
    """`NaN` is not JSON, and half_life is nan whenever a pair does not revert."""
    assert ll._n(value) is None


def test_lag_labels_say_which_way_round_they_go():
    """
    A signed lag column makes every reader re-derive the direction. These
    cannot be misread.
    """
    assert ll.lag_labels(3) == ["S 领先 3 天", "S 领先 2 天", "S 领先 1 天", "同日",
                                "T 领先 1 天", "T 领先 2 天", "T 领先 3 天"]


def test_the_label_list_lines_up_with_the_lag_list():
    for max_lag in (1, 5, 10):
        assert len(ll.lag_labels(max_lag)) == 2 * max_lag + 1


def test_the_stock_itself_is_never_its_own_peer(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "lead_lag_stats",
                        types.SimpleNamespace(fetch_qfq_returns=lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should have refused before fetching"))))
    with pytest.raises(LookupError, match="至少选择"):
        ll.analyse("600519", ["600519", "600519"])


def test_too_many_peers_is_refused_before_any_fetch(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "lead_lag_stats",
                        types.SimpleNamespace(fetch_qfq_returns=lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should have refused before fetching"))))
    with pytest.raises(LookupError, match="最多"):
        ll.analyse("600519", [f"{i:06d}" for i in range(ll.MAX_PEERS + 1)])
