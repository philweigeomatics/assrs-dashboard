"""
sector_affinity.analyse on synthetic series, where the right answer is known.

Offline — no database, no Tushare. Builds sectors with a deliberate structure
(one the stock tracks, one it mirrors, one unrelated, and a benchmark it also
tracks) and checks the panel reports that structure rather than noise.

    python -m pytest api/tests/test_sector_affinity.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import sector_affinity as sa  # noqa: E402

N = 400
DATES = pd.bdate_range("2024-01-02", periods=N)


def _price(returns: np.ndarray) -> pd.Series:
    return pd.Series(100 * np.cumprod(1 + returns), index=DATES)


@pytest.fixture()
def world():
    rng = np.random.default_rng(7)
    stock = rng.normal(0, 0.02, N)
    noise = lambda s: rng.normal(0, s, N)  # noqa: E731

    rets = {
        "跟随板块": stock + noise(0.004),        # nearly the same series
        "镜像板块": -stock + noise(0.004),       # moves opposite
        "无关板块": rng.normal(0, 0.02, N),      # independent
        sa.CSI300_LABEL: stock + noise(0.006),   # market, also tracks the stock
    }
    return _price(stock), {k: pd.Series(v, index=DATES) for k, v in rets.items()}


def test_reports_the_structure_it_was_given(world):
    close, rets = world
    out = sa.analyse(close, rets, window=20)

    by = {s["name"]: s for s in out["sectors"]}
    assert by["跟随板块"]["r"] > 0.8
    assert by["镜像板块"]["r"] < -0.8
    assert abs(by["无关板块"]["r"]) < 0.5

    # Sorted by current correlation, highest first.
    rs = [s["r"] for s in out["sectors"]]
    assert rs == sorted(rs, reverse=True)

    # Every series is as long as the date axis, or the chart draws crooked.
    assert all(len(s["series"]) == len(out["dates"]) for s in out["sectors"])


def test_benchmark_never_competes_as_a_sector(world):
    """沪深300 is drawn, but it must not lead, rank, or count as a theme."""
    close, rets = world
    out = sa.analyse(close, rets, window=20)

    bench = next(s for s in out["sectors"] if s["name"] == sa.CSI300_LABEL)
    assert bench["benchmark"] is True
    assert bench["top"] is False, "the market must not take a sector line slot"
    assert bench["r"] is not None, "but it is still measured and drawn"

    assert out["summary"]["top"] != sa.CSI300_LABEL
    assert all(r["sector"] != sa.CSI300_LABEL for r in out["dominant"])
    assert all(l["sector"] != sa.CSI300_LABEL for l in out["summary"]["leaders"])


@pytest.fixture()
def rotating():
    """
    A stock that changes theme halfway: 前半场 tracks it for the first half of
    the history, 后半场 for the second. Anything checking the rotation strip
    needs a series that actually rotates — with one unbroken run, an
    off-by-one at a run boundary has no boundary to show up at.
    """
    rng = np.random.default_rng(11)
    stock = rng.normal(0, 0.02, N)
    half = N // 2
    first, second = rng.normal(0, 0.02, N), rng.normal(0, 0.02, N)
    first[:half] = stock[:half] + rng.normal(0, 0.003, half)
    second[half:] = stock[half:] + rng.normal(0, 0.003, N - half)

    return _price(stock), {
        "前半场": pd.Series(first, index=DATES),
        "后半场": pd.Series(second, index=DATES),
        "无关板块": pd.Series(rng.normal(0, 0.02, N), index=DATES),
    }


def test_dominant_runs_tile_the_axis(rotating):
    """The strip must cover every bar exactly once, with no repeats."""
    close, rets = rotating
    out = sa.analyse(close, rets, window=20)
    runs, n = out["dominant"], len(out["dates"])

    assert len(runs) >= 2, "this fixture rotates; a single run tests nothing"
    assert runs[0]["from"] == 0
    assert runs[-1]["to"] == n - 1
    for a, b in zip(runs, runs[1:]):
        assert b["from"] == a["to"] + 1, "a gap or overlap between runs"
        assert a["sector"] != b["sector"], "adjacent runs must be different sectors"

    # The rotation count IS the number of switches.
    assert out["summary"]["rotations"] == len(runs) - 1

    # Every bar is claimed by exactly one run.
    covered = [i for r in runs for i in range(r["from"], r["to"] + 1)]
    assert covered == list(range(n))


def test_the_leader_changes_when_the_stock_changes_theme(rotating):
    """前半场 must lead early and 后半场 late — not the other way round."""
    close, rets = rotating
    out = sa.analyse(close, rets, window=20)

    leader_at = {}
    for r in out["dominant"]:
        for i in range(r["from"], r["to"] + 1):
            leader_at[i] = r["sector"]

    n = len(out["dates"])
    assert leader_at[int(n * 0.15)] == "前半场"
    assert leader_at[n - 1] == "后半场"
    assert {l["sector"] for l in out["summary"]["leaders"]} >= {"前半场", "后半场"}


def test_current_r_is_the_latest_value_not_the_window_average(rotating):
    """
    后半场 is uncorrelated early and near-perfect late. Its CURRENT r must
    read near 1 while its window mean stays far below — reporting the mean in
    the bar chart would understate exactly the sector that matters now.
    """
    close, rets = rotating
    out = sa.analyse(close, rets, window=20)
    by = {s["name"]: s for s in out["sectors"]}

    # The window shown is the last 252 of 400 bars, so about a fifth of it
    # predates the switch: a mean around 0.8 against a current value near 1.
    assert by["后半场"]["r"] > 0.9
    assert by["后半场"]["mean_r"] < 0.85
    assert by["后半场"]["r"] - by["后半场"]["mean_r"] > 0.1

    # And the reverse for the sector it has left behind.
    assert by["前半场"]["r"] < by["前半场"]["mean_r"]


def test_a_stock_glued_to_one_sector_does_not_rotate(world):
    """跟随板块 tracks it so closely that nothing else should ever lead."""
    close, rets = world
    out = sa.analyse(close, rets, window=60)

    assert out["summary"]["top"] == "跟随板块"
    assert out["summary"]["rotations"] == 0
    assert out["summary"]["verdict"] == "low"
    assert out["summary"]["n_leaders"] == 1


def test_membership_is_reported_and_self_indexing_is_flagged(world):
    close, rets = world
    out = sa.analyse(close, rets, window=20, ticker="000001",
                     members={"跟随板块": {"000001"}, "无关板块": {"999999"}})

    by = {s["name"]: s for s in out["sectors"]}
    assert by["跟随板块"]["member"] is True
    assert by["无关板块"]["member"] is False
    assert out["summary"]["top_is_member"] is True
    # r ≈ 0.98 against an index it belongs to — arithmetic, not a discovery.
    assert out["summary"]["self_index"] is True


def test_membership_absent_by_default(world):
    close, rets = world
    out = sa.analyse(close, rets, window=20)
    assert all(s["member"] is False for s in out["sectors"])
    assert out["summary"]["self_index"] is False


def test_rejects_windows_it_cannot_compute(world):
    close, rets = world
    with pytest.raises(LookupError):
        sa.analyse(close, rets, window=7)
    with pytest.raises(LookupError):
        sa.analyse(close, {}, window=20)
    with pytest.raises(LookupError):
        sa.analyse(close, {sa.CSI300_LABEL: rets[sa.CSI300_LABEL]}, window=20)


def test_too_short_an_overlap_is_skipped_not_faked(world):
    """A sector with almost no overlapping history must not appear at all."""
    close, rets = world
    rets = dict(rets)
    rets["新板块"] = rets["无关板块"].tail(8)

    out = sa.analyse(close, rets, window=20)
    assert "新板块" not in {s["name"] for s in out["sectors"]}
