"""
Building an allocation: what it refuses, and what it reports alongside.

Offline. The optimiser's own arithmetic is tested in test_optimise.py; what
is tested here is everything around it — refusing to mix two trading
calendars, surviving one unpriceable ticker, and never showing a curve
without the benchmark that makes it readable.

    python -m pytest api/tests/test_portfolio_build.py -q
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import portfolio_build as pb  # noqa: E402

N = 400
IDX = pd.bdate_range("2024-01-02", periods=N)


def walk(seed, vol=0.015, drift=0.0003):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, N))), index=IDX)


class FakeMarket:
    def __init__(self, frames):
        self.frames = frames

    def fetch_ohlcv(self, symbol):
        s = self.frames.get(symbol)
        return None if s is None else pd.DataFrame({"Close": s})


@pytest.fixture
def cn(monkeypatch):
    """Four A-share names, priced."""
    frames = {t: walk(i) for i, t in enumerate(
        ["600519", "000001", "600036", "300750"])}
    mod = types.ModuleType("markets")
    mod.canonical = lambda s: s
    mod.split = lambda s: ("US", s.split(":")[-1]) if s.startswith("US:") else ("CN", s)
    mod.get = lambda s: FakeMarket(frames)
    monkeypatch.setitem(sys.modules, "markets", mod)

    dm = types.ModuleType("data_manager")
    dm.get_all_stock_basic = lambda: [
        {"ticker": "600519", "name": "贵州茅台"},
        {"ticker": "000001", "name": "平安银行"},
        {"ticker": "600036", "name": "招商银行"},
        {"ticker": "300750", "name": "宁德时代"}]
    dm.get_index_data_live = lambda code, lookback_days=0: pd.DataFrame(
        {"Close": walk(99)})
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    return frames


# ── one market per portfolio ─────────────────────────────────────────────────
def test_a_single_market_is_fine(cn):
    assert pb.market_of(["600519", "000001"]) == "CN"


def test_mixing_two_markets_is_refused(cn):
    """
    Not a style rule. The calendars differ, so the intersection quietly drops
    every holiday on both sides and the covariance ends up measured on a
    sample neither market traded.
    """
    with pytest.raises(LookupError, match="同一个市场"):
        pb.market_of(["600519", "US:AAPL"])


def test_the_refusal_names_the_markets_involved(cn):
    with pytest.raises(LookupError) as e:
        pb.market_of(["600519", "US:AAPL"])
    assert "CN" in str(e.value) and "US" in str(e.value)


def test_an_empty_list_is_refused(cn):
    with pytest.raises(LookupError):
        pb.market_of([])


# ── loading prices ───────────────────────────────────────────────────────────
def test_prices_are_aligned_and_trimmed_to_the_lookback(cn):
    px = pb.load_closes(["600519", "000001"], 120)
    assert list(px.columns) == ["600519", "000001"]
    assert len(px) == 120


def test_one_unpriceable_ticker_does_not_empty_the_frame(monkeypatch, cn):
    """
    The failure this guards: an all-NaN column plus a row-wise dropna wipes
    every row, and the screen reports "0 sessions" for what is one bad
    ticker.
    """
    frames = dict(cn)
    frames["BROKEN"] = pd.Series(np.nan, index=IDX)
    mod = sys.modules["markets"]
    monkeypatch.setattr(mod, "get", lambda s: FakeMarket(frames))

    px = pb.load_closes(["600519", "000001", "BROKEN"], 200)
    assert "BROKEN" not in px.columns
    assert len(px) == 200


def test_a_ticker_with_no_data_at_all_is_reported_not_silently_dropped(
        monkeypatch, cn):
    mod = sys.modules["markets"]
    monkeypatch.setattr(mod, "get", lambda s: FakeMarket(cn))
    px = pb.load_closes(["600519", "000001", "999999"], 200)
    assert px.attrs["missing"] == ["999999"]


def test_too_little_overlap_is_refused_with_the_count(cn):
    frames = {"A": walk(1).iloc[-20:], "B": walk(2).iloc[-20:]}
    mod = sys.modules["markets"]
    mod.get = lambda s: FakeMarket(frames)
    with pytest.raises(LookupError, match="共同交易日"):
        pb.load_closes(["A", "B"], 242)


def test_everything_failing_is_a_fetch_error_not_a_lookup_error(cn):
    """
    Different failures, different meanings: "your tickers do not overlap" is
    the caller's problem, "the data source is down" is not.
    """
    mod = sys.modules["markets"]
    mod.get = lambda s: FakeMarket({})
    with pytest.raises(RuntimeError, match="行情"):
        pb.load_closes(["600519"], 242)


# ── returns ──────────────────────────────────────────────────────────────────
def test_daily_returns_by_default(cn):
    px = pb.load_closes(["600519", "000001"], 100)
    r = pb.returns_of(px)
    assert len(r) == len(px) - 1


def test_a_longer_duration_measures_over_that_many_sessions(cn):
    px = pb.load_closes(["600519", "000001"], 100)
    r5 = pb.returns_of(px, 5)
    expected = px["600519"].iloc[5] / px["600519"].iloc[0] - 1
    assert r5["600519"].iloc[0] == pytest.approx(expected)


# ── the build ────────────────────────────────────────────────────────────────
def test_a_build_returns_weights_that_sum_to_one(cn):
    out = pb.build(["600519", "000001", "600036", "300750"], method="min_var")
    total = sum(h["weight_pct"] for h in out["holdings"])
    assert total == pytest.approx(100.0, abs=0.5)


def test_every_holding_carries_its_name(cn):
    out = pb.build(["600519", "000001", "600036"], method="equal")
    names = {h["t"]: h["n"] for h in out["holdings"]}
    assert names["600519"] == "贵州茅台"


def test_the_cap_actually_applied_is_reported_not_the_one_asked_for(cn):
    """
    effective_cap never goes tighter than 2/n, so a 10% cap on three names
    is really 66%. Showing the asked-for number would make the optimiser
    look like it ignored the setting.
    """
    out = pb.build(["600519", "000001", "600036"], method="min_var", cap=0.10)
    assert out["cap_asked_pct"] == 10.0
    assert out["cap_pct"] > 10.0


def test_a_single_stock_is_refused(cn):
    with pytest.raises(LookupError, match="至少需要两只"):
        pb.build(["600519"], method="min_var")


def test_duplicates_collapse_rather_than_double_counting(cn):
    out = pb.build(["600519", "600519", "000001"], method="equal")
    assert len(out["holdings"]) == 2


def test_the_same_stock_twice_is_not_two_stocks(cn):
    """
    Without the de-duplication this passes the "at least two" check with a
    list of one name, and the optimiser dutifully returns 100% of it.
    """
    with pytest.raises(LookupError, match="至少需要两只"):
        pb.build(["600519", "600519"], method="equal")


@pytest.mark.parametrize("raw, expect", [
    # A dust weight is removed and its share is redistributed, not lost.
    ({"A": 0.500, "B": 0.497, "C": 0.003}, {"A": 0.5015, "B": 0.4985}),
    ({"A": 0.6, "B": 0.4}, {"A": 0.6, "B": 0.4}),          # nothing to prune
    ({"A": 0.002, "B": 0.003}, {}),                         # all dust
])
def test_pruning_dust_keeps_the_book_at_one(raw, expect):
    """
    A solver usually sends dust to exactly zero, so this path is unreachable
    from most fixtures — and matters on the day it is not. Without the
    renormalisation the book adds to 99.7% and every number built on it is
    scaled down by the difference.
    """
    out = pb.prune(pd.Series(raw))
    assert set(out.index) == set(expect)
    if len(out):
        assert out.sum() == pytest.approx(1.0)
    for k, v in expect.items():
        assert out[k] == pytest.approx(v, abs=1e-4)


def test_weights_still_sum_to_one_after_the_dust_filter(monkeypatch, cn):
    """
    Weights under MIN_WEIGHT are dropped as rounding artefacts. Dropping
    without renormalising leaves a book that adds up to 97%, and every
    number derived from it is quietly scaled down.
    """
    frames = dict(cn)
    # A name so volatile that min-variance gives it essentially nothing.
    frames["WILD"] = walk(77, vol=0.15)
    mod = sys.modules["markets"]
    monkeypatch.setattr(mod, "get", lambda s: FakeMarket(frames))

    out = pb.build(["600519", "000001", "600036", "300750", "WILD"],
                   method="min_var")
    wild = next(h for h in out["holdings"] if h["t"] == "WILD")
    assert wild["weight_pct"] < 0.5, "fixture did not produce a dust weight"
    assert sum(h["weight_pct"] for h in out["holdings"]) == pytest.approx(
        100.0, abs=0.2)


# ── what is drawn beside the result ──────────────────────────────────────────
def test_equal_weight_is_always_reported_next_to_the_optimised_one(cn):
    """
    An optimiser always produces a portfolio and in-sample it always looks
    good. Equal weight is the line that makes the result readable.
    """
    out = pb.build(["600519", "000001", "600036", "300750"], method="max_sharpe")
    assert out["equal_stats"]["ann_vol_pct"] > 0
    assert len(out["equal_curve"]) == len(out["curve"]) == len(out["dates"])


def test_the_benchmark_is_on_the_portfolio_s_own_dates(cn):
    out = pb.build(["600519", "000001", "600036"], method="min_var")
    assert out["benchmark"] is not None
    assert len(out["benchmark"]["curve"]) == len(out["dates"])


def test_a_missing_benchmark_is_null_rather_than_a_crash(monkeypatch, cn):
    dm = sys.modules["data_manager"]
    monkeypatch.setattr(dm, "get_index_data_live",
                        lambda *a, **kw: pd.DataFrame())
    out = pb.build(["600519", "000001"], method="equal")
    assert out["benchmark"] is None
    assert out["curve"], "the portfolio curve survived"


def test_the_frontier_is_a_curve_not_a_point(cn):
    out = pb.build(["600519", "000001", "600036", "300750"], method="min_var")
    f = out["frontier"]
    assert len(f) >= 5
    assert all(p["vol_pct"] > 0 for p in f)
    # Rising return should not come free of rising risk somewhere on it.
    assert max(p["ret_pct"] for p in f) > min(p["ret_pct"] for p in f)


def test_the_correlation_matrix_is_square_and_labelled(cn):
    out = pb.build(["600519", "000001", "600036"], method="equal")
    c = out["correlation"]
    assert len(c["labels"]) == 3
    assert all(len(row) == 3 for row in c["rows"])
    assert all(c["rows"][i][i] == 1.0 for i in range(3))


def test_the_curve_starts_at_zero_percent(cn):
    out = pb.build(["600519", "000001"], method="equal")
    assert out["curve"][0] == pytest.approx(0.0, abs=1e-6)


# ── the payload ──────────────────────────────────────────────────────────────
def test_the_payload_is_json_with_no_nan(cn):
    import json
    out = pb.build(["600519", "000001", "600036", "300750"], method="min_var")
    json.dumps(out, allow_nan=False)


def test_an_unknown_method_is_refused(cn):
    with pytest.raises(LookupError, match="未知的优化方法"):
        pb.build(["600519", "000001"], method="not_a_method")
