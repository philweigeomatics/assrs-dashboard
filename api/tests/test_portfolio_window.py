"""
The analysis window: regime legs, per-stock stats, and the optimiser's closes.

Offline.

    python -m pytest api/tests/test_portfolio_window.py -q
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
import portfolio_window as pw  # noqa: E402

IDX = pd.bdate_range("2022-01-03", periods=900)


def walk(seed, index=IDX, vol=0.015, drift=0.0003):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, len(index)))), index=index)


def path(*knots):
    """A straight-line price path through (session, price) points."""
    xs, ys = zip(*knots)
    return pd.Series(np.interp(range(xs[-1] + 1), xs, ys),
                     index=pd.bdate_range("2022-01-03", periods=xs[-1] + 1))


@pytest.fixture
def cn(monkeypatch):
    frames = {"600519": walk(1), "000001": walk(2), "600036": walk(3),
              "688981": walk(4).iloc[500:]}

    class Fake:
        def fetch_closes_since(self, code, start):
            s = frames.get(code)
            return None if s is None else s[s.index >= pd.Timestamp(start)]

        def fetch_ohlcv(self, code, years=3):
            s = frames.get(code)
            return None if s is None else pd.DataFrame({"Close": s})

    mod = types.ModuleType("markets")
    mod.canonical = lambda s: s
    mod.split = lambda s: ("US", s.split(":")[-1]) if s.startswith("US:") else ("CN", s)
    mod.get = lambda s: Fake()
    mod.parse = lambda s: (Fake(), mod.split(s)[1])
    monkeypatch.setitem(sys.modules, "markets", mod)

    dm = types.ModuleType("data_manager")
    dm.get_all_stock_basic = lambda: []
    dm.get_index_data_live = lambda *a, **k: pd.DataFrame({"Close": walk(9)})
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    monkeypatch.setattr(pb, "_sectors", lambda *a, **k: {}, raising=False)
    return frames


# ── regimes ──────────────────────────────────────────────────────────────────
def test_legs_alternate_between_the_real_highs_and_lows():
    s = path((0, 100), (100, 200), (200, 120), (300, 180))
    legs = pw.regimes(s, 0.20)
    assert [x["kind"] for x in legs] == ["up", "down", "up"]
    assert legs[0]["to"] == str(s.index[100].date())
    assert legs[1]["to"] == str(s.index[200].date())
    assert legs[0]["change_pct"] == pytest.approx(100, abs=0.1)
    assert legs[1]["change_pct"] == pytest.approx(-40, abs=0.1)


def test_a_turn_is_only_confirmed_once_price_has_reversed_by_the_threshold():
    s = path((0, 100), (100, 200), (200, 100))
    top = pw.regimes(s, 0.20)[0]
    # 20% off the 200 high is 160, reached 40% of the way down the fall.
    assert top["confirmed"] == str(s.index[140].date())


def test_the_last_leg_is_marked_as_unfinished():
    s = path((0, 100), (100, 200), (200, 120), (300, 180))
    legs = pw.regimes(s, 0.20)
    assert legs[-1]["open"] is True and legs[-1]["confirmed"] is None
    assert all(not x["open"] for x in legs[:-1])
    assert legs[-1]["to"] == str(s.index[-1].date())


def test_a_wider_threshold_ignores_the_smaller_swings():
    s = path((0, 100), (100, 200), (130, 165), (230, 260), (330, 120))
    assert len(pw.regimes(s, 0.15)) > len(pw.regimes(s, 0.30))


def test_a_market_that_never_reverses_is_one_leg():
    legs = pw.regimes(path((0, 100), (300, 110)), 0.20)
    assert len(legs) == 1 and legs[0]["kind"] == "up" and legs[0]["open"]


def test_legs_join_up_with_no_gaps():
    s = walk(7, vol=0.03)
    legs = pw.regimes(s, 0.20)
    assert len(legs) > 2
    assert legs[0]["from"] == str(s.index[0].date())
    assert legs[-1]["to"] == str(s.index[-1].date())
    for a, b in zip(legs, legs[1:]):
        assert a["to"] == b["from"] and a["kind"] != b["kind"]


# ── one stock, one window ────────────────────────────────────────────────────
def test_window_stats_use_only_the_days_inside_the_window():
    s = walk(5)
    a, b = str(IDX[100].date()), str(IDX[400].date())
    got = pw.window_stats(s, a, b, 0.03, None)
    assert got["days"] == 300 and got["from"] == a
    assert got["change_pct"] == pytest.approx((s.iloc[400] / s.iloc[100] - 1) * 100, abs=0.01)
    assert got["partial"] is False


def test_window_stats_match_the_optimisers_arithmetic():
    s = walk(5)
    a, b = str(IDX[100].date()), str(IDX[400].date())
    daily = s.loc[a:b].pct_change().dropna().to_frame("x")
    want = pb.stats_of(daily, pd.Series({"x": 1.0}), 0.03)
    got = pw.window_stats(s, a, b, 0.03, None)
    for key in ("ann_return_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct"):
        assert got[key] == want[key]


def test_a_suspended_day_counts_as_a_flat_day_like_the_optimiser():
    s = walk(5)
    suspended = s.drop(IDX[150:170])
    a, b = str(IDX[100].date()), str(IDX[400].date())
    cal = IDX[100:401]
    got = pw.window_stats(suspended, a, b, 0.03, None, cal)
    assert got["days"] == 300
    held = suspended.reindex(IDX).ffill().loc[a:b]
    want = pb.stats_of(held.pct_change().dropna().to_frame("x"), pd.Series({"x": 1.0}), 0.03)
    assert got["ann_return_pct"] == want["ann_return_pct"]


def test_beta_of_a_stock_moving_twice_the_index_is_two():
    bench = walk(6).pct_change().dropna()
    assert pw.beta_of(bench * 2, bench) == pytest.approx(2.0, abs=0.001)


def test_too_few_shared_days_gives_no_beta_rather_than_a_noisy_one():
    bench = walk(6).pct_change().dropna().iloc[:5]
    assert pw.beta_of(bench * 2, bench) is None


def test_a_stock_listed_inside_the_window_is_flagged():
    s = walk(5).iloc[300:]
    got = pw.window_stats(s, str(IDX[100].date()), str(IDX[600].date()), 0.03, None)
    assert got["partial"] is True and got["from"] == str(IDX[300].date())


def test_a_stock_with_nothing_in_the_window_is_empty_not_an_error():
    s = walk(5).iloc[700:]
    got = pw.window_stats(s, str(IDX[100].date()), str(IDX[400].date()), 0.03, None)
    assert got["days"] == 0 and got["change_pct"] is None


def test_stats_covers_the_window_and_every_leg(cn):
    index = walk(7, vol=0.03)
    a, b = str(IDX[100].date()), str(IDX[400].date())
    out = pw.stats(["600519", "688981"], a, b, index_close=index, code="000300.SH",
                   pct=0.20, history=pw.fetch_history)
    assert [r["t"] for r in out["rows"]] == ["600519", "688981"]
    assert out["index"]["label"] == "沪深300" and out["index"]["beta"] == pytest.approx(1.0)
    assert out["sessions"] == 300
    assert len(out["regimes"]) == len(pw.regimes(index, 0.20))
    assert out["regimes"][0]["open"] is True            # newest first
    assert len(out["cells"]["600519"]) == len(out["regimes"])


def test_a_stock_not_listed_during_a_leg_has_no_number_for_it(cn):
    index = walk(7, vol=0.03)
    out = pw.stats(["688981"], str(IDX[600].date()), str(IDX[800].date()),
                   index_close=index, code="000300.SH", history=pw.fetch_history)
    listed = IDX[500]
    for leg, cell in zip(out["regimes"], out["cells"]["688981"]):
        if pd.Timestamp(leg["from"]) < listed - pd.Timedelta(days=pw.LATE_START_DAYS):
            assert cell is None


def test_a_stock_that_cannot_be_read_is_named(cn):
    out = pw.stats(["600519", "999999"], str(IDX[100].date()), str(IDX[400].date()),
                   index_close=walk(7), code="000300.SH", history=pw.fetch_history)
    assert out["missing"] == ["999999"]


def test_us_stocks_are_refused_for_now(cn):
    with pytest.raises(LookupError):
        pw.stats(["US:AAPL", "US:MSFT"], "2023-01-01", "2024-01-01",
                 index_close=walk(7), code="000300.SH", history=pw.fetch_history)


def test_only_the_listed_indices_can_be_asked_for():
    with pytest.raises(LookupError):
        pw.index_history("000905.SH")


# ── correlation inside the window ────────────────────────────────────────────
def corr_of(c, a, b):
    return c["rows"][c["labels"].index(a)][c["labels"].index(b)]


def test_correlation_is_measured_only_inside_the_window():
    # The second name copies the first for 400 sessions, then mirrors it.
    r = walk(11).pct_change().fillna(0)
    flip = pd.Series(np.where(np.arange(len(IDX)) < 400, 1, -1), index=IDX)
    a = 100 * (1 + r).cumprod()
    b = 100 * (1 + r * flip).cumprod()
    frames = {"a": a, "b": b}
    assert corr_of(pw.window_correlation(frames, IDX[50:350]), "a", "b") == pytest.approx(1)
    assert corr_of(pw.window_correlation(frames, IDX[450:800]), "a", "b") == pytest.approx(-1)


def test_correlation_matches_the_optimiser_when_every_name_is_listed(cn):
    a, b = str(IDX[200].date()), str(IDX[600].date())
    syms = ["600519", "000001", "600036"]
    d = pb.build(syms, max_weight=0.6, mode="equal", start=a, end=b)
    got = pw.stats(syms, a, b, index_close=walk(7), code="000300.SH",
                   history=pw.fetch_history)["correlation"]
    assert set(got["labels"]) == set(syms)
    for x in syms:
        for y in syms:
            assert corr_of(got, x, y) == pytest.approx(corr_of(d["correlation"], x, y), abs=1e-3)


def test_a_late_listing_only_blanks_its_own_pairs():
    frames = {"old1": walk(1), "old2": walk(2), "new": walk(3).iloc[340:]}
    c = pw.window_correlation(frames, IDX[100:350])     # 10 shared days with "new"
    assert corr_of(c, "new", "old1") is None and corr_of(c, "old2", "new") is None
    assert corr_of(c, "old1", "old2") is not None


def test_a_late_listing_with_enough_days_uses_the_days_it_shares():
    frames = {"old": walk(1), "new": walk(3).iloc[300:]}
    c = pw.window_correlation(frames, IDX[100:400])
    daily = pd.DataFrame(frames).loc[IDX[300]:IDX[399]].pct_change().iloc[1:]
    assert corr_of(c, "old", "new") == pytest.approx(daily["old"].corr(daily["new"]), abs=1e-3)


def test_a_suspended_day_is_a_flat_day_for_correlation_too():
    frames = {"held": walk(1).drop(IDX[150:170]), "other": walk(2)}
    c = pw.window_correlation(frames, IDX[100:401])
    held = frames["held"].reindex(IDX).ffill().loc[IDX[100]:IDX[400]].pct_change().iloc[1:]
    other = frames["other"].loc[IDX[100]:IDX[400]].pct_change().iloc[1:]
    assert corr_of(c, "held", "other") == pytest.approx(held.corr(other), abs=1e-3)


def test_a_name_with_no_prices_in_the_window_has_no_correlation():
    frames = {"old": walk(1), "gone": walk(2).iloc[700:]}
    c = pw.window_correlation(frames, IDX[100:400])
    assert corr_of(c, "old", "gone") is None and corr_of(c, "gone", "gone") is None


# ── the optimiser on a window ────────────────────────────────────────────────
def test_the_window_sets_the_optimisers_dates(cn):
    a, b = str(IDX[100].date()), str(IDX[400].date())
    px = pw.window_closes(["600519", "000001"], a, b)
    assert str(px.index[0].date()) == a and str(px.index[-1].date()) == b


def test_a_window_too_short_for_a_covariance_is_refused(cn):
    with pytest.raises(LookupError, match="共同交易日"):
        pw.window_closes(["600519", "000001"], str(IDX[100].date()), str(IDX[130].date()))


def test_a_late_listing_moves_the_start_and_says_which(cn):
    px = pw.window_closes(["600519", "688981"], str(IDX[100].date()), str(IDX[800].date()))
    assert str(px.index[0].date()) == str(IDX[500].date())
    assert px.attrs["limited_by"] == ("688981", str(IDX[500].date()))


def test_build_and_weigh_price_the_same_window(cn):
    a, b = str(IDX[200].date()), str(IDX[600].date())
    syms = ["600519", "000001", "600036"]
    d = pb.build(syms, max_weight=0.6, mode="equal", start=a, end=b)
    assert d["window"] is True and d["from"] == a and d["to"] == b
    assert d["lookback"] == 400
    out = pb.weigh(syms, {h["t"]: h["weight_pct"] for h in d["holdings"]},
                   start=d["from"], end=d["to"])
    assert out["dates"] == d["dates"]
    assert out["curve"][-1] == pytest.approx(d["curve"][-1], abs=0.05)


def test_build_without_a_window_still_uses_the_lookback(cn):
    d = pb.build(["600519", "000001", "600036"], max_weight=0.6, lookback=120)
    assert d["window"] is False and d["lookback"] == 120


def test_each_holding_carries_its_own_stats_for_the_window(cn):
    a, b = str(IDX[200].date()), str(IDX[600].date())
    d = pb.build(["600519", "000001", "600036"], max_weight=0.6, mode="equal", start=a, end=b)
    solo = next(h for h in d["holdings"] if h["t"] == "600519")["solo"]
    want = pw.window_stats(cn["600519"], a, b, 0.03, None)
    assert solo["ann_vol_pct"] == want["ann_vol_pct"]
    assert solo["sharpe"] == want["sharpe"]
