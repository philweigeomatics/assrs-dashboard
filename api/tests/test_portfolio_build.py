"""
The ported optimiser: max Sharpe, minimum variance at a target, and the
assessment that grades the result.

Offline. The thresholds being checked are the Streamlit page's, deliberately
— the one thing a port must not do is make the same portfolio score
differently in the two apps.

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

    def fetch_ohlcv(self, code, years=3):
        # Keyed by the BARE code, as a real adapter is.
        s = self.frames.get(code)
        return None if s is None else pd.DataFrame({"Close": s})


@pytest.fixture
def cn(monkeypatch):
    frames = {t: walk(i) for i, t in enumerate(
        ["600519", "000001", "600036", "300750"])}
    mod = types.ModuleType("markets")
    mod.canonical = lambda s: s
    mod.split = lambda s: ("US", s.split(":")[-1]) if s.startswith("US:") else ("CN", s)
    mod.get = lambda s: FakeMarket(frames)
    mod.parse = lambda s: (FakeMarket(frames), mod.split(s)[1])
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


def moments_of(frames, cols, duration=1):
    px = pd.concat([frames[c].rename(c) for c in cols], axis=1).dropna()
    return pb.moments(px, duration)


# ── weights come off the frontier ────────────────────────────────────────────
def test_no_target_maximises_sharpe(cn):
    """The page's default: target_return None means maximise Sharpe."""
    _r, mu, cov, _d = moments_of(cn, list(cn))
    w = pb.optimise_weights(mu, cov, target_return=None, max_weight=1.0, rf=0.03)

    best = (pb.performance(w, mu.to_numpy(), cov.to_numpy())[0] - 0.03) \
        / pb.performance(w, mu.to_numpy(), cov.to_numpy())[1]
    # No random weight vector should beat it by a meaningful margin.
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = rng.random(len(mu))
        c = c / c.sum()
        r, v = pb.performance(c, mu.to_numpy(), cov.to_numpy())
        assert (r - 0.03) / v <= best + 1e-6


def test_a_target_return_is_actually_hit(cn):
    """
    The whole point of the frontier being a menu: asking for a return has to
    produce weights that reach it, not weights that merely look sensible.
    """
    _r, mu, cov, _d = moments_of(cn, list(cn))
    lo = pb.performance(pb.min_variance(mu, cov, max_weight=1.0),
                        mu.to_numpy(), cov.to_numpy())[0]
    target = (lo + float(mu.max())) / 2

    w = pb.optimise_weights(mu, cov, target_return=target, max_weight=1.0)
    got, _vol = pb.performance(w, mu.to_numpy(), cov.to_numpy())
    assert got == pytest.approx(target, abs=1e-4)


def test_a_target_gives_the_least_variance_that_reaches_it(cn):
    _r, mu, cov, _d = moments_of(cn, list(cn))
    lo = pb.performance(pb.min_variance(mu, cov, max_weight=1.0),
                        mu.to_numpy(), cov.to_numpy())[0]
    target = (lo + float(mu.max())) / 2
    w = pb.optimise_weights(mu, cov, target_return=target, max_weight=1.0)
    _r2, vol = pb.performance(w, mu.to_numpy(), cov.to_numpy())

    rng = np.random.default_rng(1)
    for _ in range(300):
        c = rng.random(len(mu))
        c = c / c.sum()
        r, v = pb.performance(c, mu.to_numpy(), cov.to_numpy())
        if abs(r - target) < 1e-3:
            assert v >= vol - 1e-6


def test_asking_for_more_return_than_exists_is_refused(cn):
    _r, mu, cov, _d = moments_of(cn, list(cn))
    with pytest.raises(LookupError, match="优化失败"):
        pb.optimise_weights(mu, cov, target_return=float(mu.max()) * 10,
                            max_weight=0.3)


def test_the_cap_is_respected(cn):
    _r, mu, cov, _d = moments_of(cn, list(cn))
    w = pb.optimise_weights(mu, cov, target_return=None, max_weight=0.30)
    assert w.max() <= 0.30 + 1e-6


# ── the frontier ─────────────────────────────────────────────────────────────
def test_the_frontier_starts_at_minimum_variance_not_the_worst_stock(cn):
    """
    Below min-variance the parabola bends back: less return for no less
    risk. Starting at the worst stock's mean draws that inefficient tail and
    invites picking from it.
    """
    _r, mu, cov, _d = moments_of(cn, list(cn))
    pts = pb.efficient_frontier(mu, cov, points=12, max_weight=1.0)
    assert len(pts) >= 5

    mv_ret, mv_vol = pb.performance(pb.min_variance(mu, cov, max_weight=1.0),
                                    mu.to_numpy(), cov.to_numpy())
    assert min(p["ret_pct"] for p in pts) >= mv_ret * 100 - 0.5
    # Nothing on it is less volatile than the min-variance portfolio.
    assert min(p["vol_pct"] for p in pts) >= mv_vol * 100 - 0.5


def test_the_frontier_only_goes_up(cn):
    """Risk buys return along it — that is what makes it efficient."""
    _r, mu, cov, _d = moments_of(cn, list(cn))
    pts = pb.efficient_frontier(mu, cov, points=14, max_weight=1.0)
    vols = [p["vol_pct"] for p in pts]
    rets = [p["ret_pct"] for p in pts]
    assert rets == sorted(rets)
    assert vols[-1] >= vols[0]


def test_each_frontier_point_carries_the_target_that_produced_it(cn):
    """The UI re-solves from this, so it has to come back."""
    _r, mu, cov, _d = moments_of(cn, list(cn))
    pts = pb.efficient_frontier(mu, cov, points=8, max_weight=1.0)
    assert all("target" in p for p in pts)

    p = pts[len(pts) // 2]
    w = pb.optimise_weights(mu, cov, target_return=p["target"], max_weight=1.0)
    got = pb.performance(w, mu.to_numpy(), cov.to_numpy())[0]
    assert got * 100 == pytest.approx(p["ret_pct"], abs=0.05)


# ── the risk metrics ─────────────────────────────────────────────────────────
def test_effective_bets_counts_independent_positions():
    assert pb.effective_bets(np.array([0.25] * 4)) == pytest.approx(4.0)
    assert pb.effective_bets(np.array([1.0])) == pytest.approx(1.0)
    # Concentrated books have fewer effective bets than names.
    assert pb.effective_bets(np.array([0.9, 0.05, 0.05])) < 1.3


def test_diversification_ratio_is_one_for_a_single_asset():
    cov = pd.DataFrame([[0.04]])
    assert pb.diversification_ratio(np.array([1.0]), cov.to_numpy()) == \
        pytest.approx(1.0)


def test_diversification_ratio_exceeds_one_when_assets_are_uncorrelated():
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    assert pb.diversification_ratio(np.array([0.5, 0.5]), cov) > 1.3


def test_var_is_the_percentile_and_cvar_is_worse_than_it():
    rng = np.random.default_rng(3)
    daily = pd.DataFrame({"A": rng.normal(0, 0.02, 500)})
    w = np.array([1.0])
    var = pb.var_at(daily, w, 0.95)
    cvar = pb.cvar_at(daily, w, 0.95)
    assert var < 0 and cvar < var, "CVaR must be the deeper loss"


def test_a_symmetric_series_has_a_tail_ratio_near_one():
    rng = np.random.default_rng(4)
    daily = pd.DataFrame({"A": rng.normal(0, 0.02, 4000)})
    assert pb.tail_ratio(daily, np.array([1.0]), 0.95) == pytest.approx(1.0, abs=0.12)


def test_worst_days_returns_the_worst_and_the_average_of_the_worst_n():
    daily = pd.DataFrame({"A": [-0.10, -0.05, -0.04, 0.01, 0.02]})
    worst, avg5 = pb.worst_days(daily, np.array([1.0]), 3)
    assert worst == pytest.approx(-0.10)
    assert avg5 == pytest.approx((-0.10 - 0.05 - 0.04) / 3)


# ── the assessment, at the page's thresholds ─────────────────────────────────
def metrics(**over):
    base = {"enb": 4.0, "div_ratio": 1.35, "tail_95": 1.3, "tail_99": 1.3,
            "worst_day_pct": -2.0, "avg_worst5_pct": -1.5}
    return {**base, **over}


def test_a_good_portfolio_scores_full_marks():
    a = pb.assess(1.8, metrics(), n_assets=5)
    assert a["score"] == 100
    assert a["tone"] == "good"
    assert a["warnings"] == []


def test_each_dimension_is_worth_what_the_page_gave_it():
    """20 / 20 / 15 / 25 / 20 — dropping one band must cost exactly that."""
    full = pb.assess(1.8, metrics(), n_assets=5)["score"]
    assert full - pb.assess(0.2, metrics(), n_assets=5)["score"] == 15    # 20→5
    assert full - pb.assess(1.8, metrics(enb=1.0), n_assets=5)["score"] == 15
    assert full - pb.assess(1.8, metrics(div_ratio=0.9), n_assets=5)["score"] == 15
    assert full - pb.assess(1.8, metrics(tail_95=0.7, tail_99=0.7),
                            n_assets=5)["score"] == 20
    assert full - pb.assess(1.8, metrics(worst_day_pct=-12.0,
                                         avg_worst5_pct=-9.0),
                            n_assets=5)["score"] == 20


@pytest.mark.parametrize("score_at, tone", [
    (1.8, "good"),        # 100
    (0.2, "warn"),        # 85 → still good? see below
])
def test_the_verdict_follows_the_score(score_at, tone):
    a = pb.assess(score_at, metrics(), n_assets=5)
    assert (a["tone"] == "good") == (a["score"] >= 80)
    assert (a["tone"] == "bad") == (a["score"] < 40)


def test_a_weak_portfolio_is_called_weak():
    a = pb.assess(0.1, metrics(enb=1.1, div_ratio=0.95, tail_95=0.5,
                               tail_99=0.5, worst_day_pct=-12.0,
                               avg_worst5_pct=-9.0), n_assets=6)
    assert a["score"] < 40 and a["tone"] == "bad"
    assert len(a["warnings"]) >= 4


def test_the_reasons_name_the_numbers_that_produced_them():
    a = pb.assess(0.3, metrics(), n_assets=5)
    assert any("0.30" in w for w in a["warnings"]), a["warnings"]


# ── the build ────────────────────────────────────────────────────────────────
def test_a_build_returns_a_full_result(cn):
    out = pb.build(list(cn), max_weight=0.30)
    assert out["mode"] == "max_sharpe"
    assert sum(h["weight_pct"] for h in out["holdings"]) == pytest.approx(100, abs=0.5)
    for key in ("frontier", "risk", "assessment", "correlation", "curve",
                "equal_curve", "benchmark", "opt"):
        assert key in out, key


def test_building_at_a_target_reports_the_target(cn):
    out0 = pb.build(list(cn), max_weight=1.0)
    target = out0["frontier"][len(out0["frontier"]) // 2]["target"]
    out = pb.build(list(cn), target_return=target, max_weight=1.0)

    assert out["mode"] == "target"
    assert out["opt"]["ann_return_pct"] == pytest.approx(target * 100, abs=0.2)


def test_a_higher_target_gives_a_more_concentrated_book(cn):
    """Chasing return along the frontier costs diversification — it should show."""
    out0 = pb.build(list(cn), max_weight=1.0)
    f = out0["frontier"]
    low = pb.build(list(cn), target_return=f[0]["target"], max_weight=1.0)
    high = pb.build(list(cn), target_return=f[-1]["target"], max_weight=1.0)
    assert high["risk"]["enb"] <= low["risk"]["enb"] + 1e-6


def test_the_optimum_sits_on_the_frontier(cn):
    """
    The dot is drawn on the curve, so it has to be computed from the same
    annualised moments rather than from the realised daily series.
    """
    out = pb.build(list(cn), max_weight=1.0)
    pts = out["frontier"]
    vol = out["opt"]["ann_vol_pct"]
    near = min(pts, key=lambda p: abs(p["vol_pct"] - vol))
    assert out["opt"]["ann_return_pct"] <= near["ret_pct"] + 0.75


def test_daily_metrics_stay_daily_when_duration_is_longer(cn):
    """
    Overlapping N-day returns are right for the covariance and wrong for VaR.
    The page documents this; losing it inflates every risk number.
    """
    one = pb.build(list(cn), max_weight=1.0, duration=1)
    five = pb.build(list(cn), max_weight=1.0, duration=5)
    assert one["lookback"] == five["lookback"]
    assert len(one["curve"]) == len(five["curve"])


def test_names_and_missing_tickers_are_reported(monkeypatch, cn):
    frames = dict(cn)
    mod = sys.modules["markets"]
    monkeypatch.setattr(mod, "parse",
                        lambda s: (FakeMarket(frames), mod.split(s)[1]))
    out = pb.build([*cn, "999999"], max_weight=0.5)
    assert out["missing"] == ["999999"]
    assert {h["t"]: h["n"] for h in out["holdings"]}["600519"] == "贵州茅台"


def test_mixing_markets_is_refused(cn):
    with pytest.raises(LookupError, match="同一个市场"):
        pb.build(["600519", "US:AAPL"])


def test_the_same_stock_twice_is_not_two_stocks(cn):
    with pytest.raises(LookupError, match="至少需要两只"):
        pb.build(["600519", "600519"])


def test_one_unpriceable_ticker_does_not_empty_the_frame(monkeypatch, cn):
    frames = dict(cn)
    frames["BROKEN"] = pd.Series(np.nan, index=IDX)
    mod = sys.modules["markets"]
    monkeypatch.setattr(mod, "parse",
                        lambda s: (FakeMarket(frames), mod.split(s)[1]))
    px = pb.load_closes([*cn, "BROKEN"], 200)
    assert "BROKEN" not in px.columns and len(px) == 201


def test_too_little_overlap_is_refused(cn):
    frames = {"A": walk(1).iloc[-20:], "B": walk(2).iloc[-20:]}
    mod = sys.modules["markets"]
    mod.parse = lambda s: (FakeMarket(frames), mod.split(s)[1])
    with pytest.raises(LookupError, match="共同交易日"):
        pb.load_closes(["A", "B"], 242)


def test_everything_failing_is_a_fetch_error(cn):
    mod = sys.modules["markets"]
    mod.parse = lambda s: (FakeMarket({}), mod.split(s)[1])
    with pytest.raises(RuntimeError, match="行情"):
        pb.load_closes(["600519"], 242)


@pytest.mark.parametrize("raw, expect", [
    ({"A": 0.500, "B": 0.497, "C": 0.003}, {"A": 0.5015, "B": 0.4985}),
    ({"A": 0.6, "B": 0.4}, {"A": 0.6, "B": 0.4}),
    ({"A": 0.002, "B": 0.003}, {}),
])
def test_pruning_dust_keeps_the_book_at_one(raw, expect):
    out = pb.prune(pd.Series(raw))
    assert set(out.index) == set(expect)
    if len(out):
        assert out.sum() == pytest.approx(1.0)
    for k, v in expect.items():
        assert out[k] == pytest.approx(v, abs=1e-4)


def test_equal_weight_is_always_beside_the_result(cn):
    out = pb.build(list(cn), max_weight=1.0)
    assert out["equal_stats"]["ann_vol_pct"] > 0
    assert len(out["equal_curve"]) == len(out["curve"]) == len(out["dates"])


def test_the_benchmark_is_on_the_portfolio_s_dates(cn):
    out = pb.build(list(cn), max_weight=1.0)
    assert out["benchmark"] is not None
    assert len(out["benchmark"]["curve"]) == len(out["dates"])


def test_a_missing_benchmark_is_null_not_a_crash(monkeypatch, cn):
    monkeypatch.setattr(sys.modules["data_manager"], "get_index_data_live",
                        lambda *a, **kw: pd.DataFrame())
    out = pb.build(list(cn), max_weight=1.0)
    assert out["benchmark"] is None and out["curve"]


def test_all_three_lines_share_one_origin(cn):
    """
    The benchmark is rebased to its own first close. Without a leading zero
    on the portfolio the two would start a day apart, and every gap read off
    the chart would be wrong by that day.
    """
    out = pb.build(list(cn), max_weight=1.0)
    assert out["curve"][0] == pytest.approx(0.0, abs=1e-6)
    assert out["equal_curve"][0] == pytest.approx(0.0, abs=1e-6)
    assert out["benchmark"]["curve"][0] == pytest.approx(0.0, abs=1e-6)
    n = len(out["dates"])
    assert len(out["curve"]) == len(out["equal_curve"]) == n
    assert len(out["benchmark"]["curve"]) == n


def test_annualisation_is_the_page_s_242_not_252():
    """Changing it moves every Sharpe by 2% with nothing on screen to explain it."""
    assert pb.TRADING_DAYS == 242


def test_the_payload_is_json_with_no_nan(cn):
    import json
    json.dumps(pb.build(list(cn), max_weight=0.3), allow_nan=False)


# ── the frontier's top end ───────────────────────────────────────────────────
def test_reachable_return_is_the_cap_filled_greedily():
    """Three names at the 30% cap plus 10% of the fourth — not the best stock."""
    mu = pd.Series([0.80, 0.40, 0.30, 0.20, 0.10, -0.20],
                   index=list("abcdef"))
    got = pb.reachable_return(mu, max_weight=0.30)
    want = 0.30 * (0.80 + 0.40 + 0.30) + 0.10 * 0.20
    assert got == pytest.approx(want)
    assert got < float(mu.max())


def test_reachable_return_respects_a_weight_floor():
    """A floor forces budget into the laggards, so the ceiling comes down."""
    mu = pd.Series([0.80, 0.40, 0.30, 0.20, 0.10, -0.20], index=list("abcdef"))
    free = pb.reachable_return(mu, max_weight=0.30)
    floored = pb.reachable_return(mu, max_weight=0.30, min_weight=0.10)
    assert floored < free
    assert floored == pytest.approx(
        0.10 * mu.sum() + 0.20 * (0.80 + 0.40) + 0.00 * 0.30)


def test_reachable_return_is_equal_weight_when_the_cap_binds_exactly():
    mu = pd.Series([0.5, 0.3, 0.1, -0.1], index=list("abcd"))
    assert pb.reachable_return(mu, max_weight=0.25) == pytest.approx(
        float(mu.mean()))


def test_frontier_spans_to_a_solvable_top_and_keeps_its_points():
    """
    The bug this guards: with the top end at mu.max() most targets were
    infeasible under the cap, were dropped, and the curve ended short of the
    best portfolio you could actually hold.
    """
    mu, cov = _skewed_moments()
    pts = pb.efficient_frontier(mu, cov, points=40, max_weight=0.30)
    # Nearly every requested point survives, rather than a quarter of them.
    assert len(pts) >= 36
    top = max(p["ret_pct"] for p in pts)
    ceiling = pb.reachable_return(mu, max_weight=0.30) * 100
    assert top == pytest.approx(ceiling, abs=0.25)


def test_max_sharpe_point_is_not_above_its_own_frontier():
    """The cyan dot has to land on the curve, because the curve is the menu."""
    mu, cov = _skewed_moments()
    w = pb.optimise_weights(mu, cov, max_weight=0.30)
    ret, _ = pb.performance(w, mu.to_numpy(), cov.to_numpy())
    top = max(p["ret_pct"] for p in pb.efficient_frontier(
        mu, cov, points=40, max_weight=0.30))
    assert ret * 100 <= top + 0.25


def _skewed_moments():
    """One runaway name, as live A-share data had — that is what broke it."""
    rng = np.random.default_rng(11)
    n = 300
    drift = np.array([0.0030, 0.0006, 0.0005, 0.0004, 0.0002, -0.0004])
    px = rng.normal(drift, 0.018, size=(n, len(drift)))
    rets = pd.DataFrame(px, columns=list("abcdef"))
    mu = rets.mean() * pb.TRADING_DAYS
    cov = rets.cov() * pb.TRADING_DAYS
    return mu, cov


# ── adapters take bare codes, not canonical symbols ──────────────────────────
def test_load_closes_hands_the_adapter_a_bare_code(monkeypatch):
    """
    The US bug: `US:AAPL` went straight to Yahoo, which 404s on it. A-shares
    hid it for months — a CN canonical symbol IS its bare code.
    """
    seen: list[str] = []

    class Fake:
        def fetch_ohlcv(self, code, years=3):
            seen.append(code)
            idx = pd.bdate_range("2024-01-01", periods=120)
            return pd.DataFrame({"Close": np.linspace(10, 20, 120)}, index=idx)

    import markets
    monkeypatch.setattr(markets, "parse", lambda s: (Fake(), s.split(":", 1)[-1]))
    px = pb.load_closes(["US:AAPL", "US:MSFT"], 60)

    assert seen == ["AAPL", "MSFT"]          # not US:AAPL
    assert list(px.columns) == ["US:AAPL", "US:MSFT"]   # keyed canonically


def test_unfetchable_symbols_report_which_and_why(monkeypatch):
    """The old message was 'try again later' — a real bug sent back as weather."""
    import markets

    def boom(sym):
        raise LookupError(f"行情源不可用：{sym}")

    monkeypatch.setattr(markets, "parse", boom)
    with pytest.raises(RuntimeError) as e:
        pb.load_closes(["US:AAPL", "US:MSFT"], 60)
    assert "US:AAPL" in str(e.value)
    assert "行情源不可用" in str(e.value)
    assert "try again" not in str(e.value).lower()


def test_names_fall_back_to_the_ticker_when_lookup_fails(monkeypatch):
    """A slow or dead name service must not take the optimisation with it."""
    import markets

    class Dead:
        def resolve(self, code):
            raise TimeoutError("nope")

    monkeypatch.setattr(markets, "parse", lambda s: (Dead(), s.split(":", 1)[-1]))
    assert pb._names(["US:AAPL", "US:MSFT"], "US") == {
        "US:AAPL": "AAPL", "US:MSFT": "MSFT"}


def test_names_use_the_resolved_company_name(monkeypatch):
    import markets
    from markets.base import StockRef

    class Live:
        def resolve(self, code):
            return StockRef(symbol=code, name=f"{code} Inc.", exchange="NMS")

    monkeypatch.setattr(markets, "parse", lambda s: (Live(), s.split(":", 1)[-1]))
    assert pb._names(["US:AAPL"], "US") == {"US:AAPL": "AAPL Inc."}


# ── industry exposure ────────────────────────────────────────────────────────
def _sector_stub(monkeypatch, mapping):
    monkeypatch.setattr(pb, "_sectors", lambda syms, market: mapping)


def test_industries_collapse_tickers_into_sectors(monkeypatch):
    _sector_stub(monkeypatch, {"a": "银行", "b": "银行", "c": "白酒"})
    out = pb.industries(pd.Series({"a": 0.3, "b": 0.3, "c": 0.4}),
                        {"a": "A", "b": "B", "c": "C"}, "CN")
    top = out["by_industry"][0]
    assert top["industry"] == "银行"
    assert top["weight_pct"] == pytest.approx(60.0)
    assert top["count"] == 2


def test_industries_flag_a_sector_bet(monkeypatch):
    """Six names at 16% each is not diversified if five are one sector."""
    _sector_stub(monkeypatch, {"a": "银行", "b": "银行", "c": "白酒"})
    out = pb.industries(pd.Series({"a": 0.4, "b": 0.3, "c": 0.3}),
                        {}, "CN")
    assert out["tone"] == "bad" and "银行" in out["note"]

    _sector_stub(monkeypatch, {"a": "银行", "b": "白酒", "c": "半导体"})
    ok = pb.industries(pd.Series({"a": 0.34, "b": 0.33, "c": 0.33}), {}, "CN")
    assert ok["tone"] == "good"


def test_unclassified_is_named_not_dropped(monkeypatch):
    """Dropping it would make the weights stop summing to 100%."""
    _sector_stub(monkeypatch, {"a": "银行", "b": None})
    out = pb.industries(pd.Series({"a": 0.5, "b": 0.5}), {}, "CN")
    assert sum(b["weight_pct"] for b in out["by_industry"]) == pytest.approx(100.0)
    assert any(b["industry"] == pb.UNCLASSIFIED for b in out["by_industry"])


def test_zero_weight_holdings_are_not_industry_exposure(monkeypatch):
    _sector_stub(monkeypatch, {"a": "银行", "b": "白酒"})
    out = pb.industries(pd.Series({"a": 1.0, "b": 0.0}), {}, "CN")
    assert [b["industry"] for b in out["by_industry"]] == ["银行"]


def test_cn_sectors_match_a_tushare_suffixed_code(monkeypatch):
    """
    A saved fund stores `601066.SH`; stock_basic is keyed `601066`. Matching
    those literally put every holding of every fund into 未分类.
    """
    import types as _t

    frame = pd.DataFrame({"symbol": ["601066", "600519"],
                          "industry": ["证券", "白酒"]})
    fake = _t.ModuleType("db_manager")
    fake.db = _t.SimpleNamespace(read_table=lambda *a, **k: frame)
    monkeypatch.setitem(sys.modules, "db_manager", fake)
    got = pb._sectors(["601066.SH", "600519"], "CN")
    assert got == {"601066.SH": "证券", "600519": "白酒"}


# ── reference points and hand-set weights ────────────────────────────────────
def test_singles_put_every_stock_in_risk_return_space(cn):
    mu, cov = moments_of(cn, list(cn))[1:3]
    pts = pb.singles(mu, cov, {t: f"名{t}" for t in cn})
    assert len(pts) == len(cn)
    one = pts[0]
    assert one["vol_pct"] == pytest.approx(float(np.sqrt(cov.iloc[0, 0])) * 100, abs=0.01)
    assert one["ret_pct"] == pytest.approx(float(mu.iloc[0]) * 100, abs=0.01)


def test_hand_set_weights_land_where_the_maths_says(cn):
    cols = list(cn)
    _, mu, cov, daily = moments_of_full(cn, cols)
    w = pd.Series({c: 1.0 / len(cols) for c in cols})
    got = pb.evaluate(mu, cov, daily, w, rf=0.03)
    ret, vol = pb.performance(w.to_numpy(), mu.to_numpy(), cov.to_numpy())
    assert got["ann_return_pct"] == pytest.approx(ret * 100, abs=0.01)
    assert got["ann_vol_pct"] == pytest.approx(vol * 100, abs=0.01)
    assert got["sharpe"] == pytest.approx((ret - 0.03) / vol, abs=0.01)


def test_an_unbalanced_allocation_is_reported_not_normalised(cn):
    """A user 3% short must see that, not a silently corrected answer."""
    cols = list(cn)
    short = {c: 20.0 for c in cols}          # 4 names × 20% = 80%
    out = pb.weigh(cols, short, lookback=200)
    assert out["sum_pct"] == pytest.approx(80.0, abs=0.01)
    assert out["balanced"] is False

    whole = {c: 25.0 for c in cols}
    assert pb.weigh(cols, whole, lookback=200)["balanced"] is True


def test_weighing_rejects_an_empty_allocation(cn):
    with pytest.raises(LookupError, match="大于 0"):
        pb.weigh(list(cn), {c: 0.0 for c in cn}, lookback=200)


def moments_of_full(frames, cols, duration=1):
    px = pd.concat([frames[c].rename(c) for c in cols], axis=1).dropna()
    return pb.moments(px, duration)
