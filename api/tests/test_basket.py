"""
pair_compare.basket — ranking several stocks, and the verdict about the laggard.

The ranking itself is arithmetic. The part worth testing is the judgement:
"they all move together, so sell the one moving least" is only sound when the
basket really is one trade and the laggard is genuinely weaker rather than just
lower-beta. Each of those three situations is built here with a known answer.

    python -m pytest api/tests/test_basket.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pair_compare as pc  # noqa: E402

N = 400
DATES = pd.bdate_range("2024-01-02", periods=N)


def _prices(logrets) -> pd.Series:
    lr = np.asarray(logrets, dtype=float)[:len(DATES) - 1]
    return pd.Series(100 * np.exp(np.concatenate([[0.0], np.cumsum(lr)])), index=DATES)


@pytest.fixture()
def market():
    rng = np.random.default_rng(5)
    return rng.normal(0.0004, 0.010, N)


@pytest.fixture()
def rising_market():
    """A market whose REALISED return over the window is firmly positive."""
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.010, N)
    r -= r[:N - 1].mean()          # the bars _prices actually uses
    return r + 0.0008


def _stock(market, beta, alpha=0.0, noise=0.0, seed=0):
    """beta x market + alpha, in log space, with optional idiosyncratic noise."""
    r = beta * np.asarray(market) + alpha
    if noise:
        r = r + np.random.default_rng(seed).normal(0, noise, len(r))
    return _prices(r)


def _run(closes, bench, **kw):
    return pc.basket(closes, bench_close=_prices(bench), window="all", **kw)


# ── the ranking ──────────────────────────────────────────────────────────────
def test_stocks_come_back_ranked_by_return(market):
    out = _run({
        "A": _stock(market, 1.2, 0.0006),
        "B": _stock(market, 1.2, 0.0003),
        "C": _stock(market, 1.2, 0.0000),
    }, market)

    assert [s["symbol"] for s in out["stocks"]] == ["A", "B", "C"]
    assert [s["rank_return"] for s in out["stocks"]] == [1, 2, 3]
    rets = [s["total_return_pct"] for s in out["stocks"]]
    assert rets == sorted(rets, reverse=True)


def test_vs_basket_is_relative_to_the_equal_weight_average(market):
    out = _run({"A": _stock(market, 1.4, 0.0006), "B": _stock(market, 1.0)}, market)

    # The leader beats the basket, the laggard trails it, and the basket's own
    # return sits between the two.
    lead, lag = out["stocks"][0], out["stocks"][-1]
    assert lead["vs_basket"] > 1.0 > lag["vs_basket"]
    assert lag["total_return_pct"] < out["basket"]["total_return_pct"] < lead["total_return_pct"]


def test_correlation_matrix_is_square_symmetric_and_unit_diagonal(market):
    out = _run({"A": _stock(market, 1.2, noise=0.004, seed=1),
                "B": _stock(market, 1.0, noise=0.004, seed=2),
                "C": _stock(market, 0.8, noise=0.004, seed=3)}, market)

    m = out["correlation"]
    n = len(out["symbols"])
    assert len(m) == n and all(len(row) == n for row in m)
    for i in range(n):
        assert m[i][i] == pytest.approx(1.0, abs=1e-9)
        for j in range(n):
            assert m[i][j] == pytest.approx(m[j][i], abs=1e-9)


def test_peer_correlation_excludes_the_stock_itself(market):
    """
    A stock is always correlated with a group it belongs to. Measuring against
    the rest is what makes the number mean "does it move with the others".
    """
    out = _run({"A": _stock(market, 1.0, noise=0.004, seed=1),
                "B": _stock(market, 1.0, noise=0.004, seed=2),
                "Z": _prices(np.random.default_rng(9).normal(0, 0.02, N))}, market)

    by = {s["symbol"]: s for s in out["stocks"]}
    # A and B move together and Z does not. A's figure averages over B AND Z,
    # so the meaningful assertion is the ordering, not an absolute level.
    assert by["Z"]["corr_to_peers"] < by["A"]["corr_to_peers"]
    assert by["Z"]["corr_to_peers"] < 0.2
    assert all(s["corr_to_peers"] < 1.0 for s in out["stocks"])


# ── the judgement ────────────────────────────────────────────────────────────
def test_a_genuinely_weaker_laggard_is_called_weak(market):
    """Same beta, less alpha: the one case that endorses selling the laggard."""
    out = _run({
        "LEAD": _stock(market, 1.2, 0.0008, noise=0.003, seed=1),
        "MID": _stock(market, 1.2, 0.0004, noise=0.003, seed=2),
        "LAG": _stock(market, 1.2, 0.0000, noise=0.003, seed=3),
    }, market)

    assert out["basket"]["cohesive"] is True
    assert out["stocks"][-1]["symbol"] == "LAG"
    kinds = {v["kind"] for v in out["verdicts"]}
    assert "weak" in kinds
    weak = next(v for v in out["verdicts"] if v["kind"] == "weak")
    assert weak["symbol"] == "LAG"


def test_a_low_beta_laggard_is_defended_not_condemned(rising_market):
    """
    It rises less because it is less exposed, not because it is worse. Selling
    it for lagging is selling low volatility to buy high volatility.

    Needs a market that actually rose: in a falling one the HIGH-beta name is
    the laggard, which is the same rule pointing the other way.
    """
    market = rising_market
    out = _run({
        "HOT": _stock(market, 1.8, 0.0002, noise=0.003, seed=1),
        "MID": _stock(market, 1.5, 0.0002, noise=0.003, seed=2),
        "CALM": _stock(market, 0.5, 0.0004, noise=0.003, seed=3),
    }, market)

    assert out["stocks"][-1]["symbol"] == "CALM"
    verdict = next(v for v in out["verdicts"] if v["symbol"] == "CALM")
    assert verdict["kind"] == "ok"
    # It trails on return but not on alpha — the whole point.
    by = {s["symbol"]: s for s in out["stocks"]}
    assert by["CALM"]["rank_return"] == 3
    assert by["CALM"]["rank_alpha"] < 3


def test_an_uncorrelated_basket_is_flagged_before_anything_else(market):
    """If they do not move together, the premise of the strategy is absent."""
    rng = np.random.default_rng(11)
    out = _run({
        "A": _prices(rng.normal(0.0008, 0.02, N)),
        "B": _prices(rng.normal(0.0004, 0.02, N)),
        "C": _prices(rng.normal(0.0000, 0.02, N)),
    }, market)

    assert out["basket"]["cohesive"] is False
    assert out["basket"]["avg_correlation"] < 0.5
    assert any(v["kind"] == "warn" and v["symbol"] is None for v in out["verdicts"])


def test_an_outsider_laggard_is_named_as_a_different_trade(market):
    """
    The group moves together, but the laggard does not move with it. Swapping
    it for the leader changes the exposure, not just the speed.
    """
    out = _run({
        "A": _stock(market, 1.3, 0.0006, noise=0.002, seed=1),
        "B": _stock(market, 1.3, 0.0005, noise=0.002, seed=2),
        "ODD": _prices(np.random.default_rng(23).normal(-0.0002, 0.02, N)),
    }, market)

    assert out["stocks"][-1]["symbol"] == "ODD"
    v = next(v for v in out["verdicts"] if v["symbol"] == "ODD")
    assert v["kind"] == "warn"
    # The "genuinely weaker" verdict must NOT also be issued — it would be
    # advice to sell, resting on a premise this basket fails.
    assert not any(x["kind"] == "weak" for x in out["verdicts"])


# ── guards ───────────────────────────────────────────────────────────────────
def test_only_days_every_stock_traded_are_used(market):
    a = _stock(market, 1.2, 0.0004)
    b = _stock(market, 1.0)
    c = _stock(market, 1.1)
    b = b.drop(b.index[100:160])

    out = _run({"A": a, "B": b, "C": c}, market)
    assert out["bars"] == N - 60


def test_refuses_baskets_it_cannot_rank(market):
    a = _stock(market, 1.2)
    with pytest.raises(LookupError):
        pc.basket({"A": a}, window="all")
    with pytest.raises(LookupError):
        pc.basket({str(i): a for i in range(pc.MAX_BASKET + 1)}, window="all")
    with pytest.raises(LookupError):
        pc.basket({"A": a.head(20), "B": a.head(20)}, window="all")


def test_the_payload_is_json_safe(market):
    import json
    out = _run({"A": _stock(market, 1.2, 0.0004), "B": _stock(market, 1.0)}, market,
               labels={"A": "甲", "B": "乙"}, benchmark_label="标普500", market="US")
    json.dumps(out, allow_nan=False)
    assert out["stocks"][0]["label"] in ("甲", "乙")
    assert out["benchmark"]["label"] == "标普500"
    assert out["market"] == "US"
