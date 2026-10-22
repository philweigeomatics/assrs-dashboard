"""
波动与反弹 — do the bigger swingers win back more after a market fall?

The episode tests pin the definition that matters for A-shares: 沪深300 has
never regained its February 2021 high, so "a drawdown lasts until the peak
is regained" would make six years one episode. Each correction counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import swing as W  # noqa: E402


def path(points, start="2021-01-04"):
    """A price series that runs in straight lines (in log terms) between waypoints."""
    vals = []
    for (a, n_a), (b, _) in zip(points, points[1:]):
        vals.extend(np.exp(np.linspace(np.log(a), np.log(b), n_a, endpoint=False)))
    vals.append(points[-1][0])
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


# ── episodes ─────────────────────────────────────────────────────────────────
def test_each_correction_in_a_bear_market_counts_on_its_own():
    """Down 20, bounce 15, down 20 again — never back to the first high."""
    m = path([(100, 40), (80, 30), (92, 40), (73.6, 30), (85, 1)])
    eps = W.episodes(m, 10)
    assert len(eps) == 2
    assert eps[0]["fall_pct"] == pytest.approx(-20, abs=0.5)
    assert eps[1]["fall_pct"] == pytest.approx(-20, abs=0.5)


def test_a_dip_below_the_threshold_is_not_an_episode():
    m = path([(100, 40), (94, 30), (110, 1)])
    assert W.episodes(m, 8) == []


def test_a_fall_still_in_progress_is_open():
    m = path([(100, 40), (85, 20), (86, 1)])
    eps = W.episodes(m, 10)
    assert len(eps) == 1 and eps[0]["open"] and eps[0]["turned"] is None


def test_the_trough_is_the_lowest_close_not_where_the_bounce_was_confirmed():
    m = path([(100, 30), (80, 10), (90, 1)])
    e = W.episodes(m, 10)[0]
    assert m[e["trough"]] == pytest.approx(80, rel=1e-6)


# ── the moves through an episode ─────────────────────────────────────────────
def test_stocks_are_measured_on_the_markets_dates_down_and_back_up():
    m = path([(100, 40), (80, 70), (100, 1)])
    p = pd.DataFrame({"hi": m ** 2 / 100, "lo": m ** 0.5 * 10}, index=m.index)
    eps = W.episodes(m, 10)
    mv = W.episode_moves(p, m, eps)[0]
    assert mv["market"]["fall"] == pytest.approx(-20, abs=0.1)
    assert mv["stocks"]["hi"]["fall"] == pytest.approx(-36, abs=0.2)       # 0.8² − 1
    assert mv["stocks"]["lo"]["fall"] == pytest.approx(-10.6, abs=0.2)     # √0.8 − 1
    assert mv["stocks"]["hi"]["reb60"] > mv["market"]["reb60"] > mv["stocks"]["lo"]["reb60"]


def test_a_rebound_past_the_end_of_the_data_is_unknown_not_zero():
    m = path([(100, 40), (80, 10), (90, 1)])
    p = pd.DataFrame({"a": m}, index=m.index)
    mv = W.episode_moves(p, m, W.episodes(m, 10))[0]
    assert mv["market"]["reb60"] is None and mv["stocks"]["a"]["reb60"] is None


def test_the_confirmed_entry_is_the_first_close_three_percent_off_the_low():
    m = path([(100, 40), (80, 20), (100, 60)])
    v = m.to_numpy()
    t = int(np.argmin(v))
    c = W._confirm_i(v, t)
    assert v[c] >= v[t] * 1.03 and v[c - 1] < v[t] * 1.03


def test_waiting_for_confirmation_costs_part_of_the_rebound():
    """Rebounds are front-loaded: the sharp first leg is what waiting misses."""
    m = path([(100, 40), (80, 8), (92, 100), (96, 1)])
    p = pd.DataFrame({"a": m}, index=m.index)
    mv = W.episode_moves(p, m, W.episodes(m, 10))[0]
    assert mv["market"]["reb60_confirm"] < mv["market"]["reb60"]


# ── the multiples ────────────────────────────────────────────────────────────
def test_up_and_down_day_multiples_see_the_asymmetry():
    rng = np.random.default_rng(0)
    mret = pd.Series(rng.normal(0, 0.01, 500))
    r = pd.Series(np.where(mret > 0, 2.0 * mret, 1.0 * mret))
    ab = W.asym_beta(r, mret)
    assert ab["beta_up"] == pytest.approx(2.0, abs=0.01)
    assert ab["beta_down"] == pytest.approx(1.0, abs=0.01)


def test_the_multiples_are_capture_ratios_like_the_ranking_table():
    """Same definition as pair_compare._capture, so the page never shows two numbers for one thing."""
    import pair_compare
    rng = np.random.default_rng(1)
    mret = pd.Series(rng.normal(0, 0.01, 300))
    r = 1.3 * mret + pd.Series(rng.normal(0, 0.01, 300))
    ab = W.asym_beta(r, mret)
    # Two decimals here, one decimal of a percentage there: equal within rounding.
    assert ab["beta_up"] * 100 == pytest.approx(pair_compare._capture(r, mret, True), abs=0.51)
    assert ab["beta_down"] * 100 == pytest.approx(pair_compare._capture(r, mret, False), abs=0.51)


# ── summaries and words ──────────────────────────────────────────────────────
def test_a_stock_that_rose_through_the_falls_gets_no_rebound_to_fall_ratio():
    moves = [{"open": False, "market": {"fall": -10, "reb60": 5},
              "stocks": {"a": {"fall": 3.0, "reb60": 20.0}}}]
    s = W.summarise(moves, ["a"], "reb60")["a"]
    assert s["ratio"] is None and s["beat"] == 1


def test_the_ratio_is_rebound_over_the_fall():
    moves = [{"open": False, "market": {"fall": -10, "reb60": 5},
              "stocks": {"a": {"fall": -20.0, "reb60": 30.0}}}]
    assert W.summarise(moves, ["a"], "reb60")["a"]["ratio"] == pytest.approx(1.5)


def _three(seed=0):
    rng = np.random.default_rng(seed)
    n = 1200
    idx = pd.bdate_range("2020-11-02", periods=n)
    mr = rng.normal(0.0003, 0.01, n)
    mr[200:240] -= 0.006
    mr[600:640] -= 0.005
    mr[900:930] -= 0.006
    m = pd.Series(100 * np.exp(np.cumsum(mr)), index=idx)
    p = pd.DataFrame({f"S{k}": 100 * np.exp(np.cumsum(b * mr + rng.normal(0, 0.006, n)))
                      for k, b in enumerate([0.6, 1.2, 2.2])}, index=idx)
    return p, m


def test_it_answers_the_question_and_says_how_many_drawdowns_it_rests_on():
    p, m = _three()
    out = W.analyse(p.tail(253), m.tail(253), p, m, labels={}, bench_label="IDX", threshold=10)
    text = "".join(out["conclusions"])
    assert "波动大 = 反弹多" in text
    assert out["episode_count"] >= 3
    assert out["summary"]["S2"]["avg_rebound"] > out["summary"]["S0"]["avg_rebound"]
    assert out["summary"]["S2"]["avg_fall"] < out["summary"]["S0"]["avg_fall"]


def test_an_open_drawdown_is_counted_separately():
    m = path([(100, 40), (80, 70), (100, 40), (85, 1)])
    p = pd.DataFrame({"a": m, "b": m ** 1.5 / 10}, index=m.index)
    out = W.analyse(p, m, p, m, labels={}, bench_label="IDX", threshold=10)
    assert any("仍在进行中" in c for c in out["conclusions"])


def test_where_the_market_is_now():
    m = path([(100, 200), (85, 40), (87, 1)])
    now = W.market_now(m, 10)
    assert now["dd_from_high_pct"] == pytest.approx(-13, abs=0.2)
    assert now["in_drawdown"] and now["off_low_pct"] > 0


def test_only_offered_thresholds_and_entries():
    p, m = _three()
    with pytest.raises(ValueError):
        W.analyse(p, m, p, m, labels={}, bench_label="X", threshold=7)
    with pytest.raises(ValueError):
        W.analyse(p, m, p, m, labels={}, bench_label="X", entry="later")


def test_without_the_index_it_still_compares_volatility():
    p, _ = _three()
    out = W.analyse(p, None, p, None, labels={}, bench_label=None)
    assert out["episodes"] == [] and out["now"] is None
    assert out["stats"]["S2"]["vol_annual_pct"] > out["stats"]["S0"]["vol_annual_pct"]
