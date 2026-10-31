"""
美股行业 ETF — level, structure and stretch for each US industry.

The arithmetic is pinned by hand-built series; the parts that could mislead
— ranking a stretch against the ETF's own history, counting past stretches
as episodes rather than days, and the rank-0 ETF that a falsy check dropped
— are pinned by name.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import us_etf as U  # noqa: E402

IDX = pd.bdate_range("2016-01-04", periods=2500)


def walk(seed, vol=0.012, drift=0.0003, n=len(IDX)):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), IDX[:n])


def board(**override):
    closes = {t: walk(i) for i, (t, _, _) in enumerate(U.UNIVERSE)}
    closes.update(override)
    return pd.DataFrame(closes)


# ── one ETF ──────────────────────────────────────────────────────────────────
def test_distances_from_the_averages_are_what_they_say():
    s = walk(1)
    r = U.row("XLK", s, None)
    assert r["d200"] == pytest.approx((s.iloc[-1] / s.tail(200).mean() - 1) * 100, abs=0.06)
    assert r["d50"] == pytest.approx((s.iloc[-1] / s.tail(50).mean() - 1) * 100, abs=0.06)


def test_the_trend_state_reads_both_averages():
    assert U.trend_state(True, True) == "多头"
    assert U.trend_state(False, True) == "回调"
    assert U.trend_state(True, False) == "反弹"
    assert U.trend_state(False, False) == "空头"


def test_52_week_position_and_drawdown():
    s = pd.Series(np.r_[np.linspace(50, 150, 400), np.linspace(150, 100, 100)], IDX[:500])
    r = U.row("XLE", s, None)
    assert r["hi52"] == pytest.approx(150) and r["from_high"] == pytest.approx(-33.3, abs=0.1)
    assert 0 <= r["range_pos"] <= 1


def test_the_cross_is_dated():
    s = pd.Series(np.r_[np.linspace(100, 200, 400), np.linspace(200, 120, 120)], IDX[:520])
    r = U.row("XLE", s, None)
    assert r["golden"] is False and r["cross_days"] is not None and r["cross_days"] < 120


def test_relative_strength_against_spy():
    spy = pd.Series(np.linspace(100, 110, 300), IDX[:300])
    etf = pd.Series(np.linspace(100, 130, 300), IDX[:300])
    r = U.row("XLK", etf, spy)
    expect = ((etf.iloc[-1] / etf.iloc[-64]) / (spy.iloc[-1] / spy.iloc[-64]) - 1) * 100
    assert r["rs_3m"] == pytest.approx(expect, abs=0.06)


def test_too_little_history_is_left_out_not_guessed():
    assert U.row("AIQ", walk(2, n=150), None) is None


# ── stretch, ranked against itself ───────────────────────────────────────────
def test_the_stretch_is_ranked_within_the_etfs_own_history():
    """Today at a ten-year high above its average → near the top of ITS range."""
    s = walk(3, vol=0.008, drift=0.0)
    s.iloc[-30:] = s.iloc[-31] * np.linspace(1.0, 1.3, 30)
    r = U.row("XLU", s, None)
    assert r["stretch_rank"] >= 95


def test_past_stretches_are_counted_as_episodes_not_days():
    """Three long stretches of 100 days each are 3 outcomes, not 300."""
    n = 1200
    base = np.full(n, 100.0)
    for start in (300, 600, 900):
        base[start:start + 100] = 130.0
    s = pd.Series(base, IDX[:n])
    dist = U.stretch_history(s)
    a = U.forward_after(s, dist, above=True, level=20)
    assert a is not None and a["episodes"] == 3


def test_episodes_whose_future_has_not_happened_are_left_out():
    n = 1100
    base = np.full(n, 100.0)
    for start in (250, 520, 790):            # each clear of the last one's 200-day window
        base[start:start + 60] = 130.0
    base[-30:] = 130.0                      # stretched now, no 63 days after yet
    s = pd.Series(base, IDX[:n])
    a = U.forward_after(s, U.stretch_history(s), above=True, level=20)
    assert a["episodes"] == 3


def test_fewer_than_three_comparable_stretches_gives_no_answer():
    s = walk(4)
    assert U.forward_after(s, U.stretch_history(s), above=True, level=500) is None
    base = np.full(900, 100.0)
    for start in (250, 520):                 # two clean episodes — still not enough
        base[start:start + 60] = 130.0
    two = pd.Series(base, IDX[:900])
    assert U.forward_after(two, U.stretch_history(two), above=True, level=20) is None


# ── the board ────────────────────────────────────────────────────────────────
def test_breadth_counts_the_industries_not_the_broad_market():
    out = U.analyse(board())
    n_industry = sum(1 for _, _, g in U.UNIVERSE if g != "大盘")
    assert out["breadth"]["n"] == n_industry
    rows = [r for r in out["rows"] if r["group"] != "大盘"]
    assert out["breadth"]["above200"] == sum(r["d200"] > 0 for r in rows)


def test_the_most_washed_out_etf_is_named_even_at_rank_zero():
    """`rank or 100` turned rank 0 into 100 and dropped URA from the list."""
    crash = walk(7, vol=0.01, drift=0.0)
    crash.iloc[-40:] = crash.iloc[-41] * np.linspace(1.0, 0.55, 40)
    out = U.analyse(board(URA=crash))
    ura = next(r for r in out["rows"] if r["ticker"] == "URA")
    assert ura["stretch_rank"] == 0
    assert any("URA" in c for c in out["conclusions"])


def test_a_record_stretch_says_it_has_no_precedent():
    hot = walk(8, vol=0.006, drift=0.0)
    hot.iloc[-40:] = hot.iloc[-41] * np.linspace(1.0, 1.4, 40)
    out = U.analyse(board(XLV=hot))
    assert any("XLV" in c and "先例" in c for c in out["conclusions"])


def test_missing_tickers_are_reported():
    frame = board().drop(columns=["URA", "LIT"])
    assert set(U.analyse(frame)["missing"]) == {"URA", "LIT"}


def test_detail_carries_its_own_bands():
    frame = board()
    d = U.detail(frame, "SMH")
    assert len(d["dates"]) == len(d["close"]) == len(d["ma200"]) == len(d["stretch"])
    assert d["bands"][10] < d["bands"][50] < d["bands"][90]
    with pytest.raises(LookupError):
        U.detail(frame, "NOPE")


# ── the clock ────────────────────────────────────────────────────────────────
def test_live_only_while_new_york_trades_and_the_bar_is_todays():
    ny = ZoneInfo("America/New_York")
    wed_noon = datetime(2026, 10, 28, 12, 0, tzinfo=ny)
    assert U.session_state(pd.Timestamp("2026-10-28"), wed_noon) == "live"
    assert U.session_state(pd.Timestamp("2026-10-27"), wed_noon) == "closed"
    assert U.session_state(pd.Timestamp("2026-10-28"), datetime(2026, 10, 28, 17, 0, tzinfo=ny)) == "closed"
    assert U.session_state(pd.Timestamp("2026-10-31"), datetime(2026, 10, 31, 12, 0, tzinfo=ny)) == "closed"
