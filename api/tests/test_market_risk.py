"""
🚦 市场风险 — the five risk tiles.

What is worth pinning: that the percentile is measured against the regime and
nothing before it, that a half-reported session cannot pass for a liquidity
collapse, that 涨停 is counted at quoted precision, and that a session which
is not published yet is never stored as a session with zero 涨停.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import market_risk as mr  # noqa: E402


def _dates(n: int, start: str = "2024-10-14") -> list[str]:
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(start, periods=n)]


# ── percentile & grade ───────────────────────────────────────────────────────
def test_percentile_ignores_everything_before_the_anchor():
    """A pre-regime spike must not make today look ordinary."""
    before = pd.Series([1000.0] * 30, index=_dates(30, "2024-08-01"))
    after = pd.Series(np.arange(1, 41, dtype=float), index=_dates(40))
    pct, n = mr.percentile(pd.concat([before, after]))
    assert n == 40
    assert pct == pytest.approx(100.0)


def test_percentile_is_the_share_at_or_below_today():
    s = pd.Series([5.0, 1, 2, 3, 4] * 4 + [3.0], index=_dates(21))
    pct, _ = mr.percentile(s)
    # 3 is >= 1, 2, 3 in each of four blocks, plus itself.
    assert pct == pytest.approx(13 / 21 * 100, abs=0.1)


def test_percentile_refuses_a_thin_sample():
    pct, n = mr.percentile(pd.Series([1.0] * 5, index=_dates(5)))
    assert pct is None and n == 5
    assert mr.grade("vol", pct) == ("样本不足", 0)


@pytest.mark.parametrize("key,pct,state,level", [
    ("turnover", 5, "枯竭", 2), ("turnover", 20, "偏冷", 1),
    ("turnover", 95, "过热", 1), ("margin", 100, "过热", 2),
    ("valuation", 50, "合理", 0), ("vol", 80, "偏高", 1),
])
def test_grade_bands(key, pct, state, level):
    assert mr.grade(key, pct) == (state, level)


# ── turnover & margin ────────────────────────────────────────────────────────
def test_turnover_drops_a_session_where_one_exchange_is_missing():
    """Summing only Shanghai on a day Shenzhen was late would halve 成交额."""
    info = pd.DataFrame({
        "trade_date": ["20261023", "20261023", "20261026", "20261023"],
        "ts_code": ["SH_MARKET", "SZ_MARKET", "SH_MARKET", "SH_FUND"],
        "amount": [9000.0, 10000.0, 8000.0, 5000.0],
    })
    s = mr.turnover_series(info)
    assert list(s.index) == ["20261023"]
    assert s.iloc[0] == pytest.approx(19000.0)       # funds are not stocks


def test_margin_share_uses_sse_and_szse_only():
    margin = pd.DataFrame({
        "trade_date": ["20261023"] * 3,
        "exchange_id": ["SSE", "SZSE", "BSE"],
        "rzmre": [8.0e10, 7.0e10, 6.0e8],
    })
    buy = mr.margin_buy_series(margin)
    assert buy.iloc[0] == pytest.approx(1500.0)       # 亿元
    share = mr.margin_share(buy, pd.Series([20000.0], index=["20261023"]))
    assert share.iloc[0] == pytest.approx(7.5)


def test_realized_vol_is_annualised_percent():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 400)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=_dates(400, "2024-01-01"))
    v = mr.realized_vol(close, 20)
    assert v.median() == pytest.approx(0.01 * np.sqrt(252) * 100, rel=0.2)


# ── limit-up stats ───────────────────────────────────────────────────────────
def test_limit_stats_counts_sealed_broken_and_down():
    daily = pd.DataFrame({
        "ts_code": ["A", "B", "C", "D", "E"],
        # A sealed (float noise above the quoted limit), B touched and broke,
        # C closed at limit-down, D an ordinary day, E no limit (new listing).
        "close": [10.89, 10.50, 9.01, 10.0, 50.0],
        "high": [10.89, 10.89, 9.80, 10.2, 60.0],
    })
    limits = pd.DataFrame({
        "ts_code": ["A", "B", "C", "D"],
        "up_limit": [10.890000001, 10.89, 11.01, 11.0],
        "down_limit": [8.91, 8.91, 9.01, 9.0],
    })
    got = mr.limit_stats(daily, limits)
    assert got == {"up_count": 1, "down_count": 1, "touched": 2, "total": 4}


def test_limit_frame_broken_rate_is_blank_when_nothing_touched():
    f = mr.limit_frame({
        "20261023": {"up_count": 80, "down_count": 5, "touched": 100, "total": 5000},
        "20261026": {"up_count": 0, "down_count": 40, "touched": 0, "total": 5000},
    })
    assert f.loc["20261023", "broken_rate"] == pytest.approx(20.0)
    assert np.isnan(f.loc["20261026", "broken_rate"])


class _Pro:
    """Tushare stand-in: one published session, one not yet."""

    def __init__(self):
        self.calls = []

    def daily(self, trade_date, fields=None):
        self.calls.append(trade_date)
        if trade_date == "20261028":
            return pd.DataFrame()
        return pd.DataFrame({"ts_code": ["A"], "close": [11.0], "high": [11.0]})

    def stk_limit(self, trade_date):
        return pd.DataFrame({"ts_code": ["A"], "up_limit": [11.0], "down_limit": [9.0]})


class _Db:
    def __init__(self, rows=None, broken=False):
        self.rows = rows or []
        self.broken = broken
        self.written = []

    def read_table(self, table):
        if self.broken:
            raise RuntimeError("relation does not exist")
        return pd.DataFrame(self.rows)

    def insert_records(self, table, records, upsert=False):
        if self.broken:
            raise RuntimeError("relation does not exist")
        self.written.extend(records)


@pytest.fixture(autouse=True)
def _clean_store():
    mr._mem.clear()
    mr._filling = False
    yield
    mr._mem.clear()


def test_an_unpublished_session_is_not_stored_as_zero():
    db = _Db()
    rows, left = mr.limit_sessions(_Pro(), ["20261027", "20261028"], db,
                                   background=False)
    assert set(rows) == {"20261027"}
    assert [r["trade_date"] for r in db.written] == ["20261027"]
    assert left == 0


def test_stored_sessions_are_not_fetched_again():
    db = _Db(rows=[{"trade_date": "20261027", "up_count": 1, "down_count": 0,
                    "touched": 1, "total": 1}])
    pro = _Pro()
    mr.limit_sessions(pro, ["20261027"], db, background=False)
    assert pro.calls == []


def test_sync_budget_takes_the_newest_gaps_first():
    pro = _Pro()
    dates = ["20261021", "20261022", "20261023", "20261026", "20261027"]
    rows, left = mr.limit_sessions(pro, dates, _Db(), sync_budget=2,
                                   background=False)
    assert sorted(pro.calls) == ["20261026", "20261027"]
    assert left == 3


def test_a_missing_table_degrades_to_memory():
    db = _Db(broken=True)
    rows, _ = mr.limit_sessions(_Pro(), ["20261027"], db, background=False)
    assert "20261027" in rows
    assert "20261027" in mr._mem


def test_panic_overrides_a_normal_limit_up_count():
    n = 30
    rows = {d: {"up_count": 50, "down_count": 5, "touched": 70, "total": 5000}
            for d in _dates(n)}
    last = _dates(n)[-1]
    rows[last] = {"up_count": 50, "down_count": 300, "touched": 70, "total": 5000}
    t = mr.limits_tile(mr.limit_frame(rows), 0)
    assert (t["state"], t["level"]) == ("恐慌", 2)
