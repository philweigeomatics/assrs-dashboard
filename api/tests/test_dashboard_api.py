"""
dashboard_api: the shaping, not the arithmetic.

Everything here runs against a stand-in data_manager, so no database and no
Tushare. What is being tested is the part that actually breaks in production:
a NaN reaching the client as a bare `NaN` token, a sector box disagreeing with
the stocks drawn inside it, and "the data source failed" rendering as "the
market was flat".

    python -m pytest api/tests/test_dashboard_api.py -q
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

from api import dashboard_api as da  # noqa: E402


class FakeTushare:
    def __init__(self, daily=None, top=None):
        self._daily, self._top = daily, top or {}

    def daily(self, trade_date, fields=None):
        if self._daily is None:
            raise RuntimeError("tushare is down")
        return self._daily

    def top_list(self, trade_date):
        return self._top.get(trade_date)


def fake_dm(monkeypatch, **attrs) -> types.ModuleType:
    mod = types.ModuleType("data_manager")
    mod.init_tushare = lambda: None
    mod.TUSHARE_API = None
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, "data_manager", mod)
    return mod


# ── the JSON safety net ──────────────────────────────────────────────────────
@pytest.mark.parametrize("value", [np.nan, None, "", "abc", pd.NA, float("nan")])
def test_unusable_numbers_become_null_not_nan(value):
    """`NaN` is not JSON. One of these in a payload breaks the whole page."""
    assert da._num(value) is None


def test_real_numbers_survive():
    assert da._num("3.5") == 3.5 and da._num(np.float64(2)) == 2.0


@pytest.mark.parametrize("raw, want", [
    ("20260917", "2026-09-17"), ("2026-09-17", "2026-09-17"), (20260917, "2026-09-17"),
])
def test_trade_dates_are_normalised(raw, want):
    assert da._iso(raw) == want


# ── 热力图 ───────────────────────────────────────────────────────────────────
def heatmap_dm(monkeypatch, *, daily=None):
    basic = pd.DataFrame({
        "ticker": ["600000", "600001", "300001", "300002"],
        "circ_mv_yi": [1000.0, 500.0, 200.0, np.nan],
        "trade_date": ["20260917"] * 4,
    })
    return fake_dm(
        monkeypatch,
        get_sector_stock_map=lambda: {"银行": ["600000", "600001"],
                                      "半导体": ["300001", "300002"]},
        get_daily_basic_for_tickers=lambda t: basic,
        get_all_stock_basic=lambda: [{"ticker": "600000", "name": "浦发银行"},
                                     {"ticker": "600001", "name": "邯郸钢铁"},
                                     {"ticker": "300001", "name": "特锐德"}],
        TUSHARE_API=FakeTushare(daily=daily),
    )


def moves(pairs) -> pd.DataFrame:
    return pd.DataFrame({"ts_code": [f"{t}.SH" for t, _ in pairs],
                         "pct_chg": [p for _, p in pairs]})


def test_a_sector_box_agrees_with_the_stocks_drawn_inside_it():
    """Cap-weighted, from the same per-stock numbers the leaves carry."""
    with pytest.MonkeyPatch.context() as mp:
        heatmap_dm(mp, daily=moves([("600000", 3.0), ("600001", -3.0),
                                    ("300001", 5.0)]))
        res = da.heatmap()

    bank = next(s for s in res["sectors"] if s["name"] == "银行")
    assert bank["pct"] == pytest.approx((3.0 * 1000 - 3.0 * 500) / 1500, abs=0.01)
    assert bank["mcap"] == 1500.0
    assert res["trade_date"] == "2026-09-17"


def test_a_stock_with_no_market_cap_is_left_out_rather_than_sized_zero():
    with pytest.MonkeyPatch.context() as mp:
        heatmap_dm(mp, daily=moves([("300001", 5.0)]))
        res = da.heatmap()

    chips = next(s for s in res["sectors"] if s["name"] == "半导体")
    assert [s["t"] for s in chips["stocks"]] == ["300001"]


def test_a_failed_move_fetch_is_declared_not_drawn_as_a_flat_market():
    """
    Every box grey because Tushare refused is not the same picture as every box
    grey because nothing moved, and the page has to be able to tell you which.
    """
    with pytest.MonkeyPatch.context() as mp:
        heatmap_dm(mp, daily=None)      # the fake raises
        res = da.heatmap()

    assert res["has_moves"] is False
    assert all(s["pct"] == 0.0 for s in res["sectors"])

    with pytest.MonkeyPatch.context() as mp:
        heatmap_dm(mp, daily=moves([("600000", 1.0)]))
        assert da.heatmap()["has_moves"] is True


def test_sectors_and_stocks_come_back_largest_first():
    with pytest.MonkeyPatch.context() as mp:
        heatmap_dm(mp, daily=moves([("600000", 1.0)]))
        res = da.heatmap()

    assert [s["name"] for s in res["sectors"]] == ["银行", "半导体"]
    assert [s["t"] for s in res["sectors"][0]["stocks"]] == ["600000", "600001"]


def test_no_market_cap_data_at_all_is_an_error_not_an_empty_map():
    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, get_sector_stock_map=lambda: {"银行": ["600000"]},
                get_daily_basic_for_tickers=lambda t: pd.DataFrame())
        with pytest.raises(LookupError, match="daily_basic"):
            da.heatmap()


# ── 市场宽度 ─────────────────────────────────────────────────────────────────
def breadth_frame() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-17"])
    return pd.DataFrame({"银行": [0.2, 0.3, 0.9],
                         "半导体": [0.8, 0.7, 0.1],
                         "空板块": [np.nan] * 3}, index=idx)


def test_breadth_is_oldest_first_and_ranked_by_today():
    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, load_market_breadth_from_db=breadth_frame)
        res = da.breadth()

    assert res["dates"] == ["2026-09-15", "2026-09-16", "2026-09-17"]
    assert [s["name"] for s in res["sectors"]] == ["银行", "半导体"]
    assert res["sectors"][0]["values"] == [0.2, 0.3, 0.9]


def test_a_sector_with_no_readings_is_dropped_not_shown_as_zero():
    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, load_market_breadth_from_db=breadth_frame)
        res = da.breadth()
    assert "空板块" not in [s["name"] for s in res["sectors"]]
    assert res["total"] == 2 and res["hot"] == 1


def test_an_empty_breadth_table_names_the_table():
    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, load_market_breadth_from_db=lambda: None)
        with pytest.raises(LookupError, match="market_breadth"):
            da.breadth()


# ── 龙虎榜 ───────────────────────────────────────────────────────────────────
def top_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["600519.SH", "000001.SZ"],
        "name": ["贵州茅台", "平安银行"],
        "close": [1500.0, 11.2],
        "pct_change": [5.1, -3.2],
        "net_amount": [1.23e8, -4.5e7],
        "net_rate": [2.2, -1.1],
        "reason": ["日涨幅偏离值达7%", "日跌幅偏离值达7%"],
    })


def test_the_top_list_walks_back_to_the_last_session_that_has_one():
    """A Saturday request must return Friday's list, not nothing."""
    from datetime import datetime, timedelta
    now = datetime.now(da.SHANGHAI)
    two_days_ago = (now - timedelta(days=2)).strftime("%Y%m%d")

    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, TUSHARE_API=FakeTushare(top={two_days_ago: top_frame()}))
        res = da.top_list()

    assert res["trade_date"] == da._iso(two_days_ago)
    assert [r["t"] for r in res["rows"]] == ["600519", "000001"]


def test_net_amount_is_converted_to_the_unit_it_is_quoted_in():
    """1.23e8 元 is 12,300 万元. A missing divide is a 10,000× error."""
    today = __import__("datetime").datetime.now(da.SHANGHAI).strftime("%Y%m%d")
    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, TUSHARE_API=FakeTushare(top={today: top_frame()}))
        rows = da.top_list()["rows"]

    assert rows[0]["net"] == pytest.approx(12300.0)
    assert rows[1]["net"] == pytest.approx(-4500.0)


def test_rows_are_ranked_by_net_flow():
    today = __import__("datetime").datetime.now(da.SHANGHAI).strftime("%Y%m%d")
    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, TUSHARE_API=FakeTushare(top={today: top_frame()}))
        nets = [r["net"] for r in da.top_list()["rows"]]
    assert nets == sorted(nets, reverse=True)


def test_no_top_list_anywhere_in_the_window_is_an_error():
    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp, TUSHARE_API=FakeTushare(top={}))
        with pytest.raises(LookupError, match="龙虎榜"):
            da.top_list(lookback_days=3)


# ── 市场杠杆 ─────────────────────────────────────────────────────────────────
def test_leverage_frames_become_lists_and_failures_are_kept():
    series = pd.DataFrame({"period": ["2026-09-16", "2026-09-17"], "value": [18000.0, 18120.0]})
    detail = pd.DataFrame({"period": ["2026-09-17"], "rzye": [17900.0],
                           "rqye": [220.0], "rzmre": [1500.0], "rzche": [1400.0],
                           "net_fin": [100.0]})
    fake = types.ModuleType("market_leverage")
    fake.fetch_all = lambda api: [
        {"market": "CN", "label": "两融余额", "ok": True, "unit": " 亿元", "freq": "daily",
         "latest": 18120.0, "prev": 18000.0, "asof": "2026-09-17", "note": "n",
         "error": None, "series": series, "cn_detail": detail},
        {"market": "US", "label": "US margin debt", "ok": False, "unit": " $bn",
         "freq": "monthly", "latest": None, "prev": None, "asof": None, "note": "n",
         "error": "FINRA unreachable", "series": None, "cn_detail": None},
    ]

    with pytest.MonkeyPatch.context() as mp:
        fake_dm(mp)
        mp.setitem(sys.modules, "market_leverage", fake)
        out = da.leverage()

    assert len(out) == 2
    assert out[0]["series"] == [{"period": "2026-09-16", "value": 18000.0},
                                {"period": "2026-09-17", "value": 18120.0}]
    assert out[0]["detail"][0]["net_fin"] == 100.0
    assert out[0]["unit"] == "亿元"
    # The failed market is present with its reason — not silently dropped.
    assert out[1]["ok"] is False and out[1]["error"] == "FINRA unreachable"
    assert out[1]["series"] == [] and out[1]["detail"] == []
