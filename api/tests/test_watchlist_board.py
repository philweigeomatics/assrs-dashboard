"""
The watchlist, as something you can watch.

It used to be 代码 / 名称 / 加入日期 / 操作 — a list that could only tell you
what you already knew. Worse, the nightly scan had been writing price, RSI
and parsed signals for every A-share on it all along, and this page never
read that table.

What these pin is the arithmetic and the joins. The speed that made it
practical — Tushare and Yahoo both taking a batch, 81 + 23 names in 7.3s
rather than a hundred calls — is a property of the APIs, checked by hand.
"""

from __future__ import annotations

import os
import sys
import types
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import watchlist_board as wb  # noqa: E402


def _series(values, start="20260901"):
    idx = [str(int(start) + i) for i in range(len(values))]
    return pd.Series(values, index=idx, dtype=float)


# ── moves ────────────────────────────────────────────────────────────────────
def test_move_is_measured_against_the_close_n_sessions_back():
    s = _series([100, 101, 102, 103, 104, 105])
    assert wb._move(s, 1) == pytest.approx(0.96, abs=0.01)   # 105/104 - 1
    assert wb._move(s, 5) == pytest.approx(5.0, abs=0.01)    # 105 vs 100


def test_a_move_needs_enough_history_rather_than_inventing_one():
    s = _series([100, 101])
    assert wb._move(s, 1) is not None
    assert wb._move(s, 5) is None
    assert wb._move(s, 20) is None


def test_a_zero_base_does_not_divide():
    assert wb._move(_series([0.0, 5.0]), 1) is None


# ── forward adjustment ───────────────────────────────────────────────────────
def _tushare(daily, adj=None):
    mod = types.ModuleType("data_manager")
    mod.init_tushare = lambda: True

    class Api:
        def daily(self, ts_code=None, start_date=None, end_date=None):
            return daily

        def adj_factor(self, ts_code=None, start_date=None, end_date=None):
            if adj is None:
                raise RuntimeError("no adj")
            return adj

    mod.TUSHARE_API = Api()
    mod.get_tushare_ticker = lambda t: t if "." in t else f"{t}.SH"
    return mod


def test_closes_are_forward_adjusted_so_a_split_is_not_a_cliff(monkeypatch):
    """
    `daily` is 不复权. A 10送10 halves the raw close, which would draw a drop
    in the sparkline that nobody holding the stock experienced.
    """
    daily = pd.DataFrame({
        "ts_code": ["600519.SH"] * 3,
        "trade_date": ["20260901", "20260902", "20260903"],
        "close": [100.0, 50.0, 51.0],          # raw: halves on the split
    })
    adj = pd.DataFrame({
        "ts_code": ["600519.SH"] * 3,
        "trade_date": ["20260901", "20260902", "20260903"],
        "adj_factor": [2.0, 1.0, 1.0],
    })
    monkeypatch.setitem(sys.modules, "data_manager", _tushare(daily, adj))
    got = wb._cn_bars(["600519.SH"], "20260901", "20260903")["600519.SH"]

    # Adjusted back to the latest factor: no step between day 1 and day 2.
    assert list(got) == pytest.approx([200.0, 50.0, 51.0]) or \
        got.iloc[0] == pytest.approx(200.0)
    assert wb._move(got, 1) == pytest.approx(2.0, abs=0.01)


def test_a_missing_adjustment_falls_back_to_raw_rather_than_failing(monkeypatch):
    daily = pd.DataFrame({
        "ts_code": ["600519.SH"] * 2,
        "trade_date": ["20260901", "20260902"],
        "close": [100.0, 110.0],
    })
    monkeypatch.setitem(sys.modules, "data_manager", _tushare(daily, None))
    got = wb._cn_bars(["600519.SH"], "20260901", "20260902")
    assert list(got["600519.SH"]) == pytest.approx([100.0, 110.0])


def test_codes_are_chunked_so_one_call_cannot_hit_the_row_cap(monkeypatch):
    """81 codes × 65 sessions came to 5,265 rows against a ~6,000 cap."""
    seen = []

    mod = types.ModuleType("data_manager")
    mod.init_tushare = lambda: True

    class Api:
        def daily(self, ts_code=None, **kw):
            seen.append(ts_code.split(","))
            return pd.DataFrame(columns=["ts_code", "trade_date", "close"])

        def adj_factor(self, **kw):
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])

    mod.TUSHARE_API = Api()
    monkeypatch.setitem(sys.modules, "data_manager", mod)

    wb._cn_bars([f"{600000 + i}.SH" for i in range(95)], "20260901", "20260930")
    assert len(seen) > 1
    assert all(len(c) <= wb.CHUNK for c in seen)
    assert sum(len(c) for c in seen) == 95


def test_no_codes_is_no_call(monkeypatch):
    def boom():
        raise AssertionError("should not have initialised Tushare")

    mod = types.ModuleType("data_manager")
    mod.init_tushare = boom
    monkeypatch.setitem(sys.modules, "data_manager", mod)
    assert wb._cn_bars([], "20260901", "20260930") == {}


# ── the scan this page never read ────────────────────────────────────────────
def test_signals_come_from_the_newest_scan_only(monkeypatch):
    rows = pd.DataFrame([
        {"user_id": 1, "scan_date": "2026-09-30", "ticker": "600031",
         "signals": "▲ MACD Bullish Crossover", "signal_count": 1,
         "rsi": 34.8},
        {"user_id": 1, "scan_date": "2026-09-29", "ticker": "600031",
         "signals": "▼ Bearish Squeeze Drop 🩸", "signal_count": 1,
         "rsi": 30.0},
    ])
    fake = types.ModuleType("db_manager")
    fake.db = types.SimpleNamespace(read_table=lambda *a, **k: rows)
    monkeypatch.setitem(sys.modules, "db_manager", fake)

    got = wb._signals(1)
    assert got["600031"]["scan_date"] == "2026-09-30"
    assert got["600031"]["rsi"] == pytest.approx(34.8)
    assert [s["label"] for s in got["600031"]["signals"]] == ["MACD 金叉"]


def test_signal_direction_travels_so_the_badge_can_be_coloured(monkeypatch):
    rows = pd.DataFrame([{
        "user_id": 1, "scan_date": "2026-09-30", "ticker": "X",
        "signals": "▼ Bearish Squeeze Drop 🩸", "signal_count": 1, "rsi": 20.0}])
    fake = types.ModuleType("db_manager")
    fake.db = types.SimpleNamespace(read_table=lambda *a, **k: rows)
    monkeypatch.setitem(sys.modules, "db_manager", fake)
    assert wb._signals(1)["X"]["signals"][0]["dir"] == "bear"


def test_a_missing_signals_table_loses_the_badges_not_the_board(monkeypatch):
    fake = types.ModuleType("db_manager")

    def boom(*a, **k):
        raise RuntimeError("PGRST205")

    fake.db = types.SimpleNamespace(read_table=boom)
    monkeypatch.setitem(sys.modules, "db_manager", fake)
    assert wb._signals(1) == {}


# ── the row ──────────────────────────────────────────────────────────────────
class _Session:
    """Stands in for session_state."""

    def __init__(self, complete):
        self.complete = complete

    def state(self, market, day):
        return types.SimpleNamespace(
            complete=self.complete, elapsed_pct=53.0, local_time="12:55")


def test_a_north_american_row_says_whether_its_bar_is_finished():
    """
    Yahoo serves a running bar while the session is open: the move is real,
    the day behind it is not over.
    """
    row = wb._row({"t": "US:NOK", "n": "Nokia", "market": "US", "added": None},
                  "NOK", _series([10.0, 10.8]), None, _Session(False))
    assert row["session"] == {"complete": False, "elapsed_pct": 53.0,
                              "local_time": "12:55"}


def test_an_a_share_row_carries_no_session_block():
    """Tushare publishes after the close, so there is nothing to qualify."""
    row = wb._row({"t": "600519", "n": "贵州茅台", "market": "CN", "added": None},
                  "600519.SH", _series([100.0, 101.0]), None, _Session(True))
    assert row["session"] is None


def test_a_row_with_no_price_still_appears():
    """A name added a minute ago must not vanish until the next fetch."""
    row = wb._row({"t": "US:NEW", "n": "New", "market": "US", "added": None},
                  "NEW", None, None, _Session(True))
    assert row["price"] is None and row["spark"] == []
    assert row["t"] == "US:NEW"


def test_the_sparkline_is_capped_at_the_window():
    row = wb._row({"t": "X", "n": "X", "market": "CN", "added": None}, "X.SH",
                  _series([float(i) for i in range(1, 200)]), None, _Session(True))
    assert len(row["spark"]) == wb.SPARK_POINTS
    assert row["spark"][-1] == pytest.approx(199.0)      # newest kept


def test_the_row_reports_which_bar_the_price_came_from():
    row = wb._row({"t": "X", "n": "X", "market": "CN", "added": None}, "X.SH",
                  _series([1.0, 2.0], start="20260901"), None, _Session(True))
    assert row["last_bar"] == "2026-09-02"


def test_scan_fields_are_attached_when_present():
    hit = {"scan_date": "2026-09-30", "rsi": 34.8, "signal_count": 2,
           "signals": [{"id": "macd_cross_up", "label": "MACD 金叉",
                        "dir": "bull", "group": "MACD"}]}
    row = wb._row({"t": "600031", "n": "三一重工", "market": "CN", "added": None},
                  "600031.SH", _series([1.0, 2.0]), hit, _Session(True))
    assert row["rsi"] == 34.8 and row["signal_count"] == 2
    assert row["signals"][0]["label"] == "MACD 金叉"
