"""
alerts_feed: parsing the nightly snapshot back into something filterable.

Offline — the rows are built by hand in the exact shape data_manager hands
back. The risk this guards is specific: the signals live inside ONE joined
string, so a parser that silently fails to match a label does not crash, it
just quietly drops that signal out of the filters and the stock stops
appearing when you ask for it.

    python -m pytest api/tests/test_alerts_feed.py -q
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import alerts_feed as af  # noqa: E402

BOX = "箱体下沿 [12.30-15.60] 位置5% 触4/3 质0.82"


def _row(ticker, name, bias, signals, count, price=10.0, rsi=50.0):
    return {"Ticker": ticker, "Name": name, "Type": bias, "Signals": signals,
            "Signal_Count": count, "Price": price, "RSI": rsi, "ADX": 20.0,
            "MACD": 0.1, "Volume": 1e6}


@pytest.fixture()
def feed():
    rows = pd.DataFrame([
        _row("000001", "甲", "🚀 Bullish", f"▲ MACD Bottoming{af.SEPARATOR}▲ {BOX}", 2),
        _row("000002", "乙", "⚠️ Bearish", f"▼ RSI Peaking{af.SEPARATOR}▼ MACD Peaking", 2),
        _row("000003", "丙", "⚖️ Mixed", f"▲ MACD Bottoming{af.SEPARATOR}▼ RSI Peaking", 2),
        _row("000004", "丁", "🚀 Bullish", "▲ DI Screaming Buy 🚀", 1),
    ])
    chips = pd.DataFrame([
        {"ticker": "000001", "decay": 1.0, "setup_score": 0.81, "setup_label": "🔴 单峰密集·上方轻",
         "winner_rate": 0.9, "concentration": 0.12, "pct_from_peak": -2.0,
         "n_peaks": 1, "peak_price": 10.2, "converged": True},
        {"ticker": "000002", "decay": 1.0, "setup_score": 0.31, "setup_label": "🟢 上方套牢重",
         "winner_rate": 0.1, "concentration": 0.30, "pct_from_peak": -30.0,
         "n_peaks": 3, "peak_price": 14.0, "converged": False},
    ])
    members = {"银行": {"000001", "000002"}, "白酒": {"000003"}}
    return af.build(rows, chips, members, "2026-09-15", age_days=1)


# ── parsing ──────────────────────────────────────────────────────────────────
def test_direction_marks_are_read_not_left_in_the_label():
    out = af.parse_signals(f"▲ MACD Bottoming{af.SEPARATOR}▼ RSI Peaking")
    assert [s["dir"] for s in out] == ["bull", "bear"]
    assert [s["cn"] for s in out] == ["MACD 筑底", "RSI 见顶"]
    assert all("▲" not in s["cn"] and "▼" not in s["cn"] for s in out)


def test_box_numbers_come_out_of_the_bracketed_tag():
    (s,) = af.parse_signals(f"▲ {BOX}")
    assert s["id"] == "box_support"
    assert s["group"] == af.GROUP_BOX
    assert s["detail"] == {
        "bot": 12.30, "top": 15.60, "position_pct": 5.0,
        "touches_top": 4, "touches_bot": 3, "quality": 0.82,
        # 15.60/12.30 - 1
        "height_pct": 26.8,
    }


def test_every_box_status_is_recognised_with_the_right_direction():
    for prefix, sid, direction in (("箱体下沿", "box_support", "bull"),
                                   ("箱体上沿", "box_resistance", "bear"),
                                   ("箱体突破", "box_breakout", "bull"),
                                   ("箱体跌破", "box_breakdown", "bear")):
        tag = f"{prefix} [10.00-12.00] 位置50% 触3/3 质0.70"
        (s,) = af.parse_signals(tag)
        assert (s["id"], s["dir"]) == (sid, direction)


def test_an_unknown_signal_is_surfaced_not_swallowed():
    """A signal nobody can filter by is still one you need to see."""
    (s,) = af.parse_signals("▲ Some Brand New Signal")
    assert s["id"] == "other"
    assert s["cn"] == "Some Brand New Signal"
    assert s["dir"] == "bull"


def test_empty_and_blank_strings_produce_nothing():
    assert af.parse_signals("") == []
    assert af.parse_signals("   ") == []


def test_catalog_must_cover_every_label_the_scanner_writes(monkeypatch):
    """
    The guard that matters: rename a signal in watchlist_scan and this module
    must fail loudly, not drop it from the filters.
    """
    import watchlist_scan as ws
    af._check_catalog()          # the real catalog is complete today

    monkeypatch.setitem(ws.BULL_BOOL, "MACD_Bottoming", "MACD Basing (renamed)")
    with pytest.raises(RuntimeError, match="MACD Basing"):
        af._check_catalog()


# ── the feed ─────────────────────────────────────────────────────────────────
def test_every_stock_survives_with_its_signals(feed):
    assert [s["t"] for s in feed["stocks"]] == ["000001", "000004", "000003", "000002"]
    first = feed["stocks"][0]
    assert [x["id"] for x in first["signals"]] == ["macd_bottom", "box_support"]


def test_bullish_first_then_most_signals(feed):
    """The scanner's own ranking, preserved."""
    biases = [s["bias"] for s in feed["stocks"]]
    assert biases == ["🚀 Bullish", "🚀 Bullish", "⚖️ Mixed", "⚠️ Bearish"]
    bulls = [s for s in feed["stocks"] if s["bias"] == "🚀 Bullish"]
    assert bulls[0]["signal_count"] >= bulls[-1]["signal_count"]


def test_chip_structure_is_joined_on_and_missing_is_none(feed):
    by = {s["t"]: s for s in feed["stocks"]}
    assert by["000001"]["chips"]["setup_score"] == 0.81
    assert by["000001"]["chip_shape"] == "🔴 单峰密集·上方轻"
    assert by["000002"]["chips"]["converged"] is False
    assert by["000003"]["chips"] is None
    assert by["000003"]["chip_shape"] is None


def test_sectors_are_attached_and_the_rest_are_reachable(feed):
    by = {s["t"]: s for s in feed["stocks"]}
    assert by["000001"]["sectors"] == ["银行"]
    assert by["000004"]["sectors"] == []

    names = {f["name"]: f["count"] for f in feed["facets"]["sectors"]}
    assert names["银行"] == 2
    # 000004 is in no sector index — without this facet the sector filter
    # would make it unreachable with no sign it exists.
    assert names[af.UNCLASSIFIED] == 1


def test_facet_counts_equal_what_filtering_would_return(feed):
    """
    A facet that says 3 and then returns 2 rows is worse than no facet. Counts
    are over the whole feed, so each one must match a scan of the stock list.
    """
    for f in feed["facets"]["signals"]:
        actual = sum(1 for s in feed["stocks"]
                     if any(x["id"] == f["id"] for x in s["signals"]))
        assert f["count"] == actual, f"signal facet {f['id']}"

    for f in feed["facets"]["sectors"]:
        actual = sum(1 for s in feed["stocks"]
                     if (s["sectors"] or [af.UNCLASSIFIED]).count(f["name"]))
        assert f["count"] == actual, f"sector facet {f['name']}"

    for f in feed["facets"]["bias"]:
        assert f["count"] == sum(1 for s in feed["stocks"] if s["bias"] == f["id"])

    for f in feed["facets"]["shapes"]:
        assert f["count"] == sum(1 for s in feed["stocks"] if s["chip_shape"] == f["name"])


def test_a_signal_firing_twice_counts_the_stock_once():
    """Otherwise the facet number stops matching the number of rows shown."""
    rows = pd.DataFrame([
        _row("000001", "甲", "🚀 Bullish",
             f"▲ MACD Bottoming{af.SEPARATOR}▲ MACD Bottoming", 2),
    ])
    out = af.build(rows, None, {}, "2026-09-15", age_days=0)
    facet = {f["id"]: f["count"] for f in out["facets"]["signals"]}
    assert facet["macd_bottom"] == 1
    assert len(out["stocks"][0]["signals"]) == 2   # both still shown


def test_staleness_tolerates_a_weekend_but_not_a_missed_run():
    rows = pd.DataFrame([_row("000001", "甲", "🚀 Bullish", "▲ MACD Bottoming", 1)])
    for age, stale in ((0, False), (3, False), (4, False), (5, True), (20, True)):
        out = af.build(rows, None, {}, "2026-09-15", age_days=age)
        assert out["stale"] is stale, f"age {age}"
        assert out["age_days"] == age


def test_an_empty_scan_is_a_valid_empty_feed():
    """A quiet market must render as "nothing fired", not as an error."""
    out = af.build(pd.DataFrame(columns=["Ticker", "Name", "Type", "Signals",
                                         "Signal_Count", "Price", "RSI", "ADX",
                                         "MACD", "Volume"]),
                   None, {}, "2026-09-15", age_days=1)
    assert out["stocks"] == []
    assert out["facets"]["signals"] == []
    assert out["scan_date"] == "2026-09-15"


def test_the_whole_feed_is_json_safe():
    import json
    rows = pd.DataFrame([_row("000001", "甲", "🚀 Bullish", f"▲ {BOX}", 1, price=float("nan"))])
    out = af.build(rows, None, {}, "2026-09-15", age_days=1)
    json.dumps(out, allow_nan=False)     # raises if a NaN slipped through
    assert out["stocks"][0]["price"] is None
