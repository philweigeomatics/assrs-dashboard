"""
日历 — the economic release feed and the A-share disclosure calendar.

Most of what is pinned here is a response to how Tushare's eco_cal behaves,
measured against the live API rather than read off the documentation:

  · at most 100 rows per call, whatever end_date says
  · `offset` pages, but the underlying order is not stable, so paging a
    multi-day range both duplicates and drops rows
  · 20 calls a minute

Hence one request per day, paged within the day. These tests exist so that
an "optimisation" back to a single ranged request fails loudly.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import calendar_data as cal  # noqa: E402


@pytest.fixture
def no_db(monkeypatch):
    """
    data_manager is the single source of truth for the ts_code mapping, but
    importing it opens a database connection. The mapping itself is pure
    string logic, so the stub reproduces it rather than the tests needing a
    live Supabase.
    """
    import types

    mod = types.ModuleType("data_manager")

    def to_ts(ticker):
        code = str(ticker).split(".")[0]
        if code.startswith(("43", "83", "87", "92")):
            return f"{code}.BJ"
        if code.startswith("6"):
            return f"{code}.SH"
        if code.startswith(("0", "2", "3")):
            return f"{code}.SZ"
        return code

    mod.get_tushare_ticker = to_ts
    monkeypatch.setitem(sys.modules, "data_manager", mod)
    return mod


class _Api:
    """Stands in for Tushare, with the 100-row ceiling reproduced."""

    def __init__(self, per_day: dict[str, int]):
        self.per_day = per_day
        self.calls: list[tuple[str, str, int]] = []

    def eco_cal(self, start_date=None, end_date=None, offset=0, **kw):
        self.calls.append((start_date, end_date, offset))
        total = self.per_day.get(start_date, 0)
        rows = [{"date": start_date, "time": f"{i // 60:02d}:{i % 60:02d}",
                 "currency": "USD", "country": "美国", "event": f"e{i}",
                 "value": None, "pre_value": None, "fore_value": None}
                for i in range(total)]
        window = rows[offset:offset + cal.PAGE]
        return pd.DataFrame(window)


@pytest.fixture(autouse=True)
def _no_cache():
    cal._cache.clear()
    yield
    cal._cache.clear()


def _use(monkeypatch, api):
    monkeypatch.setattr(cal, "_api", lambda: api)
    return api


# ── the 100-row ceiling ──────────────────────────────────────────────────────
def test_a_busy_day_is_paged_until_it_is_exhausted(monkeypatch):
    """
    30 September 2026 really did hold 147 distinct events. One call returns
    100 of them, and stopping there would drop a third of the day.
    """
    api = _use(monkeypatch, _Api({"20260930": 147}))
    rows = cal.eco_day(date(2026, 9, 30))
    assert len(rows) == 147
    assert [c[2] for c in api.calls] == [0, 100]


def test_a_quiet_day_costs_exactly_one_call(monkeypatch):
    api = _use(monkeypatch, _Api({"20260929": 59}))
    assert len(cal.eco_day(date(2026, 9, 29))) == 59
    assert len(api.calls) == 1


def test_a_day_landing_exactly_on_the_ceiling_is_still_checked(monkeypatch):
    """100 rows back is indistinguishable from "there may be more"."""
    api = _use(monkeypatch, _Api({"20260930": 100}))
    assert len(cal.eco_day(date(2026, 9, 30))) == 100
    assert len(api.calls) == 2      # the second comes back empty


def test_paging_a_single_day_is_bounded(monkeypatch):
    """A changed API must not turn one day into an unbounded loop."""
    api = _use(monkeypatch, _Api({"20260930": 10_000}))
    cal.eco_day(date(2026, 9, 30))
    assert len(api.calls) <= cal.MAX_PAGES


def test_each_day_is_requested_on_its_own(monkeypatch):
    """
    Never as a range. A request for 29 Sep – 1 Oct returned 100 rows when
    those days hold 253 events, so a ranged call silently loses data.
    """
    api = _use(monkeypatch, _Api({"20260929": 59, "20260930": 147,
                                  "20261001": 91}))
    cal.economic(date(2026, 9, 29), days=3)
    for start, end, _ in api.calls:
        assert start == end


def test_duplicate_rows_from_tushare_are_collapsed(monkeypatch):
    class Dupes(_Api):
        def eco_cal(self, start_date=None, **kw):
            row = {"date": start_date, "time": "09:00", "currency": "CNY",
                   "country": "中国", "event": "同一件事",
                   "value": None, "pre_value": None, "fore_value": None}
            return pd.DataFrame([row, row])

    _use(monkeypatch, Dupes({}))
    assert len(cal.eco_day(date(2026, 9, 29))) == 1


# ── caching ──────────────────────────────────────────────────────────────────
def test_a_finished_day_is_fetched_once_and_then_remembered(monkeypatch):
    """What makes paging back and forth through weeks free."""
    api = _use(monkeypatch, _Api({"20240102": 20}))
    cal.eco_day(date(2024, 1, 2))
    cal.eco_day(date(2024, 1, 2))
    assert len(api.calls) == 1


def test_a_week_gathers_every_day_in_its_padded_span(monkeypatch):
    """
    Grouping moved to the client when it turned out Tushare's dates are
    Beijing dates — see test_the_range_is_padded_so_any_viewer_gets_a_whole_week.
    What the server still owes is every event in the span, once.
    """
    _use(monkeypatch, _Api({"20260929": 3, "20260930": 2}))
    got = cal.economic(date(2026, 9, 29), days=2)
    assert got["total"] == 5
    assert {e["date"] for e in got["events"]} == {"2026-09-29", "2026-09-30"}
    assert all(e["at"] for e in got["events"])


def test_a_failing_day_is_reported_rather_than_returned_empty(monkeypatch):
    class Dead:
        def eco_cal(self, **kw):
            raise RuntimeError("频率超限")

    _use(monkeypatch, Dead())
    with pytest.raises(RuntimeError, match="经济日历读取失败"):
        cal.eco_day(date(2026, 9, 29))


# ── reporting periods ────────────────────────────────────────────────────────
def test_a_quarter_opens_before_it_closes(monkeypatch):
    """
    Measured on 29 September 2026, one day before Q3 closed: Tushare already
    held 2,323 disclosure rows for the period and the watchlist had 36
    October dates booked. Waiting for October hides the calendar during the
    fortnight it is most useful.
    """
    labels = [p["label"] for p in cal.periods(date(2026, 9, 29))]
    assert "2026 三季报" in labels


def test_a_quarter_that_is_still_far_off_is_not_offered():
    """Offering it empty reads as a failure rather than as "not yet"."""
    labels = [p["label"] for p in cal.periods(date(2026, 5, 15))]
    assert "2026 中报" not in labels
    assert "2026 三季报" not in labels


def test_periods_run_oldest_to_newest_so_the_last_is_the_current_one():
    got = cal.periods(date(2026, 11, 1))
    assert [p["end"] for p in got] == ["20251231", "20260331", "20260630",
                                       "20260930"]


# ── disclosure status ────────────────────────────────────────────────────────
TODAY = date(2026, 10, 15)


@pytest.mark.parametrize("actual, effective, want", [
    ("20261010", "20261010", "reported"),
    (None, "20261020", "scheduled"),
    (None, "20261010", "overdue"),      # the date passed, nothing filed
    (None, None, "unknown"),
    (None, "notadate", "unknown"),
])
def test_status_classification(actual, effective, want):
    assert cal._status(actual, effective, TODAY) == want


def test_overdue_is_its_own_state_not_folded_into_scheduled():
    """It is the one row on the page that needs chasing."""
    assert cal._status(None, "20261001", TODAY) == "overdue"
    assert cal._status(None, "20261101", TODAY) == "scheduled"


# ── the watchlist join ───────────────────────────────────────────────────────
def _frame(rows):
    return pd.DataFrame(rows, columns=["ts_code", "ann_date", "end_date",
                                       "pre_date", "actual_date"])


def test_only_a_shares_are_looked_up(monkeypatch, no_db):
    """A US holding has no A-share disclosure date; saying so would be noise."""
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20261001", "20260930", "20261025", None]]))
    # with_na=False keeps this about the Tushare join; the Yahoo side has its
    # own tests and would otherwise put this one on the network.
    got = cal.earnings("20260930", [
        {"t": "600519", "n": "贵州茅台"},
        {"t": "US:AAPL", "n": "Apple"},
    ], ref=TODAY, with_na=False)
    assert got["watched"]["cn"] == 1
    assert [r["n"] for r in got["cn_rows"]] == ["贵州茅台"]


def test_the_shown_date_prefers_what_actually_happened(monkeypatch, no_db):
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20261001", "20260930", "20261025", "20261022"]]))
    row = cal.earnings("20260930", [{"t": "600519", "n": "贵州茅台"}],
                       ref=TODAY, with_na=False)["rows"][0]
    assert row["date"] == "2026-10-22"
    assert row["status"] == "reported"
    assert row["moved"] is True          # filed early, against its own estimate


def test_a_company_that_kept_its_date_is_not_flagged(monkeypatch, no_db):
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20261001", "20260930", "20261025", "20261025"]]))
    assert cal.earnings("20260930", [{"t": "600519", "n": "贵州茅台"}],
                        ref=TODAY, with_na=False)["rows"][0]["moved"] is False


def test_a_watched_stock_with_no_row_is_named_not_silently_absent(monkeypatch, no_db):
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20261001", "20260930", "20261025", None]]))
    got = cal.earnings("20260930", [
        {"t": "600519", "n": "贵州茅台"}, {"t": "000001", "n": "平安银行"},
    ], ref=TODAY, with_na=False)
    assert got["missing"]["cn"] == ["平安银行"]


def test_rows_are_bucketed_by_date_for_the_grid(monkeypatch, no_db):
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20261001", "20260930", "20261025", None],
        ["000001.SZ", "20261001", "20260930", "20261025", None],
        ["600036.SH", "20261001", "20260930", "20261026", None]]))
    got = cal.earnings("20260930", [
        {"t": "600519", "n": "贵州茅台"}, {"t": "000001", "n": "平安银行"},
        {"t": "600036", "n": "招商银行"}], ref=TODAY, with_na=False)
    assert [(b["date"], len(b["rows"])) for b in got["by_date"]] == [
        ("2026-10-25", 2), ("2026-10-26", 1)]


def test_an_empty_watchlist_is_not_a_tushare_call(monkeypatch, no_db):
    def boom(period):
        raise AssertionError("should not have been called")

    monkeypatch.setattr(cal, "_disclosures", boom)
    got = cal.earnings("20260930", [{"t": "US:AAPL", "n": "Apple"}],
                       ref=TODAY, with_na=False)
    assert got["cn_rows"] == [] and got["watched"]["cn"] == 0


# ── the month the grid opens on ──────────────────────────────────────────────
def test_the_calendar_opens_on_the_month_that_holds_the_disclosures(monkeypatch, no_db):
    """
    H1 2026 put 80 of 81 watchlist disclosures in August and one in July.
    Opening on today's month would show an empty grid a page away from
    everything, which is why the Streamlit page picked the busiest month too.
    """
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20260801", "20260630", "20260731", None],
        ["000001.SZ", "20260801", "20260630", "20260826", None],
        ["600036.SH", "20260801", "20260630", "20260826", None]]))
    got = cal.earnings("20260630", [
        {"t": "600519", "n": "贵州茅台"}, {"t": "000001", "n": "平安银行"},
        {"t": "600036", "n": "招商银行"}], ref=date(2026, 10, 20), with_na=False)
    assert got["focus"] == "2026-08"
    assert got["months"] == [{"ym": "2026-08", "count": 2},
                             {"ym": "2026-07", "count": 1}]


def test_a_period_with_no_dates_still_names_a_month_to_open_on(monkeypatch, no_db):
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([]))
    got = cal.earnings("20260930", [{"t": "600519", "n": "贵州茅台"}],
                       ref=date(2026, 10, 20), with_na=False)
    assert got["months"] == []
    assert got["focus"] == "2026-10"      # falls back to the reference month


# ── Beijing time, including the date ─────────────────────────────────────────
def test_every_event_carries_an_unambiguous_instant(monkeypatch):
    """
    Tushare gives Beijing time and says so nowhere. Without the offset the
    client has to guess, and 28% of a real week guessed wrong.
    """
    _use(monkeypatch, _Api({"20261008": 1}))
    row = cal.eco_day(date(2026, 10, 8))[0]
    assert row["at"] == "2026-10-08T00:00:00+08:00"


@pytest.mark.parametrize("day, clock, want", [
    ("2026-10-08", "02:00", "2026-10-08T02:00:00+08:00"),
    ("2026-03-12", "20:30", "2026-03-12T20:30:00+08:00"),
    ("2026-10-08", "", None),          # no time given
    ("2026-10-08", "xx:yy", None),     # unparseable
    ("not-a-date", "02:00", None),
])
def test_instant_building(day, clock, want):
    assert cal._instant(day, clock) == want


def test_an_event_with_no_time_keeps_its_beijing_date(monkeypatch):
    """
    Placing it at midnight would shift it into another day for half the
    world; falling back to the label is the honest answer.
    """
    class NoTime(_Api):
        def eco_cal(self, start_date=None, **kw):
            return pd.DataFrame([{"date": start_date, "time": "",
                                  "currency": "CNY", "country": "中国",
                                  "event": "全天事件", "value": None,
                                  "pre_value": None, "fore_value": None}])

    _use(monkeypatch, NoTime({}))
    row = cal.eco_day(date(2026, 10, 8))[0]
    assert row["at"] is None and row["date"] == "2026-10-08"


def test_the_range_is_padded_so_any_viewer_gets_a_whole_week(monkeypatch):
    """
    Beijing is UTC+8 and the world runs UTC-12..UTC+14, so an event moves at
    most one day either way. Without the padding the first and last columns
    would be short for everyone outside UTC+8.
    """
    api = _use(monkeypatch, _Api({}))
    got = cal.economic(date(2026, 10, 5), days=7)

    asked = {c[0] for c in api.calls}
    assert "20261004" in asked and "20261012" in asked      # a day either side
    assert len(asked) == 9
    assert got["from"] == "2026-10-05" and got["to"] == "2026-10-11"
    assert got["fetched_from"] == "2026-10-04"
    assert got["fetched_to"] == "2026-10-12"


def test_the_payload_names_the_source_timezone(monkeypatch):
    """Stated rather than assumed — the whole bug was an unstated assumption."""
    _use(monkeypatch, _Api({}))
    assert cal.economic(date(2026, 10, 5), days=7)["source_tz"] == "Asia/Shanghai"


def test_events_come_back_flat_for_the_client_to_group(monkeypatch):
    """
    Which day an event belongs to depends on who is looking, so the server
    does not decide.
    """
    _use(monkeypatch, _Api({"20261005": 2, "20261006": 3}))
    got = cal.economic(date(2026, 10, 5), days=7)
    assert isinstance(got["events"], list)
    assert got["total"] == len(got["events"]) == 5
    assert "days" not in got


# ── US and Canadian earnings ─────────────────────────────────────────────────
def _yf(frames):
    """Stand in for yfinance; `frames` maps a bare code to a DataFrame."""
    import types

    class Ticker:
        def __init__(self, code):
            self.code = code

        def get_earnings_dates(self, limit=16):
            return frames.get(self.code)

    mod = types.ModuleType("yfinance")
    mod.Ticker = Ticker
    return mod


def _edf(rows):
    """
    [(iso stamp, est, reported, surprise)] → what yfinance returns.

    Parsed through UTC then converted, because a DatetimeIndex cannot hold
    mixed offsets — and a range spanning a DST change has them. yfinance
    returns one market timezone for the same reason.
    """
    idx = pd.DatetimeIndex(
        [pd.Timestamp(r[0]) for r in rows], name="Earnings Date", tz="UTC"
    ).tz_convert("America/New_York")
    return pd.DataFrame(
        {"EPS Estimate": [r[1] for r in rows],
         "Reported EPS": [r[2] for r in rows],
         "Surprise(%)": [r[3] for r in rows]}, index=idx)


@pytest.mark.parametrize("stamp, want", [
    ("2026-10-29 08:00:00-04:00", "盘前"),
    ("2026-10-29 07:00:00-05:00", "盘前"),
    ("2026-10-29 09:29:00-04:00", "盘前"),
    ("2026-10-29 16:00:00-04:00", "盘后"),
    ("2026-10-29 15:00:00-05:00", "盘中"),
    ("2026-10-29 12:00:00-04:00", "盘中"),
])
def test_the_hour_says_before_or_after_the_bell(stamp, want):
    """
    The part a date alone loses. 08:00 means the print lands before the open
    and the gap is tomorrow's; 16:00 means it lands after the close.
    """
    assert cal._when(pd.Timestamp(stamp)) == want


def test_na_earnings_reads_yahoo_for_north_american_names(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _yf({
        "AAPL": _edf([("2026-10-29 16:00:00-04:00", 1.98, None, None),
                      ("2026-07-30 16:00:00-04:00", 1.89, 2.02, 6.74)]),
    }))
    got = cal.na_earnings([{"t": "US:AAPL", "n": "Apple"}],
                          ref=date(2026, 10, 1))
    assert got["watched"] == 1
    assert [r["date"] for r in got["rows"]] == ["2026-07-30", "2026-10-29"]
    assert [r["status"] for r in got["rows"]] == ["reported", "scheduled"]
    assert got["rows"][0]["when"] == "盘后"
    assert got["rows"][0]["surprise_pct"] == pytest.approx(6.74)


def test_a_share_names_are_not_sent_to_yahoo(monkeypatch):
    """Tushare's filing calendar covers them; Yahoo has nothing to add."""
    monkeypatch.setitem(sys.modules, "yfinance", _yf({}))
    got = cal.na_earnings([{"t": "600519", "n": "贵州茅台"}],
                          ref=date(2026, 10, 1))
    assert got["watched"] == 0 and got["rows"] == []


def test_an_etf_with_no_earnings_is_named_not_an_error(monkeypatch):
    """
    Two of 23 real watchlist names were ETFs. They do not report earnings, so
    an empty answer is correct and belongs in `missing`, not in a log.
    """
    monkeypatch.setitem(sys.modules, "yfinance", _yf({"KLIP": None}))
    got = cal.na_earnings([{"t": "US:KLIP", "n": "KraneShares"}],
                          ref=date(2026, 10, 1))
    assert got["rows"] == [] and got["missing"] == ["KraneShares"]


def test_a_broken_yahoo_call_loses_one_name_not_the_calendar(monkeypatch):
    import types

    class Dead:
        def __init__(self, code):
            pass

        def get_earnings_dates(self, limit=16):
            raise RuntimeError("429")

    mod = types.ModuleType("yfinance")
    mod.Ticker = Dead
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    got = cal.na_earnings([{"t": "US:AAPL", "n": "Apple"}],
                          ref=date(2026, 10, 1))
    assert got["rows"] == [] and got["missing"] == ["Apple"]


def test_dates_far_outside_the_window_are_dropped(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _yf({
        "AAPL": _edf([("2019-01-29 16:00:00-05:00", 1.0, 1.1, 10.0),
                      ("2026-10-29 16:00:00-04:00", 1.98, None, None)]),
    }))
    got = cal.na_earnings([{"t": "US:AAPL", "n": "Apple"}],
                          ref=date(2026, 10, 1))
    assert [r["date"] for r in got["rows"]] == ["2026-10-29"]


# ── one calendar, both markets ───────────────────────────────────────────────
def test_both_markets_land_in_the_same_day_bucket(monkeypatch, no_db):
    """
    A real October had 金盘科技 and ASML on the same day. One calendar, or the
    user reads two.
    """
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20261001", "20260930", "20261014", None]]))
    monkeypatch.setitem(sys.modules, "yfinance", _yf({
        "ASML": _edf([("2026-10-14 16:00:00-04:00", 6.6, None, None)]),
    }))
    got = cal.earnings("20260930", [
        {"t": "600519", "n": "贵州茅台"}, {"t": "US:ASML", "n": "ASML"},
    ], ref=date(2026, 10, 1))

    day = next(b for b in got["by_date"] if b["date"] == "2026-10-14")
    assert {r["market"] for r in day["rows"]} == {"CN", "NA"}
    assert got["watched"] == {"cn": 1, "na": 1, "total": 2}


def test_every_row_carries_a_market_so_the_chip_can_say_which(monkeypatch, no_db):
    monkeypatch.setattr(cal, "_disclosures", lambda p: _frame([
        ["600519.SH", "20261001", "20260930", "20261014", None]]))
    monkeypatch.setitem(sys.modules, "yfinance", _yf({
        "ASML": _edf([("2026-10-20 16:00:00-04:00", 6.6, None, None)])}))
    got = cal.earnings("20260930", [
        {"t": "600519", "n": "贵州茅台"}, {"t": "US:ASML", "n": "ASML"},
    ], ref=date(2026, 10, 1))
    assert all(r.get("market") in ("CN", "NA") for r in got["rows"])


def test_a_watchlist_with_no_a_shares_still_gets_a_calendar(monkeypatch, no_db):
    """The empty-watchlist path used to return nothing at all."""
    monkeypatch.setitem(sys.modules, "yfinance", _yf({
        "ASML": _edf([("2026-10-20 16:00:00-04:00", 6.6, None, None)])}))
    got = cal.earnings("20260930", [{"t": "US:ASML", "n": "ASML"}],
                       ref=date(2026, 10, 1))
    assert len(got["na_rows"]) == 1
    assert got["focus"] == "2026-10"
    assert got["watched"]["total"] == 1
