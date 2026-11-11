"""
宏观数据 → 指数: which session a release lands on, per index, and its size.

Offline.

    python -m pytest api/tests/test_eco_impact.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import eco_impact as ei  # noqa: E402


def _ns(s: str) -> int:
    return int(pd.Timestamp(s).value)


def _px(days, start=100.0, step=0.01) -> pd.Series:
    idx = pd.DatetimeIndex(days)
    return pd.Series(start * (1 + step) ** np.arange(len(idx)), index=idx)


def no_calendar(market, start):
    return []


def board(code, days, now="2030-01-01T00:00:00+00:00", upcoming=no_calendar, px=None):
    return ei.Board(code, px if px is not None else _px(days), now_ns=_ns(now),
                    upcoming=upcoming)


# ── naming and values ────────────────────────────────────────────────────────
@pytest.mark.parametrize("event, series", [
    ("美国CPI月率(%)(九月)", "美国CPI月率(%)"),
    ("中国GDP年率(%)(第三季度)", "中国GDP年率(%)"),
    ("美国初请失业金人数(11月8日当周)", "美国初请失业金人数"),
    ("中国社会融资规模(人民币十亿元)(十二月)", "中国社会融资规模(人民币十亿元)"),
    ("美国季调后非农就业人口变动", "美国季调后非农就业人口变动"),
    ("美国密歇根大学消费者信心指数(11/14)", "美国密歇根大学消费者信心指数"),
])
def test_series_strips_only_the_period(event, series):
    assert ei.series_of(event) == series


@pytest.mark.parametrize("raw, value", [
    ("4.9%", 4.9), ("2,490.0B", 2.49e12), ("-12.5K", -12500.0), ("0.3", 0.3),
    (None, None), ("", None), ("--", None), (float("nan"), None),
])
def test_num(raw, value):
    assert ei.num(raw) == value


# ── the timing rule ──────────────────────────────────────────────────────────
US_DAYS = pd.bdate_range("2026-09-01", "2026-11-20")
CN_DAYS = pd.bdate_range("2026-09-01", "2026-11-20")


def test_us_release_before_the_open_lands_on_that_us_day_and_the_next_cn_day():
    # US CPI, 08:30 New York on Tuesday 10 November = 05:30 Vancouver.
    t = _ns("2026-11-10T08:30:00-05:00")
    assert board("^GSPC", US_DAYS).react(t)["day"] == "2026-11-10"
    # Shanghai closed at 15:00 Beijing on the 10th, long before; it reacts
    # on Wednesday the 11th.
    assert board("000300.SH", CN_DAYS).react(t)["day"] == "2026-11-11"


def test_us_release_during_the_us_session_still_counts_that_day():
    t = _ns("2026-11-10T10:00:00-05:00")          # ISM, 10:00 New York
    assert board("^GSPC", US_DAYS).react(t)["day"] == "2026-11-10"


def test_cn_morning_release_lands_on_that_day_for_both():
    # 中国CPI at 09:30 Beijing on Wednesday 11 November — 17:30 Tuesday in
    # Vancouver. Shanghai closes six hours later; New York that afternoon.
    t = _ns("2026-11-11T09:30:00+08:00")
    assert board("000300.SH", CN_DAYS).react(t)["day"] == "2026-11-11"
    assert board("^GSPC", US_DAYS).react(t)["day"] == "2026-11-11"


def test_release_after_the_a_share_close_is_the_next_session():
    t = _ns("2026-11-13T16:00:00+08:00")          # 社融, Friday 16:00 Beijing
    assert board("000300.SH", CN_DAYS).react(t)["day"] == "2026-11-16"
    # …and New York had not closed yet that Friday.
    assert board("^GSPC", US_DAYS).react(t)["day"] == "2026-11-13"


def test_release_exactly_at_the_close_is_the_next_session():
    t = _ns("2026-11-12T15:00:00+08:00")
    assert board("000300.SH", CN_DAYS).react(t)["day"] == "2026-11-13"


def test_weekend_release_waits_for_monday():
    t = _ns("2026-11-08T09:30:00+08:00")          # Sunday
    assert board("000300.SH", CN_DAYS).react(t)["day"] == "2026-11-09"
    assert board("^GSPC", US_DAYS).react(t)["day"] == "2026-11-09"


def test_a_shut_exchange_is_skipped_without_knowing_why():
    # 春节: no bars 2026-02-16 … 02-23. A release on the 13th after the close
    # belongs to the 24th.
    days = [d for d in pd.bdate_range("2026-01-05", "2026-03-06")
            if not (pd.Timestamp("2026-02-16") <= d <= pd.Timestamp("2026-02-23"))]
    t = _ns("2026-02-13T16:00:00+08:00")
    assert board("000300.SH", days).react(t)["day"] == "2026-02-24"


def test_new_york_close_follows_daylight_saving():
    # 16:00 New York is 20:00 UTC in summer and 21:00 UTC in winter. A release
    # at 20:30 UTC is after the close on 2 October, before it on 6 November.
    days = pd.bdate_range("2026-09-01", "2026-11-30")
    b = board("^GSPC", days)
    assert b.react(_ns("2026-10-02T20:30:00+00:00"))["day"] == "2026-10-05"
    assert b.react(_ns("2026-11-06T20:30:00+00:00"))["day"] == "2026-11-06"


def test_reaction_is_that_sessions_close_to_close_return():
    days = pd.bdate_range("2026-11-02", "2026-11-13")
    px = pd.Series([100, 101, 99, 99, 103, 103, 100, 102, 102, 102.0], index=days)
    b = board("000300.SH", days, px=px)
    r = b.react(_ns("2026-11-05T21:00:00+08:00"))   # after Thursday's close
    assert r["day"] == "2026-11-06" and r["ret"] == pytest.approx(4.04, abs=0.01)
    assert r["done"] is True


def test_before_the_first_bar_there_is_no_prior_close():
    days = pd.bdate_range("2026-11-02", "2026-11-13")
    assert board("000300.SH", days).react(_ns("2026-11-01T09:00:00+08:00")) is None


# ── a session that has not closed ────────────────────────────────────────────
def test_live_bar_is_pending_not_a_close():
    days = pd.bdate_range("2026-11-02", "2026-11-11")
    # It is 11:00 Beijing on the 11th: that bar is the session in progress.
    b = board("000300.SH", days, now="2026-11-11T11:00:00+08:00")
    r = b.react(_ns("2026-11-11T09:30:00+08:00"))
    assert r == {"day": "2026-11-11", "done": False}
    # The day before is final.
    assert b.react(_ns("2026-11-10T09:30:00+08:00"))["done"] is True


def test_future_release_reads_the_calendar_ahead():
    days = pd.bdate_range("2026-11-02", "2026-11-10")
    calls = []

    def upcoming(market, start):
        calls.append((market, start))
        return [date(2026, 11, 11), date(2026, 11, 12)]

    b = board("000300.SH", days, now="2026-11-10T20:00:00+08:00", upcoming=upcoming)
    assert calls == [("CN", date(2026, 11, 11))]
    assert b.react(_ns("2026-11-11T16:00:00+08:00")) == {"day": "2026-11-12", "done": False}
    assert b.react(_ns("2026-11-20T09:00:00+08:00")) is None


def test_upcoming_us_skips_nyse_holidays():
    got = ei.upcoming_sessions("US", date(2026, 11, 23), days=7)
    assert date(2026, 11, 26) not in got        # Thanksgiving
    assert date(2026, 11, 27) in got            # the half day still trades
    assert date(2026, 11, 28) not in got        # Saturday


@pytest.mark.parametrize("y, expect", [
    (2026, {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
            date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
            date(2026, 11, 26), date(2026, 12, 25)}),
    # New Year on a Saturday is not observed on the Friday before.
    (2022, {date(2022, 1, 17), date(2022, 2, 21), date(2022, 4, 15), date(2022, 5, 30),
            date(2022, 6, 20), date(2022, 7, 4), date(2022, 9, 5), date(2022, 11, 24),
            date(2022, 12, 26)}),
])
def test_nyse_holidays(y, expect):
    assert ei.nyse_holidays(y) == expect


# ── a normal day ─────────────────────────────────────────────────────────────
def test_normal_day_is_the_mean_so_an_ordinary_day_averages_one():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2024-01-01", periods=800)
    px = pd.Series(100 * np.exp(np.cumsum(rng.standard_t(4, 800) * 0.01)), index=days)
    b = board("^GSPC", days, px=px)
    xs = [b.react(ei.close_instants([d], "US")[0] - 3600 * 10**9)["x"] for d in days[100:]]
    assert np.mean(xs) == pytest.approx(1.0, abs=0.08)


def test_x_needs_enough_history():
    days = pd.bdate_range("2026-11-02", periods=10)
    r = board("^GSPC", days).react(_ns("2026-11-05T12:00:00-05:00"))
    assert r["ret"] is not None and r["x"] is None


# ── the week ─────────────────────────────────────────────────────────────────
def _ev(at, country, event, **kw):
    t = pd.Timestamp(at).tz_convert("Asia/Shanghai")
    return {"at": at, "date": t.strftime("%Y-%m-%d"), "time": t.strftime("%H:%M"),
            "country": country, "event": event, **kw}


def test_week_maps_only_us_and_china_with_a_time():
    evs = [
        _ev("2026-11-10T08:30:00-05:00", "美国", "美国CPI月率(%)(十月)"),
        _ev("2026-11-11T09:30:00+08:00", "中国", "中国CPI年率(%)(十月)"),
        _ev("2026-11-10T10:00:00+01:00", "德国", "德国ZEW经济景气指数"),
        {**_ev("2026-11-10T08:30:00-05:00", "美国", "美国假日"), "at": None},
        # eco_cal's country filter leaks; a row whose name is another
        # country's is not this country's release.
        _ev("2026-11-11T08:30:00+08:00", "中国", "澳大利亚失业率"),
    ]
    out = ei.week(evs, [board("^GSPC", US_DAYS), board("000300.SH", CN_DAYS)])
    assert [i["code"] for i in out["indexes"]] == ["^GSPC", "000300.SH"]
    assert len(out["reactions"]) == 2
    cpi = out["reactions"][ei.event_key(evs[0])]
    assert cpi["^GSPC"]["day"] == "2026-11-10" and cpi["000300.SH"]["day"] == "2026-11-11"


def test_event_key_matches_the_client():
    e = _ev("2026-11-10T08:30:00-05:00", "美国", "美国CPI月率(%)(十月)")
    assert ei.event_key(e) == "2026-11-10|21:30|美国|美国CPI月率(%)(十月)"


# ── one series' history ──────────────────────────────────────────────────────
def _row(at, value=None, forecast=None, event="中国社会融资规模(人民币十亿元)(十月)"):
    return {"at": at, "event": event, "value": value, "forecast": forecast, "prev": None}


def test_dedupe_drops_a_scheduled_twin_a_day_away():
    rows = [
        _row("2026-11-13T17:00:00+08:00"),                       # scheduled, no value
        _row("2026-11-14T09:00:00+08:00", "810.0B", "1,230.0B"),   # the real one
        _row("2026-12-12T17:00:00+08:00"),                       # not out yet: stays
    ]
    got = ei.dedupe(rows)
    assert [r["at"] for r in got] == ["2026-11-14T09:00:00+08:00", "2026-12-12T17:00:00+08:00"]


def test_dedupe_same_instant_any_offset():
    rows = [_row("2026-11-14T01:00:00+00:00", "810.0B"),
            _row("2026-11-14T09:00:00+08:00", "810.0B")]
    assert len(ei.dedupe(rows)) == 1


def test_dedupe_keeps_weekly_neighbours():
    rows = [_row(f"2026-11-{d:02d}T21:30:00+08:00", "220K", event="美国初请失业金人数")
            for d in (5, 12, 19)]
    assert len(ei.dedupe(rows)) == 3


def test_history_summary():
    days = pd.bdate_range("2026-01-05", "2026-06-30")
    px = pd.Series(100.0, index=days)
    # Up 2% on the two "above" sessions, down 1% on the "below" one.
    for d, r in (("2026-03-11", 0.02), ("2026-04-10", -0.01), ("2026-05-12", 0.02)):
        px.loc[d:] *= 1 + r
    px = px * (1 + 0.001 * np.sin(np.arange(len(px))))      # a little daily noise
    b = board("000300.SH", days, px=px)
    rows = [
        _row("2026-03-11T09:30:00+08:00", "0.5%", "0.3%", "中国CPI月率(%)(二月)"),
        _row("2026-04-10T09:30:00+08:00", "0.1%", "0.3%", "中国CPI月率(%)(三月)"),
        _row("2026-05-12T09:30:00+08:00", "0.4%", "0.2%", "中国CPI月率(%)(四月)"),
        _row("2026-06-10T09:30:00+08:00", "0.2%", "0.2%", "中国CPI月率(%)(五月)"),
        _row("2026-07-09T09:30:00+08:00", None, "0.1%", "中国CPI月率(%)(六月)"),  # future
    ]
    out = ei.history(rows, [b])
    assert [r["dir"] for r in out["rows"]] == [None, 0, 1, -1, 1]      # newest first
    assert out["rows"][0]["react"]["000300.SH"] is None                  # past the data
    s = out["summary"]["000300.SH"]
    assert s["n"] == 4
    assert s["above"]["n"] == 2 and s["above"]["ret"] == pytest.approx(2.0, abs=0.3)
    assert s["below"]["n"] == 1 and s["below"]["ret"] == pytest.approx(-1.0, abs=0.3)
    assert s["mean_x"] > 3                     # these sessions dwarf the noise


# ── storage ──────────────────────────────────────────────────────────────────
def test_rows_from_frame():
    df = pd.DataFrame([
        {"date": "20261110", "time": "21:30", "event": "美国CPI月率(%)(十月)",
         "value": "0.3%", "fore_value": "0.2%", "pre_value": "0.4%"},
        # The same release twice in one page (an upsert would reject it).
        {"date": "20261110", "time": "21:30", "event": "美国CPI月率(%)(十月)",
         "value": None, "fore_value": "0.2%", "pre_value": None},
        {"date": "20261110", "time": "", "event": "美国退伍军人节", "value": None,
         "fore_value": None, "pre_value": None},
        {"date": "20261110", "time": "08:30", "event": "澳大利亚就业人数", "value": "1K",
         "fore_value": None, "pre_value": None},
    ])
    rows = ei.rows_from_frame(df, "美国")
    assert rows == [{
        "country": "美国", "event": "美国CPI月率(%)(十月)", "series": "美国CPI月率(%)",
        "at": "2026-11-10T21:30:00+08:00", "value": "0.3%", "forecast": "0.2%",
        "prev": "0.4%"}]


def test_fetch_range_halves_a_full_page():
    class Pro:
        def __init__(self):
            self.calls = []

        def eco_cal(self, country, start_date, end_date):
            self.calls.append((start_date, end_date))
            n = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
            return pd.DataFrame({"x": range(min(100, 30 * n))})

    pro = Pro()
    df = ei.fetch_range(pro, "美国", date(2026, 11, 1), date(2026, 11, 4), gap_s=0)
    assert pro.calls[0] == ("20261101", "20261104")
    assert len(df) == 120                        # four days of 30, none lost


def test_load_series_says_how_to_create_a_missing_table():
    class DB:
        def read_all(self, *a, **k):
            raise RuntimeError("{'code': 'PGRST205', 'message': \"Could not find the "
                               "table 'public.eco_release' in the schema cache\"}")

    with pytest.raises(LookupError, match="20261111_eco_release.sql"):
        ei.load_series("美国", "美国CPI月率(%)", db=DB())
