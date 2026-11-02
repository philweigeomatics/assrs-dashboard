"""
💳 杠杆周期 — the rules, the stage sequence, and the data plumbing.

The rules are worked through on series built to show each one; the stage
history is checked for hindsight (a day's verdict must not change when later
days arrive); the fetchers are checked for the two things that would quietly
corrupt the record — a half-reported day and a Shanghai series that changes
definition in 2019 — and the ETF flow for share splits.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import leverage_cycle as L  # noqa: E402

DAYS = pd.bdate_range("2013-01-02", "2020-12-31")


def market(balance=None, index=None, seed=0):
    """A calm market: everything drifting gently, so each test changes only what it is about."""
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    p = 3000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))) if index is None else index
    b = np.linspace(8000, 9000, n) if balance is None else balance
    return pd.DataFrame({
        "rzye": b, "rzmre": b * 0.06, "rzche": b * 0.06, "rqye": 50.0,
        "float_mv": 400000.0 * p / p[0], "amount": 6000.0, "idx_close": p,
        "etf_flow": rng.normal(0, 5, n),
    }, index=DAYS)


# ── the rules ────────────────────────────────────────────────────────────────
def test_a_falling_balance_well_off_its_high_is_deleveraging():
    n = len(DAYS)
    b = np.linspace(8000, 9000, n)
    b[-60:] = np.linspace(9000, 7800, 60)                     # −13% over three months
    s = L.signals(market(balance=b))
    assert s["1a"].iloc[-1] == 1 and s["1b"].iloc[-1] == 1
    assert s["1a"].iloc[-100] == 0 and s["1b"].iloc[-100] == 0


def test_a_new_index_low_the_debt_does_not_make_is_a_divergence():
    n = len(DAYS)
    p = np.full(n, 4000.0)
    p[-80:-40] = np.linspace(4000, 3000, 40)                  # first leg down
    p[-40:-20] = np.linspace(3000, 3300, 20)
    p[-20:] = np.linspace(3300, 2900, 20)                     # a lower low, today
    b = np.full(n, 9000.0)
    b[-80:-40] = np.linspace(9000, 8000, 40)                  # debt falls with the first leg
    b[-40:] = 8100.0                                          # and then holds above that low
    s = L.signals(market(balance=b, index=p))
    assert s["3a"].iloc[-1] == 1


def test_a_joint_new_low_today_undoes_an_earlier_divergence():
    """The latest index low is the one judged: an earlier divergence does not survive a joint low."""
    n = len(DAYS)
    p = np.full(n, 4000.0)
    p[-70:-20] = np.linspace(4000, 3000, 50)
    p[-20:-12] = np.linspace(2995, 2990, 8)                   # a low the debt does not follow …
    p[-12:] = np.linspace(2985, 2800, 12)                     # … then a lower one it does, today
    b = np.full(n, 9000.0)
    b[-70:-20] = np.linspace(9000, 8000, 50)
    b[-20:-12] = 8200.0
    b[-12:] = np.linspace(8000, 7500, 12)
    s = L.signals(market(balance=b, index=p))
    assert s["3a"].iloc[-13] == 1 and s["3a"].iloc[-1] == 0


def test_the_debt_turning_up_after_the_index_stops_falling():
    n = len(DAYS)
    p = np.full(n, 4000.0)
    p[-90:-15] = np.linspace(4000, 3000, 75)
    p[-15:] = np.linspace(3000, 3200, 15)                     # off the low for 15 sessions, +6.7%
    b = np.full(n, 9000.0)
    b[-90:-10] = np.linspace(9000, 7500, 80)
    b[-10:] = np.linspace(7500, 7700, 10)                     # turned up in the last 10
    s = L.signals(market(balance=b, index=p))
    assert s["3c"].iloc[-1] == 1
    assert s["3c"].iloc[-30] == 0                             # still falling then


def test_etf_inflows_need_both_size_and_persistence():
    m = market()
    m.loc[m.index[-20:], "etf_flow"] = 10.0
    assert L.signals(m)["4a"].iloc[-1] == 1
    m.loc[m.index[-20:-9], "etf_flow"] = -1.0                 # positive total, but only 9 days in
    assert L.signals(m)["4a"].iloc[-1] == 0


def test_the_worst_themes_refusing_new_lows():
    days = DAYS
    n = len(days)
    themes = pd.DataFrame({f"T{i}": np.linspace(100, 120, n) for i in range(12)}, index=days)
    for i in range(5):                                        # five leaders down …
        themes.iloc[-80:-20, i] = np.linspace(120, 70, 60)
        themes.iloc[-20:, i] = np.linspace(70, 78, 20)        # … that have stopped falling
    s = L.signals(market(), themes)
    assert s["4b"].iloc[-1] == 1
    themes.iloc[-5:, 0:2] = 60.0                              # two of them break down again
    assert L.signals(market(), themes)["4b"].iloc[-1] == 0


def test_a_rule_without_its_data_is_unknown_not_false():
    m = market()
    m["etf_flow"] = np.nan
    s = L.signals(m)
    assert s["4a"].isna().all() and s["4b"].isna().all()
    assert s["n4"].iloc[-1] == 1                              # only the volatility rule reaches


# ── the stages ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(6))
def test_every_confirmation_has_a_bottom_and_every_bottom_a_clear_out_before_it(seed):
    """Debt rising with the market calm is a bull market, not a bottom confirmed — unless a bottom came first."""
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    b = 9000 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))     # a wandering balance: cycles of its own
    m = market(balance=b, seed=seed)
    s = L.signals(m)
    c, b, r = (s[k].to_numpy() for k in ("is_clearing", "is_bottom", "is_right"))
    for i in np.flatnonzero(r):
        assert b[max(0, i - L.RIGHT_WINDOW + 1):i + 1].any()
    for i in np.flatnonzero(b):
        assert c[max(0, i - L.RIGHT_WINDOW + 1):i + 1].any()
    assert c.any()                                            # the scenario does exercise the stages


def test_low_leverage_and_a_flat_debt_without_a_clear_out_is_not_a_bottom():
    """② and ③ can hold in a market that never deleveraged; a bottom needs the clear-out before it."""
    n = len(DAYS)
    p = 3000 * np.exp(np.linspace(0, 1.2, n))                  # a long bull market …
    p[-10:] = p[-11] * np.linspace(1, 0.95, 10)               # … with a dip at the end
    b = np.linspace(8000, 9000, n)                            # debt growing slower than the market
    m = market(balance=b, index=p)
    m["rzmre"] = np.linspace(600, 300, n)                     # borrowed buying fading
    s = L.signals(m)
    assert s["L2"].iloc[-1] >= 2 and s["3b"].iloc[-1] == 1    # it looks like a bottom …
    assert s["is_clearing"].iloc[-200:].sum() == 0            # … with no clear-out before it
    assert not s["is_bottom"].iloc[-1]


def test_stages_follow_their_sequence():
    n = len(DAYS)
    p = np.full(n, 4000.0)
    p[-300:-120] = np.linspace(4000, 2600, 180)               # a long fall
    p[-120:-60] = 2600.0 * (1 + 0.002 * np.sin(np.arange(60)))
    p[-60:] = np.linspace(2600, 3000, 60)
    b = np.full(n, 12000.0)
    b[-300:-120] = np.linspace(12000, 7000, 180)
    b[-120:-40] = 7000.0
    b[-40:] = np.linspace(7000, 7600, 40)
    s = L.signals(market(balance=b, index=p))
    hits = {k: np.flatnonzero(s[f"is_{k}"].to_numpy()) for k in ("clearing", "bottom", "right")}
    first = {k: (int(v[0]) if len(v) else None) for k, v in hits.items()}
    assert first["clearing"] is not None and first["bottom"] is not None   # the scenario reaches a bottom
    assert first["clearing"] <= first["bottom"]
    if first["right"] is not None:
        assert first["bottom"] <= first["right"]
    assert (s["stage"].iloc[-320:] != "none").any()


def test_no_hindsight_a_days_verdict_never_changes_when_later_days_arrive():
    full = L.signals(market(seed=3))
    cut = DAYS[-400]
    part = L.signals(market(seed=3).loc[:cut])
    cols = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "4a", "4c", "R1_rank", "R2_rank", "stage"]
    a, b = full.loc[:cut, cols], part[cols]
    assert a.astype(str).equals(b.astype(str))


def test_ranks_start_in_2014_not_in_the_pilot_years():
    m = market()
    m.loc[:"2013-12-31", "rzye"] = 10.0                       # the pilot: almost no margin debt
    s = L.signals(m)
    assert s["R1_rank"].loc[:"2013-12-31"].isna().all()
    assert s["R1_rank"].loc["2016":].min() < 0.05            # the pilot years do not count as lows


def test_episodes_count_entries_not_days():
    s = pd.DataFrame({"stage": ["none"] * 100 + ["clearing"] * 30 + ["none"] * 10 + ["clearing"] * 5
                      + ["none"] * 40 + ["clearing"] * 5 + ["none"] * 300},
                     index=pd.bdate_range("2015-01-01", periods=490))
    p = pd.Series(np.linspace(100, 150, 490), s.index)
    e = L.episodes(s, p)
    assert e["clearing"]["n"] == 2                            # the 10-day gap is the same visit
    assert e["clearing"]["list"][-1]["date"] == str(s.index[100].date())    # the first visit, listed last
    assert e["base"]["median63"] is not None


def test_the_board_is_plain_json_from_2014():
    b = L.board(market())
    json.dumps(b, allow_nan=False)
    assert b["series"]["dates"][0] >= "2014-01-01"
    assert len(b["series"]["stage"]) == len(b["series"]["dates"])
    assert [l["key"] for l in b["layers"]] == ["1", "2", "3", "4"]
    assert b["stage"] in L.STAGES


# ── data plumbing ────────────────────────────────────────────────────────────
def test_a_share_split_is_not_a_flow():
    days = pd.bdate_range("2024-01-01", periods=4)
    shares = pd.DataFrame({"X": [1000.0, 1100.0, 2200.0, 2200.0]}, index=days)   # 万份
    closes = pd.DataFrame({"X": [4.0, 4.0, 2.0, 2.0]}, index=days)               # 1-for-2 split on day 3
    f = L.etf_flows(shares, closes)["X"]
    assert f.iloc[1] == pytest.approx(100 * 4.0 / 1e4)                           # 100万份 × 4元 = 0.04 亿
    assert f.iloc[2] == 0.0


class FakePro:
    def __init__(self, margin=None, info=None):
        self._margin, self._info, self.calls = margin, info or {}, []

    def margin(self, start_date, end_date, fields=None):
        self.calls.append(("margin", start_date, end_date))
        m = self._margin
        return m[(m.trade_date >= start_date) & (m.trade_date <= end_date)]

    def daily_info(self, ts_code, start_date, end_date, fields=None):
        self.calls.append(("info", ts_code, start_date, end_date))
        df = self._info.get(ts_code, pd.DataFrame(columns=["trade_date", "float_mv", "amount"]))
        return df[(df.trade_date >= start_date) & (df.trade_date <= end_date)]


def test_a_day_one_exchange_did_not_report_is_dropped_not_halved():
    m = pd.DataFrame([
        {"trade_date": "20240102", "exchange_id": "SSE", "rzye": 8e11, "rzmre": 4e10, "rzche": 4e10, "rqye": 1e9},
        {"trade_date": "20240102", "exchange_id": "SZSE", "rzye": 7e11, "rzmre": 3e10, "rzche": 3e10, "rqye": 1e9},
        {"trade_date": "20240103", "exchange_id": "SSE", "rzye": 8e11, "rzmre": 4e10, "rzche": 4e10, "rqye": 1e9},
    ])
    f = L.fetch_margin(FakePro(margin=m), "20240101", "20240131")
    assert list(f.index) == ["20240102"] and f.loc["20240102", "rzye"] == pytest.approx(15000)   # 亿元


def test_shanghai_is_spliced_at_2019_and_needs_shenzhen_on_the_same_day():
    row = lambda d, mv: {"trade_date": d, "float_mv": mv, "amount": 1000.0}
    info = {
        "SH_A": pd.DataFrame([row("20181228", 230000.0), row("20190102", 229000.0)]),
        "SH_MARKET": pd.DataFrame([row("20190102", 230000.0), row("20190103", 231000.0)]),
        "SZ_MARKET": pd.DataFrame([row("20181228", 120000.0), row("20190102", 121000.0)]),
    }
    pro = FakePro(info=info)
    f = L.fetch_market(pro, "20181201", "20190131")
    assert list(f.index) == ["20181228", "20190102"]                     # 01-03: no Shenzhen row
    assert f.loc["20181228", "float_mv"] == 350000.0                     # SH_A before the splice
    assert f.loc["20190102", "float_mv"] == 351000.0                     # SH_MARKET from it
    assert ("info", "SH_A", "20181201", "20181231") in pro.calls
