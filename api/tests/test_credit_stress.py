"""
🏦 信用与债务 — the six rules, the publication lags that keep the record
honest, and the record itself.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import credit_stress as C  # noqa: E402

DAYS = pd.bdate_range("2004-01-02", "2012-12-31")


def us(baa=None, bizd=None, bkln=None, nfci=None, sloos=None, seed=0):
    """A calm market with every input; each test changes only what it is about."""
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), DAYS)
    px = pd.DataFrame({
        "SPY": spy,
        "BIZD": pd.Series(20.0 if bizd is None else bizd, DAYS),
        "BKLN": pd.Series(25.0 if bkln is None else bkln, DAYS),
        "HYG": 80.0, "LQD": 110.0, "ORCL": 50.0, "CRWV": np.nan,
    }, index=DAYS)
    weeks = pd.date_range(DAYS[0], DAYS[-1], freq="W-FRI")
    quarters = pd.date_range("2003-01-01", DAYS[-1], freq="QS")
    fred = {
        "BAA10Y": pd.Series(2.0 if baa is None else baa, DAYS, dtype=float),
        "BAMLH0A0HYM2": pd.Series(4.0, DAYS), "BAMLC0A0CM": pd.Series(1.2, DAYS),
        "BAMLC0A4CBBB": pd.Series(1.6, DAYS),
        "NFCI": pd.Series(-0.5 if nfci is None else nfci, weeks, dtype=float),
        "DRTSCILM": pd.Series(0.0 if sloos is None else sloos, quarters, dtype=float),
        "BCNSDODNS": pd.Series(np.linspace(8e6, 1e7, len(quarters)), quarters),
        "QCNPAM770A": pd.Series(150.0, quarters), "QUSPAM770A": pd.Series(140.0, quarters),
    }
    return {"fred": fred, "px": px}


def last(us_):
    return C.signals(C.frame(us_)).iloc[-1]


def test_a_calm_market_holds_nothing():
    s = last(us())
    assert s[[k for k, _, _ in C.RULES]].sum() == 0 and s["level"] == "calm"


def test_each_rule_fires_on_its_own_case():
    n = len(DAYS)
    baa = np.full(n, 2.0)
    baa[-20:] = 2.7                                                  # +70 bp from the 3-month low
    assert last(us(baa=baa))["c1"] == 1
    hi = np.r_[np.full(n - 600, 1.5), np.full(600, 4.0)]            # the last 600 days are the widest ever
    assert last(us(baa=hi))["c2"] == 1
    b = np.full(n, 20.0)
    b[-30:] = 15.0                                                   # −25% from its high
    assert last(us(bizd=b))["c3"] == 1
    k = np.full(n, 25.0)
    k[-30:] = 23.5                                                   # −6%
    assert last(us(bkln=k))["c4"] == 1
    weeks = pd.date_range(DAYS[0], DAYS[-1], freq="W-FRI")
    nf = np.full(len(weeks), -0.6)
    nf[-5:] = -0.3                                                   # +0.3 in 13 weeks, still below 0
    assert last(us(nfci=nf))["c5"] == 1
    quarters = pd.date_range("2003-01-01", DAYS[-1], freq="QS")
    sl = np.zeros(len(quarters))
    sl[-3:] = 35.0
    assert last(us(sloos=sl))["c6"] == 1


def test_three_rules_make_it_stress_two_make_it_weak():
    n = len(DAYS)
    baa = np.full(n, 2.0); baa[:700] = 4.0; baa[-20:] = 2.7         # widening, but not wide by history
    b = np.full(n, 20.0); b[-30:] = 15.0
    k = np.full(n, 25.0); k[-30:] = 23.5
    assert last(us(baa=baa))["c2"] == 0
    assert last(us(baa=baa, bizd=b))["level"] == "weak"
    assert last(us(baa=baa, bizd=b, bkln=k))["level"] == "stress"


def test_a_series_enters_on_the_day_it_was_published():
    """The survey dated 1 October is not known on 1 October: about five weeks later."""
    quarters = pd.date_range("2003-01-01", DAYS[-1], freq="QS")
    sl = np.zeros(len(quarters))
    q = list(quarters).index(pd.Timestamp("2012-10-01"))
    sl[q] = 50.0
    f = C.frame(us(sloos=sl))
    assert f.loc["2012-10-15", "DRTSCILM"] == 0.0                  # still the July survey
    assert f.loc["2012-11-13", "DRTSCILM"] == 50.0                 # 40 days on


def test_no_hindsight_a_days_reading_never_changes_when_later_days_arrive():
    rng = np.random.default_rng(5)
    n = len(DAYS)
    baa = 2 + np.cumsum(rng.normal(0, 0.03, n)).clip(-1.5, 3)
    full = us(baa=baa, seed=5)
    cut = DAYS[-500]
    part = {"fred": {k: v.loc[:cut] for k, v in full["fred"].items()}, "px": full["px"].loc[:cut]}
    a = C.signals(C.frame(full)).loc[:cut]
    b = C.signals(C.frame(part))
    cols = ["c1", "c2", "c3", "c4", "c5", "c6", "level"]
    assert a[cols].astype(str).equals(b[cols].astype(str))


def test_the_record_counts_entries_and_compares_with_any_day():
    idx = pd.bdate_range("2010-01-01", periods=400)
    level = pd.Series(["calm"] * 100 + ["stress"] * 30 + ["calm"] * 10 + ["stress"] * 5 + ["calm"] * 255, idx)
    spy = pd.Series(np.linspace(100, 140, 400), idx)
    r = C.record(level, spy)
    assert r["stress"]["n"] == 1 and r["weak"]["n"] == 1             # a 10-day gap is the same visit
    assert r["stress"]["list"][0]["date"] == str(idx[100].date())
    assert r["base"]["median63"] is not None and r["stress"]["dd63"] == 0.0


def test_the_board_is_plain_json():
    b = C.board(us(), {"tsf_yoy": pd.Series([9.0, 8.5], pd.to_datetime(["2026-08-01", "2026-09-01"]))})
    json.dumps(b, allow_nan=False)
    assert [r["key"] for r in b["rules"]] == ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert b["level"] == "calm" and b["china"]["tsf_yoy"]["v"] == [9.0, 8.5]
