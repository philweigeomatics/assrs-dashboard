"""index_audit — finding stored days that are missing, lagging, falsely halted or simply wrong."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import index_audit as IA  # noqa: E402

S = pd.DatetimeIndex(pd.to_datetime(["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"]))
CLOSES = pd.DataFrame({"A": [10.0, 11.0, 12.1, 12.1, 13.31], "B": [20.0, 20.0, 18.0, 19.8, 19.8]}, index=S)


def day(d, a, b, cash=0.0, h=None):
    """A stored day holding A and B at these values: the level is their sum plus cash."""
    st = {"v": {"A": a, "B": b}, "cash": cash}
    if h:
        st["h"] = h
    return {"d": d, "close": a + b + cash, "state": st}


def right():
    """Each day's values are yesterday's times the fresh returns: the record is correct."""
    out, a, b = [], 50.0, 50.0
    for i, s in enumerate(S):
        if i:
            a *= CLOSES["A"].iloc[i] / CLOSES["A"].iloc[i - 1]
            b *= CLOSES["B"].iloc[i] / CLOSES["B"].iloc[i - 1]
        out.append(day(str(s.date()), a, b))
    return out


def kinds(r):
    return [(i["d"], i["kind"]) for i in r["issues"]]


def test_a_correct_record_is_clean():
    r = IA.audit(right(), CLOSES, S)
    assert r["issues"] == [] and r["checked"] == 4 and r["behind"] == 0


def test_a_member_stored_as_halted_on_a_day_it_traded_is_found_with_the_wrong_level():
    # 11-19: B "halted" — held at 50 instead of falling 10% to 45.
    days = right()
    d = days[2]
    d["state"]["h"] = ["B"]
    d["state"]["v"]["B"] = 50.0
    d["close"] = d["state"]["v"]["A"] + 50.0
    r = IA.audit(days, CLOSES, S)
    assert ("2026-11-19", "halted") in kinds(r) and ("2026-11-19", "mismatch") in kinds(r)
    assert r["repair_from"] == "2026-11-19"


def test_a_missing_session_and_a_lagging_index_are_reported():
    days = right()
    del days[3]                                   # 11-20 never stored
    r = IA.audit(days[:-1], CLOSES, S)            # ...and 11-23 not yet either
    assert ("2026-11-20", "missing") not in kinds(r)    # after the last stored day it is "behind"
    assert r["behind"] == 2 and r["behind_from"] == "2026-11-20"
    r = IA.audit(days, CLOSES, S)
    assert ("2026-11-20", "missing") in kinds(r)


def test_a_genuine_halt_with_no_price_is_not_flagged():
    closes = CLOSES.copy()
    closes.loc["2026-11-19", "B"] = float("nan")             # suspended: no price that day
    days = right()
    days[2]["state"]["h"] = ["B"]
    days[2]["state"]["v"]["B"] = days[1]["state"]["v"]["B"]  # held flat, correctly
    days[2]["close"] = days[2]["state"]["v"]["A"] + days[2]["state"]["v"]["B"]
    # 11-20 is B's return from its last price (18 is gone: from 20) — rebuild it so.
    a = days[2]["state"]["v"]["A"] * 12.1 / 12.1
    b = days[2]["state"]["v"]["B"] * 19.8 / 20.0
    days[3] = day("2026-11-20", a, b)
    days[4] = day("2026-11-23", a * 13.31 / 12.1, b)
    r = IA.audit(days, closes, S)
    assert [k for k in kinds(r) if k[0] == "2026-11-19"] == []


def test_a_holding_the_download_no_longer_has_is_unverifiable_not_wrong():
    r = IA.audit(right(), CLOSES.drop(columns=["B"]), S)
    assert r["issues"] == [] and r["unverifiable"] == 4


def test_only_days_from_since_are_checked():
    days = right()
    days[1]["close"] *= 1.05                       # an old error on 11-18
    r = IA.audit(days, CLOSES, S, since="2026-11-20")
    assert all(i["d"] >= "2026-11-20" for i in r["issues"])


def test_a_stock_resuming_after_a_suspension_earns_its_gap_on_the_day_it_trades():
    """中微公司: no prices 12-24..12-31, back on 01-05 +14% — stored right, must not be flagged."""
    closes = CLOSES.copy()
    closes.loc["2026-11-18":"2026-11-19", "B"] = float("nan")     # suspended 11-18, 11-19
    days, a, b = [], 50.0, 50.0
    for i, s in enumerate(S):
        if i:
            a *= CLOSES["A"].iloc[i] / CLOSES["A"].iloc[i - 1]
        if str(s.date()) == "2026-11-20":
            b *= 19.8 / 20.0                                       # from its last price, 11-17
        elif str(s.date()) == "2026-11-23":
            b *= 19.8 / 19.8
        h = ["B"] if str(s.date()) in ("2026-11-18", "2026-11-19") else None
        days.append(day(str(s.date()), a, b, h=h))
    r = IA.audit(days, closes, S)
    assert r["issues"] == []
