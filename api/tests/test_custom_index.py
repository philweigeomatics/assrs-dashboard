"""
自建指数 — the ledger behind a custom index.

Every case here is small enough to work out by hand; the ones that matter
most are pinned by name: the level never jumps at a change, contribution
adds up to the move, buy-and-hold is never rebalanced behind your back,
and a halted stock is carried rather than traded.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import custom_index as C  # noqa: E402

IDX = pd.bdate_range("2025-01-02", "2025-06-30")


def frame(**series):
    return pd.DataFrame({k: pd.Series(v, IDX[:len(v)]) if not isinstance(v, pd.Series) else v
                         for k, v in series.items()}).reindex(IDX)


def mem(*tickers, added="2025-01-02", **extra):
    return [{"ticker": t, "added_on": added, "removed_on": None, **extra} for t in tickers]


def run(closes, members, method="equal", rebalance="M", **kw):
    return C.build(closes, members, method=method, rebalance=rebalance, base_date="2025-01-02", **kw)


def path(start, end, n=len(IDX)):
    return np.linspace(start, end, n)


# ── the methods ──────────────────────────────────────────────────────────────
def test_buy_and_hold_is_the_average_of_price_relatives():
    a, b = path(10, 30), path(50, 40)
    r = run(frame(A=a, B=b), mem("A", "B"), method="equal_hold")
    expect = 1000 * (a / a[0] + b / b[0]) / 2
    assert np.allclose(r["level"].to_numpy(), expect)


def test_equal_weight_resets_at_each_new_month():
    a = np.r_[np.linspace(10, 20, 21), np.full(len(IDX) - 21, 20.0)]   # doubles in January
    r = run(frame(A=a, B=np.full(len(IDX), 5.0)), mem("A", "B"))
    w = r["weights"]
    jan_end = w.loc["2025-01-31"]
    assert jan_end["A"] > 0.6                                   # drifted
    feb1 = w.loc["2025-02-03"]
    assert feb1["A"] == pytest.approx(0.5) and feb1["B"] == pytest.approx(0.5)
    assert any(e["kind"] == "rebalance" and e["date"] == "2025-02-03" for e in r["events"])


def test_rebalancing_changes_the_result_versus_holding():
    rng = np.random.default_rng(3)
    a = 10 * np.exp(np.cumsum(rng.normal(0, 0.03, len(IDX))))
    b = 10 * np.exp(np.cumsum(rng.normal(0, 0.03, len(IDX))))
    eq = run(frame(A=a, B=b), mem("A", "B"), method="equal")["level"]
    hold = run(frame(A=a, B=b), mem("A", "B"), method="equal_hold")["level"]
    assert abs(eq.iloc[-1] - hold.iloc[-1]) > 1


def test_never_rebalancing_an_equal_weight_index_is_buy_and_hold():
    a, b = path(10, 30), path(50, 40)
    never = run(frame(A=a, B=b), mem("A", "B"), method="equal", rebalance="never")["level"]
    hold = run(frame(A=a, B=b), mem("A", "B"), method="equal_hold")["level"]
    assert np.allclose(never, hold)


def test_a_newcomer_to_an_unrebalanced_index_does_not_reset_the_others():
    """Buy-and-hold means the winners keep their lead when someone joins."""
    a = np.r_[np.linspace(10, 30, 40), np.full(len(IDX) - 40, 30.0)]
    b = np.full(len(IDX), 10.0)
    members = mem("A", "B") + [{"ticker": "C", "added_on": "2025-03-10"}]
    for method, rebalance in (("equal_hold", "M"), ("equal", "never")):
        w = run(frame(A=a, B=b, C=b), members, method=method, rebalance=rebalance)["weights"].loc["2025-03-10"]
        assert w["C"] == pytest.approx(1 / 3)
        assert w["A"] / w["B"] == pytest.approx(3.0)


def test_quarterly_rebalances_only_at_quarter_starts():
    rng = np.random.default_rng(9)
    cl = frame(A=10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(IDX)))), B=np.full(len(IDX), 5.0))
    r = run(cl, mem("A", "B"), rebalance="Q")
    assert [e["date"] for e in r["events"] if e["kind"] == "rebalance"] == ["2025-04-01"]


def test_cap_weight_follows_the_caps():
    caps = frame(A=np.full(len(IDX), 300.0), B=np.full(len(IDX), 100.0))
    r = run(frame(A=np.full(len(IDX), 10.0), B=np.full(len(IDX), 10.0)), mem("A", "B"), method="cap", caps=caps)
    assert r["weights"].iloc[0]["A"] == pytest.approx(0.75)


def test_capped_weights_respect_the_cap_and_redistribute():
    w = C.capped({"A": 80, "B": 10, "C": 5, "D": 5, "E": 5, "F": 5, "G": 5, "H": 5, "I": 5, "J": 5, "K": 5, "L": 5}, 0.10)
    assert max(w.values()) <= 0.10 + 1e-9
    assert sum(w.values()) == pytest.approx(1)
    assert w["A"] == pytest.approx(0.10)
    assert w["C"] == pytest.approx(w["D"])                      # the uncapped keep their proportions


def test_a_cap_that_cannot_hold_is_equal_weight():
    w = C.capped({"A": 9, "B": 1, "C": 1}, 0.10)
    assert all(v == pytest.approx(1 / 3) for v in w.values())


def test_price_weight_holds_equal_units():
    r = run(frame(A=np.full(len(IDX), 300.0), B=np.full(len(IDX), 100.0)), mem("A", "B"), method="price")
    assert r["weights"].iloc[0]["A"] == pytest.approx(0.75)


def test_custom_weights_are_restored_at_rebalance():
    a = path(10, 20)
    r = run(frame(A=a, B=np.full(len(IDX), 5.0)),
            [{"ticker": "A", "added_on": "2025-01-02", "weight_pct": 70},
             {"ticker": "B", "added_on": "2025-01-02", "weight_pct": 30}], method="custom")
    assert r["weights"].loc["2025-03-03"]["A"] == pytest.approx(0.70)


def test_missing_caps_are_guessed_and_named():
    caps = frame(A=np.full(len(IDX), 300.0))
    r = run(frame(A=np.full(len(IDX), 10.0), B=np.full(len(IDX), 10.0)), mem("A", "B"), method="cap", caps=caps)
    assert r["guessed_caps"] == ["B"]


# ── changes never move the level ─────────────────────────────────────────────
def test_adding_a_member_does_not_move_the_level_and_it_rides_from_the_next_day():
    a = path(10, 20)
    c = np.r_[np.full(40, 100.0), np.linspace(100, 300, len(IDX) - 40)]
    members = mem("A") + [{"ticker": "C", "added_on": "2025-02-14", "removed_on": None}]
    with_c = run(frame(A=a, C=c), members)["level"]
    alone = run(frame(A=a), mem("A"))["level"]
    assert with_c.loc[:"2025-02-14"].equals(alone.loc[:"2025-02-14"])
    assert with_c.iloc[-1] != pytest.approx(alone.iloc[-1])


def test_removing_a_member_sells_it_at_that_close():
    a, b = path(10, 20), path(10, 40)
    members = [{"ticker": "A", "added_on": "2025-01-02"},
               {"ticker": "B", "added_on": "2025-01-02", "removed_on": "2025-03-03"}]
    r = run(frame(A=a, B=b), members, method="equal_hold")
    lv = r["level"]
    i = lv.index.get_loc(pd.Timestamp("2025-03-03"))
    # rode the 3 March move, then nothing of B after it
    assert r["weights"].iloc[i + 1]["B"] == 0
    after = lv.iloc[i:] / lv.iloc[i]
    assert np.allclose(after, a[i:] / a[i])
    ev = [e for e in r["events"] if e["kind"] == "remove"]
    assert ev == [{"date": "2025-03-03", "kind": "remove", "ticker": "B", "reason": "user"}]


def test_contribution_adds_up_to_the_move():
    rng = np.random.default_rng(5)
    cl = frame(**{t: 10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(IDX)))) for t in "ABCD"})
    members = mem("A", "B", "C") + [{"ticker": "D", "added_on": "2025-03-10"}]
    r = run(cl, members)
    lv = r["level"]
    for days in (1, 5, 21, 63, None):
        c = C.contribution(lv, r["dv"], days)
        start = 0 if days is None else len(lv) - 1 - days
        assert sum(c.values()) == pytest.approx((lv.iloc[-1] / lv.iloc[start] - 1) * 100, abs=1e-9)


# ── halts and delistings ─────────────────────────────────────────────────────
def test_a_halted_stock_is_carried_and_not_traded_at_a_rebalance():
    a = path(10, 20)
    b = pd.Series(np.full(len(IDX), 10.0), IDX)
    b.loc["2025-01-27":"2025-02-10"] = np.nan                  # halted across the Feb rebalance
    r = run(frame(A=a, B=b), mem("A", "B"))
    w = r["weights"]
    # B's units untouched on 3 Feb: its value share falls as A rises around it
    before, at = w.loc["2025-01-31"]["B"], w.loc["2025-02-03"]["B"]
    assert at == pytest.approx(before * r["level"].loc["2025-01-31"] / r["level"].loc["2025-02-03"])
    assert at != pytest.approx(0.5)
    # and is reset at the next rebalance once it trades
    assert w.loc["2025-03-03"]["B"] == pytest.approx(0.5)


def test_a_newcomer_that_is_halted_waits_until_it_trades():
    a = path(10, 20)
    b = pd.Series(np.full(len(IDX), 10.0), IDX)
    b.loc["2025-02-03":"2025-02-12"] = np.nan
    members = mem("A") + [{"ticker": "B", "added_on": "2025-02-05"}]
    r = run(frame(A=a, B=b), members)
    adds = [e for e in r["events"] if e["kind"] == "add"]
    assert adds[0]["date"] == "2025-02-13"


def test_a_delisted_member_leaves_at_its_last_price_and_the_level_is_continuous():
    a = path(10, 20)
    b = pd.Series(path(10, 15), IDX)
    b.loc["2025-04-01":] = np.nan
    members = [{"ticker": "A", "added_on": "2025-01-02"},
               {"ticker": "B", "added_on": "2025-01-02", "removed_on": "2025-03-31", "removal": "delisted"}]
    r = run(frame(A=a, B=b), members, method="equal_hold")
    lv = r["level"]
    assert lv.pct_change().abs().max() < 0.02                  # no gap anywhere
    assert r["weights"].loc["2025-04-01"]["B"] == 0
    assert r["events"][-1]["reason"] == "delisted"


def test_detect_exits_needs_the_gap_and_the_exchange_when_asked():
    a = pd.Series(path(10, 20), IDX)
    b = pd.Series(path(10, 15), IDX)
    b.iloc[-12:] = np.nan
    c = pd.Series(path(10, 15), IDX)
    c.iloc[-5:] = np.nan
    cl = frame(A=a, B=b, C=c)
    assert set(C.detect_exits(cl, gap=10)) == {"B"}
    assert C.detect_exits(cl, gap=10)["B"] == IDX[-13]
    assert C.detect_exits(cl, gap=10, confirmed=set()) == {}


def test_nothing_starts_before_the_first_member_trades():
    a = pd.Series(np.nan, IDX)
    a.loc["2025-02-03":] = 10.0
    r = run(frame(A=a), mem("A"))
    assert str(r["level"].index[0].date()) == "2025-02-03" and r["level"].iloc[0] == 1000


# ── reading it ───────────────────────────────────────────────────────────────
def test_next_rebalance_is_the_start_of_the_next_period():
    d = pd.Timestamp("2025-05-14")
    assert C.next_rebalance(d, "M") == "2025-06-01"
    assert C.next_rebalance(d, "Q") == "2025-07-01"
    assert C.next_rebalance(d, "H") == "2025-07-01"
    assert C.next_rebalance(d, "Y") == "2026-01-01"
    assert C.next_rebalance(pd.Timestamp("2025-12-03"), "M") == "2026-01-01"
    assert C.next_rebalance(d, "never") is None


def test_summary_marks_status_drift_and_the_live_boundary():
    a, b = path(10, 30), path(10, 10)
    cl = frame(A=a, B=b, Z=np.full(len(IDX), np.nan))
    members = [{"id": 1, "ticker": "A", "added_on": "2025-01-02"},
               {"id": 2, "ticker": "B", "added_on": "2025-01-02"},
               {"id": 3, "ticker": "Z", "added_on": "2025-01-02"}]
    r = run(cl, members, rebalance="Q")
    s = C.summarise(r, members=members, names={}, method="equal", rebalance="Q", cap=0.1,
                    bench=pd.Series(path(100, 110), IDX), live_from="2025-05-01", closes=cl, caps=None)
    by = {m["ticker"]: m for m in s["members"]}
    assert by["A"]["status"] == "active" and by["A"]["target"] == pytest.approx(50)
    assert by["A"]["weight"] > 50
    assert by["Z"]["status"] == "nodata"
    assert s["live_from"] == "2025-05-01"
    assert s["chart"]["bench"][0] == pytest.approx(1000)
    assert s["stats"]["ret"]["all"] == pytest.approx((s["chart"]["level"][-1] / 1000 - 1) * 100, abs=0.01)
