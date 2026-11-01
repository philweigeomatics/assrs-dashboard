"""
自建指数 — storage and the exit record, against an in-memory database.

The two things that would hurt quietly: one user reaching another's index,
and an acquired member vanishing from history once Yahoo forgets it.
"""

from __future__ import annotations

import os
import sys

from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api import index_api as A  # noqa: E402

CAL = pd.bdate_range("2025-01-02", "2025-06-30")


class FakeDB:
    def __init__(self):
        self.t: dict[str, list[dict]] = {"custom_index": [], "custom_index_member": []}
        self.seq = 0

    def _match(self, r, f):
        return all(str(r.get(k)) == str(v) for k, v in (f or {}).items())

    def read_table(self, name, filters=None, **_):
        if name not in self.t:
            raise RuntimeError(f'relation "public.{name}" does not exist')
        return pd.DataFrame([dict(r) for r in self.t[name] if self._match(r, filters)])

    def insert_records(self, name, recs, upsert=False):
        for r in recs:
            self.seq += 1
            self.t[name].append({"id": self.seq, **r})

    def update_records(self, name, vals, filters):
        for r in self.t[name]:
            if self._match(r, filters):
                r.update(vals)

    def delete_records(self, name, filters):
        self.t[name] = [r for r in self.t[name] if not self._match(r, filters)]


@pytest.fixture
def db(monkeypatch):
    d = FakeDB()
    monkeypatch.setattr(A, "_db", lambda: d)
    A._result_cache._d.clear()
    A._price_cache._d.clear()
    return d


def prices(gone_after: str | None = None, vanish: bool = False):
    a = pd.Series(np.linspace(10, 20, len(CAL)), CAL)
    b = pd.Series(np.linspace(30, 33, len(CAL)), CAL)
    if gone_after:
        b[b.index > gone_after] = np.nan
    closes = pd.DataFrame({"US:AAA": a} if vanish else {"US:AAA": a, "US:BBB": b})
    return closes, pd.Series(np.linspace(100, 110, len(CAL)), CAL)


def make(user=1, **kw):
    body = dict(name="光模块", market="US", method="equal", rebalance="M", cap_pct=None,
                base_date="2025-01-02",
                members=[{"ticker": "AAA", "name": "A Corp"}, {"ticker": "US:BBB", "name": "B Corp"}])
    body.update(kw)
    return A.create_index(user, **body)["id"]


def test_one_user_cannot_reach_anothers_index(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", lambda t, s: prices())
    iid = make(user=1)
    with pytest.raises(LookupError):
        A.compute(2, iid)
    with pytest.raises(LookupError):
        A.delete_index(2, iid)
    with pytest.raises(LookupError):
        A.add_member(2, iid, ticker="CCC", name=None, added_on=None)
    assert A.list_indexes(2)["indexes"] == []
    assert len(A.list_indexes(1)["indexes"]) == 1


def test_validation(db):
    with pytest.raises(ValueError):
        make(members=[{"ticker": "AAA"}])                       # one name is not an index
    with pytest.raises(ValueError):
        make(market="CN", members=[{"ticker": "AAPL"}, {"ticker": "600519"}])
    with pytest.raises(ValueError):
        make(method="cap_capped", cap_pct=80)
    with pytest.raises(ValueError):
        make(base_date="2099-01-01")
    make()
    with pytest.raises(ValueError):
        make()                                                  # same name twice


def test_tickers_are_stored_in_the_apps_own_form(db):
    iid = make(market="CN", members=[{"ticker": "600519.SH"}, {"ticker": "000858"}])
    assert sorted(m["ticker"] for m in A._members(iid)) == ["000858", "600519"]
    iid = make(name="x")
    assert sorted(m["ticker"] for m in A._members(iid)) == ["US:AAA", "US:BBB"]


def test_an_exit_is_written_down_once_and_outlives_yahoo(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", lambda t, s: prices(gone_after="2025-05-30"))
    iid = make(method="equal_hold")
    first = A.compute(1, iid)
    row = next(m for m in A._members(iid) if m["ticker"] == "US:BBB")
    assert row["removal"] == "delisted" and row["removed_on"] == "2025-05-30"
    assert row["exit_price"] == pytest.approx(prices(gone_after="2025-05-30")[0]["US:BBB"].dropna().iloc[-1])
    assert next(m for m in first["members"] if m["ticker"] == "US:BBB")["status"] == "delisted"

    # Weeks later Yahoo has nothing for BBB at all — the history must not move.
    monkeypatch.setattr(A, "_us_closes", lambda t, s: prices(vanish=True))
    A._result_cache._d.clear()
    A._price_cache._d.clear()
    later = A.compute(1, iid)
    assert later["chart"]["level"] == first["chart"]["level"]


def test_the_rules_are_fixed_once_created(db):
    """Only the name and the comparison can change; the index itself cannot be redefined."""
    import inspect
    iid = make()
    before = dict(A._own(1, iid))
    A.update_index(1, iid, name="改名", bench="SMH")
    after = A._own(1, iid)
    assert after["name"] == "改名" and after["bench"] == "SMH"
    for k in ("method", "rebalance", "cap_pct", "base_date", "rules_updated_at"):
        assert str(after.get(k)) == str(before.get(k))
    assert set(inspect.signature(A.update_index).parameters) == {"user_id", "index_id", "name", "bench"}
    assert not hasattr(A, "update_rules")


def test_removal_keeps_history_and_purge_does_not(db):
    iid = make()
    mid = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:BBB")
    with pytest.raises(ValueError):
        A.remove_member(1, iid, mid, mode="end", on="2025-01-02")
    A.remove_member(1, iid, mid, mode="end", on="2025-03-03")
    assert next(m for m in A._members(iid) if int(m["id"]) == mid)["removed_on"] == "2025-03-03"
    A.remove_member(1, iid, mid, mode="purge", on=None)
    assert all(int(m["id"]) != mid for m in A._members(iid))


def test_a_missing_table_names_the_migration(monkeypatch):
    class Empty(FakeDB):
        def __init__(self):
            super().__init__()
            self.t = {}
    monkeypatch.setattr(A, "_db", lambda: Empty())
    with pytest.raises(A.MissingTables, match="20261031_custom_index.sql"):
        A.list_indexes(1)


# ── the dashboard and the daily job ──────────────────────────────────────────
LONG = pd.bdate_range("2024-01-02", "2025-06-30")


def long_prices(calls, gone=None):
    def fetch(tickers, start):
        calls.append(tuple(tickers))
        data = {t: pd.Series(np.linspace(10 + i, 20 + 3 * i, len(LONG)), LONG) for i, t in enumerate(sorted(tickers))}
        if gone and gone in data:
            data[gone][data[gone].index > "2025-05-30"] = np.nan
        return pd.DataFrame(data), pd.Series(np.linspace(100, 120, len(LONG)), LONG)
    return fetch


@pytest.fixture
def fresh(db):
    A._union.clear()
    yield db
    A._union.clear()


def test_the_board_downloads_once_for_all_indexes(fresh, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", long_prices(calls))
    make(name="一", base_date="2024-01-02")
    make(name="二", base_date="2024-03-01", members=[{"ticker": "CCC"}, {"ticker": "AAA"}])
    out = A.board_rows(1, "US")
    assert len(calls) == 1 and set(calls[0]) == {"US:AAA", "US:BBB", "US:CCC"}
    assert sorted(r["name"] for r in out["rows"]) == ["一", "二"]
    assert all(r["group"] == "我的指数" and r["index_id"] for r in out["rows"])
    assert A.board_rows(1, "CN") == {"rows": [], "young": [], "failed": []}


def test_a_young_buy_and_hold_index_is_listed_not_given_readings(fresh, monkeypatch):
    """Buy-and-hold started earlier is a different portfolio, so it is not extended."""
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    make(name="新", base_date="2025-03-03", method="equal_hold")
    out = A.board_rows(1, "US")
    assert out["rows"] == [] and out["young"][0]["name"] == "新" and out["young"][0]["hold"] is True


def walk_prices(calls):
    def fetch(tickers, start):
        calls.append(str(start))
        rng = np.random.default_rng(11)
        data = {t: pd.Series(20 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(LONG)))), LONG)
                for t in sorted(tickers)}
        return pd.DataFrame(data), pd.Series(np.linspace(100, 120, len(LONG)), LONG)
    return fetch


def test_a_recent_rebalancing_index_is_read_over_the_longer_past(fresh, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", walk_prices(calls))
    iid = make(name="新", base_date="2025-03-03", rebalance="Q")
    out = A.board_rows(1, "US")
    assert out["young"] == [] and len(out["rows"]) == 1
    row = out["rows"][0]
    assert row["extended_from"] == "2024-01-02"
    real = A._level(A.compute(1, iid))
    assert row["price"] == pytest.approx(real.iloc[-1], abs=0.01)
    assert len(calls) == 1                                      # still one download


def test_extension_matches_the_real_index_after_a_rebalance(fresh, monkeypatch):
    """The whole case for extending: identical daily moves once the weights have been reset."""
    monkeypatch.setattr(A, "_us_closes", walk_prices([]))
    iid = make(name="新", base_date="2025-03-03", rebalance="Q")
    real = A._level(A.compute(1, iid))
    ext, since = A.extended_level(1, iid)
    assert since == "2024-01-02" and ext.index[0] < real.index[0]
    after = real.index[real.index >= "2025-04-02"]               # first quarterly reset is 1 April
    # the real level arrives rounded to 0.01 points; nothing else may differ
    assert np.allclose(ext.loc[after].pct_change().dropna(), real.loc[after].pct_change().dropna(), atol=2e-5)
    before = real.index[(real.index > "2025-03-03") & (real.index < "2025-04-01")]
    assert not np.allclose(ext.loc[before].pct_change().dropna(), real.loc[before].pct_change().dropna(), atol=2e-5)
    assert ext.iloc[-1] == pytest.approx(real.iloc[-1])


def test_an_index_already_starting_long_ago_is_not_extended(fresh, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", walk_prices([]))
    monkeypatch.setattr(A, "_extend_from", lambda m: date(2024, 1, 2))
    iid = make(name="老", base_date="2024-01-02")
    lv, since = A.extended_level(1, iid)
    assert since is None and lv.equals(A._level(A.compute(1, iid)))


def test_the_daily_job_computes_everyone_and_reports_new_exits(fresh, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", long_prices(calls, gone="US:BBB"))
    make(user=1, name="甲", base_date="2024-01-02")
    make(user=2, name="乙", base_date="2024-01-02", members=[{"ticker": "AAA"}, {"ticker": "CCC"}])
    report = A.refresh_all()
    assert len(calls) == 1
    by = {r["name"]: r for r in report}
    assert by["甲"]["ok"] and by["甲"]["new_exits"] == ["US:BBB"]
    assert by["乙"]["ok"] and by["乙"]["new_exits"] == []
    # the second run finds nothing new
    A._result_cache._d.clear()
    assert {r["name"]: r["new_exits"] for r in A.refresh_all()} == {"甲": [], "乙": []}


def test_stretch_detail_is_the_boards_view_of_an_index(fresh, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    iid = make(base_date="2024-01-02")
    d = A.stretch_detail(1, iid)
    assert d["name"] == "光模块" and len(d["dates"]) == len(d["close"]) == len(d["stretch"])
    with pytest.raises(LookupError):
        A.stretch_detail(2, iid)


def test_a_later_addition_keeps_its_date_in_the_longer_past(fresh, monkeypatch):
    """Only the founders move back. A member added in May is not in the April index, extended or not."""
    monkeypatch.setattr(A, "_us_closes", walk_prices([]))
    iid = make(name="新", base_date="2025-03-03", rebalance="Q")
    A.add_member(1, iid, ticker="CCC", name="C", added_on="2025-05-01")
    real = A._level(A.compute(1, iid))
    ext, _ = A.extended_level(1, iid)
    april = real.index[(real.index >= "2025-04-02") & (real.index <= "2025-05-01")]
    assert np.allclose(ext.loc[april].pct_change().dropna(), real.loc[april].pct_change().dropna(), atol=2e-5)


# ── the comparison line ──────────────────────────────────────────────────────
def test_an_index_can_be_compared_with_an_industry_etf(fresh, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", walk_prices([]))
    smh = pd.Series(np.linspace(200, 260, len(LONG)), LONG)
    monkeypatch.setattr(A, "BENCH_SOURCE", lambda market, code: smh if (market, code) == ("US", "SMH") else None)
    iid = make(name="对比", base_date="2024-06-03", bench="SMH")
    out = A.compute(1, iid)
    assert out["bench_label"] == "SMH 半导体" and out["bench_note"] is None
    start = smh.loc["2024-06-03"]
    assert out["bench_ret"]["all"] == pytest.approx((smh.iloc[-1] / start - 1) * 100, abs=0.01)
    assert A.list_indexes(1)["indexes"][0]["bench_label"] == "SMH 半导体"
    row = A.board_rows(1, "US")["rows"][0]
    assert row["rs_vs"] == "SMH"


def test_an_unreadable_comparison_falls_back_to_the_market_and_says_so(fresh, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", walk_prices([]))
    monkeypatch.setattr(A, "BENCH_SOURCE", None)
    iid = make(name="对比", base_date="2024-06-03", bench="SMH")
    out = A.compute(1, iid)
    assert out["bench_label"] == "SPY（含分红）" and "SMH" in out["bench_note"]


def test_the_comparison_must_come_from_the_boards_list(db):
    with pytest.raises(ValueError):
        make(bench="NVDA")
    with pytest.raises(ValueError):
        make(market="CN", members=[{"ticker": "600519"}, {"ticker": "000858"}], bench="SMH")
    make(name="a", market="CN", members=[{"ticker": "600519"}, {"ticker": "000858"}], bench="931160.CSI")


def test_changing_the_comparison_does_not_move_the_live_boundary(db):
    iid = make(base_date="2025-01-02")
    before = A._own(1, iid)["rules_updated_at"]
    A.update_index(1, iid, name="光模块", bench="AIQ")
    assert A._own(1, iid)["rules_updated_at"] == before and A._own(1, iid)["bench"] == "AIQ"
    with pytest.raises(ValueError):
        A.update_index(1, iid, name="光模块", bench="NVDA")


def test_an_index_without_a_comparison_saves_without_the_column(db):
    """A database migrated before `bench` existed must still accept a plain index."""
    make()
    assert "bench" not in db.t["custom_index"][0]


def test_prices_are_fetched_from_before_the_start_for_volatility(fresh, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", walk_prices(calls))
    iid = make(base_date="2025-01-02", method="inv_vol", rebalance="Q")
    out = A.compute(1, iid)
    assert calls == [str(date(2025, 1, 2) - A.WARMUP)]
    assert out["start"] == "2025-01-02" and out["chart"]["level"][0] == 1000


def test_a_mixed_list_is_valid_json(db):
    """One capped index beside an uncapped one made cap_pct a float column — NaN broke the list."""
    import json
    make(name="a")
    make(name="b", method="cap_capped", cap_pct=15)
    out = A.list_indexes(1)
    json.dumps(out, allow_nan=False)
    assert {r["name"]: r["cap_pct"] for r in out["indexes"]} == {"a": None, "b": 15.0}


# ── copying an index ─────────────────────────────────────────────────────────
def test_a_copy_has_the_same_members_over_the_same_dates_under_another_rule(fresh, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([], gone="US:BBB"))
    src = make(name="源", base_date="2024-01-02", members=[{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}])
    A.add_member(1, src, ticker="DDD", name="D", added_on="2024-06-03")
    ccc = next(int(m["id"]) for m in A._members(src) if m["ticker"] == "US:CCC")
    A.remove_member(1, src, ccc, mode="end", on="2025-02-03")
    A.compute(1, src)                                         # records BBB's exit
    out = A.copy_index(1, src, name="副本", method="inv_vol", rebalance="Q", cap_pct=None, bench="SMH")
    assert out["members"] == 4
    strip = lambda ms: sorted((m["ticker"], m["added_on"], m.get("removed_on"), m.get("removal"), m.get("exit_price"),
                               bool(m.get("frozen_closes"))) for m in ms)
    assert strip(A._members(out["id"])) == strip(A._members(src))
    cp, orig = A._own(1, out["id"]), A._own(1, src)
    assert cp["method"] == "inv_vol" and cp["rebalance"] == "Q" and cp["bench"] == "SMH"
    assert cp["base_date"] == orig["base_date"] and cp["market"] == orig["market"]
    assert cp["rules_updated_at"] > orig["rules_updated_at"]        # the copy's rule is hindsight until now
    a, b = A.compute(1, src), A.compute(1, out["id"])
    assert a["start"] == b["start"] and a["chart"]["level"] != b["chart"]["level"]
    kinds = lambda d: [(e["date"], e["kind"], e.get("ticker")) for e in d["events"] if e["kind"] in ("add", "remove")]
    assert kinds(a) == kinds(b)                                 # same membership events, different weighting


def test_a_copy_is_owner_only_and_named_uniquely(db):
    src = make()
    with pytest.raises(LookupError):
        A.copy_index(2, src, name="偷", method="equal", rebalance="M", cap_pct=None)
    with pytest.raises(ValueError):
        A.copy_index(1, src, name="光模块", method="equal", rebalance="M", cap_pct=None)
    with pytest.raises(ValueError):
        A.copy_index(1, src, name="自定", method="custom", rebalance="M", cap_pct=None)


# ── nothing watched is rewritten ─────────────────────────────────────────────
def aged(db, iid):
    """The index was made long ago: its first day, when the backfill can still be edited, is over."""
    for r in db.t["custom_index"]:
        if r["id"] == iid:
            r["created_at"] = "2025-01-02T10:00:00"


def test_on_the_first_day_members_can_be_back_dated(db):
    iid = make(base_date="2025-01-02")
    assert A.list_indexes(1)["indexes"][0]["settling"] is True
    A.add_member(1, iid, ticker="CCC", name="C", added_on="2025-03-03")
    bbb = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:BBB")
    A.remove_member(1, iid, bbb, mode="end", on="2025-04-01")
    aaa = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:AAA")
    A.remove_member(1, iid, aaa, mode="purge", on=None)
    assert sorted(m["ticker"] for m in A._members(iid)) == ["US:BBB", "US:CCC"]


def test_after_the_first_day_members_change_from_today_only(db, monkeypatch):
    monkeypatch.setattr(A, "_today", lambda market: date(2026, 10, 30))
    iid = make(base_date="2025-01-02")
    aged(db, iid)
    assert A.list_indexes(1)["indexes"][0]["settling"] is False
    with pytest.raises(ValueError, match="从今天收盘生效"):
        A.add_member(1, iid, ticker="CCC", name="C", added_on="2026-10-29")
    assert A.add_member(1, iid, ticker="CCC", name="C", added_on=None)["added_on"] == "2026-10-30"
    bbb = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:BBB")
    with pytest.raises(ValueError, match="从今天收盘生效"):
        A.remove_member(1, iid, bbb, mode="end", on="2026-10-01")
    with pytest.raises(ValueError, match="移出"):
        A.remove_member(1, iid, bbb, mode="purge", on=None)      # it has been in the index for a year
    A.remove_member(1, iid, bbb, mode="end", on=None)
    assert next(m for m in A._members(iid) if m["ticker"] == "US:BBB")["removed_on"] == "2026-10-30"
    ccc = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:CCC")
    A.remove_member(1, iid, ccc, mode="purge", on=None)      # added today: it has not moved the index yet
    assert "US:CCC" not in {m["ticker"] for m in A._members(iid)}


def test_the_first_day_is_the_markets_own_day(db, monkeypatch):
    """17:00 UTC on 31 Oct is still the 31st in New York but already the 1st in Shanghai."""
    idx = {"market": "CN", "created_at": "2026-10-31T17:00:00"}
    monkeypatch.setattr(A, "_today", lambda market: date(2026, 11, 1) if market == "CN" else date(2026, 10, 31))
    assert A._settling(idx) and A._settling({**idx, "market": "US"})
    monkeypatch.setattr(A, "_today", lambda market: date(2026, 10, 31))
    assert not A._settling(idx)
    assert not A._settling({"market": "US", "created_at": None})


def test_a_detail_says_whether_the_first_day_is_still_on(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", lambda t, s: prices())
    iid = make()
    assert A.compute(1, iid)["index"]["settling"] is True
    aged(db, iid)
    out = A.compute(1, iid)                                   # cached ledger, fresh flag
    assert out["index"]["settling"] is False and out["today"] == str(A._today("US"))
