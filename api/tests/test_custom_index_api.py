"""
自建指数 — storage and the exit record, against an in-memory database.

The two things that would hurt quietly: one user reaching another's index,
and an acquired member vanishing from history once Yahoo forgets it.
"""

from __future__ import annotations

import os
import sys

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api import index_api as A  # noqa: E402

CAL = pd.bdate_range("2025-01-02", "2025-06-30")


class FakeDB:
    #: Upserts replace the row with the same key, as the real primary keys do.
    KEYS = {"custom_index_day": ("index_id", "d")}

    def __init__(self):
        self.t: dict[str, list[dict]] = {"custom_index": [], "custom_index_member": [], "custom_index_day": []}
        self.seq = 0

    def _match(self, r, f):
        return all(str(r.get(k)) == str(v) for k, v in (f or {}).items())

    def read_table(self, name, filters=None, columns="*", order_by=None, limit=None):
        if name not in self.t:
            raise RuntimeError(f'relation "public.{name}" does not exist')
        rows = [dict(r) for r in self.t[name] if self._match(r, filters)]
        for o in reversed([o.strip() for o in (order_by or "").split(",") if o.strip()]):
            rows.sort(key=lambda r: str(r.get(o.lstrip("-"))), reverse=o.startswith("-"))
        return pd.DataFrame(rows[:limit] if limit else rows)

    def read_all(self, name, filters=None, columns="*", order_by=None, in_=None, page=1000):
        df = self.read_table(name, filters, columns, order_by)
        if in_ and not df.empty:
            df = df[df[in_[0]].astype(str).isin([str(v) for v in in_[1]])]
        return df

    def insert_records(self, name, recs, upsert=False):
        if name not in self.t:
            raise RuntimeError(f'relation "public.{name}" does not exist')
        key = self.KEYS.get(name)
        for r in recs:
            r = {k: (dict(v) if isinstance(v, dict) else v) for k, v in r.items()}
            hit = next((x for x in self.t[name] if all(str(x.get(k)) == str(r.get(k)) for k in key)), None)                 if upsert and key else None
            if hit is not None:
                hit.update(r)
                continue
            self.seq += 1
            self.t[name].append({"id": self.seq, **r})

    def update_records(self, name, vals, filters):
        for r in self.t[name]:
            if self._match(r, filters):
                r.update(vals)

    def delete_records(self, name, filters):
        self.t[name] = [r for r in self.t[name] if not self._match(r, filters)]

    def days(self, iid):
        return sorted((r for r in self.t["custom_index_day"] if r["index_id"] == iid), key=lambda r: r["d"])


@pytest.fixture
def db(monkeypatch):
    d = FakeDB()
    monkeypatch.setattr(A, "_db", lambda: d)
    # api.main, once any test imports it, points this at the live dashboard caches.
    monkeypatch.setattr(A, "BENCH_SOURCE", None)
    A._series_cache._d.clear()
    A._price_cache._d.clear()
    A._catch_up_tried.clear()
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
    with pytest.raises(A.MissingTables, match="20261101_custom_index_day.sql"):
        A.list_indexes(1)


# ── the dashboard and the daily job ──────────────────────────────────────────
LONG = pd.bdate_range("2024-01-02", "2025-06-30")


def long_prices(calls, gone=None, end=None, revise=None):
    """Prices up to `end` (a later download sees more days); `revise` rescales every price before a date."""
    def fetch(tickers, start):
        calls.append(tuple(tickers))
        cal = LONG[LONG <= end] if end else LONG
        data = {t: pd.Series(np.linspace(10 + i, 20 + 3 * i, len(LONG)), LONG).reindex(cal)
                for i, t in enumerate(sorted(tickers))}
        for t in data:                                         # some wiggle, so weights matter
            data[t] = data[t] * (1 + 0.03 * np.sin(np.arange(len(cal)) / 5 + len(t)))
        if gone and gone in data:
            data[gone][data[gone].index > "2025-05-30"] = np.nan
        if revise:
            for t in data:
                data[t][data[t].index < revise] *= 0.9
        return pd.DataFrame(data), pd.Series(np.linspace(100, 120, len(LONG)), LONG).reindex(cal)
    return fetch


def walk_prices(calls):
    def fetch(tickers, start):
        calls.append(str(start))
        rng = np.random.default_rng(11)
        data = {t: pd.Series(20 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(LONG)))), LONG)
                for t in sorted(tickers)}
        return pd.DataFrame(data), pd.Series(np.linspace(100, 120, len(LONG)), LONG)
    return fetch


def fail(*_):
    raise AssertionError("downloaded a price")


def test_an_index_is_computed_once_then_read_from_its_record(db, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", long_prices(calls))
    iid = make(base_date="2024-03-01", rebalance="Q")
    first = A.compute(1, iid)
    stored = db.days(iid)
    assert len(calls) == 1 and stored[0]["d"] == "2024-03-01" and stored[-1]["d"] == "2025-06-30"
    assert first["chart"]["level"][-1] == pytest.approx(stored[-1]["close"], abs=0.01)
    # Every page after that reads the record and downloads nothing.
    monkeypatch.setattr(A, "_us_closes", fail)
    A._price_cache._d.clear()
    assert A.compute(1, iid)["chart"] == first["chart"]
    assert A.board_rows(1, "US")["rows"][0]["index_id"] == iid
    assert len(A.stretch_detail(1, iid)["dates"]) > 0


def test_the_board_reads_stored_records_for_any_number_of_indexes(db, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", long_prices(calls))
    for k in range(5):
        make(name=f"指数{k}", base_date="2024-01-02", members=[{"ticker": "AAA"}, {"ticker": f"C{k}"}])
    A.board_rows(1, "US")
    assert len(calls) == 5                                      # each built once, the first time it is seen
    monkeypatch.setattr(A, "_us_closes", fail)
    A._price_cache._d.clear()
    A._series_cache._d.clear()
    out = A.board_rows(1, "US")
    assert len(out["rows"]) == 5 and out["failed"] == []
    assert all(r["group"] == "我的指数" for r in out["rows"])
    assert A.board_rows(1, "CN") == {"rows": [], "young": [], "failed": []}


def test_a_young_buy_and_hold_index_is_listed_not_given_readings(db, monkeypatch):
    """Buy-and-hold started earlier is a different portfolio, so it has no backtest."""
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    make(name="新", base_date="2025-03-03", method="equal_hold")
    out = A.board_rows(1, "US")
    assert out["rows"] == [] and out["young"][0]["name"] == "新" and out["young"][0]["hold"] is True


def test_the_job_carries_an_index_forward_and_never_touches_a_stored_day(db, monkeypatch):
    calls, starts = [], []
    monkeypatch.setattr(A, "_us_closes", long_prices(calls, end="2025-04-15"))
    iid = make(base_date="2024-03-01", method="inv_vol", rebalance="M")
    A.compute(1, iid)
    before = {r["d"]: (r["close"], r["state"]) for r in db.days(iid)}
    assert max(before) == "2025-04-15"
    # Next download: more days, and every older price revised (a dividend, a correction).
    A._price_cache._d.clear()
    later = long_prices(calls, end="2025-06-30", revise="2025-04-01")
    monkeypatch.setattr(A, "_us_closes", lambda t, s: (starts.append(s), later(t, s))[1])
    report = A.refresh_all()
    assert report[0]["ok"] and report[0]["added"] == len(LONG[(LONG > "2025-04-15")])
    after = {r["d"]: (r["close"], r["state"]) for r in db.days(iid)}
    assert all(after[d] == v for d, v in before.items())         # not one stored day moved
    # One download, reaching back only far enough for three months of returns.
    assert len(calls) == 2 and starts == [date(2025, 4, 15) - A.WARMUP]
    assert A.refresh_all()[0]["added"] == 0                      # nothing new, nothing written


def test_carrying_forward_is_the_same_index_as_computing_it_whole(db, monkeypatch):
    """Stored up to April and continued, or computed to June in one go: the same numbers."""
    monkeypatch.setattr(A, "_us_closes", long_prices([], end="2025-04-15"))
    members = [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}]
    iid = make(name="分段", base_date="2024-03-01", method="inv_vol", rebalance="M", members=members)
    A.compute(1, iid)
    A._price_cache._d.clear()
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    A.refresh_all()
    whole = make(name="一次", base_date="2024-03-01", method="inv_vol", rebalance="M", members=members)
    A.compute(1, whole)
    a = [r["close"] for r in db.days(iid)]
    b = [r["close"] for r in db.days(whole)]
    assert len(a) == len(b) and np.allclose(a, b, rtol=1e-6)


def test_a_session_is_not_stored_before_it_has_settled(db, monkeypatch):
    import pytz
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    monkeypatch.setattr(A, "_now", lambda m: pytz.timezone("America/New_York").localize(datetime(2025, 6, 30, 14, 0)))
    iid = make(base_date="2024-03-01")
    A.compute(1, iid)
    assert db.days(iid)[-1]["d"] == "2025-06-27"                 # the 30th is still trading
    monkeypatch.setattr(A, "_now", lambda m: pytz.timezone("America/New_York").localize(datetime(2025, 6, 30, 17, 30)))
    A._price_cache._d.clear()
    A.refresh_all()
    assert db.days(iid)[-1]["d"] == "2025-06-30"


def test_a_holding_missing_from_the_download_stops_the_day_rather_than_freezing_it(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([], end="2025-04-15"))
    iid = make(base_date="2024-03-01")
    A.compute(1, iid)
    n = len(db.days(iid))
    A._price_cache._d.clear()
    full = long_prices([])
    monkeypatch.setattr(A, "_us_closes", lambda t, s: (full(t, s)[0].drop(columns=["US:BBB"]), full(t, s)[1]))
    report = A.refresh_all()
    assert not report[0]["ok"] and "US:BBB" in report[0]["error"]
    assert len(db.days(iid)) == n


def test_the_job_writes_an_exit_down_and_reports_it(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([], end="2025-05-15"))
    make(user=1, name="甲", base_date="2024-01-02")
    make(user=2, name="乙", base_date="2024-01-02", members=[{"ticker": "AAA"}, {"ticker": "CCC"}])
    assert all(r.get("rebuilt") for r in A.refresh_all())
    A._price_cache._d.clear()
    monkeypatch.setattr(A, "_us_closes", long_prices([], gone="US:BBB"))
    by = {r["name"]: r for r in A.refresh_all()}
    assert by["甲"]["ok"] and by["甲"]["new_exits"] == ["US:BBB"]
    assert by["乙"]["ok"] and by["乙"]["new_exits"] == []
    iid = next(int(r["id"]) for r in db.t["custom_index"] if r["name"] == "甲")
    held = db.days(iid)[-1]["state"]["v"]
    assert "US:BBB" not in held and set(held) == {"US:AAA"}


def test_a_read_after_the_close_stores_the_new_session_without_waiting_for_the_job(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([], end="2025-06-27"))
    iid = make(base_date="2024-03-01")
    A.compute(1, iid)
    assert db.days(iid)[-1]["d"] == "2025-06-27"
    A._price_cache._d.clear()
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    spy = pd.Series(np.linspace(400, 500, len(LONG)), LONG)
    monkeypatch.setattr(A, "BENCH_SOURCE", lambda market, code: spy if code == "SPY" else None)
    out = A.compute(1, iid)
    assert out["as_of"] == "2025-06-30" and db.days(iid)[-1]["d"] == "2025-06-30"


def test_on_the_first_day_an_edit_recomputes_the_record(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    iid = make(base_date="2024-03-01")
    A.compute(1, iid)
    A.add_member(1, iid, ticker="CCC", name="C", added_on="2024-06-03")
    assert db.days(iid) == []                                   # dropped, to be computed again
    out = A.compute(1, iid)
    assert any(e["kind"] == "add" and e["ticker"] == "US:CCC" and e["date"] == "2024-06-03" for e in out["events"])


def test_after_the_first_day_a_change_lands_on_the_next_unstored_close(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    iid = make(base_date="2024-03-01")
    A.compute(1, iid)
    aged(db, iid)
    monkeypatch.setattr(A, "_today", lambda m: date(2025, 6, 30))   # today's close already stored
    assert A.add_member(1, iid, ticker="CCC", name="C", added_on=None)["added_on"] == "2025-07-01"
    ccc = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:CCC")
    A.remove_member(1, iid, ccc, mode="purge", on=None)          # never in a stored day: may go
    bbb = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:BBB")
    A.remove_member(1, iid, bbb, mode="end", on=None)
    assert next(m for m in A._members(iid) if m["ticker"] == "US:BBB")["removed_on"] == "2025-07-01"
    assert db.days(iid)[-1]["d"] == "2025-06-30"                # nothing stored was dropped


def test_open_and_turnover_are_stored_and_split_the_move(db, monkeypatch):
    def fetch(tickers, start):
        closes, bench = long_prices([])(tickers, start)
        opens = closes * 1.01                                   # every stock opens 1% above its close
        return {"close": closes, "open": opens, "amount": closes * 0 + 1e6, "bench": bench}
    monkeypatch.setattr(A, "_us_closes", fetch)
    iid = make(base_date="2024-03-01")
    out = A.compute(1, iid)
    d = db.days(iid)
    assert d[0]["open"] is None and d[1]["open"] is not None
    assert d[-1]["amount"] == pytest.approx(2e6)
    sp = out["stats"]["split"]["1d"]
    day = (d[-1]["close"] / d[-2]["close"] - 1) * 100
    assert (1 + sp["gap"] / 100) * (1 + sp["session"] / 100) == pytest.approx(1 + day / 100, abs=1e-3)
    assert out["stats"]["amount"] == pytest.approx(2e6) and out["stats"]["amount_x20"] == pytest.approx(1.0)


def test_a_recent_rebalancing_index_is_read_over_its_backtest(db, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", walk_prices(calls))
    iid = make(name="新", base_date="2025-03-03", rebalance="Q")
    out = A.board_rows(1, "US")
    assert out["young"] == [] and len(out["rows"]) == 1
    row = out["rows"][0]
    assert row["extended_from"] == "2024-01-02"
    assert row["price"] == pytest.approx(db.days(iid)[-1]["close"], abs=0.01)
    assert len(calls) == 1                                      # the index and its backtest from one download


def test_the_backtest_is_the_same_rules_run_earlier_and_joined_on_the_start(db, monkeypatch):
    """Before the start: the founders under the same rules from earlier. From the start: the index itself."""
    import custom_index as C
    monkeypatch.setattr(A, "_us_closes", walk_prices([]))
    iid = make(name="新", base_date="2025-03-03", rebalance="Q")
    A.compute(1, iid)
    idx = A._own(1, iid)
    lv, since = A._series(idx, A._days(iid))
    closes, _ = walk_prices([])(["US:AAA", "US:BBB"], None)
    early = [{"ticker": t, "added_on": "2024-01-02", "removed_on": None} for t in ("US:AAA", "US:BBB")]
    replay = C.build(closes, early, method="equal", rebalance="Q", base_date="2024-01-02")["level"]
    pre = lv.index[lv.index <= "2025-03-03"]
    assert np.allclose(lv.loc[pre].pct_change().dropna(), replay.loc[pre].pct_change().dropna(), atol=1e-5)
    own = pd.Series([r["close"] for r in A._days(iid)], pd.to_datetime([r["d"] for r in A._days(iid)]))
    assert np.allclose(lv.loc[own.index].to_numpy(), own.to_numpy(dtype=float))


def test_an_index_already_starting_long_ago_has_no_backtest(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", walk_prices([]))
    monkeypatch.setattr(A, "_extend_from", lambda m: date(2024, 1, 2))
    iid = make(name="老", base_date="2024-01-02")
    A.compute(1, iid)
    lv, since = A._series(A._own(1, iid), A._days(iid))
    assert since is None and len(lv) == len(A._days(iid))


def test_stretch_detail_is_the_boards_view_of_an_index(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    iid = make(base_date="2024-01-02")
    d = A.stretch_detail(1, iid)
    assert d["name"] == "光模块" and len(d["dates"]) == len(d["close"]) == len(d["stretch"])
    with pytest.raises(LookupError):
        A.stretch_detail(2, iid)


def test_deleting_an_index_deletes_its_record(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([]))
    iid = make()
    A.compute(1, iid)
    assert db.days(iid)
    A.delete_index(1, iid)
    assert db.days(iid) == [] and A._members(iid) == []


# ── the comparison line ──────────────────────────────────────────────────────
def test_an_index_can_be_compared_with_an_industry_etf(db, monkeypatch):
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


def test_an_unreadable_comparison_falls_back_to_the_market_and_says_so(db, monkeypatch):
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


def test_prices_are_fetched_from_before_the_start_for_volatility(db, monkeypatch):
    calls = []
    monkeypatch.setattr(A, "_us_closes", walk_prices(calls))
    iid = make(base_date="2025-01-02", method="inv_vol", rebalance="Q")
    out = A.compute(1, iid)
    assert calls == [str(A._extend_from("US") - A.WARMUP)]      # its backtest's start, less the warm-up
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
def test_a_copy_has_the_same_members_over_the_same_dates_under_another_rule(db, monkeypatch):
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
    with pytest.raises(ValueError, match="从下一个收盘生效"):
        A.add_member(1, iid, ticker="CCC", name="C", added_on="2026-10-29")
    assert A.add_member(1, iid, ticker="CCC", name="C", added_on=None)["added_on"] == "2026-10-30"
    bbb = next(int(m["id"]) for m in A._members(iid) if m["ticker"] == "US:BBB")
    with pytest.raises(ValueError, match="从下一个收盘生效"):
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


# ── an adjustment that has not reached the data yet is not stored ────────────
def test_an_a_share_move_beyond_the_limit_is_held_back_until_the_factor_arrives(db, monkeypatch):
    """A 10送10 without its factor reads as -50% — impossible under a ±20% limit, so the day waits."""
    cal = LONG[LONG <= "2025-04-15"]
    def fetch(end, broken):
        def f(codes, start):
            c = LONG[LONG <= end]
            a = pd.Series(np.linspace(10, 12, len(LONG)), LONG).reindex(c)
            b = pd.Series(np.linspace(30, 33, len(LONG)), LONG).reindex(c)
            if broken:
                b.loc[b.index > "2025-04-15"] /= 2                     # the raw halving, unadjusted
            return pd.DataFrame({"600519": a, "000858": b}), pd.Series(np.linspace(3000, 3300, len(LONG)), LONG).reindex(c)
        return f
    monkeypatch.setattr(A, "_cn_closes", fetch("2025-04-15", False))
    monkeypatch.setattr(A, "_cn_names", lambda: {}, raising=False)
    import api.portfolio_api as P
    monkeypatch.setattr(P, "_cn_names", lambda: {})
    iid = make(market="CN", base_date="2024-03-01", members=[{"ticker": "600519"}, {"ticker": "000858"}])
    A.compute(1, iid)
    assert db.days(iid)[-1]["d"] == str(cal[-1].date())
    A._price_cache._d.clear()
    monkeypatch.setattr(A, "_cn_closes", fetch("2025-04-22", True))
    r = A.refresh_all()[0]
    assert not r["ok"] and "涨跌停" in r["error"] and db.days(iid)[-1]["d"] == "2025-04-15"
    A._price_cache._d.clear()
    monkeypatch.setattr(A, "_cn_closes", fetch("2025-04-22", False))  # the factor has arrived
    assert A.refresh_all()[0]["ok"] and db.days(iid)[-1]["d"] == "2025-04-22"


def test_a_us_split_yahoo_has_not_applied_is_held_back_but_a_real_crash_is_stored(db, monkeypatch):
    def fetch(end, crash):
        def f(tickers, start):
            closes, bench = long_prices([], end=end)(tickers, start)
            if crash:
                closes.loc[closes.index > "2025-04-15", "US:BBB"] *= 0.1
            return closes, bench
        return f
    monkeypatch.setattr(A, "_us_closes", fetch("2025-04-15", False))
    iid = make(base_date="2024-03-01")
    A.compute(1, iid)
    A._price_cache._d.clear()
    monkeypatch.setattr(A, "_us_closes", fetch("2025-04-22", True))
    monkeypatch.setattr(A, "_split_on", lambda t, d: t == "US:BBB")     # Yahoo lists a 10:1 split that day
    r = A.refresh_all()[0]
    assert not r["ok"] and "拆股" in r["error"] and db.days(iid)[-1]["d"] == "2025-04-15"
    monkeypatch.setattr(A, "_split_on", lambda t, d: False)            # no split: a genuine -90% day
    assert A.refresh_all()[0]["ok"] and db.days(iid)[-1]["d"] == "2025-04-22"


def test_a_change_made_at_the_weekend_takes_effect_on_monday(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([], end="2025-06-27"))
    iid = make(base_date="2024-03-01")
    A.compute(1, iid)
    aged(db, iid)
    monkeypatch.setattr(A, "_today", lambda m: date(2025, 6, 28))     # Saturday; Friday is stored
    assert A.add_member(1, iid, ticker="CCC", name="C", added_on=None)["added_on"] == "2025-06-30"


def test_a_download_that_comes_back_empty_says_so_rather_than_blaming_one_stock(db, monkeypatch):
    monkeypatch.setattr(A, "_us_closes", long_prices([], end="2025-04-15"))
    iid = make(base_date="2024-03-01", members=[{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}])
    A.compute(1, iid)
    A._price_cache._d.clear()
    full = long_prices([])
    monkeypatch.setattr(A, "_us_closes", lambda t, s: (full(t, s)[0].iloc[:, :0], full(t, s)[1]))
    r = A.refresh_all()[0]
    assert not r["ok"] and "整体为空" in r["error"]
