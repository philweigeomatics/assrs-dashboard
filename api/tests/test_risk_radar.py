"""
🧭 风险雷达 — the weighting, the hindsight rule, the sensitivity regression,
the AI output validation, and a refresh run against a fake database.

The scoring tests pin the properties the page's wording depends on: downside
and upside are never netted, too little evidence reads 证据不足 rather than
低, and nothing dated after the as-of day counts. The refresh tests pin the
cost rule — an item is sent to the AI once, ever — and that a broken feed is
reported rather than raised.

    python -m pytest api/tests/test_risk_radar.py -q
"""

from __future__ import annotations

import os
import sys
import types
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import risk_ai  # noqa: E402
import risk_radar as R  # noqa: E402
import risk_sources as S  # noqa: E402

TODAY = date(2026, 11, 2)


def ev(risk="earnings", dirn="down", age=0, kind="filing", strength=2, fact=True, title=None):
    d = (TODAY - timedelta(days=age)).isoformat()
    title = title or f"{risk}-{dirn}-{age}-{kind}-{strength}"
    return {"eid": R.evidence_id("t", title, d), "d": d, "source": "t", "kind": kind,
            "title": title, "url": None, "risk": risk, "dir": dirn, "strength": strength,
            "fact": fact, "note": None, "tagged_by": "rule"}


# ── weights ──────────────────────────────────────────────────────────────────
def test_an_item_halves_at_its_half_life():
    fresh = R.weight(ev(age=0), TODAY)
    aged = R.weight(ev(age=R.HALF_LIFE["filing"]), TODAY)
    assert aged == pytest.approx(fresh / 2)


def test_future_stale_and_untagged_items_carry_nothing():
    assert R.weight(ev(age=-1), TODAY) == 0
    assert R.weight(ev(age=R.WINDOW_DAYS + 1), TODAY) == 0
    assert R.weight({**ev(), "risk": None}, TODAY) == 0
    assert R.weight({**ev(), "risk": "made_up"}, TODAY) == 0


def test_an_opinion_counts_less_than_a_fact():
    assert R.weight(ev(fact=False), TODAY) == pytest.approx(R.OPINION_WEIGHT * R.weight(ev(), TODAY))


def test_downside_and_upside_are_scored_separately_not_netted():
    items = [ev("commodities", "down", a, strength=3) for a in range(3)] + \
            [ev("commodities", "up", a, strength=3) for a in range(3)]
    s = R.score(items, "commodities", "高", TODAY)
    assert s["down"] > 50 and s["up"] > 50 and s["down"] == s["up"]


def test_more_exposure_means_more_pressure_from_the_same_evidence():
    items = [ev("rates", "down", a) for a in range(4)]
    assert R.score(items, "rates", "高", TODAY)["down"] > R.score(items, "rates", "低", TODAY)["down"]


def test_too_little_evidence_is_not_low_risk():
    assert R.tier(90, R.MIN_EVIDENCE - 1) == "证据不足"
    assert R.tier(90, R.MIN_EVIDENCE) == "高"
    assert [R.tier(v, 5) for v in (70, 55, 40, 25, 0)] == ["高", "偏高", "中", "偏低", "低"]


def test_a_later_item_never_changes_an_earlier_day():
    items = [ev("legal", "down", a) for a in (10, 20, 30)]
    before = R.score(items, "legal", "中", TODAY - timedelta(days=7))
    items.append(ev("legal", "down", 0, strength=3))
    assert R.score(items, "legal", "中", TODAY - timedelta(days=7)) == before
    assert R.score(items, "legal", "中", TODAY)["down"] > before["down"]


# ── 敞口 ──────────────────────────────────────────────────────────────────────
def _walk(rng, n, vol=0.01):
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, n))),
                     index=pd.bdate_range("2022-01-03", periods=n))


def test_the_regression_finds_a_real_factor_and_ignores_a_market_proxy():
    rng = np.random.default_rng(1)
    n = 800
    mkt = _walk(rng, n)
    copper = _walk(rng, n, 0.015)
    lookalike = mkt * np.exp(rng.normal(0, 0.002, n))        # copper-shaped name, market inside
    r = 1.0 * np.log(mkt).diff() + 0.5 * np.log(copper).diff() + rng.normal(0, 0.003, n)
    stock = pd.Series(100 * np.exp(np.nancumsum(r)), index=mkt.index)
    spec = [{"key": "market", "label": "M", "risk": "market"},
            {"key": "copper", "label": "Cu", "risk": "commodities"},
            {"key": "gold", "label": "Au", "risk": "commodities"}]
    got = {x["key"]: x for x in R.measure_exposure(
        stock, {"market": mkt, "copper": copper, "gold": lookalike}, spec, mkt.index[0])}
    assert got["market"]["beta"] == pytest.approx(1.0, abs=0.15)
    assert got["copper"]["beta"] == pytest.approx(0.5, abs=0.1) and got["copper"]["level"] == "高"
    # Its own movement is noise to this stock: |t| is chance (−1.6 on this
    # seed), never the 高 a plain correlation with the stock would give it.
    assert abs(got["gold"]["t"]) < 3 and got["gold"]["level"] != "高"
    assert R.exposure_by_risk(list(got.values()))["commodities"]["key"] == "copper"


def test_a_factor_move_is_filed_by_its_effect_on_this_stock():
    rng = np.random.default_rng(2)
    oil = _walk(rng, 400, 0.01)
    oil.iloc[-20:] = oil.iloc[-21] * np.linspace(1.0, 1.25, 20)   # +25% in a month
    reading = {"key": "oil", "label": "油", "risk": "commodities", "rate": False,
               "beta": -0.4, "t": -4.0, "r2": 0.3, "weeks": 80, "level": "高"}
    items = R.factor_move_items([reading], {"oil": oil}, oil.index[-1].date())
    assert len(items) == 1
    assert items[0]["dir"] == "down" and items[0]["risk"] == "commodities"
    assert items[0]["tagged_by"] == "rule" and items[0]["strength"] >= 2
    assert R.factor_move_items([{**reading, "level": "低"}], {"oil": oil}, oil.index[-1].date()) == []


# ── the board ────────────────────────────────────────────────────────────────
def test_an_empty_board_says_insufficient_not_safe():
    b = R.assemble("300750", "X", "CN", [], profile=None, readings=[], synthesis=None,
                   sources=[], as_of=TODAY)
    assert b["summary"]["down"] is None and b["summary"]["down_tier"] == "证据不足"
    assert all(r["tier"] == "证据不足" for r in b["risks"])
    assert len(b["risks"]) == len(R.TAXONOMY)


def test_the_top_risk_and_the_summary_use_only_risks_with_enough_evidence():
    items = [ev("trade", "down", a, strength=3) for a in range(5)] + \
            [ev("legal", "down", 0, strength=3)]               # one strong item: not enough
    b = R.assemble("US:X", "X", "US", items, profile=None, readings=[], synthesis=None,
                   sources=[], as_of=TODAY)
    assert b["summary"]["top"]["id"] == "trade"
    legal = next(r for r in b["risks"] if r["id"] == "legal")
    assert legal["tier"] == "证据不足" and legal["n"] == 1
    trade = next(r for r in b["risks"] if r["id"] == "trade")
    assert trade["d7_down"] == trade["down"] - R.score(items, "trade", "中", TODAY - timedelta(days=7))["down"]
    assert len(trade["trend"]["down"]) == R.TREND_DAYS and trade["trend"]["down"][-1] == trade["down"]


def test_measured_exposure_beats_the_ai_judgement():
    reading = {"key": "rate", "label": "10Y", "risk": "rates", "rate": True,
               "beta": -0.05, "t": -3.5, "r2": 0.2, "weeks": 100, "level": "高"}
    b = R.assemble("US:X", "X", "US", [], readings=[reading], synthesis=None, sources=[],
                   profile={"exposure": {"rates": {"level": "低", "why": "?"},
                                         "legal": {"level": "高", "why": "诉讼多"}}},
                   as_of=TODAY)
    by = {r["id"]: r for r in b["risks"]}
    assert (by["rates"]["exposure"], by["rates"]["exposure_source"]) == ("高", "measured")
    assert (by["legal"]["exposure"], by["legal"]["exposure_source"]) == ("高", "ai")
    assert by["fx"]["exposure_source"] == "default"


# ── AI output validation ─────────────────────────────────────────────────────
def test_invented_risks_and_directions_are_dropped_not_guessed():
    assert risk_ai._clean_tag({"risk": "vibes", "dir": "down"}) is None
    assert risk_ai._clean_tag({"risk": "earnings", "dir": "sideways"}) is None
    assert risk_ai._clean_tag({"risk": None})["risk"] is None            # irrelevant, kept
    t = risk_ai._clean_tag({"risk": "earnings", "dir": "up", "strength": 9, "fact": False})
    assert t["strength"] == 3 and t["fact"] is False


def test_citations_are_rewritten_to_evidence_ids(monkeypatch):
    items = [ev("earnings", "up", 0, title="a"), ev("earnings", "down", 1, title="b")]
    b = R.assemble("X", "X", "CN", items, profile=None, readings=[], synthesis=None,
                   sources=[], as_of=TODAY)
    monkeypatch.setattr(risk_ai.ai_client, "call_json", lambda *a, **k: {"risks": {
        "earnings": {"bear": "毛利承压[E2]", "bull": "订单增长[E1][E9]", "watch": ["年报"]},
        "not_a_risk": {"bear": "x", "bull": "y"}}})
    out = risk_ai.synthesize("X", "CN", b["risks"])["risks"]
    ranked = next(r for r in b["risks"] if r["id"] == "earnings")["evidence"]
    assert out["earnings"]["bear"] == f"毛利承压[#{ranked[1]['eid']}]"
    assert out["earnings"]["bull"] == f"订单增长[#{ranked[0]['eid']}]"   # E9 does not exist
    assert "not_a_risk" not in out


# ── sources ──────────────────────────────────────────────────────────────────
def test_an_upcoming_unlock_keeps_one_id_while_its_date_moves():
    a = S._item("ts_share_float", "data", "2026-11-02", "将解禁 1 亿股", key="unlock|2026-12-15",
                risk="holders", dir="down", strength=2)
    b = S._item("ts_share_float", "data", "2026-11-09", "将解禁 1 亿股", key="unlock|2026-12-15",
                risk="holders", dir="down", strength=2)
    assert a["eid"] == b["eid"] and a["d"] != b["d"]


def test_market_wide_news_is_kept_for_the_company_and_capped_for_keywords(monkeypatch):
    rows = [{"标题": "宁德时代发布新电池", "摘要": "", "发布时间": "2026-11-02 09:00", "链接": "u0"}]
    rows += [{"标题": f"储能招标 {i}", "摘要": "", "发布时间": "2026-11-02 09:00", "链接": f"u{i}"}
             for i in range(1, 40)]
    rows += [{"标题": "无关新闻", "摘要": "", "发布时间": "2026-11-02 09:00", "链接": "x"}]
    fake = types.SimpleNamespace(stock_info_global_em=lambda: pd.DataFrame(rows))
    monkeypatch.setitem(sys.modules, "akshare", fake)
    got = S.cn_feed("news_em")("300750", aliases=["宁德时代"], keywords=["储能"])
    assert got[0]["title"] == "宁德时代发布新电池"
    assert len(got) == 1 + S.MACRO_PER_FEED
    assert not any(g["title"] == "无关新闻" for g in got)


def test_a_broken_feed_is_reported_not_raised(monkeypatch):
    def broken(code, **_):
        raise ConnectionError("site changed")
    monkeypatch.setitem(S.SOURCES, "CN", [("a", "A", broken),
                                          ("b", "B", lambda code, **_: [ev(title="ok")])])
    items, status = S.collect("300750", "CN", "X", {})
    assert [i["title"] for i in items] == ["ok"]
    assert status[0]["ok"] is False and "site changed" in status[0]["reason"]
    assert {k: status[1][k] for k in ("source", "ok", "n", "reason")} == \
        {"source": "b", "ok": True, "n": 1, "reason": None}


def test_a_hanging_feed_is_cut_off_and_reported(monkeypatch):
    import threading
    release = threading.Event()

    def hangs(code, **_):
        release.wait(5)
        return [ev(title="late")]
    monkeypatch.setitem(S.SOURCES, "CN", [("slow", "Slow", hangs),
                                          ("fast", "Fast", lambda code, **_: [ev(title="ok")])])
    items, status = S.collect("300750", "CN", "X", {}, deadline=1)
    release.set()
    assert [i["title"] for i in items] == ["ok"]
    assert status[0]["ok"] is False and "1s" in status[0]["reason"]


def test_one_announcement_of_many_holders_is_one_event(monkeypatch):
    rows = pd.DataFrame({
        "ann_date": ["20261020"] * 5 + ["20261025"],
        "holder_name": [f"高管{i}" for i in range(5)] + ["大股东"],
        "in_de": ["DE"] * 5 + ["IN"],
        "change_ratio": [0.02] * 5 + [0.6],
        "change_vol": [1e4] * 5 + [5e6]})
    monkeypatch.setattr(S, "_pro", lambda: types.SimpleNamespace(stk_holdertrade=lambda **k: rows))
    monkeypatch.setattr(S, "_ts_code", lambda code: f"{code}.SZ")
    got = sorted(S.cn_holder_trades("300750"), key=lambda g: g["d"])
    assert [(g["dir"], g["strength"]) for g in got] == [("down", 1), ("up", 2)]
    assert "5 条" in got[0]["title"]


def test_an_immaterial_day_of_holder_trades_is_not_evidence(monkeypatch):
    rows = pd.DataFrame({"ann_date": ["20261020"], "holder_name": ["机构"], "in_de": ["DE"],
                         "change_ratio": [0.001], "change_vol": [9e4]})
    monkeypatch.setattr(S, "_pro", lambda: types.SimpleNamespace(stk_holdertrade=lambda **k: rows))
    monkeypatch.setattr(S, "_ts_code", lambda code: f"{code}.SZ")
    assert S.cn_holder_trades("300750") == []


def _edgar(monkeypatch, recent, form4=None):
    class Resp:
        def json(self):
            return {"filings": {"recent": recent}}
    monkeypatch.setattr(S, "cik_for", lambda t: 1)
    monkeypatch.setattr(S, "_sec_get", lambda url: Resp())
    monkeypatch.setattr(S, "_form4", lambda cik, acc, doc: (form4 or {})[acc])
    monkeypatch.setattr(S, "date", types.SimpleNamespace(today=lambda: TODAY))
    return S.us_edgar("X")


def test_eight_k_items_are_filed_by_what_the_form_means(monkeypatch):
    got = _edgar(monkeypatch, {
        "form": ["8-K", "8-K", "144", "144", "144", "10-Q"],
        "filingDate": [TODAY.isoformat()] * 6,
        "items": ["2.02,9.01", "4.02,5.02", "", "", "", ""],
        "accessionNumber": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "primaryDocument": ["x"] * 6})
    assert sorted((g["risk"], g["dir"], g["strength"]) for g in got) == [
        ("governance", "down", 1), ("governance", "down", 3), ("holders", "down", 1)]
    assert "3 份 Form 144" in next(g["title"] for g in got if g["source"] == "sec_144")


def test_planned_insider_sales_are_skipped_and_buys_are_strong(monkeypatch):
    got = _edgar(monkeypatch, {
        "form": ["4", "4", "4"], "filingDate": [TODAY.isoformat()] * 3, "items": [""] * 3,
        "accessionNumber": ["p", "s", "b"], "primaryDocument": ["x"] * 3},
        form4={"p": {"who": "CEO", "buy": 0, "sell": 6e7, "planned": True, "url": "u"},
               "s": {"who": "CFO", "buy": 0, "sell": 2e7, "planned": False, "url": "u"},
               "b": {"who": "DIR", "buy": 2e6, "sell": 0, "planned": False, "url": "u"}})
    assert sorted((g["title"].split()[0], g["dir"], g["strength"]) for g in got) == [
        ("CFO", "down", 2), ("DIR", "up", 3)]


# ── refresh, against a fake database ─────────────────────────────────────────
class FakeDB:
    KEYS = {R.EVIDENCE_TABLE: ("symbol", "eid"), R.CACHE_TABLE: ("symbol", "section")}

    def __init__(self, broken=False):
        self.rows = {R.EVIDENCE_TABLE: {}, R.CACHE_TABLE: {}}
        self.broken = broken

    def _match(self, table, filters):
        if self.broken:
            raise RuntimeError("relation does not exist")
        return [r for r in self.rows[table].values()
                if all(r.get(k) == v for k, v in (filters or {}).items())]

    def read_all(self, table, filters=None, **_):
        return pd.DataFrame(self._match(table, filters))

    def read_table(self, table, filters=None, **_):
        return pd.DataFrame(self._match(table, filters))

    def insert_records(self, table, records, upsert=False):
        if self.broken:
            raise RuntimeError("relation does not exist")
        for r in records:
            self.rows[table][tuple(r[k] for k in self.KEYS[table])] = dict(r)


@pytest.fixture
def stubbed(monkeypatch):
    calls = {"tag": 0, "tagged": 0, "profile": 0}
    feed = [ev(title=f"news {i}", age=i, kind="news") for i in range(5)]
    for f in feed:
        f.update(risk=None, dir=None, strength=None, fact=None, tagged_by=None)
    rule = [ev("holders", "down", 0, title="unlock")]

    def tag(batch, name, market):
        calls["tag"] += 1
        calls["tagged"] += len(batch)
        return {it["eid"]: {"risk": "earnings", "dir": "up", "strength": 2, "fact": True,
                            "note": "n"} for it in batch}

    def profile(*a):
        calls["profile"] += 1
        return {"aliases": ["X"], "keywords": [], "exposure": {}}

    monkeypatch.setattr(risk_ai, "tag", tag)
    monkeypatch.setattr(risk_ai, "profile", profile)
    monkeypatch.setattr(risk_ai, "synthesize", lambda name, m, risks: {"risks": {}})
    monkeypatch.setattr(S, "collect", lambda *a: ([dict(x) for x in feed + rule],
                                                  [{"source": "t", "label": "T", "ok": True,
                                                    "n": 6, "reason": None}]))
    monkeypatch.setattr(R, "fetch_factors", lambda m: {})
    monkeypatch.setattr(S, "_pro", lambda: None)
    import markets
    monkeypatch.setattr(markets, "parse", lambda s: (types.SimpleNamespace(
        fetch_ohlcv=lambda code, years=4: None), "300750"))
    return calls


def test_a_refresh_files_new_items_once_and_stores_them(stubbed):
    db = FakeDB()
    b = R.refresh("300750", "X", "CN", db=db)
    assert stubbed["tagged"] == 5 and stubbed["profile"] == 1
    assert len(db.rows[R.EVIDENCE_TABLE]) == 6 and b["persisted"] is True
    earnings = next(r for r in b["risks"] if r["id"] == "earnings")
    assert earnings["n"] == 5 and earnings["up"] > 0

    R.refresh("300750", "X", "CN", db=db)                     # the same items again
    assert stubbed["tagged"] == 5 and stubbed["profile"] == 1  # nothing re-sent, profile cached
    assert R.board("300750", "X", "CN", db=db)["summary"]["stored"] == 6


def test_the_cases_are_written_from_stored_evidence_as_a_second_step(stubbed, monkeypatch):
    db = FakeDB()
    first = R.refresh("300750", "X", "CN", db=db, synthesize=False)
    assert first["synthesized_at"] is None
    seen = {}

    def synth(name, market, risks):
        seen["n"] = sum(len(r["evidence"]) for r in risks)
        return {"risks": {"earnings": {"bear": "暂无证据", "bull": "增长", "watch": []}}}
    monkeypatch.setattr(risk_ai, "synthesize", synth)
    b = R.synthesize("300750", "X", "CN", db=db)
    assert seen["n"] == 6 and b["synthesized_at"]
    assert next(r for r in b["risks"] if r["id"] == "earnings")["bull"] == "增长"
    assert R.board("300750", "X", "CN", db=db)["risks"][8]["bull"] == "增长"
    with pytest.raises(RuntimeError, match="migrations"):
        R.synthesize("300750", "X", "CN", db=FakeDB(broken=True))


def test_a_refresh_without_its_tables_still_returns_a_board(stubbed):
    b = R.refresh("300750", "X", "CN", db=FakeDB(broken=True))
    assert b["persisted"] is False
    assert next(r for r in b["risks"] if r["id"] == "earnings")["n"] == 5
    with pytest.raises(RuntimeError, match="migrations"):
        R.board("300750", "X", "CN", db=FakeDB(broken=True))
