"""
Two watchlists in one table, and the jobs that must not trip over each other.

Offline. The table has no market column and does not need one: the canonical
symbol already says which market a row belongs to — a bare six-digit code is
an A-share, "US:AAPL" is not — which is the whole reason nothing stored before
today had to be migrated.

What that buys has to be paid for in exactly one place: each nightly run must
take only its own market. The A-share job runs at 20:00 Beijing against
Tushare; handing it "US:AAPL" would log an error a night, per symbol, with
nobody watching. That is what most of this file is about.

    python -m pytest api/tests/test_watchlist_markets.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import scan_watchlists as sw  # noqa: E402
import supply_chain as sc  # noqa: E402


# ── which job takes which symbol ─────────────────────────────────────────────
@pytest.mark.parametrize("symbol, cn, na", [
    ("600519", True, False),
    ("000001", True, False),
    ("US:AAPL", False, True),
    ("CA:SHOP.TO", False, True),
])
def test_each_nightly_run_takes_only_its_own_market(symbol, cn, na):
    assert sw._in_market(symbol, "CN") is cn
    assert sw._in_market(symbol, "NA") is na


def test_all_takes_everything():
    for symbol in ("600519", "US:AAPL", "CA:SHOP.TO", "nonsense!!"):
        assert sw._in_market(symbol, "ALL") is True


def test_a_row_nothing_can_parse_is_left_to_neither_job():
    """
    A corrupt row is one job's bug, not both jobs' crash. It still shows on
    the page — under its own group — so it can be deleted.
    """
    assert sw._in_market("not a ticker", "CN") is False
    assert sw._in_market("not a ticker", "NA") is False
    assert sw._in_market("", "CN") is False


def test_a_mixed_watchlist_splits_cleanly_with_nothing_lost():
    mixed = ["600519", "US:AAPL", "000001", "CA:SHOP.TO", "US:NVDA"]
    cn = [t for t in mixed if sw._in_market(t, "CN")]
    na = [t for t in mixed if sw._in_market(t, "NA")]

    assert cn == ["600519", "000001"]
    assert na == ["US:AAPL", "CA:SHOP.TO", "US:NVDA"]
    assert sorted(cn + na) == sorted(mixed)       # partition, not a filter


# ── the supply-chain prompt ──────────────────────────────────────────────────
def test_the_schema_is_identical_in_every_market():
    """
    The graph a chart draws must not change shape by exchange. Only the
    sourcing instruction differs.
    """
    bodies = {sc._prompt(m)[0].split("CRITICAL RULES")[0].split("\n", 1)[1]
              for m in ("CN", "US", "CA")}
    assert len(bodies) == 1


@pytest.mark.parametrize("market, expect, forbid", [
    ("CN", "cninfo.com.cn", "10-K"),
    ("US", "10-K", "cninfo.com.cn"),
    ("CA", "SEDAR+", "cninfo.com.cn"),
])
def test_each_market_is_pointed_at_filings_that_exist_for_it(market, expect, forbid):
    """
    Telling a model to verify a US filer against cninfo.com.cn is telling it
    to consult a database that does not contain the company. What comes back
    is invention, and it looks exactly like recall.
    """
    prompt = sc._prompt(market)[0]
    assert expect in prompt
    assert forbid not in prompt or f"Do NOT cite" in prompt


def test_an_unknown_market_falls_back_to_a_shares_rather_than_breaking():
    prompt, label = sc._prompt("ZZ")
    assert prompt == sc._prompt("CN")[0]
    assert "A-share" in label


def test_the_json_schema_survives_substitution():
    """
    The prompt embeds a JSON schema, so str.format would have to have every
    brace doubled. One missed brace silently breaks the whole prompt.
    """
    for market in ("CN", "US", "CA"):
        prompt = sc._prompt(market)[0]
        assert '"macro_sectors": [' in prompt
        assert '{"source":' in prompt
        assert "{speciality}" not in prompt and "{sources}" not in prompt


def test_the_company_label_matches_the_market():
    assert sc._prompt("US")[1] == "US-listed"
    assert sc._prompt("CA")[1] == "Canadian-listed"
    assert sc._prompt("CN")[1] == "Chinese A-share"


# ── the filter where it actually runs ────────────────────────────────────────
class Scanned(Exception):
    """Carries the ticker list main() decided to scan, and stops it there."""

    def __init__(self, tickers):
        super().__init__("captured")
        self.tickers = list(tickers)


def run_main(monkeypatch, watchlists, argv):
    """main() up to the point it hands tickers to the scanner, and no further."""
    import types

    dm = types.ModuleType("data_manager")
    dm.get_all_watchlists = lambda: watchlists
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    monkeypatch.setitem(sys.modules, "watchlist_scan", types.ModuleType("watchlist_scan"))
    monkeypatch.setattr(sw, "_run", lambda tickers, workers, label: (_ for _ in ()).throw(
        Scanned(tickers)))
    monkeypatch.setattr(sys, "argv", ["scan_watchlists.py", *argv])

    try:
        sw.main()
    except Scanned as got:
        return got.tickers
    return []


def test_the_a_share_run_never_reaches_a_us_symbol(monkeypatch):
    """
    The one that matters. Without this filter the 20:00 Beijing job fetches
    US tickers from Tushare and logs an error a night, per symbol, with nobody
    watching.
    """
    books = {1: ["600519", "US:AAPL"], 2: ["000001", "CA:SHOP.TO", "US:AAPL"]}
    assert run_main(monkeypatch, books, ["--market", "CN"]) == ["000001", "600519"]


def test_the_north_american_run_never_reaches_an_a_share(monkeypatch):
    books = {1: ["600519", "US:AAPL"], 2: ["000001", "CA:SHOP.TO"]}
    assert run_main(monkeypatch, books, ["--market", "NA"]) == ["CA:SHOP.TO", "US:AAPL"]


def test_the_default_run_is_the_a_share_one(monkeypatch):
    """The schedule that already exists keeps doing exactly what it did."""
    books = {1: ["600519", "US:AAPL"]}
    assert run_main(monkeypatch, books, []) == ["600519"]


def test_a_watchlist_with_nothing_for_this_market_exits_quietly(monkeypatch):
    """Not an error: a user may hold only A-shares, and the NA job still runs."""
    assert run_main(monkeypatch, {1: ["600519"]}, ["--market", "NA"]) == []


# ── the market reaches the model ─────────────────────────────────────────────
def test_the_generator_sends_the_market_specific_prompt(monkeypatch):
    import types

    seen = {}

    def call_json(system, user, **kw):
        seen["system"], seen["user"] = system, user
        return {"products": ["x"]}

    fake = types.ModuleType("ai_client")
    fake.call_json = call_json
    monkeypatch.setattr(sc, "ai_client", fake)

    sc.generate_supply_chain_graph("US:AAPL", "Apple Inc.", "US")
    assert "10-K" in seen["system"] and "cninfo.com.cn" not in seen["system"]
    assert "US-listed company" in seen["user"]

    sc.generate_supply_chain_graph("600519", "贵州茅台", "CN")
    assert "cninfo.com.cn" in seen["system"]
    assert "Chinese A-share company" in seen["user"]


def test_the_generator_still_defaults_to_a_shares(monkeypatch):
    """Every existing caller passes no market and must not change behaviour."""
    import types

    seen = {}
    fake = types.ModuleType("ai_client")
    fake.call_json = lambda system, user, **kw: seen.update(system=system) or {"products": ["x"]}
    monkeypatch.setattr(sc, "ai_client", fake)

    sc.generate_supply_chain_graph("600519", "贵州茅台")
    assert "cninfo.com.cn" in seen["system"]
