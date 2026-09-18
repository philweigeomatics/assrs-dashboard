"""
portfolio: one book out of several accounts, and what its risk really is.

Offline. Two groups of tests, for the two ways this goes wrong quietly:

  * Consolidation. The same name in a TFSA and a margin account is ONE holding
    with two account rows, and a USD position must never be counted as though
    it were CAD — a book a third smaller than it is looks entirely plausible.
  * Risk. The statistics are built from synthetic series whose answers are
    known in advance: a book that IS the index has beta 1 and zero tracking
    error; a book of twice the index has beta 2; risk contributions sum to the
    portfolio's volatility and to 100%.

    python -m pytest api/tests/test_portfolio.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import portfolio as pf  # noqa: E402

BENCH = "^GSPC"


# ── symbols ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("symbol, exchange, want", [
    ("AAPL", "NASDAQ", "AAPL"),
    ("NVDA", "", "NVDA"),
    ("BRK.B", "NYSE", "BRK-B"),               # Yahoo writes class shares with a dash
    ("ENB.TO", "TSX", "ENB.TO"),
    ("ENB", "TSX", "ENB.TO"),                 # Questrade sometimes omits the suffix
    ("XYZ.VN", "TSXV", "XYZ.V"),              # the one spelling the two disagree on
    ("XYZ.VN", "", "XYZ.V"),
    ("ABC", "CNSX", "ABC.CN"),
])
def test_questrade_tickers_map_to_yahoo_spelling(symbol, exchange, want):
    assert pf.to_yahoo(symbol, exchange) == want


def test_something_unmappable_returns_nothing_rather_than_a_guess():
    """
    A wrong mapping prices one holding off another company's chart and never
    announces itself. An empty answer produces a visible warning instead.
    """
    assert pf.to_yahoo("", "NYSE") is None
    assert pf.to_yahoo("AAPL 20Jan26C200.00", "") is None


@pytest.mark.parametrize("quote_type, security_type, want", [
    ("ETF", "Stock", "ETF"),                  # Questrade calls every ETF a Stock
    ("EQUITY", "Stock", "股票"),
    (None, "Stock", "股票"),
    (None, "Option", "期权"),
    ("MUTUALFUND", "MutualFund", "基金"),
])
def test_etfs_are_identified_from_yahoo_because_questrade_cannot(
        quote_type, security_type, want):
    assert pf.kind_of(quote_type, security_type) == want


# ── consolidation ────────────────────────────────────────────────────────────
ACCOUNTS = [
    {"id": "A1", "type": "TFSA", "label": "免税账户 TFSA"},
    {"id": "A2", "type": "Margin", "label": "保证金 Margin"},
    {"id": "A3", "type": "RRSP", "label": "退休账户 RRSP"},
]

META = {
    "AAPL": {"name": "Apple", "currency": "USD", "exchange": "NASDAQ",
             "kind": "股票", "yahoo": "AAPL"},
    "VFV.TO": {"name": "Vanguard S&P 500", "currency": "CAD", "exchange": "TSX",
               "kind": "ETF", "yahoo": "VFV.TO"},
}


def pos(symbol, qty, price, avg, **kw):
    return {"symbol": symbol, "openQuantity": qty, "currentPrice": price,
            "averageEntryPrice": avg, "currentMarketValue": qty * price,
            "totalCost": qty * avg, "openPnl": qty * (price - avg), **kw}


def bal(cad_cash=0.0, usd_cash=0.0):
    return {"perCurrencyBalances": [
        {"currency": "CAD", "cash": cad_cash, "marketValue": 0, "totalEquity": cad_cash},
        {"currency": "USD", "cash": usd_cash, "marketValue": 0, "totalEquity": usd_cash},
    ]}


def book(**over):
    args = dict(
        accounts=ACCOUNTS,
        positions={
            "A1": [pos("AAPL", 10, 200.0, 150.0), pos("VFV.TO", 50, 140.0, 100.0)],
            "A2": [pos("AAPL", 5, 200.0, 180.0)],
            "A3": [],
        },
        balances={"A1": bal(1000, 200), "A2": bal(500), "A3": bal(0)},
        meta=META, base="CAD", rates={"USD": 1.40}, rate_source="questrade",
    )
    args.update(over)
    return pf.consolidate(**args)


def holding(res, symbol):
    return next(h for h in res["holdings"] if h["symbol"] == symbol)


def test_the_same_name_in_two_accounts_is_one_holding():
    """"How much AAPL do I own" must not need mental arithmetic."""
    res = book()
    apple = holding(res, "AAPL")

    assert apple["quantity"] == 15
    assert len(apple["accounts"]) == 2
    assert apple["split"] is True
    assert {a["id"] for a in apple["accounts"]} == {"A1", "A2"}


def test_average_cost_stays_per_account_as_well_as_combined():
    """
    ACB is tracked per account, a TFSA has no cost base for tax at all, and an
    RRSP loss is not harvestable. One blended number matches nothing.
    """
    apple = holding(book(), "AAPL")
    by_account = {a["id"]: a["avg_cost"] for a in apple["accounts"]}

    assert by_account == {"A1": 150.0, "A2": 180.0}
    # The blended figure is still there for sizing, and is cost-weighted.
    assert apple["avg_cost"] == pytest.approx((10 * 150 + 5 * 180) / 15)


def test_market_value_is_reported_per_account_and_in_total():
    res = book()
    accounts = {a["id"]: a for a in res["accounts"]}

    # TFSA: 10 AAPL at $200 USD × 1.40, plus 50 VFV at C$140.
    assert accounts["A1"]["market_value_base"] == pytest.approx(10 * 200 * 1.40 + 50 * 140)
    assert accounts["A2"]["market_value_base"] == pytest.approx(5 * 200 * 1.40)
    assert accounts["A3"]["market_value_base"] == 0
    assert res["totals"]["market_value"] == pytest.approx(
        sum(a["market_value_base"] for a in res["accounts"]))


def test_a_us_holding_is_converted_not_counted_as_canadian():
    """A third of the book quietly missing is the failure that looks fine."""
    apple = holding(book(), "AAPL")
    assert apple["market_value"] == pytest.approx(3000.0)         # USD, as reported
    assert apple["market_value_base"] == pytest.approx(4200.0)    # CAD


def test_a_missing_exchange_rate_is_a_warning_not_a_one_to_one():
    res = book(rates={})            # no USD rate at all
    # Still listed — it converts to zero, and deleting it as "dust" would hide
    # the very holding the warning is about.
    apple = holding(res, "AAPL")

    assert apple["market_value_base"] == 0
    assert apple["market_value"] == pytest.approx(3000.0)
    assert any("AAPL" in w and "USD" in w for w in res["warnings"])
    # The CAD side is unaffected and still totals correctly.
    assert res["totals"]["market_value"] == pytest.approx(50 * 140)


def test_cash_is_converted_too():
    res = book()
    total_cash = 1000 + 200 * 1.40 + 500
    assert res["totals"]["cash"] == pytest.approx(total_cash)
    assert res["totals"]["equity"] == pytest.approx(
        res["totals"]["market_value"] + total_cash)


def test_weights_are_of_the_whole_book_and_sum_to_a_hundred():
    res = book()
    assert sum(h["weight_pct"] for h in res["holdings"]) == pytest.approx(100.0)
    # Largest first, AFTER conversion: 50 × C$140 beats 15 × US$200 × 1.40,
    # which is the whole reason the sort happens on the base-currency value.
    assert [h["symbol"] for h in res["holdings"]] == ["VFV.TO", "AAPL"]


def test_holdings_are_ranked_after_conversion_not_before():
    """
    US$3,000 outranks C$3,500 once it is C$4,200. Ranking on the native number
    puts the wrong name at the top of the page and in every "biggest position"
    reading taken from it.
    """
    meta = {**META, "BCE.TO": {"name": "BCE", "currency": "CAD", "exchange": "TSX",
                               "kind": "股票", "yahoo": "BCE.TO"}}
    res = book(meta=meta, positions={
        "A1": [pos("AAPL", 15, 200.0, 150.0)],       # US$3,000 → C$4,200
        "A2": [pos("BCE.TO", 100, 35.0, 30.0)],      # C$3,500
        "A3": [],
    })
    assert [h["symbol"] for h in res["holdings"]] == ["AAPL", "BCE.TO"]


def test_a_position_closed_today_is_not_a_holding():
    res = book(positions={"A1": [pos("AAPL", 0, 200.0, 150.0)], "A2": [], "A3": []})
    assert res["holdings"] == []
    assert res["totals"]["positions"] == 0


def test_a_residual_position_is_dropped_but_still_counted():
    """
    Questrade keeps rows for fractional residue worth a fraction of a cent.
    They cannot be sold and cannot move the book, but a row that vanishes
    without trace is worse than one that is merely small.
    """
    res = book(positions={
        "A1": [pos("AAPL", 10, 200.0, 150.0), pos("VFV.TO", 0.00002, 140.0, 100.0)],
        "A2": [], "A3": [],
    })
    assert [h["symbol"] for h in res["holdings"]] == ["AAPL"]
    assert res["totals"]["dust"] == 1
    assert res["totals"]["positions"] == 1


def test_dust_is_judged_in_the_positions_own_currency():
    """Converted value is zero when a rate is missing; that is not dust."""
    res = book(rates={}, positions={"A1": [pos("AAPL", 10, 200.0, 150.0)],
                                    "A2": [], "A3": []})
    assert [h["symbol"] for h in res["holdings"]] == ["AAPL"]
    assert res["totals"]["dust"] == 0


def test_every_holding_carries_a_machine_readable_group():
    """The Chinese label is for reading; the client filters on this."""
    res = book()
    assert {h["symbol"]: h["group"] for h in res["holdings"]} == {
        "AAPL": "stock", "VFV.TO": "etf"}


@pytest.mark.parametrize("kind, want", [
    ("股票", "stock"), ("ETF", "etf"), ("基金", "etf"),
    ("期权", "other"), ("债券", "other"),
])
def test_groups_cover_every_kind(kind, want):
    assert pf.group_of(kind) == want


def test_the_mix_splits_by_currency_and_by_instrument_kind():
    mix = book()["mix"]
    assert {m["name"] for m in mix["currency"]} == {"USD", "CAD"}
    assert {m["name"] for m in mix["kind"]} == {"股票", "ETF"}
    assert sum(m["pct"] for m in mix["currency"]) == pytest.approx(100.0)


def test_a_holding_with_no_price_source_is_flagged():
    meta = {**META, "AAPL": {**META["AAPL"], "yahoo": None}}
    res = book(meta=meta)
    assert any("AAPL" in w for w in res["warnings"])


def test_the_rate_actually_used_is_reported():
    res = book()
    assert res["fx"]["rates"]["USD"] == 1.40
    assert res["fx"]["source"] == "questrade"


# ── the broker's own FX ──────────────────────────────────────────────────────
def test_the_rate_is_read_out_of_the_brokers_own_combined_balances():
    """
    Questrade states the whole account in each currency, so the ratio IS the
    rate it used — which keeps the page's totals matching the statement.
    """
    combined = [{"currency": "CAD", "totalEquity": 14000.0},
                {"currency": "USD", "totalEquity": 10000.0}]
    assert pf.implied_fx(combined, "CAD", "USD") == pytest.approx(1.40)


def test_an_implausible_implied_rate_is_refused():
    """An empty sleeve produces a ratio that would rescale the whole book."""
    assert pf.implied_fx([{"currency": "CAD", "totalEquity": 14000.0},
                          {"currency": "USD", "totalEquity": 1.0}], "CAD", "USD") is None
    assert pf.implied_fx([{"currency": "CAD", "totalEquity": 0.0},
                          {"currency": "USD", "totalEquity": 100.0}], "CAD", "USD") is None
    assert pf.implied_fx([], "CAD", "USD") is None


# ── risk ─────────────────────────────────────────────────────────────────────
def prices(n=600, seed=0, extra=None) -> pd.DataFrame:
    """A benchmark, a copy of it, a 2× version, and an uncorrelated name."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    bench = np.r_[0, rng.normal(0.0004, 0.010, n - 1)]
    noise = np.r_[0, rng.normal(0.0004, 0.010, n - 1)]
    cols = {
        BENCH: 100 * np.exp(np.cumsum(bench)),
        "SAME": 100 * np.exp(np.cumsum(bench)),
        "TWICE": 100 * np.exp(np.cumsum(2 * bench)),
        "OTHER": 100 * np.exp(np.cumsum(noise)),
    }
    cols.update(extra or {})
    return pd.DataFrame(cols, index=idx)


def test_a_book_that_is_the_index_has_beta_one_and_no_tracking_error():
    r = pf.risk({"SAME": 1.0}, prices(), BENCH)
    assert r["beta"] == pytest.approx(1.0, abs=0.01)
    assert r["r2"] == pytest.approx(1.0, abs=0.01)
    assert r["tracking_error_pct"] == pytest.approx(0.0, abs=0.05)


def test_a_book_of_twice_the_index_has_beta_two():
    r = pf.risk({"TWICE": 1.0}, prices(), BENCH)
    assert r["beta"] == pytest.approx(2.0, abs=0.02)
    assert r["ann_vol_pct"] > pf.risk({"SAME": 1.0}, prices(), BENCH)["ann_vol_pct"]


def test_diversifying_into_an_uncorrelated_name_lowers_the_volatility():
    both = pf.risk({"SAME": 0.5, "OTHER": 0.5}, prices(), BENCH)
    one = pf.risk({"SAME": 1.0}, prices(), BENCH)
    assert both["ann_vol_pct"] < one["ann_vol_pct"]


def test_risk_contributions_account_for_the_whole_portfolio():
    """
    Marginal contributions sum to the portfolio volatility, so as percentages
    they sum to 100. A 4% position in something wild can carry more risk than
    a 20% position in a utility, and that only shows if the decomposition is
    right.
    """
    r = pf.risk({"SAME": 0.7, "OTHER": 0.2, "TWICE": 0.1}, prices(), BENCH)
    assert sum(h["risk_pct"] for h in r["holdings"]) == pytest.approx(100.0, abs=0.5)
    assert [h["symbol"] for h in r["holdings"]][0] == "SAME"   # ranked by risk


def test_a_leveraged_sleeve_carries_more_risk_than_its_weight():
    r = pf.risk({"SAME": 0.9, "TWICE": 0.1}, prices(), BENCH)
    twice = next(h for h in r["holdings"] if h["symbol"] == "TWICE")
    assert twice["risk_pct"] > twice["weight_pct"]


def test_concentration_says_how_few_names_the_book_really_is():
    """Ten holdings with one at 55% is not ten holdings."""
    px = prices(extra={f"N{i}": 100 * np.exp(np.cumsum(
        np.r_[0, np.random.default_rng(i).normal(0, 0.01, 599)])) for i in range(9)})
    weights = {"SAME": 0.55, **{f"N{i}": 0.05 for i in range(9)}}
    c = pf.risk(weights, px, BENCH)["concentration"]

    assert c["positions"] == 10
    assert c["top1_pct"] == pytest.approx(55.0, abs=0.1)
    assert c["effective_n"] < 4


def test_the_payload_says_it_is_a_replay_of_todays_weights():
    """It is not the account's performance record, and must not be read as one."""
    r = pf.risk({"SAME": 1.0}, prices(), BENCH)
    assert "当前持仓" in r["basis"]


def test_it_reports_how_much_of_the_book_it_could_actually_price():
    """A beta for 70% of a portfolio is not a beta for the portfolio."""
    r = pf.risk({"SAME": 0.7, "UNPRICEABLE": 0.3}, prices(), BENCH)
    assert r["covered_pct"] == pytest.approx(70.0, abs=0.1)

    # And the part it CAN price is measured at full weight. Leaving the 0.7 in
    # place would report a book of 70% cash — beta 0.7, volatility a third
    # short — rather than what the priced holdings actually did.
    assert r["beta"] == pytest.approx(1.0, abs=0.01)
    assert r["ann_vol_pct"] == pytest.approx(
        pf.risk({"SAME": 1.0}, prices(), BENCH)["ann_vol_pct"], abs=0.01)


def test_a_holding_with_too_little_history_of_its_own_is_dropped():
    with pytest.raises(LookupError, match="足够的历史行情"):
        pf.risk({"SAME": 1.0}, prices(n=60), BENCH)


def test_a_late_listing_is_dropped_rather_than_halving_everyone_elses_window():
    """
    A holding with 200 sessions of its own passes on its own merits, but the
    SHARED window can only start where it does. Measuring nine names over
    three years beats measuring ten over eight months.
    """
    px = prices(n=600)
    px["LATE"] = px["OTHER"]
    px.loc[px.index[:-200], "LATE"] = float("nan")

    alone = pf.risk({"LATE": 1.0}, px, BENCH)
    assert alone["sessions"] < 220              # on its own it is measurable

    both = pf.risk({"SAME": 0.7, "LATE": 0.3}, px, BENCH)
    assert both["sessions"] > 500               # …but not at everyone's expense
    assert both["excluded"] == [{"symbol": "LATE", "reason": "上市时间过短"}]


# ── the failure that produced "0 trading days available" ─────────────────────
def test_one_unpriceable_holding_does_not_delete_the_whole_report():
    """
    The first live portfolio held a name Yahoo has no data for. An all-NaN
    column plus dropna(how="any") returns zero rows, and the panel reported
    "可用历史仅 0 个交易日" for a book with three years of history in it.
    """
    px = prices()
    px["DEAD"] = float("nan")

    r = pf.risk({"SAME": 0.6, "OTHER": 0.3, "DEAD": 0.1}, px, BENCH)
    assert r["sessions"] > 500
    assert [e["symbol"] for e in r["excluded"]] == ["DEAD"]
    assert r["excluded"][0]["reason"] == "行情缺失"
    assert r["covered_pct"] == pytest.approx(90.0, abs=0.1)


def test_a_recent_listing_does_not_truncate_everyone_elses_history():
    """
    One holding that IPO'd last month would otherwise cut the whole report to
    a month. It is dropped and named instead.
    """
    px = prices()
    px["NEW"] = px["OTHER"]
    px.loc[px.index[:-30], "NEW"] = float("nan")

    r = pf.risk({"SAME": 0.8, "NEW": 0.2}, px, BENCH)
    assert r["sessions"] > 500
    assert r["excluded"] == [{"symbol": "NEW", "reason": "上市时间过短"}]


def test_no_data_and_too_little_data_are_told_apart():
    """
    One is usually a wrong ticker mapping, the other is a real holding that
    has not existed for long. Different problems, different fixes.
    """
    px = prices()
    px["DEAD"] = float("nan")
    px["NEW"] = px["OTHER"]
    px.loc[px.index[:-30], "NEW"] = float("nan")

    r = pf.risk({"SAME": 0.8, "DEAD": 0.1, "NEW": 0.1}, px, BENCH)
    assert {e["symbol"]: e["reason"] for e in r["excluded"]} == {
        "DEAD": "行情缺失", "NEW": "上市时间过短"}


def test_holidays_are_filled_rather_than_thrown_away():
    """
    TSX and NYSE do not share a calendar. Dropping every date one of them was
    shut discards real sessions for the OTHER market, which did trade.

    Pinned against the no-gap case rather than a loose floor: ten missing days
    out of six hundred slips under any threshold loose enough to be safe.
    """
    clean = prices()
    holed = prices()
    holed.loc[holed.index[100:110], "OTHER"] = float("nan")

    a = pf.risk({"SAME": 0.5, "OTHER": 0.5}, clean, BENCH)
    b = pf.risk({"SAME": 0.5, "OTHER": 0.5}, holed, BENCH)

    assert b["sessions"] == a["sessions"]
    assert b["excluded"] == []


def test_nothing_measurable_at_all_names_what_was_missing():
    px = prices()
    px["DEAD"] = float("nan")
    with pytest.raises(LookupError, match="DEAD"):
        pf.risk({"DEAD": 1.0}, px, BENCH)


def test_a_missing_benchmark_is_refused_by_name():
    with pytest.raises(LookupError, match=r"\^IXIC"):
        pf.risk({"SAME": 1.0}, prices(), "^IXIC")


def test_nothing_priceable_at_all_is_refused():
    with pytest.raises(LookupError, match="行情"):
        pf.risk({"NOPE": 1.0}, prices(), BENCH)


def test_capture_ratios_are_about_a_hundred_for_an_index_tracker():
    r = pf.risk({"SAME": 1.0}, prices(), BENCH)
    assert r["up_capture_pct"] == pytest.approx(100.0, abs=1.0)
    assert r["down_capture_pct"] == pytest.approx(100.0, abs=1.0)


def test_drawdown_and_var_are_losses_not_magnitudes():
    r = pf.risk({"OTHER": 1.0}, prices(), BENCH)
    assert r["max_drawdown_pct"] < 0
    assert r["var95_pct"] < 0
    assert r["worst_day_pct"] <= r["var95_pct"]


def test_the_benchmarks_own_row_claims_no_beta_against_itself():
    r = pf.risk({"OTHER": 1.0}, prices(), BENCH)["benchmark_stats"]
    assert r["beta"] is None and r["up_capture_pct"] is None
    assert r["ann_vol_pct"] > 0
