"""
The Questrade activity ledger, and the trap inside it.

Money moved between a user's own accounts appears twice — a withdrawal in
one, a deposit in the other. On a real year the accounts showed USD 16,588.56
of deposits and USD 15,788.56 of it had simply come from the margin account.
Adding the column up without pairing those legs overstates new money by
nearly twentyfold, which is exactly the number someone would carry onto a
contribution-room calculation.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import questrade as qt  # noqa: E402
from api import questrade_api as qa  # noqa: E402


# ── the 30-day window ────────────────────────────────────────────────────────
class _Recorder:
    """A Questrade whose only job is to remember what windows were asked for."""

    def __init__(self, rows_per_call=0):
        self.calls: list[tuple[str, str]] = []
        self.rows_per_call = rows_per_call

    def _get(self, path, params=None, **kw):
        params = params or {}
        self.calls.append((params.get("startTime"), params.get("endTime")))
        return {"activities": [{"type": "Trades"}] * self.rows_per_call}


def _client(rec):
    c = qt.Questrade.__new__(qt.Questrade)
    c._get = rec._get
    return c


def test_the_window_constant_is_the_limit_the_api_actually_enforces():
    assert qt.ACTIVITY_WINDOW_DAYS == 30


def test_a_year_is_walked_in_windows_the_api_will_accept():
    """
    Verified against the live API: a 30-day window returns rows and 31
    returns 400. The documentation says 31; it is wrong.
    """
    rec = _Recorder()
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _client(rec).activities("X", start, end)

    assert len(rec.calls) >= 12
    for s, e in rec.calls:
        span = datetime.fromisoformat(e) - datetime.fromisoformat(s)
        # Hard-coded deliberately. Asserting against ACTIVITY_WINDOW_DAYS
        # would move with the constant and pass at 31, which the live API
        # rejects — the whole point is that the documented figure is wrong.
        assert span <= timedelta(days=30)


def test_the_windows_cover_the_year_without_overlapping():
    rec = _Recorder()
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _client(rec).activities("X", start, end)

    spans = [(datetime.fromisoformat(s), datetime.fromisoformat(e))
             for s, e in rec.calls]
    assert spans[0][0] == start
    assert spans[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        # Half-open: the next window starts after the previous one ends, so
        # an activity on the boundary is not collected twice.
        assert next_start > prev_end
        assert (next_start - prev_end) <= timedelta(seconds=2)


def test_every_window_contributes_its_rows():
    rec = _Recorder(rows_per_call=3)
    got = _client(rec).activities(
        "X", datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 4, 1, tzinfo=timezone.utc))
    assert len(got) == 3 * len(rec.calls)


# ── internal transfers ───────────────────────────────────────────────────────
def _acct(label, rows, registered=False):
    return {"id": label, "label": label, "registered": registered,
            "rows": [qa._row(r) for r in rows]}


def _move(kind, amount, day, ccy="USD"):
    return {"type": kind, "action": "CON", "netAmount": amount,
            "currency": ccy, "tradeDate": f"2025-{day}T00:00:00.000000-05:00",
            "settlementDate": f"2025-{day}T00:00:00.000000-05:00"}


def test_a_transfer_between_your_own_accounts_is_paired():
    margin = _acct("Margin", [_move("Withdrawals", -4000.0, "10-02")])
    tfsa = _acct("TFSA", [_move("Deposits", 4000.0, "10-02")], registered=True)
    pairs = qa._mark_internal([margin, tfsa])

    assert len(pairs) == 1
    assert pairs[0]["from"] == "Margin" and pairs[0]["to"] == "TFSA"
    assert margin["rows"][0]["internal"] is True
    assert tfsa["rows"][0]["internal"] is True


def test_legs_settling_a_day_apart_still_pair():
    margin = _acct("Margin", [_move("Withdrawals", -1194.80, "10-07")])
    rrsp = _acct("RRSP", [_move("Deposits", 1194.80, "10-09")], registered=True)
    assert len(qa._mark_internal([margin, rrsp])) == 1


def test_legs_a_month_apart_are_two_separate_events():
    margin = _acct("Margin", [_move("Withdrawals", -1000.0, "01-05")])
    tfsa = _acct("TFSA", [_move("Deposits", 1000.0, "03-05")], registered=True)
    assert qa._mark_internal([margin, tfsa]) == []
    assert tfsa["rows"][0]["internal"] is False


def test_different_currencies_never_pair():
    """Same number, different money. Pairing them would invent a transfer."""
    margin = _acct("Margin", [_move("Withdrawals", -1000.0, "10-02", "USD")])
    tfsa = _acct("TFSA", [_move("Deposits", 1000.0, "10-02", "CAD")],
                 registered=True)
    assert qa._mark_internal([margin, tfsa]) == []


def test_each_leg_is_consumed_once():
    """
    Two genuinely separate 1,000-dollar moves on one day, which is what the
    live account actually had. They must pair one-to-one, not collapse.
    """
    margin = _acct("Margin", [_move("Withdrawals", -1000.0, "01-30"),
                              _move("Withdrawals", -1000.0, "01-30")])
    tfsa = _acct("TFSA", [_move("Deposits", 1000.0, "01-30"),
                          _move("Deposits", 1000.0, "01-30")], registered=True)
    pairs = qa._mark_internal([margin, tfsa])
    assert len(pairs) == 2
    assert all(r["internal"] for r in margin["rows"] + tfsa["rows"])


def test_a_deposit_from_outside_is_not_marked_internal():
    """New money arriving is the thing this must not hide."""
    tfsa = _acct("TFSA", [_move("Deposits", 500.0, "06-01", "CAD")],
                 registered=True)
    margin = _acct("Margin", [])
    assert qa._mark_internal([margin, tfsa]) == []
    assert tfsa["rows"][0]["internal"] is False


def test_two_legs_in_the_same_account_do_not_pair():
    """An account cannot transfer to itself."""
    one = _acct("Margin", [_move("Withdrawals", -100.0, "05-01"),
                           _move("Deposits", 100.0, "05-01")])
    assert qa._mark_internal([one]) == []


# ── summaries ────────────────────────────────────────────────────────────────
def test_currencies_are_never_added_together():
    rows = [qa._row(_move("Deposits", 1000.0, "01-02", "CAD")),
            qa._row(_move("Deposits", 500.0, "01-03", "USD"))]
    got = qa._summarise(rows)
    deposits = got["by_type"][0]
    assert {c["currency"] for c in deposits["by_currency"]} == {"CAD", "USD"}
    assert [c["net"] for c in deposits["by_currency"]] == [1000.0, 500.0]


def test_the_summary_separates_new_money_from_shuffled_money():
    margin = _acct("Margin", [_move("Withdrawals", -4000.0, "10-02")])
    tfsa = _acct("TFSA", [_move("Deposits", 4000.0, "10-02"),
                          _move("Deposits", 800.0, "11-01")], registered=True)
    qa._mark_internal([margin, tfsa])

    deposits = qa._summarise(tfsa["rows"])["by_type"][0]
    usd = deposits["by_currency"][0]
    assert usd["net"] == 4800.0
    assert usd["internal_net"] == 4000.0
    assert usd["external_net"] == 800.0      # the only money that arrived


def test_types_come_back_in_statement_order():
    rows = [qa._row(_move("Deposits", 1.0, "01-02")),
            qa._row({"type": "Trades", "netAmount": 2.0, "currency": "USD",
                     "tradeDate": "2025-01-02T00:00:00.000000-05:00"})]
    order = [b["type"] for b in qa._summarise(rows)["by_type"]]
    assert order == ["Trades", "Deposits"]


def test_an_unfamiliar_type_is_kept_rather_than_dropped():
    """If Questrade adds a category, a row must not vanish from the ledger."""
    rows = [qa._row({"type": "Something New", "netAmount": 5.0,
                     "currency": "CAD",
                     "tradeDate": "2025-01-02T00:00:00.000000-05:00"})]
    got = qa._summarise(rows)
    assert got["count"] == 1
    assert got["by_type"][0]["type"] == "Something New"


# ── rows ─────────────────────────────────────────────────────────────────────
def test_the_row_date_is_the_trade_date_not_the_settlement_date():
    """
    Disposition date decides the tax year, and a trade on 31 December
    settling in January belongs to the year it was made in.
    """
    row = qa._row({"type": "Trades",
                   "tradeDate": "2025-12-31T00:00:00.000000-05:00",
                   "settlementDate": "2026-01-02T00:00:00.000000-05:00",
                   "netAmount": 10.0, "currency": "USD"})
    assert row["date"] == "2025-12-31"
    assert row["settled"] == "2026-01-02"


def test_a_missing_currency_does_not_become_an_empty_bucket():
    assert qa._row({"type": "Other"})["currency"] == "CAD"


@pytest.mark.parametrize("value", [None, "", "abc", float("nan")])
def test_unparseable_numbers_become_none_not_zero(value):
    """Zero is a real quantity; "no value" must not look like one."""
    assert qa._num(value) is None


# ── account classification ───────────────────────────────────────────────────
def test_the_registered_set_covers_every_sheltered_type_the_app_names():
    sheltered = {"TFSA", "RRSP", "SRRSP", "LRRSP", "LIRA", "LIF", "RIF",
                 "SRIF", "RESP", "FRESP", "FHSA"}
    assert sheltered <= qt.REGISTERED
    # The two that are not sheltered are the two that matter for a return.
    assert "Cash" not in qt.REGISTERED
    assert "Margin" not in qt.REGISTERED
    assert qt.REGISTERED <= set(qt.ACCOUNT_TYPES)


# ── searching one symbol across years ────────────────────────────────────────
def _ledger(rows_by_year):
    """Stand in for the cached per-year payloads the search is built on."""
    def fake(user_id, year):
        return {"year": year, "accounts": [
            {"id": "A1", "tail": "0001", "type": "TFSA", "label": "免税账户 TFSA",
             "registered": True, "rows": rows_by_year.get(year, []),
             "summary": {"count": 0, "by_type": []}}]}
    return fake


def _trade(date, symbol, qty, price, kind="Trades"):
    return qa._row({
        "type": kind, "action": "Buy" if qty > 0 else "Sell", "symbol": symbol,
        "quantity": qty, "price": price, "netAmount": -qty * price,
        "currency": "USD", "tradeDate": f"{date}T00:00:00.000000-05:00",
        "settlementDate": f"{date}T00:00:00.000000-05:00"})


def test_a_symbol_you_no_longer_hold_is_still_findable(monkeypatch):
    """
    The whole point. On a real account 27 symbols had been traded in three
    years and only 12 were still held, so a picker built from holdings would
    have hidden ALGM — bought and sold out completely.
    """
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-29", "ALGM", -130, 38.5)],
        2025: [_trade("2025-05-21", "ALGM", 130, 29.0)],
        2024: [],
    }))
    got = qa.traded_symbols(1, span=3)
    assert [s["symbol"] for s in got["symbols"]] == ["ALGM"]
    assert got["symbols"][0]["trades"] == 2


def test_the_search_spans_years_rather_than_one(monkeypatch):
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-29", "ALGM", -10, 38.5)],
        2025: [_trade("2025-05-21", "ALGM", 10, 29.0)],
    }))
    got = qa.symbol_activity(1, "ALGM", span=3)
    assert len(got["rows"]) == 2
    assert got["first"] == "2025-05-21" and got["last"] == "2026-01-29"


def test_rows_come_back_newest_first(monkeypatch):
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-29", "X", -10, 2.0),
               _trade("2026-03-01", "X", -10, 3.0)],
    }))
    dates = [r["date"] for r in qa.symbol_activity(1, "X", span=1)["rows"]]
    assert dates == sorted(dates, reverse=True)


def test_a_canadian_listing_is_not_the_same_symbol(monkeypatch):
    """
    META and META.TO are two listings of one company and were both traded on
    the real account. Collapsing them would merge two different books.
    """
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-05", "META", 1, 600.0),
               _trade("2026-01-06", "META.TO", 1, 820.0)],
    }))
    assert len(qa.symbol_activity(1, "META", span=1)["rows"]) == 1
    assert len(qa.symbol_activity(1, "META.TO", span=1)["rows"]) == 1


def test_the_search_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-05", "ALGM", 1, 30.0)]}))
    assert len(qa.symbol_activity(1, "algm", span=1)["rows"]) == 1


def test_an_empty_symbol_is_refused():
    with pytest.raises(LookupError, match="代码"):
        qa.symbol_activity(1, "  ", span=1)


# ── the buy/sell arithmetic ──────────────────────────────────────────────────
def test_position_is_quantity_weighted_per_currency(monkeypatch):
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-02", "X", 100, 10.0),
               _trade("2026-01-03", "X", 100, 20.0),
               _trade("2026-01-04", "X", -50, 30.0)]}))
    p = qa.symbol_activity(1, "X", span=1)["position"]["by_currency"][0]
    assert p["bought"] == 200 and p["avg_buy"] == pytest.approx(15.0)
    assert p["sold"] == 50 and p["avg_sell"] == pytest.approx(30.0)
    assert p["net"] == 150


def test_currencies_stay_apart_in_the_position_too():
    rows = [qa._row({"type": "Trades", "symbol": "X", "quantity": 10,
                     "price": 2.0, "currency": "USD",
                     "tradeDate": "2026-01-02T00:00:00.000000-05:00"}),
            qa._row({"type": "Trades", "symbol": "X", "quantity": 10,
                     "price": 3.0, "currency": "CAD",
                     "tradeDate": "2026-01-02T00:00:00.000000-05:00"})]
    got = qa._position(rows)["by_currency"]
    assert [c["currency"] for c in got] == ["CAD", "USD"]
    assert got[0]["avg_buy"] == 3.0 and got[1]["avg_buy"] == 2.0


def test_dividends_do_not_move_the_position(monkeypatch):
    """Only fills are a position; a dividend is cash."""
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-02", "X", 100, 10.0),
               _trade("2026-01-03", "X", 0, 0.0, kind="Dividends")]}))
    p = qa.symbol_activity(1, "X", span=1)["position"]["by_currency"][0]
    assert p["bought"] == 100 and p["net"] == 100


# ── the chart marks ──────────────────────────────────────────────────────────
def test_marks_carry_a_side_and_come_back_oldest_first(monkeypatch):
    """
    lightweight-charts asserts markers are time-ascending, so an unsorted
    list does not render wrong — it throws and takes the chart with it.
    """
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-03-01", "X", -10, 3.0),
               _trade("2026-01-02", "X", 10, 2.0)]}))
    marks = qa.trade_marks(1, "X", span=1)["marks"]
    assert [m["date"] for m in marks] == ["2026-01-02", "2026-03-01"]
    assert [m["side"] for m in marks] == ["buy", "sell"]
    assert marks[0]["quantity"] == 10      # absolute, not signed


def test_marks_leave_out_everything_that_is_not_a_fill(monkeypatch):
    monkeypatch.setattr(qa, "transactions", _ledger({
        2026: [_trade("2026-01-02", "X", 10, 2.0),
               _trade("2026-01-03", "X", 0, 0.0, kind="Dividends"),
               _trade("2026-01-04", "X", 0, 0.0, kind="Fees and rebates")]}))
    assert len(qa.trade_marks(1, "X", span=1)["marks"]) == 1
