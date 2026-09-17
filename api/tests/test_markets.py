"""
markets: symbol parsing and per-market conventions.

Offline — no network, no database. Symbol parsing gates every route, and the
conventions decide what colour a candle is and which bars a percentile window
may see, so both are worth pinning down.

    python -m pytest api/tests/test_markets.py -q
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import markets  # noqa: E402


# ── symbols ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("given, market, code", [
    ("600519", "CN", "600519"),
    ("000001", "CN", "000001"),
    ("CN:600519", "CN", "600519"),
    ("US:AAPL", "US", "AAPL"),
    ("us:aapl", "US", "AAPL"),          # case-insensitive, upper-cased
    ("CA:SHOP.TO", "CA", "SHOP.TO"),
    ("US:BRK.B", "US", "BRK.B"),        # share classes survive
    ("CA:RY-PS.TO", "CA", "RY-PS.TO"),  # so do hyphens
    ("  US:MSFT  ", "US", "MSFT"),
])
def test_symbols_parse_to_the_right_market(given, market, code):
    assert markets.split(given) == (market, code)


def test_a_shares_stay_bare_and_everything_else_is_prefixed():
    """
    The canonical form is what lands in search_history, watchlists and URLs.
    A-shares must stay unprefixed or every row already stored — and the whole
    Streamlit app — would need migrating.
    """
    assert markets.canonical("600519") == "600519"
    assert markets.canonical("CN:600519") == "600519"
    assert markets.canonical("us:aapl") == "US:AAPL"
    assert markets.canonical("CA:shop.to") == "CA:SHOP.TO"


@pytest.mark.parametrize("bad", [
    "", "   ", "bad!!", "US:", ":AAPL", "US::AAPL", "60051", "6005190",
    "JP:7203", "US:" + "A" * 20, "../etc/passwd", "US:AAPL;DROP",
])
def test_malformed_symbols_are_refused(bad):
    with pytest.raises(LookupError):
        markets.split(bad)


def test_a_bare_non_a_share_ticker_says_what_to_do():
    """
    'AAPL' is a plausible thing to type. The error has to name the fix, not
    just refuse — this is the message a user actually sees.
    """
    with pytest.raises(LookupError, match="US:AAPL"):
        markets.split("AAPL")


def test_six_digits_is_never_mistaken_for_a_us_ticker():
    """A bare six-digit code is an A-share, always. Nothing else may claim it."""
    assert markets.split("600519")[0] == "CN"
    assert markets.split("000001")[0] == "CN"


# ── conventions ──────────────────────────────────────────────────────────────
def test_colour_convention_is_per_market():
    """Red is up in Shanghai and down in New York; one global constant is wrong."""
    assert markets.get("CN").conv.up_is_red is True
    assert markets.get("US").conv.up_is_red is False
    assert markets.get("CA").conv.up_is_red is False


def test_each_market_has_its_own_currency_and_benchmark():
    got = {c: (markets.get(c).conv.currency, markets.get(c).conv.benchmark)
           for c in markets.CODES}
    assert got == {
        "CN": ("CNY", "000300.SH"),
        "US": ("USD", "^GSPC"),
        "CA": ("CAD", "^GSPTSE"),
    }
    # Three distinct symbols — beta against the wrong index is meaningless,
    # not merely less accurate.
    assert len({markets.get(c).conv.benchmark for c in markets.CODES}) == 3


def test_north_america_has_no_regime_break():
    """
    The 924 anchor is an A-share fact. Applied to a US stock it silently drops
    every bar before 2024-10-14 from the percentile baselines — measured at
    270 of AAPL's 752 sessions.
    """
    from analysis_engine import REGIME_ANCHOR, _regime_anchor_pos

    idx = pd.bdate_range("2023-09-18", periods=752)
    cn_anchor = markets.get("CN").conv.regime_anchor
    na_anchor = markets.get("US").conv.regime_anchor

    assert cn_anchor == REGIME_ANCHOR
    assert na_anchor == markets.NO_REGIME_BREAK
    assert _regime_anchor_pos(idx, na_anchor) == 0
    assert _regime_anchor_pos(idx, cn_anchor) > 200


def test_the_default_anchor_is_still_the_a_share_one():
    """Every existing caller passes nothing and must keep the 924 behaviour."""
    from analysis_engine import _regime_anchor_pos

    idx = pd.bdate_range("2023-09-18", periods=752)
    assert _regime_anchor_pos(idx) == _regime_anchor_pos(
        idx, markets.get("CN").conv.regime_anchor)


def test_chart_window_follows_the_market():
    """
    The display window is anchored too: an A-share chart opens at the 924
    pivot, a US chart has no such date and must show its whole history.
    """
    import ta_payload

    idx = pd.bdate_range("2023-09-18", periods=752)
    df = pd.DataFrame({"Close": range(len(idx))}, index=idx)

    cn = ta_payload.chart_window(df, markets.get("CN").conv.regime_anchor)
    na = ta_payload.chart_window(df, markets.get("US").conv.regime_anchor)

    assert len(na) == len(df)
    assert len(cn) < len(df)
    assert cn.index[0] >= markets.get("CN").conv.regime_anchor
    # Default must remain the A-share behaviour.
    assert len(ta_payload.chart_window(df)) == len(cn)


def test_get_accepts_a_market_code_or_a_symbol():
    assert markets.get("US") is markets.get("US:AAPL")
    assert markets.get("CN") is markets.get("600519")
    assert markets.parse("CA:SHOP.TO")[0] is markets.get("CA")


def test_adapters_are_cached_not_rebuilt():
    """They hold a lock and a session; a fresh one per request would defeat both."""
    assert markets.get("US") is markets.get("US")
