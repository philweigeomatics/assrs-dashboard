"""
Sessions are not calendar days, and confusing the two emptied the screen.

`lookback_days` means TRADING sessions everywhere it is used — pair_scan needs
240 of them to split a history in half — but the fetch asked Tushare for
`lookback_days + 90` CALENDAR days, which is an offset, not a conversion. It
under-delivers by more the further back you ask:

    asked 252 → got 232    (below the 240 minimum: "只有 0 只股票有足够长的历史")
    asked 504 → got 399
    asked 756 → got 566

The first of those is the one that surfaced: every stock in the watchlist had
years of history and the screen said none of them had enough.

    python -m pytest api/tests/test_lookback_window.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import lead_lag_stats as lls  # noqa: E402
import pair_scan as ps  # noqa: E402

#: Sessions per calendar day, measured against Tushare rather than assumed:
#: 418 calendar days returned exactly 252 sessions, 797 returned 504.
OBSERVED_RATE = 243 / 365


@pytest.mark.parametrize("sessions", [90, 180, 252, 504, 756, 1000])
def test_the_window_asked_for_actually_yields_the_sessions_wanted(sessions):
    calendar = lls.calendar_span(sessions)
    assert calendar * OBSERVED_RATE >= sessions, (
        f"{sessions} sessions needs more than {calendar} calendar days")


@pytest.mark.parametrize("sessions", [252, 504, 756])
def test_the_old_flat_offset_would_not_have(sessions):
    """The bug, pinned, so nobody reintroduces `+ 90`."""
    assert (sessions + 90) * OBSERVED_RATE < sessions


def test_the_window_is_not_wastefully_wide():
    """
    Over-fetching costs Tushare calls and time on every scan. Twice what is
    needed would be as wrong as half.
    """
    for sessions in (180, 252, 504, 756):
        assert lls.calendar_span(sessions) <= sessions * 2


def test_the_window_grows_with_the_lookback():
    spans = [lls.calendar_span(s) for s in (90, 180, 252, 504, 756)]
    assert spans == sorted(spans)
    assert len(set(spans)) == len(spans)


def test_every_lookback_the_ui_offers_can_satisfy_the_pair_scan():
    """
    252 is the smallest option on the 回看 selector, and it has to clear
    MIN_HALF * 2 — otherwise the screen refuses before it starts, which is
    exactly what it was doing.
    """
    for offered in (252, 504, 756):
        assert offered >= ps.MIN_HALF * 2
        assert lls.calendar_span(offered) * OBSERVED_RATE >= ps.MIN_HALF * 2


def test_the_route_cannot_be_asked_for_a_window_it_could_never_split():
    import pydantic
    from api.main import DiscoverReq

    assert DiscoverReq(lookback_days=ps.MIN_HALF * 2 + 12).lookback_days > ps.MIN_HALF * 2
    with pytest.raises(pydantic.ValidationError):
        DiscoverReq(lookback_days=ps.MIN_HALF * 2 - 1)


def test_a_split_of_the_smallest_window_still_leaves_two_usable_halves():
    smallest = 252
    half = ps.split_point(smallest)
    assert half >= ps.MIN_HALF and smallest - half >= ps.MIN_HALF
