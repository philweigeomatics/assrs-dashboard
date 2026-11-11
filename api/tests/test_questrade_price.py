"""
The price series the holdings panel draws your own fills on.

It must be the RAW close. A dividend-adjusted series sits below every old
fill by whatever has been paid out since, and on a monthly payer like QMAX
that is several percent a year — every buy would look worse than it was.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api import questrade_api as qa  # noqa: E402


def _frame():
    idx = pd.DatetimeIndex(["2023-12-29", "2024-01-02", "2024-01-30", "2024-01-31",
                            "2024-06-10"], tz="America/Toronto")
    return pd.DataFrame({
        "Close": [9.9, 10.0, 10.4, 10.2, 5.1],
        "Adj Close": [9.0, 9.1, 9.5, 9.4, 5.0],
        "Dividends": [0.0, 0.0, 0.0, 0.15, 0.0],
        "Stock Splits": [0.0, 0.0, 0.0, 0.0, 2.0],
    }, index=idx)


def test_uses_the_raw_close_from_the_window_start():
    out = qa._shape_history(_frame(), "2024-01-01")
    assert out["closes"][0] == ["2024-01-02", 10.0]
    assert [c for _, c in out["closes"]] == [10.0, 10.4, 10.2, 5.1]


def test_dividends_are_keyed_by_ex_date():
    out = qa._shape_history(_frame(), "2024-01-01")
    assert out["dividends"] == [{"ex_date": "2024-01-31", "amount": 0.15}]


def test_splits_come_along():
    out = qa._shape_history(_frame(), "2024-01-01")
    assert out["splits"] == [{"date": "2024-06-10", "ratio": 2.0}]


def test_empty_history_is_empty_not_an_error():
    out = qa._shape_history(pd.DataFrame(), "2024-01-01")
    assert out["closes"] == [] and out["dividends"] == []


def test_window_starts_on_the_first_of_january_of_the_oldest_year():
    assert qa._window_start(3).endswith("-01-01")
    assert int(qa._window_start(3)[:4]) == qa._years(span=3)[-1]
