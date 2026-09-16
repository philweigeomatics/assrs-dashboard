"""
The stock list must never cache a failure.

data_manager.get_all_stock_basic() catches its own errors and returns [], so a
momentary database problem is indistinguishable from "no stocks exist". The
route used to hand that [] to a 6-hour cache, and every search box in the app
went silently empty for six hours — with a 200 OK, so nothing looked wrong.

    python -m pytest api/tests/test_stocks_cache.py -q
"""

from __future__ import annotations

import os
import sys
import types

import pytest
from fastapi import HTTPException

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


@pytest.fixture()
def route(monkeypatch):
    """The real /stocks route function with a stub data_manager behind it."""
    from api import main

    main._stocks_cache._d.clear()

    calls = {"n": 0}
    rows: list[dict] = []

    stub = types.ModuleType("data_manager")
    stub.get_all_stock_basic = lambda: (calls.__setitem__("n", calls["n"] + 1), rows)[1]
    monkeypatch.setitem(sys.modules, "data_manager", stub)

    def set_rows(new):
        rows[:] = new

    user = types.SimpleNamespace(id=1, username="t", email="t@x", role="admin")
    return types.SimpleNamespace(call=lambda: main.stocks(user), calls=calls, set_rows=set_rows)


def test_empty_result_is_not_cached(route):
    """An empty list is a failure: 503, and the next request retries."""
    for _ in range(3):
        with pytest.raises(HTTPException) as exc:
            route.call()
        assert exc.value.status_code == 503

    assert route.calls["n"] == 3, "a failed lookup must not be served from cache"

    # The database comes back; the very next request must see it.
    route.set_rows([{"ticker": "000001", "name": "平安银行"}])
    assert route.call() == [{"t": "000001", "n": "平安银行"}]


def test_real_result_is_cached(route):
    """The cache still does its job when there is something to cache."""
    route.set_rows([{"ticker": "600519", "name": "贵州茅台"}])
    first = route.call()
    for _ in range(4):
        assert route.call() == first
    assert route.calls["n"] == 1
