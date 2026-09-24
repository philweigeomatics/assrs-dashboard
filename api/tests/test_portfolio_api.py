"""
The fund-management side: rebalancing, and the drift history behind it.

Every check the browser performs is repeated on the server, and these pin
that down — a disabled button is a courtesy to the user, not a control on
the request.
"""

from __future__ import annotations

import os
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))




# ── rebalance: the server redoes every check the browser does ────────────────
def _stub_rebalance(monkeypatch, *, owned=True, names=("600519",)):
    import types as _t

    import api.portfolio_api as pa

    monkeypatch.setattr(pa, "_own",
                        (lambda u, f: {"id": f}) if owned
                        else _raiser(LookupError("找不到这个组合")))
    dm = _t.ModuleType("data_manager")
    dm.get_tushare_ticker = lambda t: t if "." in t else f"{t}.SH"
    dm.get_stock_name_from_db = lambda c: (
        "名字" if c.split(".")[0] in names else None)
    seen = {}
    dm.execute_fund_rebalance = lambda fid, pos: (
        seen.update(fund=fid, pos=pos) or (True, "已调仓"))
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    return seen


def _raiser(exc):
    def go(*a, **k):
        raise exc
    return go


def test_rebalance_refuses_weights_that_do_not_sum_to_100(monkeypatch):
    import api.portfolio_api as pa
    _stub_rebalance(monkeypatch)
    with pytest.raises(LookupError, match="100%"):
        pa.rebalance(1, 7, [{"t": "600519", "weight_pct": 94.0}])


def test_rebalance_refuses_an_unknown_ticker(monkeypatch):
    import api.portfolio_api as pa
    _stub_rebalance(monkeypatch, names=("600519",))
    with pytest.raises(LookupError, match="找不到"):
        pa.rebalance(1, 7, [{"t": "600519", "weight_pct": 50.0},
                            {"t": "999999", "weight_pct": 50.0}])


def test_rebalance_refuses_a_fund_that_is_not_yours(monkeypatch):
    """The browser hides the tab; that is not what stops the request."""
    import api.portfolio_api as pa
    _stub_rebalance(monkeypatch, owned=False)
    with pytest.raises(LookupError, match="找不到这个组合"):
        pa.rebalance(2, 7, [{"t": "600519", "weight_pct": 100.0}])


def test_rebalance_writes_fractions_not_percents(monkeypatch):
    import api.portfolio_api as pa
    seen = _stub_rebalance(monkeypatch, names=("600519", "000001"))
    out = pa.rebalance(1, 7, [{"t": "600519", "weight_pct": 60.0},
                              {"t": "000001", "weight_pct": 40.0}])
    assert out["ok"] and out["holdings"] == 2
    assert seen["pos"] == {"600519.SH": 0.6, "000001.SH": 0.4}


def test_rebalance_drops_zero_weights_rather_than_writing_them(monkeypatch):
    seen = _stub_rebalance(monkeypatch, names=("600519", "000001"))
    import api.portfolio_api as pa
    pa.rebalance(1, 7, [{"t": "600519", "weight_pct": 100.0},
                        {"t": "000001", "weight_pct": 0.0}])
    assert list(seen["pos"]) == ["600519.SH"]


# ── drift history ────────────────────────────────────────────────────────────
def test_drift_history_never_mixes_real_with_simulated(monkeypatch):
    """
    Simulated rows are a retroactive "what this would have done". Drawing
    them continuous with measured history invents a track record.
    """
    import api.portfolio_api as pa

    rows = pd.DataFrame([
        {"fund_id": 1, "trade_date": "2026-01-02", "ts_code": "600519.SH",
         "target_weight": 0.5, "actual_weight": 0.55, "is_simulated": 1},
        {"fund_id": 1, "trade_date": "2026-05-04", "ts_code": "600519.SH",
         "target_weight": 0.5, "actual_weight": 0.58, "is_simulated": 0},
    ])
    monkeypatch.setattr(pa, "_db", lambda: types.SimpleNamespace(
        read_table=lambda *a, **k: rows))
    monkeypatch.setattr(pa, "_cn_names", lambda: {"600519": "贵州茅台"})

    out = pa._drift_series(1)
    assert out["real"]["dates"] == ["2026-05-04"]
    assert out["simulated"]["dates"] == ["2026-01-02"]
    assert out["real"]["holdings"][0]["drift_pp"] == [pytest.approx(8.0)]
    assert out["real"]["holdings"][0]["n"] == "贵州茅台"


def test_drift_history_is_empty_not_a_crash_without_rows(monkeypatch):
    import api.portfolio_api as pa
    monkeypatch.setattr(pa, "_db", lambda: types.SimpleNamespace(
        read_table=lambda *a, **k: None))
    assert pa._drift_series(1) == {"real": None, "simulated": None}


@pytest.mark.parametrize("stored, want", [
    (0.55, 55.0),      # fraction, as the rollup writes it
    (55.0, 55.0),      # a few rows arrived as percents
    (None, None),
])
def test_weights_are_read_in_whichever_unit_they_were_stored(stored, want):
    import api.portfolio_api as pa
    got = pa._pct(stored)
    assert (got is None) if want is None else got == pytest.approx(want)
