"""
行业轨迹: one stock's rolling correlation with industry indexes, market removed.

Offline.

    python -m pytest api/tests/test_industry_track.py -q
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import industry_track as it  # noqa: E402

CAL = pd.bdate_range("2017-01-02", periods=1500)
HALF = 750
PROP, SEMI, MKT = "931775.CSI", "H30184.CSI", "000985.CSI"


@pytest.fixture(autouse=True)
def fake_markets(monkeypatch):
    mod = types.ModuleType("markets")
    mod.canonical = lambda s: s
    mod.split = lambda s: ("US", s.split(":")[-1]) if s.startswith("US:") else ("CN", s)
    monkeypatch.setitem(sys.modules, "markets", mod)


def _prices(rets: np.ndarray, index=CAL) -> pd.Series:
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=index)


@pytest.fixture
def book():
    """
    A stock that trades with property for the first half and with chips for
    the second, on top of the market. Both industries also carry the market,
    so the raw correlation with either stays high throughout.
    """
    rng = np.random.default_rng(7)
    n = len(CAL)
    m = rng.normal(0, 0.018, n)        # the market dominates, as it does
    f_prop = rng.normal(0, 0.012, n)
    f_semi = rng.normal(0, 0.012, n)
    first = np.arange(n) < HALF
    s = m + np.where(first, f_prop, f_semi) + rng.normal(0, 0.006, n)
    closes = {
        MKT: _prices(m),
        PROP: _prices(m + f_prop),
        SEMI: _prices(m + f_semi),
        "931009.CSI": _prices(m + rng.normal(0, 0.012, n)),
    }
    stock = {"002080": _prices(s)}
    return stock, closes


def run(book, *, inds=(PROP, SEMI), **kw):
    stock, closes = book

    def history(sym):
        return stock[sym]

    def industry(code):
        if code not in closes:
            raise RuntimeError(f"{code} 行情读取失败")
        return closes[code]

    return it.track("002080", list(inds), calendar=CAL, history=history,
                    industry=industry, **kw)


def at(out, key, code, pos):
    return out[key][code][pos]


def test_the_switch_shows_once_the_market_is_out(book):
    out = run(book, freq="W")
    early, late = HALF // 2 + 200, HALF + 600
    assert at(out, "ex", PROP, early) > 0.5 and at(out, "ex", SEMI, early) < 0.2
    assert at(out, "ex", SEMI, late) > 0.5 and at(out, "ex", PROP, late) < 0.2


def test_raw_correlation_is_high_with_both_whichever_leads(book):
    out = run(book, freq="W")
    for pos in (HALF // 2 + 200, HALF + 600):
        assert at(out, "raw", PROP, pos) > 0.35 and at(out, "raw", SEMI, pos) > 0.35
    assert out["market_raw"][HALF + 600] > 0.4


def test_every_series_is_on_the_window_calendar(book):
    out = run(book, freq="D")
    n = len(CAL)
    assert out["dates"][0] == str(CAL[0].date()) and len(out["dates"]) == n
    assert len(out["close"]) == n and len(out["market_raw"]) == n
    assert all(len(v) == n for v in out["ex"].values())
    assert out["span"] == 60 and out["unit"] == "日"
    # The first full span is the first value: nothing before it.
    assert out["ex"][PROP][59] is None and out["ex"][PROP][61] is not None


def test_a_weekly_value_is_held_until_the_next_week_closes(book):
    out = run(book, freq="W")
    s = out["ex"][PROP]
    week = CAL[400:405]                       # Monday to Friday
    assert len({s[i] for i in range(400, 404)}) == 1          # Mon–Thu: last Friday's
    assert week[-1].dayofweek == 4


def test_a_week_the_exchange_was_shut_is_not_a_gap(book):
    stock, closes = book
    shut = (CAL >= "2019-02-04") & (CAL <= "2019-02-08")      # a whole 春节 week
    cal = CAL[~shut]
    stock = {k: v[~shut] for k, v in stock.items()}
    closes = {k: v[~shut] for k, v in closes.items()}
    out = it.track("002080", [PROP], calendar=cal, history=lambda s: stock[s],
                   industry=lambda c: closes[c], freq="W")
    after = int(np.searchsorted(cal, pd.Timestamp("2019-03-01")))
    assert out["ex"][PROP][after] is not None


def test_partial_equals_correlating_the_residuals():
    rng = np.random.default_rng(3)
    m = rng.normal(size=500)
    s = 0.8 * m + rng.normal(size=500)
    x = 0.5 * m + 0.4 * s + rng.normal(size=500)

    def resid(y):
        b = np.polyfit(m, y, 1)
        return y - np.polyval(b, m)

    want = np.corrcoef(resid(s), resid(x))[0, 1]
    got = it.partial(np.corrcoef(s, x)[0, 1], np.corrcoef(s, m)[0, 1], np.corrcoef(x, m)[0, 1])
    assert got == pytest.approx(want, abs=1e-9)


def test_partial_is_undefined_when_one_side_is_the_market():
    assert np.isnan(it.partial(0.5, 1.0, 0.5))
    got = it.partial(pd.Series([0.5, 0.5]), pd.Series([1.0, 0.3]), pd.Series([0.5, 0.2]))
    assert np.isnan(got.iloc[0]) and np.isfinite(got.iloc[1])


def test_the_window_numbers_come_from_daily_returns_inside_it(book):
    a, b = str(CAL[100].date()), str(CAL[400].date())
    out = run(book, start=a, end=b, series=False)
    assert "dates" not in out and "ex" not in out
    w = out["window"]
    assert w["from"] == a and w["days"] == 300
    assert w["rows"][PROP]["ex"] > 0.5 and w["rows"][SEMI]["ex"] < 0.2
    assert w["rows"][SEMI]["raw"] > 0.3 and w["market_raw"] > 0.4

    stock, closes = book
    r = pd.DataFrame({"s": stock["002080"], "m": closes[MKT], "x": closes[PROP]}
                     ).loc[a:b].pct_change().iloc[1:]
    want = it.partial(r.s.corr(r.x), r.s.corr(r.m), r.x.corr(r.m))
    assert w["rows"][PROP]["ex"] == pytest.approx(want, abs=1e-3)


def test_a_window_too_short_has_no_numbers(book):
    out = run(book, start=str(CAL[100].date()), end=str(CAL[110].date()), series=False)
    assert out["window"]["rows"][PROP] == {"raw": None, "ex": None}


def test_nothing_before_the_stock_listed(book):
    stock, closes = book
    late = {"002080": stock["002080"].iloc[600:]}
    out = it.track("002080", [PROP], calendar=CAL, history=lambda s: late[s],
                   industry=lambda c: closes[c], freq="D")
    assert out["listed"] == str(CAL[600].date())
    assert out["close"][599] is None and out["close"][600] is not None
    assert all(v is None for v in out["ex"][PROP][:660])
    assert out["ex"][PROP][662] is not None


def test_membership_is_marked_with_its_weight(book):
    look = {PROP: {"002080": 1.85, "600048": 9.1}, SEMI: {"688981": 8.0}}
    out = run(book, membership=lambda c: look[c])
    assert out["member"] == {PROP: 1.85, SEMI: False}


def test_an_index_with_no_constituent_list_is_unknown_not_no(book):
    out = run(book, membership=lambda c: {})
    assert out["member"] == {PROP: None, SEMI: None}


def test_a_membership_lookup_that_fails_is_unknown_not_an_error(book):
    def boom(code):
        raise RuntimeError("no points")

    assert run(book, membership=boom)["member"] == {PROP: None, SEMI: None}


def test_an_industry_that_fails_to_load_is_left_out(book):
    out = run(book, inds=(PROP, "399986.SZ"))
    assert [i["code"] for i in out["industries"]] == [PROP]
    assert out["failed"] == ["399986.SZ"]


def test_the_market_proxy_failing_is_an_error(book):
    stock, closes = book
    with pytest.raises(RuntimeError, match="中证全指"):
        it.track("002080", [PROP], calendar=CAL, history=lambda s: stock[s],
                 industry=lambda c: (_ for _ in ()).throw(RuntimeError("down")) if c == MKT
                 else closes[c])


@pytest.mark.parametrize("inds,freq,msg", [
    (["XLK"], "W", "同一个市场"),
    (["000300.SH"], "W", "同一个市场"),
    ([], "W", "至少"),
    (list(it.INDUSTRIES["CN"])[:9], "W", "最多"),
    ([PROP], "M", "频率"),
])
def test_requests_that_cannot_be_answered(book, inds, freq, msg):
    with pytest.raises(LookupError, match=msg):
        run(book, inds=inds, freq=freq)


def test_a_us_stock_reads_against_us_sectors_and_the_sp500():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, len(CAL))
    closes = {"^GSPC": _prices(m), "SMH": _prices(m + rng.normal(0, 0.01, len(CAL)))}
    out = it.track("US:NVDA", ["SMH"], calendar=CAL,
                   history=lambda s: _prices(m + rng.normal(0, 0.01, len(CAL))),
                   industry=lambda c: closes[c])
    assert out["market_proxy"] == {"code": "^GSPC", "label": "标普500"}
    assert out["member"] == {"SMH": None}


def test_the_allow_lists_hold_together():
    assert set(it.INDUSTRIES) == set(it.MARKET_PROXY)
    for market, codes in it.INDUSTRIES.items():
        assert all(it.CODE_MARKET[c] == market for c in codes)
        assert it.MARKET_PROXY[market][0] not in codes
    with pytest.raises(LookupError):
        it.industry_history("000905.SH")
