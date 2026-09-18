"""
sector_rotation: the map has to say OUT OF what, INTO what.

Offline — no database, no Tushare. The fixture is four sectors whose relative
strength is a sine wave in log space, each a quarter-cycle out of phase. That
is the cleanest possible rotation: at any instant one sector is peaking, one is
rolling over, one is bottoming and one is turning up, so a correct RRG must put
exactly one in each quadrant and send them all round clockwise. If the maths is
wrong in any way that matters, this fixture cannot stay in four corners.

    python -m pytest api/tests/test_sector_rotation.py -q
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

import sector_rotation as sr  # noqa: E402

NAMES = ("A", "B", "C", "D")


def rotating(n: int = 400, period: int = 200, amp: float = 0.25):
    """Four sectors a quarter-cycle apart, against a flat benchmark."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    k = np.arange(n)
    sectors = {
        name: pd.Series(100 * np.exp(amp * np.sin(2 * np.pi * k / period + i * np.pi / 2)),
                        index=idx)
        for i, name in enumerate(NAMES)
    }
    return sectors, pd.Series(100.0, index=idx)


def drifting(slopes: dict[str, float], n: int = 400):
    """Sectors on constant relative drifts — all outperforming, by different amounts."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n) / n
    return ({name: pd.Series(100 * np.exp(c * t), index=idx) for name, c in slopes.items()},
            pd.Series(100.0, index=idx))


def by_name(result) -> dict:
    return {s["name"]: s for s in result["sectors"]}


# ── the whole point: direction ───────────────────────────────────────────────
def test_four_phases_of_a_rotation_land_in_four_different_quadrants():
    res = sr.analyse(*rotating(), freq="d")
    quads = sorted(s["quadrant"] for s in res["sectors"])
    assert quads == ["improving", "lagging", "leading", "weakening"]


def test_the_map_names_what_money_is_leaving_and_what_it_is_entering():
    """The question the old correlation panel could not answer at all."""
    res = sr.analyse(*rotating(), freq="d")
    calls = res["calls"]

    assert len(calls["into"]) == 1 and len(calls["outof"]) == 1
    into, outof = calls["into"][0]["name"], calls["outof"][0]["name"]

    # Arriving: still behind the market, but momentum already positive.
    assert by_name(res)[into]["ratio"] < 100 < by_name(res)[into]["mom"]
    # Leaving: still ahead on price, momentum already gone.
    assert by_name(res)[outof]["mom"] < 100 < by_name(res)[outof]["ratio"]
    assert into != outof


def test_sectors_travel_clockwise():
    """
    改善 → 领先 → 走弱 → 落后. Each sector's heading must point at the quadrant
    it is on its way to — that is what makes 走弱 a warning rather than a label.
    """
    res = sr.analyse(*rotating(), freq="d")
    # Bearings are clockwise from north: NE 0–90, SE 90–180, SW 180–270, NW 270–360.
    expect = {"improving": (0, 90), "leading": (90, 180),
              "weakening": (180, 270), "lagging": (270, 360)}
    for s in res["sectors"]:
        lo, hi = expect[s["quadrant"]]
        assert s["heading"] is not None, f"{s['name']} has no heading"
        assert lo <= s["heading"] <= hi, f"{s['name']} in {s['quadrant']} heading {s['heading']}"


def test_a_heading_survives_a_leg_too_small_to_see():
    """
    A slow-moving sector still has a direction. Taking the bearing from the
    two-decimal values sent to the client rounds a smooth leg to zero and the
    arrow vanishes on exactly the sectors that are hardest to read.
    """
    res = sr.analyse(*rotating(period=4000, amp=0.02), freq="d")
    # Every step here is far below the rounding granularity of the tail.
    spans = [max(abs(s["tail"][-1]["ratio"] - s["tail"][-2]["ratio"]),
                 abs(s["tail"][-1]["mom"] - s["tail"][-2]["mom"])) for s in res["sectors"]]
    assert max(spans) <= 0.02, "fixture is not slow enough to test the rounding"
    assert all(s["heading"] is not None for s in res["sectors"])


# ── why not correlation ──────────────────────────────────────────────────────
def test_correlation_cannot_distinguish_inflow_from_outflow():
    """
    Two sectors, both correlated ~1.0 with the market, one gaining on it and
    one losing to it. Correlation is identical; the rotation map is opposite.
    This is the defect the module exists to fix, so it is pinned as a test.
    """
    n = 400
    idx = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n) / n
    # One shared market shock, so every sector genuinely co-moves with the
    # benchmark; the only difference between them is a steady relative drift.
    shock = np.cumsum(np.random.default_rng(7).normal(0, 0.012, n))
    px = lambda drift: pd.Series(100 * np.exp(shock + drift * t), index=idx)

    bench, winner, loser = px(0.30), px(0.60), px(0.05)
    filler = {"C": px(0.30), "D": px(0.32)}

    lr = np.log(pd.DataFrame({"b": bench, "w": winner, "l": loser})).diff().dropna()
    assert lr["b"].corr(lr["w"]) == pytest.approx(1.0)
    assert lr["b"].corr(lr["l"]) == pytest.approx(1.0)

    res = sr.analyse({"W": winner, "L": loser, **filler}, bench, freq="d")
    assert by_name(res)["W"]["ratio"] > 100 > by_name(res)["L"]["ratio"]


# ── the normalisation ────────────────────────────────────────────────────────
def test_sectors_are_scored_against_each_other_not_against_their_own_history():
    """
    Every sector outperforming the benchmark cannot mean every sector is
    leading — rotation is zero-sum, and a map where the whole universe sits in
    one corner has told you nothing about where to put money.
    """
    res = sr.analyse(*drifting({"A": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}), freq="d")
    quads = {s["name"]: s["quadrant"] for s in res["sectors"]}
    assert len(set(quads.values())) > 1, quads
    assert quads["A"] != quads["D"]


def test_a_universe_that_moved_identically_produces_no_ranking():
    """No spread means no information — not a division by zero."""
    same = {n: s for n, s in zip(NAMES, [None] * 4)}
    idx = pd.bdate_range("2024-01-01", periods=400)
    t = np.arange(400) / 400
    for n in NAMES:
        same[n] = pd.Series(100 * np.exp(0.3 * t), index=idx)

    res = sr.analyse(same, pd.Series(100.0, index=idx), freq="d")
    assert all(s["ratio"] == 100.0 and s["mom"] == 100.0 for s in res["sectors"])
    assert all(s["neutral"] for s in res["sectors"])
    assert res["calls"]["into"] == [] and res["calls"]["outof"] == []


def test_a_sector_sitting_on_the_crossing_point_is_not_a_call():
    res = sr.analyse(*rotating(), freq="d")
    sectors = by_name(res)
    sectors["A"]["neutral"] = True
    assert "A" not in [c["name"] for c in sr._calls(list(sectors.values()))["into"]]
    assert "A" not in sr._calls(list(sectors.values()))["leading"]


# ── the measured edge ────────────────────────────────────────────────────────
def test_each_quadrant_reports_what_it_has_actually_been_worth():
    res = sr.analyse(*rotating(), freq="d")
    rows = {r["quadrant"]: r for r in res["edge"]["rows"]}

    assert all(r["n"] > 0 for r in rows.values())
    # On a clean rotation, being in 领先 must pay more than being in 落后.
    assert rows["leading"]["edge_pp"] > rows["lagging"]["edge_pp"]
    # And 改善 (arriving) must beat 走弱 (leaving) — that ordering IS the claim
    # the whole panel makes; if it inverts on clean data the maths is backwards.
    assert rows["improving"]["edge_pp"] > rows["weakening"]["edge_pp"]


def test_the_edge_is_measured_against_the_all_quadrant_baseline():
    """
    A universe that drifted up 3% does not make "改善 returned 3%" a finding.
    The edges are deviations from the pooled mean, so they must cancel out.
    """
    res = sr.analyse(*rotating(), freq="d")
    rows = res["edge"]["rows"]
    weighted = sum(r["edge_pp"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
    assert weighted == pytest.approx(0.0, abs=0.05)


# ── the absolute trend, as a second opinion ──────────────────────────────────
def test_the_absolute_trend_never_changes_the_ranking():
    """
    Relative and absolute answer different questions. A sector can be gaining
    on a falling market while still falling itself — that is a real and
    common case, and folding the two into one score would bury it.
    """
    sectors, bench = rotating()
    plain = sr.analyse(sectors, bench, freq="d")

    # An absolute trend that falls while relative strength rises.
    dates = list(sectors["A"].index)
    trend = pd.DataFrame({n: np.linspace(0.8, 0.2, len(dates)) for n in NAMES}, index=dates)
    with_t = sr.analyse(sectors, bench, freq="d", trend=trend)

    assert [c["name"] for c in with_t["calls"]["into"]] == \
           [c["name"] for c in plain["calls"]["into"]]
    assert with_t["calls"]["into"][0]["rising"] is False
    assert with_t["calls"]["into"][0]["trend"]["delta_pp"] < 0


def test_a_sector_with_no_trend_column_reports_unknown_not_false():
    sectors, bench = rotating()
    dates = list(sectors["A"].index)
    trend = pd.DataFrame({"A": np.linspace(0.2, 0.8, len(dates))}, index=dates)
    res = sr.analyse(sectors, bench, freq="d", trend=trend)

    got = by_name(res)
    assert got["A"]["trend"] is not None
    assert got["B"]["trend"] is None
    assert all(c["rising"] is None for c in res["calls"]["into"] if c["name"] != "A")


def test_the_trend_delta_is_measured_over_the_maps_own_horizon():
    """Weekly and daily look back the same number of TRADING days, not bars."""
    sectors, bench = rotating(n=600)
    dates = list(sectors["A"].index)
    # A ramp, so the delta is proportional to how far back it looks.
    trend = pd.DataFrame({n: np.linspace(0.0, 1.0, len(dates)) for n in NAMES}, index=dates)

    week = by_name(sr.analyse(sectors, bench, freq="w", trend=trend))["A"]["trend"]
    day = by_name(sr.analyse(sectors, bench, freq="d", trend=trend))["A"]["trend"]
    assert week["delta_pp"] == pytest.approx(day["delta_pp"], abs=0.2)


# ── frequency ────────────────────────────────────────────────────────────────
def test_weekly_reads_the_weeks_last_print_not_its_average():
    """
    An RRG reads levels. The average of a week's relative strength is not any
    week's relative strength, and a mid-week spike would move it.
    """
    sectors, bench = rotating(n=600)
    spiked = {n: s.copy() for n, s in sectors.items()}
    # ONE Wednesday, inside the window the map draws — never a weekly close.
    # It has to be a single day: scaling every Wednesday scales the weekly
    # mean uniformly, and a uniform scale cancels in rel / SMA(rel), so the
    # difference between "last" and "mean" would be invisible.
    wed = [d for d in spiked["A"].index if d.weekday() == 2][-3]
    spiked["A"].loc[wed] *= 1.5

    clean = sr.analyse(sectors, bench, freq="w")
    noisy = sr.analyse(spiked, bench, freq="w")
    assert by_name(noisy)["A"]["ratio"] == by_name(clean)["A"]["ratio"]


def test_a_weekly_bar_is_dated_by_the_session_it_ends_on():
    """
    resample("W-FRI") stamps each bin with the coming Friday. Read on a
    Thursday that makes the map's "as of" a date the market has not traded yet.
    """
    sectors, bench = rotating(n=600)
    thursday = [d for d in sectors["A"].index if d.weekday() == 3][-1]
    truncated = {n: s.loc[:thursday] for n, s in sectors.items()}

    res = sr.analyse(truncated, bench.loc[:thursday], freq="w")
    assert res["asof"] == thursday.strftime("%Y-%m-%d")
    assert res["dates"][-1] == res["asof"]


def test_weekly_and_daily_both_run_and_report_their_own_parameters():
    sectors, bench = rotating(n=600)
    for freq in ("d", "w"):
        res = sr.analyse(sectors, bench, freq=freq)
        assert res["freq"] == freq
        assert res["params"] == {k: sr.PRESETS[freq][k] for k in res["params"]}
        assert len(res["dates"]) == sr.PRESETS[freq]["tail"]
        assert all(len(s["tail"]) == sr.PRESETS[freq]["tail"] for s in res["sectors"])


# ── refusals ─────────────────────────────────────────────────────────────────
def test_too_few_sectors_is_refused_rather_than_ranked():
    sectors, bench = rotating()
    with pytest.raises(LookupError, match="横截面"):
        sr.analyse({k: sectors[k] for k in ("A", "B")}, bench, freq="d")


def test_too_little_history_is_refused():
    sectors, bench = rotating(n=80)
    with pytest.raises(LookupError, match="历史数据不足"):
        sr.analyse(sectors, bench, freq="d")


def test_an_unknown_frequency_is_refused():
    sectors, bench = rotating()
    with pytest.raises(LookupError):
        sr.analyse(sectors, bench, freq="m")


def test_a_sector_whose_history_is_too_short_is_dropped_not_zero_filled():
    """A mostly-empty column would drag the cross-sectional mean it joins."""
    sectors, bench = rotating()
    sectors["E"] = sectors["A"].tail(30)          # listed last month
    res = sr.analyse(sectors, bench, freq="d")
    assert "E" not in by_name(res)
    assert len(res["sectors"]) == 4


# ── the benchmark ────────────────────────────────────────────────────────────
def proxy_dm(monkeypatch, last: str, rows: int = 400):
    """A data_manager whose PPI_MARKET_PROXY ends on `last`."""
    idx = pd.bdate_range(end=last, periods=rows)
    frame = pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"),
                          "Close": np.linspace(100, 120, rows)})
    db = types.SimpleNamespace(
        table_exists=lambda t: t == "PPI_MARKET_PROXY",
        read_table=lambda t, columns=None, order_by=None: frame.copy(),
    )
    mod = types.ModuleType("data_manager")
    mod.db = db
    monkeypatch.setitem(sys.modules, "data_manager", mod)


def test_a_current_market_proxy_is_used_as_the_benchmark():
    sectors, _ = rotating()
    newest = sectors["A"].index.max().strftime("%Y-%m-%d")
    with pytest.MonkeyPatch.context() as mp:
        proxy_dm(mp, last=newest)
        assert sr._proxy_benchmark(dict(sectors)) is not None


def test_a_stale_market_proxy_is_refused_rather_than_divided_by():
    """
    A denominator seven months behind its numerators does not fail loudly — it
    turns every sector into a rising relative line and reports the entire
    market as 领先. That is the worst kind of wrong: plausible.
    """
    sectors, _ = rotating()
    stale = (sectors["A"].index.max() - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    with pytest.MonkeyPatch.context() as mp:
        proxy_dm(mp, last=stale)
        assert sr._proxy_benchmark(dict(sectors)) is None


def test_a_proxy_with_almost_no_history_is_refused():
    sectors, _ = rotating()
    newest = sectors["A"].index.max().strftime("%Y-%m-%d")
    with pytest.MonkeyPatch.context() as mp:
        proxy_dm(mp, last=newest, rows=40)
        assert sr._proxy_benchmark(dict(sectors)) is None
