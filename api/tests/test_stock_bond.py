"""stock_bond — the stock/bond ratio, its legs, and how far the next turn is."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import stock_bond as sb  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", periods=400)


def path(*moves):
    """A price path from (sessions, total move) pieces, starting at 100."""
    out, v = [], 100.0
    for n, total in moves:
        step = (1 + total) ** (1 / n)
        for _ in range(n):
            v *= step
            out.append(v)
    return pd.Series(out, index=DAYS[:len(out)])


def test_ratio_rebases_both_and_divides():
    e = pd.Series([10.0, 11.0, 12.0] * 30, index=DAYS[:90])
    b = pd.Series([50.0, 50.0, 60.0] * 30, index=DAYS[:90])
    d = sb.ratio(e, b)
    assert d["e"].iloc[0] == d["b"].iloc[0] == d["r"].iloc[0] == 100
    assert d["r"].iloc[1] == pytest.approx(110.0)       # stocks +10%, bonds flat
    assert d["r"].iloc[2] == pytest.approx(100.0)       # both +20%


def test_legs_alternate_and_name_the_side_that_won():
    # Stocks +40% while bonds are flat, then stocks −30% while bonds +10%.
    e = path((150, 0.40), (150, -0.30), (100, 0.0))
    b = path((150, 0.00), (150, 0.10), (100, 0.0))
    d = sb.build(e, b, pct=0.10)
    sides = [x["side"] for x in d["legs"]]
    assert sides[:2] == ["equity", "bond"]
    up, down = d["legs"][0], d["legs"][1]
    assert up["equity_pct"] == pytest.approx(40.0, abs=0.5)
    assert down["bond_pct"] == pytest.approx(10.0, abs=0.5)
    # A leg's start is known when the leg before it is confirmed over, and
    # that is after the turn itself.
    assert down["start_confirmed"] == up["end_confirmed"]
    assert down["start_confirmed"] > down["from"]
    json.dumps(d, allow_nan=False)


def test_the_open_leg_says_how_far_the_next_turn_is():
    e = path((200, 0.50), (20, -0.04))                    # up, then a small dip
    b = path((220, 0.0))
    d = sb.build(e, b, pct=0.10)
    now = d["now"]
    assert now["side"] == "equity" and d["legs"][-1]["open"] is True
    # Down about 4% from the peak: a further fall to 90% of the peak confirms.
    assert now["from_extreme"] == pytest.approx(-0.04, abs=0.002)
    assert now["need"] == pytest.approx(0.90 / 0.96 - 1, abs=0.002)


def test_a_bond_leg_needs_a_rise_to_turn():
    e = path((100, 0.30), (120, -0.30))
    b = path((220, 0.0))
    now = sb.build(e, b, pct=0.10)["now"]
    assert now["side"] == "bond" and now["need"] > 0


def test_moving_together_is_a_positive_correlation():
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 300)
    e = pd.Series(100 * np.cumprod(1 + common + rng.normal(0, 0.002, 300)), index=DAYS[:300])
    b = pd.Series(100 * np.cumprod(1 + common + rng.normal(0, 0.002, 300)), index=DAYS[:300])
    assert sb.build(e, b)["now"]["corr"] > 0.8
    seesaw = pd.Series(100 * np.cumprod(1 - common + rng.normal(0, 0.002, 300)), index=DAYS[:300])
    assert sb.build(e, seesaw)["now"]["corr"] < -0.8


def test_too_little_shared_history_is_an_error_not_a_chart():
    with pytest.raises(LookupError):
        sb.ratio(path((30, 0.1)), path((30, 0.0)))
