"""
The Granger call has to survive the statsmodels version it is run against.

WHAT HAPPENED. `grangercausalitytests(..., verbose=False)` was deprecated in
statsmodels 0.14 and removed in 0.15:

    0.14.5   def grangercausalitytests(x, maxlag, addconst=True, verbose=None)
    0.15.0   def grangercausalitytests(x, maxlag, addconst=True)

requirements.txt pinned nothing, a rebuild picked up 0.15, and passing
`verbose` became a TypeError. `_granger` swallows every exception and returns
nan, so `lead_lag_test` returned None for all 79 peers and the screen said
"no lead-lag relationships" — a dependency break rendered as a finding, with
nothing in the output to distinguish it.

Local tests all passed the whole time, because locally statsmodels was 0.14.

    python -m pytest api/tests/test_granger_api.py -q
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pair_scan as ps  # noqa: E402


def series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = np.r_[0, 0, x[:-2]] * 0.8 + rng.normal(0, 0.3, n)
    return y, x


# ── the call must match whatever version is installed ────────────────────────
def test_the_call_passes_only_arguments_this_statsmodels_accepts():
    """
    The guard that would have caught it, against the installed version rather
    than a remembered one.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    accepted = set(inspect.signature(grangercausalitytests).parameters)
    # What _granger actually passes, positionally and by keyword.
    used = {"x", "maxlag"}
    assert used <= accepted, f"this statsmodels takes {sorted(accepted)}"


def test_verbose_is_not_passed_at_all():
    """
    Not "passed conditionally" — not passed. It does nothing in 0.14 but warn
    and is a TypeError in 0.15, so there is no version where sending it helps.
    """
    # Parsed, not string-matched. Splitting on the first ")" lands inside
    # column_stack([y, x]) and never reaches the keyword — which is how the
    # first version of this test passed while the bug was present — and
    # stripping the docstring by text fails when git checks the file out
    # with CRLF. The call node is the only thing that actually settles it.
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(ps._granger)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "grangercausalitytests"]
    assert calls, "the Granger call was renamed or removed"
    passed = {k.arg for c in calls for k in c.keywords}
    assert "verbose" not in passed, f"verbose is being passed again: {passed}"


def test_a_real_lead_is_still_found():
    """The fix must not have quietly broken the thing it was fixing."""
    y, x = series()
    p, lag = ps._granger(y, x, 5)
    assert not np.isnan(p)
    assert p < 0.01 and lag == 2


# ── a removed keyword must not read as "nothing found" ───────────────────────
def test_a_statsmodels_that_rejects_verbose_still_works(monkeypatch):
    """
    Simulate 0.15 exactly: a function whose signature has no `verbose`. If
    _granger ever passes it again, this fails here instead of in production.
    """
    import statsmodels.tsa.stattools as st

    real = st.grangercausalitytests

    def strict(x, maxlag, addconst=True):       # 0.15's signature, verbatim
        return real(x, maxlag, addconst)

    monkeypatch.setattr(st, "grangercausalitytests", strict)
    y, x = series()
    p, lag = ps._granger(y, x, 5)
    assert not np.isnan(p), "the 0.15 signature broke the call"
    assert p < 0.01 and lag == 2


def test_a_broken_call_is_recorded_rather_than_reported_as_no_result(monkeypatch):
    """
    The deeper bug. `except Exception: return nan` turned a TypeError into
    "this pair has no relationship", and 79 of those turned into "your
    watchlist has no lead-lag pairs". The failure now leaves a trace.
    """
    import statsmodels.tsa.stattools as st

    def exploding(*a, **kw):
        raise TypeError("grangercausalitytests() got an unexpected "
                        "keyword argument 'verbose'")

    monkeypatch.setattr(st, "grangercausalitytests", exploding)
    ps._last_failure = None
    p, _lag = ps._granger(*series(), 5)

    assert np.isnan(p)
    assert ps._last_failure and "TypeError" in ps._last_failure
    assert "verbose" in ps._last_failure


def test_the_scan_says_why_it_tested_nothing(monkeypatch):
    """
    End to end: with every Granger call failing, the funnel must not look
    like a clean empty result.
    """
    import pandas as pd
    import statsmodels.tsa.stattools as st

    def exploding(*a, **kw):
        raise TypeError("unexpected keyword argument 'verbose'")

    monkeypatch.setattr(st, "grangercausalitytests", exploding)

    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=600)
    px = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 600)))
         for i in range(6)}, index=idx)

    f = ps.scan(px, "lead-lag", min_corr=0.0)["funnel"]
    assert f["screen_tested"] == 0
    assert f["screen_skipped"] == f["shortlisted"] > 0
    assert f["screen_error"] and "TypeError" in f["screen_error"]


def test_a_healthy_scan_reports_no_error():
    """The reason field must stay empty when nothing is wrong, or it becomes
    noise nobody reads."""
    import pandas as pd

    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2024-01-01", periods=600)
    px = pd.DataFrame(
        {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 600)))
         for i in range(6)}, index=idx)

    f = ps.scan(px, "lead-lag", min_corr=0.0)["funnel"]
    assert f["screen_skipped"] == 0
    assert f["screen_error"] is None


# ── the pin ──────────────────────────────────────────────────────────────────
def test_statsmodels_is_pinned():
    """
    An unpinned scientific dependency changed its API between two deploys of
    unrelated work. The pin is the part that stops it happening again to a
    different function.
    """
    req = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read()
    line = next(l for l in req.splitlines()
                if l.strip().startswith("statsmodels"))
    assert any(op in line for op in ("<", "==", "~=")), (
        f"statsmodels is not pinned: {line!r}")


@pytest.mark.parametrize("fn", ["coint", "adfuller"])
def test_the_other_statsmodels_calls_still_match_their_signatures(fn):
    """The same class of break, for the two functions the pair screen uses."""
    import statsmodels.tsa.stattools as st

    assert hasattr(st, fn), f"statsmodels no longer exports {fn}"
    params = set(inspect.signature(getattr(st, fn)).parameters)
    needed = {"coint": {"y0", "y1"}, "adfuller": {"x", "maxlag", "autolag"}}[fn]
    assert needed <= params, f"{fn} now takes {sorted(params)}"
