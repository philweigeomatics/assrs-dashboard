"""
pair_scan.py — finding the pairs, instead of guessing which ones to test.

Picking peers by hand is the honest version of a small problem and no version
at all of the real one: you cannot know in advance which two of eighty stocks
lead each other. But searching all of them has a trap that has to be named
before anything else, because it is the entire reason this module is shaped
the way it is.

Eighty stocks make 3,160 pairs. Lead-lag tests each in both directions, so
6,320 hypothesis tests. Under the null a p-value is uniform, so a 5% threshold
returns about 316 "significant" pairs from data with no structure in it at
all. Rank those by p-value, show the top ten, and every one of them will look
compelling. This is not a subtle statistical point — it is the difference
between a screen that finds relationships and a screen that manufactures them.

Correcting for 6,320 tests with Benjamini-Hochberg is the obvious answer and a
poor one: the bar becomes so high that genuine, modest relationships are
rejected along with the noise, and the screen returns nothing forever.

So the search is a funnel, and the last stage is the one that counts:

    1. universe     every watchlist stock with enough overlapping history
    2. shortlist    pairs that simply MOVE TOGETHER, by contemporaneous
                    correlation. A descriptive statistic, not a test — it
                    costs no multiple-testing budget because nothing is being
                    concluded, only narrowed.
    3. screen       the real test, on the FIRST half of the history only
    4. confirm      the same test again on the second half, which played no
                    part in choosing the pair

A pair that survives step 4 was selected without reference to the data that
confirmed it. That is what makes it evidence rather than a coincidence found
by looking hard enough. And because only a handful of pairs reach that stage,
the correction there is mild and the screen can still find something.

The funnel counts are part of the output, not diagnostics. "3,160 → 240 → 31 →
4, against 1.6 expected by chance" tells you how much of what you are looking
at is real. The final four alone tell you nothing.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

#: Pairs must move together this much before anything is tested on them.
#: Not a significance threshold — a relevance one. Two stocks correlated 0.1
#: have no relationship worth the name even if a test happens to fire.
MIN_CORR = 0.45

#: Hard ceiling on the shortlist, whatever the correlation says. Guards the
#: runtime, and keeps the confirmation stage's correction meaningful.
MAX_CANDIDATES = 250

#: Each half of the split needs this many sessions for its test to mean
#: anything. Cointegration wants the most, so it sets the floor.
MIN_HALF = 120

#: Significance used at BOTH stages. The protection comes from the split and
#: from correcting the confirmation stage, not from a stricter threshold.
ALPHA = 0.05

KINDS = ("lead-lag", "pair-trade")


def split_point(n: int) -> int:
    """Index where the screening half ends and the confirming half begins."""
    return n // 2


def usable(prices: pd.DataFrame, min_sessions: int) -> list[str]:
    """Tickers with enough real observations to appear in both halves."""
    return [c for c in prices.columns
            if int(prices[c].notna().sum()) >= min_sessions]


def peers_of(columns, target: str) -> tuple[list[tuple], dict]:
    """
    Every other stock, paired with one target. No correlation filter.

    The filter exists to make an all-pairs sweep affordable: eighty stocks are
    3,160 pairs and 6,320 directional tests, so the field has to be narrowed
    before anything is tested. Naming a target narrows it by a different and
    better means — 79 pairs, 158 tests — and at that size there is nothing
    left to afford. Testing every peer is strictly more informative than
    testing the most correlated fifth of them.

    It is also a different QUESTION, and the better one. "Are there any
    lead-lag pairs among my eighty stocks" is a fishing expedition whose
    honest answer is usually no. "What leads 长电科技" is something a person
    actually wants to know, and the multiple-testing bill for asking it is
    forty times smaller.

    The target is always `a`, so "who leads" reads consistently downstream.
    """
    if target not in columns:
        raise LookupError(f"{target} 不在可用的自选股里 — 可能是历史太短或行情缺失")
    pairs = [(target, other, float("nan")) for other in columns if other != target]
    return pairs, {"pairs_possible": len(pairs), "pairs_correlated": len(pairs),
                   "shortlisted": len(pairs), "min_corr": None, "target": target}


def shortlist(rets: pd.DataFrame, *, min_corr: float = MIN_CORR,
              cap: int = MAX_CANDIDATES,
              within: dict[str, str] | None = None) -> tuple[list[tuple], dict]:
    """
    Pairs worth testing, by contemporaneous correlation alone.

    One correlation matrix, not 3,160 separate calculations — and no p-values,
    because nothing is being concluded here. Narrowing the field on a
    descriptive statistic is what makes the later tests affordable; it does
    bias them, which is exactly what the confirmation half exists to undo.

    `within` optionally maps ticker → group (a sector), restricting pairs to
    stocks that share one. Two stocks from unrelated industries showing a
    0.6 correlation over a year are usually telling you about the market, not
    about each other.
    """
    corr = rets.corr()
    pairs = []
    for a, b in itertools.combinations(list(rets.columns), 2):
        if within is not None and within.get(a) != within.get(b):
            continue
        r = corr.at[a, b]
        if r is not None and not math.isnan(r) and abs(r) >= min_corr:
            pairs.append((a, b, float(r)))

    total = len(list(itertools.combinations(list(rets.columns), 2)))
    pairs.sort(key=lambda p: -abs(p[2]))
    return pairs[:cap], {"pairs_possible": total, "pairs_correlated": len(pairs),
                         "shortlisted": min(len(pairs), cap),
                         "min_corr": min_corr}


# ── the two screens ──────────────────────────────────────────────────────────
def _granger(y: np.ndarray, x: np.ndarray, maxlag: int) -> tuple[float, int]:
    """Best p-value across lags for 'x helps predict y', and the lag."""
    from statsmodels.tsa.stattools import grangercausalitytests
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(np.column_stack([y, x]), maxlag=maxlag,
                                        verbose=False)
    except Exception:                                              # noqa: BLE001
        return float("nan"), 0
    best_p, best_lag = float("nan"), 0
    for lag, out in res.items():
        p = out[0]["ssr_ftest"][1]
        if math.isnan(best_p) or p < best_p:
            best_p, best_lag = float(p), int(lag)
    return best_p, best_lag


def lead_lag_test(a: pd.Series, b: pd.Series, maxlag: int = 5) -> dict | None:
    """Granger in both directions on one slice. None if too little overlap."""
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(aligned) < 60:
        return None
    av, bv = aligned["a"].to_numpy(), aligned["b"].to_numpy()

    p_ab, lag_ab = _granger(bv, av, maxlag)      # does A help predict B
    p_ba, lag_ba = _granger(av, bv, maxlag)      # does B help predict A
    best = min([p for p in (p_ab, p_ba) if not math.isnan(p)], default=float("nan"))
    if math.isnan(best):
        return None

    leads = "a" if (not math.isnan(p_ab) and p_ab <= (p_ba if not math.isnan(p_ba) else 9)) else "b"
    lag = lag_ab if leads == "a" else lag_ba

    # HOW BIG, not just whether. A Granger p-value says "knowing the leader
    # helps predict the follower"; it says nothing about by how much, and on
    # daily A-share returns the honest answer is "barely". Without a
    # magnitude beside it, "领先 4 天, q=0.018" reads as though a 5% move in
    # the leader implies a 5% move in the follower. Measured, it implies
    # about 0.3%.
    x, y = (av, bv) if leads == "a" else (bv, av)
    beta = r2 = float("nan")
    if lag and len(x) > lag + 30:
        xs, ys = x[:-lag], y[lag:]
        if xs.std() > 1e-12 and ys.std() > 1e-12:
            beta = float(np.polyfit(xs, ys, 1)[0])
            r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)

    return {"p": best, "p_ab": p_ab, "p_ba": p_ba,
            "lag": lag, "leads": leads, "n": len(aligned),
            "beta": beta, "r2": r2}


def coint_test(a: pd.Series, b: pd.Series) -> dict | None:
    """
    Engle-Granger on LOG prices, plus the OU half-life of the spread.

    Log space so the spread is a percentage deviation rather than a dollar
    gap, which keeps the hedge ratio and the half-life comparable across
    stocks at very different price levels.
    """
    from statsmodels.tsa.stattools import coint
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    aligned = aligned[(aligned > 0).all(axis=1)]
    if len(aligned) < MIN_HALF:
        return None

    la, lb = np.log(aligned["a"].to_numpy()), np.log(aligned["b"].to_numpy())
    try:
        _, p, _ = coint(la, lb)
    except Exception:                                              # noqa: BLE001
        return None
    beta = float(np.polyfit(lb, la, 1)[0])
    return {"p": float(p), "beta": beta,
            "half_life": _half_life(pd.Series(la - beta * lb)), "n": len(aligned)}


def _half_life(spread: pd.Series) -> float:
    """OU half-life in sessions; nan when the spread does not mean-revert."""
    clean = spread.dropna()
    if len(clean) < 30:
        return float("nan")
    delta = clean.diff().dropna()
    lagged = clean.shift(1).dropna()
    delta, lagged = delta.align(lagged, join="inner")
    try:
        kappa = float(np.polyfit(lagged.to_numpy(), delta.to_numpy(), 1)[0])
    except Exception:                                              # noqa: BLE001
        return float("nan")
    return float(-np.log(2) / kappa) if kappa < 0 else float("nan")


# ── the funnel ───────────────────────────────────────────────────────────────
def scan(prices: pd.DataFrame, kind: str = "pair-trade", *,
         min_corr: float = MIN_CORR, cap: int = MAX_CANDIDATES,
         maxlag: int = 5, within: dict[str, str] | None = None,
         target: str | None = None) -> dict:
    """
    Search a watchlist, and report how much of what was found is real.

    With `target`, the search is every peer against that one stock and the
    correlation filter is not applied — see peers_of. Without it, the search
    is all pairs and the filter is what makes that affordable.

    Raises LookupError when the history cannot support a split — half of a
    year is not enough for a cointegration test, and a screen that ran anyway
    would be guessing with a confident face.
    """
    if kind not in KINDS:
        raise LookupError(f"未知的扫描类型：{kind}")

    cols = usable(prices, MIN_HALF * 2)
    if len(cols) < 2:
        raise LookupError(f"只有 {len(cols)} 只股票有足够长的历史（需要 "
                          f"{MIN_HALF * 2} 个交易日），无法配对")
    px = prices[cols].ffill()
    if len(px) < MIN_HALF * 2:
        raise LookupError(f"可用历史仅 {len(px)} 个交易日，不足以切成两半"
                          f"（每半至少 {MIN_HALF} 天）")

    cut = split_point(len(px))
    train_px, test_px = px.iloc[:cut], px.iloc[cut:]
    train_r = train_px.pct_change(fill_method=None).dropna(how="all")
    test_r = test_px.pct_change(fill_method=None).dropna(how="all")

    # Shortlisted on the TRAINING half only. Using the whole history here
    # would let the confirming half influence which pairs got chosen, which
    # is the leak this design exists to prevent. Naming a target skips the
    # step entirely, so there is no selection to leak.
    if target:
        pairs, funnel = peers_of(cols, target)
    else:
        pairs, funnel = shortlist(train_r, min_corr=min_corr, cap=cap, within=within)

    screened, confirmed = [], []
    # What the screen SAW, not only what it passed. "0 前半程显著" is a
    # different statement depending on whether the best p was 0.06 or 0.51,
    # and a different one again if the test could not run at all — and none
    # of those are distinguishable from a count of zero. Cheap to carry, and
    # the only way an empty result can be argued with.
    seen: list[float] = []
    skipped = 0
    for a, b, r in pairs:
        if kind == "lead-lag":
            tr = lead_lag_test(train_r[a], train_r[b], maxlag)
        else:
            tr = coint_test(train_px[a], train_px[b])
        if tr is None:
            skipped += 1
            continue
        seen.append(float(tr["p"]))
        if tr["p"] >= ALPHA:
            continue
        screened.append((a, b, r, tr))

    for a, b, r, tr in screened:
        if kind == "lead-lag":
            te = lead_lag_test(test_r[a], test_r[b], maxlag)
        else:
            te = coint_test(test_px[a], test_px[b])
        if te is None:
            continue
        confirmed.append({"a": a, "b": b,
                          "corr": None if r != r else round(r, 3),
                          "train": tr, "test": te})

    qs = _bh([c["test"]["p"] for c in confirmed])
    rows = []
    for c, q in zip(confirmed, qs):
        rows.append(_row(c, q, kind))
    rows = [r for r in rows if r["survives"]] + [r for r in rows if not r["survives"]]
    rows.sort(key=lambda r: (not r["survives"], r["q"]))

    funnel.update({
        "targeted": bool(target),
        "universe": len(cols),
        # The screen's own evidence, so an empty result can be read.
        "screen_tested": len(seen),
        "screen_skipped": skipped,
        "screen_min_p": round(min(seen), 4) if seen else None,
        "screen_under_10": sum(1 for p in seen if p < 0.10),
        "screened": len(screened),
        "retested": len(confirmed),
        # How many cleared the holdout on the RAW threshold. This is the number
        # `expected_by_chance` is the counterpart to — comparing an already
        # corrected survivor count against an uncorrected expectation would be
        # comparing two different things and flattering the screen.
        "retest_hits": sum(1 for c in confirmed if c["test"]["p"] < ALPHA),
        # If every retest were pure noise, this many would pass anyway. For
        # lead-lag a coincidence must ALSO land the same arrow in both halves
        # to survive, and noise does that half the time — so the bar a
        # survivor count is measured against is half as high.
        "expected_by_chance": round(len(confirmed) * ALPHA
                                    * (0.5 if kind == "lead-lag" else 1.0), 2),
        "survivors": sum(1 for r in rows if r["survives"]),
        "alpha": ALPHA,
    })
    return {
        "kind": kind,
        "sessions": int(len(px)),
        "train": {"from": str(train_px.index[0].date()),
                  "to": str(train_px.index[-1].date()), "sessions": cut},
        "test": {"from": str(test_px.index[0].date()),
                 "to": str(test_px.index[-1].date()), "sessions": len(px) - cut},
        "funnel": funnel,
        "rows": rows,
    }


def _row(c: dict, q: float, kind: str) -> dict:
    tr, te = c["train"], c["test"]
    cleared = bool(q is not None and q < ALPHA)
    row = {
        "a": c["a"], "b": c["b"], "corr": c["corr"],
        "p_train": _n(tr["p"], 5), "p_test": _n(te["p"], 5),
        "q": _n(q, 5) if q is not None else None,
        "survives": cleared,
        "n_test": int(te["n"]),
    }
    if kind == "lead-lag":
        # Does the confirming half agree about WHICH one leads? A pair that
        # swaps direction between halves is not a weak lead — it is noise
        # that happened to be significant twice, and a coin flip decided the
        # arrow each time.
        #
        # This is a survival requirement rather than a column to read,
        # because measured on the real watchlist it is the thing that
        # separates the screen from a random number generator: of 75 pairs
        # that cleared BH across 14 targets, 53% agreed on direction, and
        # pure noise agrees 50% of the time. Solving the mixture says only
        # about 6% of them were real. Reading "存活 19" without the arrows
        # is reading mostly coincidences.
        agrees = tr["leads"] == te["leads"]
        row.update({
            "leads": te["leads"], "lag": int(te["lag"]),
            "same_direction": agrees,
            "survives": cleared and agrees,
            # Measured on the holdout half, so the size is out-of-sample too.
            # Named apart from the cointegration `beta`, which is a hedge
            # ratio and an entirely different quantity.
            "lead_beta": _n(te.get("beta"), 3),
            "lead_r2": _n(te.get("r2"), 4),
        })
    else:
        row.update({
            "beta": _n(te["beta"], 3),
            "half_life": _n(te["half_life"], 1),
            # A spread that takes half a year to close is cointegrated and
            # untradeable. Reported, not filtered — that is the reader's call.
            "tradeable": bool(te["half_life"] == te["half_life"]
                              and 0 < te["half_life"] <= 30),
        })
    return row


def _n(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


def _bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values over the CONFIRMATION tests only."""
    n = len(pvals)
    if not n:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, pvals[idx] * n / rank)
        q[idx] = min(prev, 1.0)
    return q
