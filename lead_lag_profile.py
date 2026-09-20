"""
lead_lag_profile.py — when did A lead B, and by how much, and for how long?

A single verdict ("A 领先 B 4 天, q=0.018") is a summary that hides the thing
worth knowing. It averages a whole history into one arrow and one number, and
we have already measured how badly that can mislead: across fourteen targets,
half the pairs that cleared the correction pointed the OTHER way in the second
half of the same history. A relationship that reverses is not a 4-day lead; it
is two different regimes, or noise, and the verdict cannot tell you which.

So this does not conclude. It lays out the history: for every rolling window,
the cross-correlation of A's returns against B's at every lag from −maxlag to
+maxlag. What comes back is a picture a person reads — a stripe that holds at
one lag for months is a behaviour worth knowing about; a speckle that lands on
a different lag every window is what two unrelated stocks look like.

WHY THERE IS A PLACEBO IN THE OUTPUT

Every rolling window has a best lag, always, including in data with no
relationship whatsoever. Show that picture alone and the eye will find
structure in it — that is what eyes do. So the same computation is run on one
series rotated in time, which destroys any real alignment while preserving
each stock's own volatility and autocorrelation exactly. The placebo is what
NOTHING looks like for this pair, at this window length, on this many
windows. Read the real one next to it or do not read it at all.

WHAT THE SHADING IS NOT

Cells above the noise band are not "significant". With eleven lags across
dozens of windows there are hundreds of cells, and at any sane threshold a
handful clear it by luck. The band is drawn so that the FAINT cells can be
ignored, not so the strong ones can be believed individually.

AND PERSISTENCE IS NOT EVIDENCE EITHER, WHICH IS THE POINT OF THE NULLS

The obvious next thought — "a stripe that holds for months must be real" — is
wrong here, and measurably so. Neighbouring windows share `window − step`
sessions, 55 of 60 at the defaults, so a chance correlation in one window is
still there in the next fifteen by construction. Measured on forty pairs of
independent random walks: the longest run of a single dominant lag had a
MEDIAN of 9 windows, 98% of pairs produced a run of 5 or more, and the worst
reached 26. A three-month stretch where A appears to lead B by three days is
what two unrelated stocks look like.

So the panel ships with a null distribution rather than a single placebo: the
same computation over several rotations of B, giving the longest run and the
episode count that THIS pair, at THIS window length, produces with the
alignment destroyed. An episode is worth a second look when it is longer than
anything the rotations managed. Below that, it is the texture of the method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Sessions in each rolling window. About three months — long enough for a
#: correlation to mean something, short enough that a regime change shows up
#: as a change rather than being averaged away.
WINDOW = 60

#: Sessions between windows. Weekly: finer just redraws the same information
#: with more overlap between neighbouring columns.
STEP = 5

#: Lags scanned either side of zero. Beyond a week, a daily lead-lag story in
#: equities is usually a story about something else.
MAX_LAG = 5

#: A run must hold this many consecutive windows before it is called an
#: episode. Two windows overlap heavily, so two in a row is one observation
#: wearing a disguise. This is a floor on what gets LISTED, not a claim that
#: anything above it is real — see the nulls.
MIN_RUN = 3

#: Rotations used to build the null distribution. Enough for a median and a
#: maximum to mean something without making the request slow.
ROTATIONS = 8

#: Rotation applied to build the placebo, as a fraction of the series. Near
#: half, so no window is compared against anything close to its own dates.
PLACEBO_SHIFT = 0.47


def _corr_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """
    Correlation of a[t−lag] with b[t]. Positive lag means A moves FIRST.

    Sign convention stated because it is the whole meaning of the output and
    it is the easy thing to get backwards.
    """
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    if len(x) < 20:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def noise_band(window: int) -> float:
    """
    |r| a single window would reach by luck about 5% of the time.

    2/sqrt(n) is the usual approximation. It is a per-cell figure and the
    panel has hundreds of cells, which is exactly why the output calls this a
    noise band and never calls a cell significant.
    """
    return 2.0 / max(1.0, float(window)) ** 0.5


def profile(a: pd.Series, b: pd.Series, *, window: int = WINDOW,
            step: int = STEP, maxlag: int = MAX_LAG) -> dict:
    """
    The rolling cross-correlation panel for one pair of return series.

    Returns dates (the LAST session of each window), the lags, and a matrix
    of correlations indexed [window][lag].
    """
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(aligned) < window + maxlag + step:
        raise LookupError(
            f"可用历史仅 {len(aligned)} 个交易日，不足以按 {window} 天的窗口滚动")

    av, bv = aligned["a"].to_numpy(), aligned["b"].to_numpy()
    lags = list(range(-maxlag, maxlag + 1))
    dates, rows = [], []
    for end in range(window, len(aligned) + 1, step):
        seg_a, seg_b = av[end - window:end], bv[end - window:end]
        rows.append([_corr_at(seg_a, seg_b, k) for k in lags])
        dates.append(str(aligned.index[end - 1].date()))

    return {"dates": dates, "lags": lags,
            "matrix": [[None if v != v else round(v, 3) for v in r] for r in rows],
            "window": window, "step": step, "band": round(noise_band(window), 3)}


def dominant(panel: dict) -> list[int | None]:
    """
    The lag with the largest |r| in each window, or None if none clears the band.

    "None" is a real and common answer, and a panel that always names a lag
    would be telling you something about the arithmetic rather than the pair.
    """
    band, lags = panel["band"], panel["lags"]
    out: list[int | None] = []
    for row in panel["matrix"]:
        best, val = None, 0.0
        for k, v in zip(lags, row):
            if v is None:
                continue
            if abs(v) > abs(val):
                best, val = k, v
        out.append(best if abs(val) >= band else None)
    return out


def episodes(panel: dict, *, min_run: int = MIN_RUN) -> list[dict]:
    """
    Stretches where one lag stayed dominant — the "instances" worth reading.

    Consecutive windows overlap by `window − step` sessions, so a run of three
    is not three independent observations. The count is reported in windows
    and in calendar dates rather than as a sample size, because it is not one.
    """
    doms = dominant(panel)
    dates, lags = panel["dates"], panel["lags"]
    out, start = [], 0
    for i in range(1, len(doms) + 1):
        if i < len(doms) and doms[i] == doms[start]:
            continue
        lag, run = doms[start], i - start
        if lag is not None and run >= min_run:
            col = lags.index(lag)
            vals = [panel["matrix"][j][col] for j in range(start, i)
                    if panel["matrix"][j][col] is not None]
            out.append({
                "lag": int(lag),
                "leads": "a" if lag > 0 else "b",
                "from": dates[start], "to": dates[i - 1],
                "windows": int(run),
                "mean_corr": round(float(np.mean(vals)), 3) if vals else None,
                "peak_corr": round(float(max(vals, key=abs)), 3) if vals else None,
            })
        start = i
    return out


def placebo(a: pd.Series, b: pd.Series, *, shift: float = PLACEBO_SHIFT,
            **kw) -> dict:
    """
    The same panel with B rotated in time — what NOTHING looks like here.

    Rotation, not reshuffling: it keeps B's own autocorrelation and volatility
    clustering intact and destroys only its alignment to A. A shuffle would
    produce a flatter, prettier placebo and would flatter the real panel by
    comparison.
    """
    by = int(len(b) * shift) or 1
    rotated = pd.Series(np.roll(b.to_numpy(), by), index=b.index, name=b.name)
    return profile(a, rotated, **kw)


def _longest(doms) -> int:
    best = run = 0
    for i, d in enumerate(doms):
        run = run + 1 if (d is not None and i and d == doms[i - 1]) else 1
        if d is not None:
            best = max(best, run)
    return best


def nulls(a: pd.Series, b: pd.Series, *, rotations: int = ROTATIONS,
          min_run: int = MIN_RUN, **kw) -> dict:
    """
    What this pair produces with the alignment destroyed, several times over.

    One rotation is one draw from a wide distribution and can land low by
    luck, which would make the real panel look special when it is ordinary.
    The rotations are spread across the history so no two are near-copies.
    """
    runs, counts, shares = [], [], []
    for i in range(max(1, rotations)):
        frac = 0.12 + 0.76 * (i / max(1, rotations - 1)) if rotations > 1 \
            else PLACEBO_SHIFT
        sham = placebo(a, b, shift=frac, **kw)
        doms = dominant(sham)
        runs.append(_longest(doms))
        counts.append(len(episodes(sham, min_run=min_run)))
        shares.append(sum(1 for d in doms if d is not None) / max(1, len(doms)))
    return {
        "rotations": len(runs),
        "longest_median": int(np.median(runs)),
        "longest_max": int(max(runs)),
        "episodes_median": float(np.median(counts)),
        "named_share_median": round(float(np.median(shares)), 3),
    }


def summarise(panel: dict, null: dict | None = None,
              *, min_run: int = MIN_RUN) -> dict:
    """
    What the panel says, in the only terms it can honestly support.

    No verdict. Per-lag share of windows, the longest run, the episodes — and
    beside each, what the rotations produced. Every episode carries
    `beats_null`, which is the only thing here that distinguishes a behaviour
    from the texture of overlapping windows, and even that is a prompt to look
    rather than a finding.
    """
    doms = dominant(panel)
    named = [d for d in doms if d is not None]
    share = {k: round(named.count(k) / len(doms), 3) for k in panel["lags"]} \
        if doms else {}
    eps = episodes(panel, min_run=min_run)

    if null:
        for e in eps:
            e["beats_null"] = bool(e["windows"] > null["longest_max"])

    out = {
        "windows": len(doms),
        "named": len(named),
        "named_share": round(len(named) / len(doms), 3) if doms else 0.0,
        # The headline, and usually the true answer. Lag 0 winning means the
        # two move on the SAME day and there is no lead to trade — which is
        # the most common honest outcome for two related stocks, and the one
        # a Granger verdict hides because it conditions away the follower's
        # own past before it looks. Excluding lag 0 to "find the lead" would
        # manufacture one out of plain co-movement.
        "sync_share": round(named.count(0) / len(named), 3) if named else 0.0,
        "share": share,
        "longest_run": _longest(doms),
        "episodes": eps,
    }
    if null:
        out["null"] = null
        out["notable"] = sum(1 for e in eps if e.get("beats_null"))
    return out
