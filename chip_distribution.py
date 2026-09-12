"""
chip_distribution.py — 筹码分布 computed locally, not bought from Tushare.

Tushare's cyq_perf / cyq_chips need 5000 points, but neither is privileged
DATA — both are DERIVED from inputs a 2000-point account already has: daily
OHLC, and the turnover rate from daily_basic. The algorithm below is the one
通达信 / 同花顺 use, so what is being replaced is the computation, not a feed.

The model
---------
Chips (筹码) are the float, indexed by the cost basis of whoever holds them.
One state vector over a price grid, summing to 1.0 — the whole float, always
held by somebody.

Each session does exactly two things:

  1. DECAY. A fraction h of the float changed hands today. Those shares'
     cost basis is no longer whatever it was — it is today's price. So every
     existing bucket is scaled by (1 - h). h is the turnover rate.

  2. DEPOSIT. That same fraction h is added back, spread across today's
     traded range [low, high], shaped as a triangle peaked at the day's
     average price. Triangular rather than flat because trading clusters
     around the average — a flat deposit would claim as much volume printed
     at the day's high as at its centre.

      chips ← chips · (1 − h) + h · triangle(low, avg, high)

Iterate over history and the float's cost structure falls out. 获利盘
(winner_rate) is then just the mass sitting below the current price.

Why the initial condition does not matter
-----------------------------------------
Seeding needs a guess, and the guess is provably irrelevant after enough
turnover: each day multiplies whatever remains of it by (1 - h), so what
survives is exactly prod(1 - h_t). That is computed and RETURNED rather than
assumed — `seed_remaining` in the result. When it is under ~1% the answer is
the market's, not the seed's; when it is high the caller is told so instead of
being handed a number that is mostly an artefact of where the walk started.

Price scale
-----------
Prices must be 前复权 (forward-adjusted), which is what this app's loader
returns: the latest bar equals the raw quote and history is restated into
today's money. Feeding raw prices across a dividend would compare a cost basis
in old money against a price in new money, and winner_rate would be wrong in
exactly the cases that matter. Turnover itself is adjustment-free — it is
shares over shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRID_BINS = 400          # resolution of the price axis
PAD = 0.03               # grid padding beyond the traded range
SEED_CONVERGED = 0.01    # seed influence below this = answer is the market's


def _tri_cdf(x: np.ndarray, a: float, c: float, b: float) -> np.ndarray:
    """CDF of the triangular law on [a, b] with peak at c, evaluated at x."""
    x = np.asarray(x, dtype=float)
    out = np.clip((x - a) / (b - a), 0.0, 1.0)   # fallback: uniform
    span = b - a
    if span <= 0:
        return (x >= b).astype(float)
    left, right = c - a, b - c
    out = np.zeros_like(x)
    if left > 0:
        m = (x > a) & (x <= c)
        out[m] = (x[m] - a) ** 2 / (span * left)
    if right > 0:
        m = (x > c) & (x < b)
        out[m] = 1.0 - (b - x[m]) ** 2 / (span * right)
    else:                                        # peak sits on the high edge
        m = (x > c) & (x < b)
        out[m] = 1.0
    if left <= 0:                                # peak sits on the low edge
        m = (x > a) & (x <= c)
        out[m] = 0.0
    out[x >= b] = 1.0
    out[x <= a] = 0.0
    return out


def _deposit(edges: np.ndarray, low: float, avg: float, high: float) -> np.ndarray:
    """
    Today's volume spread across the price bins, summing to 1.

    Integrates the triangle over each bin (CDF difference at the bin edges)
    rather than sampling its height at bin centres. Centre-sampling looks
    equivalent and is not: when a session's range is narrower than a bin — a
    一字板 prints high == low exactly, and quiet days get close — membership
    testing puts the whole day into whichever single bin happens to contain a
    centre, which both misplaces the mass by up to half a bin and turns a
    smooth deposit into a spike. Area integration is exact at every range
    width, including zero, so no degenerate branch is needed at all.
    """
    n = len(edges) - 1
    if not (np.isfinite(low) and np.isfinite(high)) or high < low:
        return np.zeros(n)
    avg = min(max(avg, low), high)

    if high - low < 1e-12:
        # Zero range: the whole session traded at one price. Place it in the
        # bin that actually contains that price.
        w = np.zeros(n)
        i = int(np.clip(np.searchsorted(edges, low) - 1, 0, n - 1))
        w[i] = 1.0
        return w

    w = np.diff(_tri_cdf(edges, low, avg, high))
    s = w.sum()
    if s <= 0:
        w = np.zeros(n)
        i = int(np.clip(np.searchsorted(edges, avg) - 1, 0, n - 1))
        w[i] = 1.0
        return w
    return w / s


def _concentration(grid: np.ndarray, chips: np.ndarray) -> float:
    """(p90-p10)/(p90+p10). 0 = one price, higher = spread out."""
    p90, p10 = _pctl(grid, chips, 0.90), _pctl(grid, chips, 0.10)
    return (p90 - p10) / (p90 + p10) if (p90 + p10) else float("nan")


def compute_chips(df: pd.DataFrame, turnover_pct: pd.Series, *,
                  decay: float = 1.0, bins: int = GRID_BINS,
                  avg_price: pd.Series | None = None,
                  snapshot_every: int = 0) -> dict:
    """
    Run the model over `df` and return the final distribution plus diagnostics.

    df            : 前复权 OHLC, DatetimeIndex, ascending
    turnover_pct  : turnover rate in PERCENT (daily_basic.turnover_rate),
                    aligned to df.index
    decay         : multiplier on turnover. >1 makes chips change hands faster
                    (the distribution forgets sooner), <1 slower. 1.0 = use the
                    exchange's own turnover, which is the honest default.
    avg_price     : the day's average price. VWAP when available; otherwise a
                    (H+L+2C)/4 proxy is used, which weights the close because
                    that is where A-share volume concentrates.
    """
    if df is None or len(df) < 20:
        return {"ok": False, "reason": "not enough history"}

    d = df.sort_index()
    h_arr = d["High"].astype(float).values
    l_arr = d["Low"].astype(float).values
    c_arr = d["Close"].astype(float).values

    if avg_price is not None:
        a_arr = pd.Series(avg_price).reindex(d.index).astype(float).values
        a_arr = np.where(np.isfinite(a_arr), a_arr, (h_arr + l_arr + 2 * c_arr) / 4)
    else:
        a_arr = (h_arr + l_arr + 2 * c_arr) / 4

    t = pd.Series(turnover_pct).reindex(d.index).astype(float)
    # A missing turnover day is a day we know nothing about; treating it as
    # zero turnover (chips frozen) is the conservative reading and beats
    # inventing a number.
    turn = np.clip(t.fillna(0.0).values / 100.0 * float(decay), 0.0, 1.0)

    lo = float(np.nanmin(l_arr)) * (1 - PAD)
    hi = float(np.nanmax(h_arr)) * (1 + PAD)
    edges = np.linspace(lo, hi, int(bins) + 1)
    grid = (edges[:-1] + edges[1:]) / 2.0      # bin centres, for reporting

    # Seed: the first bar's own deposit. Any seed works — see module docstring
    # — and its surviving weight is reported.
    chips = _deposit(edges, l_arr[0], a_arr[0], h_arr[0])
    seed_remaining = 1.0

    # The evolution is free: the loop already holds the full distribution on
    # every bar, so recording it costs a reduction per step rather than a
    # second pass. Snapshots are what make "how has this CHANGED" answerable —
    # a single end-state cannot show chips migrating up into a rally, which is
    # the whole 吸筹 / 派发 read.
    winner_hist = np.empty(len(d))
    conc_hist = np.empty(len(d))
    avg_hist = np.empty(len(d))
    peak_hist = np.empty(len(d))
    snaps, snap_dates = [], []

    for i in range(len(d)):
        k = turn[i]
        if k > 0:
            chips *= (1.0 - k)
            chips += k * _deposit(edges, l_arr[i], a_arr[i], h_arr[i])
            seed_remaining *= (1.0 - k)
        s = chips.sum()
        if s > 0:
            chips /= s                      # guard drift, mass is always 1
        winner_hist[i] = chips[grid < c_arr[i]].sum()
        conc_hist[i] = _concentration(grid, chips)
        avg_hist[i] = (grid * chips).sum()
        peak_hist[i] = grid[int(np.argmax(chips))]
        if snapshot_every and (i % snapshot_every == 0 or i == len(d) - 1):
            snaps.append(chips.copy())
            snap_dates.append(d.index[i])

    return {
        "ok": True,
        "grid": grid,
        "edges": edges,
        "chips": chips,
        "winner_history": pd.Series(winner_hist, index=d.index),
        "concentration_history": pd.Series(conc_hist, index=d.index),
        "avg_cost_history": pd.Series(avg_hist, index=d.index),
        "peak_history": pd.Series(peak_hist, index=d.index),
        "snapshots": np.array(snaps) if snaps else None,
        "snapshot_dates": snap_dates,
        "seed_remaining": float(seed_remaining),
        "converged": bool(seed_remaining < SEED_CONVERGED),
        "cum_turnover": float(turn.sum()),
        "sessions": len(d),
        "decay": float(decay),
    }


def _pctl(grid: np.ndarray, chips: np.ndarray, q: float) -> float:
    """Price below which q of the chips sit, interpolated within the bin."""
    c = np.cumsum(chips)
    if c[-1] <= 0:
        return float("nan")
    c = c / c[-1]
    return float(np.interp(q, c, grid))


def chip_metrics(res: dict, price: float) -> dict:
    """
    The cyq_perf field set, computed from our own distribution.

    Mirrors Tushare's names so this is a drop-in: his_low, his_high,
    cost_5pct … cost_95pct, weight_avg, winner_rate.
    """
    if not res.get("ok"):
        return {"ok": False, "reason": res.get("reason")}
    grid, chips = res["grid"], res["chips"]
    live = chips > 1e-9

    winner = float(chips[grid < price].sum())
    # Interpolate inside the straddling bin so winner_rate moves smoothly with
    # price instead of stepping at bin edges.
    j = int(np.searchsorted(grid, price))
    if 0 < j < len(grid) and chips[j] > 0:
        frac = (price - grid[j - 1]) / (grid[j] - grid[j - 1])
        winner += chips[j] * float(np.clip(frac, 0, 1))

    p90, p10 = _pctl(grid, chips, 0.90), _pctl(grid, chips, 0.10)
    conc = (p90 - p10) / (p90 + p10) if (p90 + p10) else float("nan")

    return {
        "ok": True,
        "his_low": float(grid[live].min()) if live.any() else float("nan"),
        "his_high": float(grid[live].max()) if live.any() else float("nan"),
        "cost_5pct": _pctl(grid, chips, 0.05),
        "cost_15pct": _pctl(grid, chips, 0.15),
        "cost_50pct": _pctl(grid, chips, 0.50),
        "cost_85pct": _pctl(grid, chips, 0.85),
        "cost_95pct": _pctl(grid, chips, 0.95),
        "weight_avg": float((grid * chips).sum() / chips.sum()),
        "winner_rate": float(np.clip(winner, 0.0, 1.0)),
        # 集中度: 0 = one price, 1 = spread everywhere. Below ~0.15 the float
        # is tightly held and a breakout meets little overhead supply.
        "concentration": float(conc),
        "trapped_rate": float(np.clip(1.0 - winner, 0.0, 1.0)),
        "price": float(price),
        "converged": res["converged"],
        "seed_remaining": res["seed_remaining"],
        "cum_turnover_pct": res["cum_turnover"] * 100.0,
        "sessions": res["sessions"],
    }


def find_peaks(grid: np.ndarray, chips: np.ndarray,
               min_share: float = 0.12, min_gap_pct: float = 8.0,
               max_valley: float = 0.60) -> list[dict]:
    """
    Locate the chip peaks — 单峰密集 vs 双峰 is the whole visual read.

    Three rules, and the third is the one that matters. Share and separation
    alone still called every smooth distribution three-peaked, because a
    histogram has ripples and any ripple is a local maximum; a term that reads
    3 on almost everything adds nothing to a ranking but noise. So a second
    peak must also be SEPARATED BY A REAL VALLEY: the lowest point between it
    and a bigger peak has to fall to `max_valley` of the smaller peak's
    height. That is what distinguishes two clusters of cost basis from one
    cluster with a bumpy top, which is exactly the 单峰 / 双峰 question.
    """
    if chips.sum() <= 0:
        return []
    # Smooth lightly first: we are looking for structure, not for every bin.
    k = max(3, len(grid) // 60)
    sm = np.convolve(chips, np.ones(k) / k, mode="same")

    cand = []
    for i in range(1, len(sm) - 1):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]:
            band = np.abs(grid - grid[i]) <= grid[i] * min_gap_pct / 200.0
            cand.append({"i": i, "price": float(grid[i]),
                         "share": float(chips[band].sum()),
                         "height": float(sm[i])})
    cand.sort(key=lambda p: -p["share"])

    kept: list[dict] = []
    for p in cand:
        if p["share"] < min_share:
            continue
        if any(abs(p["price"] - q["price"]) / q["price"] * 100 < min_gap_pct
               for q in kept):
            continue
        # Must be its own mode, not a shoulder of one already kept.
        distinct = True
        for q in kept:
            a, b = sorted((p["i"], q["i"]))
            valley = sm[a:b + 1].min()
            if valley > max_valley * min(p["height"], q["height"]):
                distinct = False
                break
        if distinct:
            kept.append(p)
    for p in kept:
        p.pop("i", None)
    return sorted(kept, key=lambda p: p["price"])


def score_setup(m: dict, *, conc_60d_ago: float | None = None) -> dict:
    """
    Rank a chip structure as a LONG setup, 0–1, with the components exposed.

    The textbook base is 低位单峰密集 with price at or just above the peak:
    one tight cluster of cost basis, little supply stranded overhead, and the
    cluster still tightening. Each term below is one of those words, kept
    separate so a ranking can be argued with rather than taken on faith.

    This is a structure score, not a forecast — it says the overhead is thin
    and the holders agree on a price, not that the stock goes up.
    """
    if not m.get("ok"):
        return {"score": float("nan"), "parts": {}, "label": "—"}

    conc = m.get("concentration", float("nan"))
    # 单峰密集: concentration under ~0.10 is tight, over ~0.35 is scattered.
    tight = float(np.clip((0.35 - conc) / 0.25, 0, 1)) if np.isfinite(conc) else 0.0

    # Overhead supply. Trapped holders above are the sellers a rally must eat
    # through, so less is better — but a winner_rate near 100% is its own risk
    # (everyone is in profit and free to leave), so the preference peaks near
    # 75% rather than running to the top.
    w = m.get("winner_rate", 0.0)
    overhead = float(np.clip(1 - abs(w - 0.75) / 0.75, 0, 1))

    # Peak as support: the dominant cluster sitting at or below price means the
    # weight of cost basis is underneath, not overhead.
    peak, px = m.get("peak_price"), m.get("price")
    if peak and px:
        rel = (px - peak) / px * 100.0          # % price sits above the peak
        support = float(np.clip((rel + 5.0) / 15.0, 0, 1))
    else:
        support = 0.5

    # Tightening = 吸筹 in progress. Unknown is scored neutral, never as a win.
    if conc_60d_ago and np.isfinite(conc) and conc_60d_ago > 0:
        tightening = float(np.clip((conc_60d_ago - conc) / (0.3 * conc_60d_ago), 0, 1))
    else:
        tightening = 0.5

    unimodal = 1.0 if m.get("n_peaks", 0) <= 1 else 0.4 if m.get("n_peaks") == 2 else 0.1

    score = (0.28 * tight + 0.24 * overhead + 0.22 * support
             + 0.14 * tightening + 0.12 * unimodal)
    label = ("🔴 单峰密集·上方轻" if score >= 0.70 else
             "🟠 结构尚可" if score >= 0.55 else
             "⚪ 一般" if score >= 0.40 else "🟢 上方套牢重")
    return {
        "score": round(float(score), 3),
        "label": label,
        "parts": {"tight": round(tight, 2), "overhead": round(overhead, 2),
                  "support": round(support, 2), "tightening": round(tightening, 2),
                  "unimodal": round(unimodal, 2)},
    }


def analyse(df: pd.DataFrame, turnover_pct: pd.Series, *,
            decay: float = 1.0, bins: int = GRID_BINS,
            avg_price: pd.Series | None = None,
            snapshot_every: int = 0) -> dict:
    """Convenience: run the model and return metrics plus the raw arrays."""
    res = compute_chips(df, turnover_pct, decay=decay, bins=bins,
                        avg_price=avg_price, snapshot_every=snapshot_every)
    if not res.get("ok"):
        return res
    m = chip_metrics(res, float(df["Close"].iloc[-1]))

    peaks = find_peaks(res["grid"], res["chips"])
    m["peaks"] = peaks
    m["n_peaks"] = len(peaks)
    m["peak_price"] = (max(peaks, key=lambda p: p["share"])["price"]
                       if peaks else float(res["grid"][int(np.argmax(res["chips"]))]))

    ch = res["concentration_history"]
    prev = float(ch.iloc[-61]) if len(ch) > 61 else None
    m["concentration_60d_ago"] = prev
    m.update({f"setup_{k}": v for k, v in score_setup(m, conc_60d_ago=prev).items()})

    for k in ("grid", "chips", "winner_history", "concentration_history",
              "avg_cost_history", "peak_history", "snapshots", "snapshot_dates"):
        m[k] = res[k]
    return m
