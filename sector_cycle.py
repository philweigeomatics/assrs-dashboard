"""
sector_cycle.py — 板块周期: where each 板块 is in its own trend cycle, and
which pairs take turns.

The trend oscillator
--------------------
For each 板块 index: how far it sits from its own 20-day average, divided by
how far it usually strays from it (the trailing 120-day spread of that same
distance). +1 means "unusually far above its trend for THIS sector", so a
calm 白酒 and a wild 光模块 read on one scale. It is the continuous version of
the red/green heatmap on the dashboard, which only says above or below.

`abs` is the 板块 index as it traded. `rel` takes 沪深300 out first — and
takes it out by each sector's own β, not by dividing by the index.

Dividing looks like the natural "relative to 沪深300" and is not: a sector
divided by the index carries (β − 1) × the market. 银行 (β ≈ 0) lags the index
on every up day and 半导体 (β ≈ 1.9) beats it, so the two "move apart"
whenever the market moves, with no money rotating between them at all. On
these 板块 the product (β_a − 1)(β_b − 1) alone explained 0.81 of how strongly
a pair see-sawed on the ratio basis, which flagged 59 pairs. Subtracting
β × 沪深300 (each sector's daily return minus its β times the index's,
compounded) leaves 18 — defensive against tech hardware — that move
opposite beyond what the market's direction explains.

Phases
------
Four, which a cycle passes through in order:

    上行  above its trend, and higher than five days ago
    回落  above its trend, but lower than five days ago
    下行  below its trend, and lower than five days ago
    回升  below its trend, but higher than five days ago

Each sector's typical stay in each phase is measured from its own history,
so "第 4 天 · 典型 9 天" compares it with itself, not with a rule of thumb.

Is a pair a see-saw?
--------------------
Three measures, each against what chance would give:

  * Lag correlation: corr(A today, B L days later) for L in −20..20; the
    most negative is the pair's see-saw, and L says who moves first.
  * Opposite-phase share: how often one is above its trend while the other
    is below.
  * Turn follow-through: after A crosses up through its trend, how often B
    crosses down within ten days either side, and the reverse.

Trends are sticky — a sector above its average today is likely above it
tomorrow — so ordinary significance tests on these series wildly overstate
the evidence. Chance here is phase-randomised surrogates of B: series with
exactly B's spectrum, so exactly its stickiness and rhythm, but random
timing and so no relation to A. Every measure is recomputed on 4,000
of them — enough that a pair can survive the correction for 325 (at 300
the smallest possible p is 1/301, and 10% FDR over 325 needs ~0.0003).
And because the most negative of 41 lags is picked, each surrogate picks
the most negative of 41 too.

The first version rotated B's own history against A's instead. On pairs
of independent sectors it called 9% significant at p<0.01: the real
alignment is the only one with no seam where B's end wraps onto its start,
and with trends this sticky the seam weakens every other alignment, so the
real one looked special when it was not. Surrogates came out at 1.3%.

With 26 sectors there are 325 pairs; the ranking is corrected for testing
all of them (Benjamini-Hochberg).

Window: from 2024-10-14 — the start of the market as it now trades after the
924 policy pivot — with the earlier bars used only to warm the oscillator.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

MA_DAYS = 20
SPREAD_DAYS = 120
SLOPE_DAYS = 5
MAX_LAG = 20
MIN_SHIFT = 60                 # minimum history, in units of this, for a pair to be tested
SURROGATES = 4000
PAIR_SURROGATES = 1000          # one pair, no correction needed
TURN_WINDOW = 10               # ± days for "B turned too"
WINDOW_START = "2024-10-14"
SPARK_DAYS = 120

PHASES = {
    "up": {"label": "上行", "order": 0},
    "topping": {"label": "回落", "order": 1},
    "down": {"label": "下行", "order": 2},
    "bottoming": {"label": "回升", "order": 3},
}


def _f(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


# ── the oscillator ───────────────────────────────────────────────────────────
def oscillator(close: pd.Series) -> pd.Series:
    """Distance from the 20-day average, in units of its own trailing spread."""
    dev = close / close.rolling(MA_DAYS, min_periods=MA_DAYS).mean() - 1
    # Yesterday's yardstick: a jump today must not shrink its own reading.
    spread = dev.rolling(SPREAD_DAYS, min_periods=SPREAD_DAYS // 2).std().shift(1)
    return dev / spread


#: Phase smoothing and dead bands. Read straight off the oscillator, half of
#: all phases on these 板块 lasted two days or less — a label that flips on
#: noise cannot answer "is it rising right now". A 3-day EMA plus hysteresis
#: (stay "above" until clearly below, "rising" until clearly falling) gives
#: a median phase of 7 days with ~8% of them two days or shorter.
PHASE_EMA = 3
ZERO_BAND = 0.3
SLOPE_BAND = 0.2


def _sticky(x: np.ndarray, band: float) -> list:
    """True/False for above/below zero, switching only once clearly across."""
    state, out = None, []
    for v in x:
        if not np.isfinite(v):
            out.append(None)
            continue
        if state is None:
            state = bool(v > 0)
        elif state and v < -band:
            state = False
        elif not state and v > band:
            state = True
        out.append(state)
    return out


def smooth(osc: pd.Series) -> pd.Series:
    return osc.ewm(span=PHASE_EMA, adjust=False).mean().where(osc.notna())


def phases(osc: pd.Series) -> pd.Series:
    sm = smooth(osc)
    above = _sticky(sm.to_numpy(), ZERO_BAND)
    rising = _sticky((sm - sm.shift(SLOPE_DAYS)).to_numpy(), SLOPE_BAND)
    lab = [None if a is None or r is None else
           ("up" if a and r else "topping" if a else "down" if not r else "bottoming")
           for a, r in zip(above, rising)]
    return pd.Series(lab, index=osc.index, dtype=object)


def runs(labels: pd.Series) -> list[tuple[str, int, int]]:
    """(phase, start position, length) for each consecutive stretch."""
    v = labels.to_numpy()
    out, start = [], None
    for i, x in enumerate(v):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            if start is not None:
                out.append((v[start], start, i - start))
                start = None
            continue
        if start is None:
            start = i
        elif x != v[start]:
            out.append((v[start], start, i - start))
            start = i
    if start is not None:
        out.append((v[start], start, len(v) - start))
    return out


def crosses(osc: np.ndarray, direction: str) -> np.ndarray:
    """
    Positions where a sector crossed its trend line — read off the same
    smoothed, hysteresis state the phase labels are, so "上穿均线 x 天前" and
    the phase on screen can never disagree. (A first version used its own
    rule, five confirming days on the raw oscillator, and reported 银行 as
    having crossed down 21 days ago while it sat at +1.4σ: a later dip of
    three days had un-confirmed the crossing back up.)
    """
    sm = pd.Series(osc).ewm(span=PHASE_EMA, adjust=False).mean().where(pd.Series(osc).notna())
    st = _sticky(sm.to_numpy(), ZERO_BAND)
    hit = np.zeros(len(osc), dtype=bool)
    for t in range(1, len(st)):
        if st[t] is None or st[t - 1] is None or st[t] == st[t - 1]:
            continue
        if (direction == "up") == bool(st[t]):
            hit[t] = True
    return hit


# ── pairs ────────────────────────────────────────────────────────────────────
def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() or 1)


def _circular_ccf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ccf[L] = corr(a_t, b_{t+L}) with b treated as circular, L = 0..n-1."""
    a, b = _z(a), _z(b)
    return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n=len(a)) / len(a)


def surrogates(b: np.ndarray, draws: int = SURROGATES, seed: int = 0) -> np.ndarray:
    """
    `draws` series with b's spectrum and random phases: b's rhythm and
    stickiness, none of its timing. Standardised, one per row.
    """
    rng = np.random.default_rng(seed)
    n = len(b)
    fb = np.fft.rfft(b - b.mean())
    ph = rng.uniform(0, 2 * np.pi, (draws, len(fb)))
    ph[:, 0] = 0
    if n % 2 == 0:
        ph[:, -1] = 0
    sb = np.fft.irfft(np.abs(fb)[None, :] * np.exp(1j * ph), n=n, axis=1)
    return (sb - sb.mean(axis=1, keepdims=True)) / sb.std(axis=1, keepdims=True)


def prepared(b: np.ndarray, surr: np.ndarray) -> dict:
    """
    What every pair test needs from one sector's surrogates, computed once:
    their transforms for the lag test and their side of the trend line for
    the opposite-phase test. With 4,000 surrogates reused across 25 pairs,
    doing this per pair instead tripled the run time.
    """
    n = surr.shape[1]
    level = b.mean() + surr * b.std()
    return {"z": surr, "f": np.fft.rfft(surr, axis=1), "n": n,
            "sign": np.sign(level)}


def _prep(b, surr):
    if surr is None:
        surr = surrogates(b)
    return surr if isinstance(surr, dict) else prepared(b, surr)


def lag_test(a: np.ndarray, b: np.ndarray, surr=None) -> dict:
    """
    The most negative correlation over lags −MAX_LAG..MAX_LAG, and how often
    a surrogate of b finds one at least as negative.
    """
    n = len(a)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    window = _circular_ccf(a, b)[lags % n]
    i = int(np.argmin(window))
    obs = float(window[i])
    sp = _prep(b, surr)
    fa = np.fft.rfft(_z(a))
    ccf = np.fft.irfft(np.conj(fa)[None, :] * sp["f"], n=n, axis=1) / n
    null = ccf[:, lags % n].min(axis=1)
    p = float((1 + np.sum(null <= obs)) / (len(null) + 1))
    return {"best_lag": int(lags[i]), "best_corr": obs, "corr0": float(window[MAX_LAG]),
            "curve": [float(v) for v in window], "p": p,
            "null_median": float(np.median(null))}


def opposite_test(a: np.ndarray, b: np.ndarray, surr=None) -> dict:
    """
    Share of days on opposite sides of their trends, against surrogates of b.

    A surrogate is centred, so it is put back on b's level before asking
    which side of zero it is on (see `prepared`) — otherwise a sector that
    spent most of the window above trend would look half the time below it.
    With ±1 signs, "opposite" is (1 − agreement)/2 and agreement is a dot
    product, so the whole null is one matrix product.
    """
    obs = float(np.mean(np.sign(a) != np.sign(b)))
    sp = _prep(b, surr)
    sa = np.sign(a)
    both = (sp["sign"] != 0) & (sa[None, :] != 0)
    agree = (sp["sign"] @ sa) / len(a)
    null = (both.mean(axis=1) - agree) / 2
    p = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    return {"share": obs, "expected": float(np.median(null)), "p": p}


def follow_test(a: np.ndarray, b: np.ndarray, a_dir: str, b_dir: str,
                surr: np.ndarray | None = None) -> dict:
    """After A crosses `a_dir`, did B cross `b_dir` within ±TURN_WINDOW days?"""
    ta = np.flatnonzero(crosses(a, a_dir))
    n = len(a)

    def rate(h: np.ndarray):
        hits, offs = 0, []
        for t in ta:
            lo, hi = max(0, t - TURN_WINDOW), min(n, t + TURN_WINDOW + 1)
            near = np.flatnonzero(h[lo:hi]) + lo - t
            if len(near):
                hits += 1
                offs.append(int(near[np.argmin(np.abs(near))]))
        return hits, offs

    hits, offs = rate(crosses(b, b_dir))
    if not len(ta):
        return {"events": 0, "hits": 0, "rate": None, "chance": None,
                "median_offset": None, "p": None}
    sp = _prep(b, surr)
    level = b.mean() + sp["z"][:100] * b.std()
    null = np.asarray([rate(crosses(row, b_dir))[0] / len(ta) for row in level])
    p = float((1 + np.sum(null >= hits / len(ta))) / (len(null) + 1))
    return {"events": int(len(ta)), "hits": int(hits),
            "rate": _f(hits / len(ta), 3), "chance": _f(float(np.mean(null)), 3),
            "median_offset": int(np.median(offs)) if offs else None,
            "p": _f(p, 4)}


def bh_q(ps: list[float]) -> list[float]:
    p = np.asarray(ps, dtype=float)
    n = len(p)
    if not n:
        return []
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()


# ── assembly ─────────────────────────────────────────────────────────────────
def build_frame(closes: dict[str, pd.Series], bench: pd.Series | None, basis: str,
                start: str = WINDOW_START) -> pd.DataFrame:
    """
    One price column per sector. For `rel`, each is rebuilt from its daily
    return minus β × 沪深300's, β fitted on the analysis window; the βs ride
    along in `frame.attrs["beta"]`.
    """
    frame = pd.DataFrame(closes).sort_index().ffill(limit=3)
    if basis != "rel":
        return frame
    if bench is None:
        raise LookupError("沪深300 数据不可用，无法计算相对口径")
    m = bench.reindex(frame.index).ffill(limit=3).pct_change()
    r = frame.pct_change()
    win = r.index >= pd.Timestamp(start)
    var = float(m[win].var())
    betas, out = {}, {}
    for c in frame.columns:
        beta = float(r[c][win].cov(m[win]) / var) if var > 0 else 1.0
        betas[c] = beta
        resid = (r[c] - beta * m).where(r[c].notna())
        out[c] = (1 + resid.fillna(0)).cumprod().where(frame[c].notna())
    res = pd.DataFrame(out)
    res.attrs["beta"] = betas
    return res


def analyse(closes: dict[str, pd.Series], bench: pd.Series | None = None,
            basis: str = "abs", start: str = WINDOW_START) -> dict:
    if basis not in ("abs", "rel"):
        raise ValueError("basis must be abs or rel")
    frame = build_frame(closes, bench, basis, start)
    betas = frame.attrs.get("beta", {})
    osc = frame.apply(oscillator)
    win = osc[osc.index >= pd.Timestamp(start)].dropna(how="all")
    if len(win) < 2 * MIN_SHIFT + 20:
        raise LookupError(f"{start} 以来只有 {len(win)} 个交易日，不足以判断周期")
    names = [c for c in win.columns if win[c].notna().sum() >= 2 * MIN_SHIFT + 20]

    board = []
    for c in names:
        o = win[c].dropna()
        ph = phases(osc[c]).reindex(o.index)
        rs = runs(ph)
        cur_phase, _, cur_len = rs[-1] if rs else (None, 0, 0)
        typical = {}
        for key in PHASES:
            done = [ln for p, st, ln in rs[:-1] if p == key]
            typical[key] = int(np.median(done)) if len(done) >= 3 else None
        up_x = np.flatnonzero(crosses(o.to_numpy(), "up"))
        dn_x = np.flatnonzero(crosses(o.to_numpy(), "down"))
        last = max([("up", i) for i in up_x] + [("down", i) for i in dn_x],
                   key=lambda x: x[1], default=None)
        sm = smooth(osc[c]).reindex(o.index)
        board.append({
            "sector": c, "beta": _f(betas.get(c)) if betas else None,
            "osc": _f(sm.iloc[-1]), "slope": _f(sm.iloc[-1] - sm.iloc[-1 - SLOPE_DAYS]),
            "phase": cur_phase, "days": int(cur_len),
            "typical": typical.get(cur_phase) if cur_phase else None,
            "typical_all": typical,
            "last_cross": ({"dir": last[0], "date": str(o.index[last[1]].date()),
                            "days_ago": int(len(o) - 1 - last[1])} if last else None),
            "spark": [_f(v) for v in sm.tail(SPARK_DAYS).to_numpy()],
        })
    board.sort(key=lambda r: (PHASES[r["phase"]]["order"] if r["phase"] else 9, -(r["osc"] or 0)))

    pairs = []
    # Surrogates depend only on B, so each sector's set is made once and
    # reused for every pair it is the B of — 25 sets, not 325.
    by_b: dict[str, list[str]] = {}
    for a, b in itertools.combinations(names, 2):
        by_b.setdefault(b, []).append(a)
    for k, (b, partners) in enumerate(by_b.items()):
        made: dict[tuple, np.ndarray] = {}
        for a in partners:
            both = win[[a, b]].dropna()
            if len(both) < 2 * MIN_SHIFT + 20:
                continue
            x, y = both[a].to_numpy(), both[b].to_numpy()
            span = (both.index[0], len(both))
            if span not in made:
                made[span] = prepared(y, surrogates(y, seed=k))
            surr = made[span]
            lt = lag_test(x, y, surr)
            op = opposite_test(x, y, surr)
            pairs.append({"a": a, "b": b, "best_lag": lt["best_lag"],
                          "best_corr": _f(lt["best_corr"], 3), "corr0": _f(lt["corr0"], 3),
                          "p": _f(lt["p"], 5), "opposite": _f(op["share"], 3),
                          "opposite_expected": _f(op["expected"], 3)})
    for pr, q in zip(pairs, bh_q([p["p"] if p["p"] is not None else 1.0 for p in pairs])):
        pr["q"] = _f(q, 4)
    pairs.sort(key=lambda p: (p["best_corr"] if p["best_corr"] is not None else 9))

    return {
        "basis": basis, "start": str(win.index[0].date()), "end": str(win.index[-1].date()),
        "days": int(len(win)), "board": board, "pairs": pairs,
        "spark_dates": [str(d.date()) for d in win.index[-SPARK_DAYS:]],
        "phases": {k: v["label"] for k, v in PHASES.items()},
    }


def pair_detail(closes: dict[str, pd.Series], a: str, b: str, bench: pd.Series | None = None,
                basis: str = "abs", start: str = WINDOW_START) -> dict:
    if a == b:
        raise ValueError("两个板块不能相同")
    for s in (a, b):
        if s not in closes:
            raise LookupError(f"没有板块「{s}」")
    frame = build_frame({a: closes[a], b: closes[b]}, bench, basis, start)
    osc = frame.apply(oscillator)
    both = osc[osc.index >= pd.Timestamp(start)].dropna()
    if len(both) < 2 * MIN_SHIFT + 20:
        raise LookupError("共同的交易日太少，无法判断")
    x, y = both[a].to_numpy(), both[b].to_numpy()
    surr = prepared(y, surrogates(y, draws=PAIR_SURROGATES, seed=1))
    lt = lag_test(x, y, surr)
    op = opposite_test(x, y, surr)
    up_dn = follow_test(x, y, "up", "down", surr)
    dn_up = follow_test(x, y, "down", "up", surr)
    ph_a = phases(osc[a]).reindex(both.index)
    ph_b = phases(osc[b]).reindex(both.index)
    dates = [str(d.date()) for d in both.index]
    ux, dx = crosses(x, "up"), crosses(x, "down")
    uy, dy = crosses(y, "up"), crosses(y, "down")

    def last_cross(u, d):
        idx = [("up", i) for i in np.flatnonzero(u)] + [("down", i) for i in np.flatnonzero(d)]
        if not idx:
            return None
        k, i = max(idx, key=lambda t: t[1])
        return {"dir": k, "date": dates[i], "days_ago": int(len(dates) - 1 - i)}

    out = {
        "a": a, "b": b, "basis": basis, "dates": dates,
        "osc_a": [_f(v) for v in x], "osc_b": [_f(v) for v in y],
        "opposite_mask": [bool(v) for v in (np.sign(x) != np.sign(y))],
        "cross_a": {"up": np.flatnonzero(ux).tolist(), "down": np.flatnonzero(dx).tolist()},
        "cross_b": {"up": np.flatnonzero(uy).tolist(), "down": np.flatnonzero(dy).tolist()},
        "lag": {"best_lag": lt["best_lag"], "best_corr": _f(lt["best_corr"], 3),
                "corr0": _f(lt["corr0"], 3), "p": _f(lt["p"], 4),
                "null_median": _f(lt["null_median"], 3),
                "curve": [_f(v, 3) for v in lt["curve"]], "lags": list(range(-MAX_LAG, MAX_LAG + 1))},
        "opposite": {"share": _f(op["share"], 3), "expected": _f(op["expected"], 3), "p": _f(op["p"], 4)},
        "a_up_b_down": up_dn, "a_down_b_up": dn_up,
        "now": {
            "a": {"phase": ph_a.iloc[-1], "osc": _f(x[-1]), "last_cross": last_cross(ux, dx)},
            "b": {"phase": ph_b.iloc[-1], "osc": _f(y[-1]), "last_cross": last_cross(uy, dy)},
        },
    }
    out["conclusions"] = pair_words(out)
    return out


def pair_words(d: dict) -> list[str]:
    a, b = d["a"], d["b"]
    lab = {k: v["label"] for k, v in PHASES.items()}
    lines = []
    lg, op = d["lag"], d["opposite"]
    sig = lg["p"] is not None and lg["p"] < 0.05
    if lg["best_corr"] is not None and lg["best_corr"] < 0:
        who = (f"{b} 比 {a} 晚 {lg['best_lag']} 天" if lg["best_lag"] > 0 else
               f"{a} 比 {b} 晚 {-lg['best_lag']} 天" if lg["best_lag"] < 0 else "两者同步")
        lines.append(
            f"最强的反向联动：{who}，相关 {lg['best_corr']:.2f}（随机对照的中位数 {lg['null_median']:.2f}）"
            + (" —— 超出随机，是跷跷板。" if sig else
               f" —— 随机对照也有 {lg['p'] * 100:.0f}% 能这么负，不足以说是规律。"))
    else:
        lines.append(f"两者的趋势周期没有反向联动（最负的相关也只有 {lg['best_corr']}）。")
    if op["share"] is not None:
        lines.append(f"{op['share'] * 100:.0f}% 的交易日一个在均线上方、另一个在下方；随机对照一般是 "
                     f"{(op['expected'] or 0) * 100:.0f}%。")
    for key, x_dir, x_how, y_dir in (("a_up_b_down", "转升", "上穿均线", "转跌"),
                                     ("a_down_b_up", "转跌", "下穿均线", "转升")):
        f = d[key]
        if f["events"]:
            lines.append(
                f"{a}{x_dir}（{x_how}）{f['events']} 次，其中 {f['hits']} 次 {b} 在前后 {TURN_WINDOW} 天内{y_dir}"
                f"（随机约 {round((f['chance'] or 0) * f['events'])} 次）"
                + (f"，中位 {b} {'晚' if (f['median_offset'] or 0) >= 0 else '早'} {abs(f['median_offset'] or 0)} 天。"
                   if f["median_offset"] is not None else "。"))
    na, nb = d["now"]["a"], d["now"]["b"]

    def state(name, s):
        lc = s["last_cross"]
        tail = (f"，{lc['days_ago']} 天前{'上穿' if lc['dir'] == 'up' else '下穿'}均线" if lc else "")
        return f"{name} {lab.get(s['phase'], '—')}（{s['osc']:+.1f}σ{tail}）"
    lines.append("现在：" + state(a, na) + "；" + state(b, nb) + "。")
    return lines
