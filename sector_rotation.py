"""
sector_rotation.py — which sector money is leaving, and which it is entering.

The old dashboard answered this with rolling CORRELATION between four
hand-picked sector pairs. That can never work, for a reason worth stating
plainly: correlation is symmetric and unsigned. corr(A, B) = corr(B, A), and it
is identical whether A soared while B crawled or the reverse. It tells you
whether two sectors moved together — never which one money went to. So the old
panel could say "rotation is happening" and was structurally incapable of
saying rotation out of WHAT, into WHAT.

What does carry direction is relative strength: each sector priced in units of
the market. If 半导体/市场 is rising, money is moving into semiconductors,
whatever the absolute tape does. And the useful part is the SECOND derivative —
relative strength *momentum* turns before relative strength itself does, which
is what makes an early call possible at all.

That is a Relative Rotation Graph (JdK RS-Ratio / RS-Momentum, Julius de Kempenaer).
Two numbers per sector, both centred on 100:

    RS-Ratio      how far the sector's relative line sits above its own trend
                  → is it outperforming the market NOW
    RS-Momentum   how fast RS-Ratio is itself changing
                  → is that outperformance building or decaying

Plot one against the other and every sector sits in one of four quadrants,
which it traverses CLOCKWISE:

        RS-Mom
          ▲
   改善   │   领先          改善 Improving  — lagging, but momentum has turned
 (买入观察)│ (持有)                          this is where money is ARRIVING
    ──────┼──────▶ RS-Ratio  领先 Leading   — outperforming and still building
   落后   │   走弱          走弱 Weakening  — still ahead, but momentum is gone
 (回避)   │ (减仓预警)                       this is where money is LEAVING
                            落后 Lagging    — out of favour and still falling

So "rotating out of" = 走弱, "rotating into" = 改善. Not an interpretation of a
correlation number — a position on a plane, with a heading.

Two things keep this from being astrology:

  * Cross-sectional normalisation. Both axes are standardised ACROSS sectors on
    each date, not against each sector's own history. Rotation is zero-sum —
    money entering one sector left another — so the question is always
    "compared with the other sectors today", and in a market-wide melt-up a
    per-sector normalisation would show everything Leading at once.

  * A measured edge, not a picture. `analyse()` returns what each quadrant has
    ACTUALLY been worth in THIS market's history: mean forward excess return
    over the benchmark, by quadrant, with sample sizes. If 改善 has not paid in
    A-shares, the panel says so. A rotation map that cannot be wrong is
    decoration.

The second column, when the market_breadth table is available, is the sector's
ABSOLUTE trend — see _trend_deltas for what that table really contains, which
is not what its name says. Relative and absolute are different questions and
routinely disagree: in a falling market the sector entering 改善 is often the
one falling least, which is a defensive rotation, not a bull one. Worth seeing
side by side; never worth averaging into one score.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BENCHMARK_TABLE = "MARKET_PROXY"
CSI300_TICKER = "000300.SH"
CSI300_LABEL = "沪深300"

#: The four quadrants, in clockwise order of the rotation they describe.
QUADRANTS = {
    "improving":  {"label": "改善", "en": "Improving",  "color": "#0ea5e9",
                   "means": "落后但动能已转正 — 资金正在流入"},
    "leading":    {"label": "领先", "en": "Leading",    "color": "#d70015",
                   "means": "跑赢且动能仍在增强 — 持有"},
    "weakening":  {"label": "走弱", "en": "Weakening",  "color": "#f59e0b",
                   "means": "仍然跑赢但动能消退 — 资金正在流出"},
    "lagging":    {"label": "落后", "en": "Lagging",    "color": "#1f7a35",
                   "means": "跑输且仍在走弱 — 回避"},
}

#: freq → (rs_window, mom_window, tail, horizon) in bars of that frequency.
#:
#: Weekly is the default because the complaint that started this module was
#: that rotation is hard to SEE. On daily bars a sector crosses a quadrant
#: boundary and crosses back within a week; the signal is real but it is buried
#: under two days of noise. Weekly bars are the standard remedy and cost
#: nothing but recency.
PRESETS = {
    "w": {"rs_window": 14, "mom_window": 5,  "tail": 8,  "horizon": 4,  "days": 5},
    "d": {"rs_window": 60, "mom_window": 20, "tail": 20, "horizon": 20, "days": 1},
}

#: Fewer sectors than this and a cross-sectional z-score is meaningless.
MIN_SECTORS = 4

#: How far from (100, 100) a sector must sit before its quadrant is called
#: rather than described as sitting on the line. In z units × 2 (see _z).
NEUTRAL_RADIUS = 0.6


# ── loading ──────────────────────────────────────────────────────────────────
def load_sector_closes() -> tuple[dict[str, pd.Series], pd.Series, str]:
    """
    ({sector: close series}, benchmark close, benchmark label) from the DB.

    The benchmark is PPI_MARKET_PROXY when it exists, because it is built the
    same cap-weighted way as the sector indices — an RRG against a differently
    constructed index measures the construction as much as the rotation. 沪深300
    is the fallback, and the label says which was used.
    """
    import data_manager

    closes: dict[str, pd.Series] = {}
    for sector in data_manager.get_sector_stock_map():
        table = f"PPI_{sector}"
        try:
            if not data_manager.db.table_exists(table):
                continue
            df = data_manager.db.read_table(table, columns="Date,Close", order_by="Date")
            if df is None or df.empty:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            s = df.dropna(subset=["Date"]).set_index("Date").sort_index()["Close"]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if len(s) > 30:
                closes[sector] = s
        except Exception:
            continue

    if not closes:
        raise LookupError("没有找到任何板块指数（PPI_*）")

    bench = _proxy_benchmark(closes)
    label = "全市场（PPI_MARKET_PROXY）"
    if bench is None:
        try:
            idx = data_manager.get_index_data_live(CSI300_TICKER, lookback_days=1200)
            bench = idx["Close"] if idx is not None and not idx.empty else None
            label = CSI300_LABEL
        except Exception:
            bench = None
    if bench is None or bench.empty:
        raise LookupError("没有可用的市场基准（PPI_MARKET_PROXY 或 沪深300）")
    return closes, bench, label


#: A benchmark this far behind the sectors it is dividing is not a benchmark.
BENCHMARK_STALE_DAYS = 10


def _proxy_benchmark(closes: dict[str, pd.Series]) -> pd.Series | None:
    """
    PPI_MARKET_PROXY, but only if it is current.

    It lives in the same table family as the sectors but is not always in the
    sector map, and at the time of writing it had stopped updating seven months
    before the sector indices did. A stale denominator does not fail loudly —
    it turns every sector into a rising relative line and reports the whole
    market as 领先. Better to notice and fall back to 沪深300.
    """
    import data_manager

    closes.pop(BENCHMARK_TABLE, None)
    try:
        if not data_manager.db.table_exists(f"PPI_{BENCHMARK_TABLE}"):
            return None
        df = data_manager.db.read_table(f"PPI_{BENCHMARK_TABLE}",
                                        columns="Date,Close", order_by="Date")
        if df is None or df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        s = pd.to_numeric(df.dropna(subset=["Date"]).set_index("Date")
                          .sort_index()["Close"], errors="coerce").dropna()
    except Exception:
        return None
    if len(s) < 120:
        return None

    newest = max(c.index.max() for c in closes.values())
    if (newest - s.index.max()).days > BENCHMARK_STALE_DAYS:
        print(f"[rotation] PPI_MARKET_PROXY 停更于 {s.index.max():%Y-%m-%d}，改用 {CSI300_LABEL}")
        return None
    return s


def load_trend() -> pd.DataFrame | None:
    """
    The market_breadth table: {Date × sector}, 0–1 per cell.

    Named "breadth" throughout this codebase and in the Streamlit page, which
    it is not. main.calculate_ppi_breadth_proxy builds each cell from the
    SECTOR INDEX's distance from its own MA20, mapped linearly from ±5% onto
    0–1 and clipped — so it saturates, which is why most cells read 0 or 1, and
    it counts no constituents at all.

    That makes it an absolute-trend flag, not corroboration: a genuine breadth
    series would count members and could therefore contradict a cap-weighted
    index dragged by two names. This cannot. It is still worth showing next to
    the relative numbers — above or below the 20-day mean in absolute terms is
    a different fact from ahead of or behind the market — but calling it
    breadth would be claiming an independence it does not have.
    """
    import data_manager
    try:
        df = data_manager.load_market_breadth_from_db()
    except Exception:
        return None
    return None if df is None or df.empty else df


# ── the computation ──────────────────────────────────────────────────────────
def _z(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise across sectors on each date, centred on 100.

    Row-wise, not column-wise: see the module docstring. ddof=0 because this is
    the whole sector universe on that date, not a sample from it. A row whose
    sectors all moved identically has no spread to normalise, and becomes
    exactly 100 everywhere rather than ±inf.
    """
    mu = frame.mean(axis=1)
    sd = frame.std(axis=1, ddof=0)
    out = frame.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0).fillna(0.0)
    return 100.0 + 2.0 * out


def _quadrant(ratio: float, mom: float) -> str:
    if ratio >= 100:
        return "leading" if mom >= 100 else "weakening"
    return "improving" if mom >= 100 else "lagging"


def _heading(rr: pd.Series, mm: pd.Series) -> float | None:
    """
    Compass bearing of the last leg, in degrees clockwise from north.

    A bearing rather than a raw dx/dy because the whole point of an RRG is that
    the rotation is clockwise: a sector in 领先 heading ~135° (south-east) is
    about to cross into 走弱, which is the sell warning. One number the client
    can draw as an arrow and a person can read as a direction.

    Computed from the FULL-precision series, never from the rounded tail sent
    to the client: two decimals is plenty to draw a dot and nowhere near enough
    to take a difference of two adjacent ones — a smooth leg rounds to zero and
    the arrow silently disappears.
    """
    if len(rr) < 2:
        return None
    dx = float(rr.iloc[-1] - rr.iloc[-2])
    dy = float(mm.iloc[-1] - mm.iloc[-2])
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _edge(ratio: pd.DataFrame, mom: pd.DataFrame, rel: pd.DataFrame,
          horizon: int) -> dict:
    """
    What each quadrant has been worth, measured on this market's own history.

    For every (date, sector) with a defined quadrant, the forward excess return
    over the benchmark `horizon` bars later. Grouped by quadrant. `rel` is the
    sector's relative line (sector / benchmark), so its forward change IS the
    excess return — no second series and no compounding error.

    Reported as an edge over the all-quadrant mean rather than as a raw number,
    for the same reason the prediction scorecard does it: in a market where
    every sector drifted up 3%, "改善 returned +3%" is not a finding.
    """
    fwd = rel.shift(-horizon) / rel - 1.0
    rows: dict[str, list[float]] = {k: [] for k in QUADRANTS}
    for col in ratio.columns:
        q = pd.DataFrame({"r": ratio[col], "m": mom[col], "f": fwd[col]}).dropna()
        for _, row in q.iterrows():
            rows[_quadrant(row["r"], row["m"])].append(float(row["f"]))

    allv = [v for lst in rows.values() for v in lst]
    base = float(np.mean(allv)) if allv else 0.0
    out = []
    for key, vals in rows.items():
        if not vals:
            out.append({"quadrant": key, "n": 0, "mean_pct": None,
                        "edge_pp": None, "win_pct": None})
            continue
        arr = np.asarray(vals, dtype=float)
        out.append({
            "quadrant": key,
            "n": int(arr.size),
            "mean_pct": round(float(arr.mean()) * 100, 2),
            "edge_pp": round(float(arr.mean() - base) * 100, 2),
            "win_pct": round(float((arr > 0).mean()) * 100, 1),
        })
    out.sort(key=lambda r: (r["edge_pp"] is None, -(r["edge_pp"] or 0)))
    return {"horizon": horizon, "baseline_pct": round(base * 100, 2), "rows": out}


def analyse(sector_closes: dict[str, pd.Series], benchmark: pd.Series, *,
            freq: str = "w", benchmark_label: str = "市场",
            trend: pd.DataFrame | None = None) -> dict:
    """
    The whole rotation map, for one frequency.

    Raises LookupError when there is not enough to say anything — too few
    sectors for a cross-sectional score, or too little overlapping history for
    the rolling windows. A rotation map computed from three sectors and forty
    bars is worse than no rotation map.
    """
    if freq not in PRESETS:
        raise LookupError(f"freq must be one of {sorted(PRESETS)}")
    p = PRESETS[freq]
    if len(sector_closes) < MIN_SECTORS:
        raise LookupError(f"至少需要 {MIN_SECTORS} 个板块指数才能做横截面比较，"
                          f"当前只有 {len(sector_closes)} 个")

    panel = pd.DataFrame(sector_closes).join(benchmark.rename("__bench__"), how="inner")
    panel = panel.dropna(subset=["__bench__"]).sort_index()
    if freq == "w":
        # Last print of each week. Not a mean: an RRG reads levels, and a
        # weekly mean of a relative line is not any week's relative strength.
        #
        # Relabelled with the real session date afterwards, because resample
        # stamps each bin with its RIGHT EDGE — on a Thursday the current bar
        # would be dated to the coming Friday, and a dashboard that reports an
        # "as of" date in the future is reporting a bug.
        stamps = pd.Series(panel.index, index=panel.index).resample("W-FRI").last()
        panel = panel.resample("W-FRI").last().dropna(how="all")
        panel.index = pd.DatetimeIndex(stamps.reindex(panel.index).to_numpy())

    need = p["rs_window"] + p["mom_window"] + p["tail"]
    if len(panel) < need:
        raise LookupError(f"历史数据不足：需要 {need} 根{'周' if freq == 'w' else '日'}线，"
                          f"只有 {len(panel)} 根")

    bench = panel.pop("__bench__")
    # Drop sectors whose history does not cover the windows — a column that is
    # NaN for most of the panel would otherwise drag the cross-sectional mean.
    panel = panel.loc[:, panel.notna().sum() >= need]
    if panel.shape[1] < MIN_SECTORS:
        raise LookupError(f"只有 {panel.shape[1]} 个板块有足够长的历史")
    panel = panel.ffill()

    rel = panel.div(bench, axis=0) * 100.0                    # the relative line
    ratio_raw = rel / rel.rolling(p["rs_window"]).mean() * 100.0
    mom_raw = ratio_raw / ratio_raw.rolling(p["mom_window"]).mean() * 100.0

    ratio = _z(ratio_raw.dropna(how="all"))
    mom = _z(mom_raw.dropna(how="all"))
    common = ratio.index.intersection(mom.index)
    ratio, mom = ratio.loc[common], mom.loc[common]
    if len(common) < p["tail"]:
        raise LookupError("滚动窗口之后剩下的历史不足以画出轨迹")

    dates = [d.strftime("%Y-%m-%d") for d in common]
    tail_idx = common[-p["tail"]:]

    tr = _trend_deltas(trend, panel.columns, p["days"] * p["horizon"])

    sectors = []
    for name in panel.columns:
        rr, mm = ratio.loc[tail_idx, name], mom.loc[tail_idx, name]
        tail = [{"date": d.strftime("%Y-%m-%d"),
                 "ratio": round(float(rr.at[d]), 2),
                 "mom": round(float(mm.at[d]), 2)}
                for d in tail_idx]
        r, m = tail[-1]["ratio"], tail[-1]["mom"]
        dist = math.hypot(r - 100.0, m - 100.0)
        sectors.append({
            "name": name,
            "ratio": r,
            "mom": m,
            "quadrant": _quadrant(r, m),
            # Near the crossing point the quadrant label is a coin flip; say so
            # rather than letting 100.02 read as "Leading".
            "neutral": dist < NEUTRAL_RADIUS,
            "distance": round(dist, 2),
            "heading": (lambda h: None if h is None else round(h, 1))(_heading(rr, mm)),
            "tail": tail,
            "trend": tr.get(name),
        })
    sectors.sort(key=lambda s: (-s["mom"], -s["ratio"]))

    return {
        "freq": freq,
        "benchmark": benchmark_label,
        "asof": dates[-1],
        "bars": len(common),
        "dates": dates[-p["tail"]:],
        "params": {k: p[k] for k in ("rs_window", "mom_window", "tail", "horizon")},
        "quadrants": QUADRANTS,
        "sectors": sectors,
        "calls": _calls(sectors),
        "edge": _edge(ratio, mom, rel.loc[common], p["horizon"]),
    }


def _trend_deltas(trend: pd.DataFrame | None, names, days: int) -> dict:
    """
    {sector: {now_pct, delta_pp}} — the absolute-trend score and its change.

    Reads the market_breadth table; see load_trend for what is actually in it.
    0 means the sector index is 5% or more BELOW its own 20-day mean, 100 means
    5% or more above, and the middle is linear. `days` is in trading days, to
    match the map's own horizon.
    """
    if trend is None or trend.empty:
        return {}
    out = {}
    for name in names:
        if name not in trend.columns:
            continue
        s = pd.to_numeric(trend[name], errors="coerce").dropna()
        if s.empty:
            continue
        now = float(s.iloc[-1])
        prev = float(s.iloc[-min(days + 1, len(s))])
        out[name] = {"now_pct": round(now * 100, 1),
                     "delta_pp": round((now - prev) * 100, 1)}
    return out


def _calls(sectors: list[dict]) -> dict:
    """
    The two lists the page exists to show: leaving, and arriving.

    Membership is the quadrant; the ORDER is momentum, because momentum is what
    moves a sector between quadrants and therefore what makes one 改善 sector a
    better candidate than another. Sectors sitting on the crossing point are
    excluded from both — being 0.1 inside a quadrant is not a call.

    `rising` is whether the absolute trend is moving the same way the relative
    signal claims. It is deliberately NOT part of the ranking: relative and
    absolute answer different questions, and folding one into the other would
    hide the case worth seeing — a sector gaining on a falling market while
    still falling itself.
    """
    def pick(q: str, reverse: bool) -> list[dict]:
        rows = [s for s in sectors if s["quadrant"] == q and not s["neutral"]]
        rows.sort(key=lambda s: s["mom"], reverse=reverse)
        return [{
            "name": s["name"],
            "ratio": s["ratio"],
            "mom": s["mom"],
            "heading": s["heading"],
            "trend": s["trend"],
            "rising": _agrees(s, up=reverse),
        } for s in rows]

    return {
        # Arriving: momentum already positive, ranked strongest first.
        "into": pick("improving", True),
        # Leaving: still ahead on price but momentum gone, weakest first —
        # that is the one furthest along the way out.
        "outof": pick("weakening", False),
        "leading": [s["name"] for s in sectors
                    if s["quadrant"] == "leading" and not s["neutral"]],
        "lagging": [s["name"] for s in sectors
                    if s["quadrant"] == "lagging" and not s["neutral"]],
    }


def _agrees(s: dict, *, up: bool) -> bool | None:
    """True when the absolute trend moved the way the relative signal claims."""
    t = s.get("trend")
    if not t or t.get("delta_pp") is None:
        return None
    return t["delta_pp"] > 0 if up else t["delta_pp"] < 0
