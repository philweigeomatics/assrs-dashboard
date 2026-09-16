"""
sector_affinity.py — how much this stock moves like each sector, and which
sector has been leading it.

Two questions, one computation:

  * 板块相关性 — rolling Pearson correlation between the stock's daily returns
    and each sector index's (PPI_<sector>) daily returns. High r means the
    stock trades like that basket, whether or not it belongs to it on paper.
  * 板块轮动 — for every day, WHICH sector had the highest correlation. Where
    that changes, the stock has switched themes. A stock that switches every
    other week has no sector identity, and its affinity numbers are noise.

沪深300 rides along as a benchmark: a stock correlated 0.8 with everything is
just correlated with the market, and the CSI 300 bar is how you see that.

Lives at the repository root, not under api/, so Streamlit pages can import it
too — the API only wraps it.
"""

from __future__ import annotations

import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
DEFAULT_WINDOW = 20

#: Trading days of rolling correlation returned to the client.
HISTORY_BARS = 252

CSI300_LABEL = "沪深300"
CSI300_TICKER = "000300.SH"
CSI300_COLOR = "#fbbf24"

#: Sector colours, assigned once so the lines, the strip and the legend agree.
PALETTE = (
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#64748b",
    "#0ea5e9", "#22c55e", "#eab308", "#f43f5e", "#a855f7",
)

#: Number of sector lines drawn in colour; the rest are grey context.
TOP_LINES = 5


def sector_members() -> dict[str, set[str]]:
    """{sector: {ticker, …}} — who is actually in each index."""
    import data_manager

    out: dict[str, set[str]] = {}
    for sector, members in data_manager.get_sector_stock_map().items():
        out[sector] = {str(m).split(".")[0] for m in (members or [])}
    return out


def load_sector_returns() -> dict[str, pd.Series]:
    """
    Daily returns of every sector index in the database, plus 沪深300.

    Independent of any one stock — cache it and correlate many tickers against
    the same dict. A sector whose table is missing, empty or too short to
    correlate is skipped rather than reported as zero correlation.
    """
    import data_manager

    out: dict[str, pd.Series] = {}
    for sector in data_manager.get_sector_stock_map():
        table = f"PPI_{sector}"
        try:
            if not data_manager.db.table_exists(table):
                continue
            df = data_manager.db.read_table(table, columns="Date,Close", order_by="Date")
            if df is None or df.empty:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            ret = df.set_index("Date").sort_index()["Close"].pct_change().dropna()
            if len(ret) > 10:
                out[sector] = ret
        except Exception:
            continue

    try:
        idx = data_manager.get_index_data_live(CSI300_TICKER, lookback_days=900)
        if idx is not None and not idx.empty:
            ret = idx["Close"].pct_change().dropna()
            if len(ret) > 10:
                out[CSI300_LABEL] = ret
    except Exception:
        pass

    return out


def _colors(names: list[str]) -> dict[str, str]:
    palette_names = [n for n in names if n != CSI300_LABEL]
    m = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(palette_names)}
    m[CSI300_LABEL] = CSI300_COLOR
    return m


def analyse(close: pd.Series, sector_returns: dict[str, pd.Series],
            window: int = DEFAULT_WINDOW, *, ticker: str | None = None,
            members: dict[str, set[str]] | None = None) -> dict:
    """
    Everything the 板块 panel draws, for one stock at one window length.

    `close` is the stock's close series indexed by date; `sector_returns` comes
    from load_sector_returns(). Raises LookupError when there is nothing to
    correlate — a missing sector table is a setup problem, not an answer.

    Pass `ticker` and `members` to mark the sectors the stock belongs to. The
    indices are market-cap weighted, so a large constituent correlates near 1.0
    with its own sector almost by construction — that is a different fact from
    an outsider that happens to trade like the basket, and the panel says which.
    """
    if window not in WINDOWS:
        raise LookupError(f"window must be one of {WINDOWS}")
    if not sector_returns:
        raise LookupError("没有找到板块指数（PPI_*）数据")

    stock_ret = close.pct_change().dropna()

    rolling: dict[str, pd.Series] = {}
    for name, sec_ret in sector_returns.items():
        aligned = pd.concat([stock_ret.rename("s"), sec_ret.rename("x")], axis=1).dropna()
        if len(aligned) < window + 5:
            continue
        rolling[name] = (aligned["s"].rolling(window).corr(aligned["x"])
                         .reindex(stock_ret.index))

    if not rolling:
        raise LookupError("与板块指数重叠的交易日不足，无法计算相关性")

    frame = pd.DataFrame(rolling).dropna(how="all").tail(HISTORY_BARS)
    if frame.empty:
        raise LookupError("滚动窗口内没有有效的相关系数")

    filled = frame.ffill()

    # 沪深300 is drawn and ranked in the bar list, but never competes as a
    # sector: it is the market. Letting it into the "which sector leads today"
    # comparison would credit rotations to the index itself and inflate the
    # count of sectors that have led.
    sector_cols = [c for c in filled.columns if c != CSI300_LABEL]
    if not sector_cols:
        raise LookupError("没有可用的板块指数（只有基准指数）")
    themes = filled[sector_cols]

    # Mean over the window shown, NOT the last value: which sector this stock
    # has been tracking lately is a steadier question than where it closed.
    mean_r = themes.mean().sort_values(ascending=False)
    top_names = mean_r.head(TOP_LINES).index.tolist()
    colors = _colors(frame.columns.tolist())

    tails = ((c, frame[c].dropna()) for c in frame.columns)
    current = {c: float(v.iloc[-1]) for c, v in tails if not v.empty}

    owned = {s for s, t in (members or {}).items() if ticker and ticker in t}

    sectors = [{
        "name": name,
        "color": colors[name],
        "benchmark": name == CSI300_LABEL,
        "member": name in owned,
        "top": name in top_names,
        "r": round(current[name], 4) if name in current else None,
        "mean_r": round(float(filled[name].mean()), 4),
        "series": [None if pd.isna(v) else round(float(v), 4) for v in filled[name]],
    } for name in frame.columns]
    sectors.sort(key=lambda s: (s["r"] is None, -(s["r"] or 0)))

    return {
        "window": window,
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "sectors": sectors,
        "dominant": _runs(themes),
        "summary": _summary(themes, current, mean_r, owned),
    }


def _runs(filled: pd.DataFrame) -> list[dict]:
    """
    The leading sector per day, run-length encoded into [from, to] bar spans.

    Sent as spans rather than one value per bar because that is what the strip
    draws, and because the number of spans IS the rotation count.
    """
    if filled.shape[1] < 2:
        return []
    dominant = filled.idxmax(axis=1).dropna()
    if dominant.empty:
        return []

    pos = {d: i for i, d in enumerate(filled.index)}
    runs: list[dict] = []
    start = pos[dominant.index[0]]
    prev = dominant.iloc[0]
    for date, name in dominant.items():
        if name != prev:
            runs.append({"from": start, "to": pos[date] - 1, "sector": prev})
            start, prev = pos[date], name
    runs.append({"from": start, "to": pos[dominant.index[-1]], "sector": prev})
    return runs


def _summary(filled: pd.DataFrame, current: dict[str, float],
             mean_r: pd.Series, owned: set[str]) -> dict:
    top = mean_r.index[0]
    series = filled[top].dropna()
    r5, r20 = series.tail(5).mean(), series.tail(20).mean()
    trend = ("strengthening" if r5 > r20 + 0.05
             else "weakening" if r5 < r20 - 0.05 else "stable")

    dominant = filled.idxmax(axis=1).dropna() if filled.shape[1] >= 2 else pd.Series(dtype=object)
    rotations = int((dominant != dominant.shift(1)).sum()) - 1 if not dominant.empty else 0
    counts = dominant.value_counts() if not dominant.empty else pd.Series(dtype=int)
    n = max(len(dominant), 1)

    # Thresholds from the Streamlit page: >40 switches in 252 sessions is a
    # stock changing theme roughly every six days — affinity means little then.
    verdict = ("high" if rotations > 40 else "low" if rotations < 8 else "moderate")

    top_r = float(current.get(top, mean_r[top]))
    return {
        "top": top,
        "top_r": round(top_r, 4),
        "top_is_member": top in owned,
        # A heavy constituent tracking its own index is arithmetic, not a
        # discovery. Say so rather than letting r = 0.99 read as insight.
        "self_index": top in owned and top_r >= 0.95,
        "trend": trend,
        "r5": round(float(r5), 4),
        "r20": round(float(r20), 4),
        "rotations": rotations,
        "verdict": verdict,
        "leaders": [{"sector": s, "days": int(c), "pct": round(c / n * 100, 1)}
                    for s, c in counts.head(5).items()],
        "n_leaders": int(dominant.nunique()) if not dominant.empty else 0,
        "sessions": int(len(filled)),
    }
