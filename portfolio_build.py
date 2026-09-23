"""
portfolio_build.py — turn a list of tickers into an allocation you can defend.

The optimiser itself lives in optimise.py and is shared with the Questrade
book. What this adds is everything around it: fetching prices for whichever
market the names belong to, aligning them, building the efficient frontier
and the correlation matrix, and simulating the result against a benchmark.

ONE MARKET PER PORTFOLIO, ENFORCED

A-shares and US stocks trade on different calendars. Optimising across both
means intersecting two session lists, and the overlap silently drops every
A-share holiday and every US holiday — a year of history becomes ten months,
and the covariance is measured on a sample neither market actually saw. It
is solvable with currency-converted, calendar-aligned series; it is not
solvable by accident. So a portfolio is A-shares or it is US names, and the
refusal is here rather than in the screen that calls it.

THE BENCHMARK IS THE POINT OF THE SIMULATION

An optimiser will always produce a portfolio, and on the data it was fitted
to that portfolio will always look good. The equity curve is drawn against
沪深300 or the S&P for the same reason the pair screen has a holdout: a
number with nothing beside it cannot be judged. Equal weight is drawn too,
because beating it is harder than it sounds and most methods do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import optimise

#: Sessions of history the covariance is measured over. 242 ≈ one A-share
#: trading year, which is what the Streamlit page used and a reasonable
#: default: shorter is noisier, longer averages across regimes.
DEFAULT_LOOKBACK = 242

#: Points on the efficient frontier. Enough to draw a curve, few enough that
#: fifty constrained solves do not dominate the request.
FRONTIER_POINTS = 24

#: Below this many overlapping sessions a covariance matrix is decoration.
MIN_SESSIONS = 60

#: What each market is measured against by default.
BENCHMARKS = {
    "CN": ("000300.SH", "沪深300"),
    "US": ("^GSPC", "标普500"),
    "CA": ("^GSPTSE", "标普/TSX"),
}


def market_of(symbols: list[str]) -> str:
    """
    The one market these names belong to, or a refusal.

    See the module docstring: mixing calendars is not a thing to do by
    accident, so it is refused rather than silently intersected.
    """
    import markets

    found = {markets.split(markets.canonical(s))[0] for s in symbols}
    if not found:
        raise LookupError("没有可用的标的")
    if len(found) > 1:
        names = "、".join(sorted(found))
        raise LookupError(
            f"一个组合只能是同一个市场的股票（这里有 {names}）—— "
            f"不同市场交易日历不同，混在一起算出来的协方差是两边都没经历过的样本")
    return found.pop()


def load_closes(symbols: list[str], lookback: int) -> pd.DataFrame:
    """
    Aligned closing prices for every symbol, newest `lookback` sessions.

    Per-column pruning before the row-wise dropna: one unpriceable ticker
    with an all-NaN column would otherwise take every row with it and leave
    "0 sessions available", which is a bug that reads like a data outage.
    """
    import markets

    frames: dict[str, pd.Series] = {}
    missing: list[str] = []
    for raw in symbols:
        sym = markets.canonical(raw)
        try:
            df = markets.get(sym).fetch_ohlcv(sym)
        except Exception:                                          # noqa: BLE001
            df = None
        if df is None or df.empty or "Close" not in df:
            missing.append(sym)
            continue
        frames[sym] = df["Close"].rename(sym)

    if not frames:
        raise RuntimeError("行情读取失败 — 请稍后重试")

    px = pd.concat(frames.values(), axis=1)
    px = px.dropna(axis=1, how="all").ffill().dropna(how="any")
    if len(px) > lookback:
        px = px.iloc[-lookback:]
    if len(px) < MIN_SESSIONS:
        raise LookupError(
            f"共同交易日只有 {len(px)} 天，少于 {MIN_SESSIONS} 天，"
            f"协方差矩阵没有意义")

    px.attrs["missing"] = missing
    return px


def returns_of(px: pd.DataFrame, duration: int = 1) -> pd.DataFrame:
    """
    Simple returns over `duration` sessions.

    Longer durations smooth the noise that makes a daily covariance matrix
    unstable. They also overlap, so the effective sample is smaller than the
    row count suggests — which is why the caller reports both.
    """
    if duration <= 1:
        return px.pct_change(fill_method=None).dropna(how="any")
    return px.pct_change(duration, fill_method=None).dropna(how="any")


def frontier(rets: pd.DataFrame, *, cap: float,
             points: int = FRONTIER_POINTS) -> list[dict]:
    """
    Minimum variance at each of a range of target returns.

    Drawn to show the shape of the trade-off, not to be picked from: the
    return axis is a historical mean, and historical means do not repeat.
    """
    from scipy.optimize import minimize

    n = rets.shape[1]
    if n < 2:
        return []
    cov = optimise._cov(rets)
    mu = rets.mean().to_numpy() * optimise.TRADING_DAYS
    hi = float(min(mu.max(), np.percentile(mu, 97)))
    lo = float(max(mu.min(), np.percentile(mu, 3)))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return []

    eff = optimise.effective_cap(cap, n)
    out = []
    for target in np.linspace(lo, hi, points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: float(w @ cov @ w), np.full(n, 1.0 / n),
                       method="SLSQP", bounds=[(0.0, eff)] * n,
                       constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9})
        if not res.success:
            continue
        w = np.clip(res.x, 0, None)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        vol = float(np.sqrt(w @ cov @ w) * np.sqrt(optimise.TRADING_DAYS))
        out.append({"vol_pct": round(vol * 100, 2),
                    "ret_pct": round(float(w @ mu) * 100, 2)})
    # Deduplicate the flat tail the solver produces at infeasible targets.
    seen, clean = set(), []
    for p in out:
        key = (p["vol_pct"], p["ret_pct"])
        if key not in seen:
            seen.add(key)
            clean.append(p)
    return clean


def prune(w: pd.Series) -> pd.Series:
    """
    Drop dust weights, then put back what dropping them removed.

    A 0.3% allocation is a rounding artefact with a commission attached, so
    it goes. But dropping without renormalising leaves a book that adds up to
    99.4%, and every number derived from it — the stats, the curve, the
    amounts an investor would actually place — is quietly scaled down by the
    difference. Separated out because a solver usually sends dust to exactly
    zero, so the renormalisation is unreachable from most fixtures and only
    matters on the day it is not.
    """
    kept = w[w >= optimise.MIN_WEIGHT]
    total = float(kept.sum())
    return kept / total if total > 0 else kept


def correlation(rets: pd.DataFrame) -> dict:
    """The correlation matrix, as labels plus rows, ready to draw."""
    c = rets.corr()
    return {
        "labels": list(c.columns),
        "rows": [[None if v != v else round(float(v), 3) for v in row]
                 for row in c.to_numpy()],
    }


def curve(px: pd.DataFrame, w: pd.Series) -> list[float]:
    """Cumulative return of a fixed-weight portfolio, rebased to 0%."""
    rets = px.pct_change(fill_method=None).fillna(0.0)
    aligned = w.reindex(px.columns).fillna(0.0)
    port = (rets * aligned).sum(axis=1)
    return [round(float(v) * 100, 3) for v in ((1 + port).cumprod() - 1)]


def benchmark_closes(market: str, index: pd.Index) -> tuple[pd.Series | None, str]:
    """
    The benchmark's closes on the portfolio's own dates, and its name.

    Reindexed onto the portfolio's sessions rather than the other way round:
    the portfolio is the thing being judged, and a benchmark that is missing
    a day should not shorten it.
    """
    code, label = BENCHMARKS.get(market, (None, ""))
    if not code:
        return None, ""

    try:
        if market == "CN":
            import data_manager
            df = data_manager.get_index_data_live(code, lookback_days=1000)
            s = None if df is None or df.empty else df["Close"]
        else:
            import yfinance as yf
            raw = yf.Ticker(code).history(period="5y", auto_adjust=True,
                                          raise_errors=False)
            s = None if raw is None or raw.empty else raw["Close"]
    except Exception as exc:                                       # noqa: BLE001
        print(f"[portfolio_build] benchmark {code}: {type(exc).__name__}: {exc}")
        return None, label
    if s is None or s.empty:
        return None, label

    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.reindex(index.tz_localize(None) if index.tz else index).ffill(), label


def build(symbols: list[str], *, method: str = "min_var",
          cap: float = optimise.DEFAULT_CAP,
          lookback: int = DEFAULT_LOOKBACK,
          duration: int = 1, rf_annual: float = 0.0) -> dict:
    """
    An allocation, the evidence around it, and what it would have done.

    Everything the screen shows comes from here so the numbers on it cannot
    disagree with each other: the weights, the stats those weights produce,
    the frontier they sit on, and the curve they would have traced.
    """
    import markets

    syms = list(dict.fromkeys(markets.canonical(s) for s in symbols))
    if len(syms) < 2:
        raise LookupError("至少需要两只股票")
    market = market_of(syms)

    px = load_closes(syms, lookback)
    rets = returns_of(px, duration)
    if rets.empty:
        raise LookupError("可用收益率数据不足")

    w = prune(optimise.weights(rets, method, cap=cap, rf_annual=rf_annual))

    names = _names(list(px.columns), market)
    equal = pd.Series(1.0 / px.shape[1], index=px.columns)
    bench, bench_label = benchmark_closes(market, px.index)

    daily = px.pct_change(fill_method=None).dropna(how="any")
    out = {
        "market": market,
        "method": method,
        "method_label": optimise.METHODS[method]["label"],
        "cap_pct": round(optimise.effective_cap(cap, px.shape[1]) * 100, 1),
        "cap_asked_pct": round(cap * 100, 1),
        "lookback": int(len(px)),
        "duration": int(duration),
        "from": str(px.index[0].date()), "to": str(px.index[-1].date()),
        "missing": px.attrs.get("missing", []),
        "holdings": [{
            "t": t, "n": names.get(t, t),
            "weight_pct": round(float(w.get(t, 0.0)) * 100, 2),
        } for t in px.columns],
        "stats": optimise.stats(daily, w),
        "equal_stats": optimise.stats(daily, equal),
        "frontier": frontier(rets, cap=cap),
        "correlation": correlation(rets),
        "dates": [str(d.date()) for d in px.index],
        "curve": curve(px, w),
        "equal_curve": curve(px, equal),
    }

    if bench is not None and bench.notna().any():
        base = bench.dropna()
        rebased = (bench / base.iloc[0] - 1) * 100
        out["benchmark"] = {
            "label": bench_label,
            "curve": [None if v != v else round(float(v), 3) for v in rebased],
        }
    else:
        out["benchmark"] = None
    return out


def _names(symbols: list[str], market: str) -> dict[str, str]:
    """Company names, so a weight table is readable without a lookup."""
    if market != "CN":
        return {s: s.split(":", 1)[-1] for s in symbols}
    try:
        import data_manager
        rows = data_manager.get_all_stock_basic() or []
        found = {r["ticker"]: r["name"] for r in rows}
        return {s: found.get(s, s) for s in symbols}
    except Exception:                                              # noqa: BLE001
        return {s: s for s in symbols}
