"""
portfolio_build.py — mean-variance optimisation, ported from the Streamlit page.

This is a port, not a reinterpretation. The optimiser, the frontier, the risk
metrics and the 100-point assessment are the ones from
pages/Portfolio_Optimization.py, with the same thresholds and the same
annualisation, so a portfolio built here gets the answer it got there.

WEIGHTS COME OFF THE FRONTIER

Two modes, exactly as the page has them:

    target_return is None   maximise the Sharpe ratio
    target_return is set    minimise variance subject to reaching that return

The second is the point of drawing a frontier at all. The curve is not
decoration beside a fixed answer — it is the menu, and choosing a point on it
re-solves for the weights that reach it with the least variance.

ANNUALISATION IS 242, NOT 252

A-shares trade about 242 sessions a year and the page used 242 throughout.
Keeping it means the numbers match; changing it would move every Sharpe ratio
by 2% for no reason a reader could see.

RETURN DURATION IS FOR THE OPTIMISER ONLY

`duration > 1` builds overlapping N-day returns, which smooths the covariance.
They overlap heavily, so they are right for the optimiser and wrong for
anything that compounds — the simulation and the daily risk metrics always use
period-1 returns. That distinction is documented in the page and is the
easiest thing to lose in a port.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A-share trading days per year. The Streamlit page used 242; so does this.
TRADING_DAYS = 242

#: Sessions of history the covariance is measured over.
DEFAULT_LOOKBACK = 242

#: Points on the frontier. The page used 50; 40 draws the same curve and
#: keeps the request to a couple of seconds.
FRONTIER_POINTS = 40

#: Below this many overlapping sessions a covariance matrix is decoration.
MIN_SESSIONS = 60

#: Weights below this are reported as zero — a 0.3% allocation is a rounding
#: artefact with a commission attached.
MIN_WEIGHT = 0.005
#: Names are a nicety — never let them hold up an optimisation.
NAME_TIMEOUT_S = 12.0

#: What each market is measured against by default.
BENCHMARKS = {
    "CN": ("000300.SH", "沪深300"),
    "US": ("^GSPC", "标普500"),
    "CA": ("^GSPTSE", "标普/TSX"),
}


# ── universe ─────────────────────────────────────────────────────────────────
def market_of(symbols: list[str]) -> str:
    """
    The one market these names belong to, or a refusal.

    A-shares and US names trade on different calendars, so optimising across
    both intersects two session lists and drops every holiday on each side.
    The covariance then describes a sample neither market traded.
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
    Aligned closes, with per-column pruning before the row-wise dropna.

    One unpriceable ticker with an all-NaN column would otherwise take every
    row with it and report "0 sessions", which is a bug that reads like a
    data outage.
    """
    import markets

    frames: dict[str, pd.Series] = {}
    missing: list[str] = []
    why: list[str] = []
    for raw in symbols:
        sym = markets.canonical(raw)
        try:
            # The adapter takes the bare, source-native code — `AAPL`, not
            # `US:AAPL`. Handing it the canonical form 404s at Yahoo, and
            # A-shares hid it: their canonical symbol IS the bare code.
            market, code = markets.parse(sym)
            df = market.fetch_ohlcv(code)
        except Exception as exc:                                   # noqa: BLE001
            why.append(f"{sym}: {type(exc).__name__}: {exc}")
            df = None
        if df is None or df.empty or "Close" not in df:
            if not why or not why[-1].startswith(f"{sym}:"):
                why.append(f"{sym}: 没有返回行情")
            missing.append(sym)
            continue
        frames[sym] = df["Close"].rename(sym)

    if not frames:
        # Say which name failed and how. The bare "try again later" this used
        # to raise sent a real bug back as a weather report.
        raise RuntimeError("行情读取失败 — " + "；".join(why[:3]))

    px = pd.concat(frames.values(), axis=1)
    px = px.dropna(axis=1, how="all").ffill().dropna(how="any")
    # lookback + 1 rows so that `lookback` RETURNS survive the pct_change.
    if len(px) > lookback + 1:
        px = px.iloc[-(lookback + 1):]
    if len(px) < MIN_SESSIONS:
        raise LookupError(
            f"共同交易日只有 {len(px)} 天，少于 {MIN_SESSIONS} 天，"
            f"协方差矩阵没有意义")

    px.attrs["missing"] = missing
    return px


def moments(px: pd.DataFrame, duration: int = 1) -> tuple:
    """
    (period returns, annualised means, annualised covariance, daily returns).

    The last is separate on purpose. Overlapping N-day returns are right for
    the covariance and wrong for anything that compounds, so the simulation
    and the daily risk metrics take the period-1 series instead.
    """
    daily = px.pct_change(fill_method=None).dropna(how="any")
    rets = (daily if duration <= 1
            else px.pct_change(duration, fill_method=None).dropna(how="any"))
    factor = TRADING_DAYS / max(1, duration)
    return rets, rets.mean() * factor, rets.cov() * factor, daily


# ── the optimiser, as the page has it ────────────────────────────────────────
def performance(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple:
    """Annualised (return, volatility) for one weight vector."""
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def min_variance(mu: pd.Series, cov: pd.DataFrame, *, max_weight: float,
                 min_weight: float = 0.0) -> np.ndarray:
    """The global minimum-variance portfolio — the frontier's left-hand end."""
    from scipy.optimize import minimize

    n = len(mu)
    mu_v, cov_v = mu.to_numpy(), cov.to_numpy()
    res = minimize(lambda w: performance(w, mu_v, cov_v)[1],
                   np.full(n, 1.0 / n), method="SLSQP",
                   bounds=tuple((min_weight, max_weight) for _ in range(n)),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    if not res.success:
        raise LookupError(f"最小方差组合求解失败：{res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum() if w.sum() > 0 else w


def optimise_weights(mu: pd.Series, cov: pd.DataFrame, *,
                     target_return: float | None = None,
                     max_weight: float = 0.30,
                     min_weight: float = 0.0,
                     rf: float = 0.03) -> np.ndarray:
    """
    Max Sharpe, or minimum variance at a target return.

    `target_return` set is the frontier case: an equality constraint on the
    portfolio's expected return, minimising variance under it. That is how a
    point on the curve becomes a set of weights.
    """
    from scipy.optimize import minimize

    n = len(mu)
    mu_v, cov_v = mu.to_numpy(), cov.to_numpy()
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if target_return is not None:
        cons.append({"type": "eq",
                     "fun": lambda w, t=target_return:
                     performance(w, mu_v, cov_v)[0] - t})

    bounds = tuple((min_weight, max_weight) for _ in range(n))
    x0 = np.full(n, 1.0 / n)

    if target_return is None:
        def neg_sharpe(w):
            r, v = performance(w, mu_v, cov_v)
            return -(r - rf) / v if v > 0 else 0.0
        res = minimize(neg_sharpe, x0, method="SLSQP",
                       bounds=bounds, constraints=cons)
    else:
        res = minimize(lambda w: performance(w, mu_v, cov_v)[1], x0,
                       method="SLSQP", bounds=bounds, constraints=cons)

    if not res.success:
        raise LookupError(f"优化失败：{res.message}")
    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else w


def reachable_return(mu: pd.Series, *, max_weight: float = 0.30,
                     min_weight: float = 0.0) -> float:
    """
    The best annualised return the weight cap actually allows.

    The obvious ceiling is the best single stock, and that is what the
    Streamlit page used — but no portfolio can hold it at 100% when
    ``max_weight`` is 30%. With six names and a 30% cap the real ceiling is
    three names at the cap plus 10% of the fourth, and on live data that came
    out 28.2% against a best-stock figure of 81.2%. Spanning the frontier to
    the latter put 29 of 40 targets outside the feasible set, so they were
    solved, failed and dropped: an eleven-point curve that stopped short of
    the best portfolio available, leaving the max-Sharpe dot floating above
    its own frontier.

    So: give every name its floor, then pour what is left into the best
    returns in ``max_weight``-sized measures.
    """
    r = np.sort(mu.to_numpy())[::-1]
    n = len(r)
    budget = 1.0 - n * min_weight
    if budget < 0:
        return float(r.mean())
    step = max_weight - min_weight
    total = float(min_weight * r.sum())
    for x in r:
        if budget <= 0 or step <= 0:
            break
        take = min(step, budget)
        total += take * float(x)
        budget -= take
    return total


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, *,
                       points: int = FRONTIER_POINTS,
                       max_weight: float = 0.30,
                       min_weight: float = 0.0) -> list[dict]:
    """
    The efficient half of the curve, starting at the global minimum variance.

    Below the min-variance portfolio the parabola bends back on itself: those
    points give less return for no less risk. Starting from the worst
    individual stock's return — the obvious implementation — draws that
    inefficient tail and invites picking from it.

    The top end is `reachable_return`, not the best stock: this curve is the
    menu the weights are chosen from, so every point on it has to be a
    portfolio you can actually hold.
    """
    try:
        lo = performance(min_variance(mu, cov, max_weight=max_weight,
                                      min_weight=min_weight),
                         mu.to_numpy(), cov.to_numpy())[0]
    except Exception:                                              # noqa: BLE001
        lo = float(mu.min())

    hi = reachable_return(mu, max_weight=max_weight, min_weight=min_weight)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []

    out = []
    for target in np.linspace(lo, hi, points):
        try:
            w = optimise_weights(mu, cov, target_return=float(target),
                                 max_weight=max_weight, min_weight=min_weight)
        except LookupError:
            continue
        r, v = performance(w, mu.to_numpy(), cov.to_numpy())
        out.append({"ret_pct": round(r * 100, 2), "vol_pct": round(v * 100, 2),
                    "target": round(float(target), 6)})
    return out


# ── risk metrics, as the page has them ───────────────────────────────────────
def effective_bets(w: np.ndarray) -> float:
    """1 / Σw² — how many independent positions this really is."""
    s = float(np.sum(w ** 2))
    return 1.0 / s if s > 0 else 0.0


def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float:
    """Weighted average volatility ÷ portfolio volatility. Above 1 is a benefit."""
    vols = np.sqrt(np.diag(cov))
    port = float(np.sqrt(w @ cov @ w))
    return float(np.sum(w * vols) / port) if port > 0 else 0.0


def var_at(rets: pd.DataFrame, w: np.ndarray, conf: float) -> float:
    """The loss the portfolio exceeds on (1-conf) of days."""
    return float(np.percentile(rets.to_numpy() @ w, (1 - conf) * 100))


def cvar_at(rets: pd.DataFrame, w: np.ndarray, conf: float) -> float:
    """Average loss on the days that breach VaR — how bad the tail actually is."""
    port = rets.to_numpy() @ w
    tail = port[port <= var_at(rets, w, conf)]
    return float(tail.mean()) if len(tail) else float("nan")


def tail_ratio(rets: pd.DataFrame, w: np.ndarray, conf: float) -> float:
    """Upper tail ÷ lower tail. Above 1 means the upside is the bigger one."""
    port = rets.to_numpy() @ w
    up = float(np.percentile(port, conf * 100))
    dn = float(np.percentile(port, (1 - conf) * 100))
    return abs(up) / abs(dn) if dn != 0 else float("nan")


def worst_days(rets: pd.DataFrame, w: np.ndarray, n: int = 5) -> tuple:
    """(worst single day, average of the worst n) — what actually happened."""
    port = np.sort(rets.to_numpy() @ w)
    if not len(port):
        return float("nan"), float("nan")
    return float(port[0]), float(port[:n].mean())


def risk_metrics(daily: pd.DataFrame, w: np.ndarray, cov: pd.DataFrame) -> dict:
    """Every metric the assessment scores, in one dict."""
    worst, avg5 = worst_days(daily, w, 5)
    return {
        "enb": round(effective_bets(w), 2),
        "div_ratio": round(diversification_ratio(w, cov.to_numpy()), 3),
        "var_95_pct": round(var_at(daily, w, 0.95) * 100, 2),
        "var_99_pct": round(var_at(daily, w, 0.99) * 100, 2),
        "cvar_95_pct": round(cvar_at(daily, w, 0.95) * 100, 2),
        "cvar_99_pct": round(cvar_at(daily, w, 0.99) * 100, 2),
        "tail_95": round(tail_ratio(daily, w, 0.95), 2),
        "tail_99": round(tail_ratio(daily, w, 0.99), 2),
        "worst_day_pct": round(worst * 100, 2),
        "avg_worst5_pct": round(avg5 * 100, 2),
    }


ASSESSMENT_MAX = 100


def assess(sharpe: float, m: dict, n_assets: int) -> dict:
    """
    A 0-100 score across five dimensions, with the reasons spelled out.

    Ported band for band. The verdict cuts (80 / 60 / 40) and every threshold
    inside are the Streamlit page's judgement, not a derivation — kept
    identical because the one thing a port must not do is make the same
    portfolio score differently in the two apps.
    """
    score = 0
    strengths: list[str] = []
    notes: list[str] = []
    warnings: list[str] = []

    # 1. Risk-adjusted return — 20
    if sharpe > 1.5:
        score += 20
        strengths.append(f"风险调整后收益优秀（夏普 {sharpe:.2f}）")
    elif sharpe > 1.0:
        score += 15
        strengths.append(f"风险调整后收益良好（夏普 {sharpe:.2f}）")
    elif sharpe > 0.5:
        score += 10
        notes.append(f"风险调整后收益中等（夏普 {sharpe:.2f}）")
    else:
        score += 5
        warnings.append(f"风险调整后收益偏低（夏普 {sharpe:.2f}）")

    # 2. Diversification quality — 20
    enb = m["enb"]
    pct = (enb / n_assets * 100) if n_assets else 0
    if pct > 70:
        score += 20
        strengths.append(f"分散度很好（{n_assets} 只里有 {enb:.1f} 个有效独立仓位）")
    elif pct > 50:
        score += 15
        strengths.append(f"分散度不错（{enb:.1f} 个有效独立仓位）")
    elif pct > 30:
        score += 10
        notes.append(f"分散度中等（{enb:.1f} 个有效独立仓位）")
    else:
        score += 5
        warnings.append(f"过于集中（只有 {enb:.1f} 个有效独立仓位）—— 单一标的风险高")

    # 3. Diversification benefit — 15
    dr = m["div_ratio"]
    if dr > 1.3:
        score += 15
        strengths.append(f"分散化收益明显（分散比 {dr:.2f}）")
    elif dr > 1.15:
        score += 10
        strengths.append(f"有实际的分散化收益（分散比 {dr:.2f}）")
    elif dr > 1.0:
        score += 5
        notes.append(f"分散化收益很小（分散比 {dr:.2f}）")
    else:
        warnings.append(f"没有分散化收益（分散比 {dr:.2f}）—— 这些股票高度相关")

    # 4. Tail risk — 25
    t95, t99 = m["tail_95"], m["tail_99"]
    if t95 > 1.2 and t99 > 1.2:
        score += 25
        strengths.append("极端行情下上行大于下行")
    elif t95 > 1.0 and t99 > 1.0:
        score += 15
        notes.append("上行与下行大致平衡")
    elif t95 < 0.8 or t99 < 0.8:
        score += 5
        warnings.append("负偏：极端行情下的下行风险大于上行空间")
    else:
        score += 10

    # 5. Extreme loss — 20
    worst, avg5 = m["worst_day_pct"], m["avg_worst5_pct"]
    if worst > -3 and avg5 > -2:
        score += 20
        strengths.append(f"极端亏损风险低（最差单日 {worst:.2f}%）")
    elif worst > -5 and avg5 > -3:
        score += 15
        notes.append(f"极端亏损风险可控（最差单日 {worst:.2f}%）")
    elif worst > -8:
        score += 8
        warnings.append(f"极端亏损风险中等（最差单日 {worst:.2f}%）")
    else:
        warnings.append(f"极端亏损风险高（最差单日 {worst:.2f}%）—— 可能出现大幅回撤")

    if score >= 80:
        verdict, tone = "组合稳健 · 可以配置", "good"
        summary = "风险调整后收益、分散度和尾部风险都站得住。"
    elif score >= 60:
        verdict, tone = "组合尚可 · 谨慎对待", "warn"
        summary = "整体可以接受，但有明确的弱项 —— 先看下面的警告。"
    elif score >= 40:
        verdict, tone = "组合勉强 · 问题明显", "warn"
        summary = "风险指标或分散度上有明显短板，建议换一组股票或换一种权重方式。"
    else:
        verdict, tone = "组合偏弱 · 不建议", "bad"
        summary = "多个维度上的风险特征都不好，这个组合值得重新考虑。"

    return {"score": score, "max": ASSESSMENT_MAX, "verdict": verdict,
            "tone": tone, "summary": summary, "strengths": strengths,
            "notes": notes, "warnings": warnings}


# ── presentation ─────────────────────────────────────────────────────────────
def correlation(rets: pd.DataFrame) -> dict:
    c = rets.corr()
    return {"labels": list(c.columns),
            "rows": [[None if v != v else round(float(v), 3) for v in row]
                     for row in c.to_numpy()]}


def curve(daily: pd.DataFrame, w: pd.Series) -> list[float]:
    """
    Cumulative return of a fixed-weight portfolio, from 0% on the base day.

    The leading zero matters: the benchmark is rebased to its own first
    close, so without it the portfolio would start at day one's return while
    the benchmark starts at the origin, and the gap between the three lines
    would be wrong by a day everywhere.
    """
    aligned = w.reindex(daily.columns).fillna(0.0)
    port = (daily * aligned).sum(axis=1)
    return [0.0] + [round(float(v) * 100, 3) for v in ((1 + port).cumprod() - 1)]


def stats_of(daily: pd.DataFrame, w: pd.Series, rf: float = 0.0) -> dict:
    """Realised annualised stats from the daily series, not the moments."""
    aligned = w.reindex(daily.columns).fillna(0.0)
    port = (daily * aligned).sum(axis=1)
    if not len(port):
        return {"ann_return_pct": None, "ann_vol_pct": None,
                "sharpe": None, "max_drawdown_pct": None}
    vol = float(port.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann = float((1 + port).prod() ** (TRADING_DAYS / len(port)) - 1)
    growth = (1 + port).cumprod()
    return {
        "ann_return_pct": round(ann * 100, 2),
        "ann_vol_pct": round(vol * 100, 2),
        "sharpe": round((ann - rf) / vol, 2) if vol else None,
        "max_drawdown_pct": round(float((growth / growth.cummax() - 1).min()) * 100, 2),
    }


def benchmark_closes(market: str, index: pd.Index) -> tuple:
    """The benchmark on the portfolio's own dates, and its label."""
    code, label = BENCHMARKS.get(market, (None, ""))
    if not code:
        return None, ""
    try:
        if market == "CN":
            import data_manager
            df = data_manager.get_index_data_live(code, lookback_days=1500)
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
    want = index.tz_localize(None) if getattr(index, "tz", None) else index
    return s.reindex(want).ffill(), label


def prune(w: pd.Series) -> pd.Series:
    """
    Drop dust weights, then put back what dropping them removed.

    Dropping without renormalising leaves a book that adds to 99.4%, and
    every number derived from it is quietly scaled down by the difference.
    """
    kept = w[w >= MIN_WEIGHT]
    total = float(kept.sum())
    return kept / total if total > 0 else kept


def _names(symbols: list[str], market: str) -> dict:
    """
    Company names, because a weight next to `AAPL AAPL` reads as a bug.

    A-shares come from the local basics table in one hit. US and Canadian
    names have to be asked for one at a time, so they go out together rather
    than in series — and any that does not answer falls back to its ticker
    instead of holding up the whole optimisation.
    """
    if market == "CN":
        try:
            import data_manager
            rows = data_manager.get_all_stock_basic() or []
            found = {r["ticker"]: r["name"] for r in rows}
            return {s: found.get(s, s) for s in symbols}
        except Exception:                                          # noqa: BLE001
            return {s: s for s in symbols}

    from concurrent.futures import ThreadPoolExecutor

    import markets

    bare = {s: s.split(":", 1)[-1] for s in symbols}

    def one(sym: str) -> tuple[str, str]:
        try:
            adapter, code = markets.parse(sym)
            ref = adapter.resolve(code)
            return sym, str(ref.name) if ref and ref.name else bare[sym]
        except Exception:                                          # noqa: BLE001
            return sym, bare[sym]

    try:
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
            return dict(pool.map(one, symbols, timeout=NAME_TIMEOUT_S))
    except Exception:                                              # noqa: BLE001
        return bare


def build(symbols: list[str], *, target_return: float | None = None,
          max_weight: float = 0.30, lookback: int = DEFAULT_LOOKBACK,
          duration: int = 1, rf: float = 0.03) -> dict:
    """
    Weights, the frontier they came off, the risk metrics and the assessment.

    `target_return` None means "maximise Sharpe"; a value means "the least
    variance that reaches this return" — which is what makes the frontier a
    menu rather than an illustration.
    """
    import markets

    syms = list(dict.fromkeys(markets.canonical(s) for s in symbols))
    if len(syms) < 2:
        raise LookupError("至少需要两只股票")
    market = market_of(syms)

    px = load_closes(syms, lookback)
    rets, mu, cov, daily = moments(px, duration)
    if rets.empty or len(mu) < 2:
        raise LookupError("可用收益率数据不足")

    raw = optimise_weights(mu, cov, target_return=target_return,
                           max_weight=max_weight, rf=rf)
    w = prune(pd.Series(raw, index=px.columns))
    w_full = w.reindex(px.columns).fillna(0.0)
    ann_ret, ann_vol = performance(w_full.to_numpy(), mu.to_numpy(), cov.to_numpy())
    sharpe = (ann_ret - rf) / ann_vol if ann_vol else 0.0

    names = _names(list(px.columns), market)
    equal = pd.Series(1.0 / px.shape[1], index=px.columns)
    bench, bench_label = benchmark_closes(market, px.index)
    metrics = risk_metrics(daily, w_full.to_numpy(), cov)

    out = {
        "market": market,
        "mode": "target" if target_return is not None else "max_sharpe",
        "target_return_pct": (round(target_return * 100, 2)
                              if target_return is not None else None),
        "max_weight_pct": round(max_weight * 100, 1),
        "rf_pct": round(rf * 100, 2),
        "lookback": int(len(daily)), "duration": int(duration),
        "from": str(px.index[0].date()), "to": str(px.index[-1].date()),
        "missing": px.attrs.get("missing", []),
        "holdings": [{"t": t, "n": names.get(t, t),
                      "weight_pct": round(float(w.get(t, 0.0)) * 100, 2)}
                     for t in px.columns],
        # From the annualised moments, so the dot sits exactly on the frontier.
        "opt": {"ann_return_pct": round(ann_ret * 100, 2),
                "ann_vol_pct": round(ann_vol * 100, 2),
                "sharpe": round(sharpe, 2)},
        "stats": stats_of(daily, w, rf),
        "equal_stats": stats_of(daily, equal, rf),
        "risk": metrics,
        "assessment": assess(sharpe, metrics, px.shape[1]),
        "frontier": efficient_frontier(mu, cov, max_weight=max_weight),
        "singles": singles(mu, cov, names),
        "industries": industries(w_full, names, market),
        "correlation": correlation(rets),
        # One more date than there are returns — the base day the curves
        # start from, so all three lines share an origin.
        "dates": [str(d.date()) for d in px.index],
        "curve": curve(daily, w),
        "equal_curve": curve(daily, equal),
    }

    if bench is not None and bench.notna().any():
        first = bench.dropna()
        rebased = (bench / first.iloc[0] - 1) * 100
        out["benchmark"] = {
            "label": bench_label,
            "curve": [None if v != v else round(float(v), 3) for v in rebased]}
    else:
        out["benchmark"] = None
    return out


# ── industry exposure ────────────────────────────────────────────────────────
#: Above this in one industry the portfolio is a sector bet wearing six tickers.
CONCENTRATED = 0.50
CROWDED = 0.35
UNCLASSIFIED = "未分类"


def industries(weights: pd.Series, names: dict, market: str) -> dict:
    """
    Where the money actually is, once the tickers are collapsed into sectors.

    Six names at 16% each looks diversified in the allocation bars and is not
    diversified at all if five of them are 白酒. Effective bets counts
    positions; this counts the thing positions are a proxy for.

    A-share industries come from stock_basic. US names carry a sector on the
    Yahoo profile, fetched together rather than in series. Anything without
    one is `未分类` — named as such rather than dropped, because a silent
    omission would make the weights stop summing to 100%.
    """
    held = weights[weights > 0]
    if held.empty:
        return {"rows": [], "by_industry": [], "top_pct": 0.0, "tone": "good",
                "note": "没有持仓"}

    sector = _sectors(list(held.index), market)
    rows = [{"t": t, "n": names.get(t, t),
             "industry": sector.get(t) or UNCLASSIFIED,
             "weight_pct": round(float(w) * 100, 2)}
            for t, w in held.items()]

    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r["industry"], []).append(r)

    by_industry = sorted(
        ({"industry": k,
          "weight_pct": round(sum(x["weight_pct"] for x in v), 2),
          "count": len(v),
          "holdings": [x["n"] for x in sorted(
              v, key=lambda x: -x["weight_pct"])]}
         for k, v in buckets.items()),
        key=lambda x: -x["weight_pct"])

    top = by_industry[0]
    top_frac = top["weight_pct"] / 100
    if top_frac > CONCENTRATED:
        tone, note = "bad", (
            f"{top['industry']} 一个行业占 {top['weight_pct']:.1f}% —— "
            f"这是一笔行业押注，不是一个分散组合")
    elif top_frac > CROWDED:
        tone, note = "warn", (
            f"{top['industry']} 占 {top['weight_pct']:.1f}%，偏重")
    else:
        tone, note = "good", f"分散在 {len(by_industry)} 个行业"

    return {"rows": sorted(rows, key=lambda r: -r["weight_pct"]),
            "by_industry": by_industry,
            "top_pct": top["weight_pct"], "tone": tone, "note": note}


def _sectors(symbols: list[str], market: str) -> dict:
    if market == "CN":
        try:
            from db_manager import db
            df = db.read_table("stock_basic", columns="symbol,industry")
            if df is None or df.empty:
                return {}
            table = {str(r["symbol"]).strip(): (
                         str(r["industry"]).strip()
                         if r.get("industry") and str(r["industry"]) != "nan"
                         else None)
                     for _, r in df.iterrows()}
            # stock_basic is keyed on the bare code, but a saved fund stores
            # the Tushare form — `601066.SH`. Matching those literally put
            # every holding of every fund into 未分类.
            return {s: table.get(s.split(".", 1)[0]) for s in symbols}
        except Exception:                                          # noqa: BLE001
            return {}

    from concurrent.futures import ThreadPoolExecutor

    import markets

    def one(sym: str):
        try:
            adapter, code = markets.parse(sym)
            prof = adapter.profile(code) or {}
            return sym, prof.get("sector") or None
        except Exception:                                          # noqa: BLE001
            return sym, None

    try:
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
            return dict(pool.map(one, symbols, timeout=NAME_TIMEOUT_S))
    except Exception:                                              # noqa: BLE001
        return {}


# ── reference points on the frontier ─────────────────────────────────────────
def singles(mu: pd.Series, cov: pd.DataFrame, names: dict) -> list[dict]:
    """
    Each stock on its own, in risk/return space.

    The frontier means nothing without them. They are what the portfolio is
    being compared against — every one of them sits below and to the right of
    the curve, and that gap IS the diversification benefit.
    """
    sd = np.sqrt(np.diag(cov.to_numpy()))
    return [{"t": t, "n": names.get(t, t),
             "ret_pct": round(float(mu[t]) * 100, 2),
             "vol_pct": round(float(s) * 100, 2)}
            for t, s in zip(mu.index, sd)]


def evaluate(mu: pd.Series, cov: pd.DataFrame, daily: pd.DataFrame,
             w: pd.Series, rf: float = 0.03) -> dict:
    """Where an arbitrary set of weights lands — the weight editor's answer."""
    v = w.reindex(mu.index).fillna(0.0).to_numpy()
    ret, vol = performance(v, mu.to_numpy(), cov.to_numpy())
    return {
        "ann_return_pct": round(ret * 100, 2),
        "ann_vol_pct": round(vol * 100, 2),
        "sharpe": round((ret - rf) / vol, 3) if vol > 0 else None,
        "realised": stats_of(daily, pd.Series(v, index=mu.index), rf),
        "risk": risk_metrics(daily, v, cov),
        "curve": curve(daily, pd.Series(v, index=mu.index)),
    }


def weigh(symbols: list[str], weights: dict, *, lookback: int = DEFAULT_LOOKBACK,
          duration: int = 1, rf: float = 0.03, max_weight: float = 0.30) -> dict:
    """
    Hand-set weights, priced on the same history the optimiser used.

    This is what makes the frontier something you can argue with: move a
    weight, see where the portfolio moves in risk/return space and how far
    off the curve it lands. The frontier and the reference points come back
    too, so the caller plots the answer against the same axes rather than
    rescaling underneath it.
    """
    import markets

    syms = list(dict.fromkeys(markets.canonical(s) for s in symbols))
    if len(syms) < 2:
        raise LookupError("至少需要两只股票")
    market = market_of(syms)

    px = load_closes(syms, lookback)
    rets, mu, cov, daily = moments(px, duration)
    if rets.empty or len(mu) < 2:
        raise LookupError("可用收益率数据不足")

    w = pd.Series({markets.canonical(k): float(v) / 100.0
                   for k, v in weights.items()})
    w = w.reindex(px.columns).fillna(0.0)
    total = float(w.sum())
    if total <= 0:
        raise LookupError("权重合计必须大于 0")

    names = _names(list(px.columns), market)
    out = evaluate(mu, cov, daily, w, rf)
    out.update({
        "market": market,
        "sum_pct": round(total * 100, 2),
        # Reported, never silently corrected: a caller that is 3% short
        # should see that, not a normalised answer that hides it.
        "balanced": abs(total - 1.0) <= 0.001,
        "holdings": [{"t": t, "n": names.get(t, t),
                      "weight_pct": round(float(w[t]) * 100, 2)}
                     for t in px.columns if w[t] > 0],
        "industries": industries(w, names, market),
        "dates": [str(d.date()) for d in px.index],
    })
    return out
