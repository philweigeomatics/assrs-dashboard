"""
pair_compare.py — the quantitative case for why one stock beat another,
and for whether a basket of them is really one trade.

The chart overlay shows THAT 长飞光纤 outran 中天科技. It cannot say why, and
"why" has only a few possible answers, each of which means something different
for whether you should own it:

  * It took more market risk. A β of 1.6 in a market that rose 20% hands you
    32% before the company does anything. That is leverage, not skill, and it
    reverses just as fast.
  * It earned more than its risk implied — alpha. Return the market's move
    does not explain.
  * It was simply more volatile, and you are looking at the good half of a
    wider distribution. Sharpe and max drawdown say whether the extra return
    paid for the extra pain.
  * Investors re-rated it: the price rose because the multiple rose, not
    because earnings did. Price = EPS × PE, so the return splits cleanly into
    those two, and a pure re-rating is the most fragile kind of gain.

So the panel answers each one and splits the gap between the two stocks into a
market-risk factor and an everything-else factor that multiply back to the full
outperformance exactly, instead of leaving you to eyeball two lines.

Conventions, stated because every number here depends on them:
  * 252 trading days a year.
  * Risk-free rate is ZERO. A-share cash yields little, the window is short,
    and a fudged rate would move Sharpe without making it more true. Sharpe
    here is therefore return-over-volatility, not excess-over-cash.
  * Beta, alpha and R² are measured against the MARKET'S OWN index — 沪深300,
    the S&P 500 or the S&P/TSX Composite — which the caller passes in and
    which is echoed back in the payload. The pair statistics (correlation,
    tracking error, information ratio) are measured between the two stocks.
  * Regressions run on LOG returns, which is what makes the attribution exact
    (see _logrets). Volatility, Sharpe and the capture ratios run on simple
    returns, because those describe moves you actually lived through.
  * Every series is cut to the SAME aligned trading days, so a stock that was
    suspended cannot score on days the other one traded.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: Windows the panel offers, in trading days. None = every aligned bar.
WINDOWS = {"60": 60, "120": 120, "252": 252, "all": None}
DEFAULT_WINDOW = "252"

#: Below this many overlapping bars the statistics are noise, not estimates.
MIN_BARS = 40


def _f(v, nd=2):
    """JSON-safe float, or None. PostgREST and JSON both reject NaN."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, nd)


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def _logrets(close: pd.Series) -> pd.Series:
    """
    Continuously compounded returns — what the regression runs on.

    Log returns ADD over time where simple returns compound, and that is the
    whole reason the attribution below is exact rather than approximate: an
    OLS fit with an intercept has residuals summing to zero, so

        log(A_end/A_start) = N·alpha + beta · log(M_end/M_start)

    holds exactly, with nothing left over. Attributing on simple returns left
    two thirds of a 288-point gap in an unexplained residual, which is no
    attribution at all. For daily moves the two betas differ in the third
    decimal; for the decomposition the difference is everything.
    """
    return np.log(close).diff().dropna()


def _max_drawdown(close: pd.Series) -> float:
    """Worst peak-to-trough fall, in percent (negative)."""
    peak = close.cummax()
    return float(((close / peak - 1).min()) * 100)


def _fit(r: pd.Series, m: pd.Series) -> tuple[float, float, float]:
    """
    (beta, DAILY log alpha, R²) from an OLS fit of `r` on `m`.

    Both series are log returns. Alpha is the intercept: the average daily
    log return left once the market's move is accounted for. With a zero
    risk-free rate this is Jensen's alpha, stated per day so the callers can
    either annualise it for display or sum it across the window for the exact
    attribution.
    """
    var = float(m.var(ddof=1))
    if var == 0 or len(r) < 2:
        return float("nan"), float("nan"), float("nan")
    beta = float(r.cov(m) / var)
    alpha_daily = float(r.mean() - beta * m.mean())
    return beta, alpha_daily, float(r.corr(m)) ** 2


def _annualise_alpha(alpha_daily: float) -> float:
    """
    A daily log alpha as the percentage return it compounds to over a year.

    Stated as a compounded figure, not alpha × 252, because that is the number
    you would actually have earned — and at the alphas A-share momentum names
    produce, the two differ by a lot.
    """
    if math.isnan(alpha_daily):
        return float("nan")
    return (math.exp(alpha_daily * TRADING_DAYS) - 1) * 100


def _capture(r: pd.Series, m: pd.Series, up: bool) -> float:
    """
    Up/down capture: of the market's average move on days it rose (or fell),
    what fraction did this stock take? 120 up / 80 down is the shape every
    manager wants; 120/130 is just leverage.
    """
    mask = m > 0 if up else m < 0
    if mask.sum() < 5:
        return float("nan")
    denom = float(m[mask].mean())
    if denom == 0:
        return float("nan")
    return float(r[mask].mean() / denom) * 100


def _profile(close: pd.Series, bench_close: pd.Series | None) -> dict:
    """
    Return, risk and market-sensitivity for one stock over the window.

    Volatility, Sharpe and the capture ratios run on SIMPLE returns, because
    they describe moves you actually experienced. Beta, alpha and R² come from
    the log-return fit, so they agree with the attribution rather than being a
    second, slightly different pair of numbers on the same screen.
    """
    r = _returns(close)
    n = len(r)
    total = float(close.iloc[-1] / close.iloc[0] - 1) * 100
    years = n / TRADING_DAYS
    vol = float(r.std(ddof=1)) * math.sqrt(TRADING_DAYS) * 100

    out = {
        "total_return_pct": _f(total),
        # Annualising a 3-month window magnifies noise into a headline, so the
        # figure is only offered once there is most of a year behind it.
        "cagr_pct": _f(((1 + total / 100) ** (1 / years) - 1) * 100) if years >= 0.75 else None,
        "vol_annual_pct": _f(vol),
        "sharpe": _f(float(r.mean()) / float(r.std(ddof=1)) * math.sqrt(TRADING_DAYS), 2)
        if float(r.std(ddof=1)) else None,
        "max_drawdown_pct": _f(_max_drawdown(close)),
        "best_day_pct": _f(float(r.max()) * 100),
        "worst_day_pct": _f(float(r.min()) * 100),
        "positive_days_pct": _f(float((r > 0).mean()) * 100, 1),
        "bars": n,
    }

    if bench_close is not None:
        mr = _returns(bench_close)
        beta, alpha_daily, r2 = _fit(_logrets(close), _logrets(bench_close))
        out.update({
            "beta": _f(beta),
            "alpha_annual_pct": _f(_annualise_alpha(alpha_daily)),
            "r2": _f(r2),
            "up_capture_pct": _f(_capture(r, mr, True), 1),
            "down_capture_pct": _f(_capture(r, mr, False), 1),
        })
    return out


def _valuation(fund: pd.DataFrame | None, close: pd.Series) -> dict | None:
    """
    Did the price rise because the multiple rose, or because earnings did?

    Price = EPS × PE exactly, so the return factorises into the change in the
    multiple and the change in trailing EPS. EPS is implied as Price ÷ PE_TTM
    rather than taken from a statement, which keeps it on the same adjusted
    price basis as everything else here.

    Returns None where PE is missing or non-positive: a loss-making company
    has no meaningful PE, and dividing by it would invent a number.
    """
    if fund is None or fund.empty or "PE_TTM" not in fund.columns:
        return None

    pe = fund["PE_TTM"].reindex(close.index, method="ffill")
    pe = pe.where(pe > 0)
    if pe.dropna().empty:
        return None

    first, last = pe.dropna().index[0], pe.dropna().index[-1]
    pe0, pe1 = float(pe[first]), float(pe[last])
    p0, p1 = float(close[first]), float(close[last])
    eps0, eps1 = p0 / pe0, p1 / pe1

    pb = fund["PB"].reindex(close.index, method="ffill") if "PB" in fund.columns else None
    mv = fund["Total_MV_Yi"].reindex(close.index, method="ffill") if "Total_MV_Yi" in fund.columns else None
    turn = fund["Turnover_Rate"].reindex(close.index, method="ffill") if "Turnover_Rate" in fund.columns else None

    return {
        "pe_start": _f(pe0), "pe_end": _f(pe1),
        "pb_start": _f(float(pb[first]), 2) if pb is not None and pd.notna(pb.get(first)) else None,
        "pb_end": _f(float(pb[last]), 2) if pb is not None and pd.notna(pb.get(last)) else None,
        "mv_yi": _f(float(mv[last]), 0) if mv is not None and pd.notna(mv.get(last)) else None,
        "turnover_avg_pct": _f(float(turn.mean()), 2) if turn is not None and turn.notna().any() else None,
        # The two halves of the move. They compound, not add:
        # (1+total) = (1+rerating) × (1+earnings).
        "rerating_pct": _f((pe1 / pe0 - 1) * 100),
        "earnings_pct": _f((eps1 / eps0 - 1) * 100),
        "priced_from": str(first.date()),
    }


def _monthly_relative(ra: pd.Series, rb: pd.Series) -> list[dict]:
    """
    Month-by-month lead: how far A ran ahead of (or behind) B.

    A single divergence number says the gap exists; these bars say WHEN it
    opened, which is usually the actual question — one earnings month can be
    the entire difference between two stocks in the same sector.
    """
    rel = ((1 + ra) / (1 + rb) - 1)
    out = []
    for period, g in rel.groupby(rel.index.to_period("M")):
        out.append({"month": str(period), "rel_pct": _f((np.prod(1 + g.values) - 1) * 100)})
    return out


def compare(a_close: pd.Series, b_close: pd.Series, *,
            bench_close: pd.Series | None = None,
            a_fund: pd.DataFrame | None = None,
            b_fund: pd.DataFrame | None = None,
            a_label: str = "A", b_label: str = "B",
            benchmark_label: str = "基准", market: str | None = None,
            window: str = DEFAULT_WINDOW) -> dict:
    """
    The full quantitative comparison of two stocks over one window.

    Every series is cut to the trading days all three (both stocks and the
    benchmark) actually have, so no statistic is computed across a gap one of
    them slept through.
    """
    if window not in WINDOWS:
        raise LookupError(f"window must be one of {sorted(WINDOWS)}")

    frame = pd.concat([a_close.rename("a"), b_close.rename("b")], axis=1).dropna()
    if bench_close is not None:
        frame = pd.concat([frame, bench_close.rename("m")], axis=1).dropna()

    bars = WINDOWS[window]
    if bars:
        # One extra bar: N returns need N+1 closes.
        frame = frame.tail(bars + 1)
    if len(frame) < MIN_BARS:
        raise LookupError(
            f"只有 {len(frame)} 个共同交易日，不足以计算（至少需要 {MIN_BARS} 个）")

    a, b = frame["a"], frame["b"]
    m = frame["m"] if "m" in frame.columns else None
    ra, rb = _returns(a), _returns(b)

    pair_beta, _pair_alpha, pair_r2 = _fit(_logrets(a), _logrets(b))
    diff = ra - rb
    te = float(diff.std(ddof=1)) * math.sqrt(TRADING_DAYS) * 100
    ir = (float(diff.mean()) * TRADING_DAYS * 100 / te) if te else float("nan")

    ret_a = float(a.iloc[-1] / a.iloc[0] - 1) * 100
    ret_b = float(b.iloc[-1] / b.iloc[0] - 1) * 100

    return {
        "window": window,
        "bars": len(frame),
        "from": str(frame.index[0].date()),
        "to": str(frame.index[-1].date()),
        "market": market,
        # Named by the caller. Hardcoding 沪深300 here labelled a US pair's
        # beta as measured against the Chinese index while it was actually
        # regressed on the S&P 500 — the number was right and the sentence
        # describing it was false, which is the worse of the two failures.
        "benchmark": {
            "label": benchmark_label,
            "total_return_pct": _f(float(m.iloc[-1] / m.iloc[0] - 1) * 100),
        } if m is not None else None,
        "a": {"label": a_label, **_profile(a, m), "valuation": _valuation(a_fund, a)},
        "b": {"label": b_label, **_profile(b, m), "valuation": _valuation(b_fund, b)},
        "pair": {
            "correlation": _f(float(ra.corr(rb)), 3),
            # How much A moves for each 1% of B — the two stocks' own beta,
            # which is what a pair trade is actually sized on.
            "beta_a_on_b": _f(pair_beta),
            "r2": _f(pair_r2, 3),
            "return_gap_pct": _f(ret_a - ret_b),
            "tracking_error_pct": _f(te),
            "information_ratio": _f(ir),
            "ratio": [_f(v, 4) for v in (a / b / (a.iloc[0] / b.iloc[0]) * 100).tolist()],
            "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
            "monthly": _monthly_relative(ra, rb),
        },
        "attribution": _attribution(a, b, m),
    }


def _attribution(a: pd.Series, b: pd.Series, m: pd.Series | None) -> dict | None:
    """
    Why one stock beat the other, split into market risk and everything else.

    Worked in log space, where an OLS fit with an intercept leaves residuals
    summing to exactly zero:

        log(A_end/A_start) = N·α_A + β_A · log(M_end/M_start)

    Subtracting the same identity for B gives the gap with nothing left over:

        log(A/B ratio) = N·(α_A − α_B) + (β_A − β_B) · log(market)

    Exponentiating turns those two terms into MULTIPLIERS that compound to the
    full outperformance:  ratio = beta_factor × alpha_factor. So "A ended 2.3×
    ahead: ×1.0 from taking more market risk, ×2.3 from itself" is exact, not
    an approximation with a residual big enough to swallow the answer.

    `residual_pct` stays in the payload as the proof — it should be zero, and
    a non-zero value means this reasoning has broken somewhere.
    """
    if m is None:
        return None
    la, lb, lm = _logrets(a), _logrets(b), _logrets(m)
    if len(lm) < MIN_BARS:
        return None

    beta_a, alpha_a, _ = _fit(la, lm)
    beta_b, alpha_b, _ = _fit(lb, lm)
    if any(math.isnan(x) for x in (beta_a, beta_b, alpha_a, alpha_b)):
        return None

    n = len(la)
    log_market = float(lm.sum())
    beta_log = (beta_a - beta_b) * log_market
    alpha_log = n * (alpha_a - alpha_b)

    ret_a = float(a.iloc[-1] / a.iloc[0] - 1) * 100
    ret_b = float(b.iloc[-1] / b.iloc[0] - 1) * 100
    ratio = float(a.iloc[-1] / a.iloc[0]) / float(b.iloc[-1] / b.iloc[0])

    return {
        "market_return_pct": _f((math.exp(log_market) - 1) * 100),
        # Two readings of the same outperformance: the plain difference in
        # returns, which is how people say it, and the ratio, which is the one
        # that factorises.
        "gap_pct": _f(ret_a - ret_b),
        "gap_ratio": _f(ratio, 3),
        # These two COMPOUND to the ratio: (1+beta)(1+alpha) = gap_ratio.
        "beta_factor_pct": _f((math.exp(beta_log) - 1) * 100),
        "alpha_factor_pct": _f((math.exp(alpha_log) - 1) * 100),
        "residual_pct": _f((math.exp(math.log(ratio) - beta_log - alpha_log) - 1) * 100, 4),
        "beta_a": _f(beta_a), "beta_b": _f(beta_b),
        "alpha_a_pct": _f(_annualise_alpha(alpha_a)),
        "alpha_b_pct": _f(_annualise_alpha(alpha_b)),
        "years": _f(n / TRADING_DAYS, 2),
    }


# ── baskets ──────────────────────────────────────────────────────────────────
#: More than this and the table stops being readable, and the fetches stop
#: being cheap.
MAX_BASKET = 8

#: Below this average pairwise correlation, "they move together" is not true
#: and the whole ranking is comparing unrelated things.
COHESIVE_R = 0.5


def basket(closes: dict, *,
           bench_close: pd.Series | None = None,
           labels: dict | None = None,
           benchmark_label: str = "基准", market: str | None = None,
           window: str = DEFAULT_WINDOW) -> dict:
    """
    Rank several stocks against each other and against their own average.

    This exists for one specific strategy: several names move together, all are
    trending up, so sell the ones moving least and concentrate into the leader.
    That is sound reasoning, but only when two things are true, and the numbers
    for both are computed here rather than assumed:

      * They really do move together. If the laggard's correlation to the rest
        is low, it is not a slower version of the same trade, it is a different
        trade, and swapping it for the leader raises concentration without
        buying the same exposure.
      * The laggard is genuinely weaker, not merely lower-beta. A stock with
        beta 0.6 in a basket averaging 1.4 SHOULD rise less; selling it for
        lagging is selling it for being what it is. Alpha controls for that, so
        both ranks are reported and the verdict names which one it fails.

    Every series is cut to the days ALL of them traded, so no stock is ranked
    over a stretch its peers slept through.
    """
    if window not in WINDOWS:
        raise LookupError(f"window must be one of {sorted(WINDOWS)}")
    symbols = list(closes)
    if len(symbols) < 2:
        raise LookupError("至少需要两只股票")
    if len(symbols) > MAX_BASKET:
        raise LookupError(f"最多 {MAX_BASKET} 只股票")

    frame = pd.concat([closes[s].rename(s) for s in symbols], axis=1).dropna()
    if bench_close is not None:
        frame = pd.concat([frame, bench_close.rename("__m")], axis=1).dropna()

    bars = WINDOWS[window]
    if bars:
        frame = frame.tail(bars + 1)
    if len(frame) < MIN_BARS:
        raise LookupError(
            f"只有 {len(frame)} 个共同交易日，不足以计算（至少需要 {MIN_BARS} 个）")

    m = frame["__m"] if "__m" in frame.columns else None
    prices = frame[symbols]
    rets = prices.pct_change().dropna()

    # The basket itself: equal-weight, rebalanced daily. This is the benchmark
    # each member is actually being judged against.
    basket_ret = rets.mean(axis=1)
    basket_total = float(np.prod(1 + basket_ret.values) - 1) * 100

    corr = rets.corr()
    rows = []
    for s in symbols:
        prof = _profile(prices[s], m)
        others = [o for o in symbols if o != s]
        # Correlation to the REST of the basket, not to a basket containing
        # itself: a stock is always correlated with a group it belongs to, and
        # the smaller the group the more that flatters it.
        peer = float(corr.loc[s, others].mean()) if others else float("nan")
        total = prof["total_return_pct"] or 0.0
        rows.append({
            "symbol": s,
            "label": (labels or {}).get(s, s),
            **prof,
            "corr_to_peers": _f(peer, 3),
            # A ratio, not a difference: "0.6x the basket" survives compounding
            # in a way that "40 points behind" does not.
            "vs_basket": _f((1 + total / 100) / (1 + basket_total / 100), 3),
        })

    by_return = sorted(rows, key=lambda r: -(r["total_return_pct"] or -1e9))
    has_alpha = all(r.get("alpha_annual_pct") is not None for r in rows)
    by_alpha = (sorted(rows, key=lambda r: -(r["alpha_annual_pct"] or -1e9))
                if has_alpha else [])
    for i, r in enumerate(by_return):
        r["rank_return"] = i + 1
    for i, r in enumerate(by_alpha):
        r["rank_alpha"] = i + 1

    n = len(symbols)
    pairs = [float(corr.iloc[i, j]) for i in range(n) for j in range(i + 1, n)]
    avg_corr = float(np.mean(pairs)) if pairs else float("nan")

    return {
        "window": window,
        "market": market,
        "bars": len(frame),
        "from": str(frame.index[0].date()),
        "to": str(frame.index[-1].date()),
        "benchmark": {
            "label": benchmark_label,
            "total_return_pct": _f(float(m.iloc[-1] / m.iloc[0] - 1) * 100),
        } if m is not None else None,
        "basket": {
            "total_return_pct": _f(basket_total),
            "avg_correlation": _f(avg_corr, 3),
            "cohesive": bool(not math.isnan(avg_corr) and avg_corr >= COHESIVE_R),
            "spread_pct": _f((by_return[0]["total_return_pct"] or 0)
                             - (by_return[-1]["total_return_pct"] or 0)),
        },
        "stocks": by_return,
        "symbols": symbols,
        "correlation": [[_f(float(corr.iloc[i, j]), 3) for j in range(n)] for i in range(n)],
        "verdicts": _verdicts(by_return, by_alpha, avg_corr),
    }


def _verdicts(by_return: list, by_alpha: list, avg_corr: float) -> list:
    """
    What the ranking actually supports, about the laggard.

    Deliberately conservative. The only case that endorses "sell the one moving
    least" is the one where it trails on BOTH raw return and alpha while really
    moving with the group. Everything else is a reason not to, and says which.
    """
    out = []
    if len(by_return) < 2:
        return out
    leader, laggard = by_return[0], by_return[-1]

    if math.isnan(avg_corr) or avg_corr < COHESIVE_R:
        out.append({
            "symbol": None, "kind": "warn",
            "text": f"这些股票的平均相关性只有 {avg_corr:.2f}，并没有在一起动。"
                    f"排名依然成立，但“换掉涨得少的那只”依赖同涨同跌这个前提，"
                    f"在这里并不成立。",
        })

    peer = laggard.get("corr_to_peers")
    if peer is not None and peer < 0.4:
        out.append({
            "symbol": laggard["symbol"], "kind": "warn",
            "text": f"{laggard['label']} 与其余标的的相关性只有 {peer:.2f}，"
                    f"它不是同一笔交易的慢速版本，而是另一笔交易。"
                    f"换成领先者会提高集中度，却换不到同样的暴露。",
        })
        return out

    if not by_alpha:
        return out

    alpha_last = by_alpha[-1]["symbol"] == laggard["symbol"]
    betas = [r.get("beta") for r in by_return if r.get("beta") is not None]
    lag_beta, lead_beta = laggard.get("beta"), leader.get("beta")
    low_beta = (lag_beta is not None and betas
                and lag_beta < float(np.mean(betas)) * 0.75)

    if alpha_last:
        out.append({
            "symbol": laggard["symbol"], "kind": "weak",
            "text": f"{laggard['label']} 在涨幅和 α 上都排最后"
                    f"（年化 α {laggard.get('alpha_annual_pct')}%，"
                    f"领先者 {leader.get('alpha_annual_pct')}%）——"
                    f"它是真的更弱，不只是波动小。",
        })
    elif low_beta:
        out.append({
            "symbol": laggard["symbol"], "kind": "ok",
            "text": f"{laggard['label']} 涨得少，但 β 只有 {lag_beta}"
                    f"（领先者 {lead_beta}），α 排第 {laggard.get('rank_alpha')}。"
                    f"它本来就该涨得少——按涨幅把它换掉，"
                    f"等于卖掉低波动的仓位去买高波动的仓位。",
        })
    else:
        out.append({
            "symbol": laggard["symbol"], "kind": "mixed",
            "text": f"{laggard['label']} 涨幅最低，但 α 排第 "
                    f"{laggard.get('rank_alpha')}，并非全面落后。"
                    f"差距更可能来自某一段时间的错位，而不是持续更弱。",
        })
    return out
