"""
optimise.py — what the weights could be, and whether changing them helps.

Mean-variance optimisation has a bad reputation it mostly deserves. Fed
historical average returns, it is an *estimation-error maximiser*: it puts
everything into whichever asset got luckiest in the sample, and the result
reliably underperforms equal weighting out of sample. The error is not in the
mathematics, it is that expected returns cannot be estimated from three years
of daily data to anything like the precision the optimiser assumes.

So this module is arranged around what IS estimable:

    最小方差  min variance   needs only the covariance matrix
    风险平价  risk parity    needs only the covariance matrix
    等权重    equal weight   needs nothing — the baseline that is hard to beat
    最大夏普  max Sharpe     needs expected returns, and is labelled accordingly

Covariances are estimated with Ledoit–Wolf shrinkage rather than the raw sample
matrix. With twenty holdings and 750 days the sample covariance is invertible
but badly conditioned, and an optimiser will happily exploit the noise in its
smallest eigenvalues — shrinkage is the standard, cheap correction.

Two more guards against corner solutions: long-only, and a cap on any single
weight. An "optimal" portfolio that is 90% one utility is a statement about the
sample, not advice.

And the part that makes this a claim rather than a picture: `walk_forward()`
fits the weights on the first half of the history and measures them on the
second, alongside the book's current weights and equal weighting. If the
optimised allocation does not beat what you already hold on data it never saw,
the page says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: Cap on any single weight. Without it the optimiser routinely proposes a
#: two-name portfolio, which is a statement about the sample rather than advice.
DEFAULT_CAP = 0.25

#: Weights below this are reported as zero — a 0.3% allocation is not a
#: position, it is a rounding artefact with a commission attached.
MIN_WEIGHT = 0.005

METHODS = {
    "min_var":     {"label": "最小方差", "en": "Minimum variance",
                    "needs_returns": False,
                    "means": "只用协方差矩阵，求整体波动最小的组合。不预测收益。"},
    "risk_parity": {"label": "风险平价", "en": "Risk parity",
                    "needs_returns": False,
                    "means": "让每只股票对组合波动的贡献相等。同样不预测收益。"},
    "equal":       {"label": "等权重", "en": "Equal weight",
                    "needs_returns": False,
                    "means": "每只一样多。基准线 —— 实证上很难被稳定超越。"},
    "max_sharpe":  {"label": "最大夏普", "en": "Maximum Sharpe",
                    "needs_returns": True,
                    "means": "用历史平均收益求风险调整后收益最高的组合。"
                             "历史收益几乎无法外推，这一项最容易过拟合，仅供参考。"},
}


def _cov(rets: pd.DataFrame) -> np.ndarray:
    """
    Ledoit–Wolf shrunk covariance, in daily units.

    The sample covariance of twenty assets over 750 days is invertible and
    badly conditioned; an optimiser finds the noise in its smallest
    eigenvalues and leans on it. Shrinkage pulls the estimate toward a scaled
    identity by a factor chosen analytically, which is the standard fix.
    """
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(rets.to_numpy()).covariance_
    except Exception:                                              # noqa: BLE001
        # Better a plain sample covariance than no optimiser at all.
        return np.cov(rets.to_numpy(), rowvar=False, ddof=1)


def effective_cap(cap: float, n: int) -> float:
    """
    The cap actually applied, which is not always the one asked for.

    Two corrections. A cap below 1/n makes "weights sum to 1" unsatisfiable, so
    the solve would fail and return whatever it happened to be holding. And a
    cap AT 1/n leaves exactly one feasible point — equal weighting — which is
    what 25% does to a five-name book: every method returns the same answer and
    the optimiser looks broken rather than constrained. So the cap is never
    tighter than twice equal weight, which always leaves room to express a
    preference while still ruling out a two-name portfolio.
    """
    return float(min(1.0, max(cap, 2.0 / n))) if n > 1 else 1.0


def _solve(objective, n: int, cap: float, x0=None) -> np.ndarray:
    from scipy.optimize import minimize

    cap = effective_cap(cap, n)
    res = minimize(
        objective,
        x0 if x0 is not None else np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = np.clip(res.x, 0.0, cap)
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / n)


def weights(rets: pd.DataFrame, method: str, *, cap: float = DEFAULT_CAP,
            rf_annual: float = 0.0) -> pd.Series:
    """Target weights for one method, as a Series indexed like `rets.columns`."""
    if method not in METHODS:
        raise LookupError(f"未知的优化方法：{method}")
    n = rets.shape[1]
    if n == 0:
        raise LookupError("没有可用于优化的标的")
    if n == 1:
        return pd.Series([1.0], index=rets.columns)

    if method == "equal":
        return pd.Series(np.full(n, 1.0 / n), index=rets.columns)

    cov = _cov(rets)

    if method == "min_var":
        w = _solve(lambda w: float(w @ cov @ w), n, cap)

    elif method == "risk_parity":
        def dispersion(w):
            port = float(np.sqrt(max(w @ cov @ w, 1e-18)))
            contrib = w * (cov @ w) / port
            return float(((contrib - port / n) ** 2).sum())
        w = _solve(dispersion, n, cap)

    else:                                        # max_sharpe
        mu = rets.mean().to_numpy() * TRADING_DAYS
        rf = rf_annual

        def neg_sharpe(w):
            vol = float(np.sqrt(max(w @ cov @ w, 1e-18)) * np.sqrt(TRADING_DAYS))
            return -((float(w @ mu) - rf) / vol) if vol > 0 else 0.0
        w = _solve(neg_sharpe, n, cap)

    w = np.where(w < MIN_WEIGHT, 0.0, w)
    total = w.sum()
    w = w / total if total > 0 else np.full(n, 1.0 / n)
    return pd.Series(w, index=rets.columns)


def stats(rets: pd.DataFrame, w: pd.Series) -> dict:
    """Annualised return, volatility and drawdown for one set of weights."""
    aligned = w.reindex(rets.columns).fillna(0.0)
    port = (rets * aligned).sum(axis=1)
    vol = float(port.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann = float((1 + port).prod() ** (TRADING_DAYS / len(port)) - 1)
    curve = (1 + port).cumprod()
    return {
        "ann_return_pct": round(ann * 100, 2),
        "ann_vol_pct": round(vol * 100, 2),
        "sharpe": round(ann / vol, 2) if vol else None,
        "max_drawdown_pct": round(float((curve / curve.cummax() - 1).min()) * 100, 2),
    }


def walk_forward(rets: pd.DataFrame, current: pd.Series, methods: list[str], *,
                 cap: float = DEFAULT_CAP, split: float = 0.5) -> dict:
    """
    Fit on the first half, measure on the second.

    This is the only part of the module that is evidence rather than
    arithmetic. In-sample, an optimiser always wins — it was chosen to. The
    question worth answering is whether weights fitted on data it had seen do
    better on data it had not, against the two benchmarks that matter: equal
    weighting, and the book you already hold.

    One split, not a rolling backtest: with three years of daily data there is
    only really one honest out-of-sample period, and a rolling version would
    imply a precision this does not have.
    """
    cut = int(len(rets) * split)
    train, test = rets.iloc[:cut], rets.iloc[cut:]
    if len(train) < 60 or len(test) < 60:
        raise LookupError("历史长度不足以做样本外检验（前后各需至少 60 个交易日）")

    rows = []
    for method in methods:
        w = weights(train, method, cap=cap)
        rows.append({
            "method": method,
            "label": METHODS[method]["label"],
            "in_sample": stats(train, w),
            "out_of_sample": stats(test, w),
        })

    current = current.reindex(rets.columns).fillna(0.0)
    current = current / current.sum() if current.sum() else current
    rows.append({
        "method": "current", "label": "当前持仓",
        "in_sample": stats(train, current),
        "out_of_sample": stats(test, current),
    })

    return {
        "train": {"from": str(train.index[0].date()), "to": str(train.index[-1].date()),
                  "sessions": len(train)},
        "test": {"from": str(test.index[0].date()), "to": str(test.index[-1].date()),
                 "sessions": len(test)},
        "rows": rows,
    }


def suggest(rets: pd.DataFrame, current: pd.Series, method: str, *,
            cap: float = DEFAULT_CAP, names: dict | None = None) -> dict:
    """
    A target allocation, what it would change, and whether it has ever helped.

    `current` is the book's present weights over the same symbols. The
    per-holding rows carry the delta, because "buy 4% more of this, sell 6% of
    that" is the actionable form and a column of target percentages is not.
    """
    if method not in METHODS:
        raise LookupError(f"未知的优化方法：{method}")
    if rets.shape[1] < 2:
        raise LookupError("至少需要两只可计价的标的才能做配置优化")

    target = weights(rets, method, cap=cap)
    current = current.reindex(rets.columns).fillna(0.0)
    current = current / current.sum() if current.sum() else current

    contrib = _risk_contrib(rets, target)
    rows = [{
        "symbol": s,
        "name": (names or {}).get(s, s),
        "current_pct": round(float(current[s]) * 100, 2),
        "target_pct": round(float(target[s]) * 100, 2),
        "delta_pct": round(float(target[s] - current[s]) * 100, 2),
        "risk_pct": round(float(contrib[s]) * 100, 2),
    } for s in rets.columns]
    rows.sort(key=lambda r: -r["target_pct"])

    return {
        "method": method,
        "label": METHODS[method]["label"],
        "means": METHODS[method]["means"],
        "overfit_risk": METHODS[method]["needs_returns"],
        "cap_pct": round(effective_cap(cap, rets.shape[1]) * 100, 1),
        "min_weight_pct": MIN_WEIGHT * 100,
        "rows": rows,
        "current": stats(rets, current),
        "target": stats(rets, target),
        "turnover_pct": round(float((target - current).abs().sum()) / 2 * 100, 1),
        # Said plainly, because a target allocation reads like a recommendation
        # whatever caveats sit beside it.
        "basis": "基于所选区间的历史协方差，样本内结果。是否值得换仓请看下方样本外检验。",
    }


def _risk_contrib(rets: pd.DataFrame, w: pd.Series) -> pd.Series:
    cov = _cov(rets)
    vec = w.to_numpy()
    vol = float(np.sqrt(max(vec @ cov @ vec, 1e-18)))
    return pd.Series(vec * (cov @ vec) / vol / vol if vol else vec, index=rets.columns)
