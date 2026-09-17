"""
strategies/pair_trade.py — 配对交易: two stocks that usually move together,
and what to do on the days they do not.

Extracted from pages/pair_trader.py unchanged in substance: same walk-forward
hedge ratio, same four statistical gates, same composite score, same buy-only
reading of the signal.

WALK-FORWARD, WHICH IS THE WHOLE POINT

The hedge ratio β is re-estimated every day by rolling OLS over the preceding
`ols_window` days, and then SHIFTED ONE DAY. Day t's spread is built from a β
that only saw data up to t−1. Fit β on the whole history instead and the spread
mean-reverts beautifully in backtest and not at all in life, because the ratio
was chosen knowing where the series ended up. Everything downstream — the
cointegration test, ADF, Hurst, the half-life, the z-score, the trade list — is
computed on that out-of-sample spread, so the statistics are a genuine
assessment rather than a description of the fit.

BUY-ONLY, BECAUSE THESE ARE A-SHARES

Textbook pairs trading shorts the rich leg and buys the cheap one. Shorting
A-shares is restricted in practice, so the signal is read one-sided: when the
spread stretches, BUY whichever leg is cheap relative to the other and reduce
the one that is rich. P&L is measured on the bought leg only — the honest
number for a trade you can actually place.

THE FOUR GATES, and what failing each one means

  * Cointegration (Engle-Granger p < 0.10) — is there a stable long-run
    relationship at all, or do they merely both drift upward?
  * ADF on the spread (p < 0.10) — is the spread itself stationary, i.e. does
    it come back, or does it wander?
  * Hurst < 0.45 — measured on spread LEVELS, which is correct because the
    spread is already mean-zero. 0.5 is a random walk; above that it trends
    and a stretched spread is more likely to stretch further.
  * Half-life 5-30 days — from an Ornstein-Uhlenbeck regression. Under 5 the
    move is gone before you can act on it; over 30 the capital is tied up for
    a quarter waiting.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Rolling window for the z-score of the spread.
Z_WINDOW = 60
#: Days of history each day's hedge ratio is estimated from.
OLS_WINDOW = 252
#: |Z| at which a pair is actionable, and at which a historical trade opens.
ENTRY_Z = 2.0
WATCH_Z = 1.5

#: Scores above this are worth looking at; the maximum possible is about 11.
GOOD_SCORE = 7.0

MAX_LEGS = 10


def hurst_rs(spread: np.ndarray) -> float:
    """
    R/S Hurst exponent on the spread LEVELS.

    Levels, not returns or first differences — the usual advice to difference
    first exists because price series are non-stationary, and a spread that
    passed the tests above already is not. H < 0.45 mean-reverts, 0.5 is a
    random walk, above 0.55 it trends.
    """
    n = len(spread)
    if n < 20:
        return 0.5
    cumdev = np.cumsum(spread - spread.mean())
    R = cumdev.max() - cumdev.min()
    S = spread.std(ddof=0)
    return math.log(R / S) / math.log(n) if S > 1e-9 else 0.5


def zscore_series(spread: np.ndarray, window: int = Z_WINDOW) -> np.ndarray:
    """Rolling z of the spread — how stretched it is against its own recent range."""
    s = pd.Series(spread)
    return ((s - s.rolling(window).mean()) / s.rolling(window).std()).values


def half_life(spread: np.ndarray) -> float:
    """
    Days for half of a deviation to decay, from ΔS_t = λ·S_{t−1} + μ + ε.

    Two guards, both required. λ ≥ 0 means the spread is not pulled back at
    all — it is diverging or random-walking. λ ≤ −2 puts the implied AR(1)
    coefficient outside the unit circle: the spread would overshoot the mean
    every single step, which is noise fitting, not mean reversion. Either way
    999 marks it untradeable rather than returning a small, inviting number.
    """
    from scipy import stats

    if len(spread) < 10:
        return 999.0
    lag = spread[:-1]
    slope = stats.linregress(lag, spread[1:] - lag).slope
    if slope >= 0 or slope <= -2:
        return 999.0
    return min(-math.log(2) / slope, 999.0)


def analyse_pair(code_a: str, code_b: str, log_prices: pd.DataFrame,
                 z_window: int = Z_WINDOW, ols_window: int = OLS_WINDOW) -> dict:
    """
    The full walk-forward assessment of one ordered pair.

    `log_prices` holds log closes for both codes on a shared index. Raises
    LookupError when the warm-up leaves too little out-of-sample history to
    say anything — which is a real answer, not a failure.
    """
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
    from statsmodels.tsa.stattools import adfuller, coint

    ya, xb = log_prices[code_a], log_prices[code_b]

    rres = RollingOLS(ya, sm.add_constant(xb), window=ols_window).fit()
    # The shift is the walk-forward: day t uses a ratio fitted on [t−window, t−1].
    params = rres.params.shift(1)
    beta, alpha = params[code_b], params["const"]

    oos = ya - (beta * xb + alpha)
    valid = oos.notna()
    if valid.sum() < z_window + 30:
        raise LookupError(
            f"{code_a}/{code_b}: {ols_window} 天热身后剩余样本外数据不足 — "
            f"需要更长的历史或更短的 OLS 窗口")

    spread = oos[valid]
    arr = spread.values
    clean_ya, clean_xb = ya[valid].values, xb[valid].values
    clean_beta = beta[valid].values

    eg_stat, eg_p, _ = coint(clean_ya, clean_xb)
    adf_stat, adf_p, *_ = adfuller(arr, maxlag=10, autolag="AIC")
    h = hurst_rs(arr)
    hl = half_life(arr)
    corr = float(np.corrcoef(np.diff(clean_ya), np.diff(clean_xb))[0, 1])

    z = zscore_series(arr, z_window)
    z_now = float(z[-1]) if not np.isnan(z[-1]) else 0.0

    # Composite, max ≈ 11. Correlation is clamped to [0, 1] so an
    # anti-correlated pair contributes nothing rather than a negative.
    score = (
        (3 if eg_p < 0.10 else 1 if eg_p < 0.20 else 0)
        + (2 if adf_p < 0.10 else 1 if adf_p < 0.15 else 0)
        + (2 if h < 0.45 else 1 if h < 0.50 else 0)
        + (1 if 5 <= hl <= 30 else 0)
        + (2 if abs(z_now) >= ENTRY_Z else 1 if abs(z_now) >= WATCH_Z else 0)
        + max(0.0, min(corr, 1.0))
    )

    return {
        "code_a": code_a, "code_b": code_b,
        "eg_p": round(eg_p, 4), "adf_p": round(adf_p, 4),
        "hurst": round(h, 3), "corr": round(corr, 3),
        "half_life": round(hl, 1), "beta_now": round(float(clean_beta[-1]), 3),
        "z_now": round(z_now, 2), "score": round(score, 2),
        "coint_ok": bool(eg_p < 0.10), "adf_ok": bool(adf_p < 0.10),
        "hurst_ok": bool(h < 0.45), "hl_ok": bool(5 <= hl <= 30),
        "dates": spread.index,
        "spread": arr,
        "z_series": z,
        "beta_series": clean_beta,
    }


def signal_for_pair(result: dict) -> tuple[str, str, str]:
    """
    (signal, buy this, reduce that) — the buy-only A-share reading.

    A negative z means A is cheap against B, so A is the one to buy. The
    labels name the leg to BUY, never a leg to short.
    """
    z, a, b = result["z_now"], result["code_a"], result["code_b"]
    if z <= -ENTRY_Z:
        return "BUY_A", a, b
    if z >= ENTRY_Z:
        return "BUY_B", b, a
    if abs(z) >= WATCH_Z:
        return "WATCH", a, b
    return "NEUTRAL", a, b


def detect_trades(z: np.ndarray, dates, prices: pd.DataFrame,
                  code_a: str, code_b: str,
                  entry_thresh: float = ENTRY_Z) -> list[dict]:
    """
    Every historical entry and exit in the out-of-sample window.

    In at a ±threshold crossing, out when the spread crosses zero. Because z
    and dates are both post-warm-up, each of these is a signal the engine
    could actually have given on the day, not one visible only in hindsight.
    """
    z_s = pd.Series(z, index=dates).dropna()
    idx = list(z_s.index)
    trades: list[dict] = []
    in_trade = False
    entry_date = entry_z = direction = None

    for i in range(1, len(z_s)):
        zp, zc, dt = float(z_s.iloc[i - 1]), float(z_s.iloc[i]), idx[i]
        if not in_trade:
            if zp > -entry_thresh and zc <= -entry_thresh:
                in_trade, entry_date, entry_z, direction = True, dt, zc, "BUY_A"
            elif zp < entry_thresh and zc >= entry_thresh:
                in_trade, entry_date, entry_z, direction = True, dt, zc, "BUY_B"
        else:
            if ((direction == "BUY_A" and zp < 0 <= zc)
                    or (direction == "BUY_B" and zp > 0 >= zc)):
                trades.append(make_trade(entry_date, dt, entry_z, zc, direction,
                                         False, prices, code_a, code_b))
                in_trade = False

    if in_trade:
        trades.append(make_trade(entry_date, idx[-1], entry_z, float(z_s.iloc[-1]),
                                 direction, True, prices, code_a, code_b))
    return trades


def make_trade(entry_date, exit_date, entry_z, exit_z, direction, is_open,
               prices: pd.DataFrame, code_a: str, code_b: str) -> dict:
    """P&L on the BOUGHT leg only — the leg an A-share account can hold."""
    buy_code = code_a if direction == "BUY_A" else code_b
    try:
        ei = prices.index.get_indexer([entry_date], method="nearest")[0]
        xi = prices.index.get_indexer([exit_date], method="nearest")[0]
        entry_price = float(prices[buy_code].iloc[ei])
        exit_price = float(prices[buy_code].iloc[xi])
        pnl = (exit_price / entry_price - 1) * 100.0
    except Exception:
        entry_price = exit_price = pnl = float("nan")

    def _r(v, nd):
        return None if (v is None or v != v) else round(v, nd)

    return {
        "entry": str(pd.Timestamp(entry_date).date()),
        "exit": str(pd.Timestamp(exit_date).date()),
        "entry_z": round(float(entry_z), 2),
        "exit_z": round(float(exit_z), 2),
        "direction": direction,
        "open": bool(is_open),
        "buy_code": buy_code,
        "entry_price": _r(entry_price, 3),
        "exit_price": _r(exit_price, 3),
        "pnl_pct": _r(pnl, 2),
    }


def rank_pairs(prices: pd.DataFrame, z_window: int = Z_WINDOW,
               ols_window: int = OLS_WINDOW) -> tuple[list[dict], list[dict]]:
    """
    Test every unique unordered pair among the supplied closes.

    (ranked results, skipped) — a pair with too little overlapping history is
    reported as skipped with its reason rather than dropped, because "these
    two have not traded together long enough" is an answer.
    """
    import itertools

    codes = list(prices.columns)
    if len(codes) < 2:
        raise LookupError("至少需要两只股票")
    if len(codes) > MAX_LEGS:
        raise LookupError(f"最多 {MAX_LEGS} 只股票")

    log_prices = np.log(prices)
    results, skipped = [], []
    for a, b in itertools.combinations(codes, 2):
        try:
            results.append(analyse_pair(a, b, log_prices, z_window, ols_window))
        except LookupError as exc:
            skipped.append({"code_a": a, "code_b": b, "why": str(exc)})
        except Exception as exc:                                # noqa: BLE001
            skipped.append({"code_a": a, "code_b": b, "why": f"计算失败：{exc}"[:160]})

    results.sort(key=lambda r: -r["score"])
    return results, skipped


SIGNAL_CN = {
    "BUY_A": "买入 A · 减持 B",
    "BUY_B": "买入 B · 减持 A",
    "WATCH": "接近信号",
    "NEUTRAL": "无信号",
}
