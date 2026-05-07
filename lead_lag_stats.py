"""
Lead-Lag Statistical Analysis — Phase 4.

Three-layer output per (T stock, peer) pair:
  1. Granger causality  — rigorous test for direction and lag order
  2. Cross-correlation  — visualisable heatmap, lag by lag
  3. Cointegration + half-life — mean-reversion exploitability check

All heavy lifting is in compute_lead_lag(); fetch_qfq_returns() handles data.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

import data_manager

try:
    from statsmodels.tsa.stattools import grangercausalitytests, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_qfq_returns(tickers: list, lookback_days: int = 180):
    """
    Fetch qfq-adjusted daily returns for all tickers in one pass.

    Returns (returns_df, prices_df):
      - prices_df : DataFrame, index=trade_date str, columns=tickers, values=adj close
      - returns_df: same shape but daily pct returns (first row dropped)
    Tickers with no data are silently omitted.
    """
    if not tickers or not data_manager.init_tushare():
        return pd.DataFrame(), pd.DataFrame()

    bj = datetime.now(pytz.timezone("Asia/Shanghai"))
    end_date   = bj.strftime("%Y%m%d")
    # fetch extra calendar days to guarantee enough trading days
    start_date = (bj - timedelta(days=lookback_days + 90)).strftime("%Y%m%d")

    prices = {}
    for t in tickers:
        df = data_manager.fetch_adjusted_daily(t, start_date, end_date)
        if df.empty:
            continue
        df = df.set_index("trade_date")["close"]
        prices[t] = df

    if not prices:
        return pd.DataFrame(), pd.DataFrame()

    prices_df = pd.DataFrame(prices).sort_index()
    # Convert "YYYYMMDD" string index to proper datetime so charts render correctly
    prices_df.index = pd.to_datetime(prices_df.index, format="%Y%m%d")
    # Keep only the last `lookback_days` trading rows
    if len(prices_df) > lookback_days:
        prices_df = prices_df.iloc[-lookback_days:]

    returns_df = prices_df.pct_change().iloc[1:]  # drop first NaN row
    return returns_df, prices_df


# ── Statistical Helpers ───────────────────────────────────────────────────────

def _estimate_beta(r_t: pd.Series, r_s: pd.Series) -> float:
    """OLS slope: how much T moves for each 1 % move in S."""
    clean = pd.concat([r_t, r_s], axis=1).dropna()
    if len(clean) < 20 or not _STATSMODELS_OK:
        return float("nan")
    try:
        X = add_constant(clean.iloc[:, 1].values)
        return float(OLS(clean.iloc[:, 0].values, X).fit().params[1])
    except Exception:
        return float("nan")


def _granger_result(y: np.ndarray, x: np.ndarray, maxlag: int = 5):
    """
    Test whether x Granger-causes y.
    Returns (min_p_value, best_lag) across all lag orders 1..maxlag.
    Lower p-value = stronger evidence that x leads y.
    """
    try:
        data = np.column_stack([y, x])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        p_by_lag = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
        best_lag = min(p_by_lag, key=p_by_lag.get)
        return float(p_by_lag[best_lag]), int(best_lag)
    except Exception:
        return float("nan"), 0


def _cross_correlations(r_t: pd.Series, r_s: pd.Series, max_lag: int = 5) -> dict:
    """
    Returns {lag: correlation} for lag in range(-max_lag, max_lag+1).
      lag > 0 : T leads S  (corr between R_T[t] and R_S[t+lag])
      lag < 0 : S leads T  (corr between R_T[t+|lag|] and R_S[t])
      lag = 0 : concurrent
    """
    aligned = pd.concat([r_t, r_s], axis=1).dropna()
    if len(aligned) < 30:
        return {k: float("nan") for k in range(-max_lag, max_lag + 1)}

    rt = aligned.iloc[:, 0].values
    rs = aligned.iloc[:, 1].values
    result = {}
    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            c = np.corrcoef(rt[:-k], rs[k:])[0, 1]
        elif k < 0:
            kk = -k
            c = np.corrcoef(rt[kk:], rs[:-kk])[0, 1]
        else:
            c = np.corrcoef(rt, rs)[0, 1]
        result[k] = float(c) if not np.isnan(c) else float("nan")
    return result


def _half_life(spread: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life: Δspread_t = κ·spread_{t-1} + ε
    half_life = -ln(2) / κ  (days).  Returns nan if κ ≥ 0 (non-mean-reverting).
    """
    if not _STATSMODELS_OK:
        return float("nan")
    try:
        clean = spread.dropna()
        if len(clean) < 30:
            return float("nan")
        delta  = clean.diff().dropna()
        lagged = clean.shift(1).dropna()
        delta, lagged = delta.align(lagged, join="inner")
        X = add_constant(lagged.values)
        kappa = OLS(delta.values, X).fit().params[1]
        if kappa >= 0:
            return float("nan")
        return float(-np.log(2) / kappa)
    except Exception:
        return float("nan")


# ── Main Analysis ─────────────────────────────────────────────────────────────

def compute_lead_lag(
    t_ticker: str,
    peer_data: list,
    returns_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    For each peer in peer_data, compute:
      - beta (T relative to peer)
      - Granger causality in both directions + p-values
      - Cross-correlations at each lag (-max_lag … +max_lag)
      - Cointegration test + OU half-life (on price levels)

    peer_data : list of {ticker, name, layer_name, layer_idx}
    Returns   : DataFrame, one row per peer; includes '_xcorrs' column for heatmap.
    """
    if not _STATSMODELS_OK:
        raise RuntimeError(
            "statsmodels is required for lead-lag analysis. "
            "Run: pip install statsmodels"
        )

    if t_ticker not in returns_df.columns:
        raise RuntimeError(f"No return data for T stock {t_ticker}.")

    r_t = returns_df[t_ticker]
    p_t = prices_df[t_ticker] if t_ticker in prices_df.columns else None

    rows = []
    for peer in peer_data:
        s = peer["ticker"]
        if s not in returns_df.columns:
            continue

        r_s = returns_df[s]
        p_s = prices_df[s] if s in prices_df.columns else None

        aligned = pd.concat([r_t, r_s], axis=1).dropna()
        n_obs = len(aligned)
        if n_obs < 30:
            continue

        t_arr = aligned.iloc[:, 0].values
        s_arr = aligned.iloc[:, 1].values

        # ── Beta ──────────────────────────────────────────────────────────────
        beta = _estimate_beta(r_t, r_s)

        # ── Granger causality ─────────────────────────────────────────────────
        # p_T_leads_S: does T's past predict S? (reject H0 → T leads S)
        # p_S_leads_T: does S's past predict T? (reject H0 → S leads T)
        p_T_leads_S, lag_T_leads_S = _granger_result(s_arr, t_arr, max_lag)
        p_S_leads_T, lag_S_leads_T = _granger_result(t_arr, s_arr, max_lag)

        # ── Cross-correlation ─────────────────────────────────────────────────
        xcorrs = _cross_correlations(r_t, r_s, max_lag)
        valid  = {k: v for k, v in xcorrs.items() if not np.isnan(v)}
        if valid:
            peak_lag  = max(valid, key=lambda k: abs(valid[k]))
            peak_corr = valid[peak_lag]
        else:
            peak_lag, peak_corr = 0, float("nan")

        # ── Cointegration + half-life ─────────────────────────────────────────
        cointegrated = False
        half_life    = float("nan")
        if p_t is not None and p_s is not None:
            try:
                pt_c, ps_c = p_t.align(p_s, join="inner")
                pt_c = pt_c.dropna()
                ps_c = ps_c.dropna()
                if len(pt_c) >= 60:
                    _, coint_p, _ = coint(pt_c.values, ps_c.values)
                    cointegrated  = bool(coint_p < 0.05)
                    if cointegrated:
                        beta_price = _estimate_beta(pt_c, ps_c)
                        spread     = pt_c - beta_price * ps_c
                        half_life  = _half_life(spread)
            except Exception:
                pass

        # ── Relationship label ────────────────────────────────────────────────
        t_sig = not np.isnan(p_T_leads_S) and p_T_leads_S < 0.05
        s_sig = not np.isnan(p_S_leads_T) and p_S_leads_T < 0.05

        if t_sig and not s_sig:
            relationship = f"T leads S by {lag_T_leads_S}d"
        elif s_sig and not t_sig:
            relationship = f"S leads T by {lag_S_leads_T}d"
        elif t_sig and s_sig:
            relationship = "Bidirectional"
        else:
            relationship = "No relationship"

        # ── Signal strength ───────────────────────────────────────────────────
        p_vals = [p for p in [p_T_leads_S, p_S_leads_T] if not np.isnan(p)]
        p_best = min(p_vals) if p_vals else float("nan")
        hl_ok  = not np.isnan(half_life) and half_life < 15

        if not np.isnan(p_best) and p_best < 0.01 and cointegrated and hl_ok:
            signal = "🔥 Strong"
        elif not np.isnan(p_best) and p_best < 0.05 and (
            cointegrated or (not np.isnan(peak_corr) and abs(peak_corr) > 0.4)
        ):
            signal = "⚡ Moderate"
        elif not np.isnan(p_best) and p_best < 0.10:
            signal = "〰 Weak"
        else:
            signal = "✗ None"

        rows.append({
            "ticker":        s,
            "name":          peer["name"],
            "layer":         peer.get("layer_name", ""),
            "layer_idx":     peer.get("layer_idx", 0),
            "n_obs":         n_obs,
            "beta":          beta,
            "p_T_leads_S":   p_T_leads_S,
            "lag_T_leads_S": lag_T_leads_S,
            "p_S_leads_T":   p_S_leads_T,
            "lag_S_leads_T": lag_S_leads_T,
            "peak_corr":     peak_corr,
            "peak_lag":      peak_lag,
            "cointegrated":  cointegrated,
            "half_life":     half_life,
            "relationship":  relationship,
            "signal":        signal,
            "_xcorrs":       xcorrs,   # kept for heatmap; excluded from display table
        })

    return pd.DataFrame(rows)


def build_heatmap_df(results_df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Extract cross-correlations from results_df into a heatmap DataFrame.
    Rows = stocks, Columns = lag labels, Values = correlation.
    """
    lags = list(range(-max_lag, max_lag + 1))
    col_labels = []
    for k in lags:
        if k < 0:
            col_labels.append(f"S+{-k}d→T")   # S leads T
        elif k == 0:
            col_labels.append("±0d")
        else:
            col_labels.append(f"T+{k}d→S")    # T leads S

    rows = {}
    for _, row in results_df.iterrows():
        label = f"{row['ticker']} {row['name']}"
        xcorrs = row.get("_xcorrs", {})
        rows[label] = [xcorrs.get(k, float("nan")) for k in lags]

    return pd.DataFrame(rows, index=col_labels).T
