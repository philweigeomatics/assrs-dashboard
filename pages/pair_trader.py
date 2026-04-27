"""
Pair Trader  —  ASSRS V3 (Out-of-Sample Engine)
────────────────────────
Enter up to 10 A-share codes. The engine tests every unique pair for
cointegration, stationarity, and mean-reversion quality, then surfaces
the top tradeable pairs with full charts and plain-English explanations.


V3 UPGRADE: Walk-Forward Optimization
Eliminates lookahead bias by calculating today's spread using a rolling OLS
hedge ratio derived EXCLUSIVELY from the preceding N days. P&L reflects true
out-of-sample trading performance.


A-SHARE NOTE: Short-selling is restricted in A-shares.
This page uses a BUY-ONLY interpretation:
  Z ≤ −2σ  →  BUY the laggard (Stock A is cheap relative to B — buy A, reduce B)
  Z ≥ +2σ  →  BUY the laggard (Stock B is cheap relative to A — buy B, reduce A)
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import data_manager as dm
from datetime import datetime, timedelta
import datetime as _dt


# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
PAIR_COLORS = [
    ("#2dd4bf", "#f59e0b"),   # pair 1: teal / amber
    ("#a78bfa", "#60a5fa"),   # pair 2: violet / blue
    ("#f87171", "#34d399"),   # pair 3: red / green
]


# ──────────────────────────────────────────────────────────────────────────────
# BEIJING TIME helper
# ──────────────────────────────────────────────────────────────────────────────
def _beijing_today() -> _dt.date:
    """Return today's date in Asia/Shanghai timezone."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime as _datetime
        return _datetime.now(ZoneInfo("Asia/Shanghai")).date()
    except Exception:
        return (_dt.datetime.utcnow() + _dt.timedelta(hours=8)).date()


# ──────────────────────────────────────────────────────────────────────────────
# PURE MATH
# ──────────────────────────────────────────────────────────────────────────────

def _hurst_rs(spread: np.ndarray) -> float:
    """
    R/S Hurst exponent calculated directly on the spread (stationary series).

    Operates on spread LEVELS — not log-returns or first differences.
    This is correct because the spread is already mean-zero and stationary.

    H < 0.45  →  mean-reverting (ideal for pairs trading)
    H = 0.50  →  random walk
    H > 0.55  →  trending (spread drifts — avoid)
    """
    n = len(spread)
    if n < 20:
        return 0.5
    mean_s = spread.mean()
    cumdev = np.cumsum(spread - mean_s)
    R = cumdev.max() - cumdev.min()
    S = spread.std(ddof=0)
    return math.log(R / S) / math.log(n) if S > 1e-9 else 0.5


def _zscore_series(spread: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Rolling Z-score of the spread.
    Z = (spread − rolling_mean) / rolling_std
    Entry signal when |Z| ≥ 2.
    """
    s = pd.Series(spread)
    rm = s.rolling(window).mean()
    rs = s.rolling(window).std()
    return ((s - rm) / rs).values


def _half_life(spread: np.ndarray) -> float:
    """
    Half-life of mean reversion via Ornstein-Uhlenbeck regression.

    Regresses ΔS_t on S_{t-1}:  ΔS_t = λ·S_{t-1} + μ + ε
    Half-life = −ln(2) / λ

    Guards:
      λ ≥ 0        → spread is diverging or random-walking
      λ ≤ −2       → implied AR(1) coefficient outside unit circle (explosive oscillation)
      Both → return 999 (non-tradeable)
    """
    if len(spread) < 10:
        return 999.0

    spread_lag  = spread[:-1]                  # S_{t-1}
    spread_diff = spread[1:] - spread_lag      # ΔS_t

    slope, _, _, _, _ = stats.linregress(spread_lag, spread_diff)

    # Both guards are required:
    #   slope >= 0  → diverging/random walk
    #   slope <= -2 → AR(1) outside unit circle (over-shoots mean every step)
    if slope >= 0 or slope <= -2:
        return 999.0

    hl = -math.log(2) / slope
    return min(hl, 999.0)


def analyse_pair(
    code_a: str, code_b: str,
    log_prices: pd.DataFrame,
    z_window: int = 60,
    ols_window: int = 252,
) -> dict:
    """
    Full walk-forward statistical analysis for one pair.

    Core mechanism: Rolling OLS with shift(1) — the hedge ratio on any given
    day is estimated using only the PRECEDING ols_window days of data.
    This eliminates lookahead bias in both the spread and the Z-score.

    Statistical tests (cointegration, ADF, Hurst, half-life) are then run on
    the resulting out-of-sample spread — a conservative and honest assessment.
    """
    from statsmodels.tsa.stattools import adfuller, coint

    ya = log_prices[code_a]
    xb = log_prices[code_b]

    # ── 1. Rolling OLS: the walk-forward engine ───────────────────────────────
    # RollingOLS(endog, exog, window) computes β_t using [t-window, t-1] data.
    X = sm.add_constant(xb)
    rols = RollingOLS(ya, X, window=ols_window)
    rres = rols.fit()

    # CRITICAL: shift(1) ensures the hedge ratio used on day t was estimated
    # using data UP TO day t-1 only. No future information leaks in.
    params = rres.params.shift(1)
    beta   = params[code_b]
    alpha  = params["const"]

    # Out-of-sample spread: residual using only historically known hedge ratio
    oos_spread = ya - (beta * xb + alpha)

    # Drop warm-up period (first ols_window rows have NaN beta)
    valid_mask = oos_spread.notna()
    if valid_mask.sum() < z_window + 30:
        raise ValueError(
            f"Insufficient OOS data after {ols_window}-day warm-up. "
            "Move the start date further back or reduce the OLS window."
        )

    clean_spread = oos_spread[valid_mask]
    clean_ya     = ya[valid_mask].values
    clean_xb     = xb[valid_mask].values
    clean_beta   = beta[valid_mask].values
    spread_arr   = clean_spread.values

    # ── 2. Statistical tests on the OOS spread ────────────────────────────────
    # These are genuinely out-of-sample since the spread itself is OOS.
    eg_stat, eg_p, _ = coint(clean_ya, clean_xb)
    adf_stat, adf_p, *_ = adfuller(spread_arr, maxlag=10, autolag="AIC")

    h  = _hurst_rs(spread_arr)           # R/S directly on spread levels
    hl = _half_life(spread_arr)          # OU regression with both guards

    ra   = np.diff(clean_ya)
    rb   = np.diff(clean_xb)
    corr = float(np.corrcoef(ra, rb)[0, 1])

    # ── 3. Rolling Z-score ────────────────────────────────────────────────────
    z     = _zscore_series(spread_arr, window=z_window)
    z_now = float(z[-1]) if not np.isnan(z[-1]) else 0.0

    # Current beta (today's hedge ratio, estimated from yesterday's window)
    beta_now = float(clean_beta[-1])

    # ── 4. Composite quality score ────────────────────────────────────────────
    # max theoretical score ≈ 3+2+2+1+2+1 = 11
    # corr is capped to [0, 1] so anti-correlated pairs don't help their score
    score = (
        (3 if eg_p  < 0.10 else 1 if eg_p  < 0.20 else 0) +
        (2 if adf_p < 0.10 else 1 if adf_p < 0.15 else 0) +
        (2 if h     < 0.45 else 1 if h     < 0.50 else 0) +
        (1 if 5 <= hl <= 30 else 0) +
        (2 if abs(z_now) >= 2.0 else 1 if abs(z_now) >= 1.5 else 0) +
        max(0.0, min(corr, 1.0))    # clamp to [0,1]: anti-correlated pairs get 0, not negative
    )

    return {
        "code_a": code_a, "code_b": code_b,
        "eg_p":      round(eg_p,   4),
        "adf_p":     round(adf_p,  4),
        "hurst":     round(h,      3),
        "corr":      round(corr,   3),
        "half_life": round(hl,     1),
        "beta_now":  round(beta_now, 3),
        "z_now":     round(z_now,  2),
        "score":     round(score,  2),
        # Series — all indexed to valid_dates (OOS period only)
        "valid_dates":  clean_spread.index,
        "spread":       spread_arr,
        "z_series":     z,
        "beta_series":  clean_beta,
        # Pass/fail flags
        "coint_ok": eg_p  < 0.10,
        "adf_ok":   adf_p < 0.10,
        "hurst_ok": h     < 0.45,
        "hl_ok":    5 <= hl <= 30,
    }


def signal_for_pair(result: dict) -> tuple[str, str, str]:
    """
    A-share buy-only interpretation.
    Returns (signal, buy_code, reduce_code).
    """
    z = result["z_now"]
    a, b = result["code_a"], result["code_b"]
    if z <= -2.0:
        return "BUY_A", a, b
    elif z >= 2.0:
        return "BUY_B", b, a
    elif abs(z) >= 1.5:
        return "WATCH", a, b
    return "NEUTRAL", a, b


# ──────────────────────────────────────────────────────────────────────────────
# TRADE DETECTION + P&L
# ──────────────────────────────────────────────────────────────────────────────

def _detect_trades(
    z: np.ndarray,
    dates,
    prices: pd.DataFrame,
    code_a: str,
    code_b: str,
    entry_thresh: float = 2.0,
) -> list:
    """
    Scan Z-score series and return all historical trade intervals.

    Entry: Z crosses ±entry_thresh
    Exit:  Z crosses 0 (convergence)

    Because z and dates are both from the OOS period (post-warm-up),
    every detected trade is a true out-of-sample signal.
    """
    z_s   = pd.Series(z, index=dates).dropna()
    z_idx = list(z_s.index)
    trades = []
    in_trade  = False
    entry_date = entry_z = direction = None

    for i in range(1, len(z_s)):
        zp = float(z_s.iloc[i - 1])
        zc = float(z_s.iloc[i])
        dt = z_idx[i]

        if not in_trade:
            if zp > -entry_thresh and zc <= -entry_thresh:
                in_trade, entry_date, entry_z, direction = True, dt, zc, "BUY_A"
            elif zp < entry_thresh and zc >= entry_thresh:
                in_trade, entry_date, entry_z, direction = True, dt, zc, "BUY_B"
        else:
            exited = False
            if direction == "BUY_A" and zp < 0 and zc >= 0:
                exited = True
            elif direction == "BUY_B" and zp > 0 and zc <= 0:
                exited = True
            if exited:
                trades.append(_make_trade(
                    entry_date, dt, entry_z, zc, direction,
                    False, prices, code_a, code_b))
                in_trade = False

    if in_trade:
        trades.append(_make_trade(
            entry_date, z_idx[-1], entry_z, float(z_s.iloc[-1]),
            direction, True, prices, code_a, code_b))
    return trades


def _make_trade(
    entry_date, exit_date, entry_z, exit_z,
    direction, is_open, prices, code_a, code_b,
) -> dict:
    """Build a trade dict with P&L on the bought stock (buy leg only)."""
    buy_code = code_a if direction == "BUY_A" else code_b
    try:
        ei = prices.index.get_indexer([entry_date], method="nearest")[0]
        xi = prices.index.get_indexer([exit_date],  method="nearest")[0]
        entry_price = float(prices[buy_code].iloc[ei])
        exit_price  = float(prices[buy_code].iloc[xi])
        pnl_pct     = (exit_price / entry_price - 1) * 100.0
    except Exception:
        entry_price = exit_price = pnl_pct = float("nan")

    return {
        "entry":       entry_date,
        "exit":        exit_date,
        "entry_z":     entry_z,
        "exit_z":      exit_z,
        "direction":   direction,
        "open":        is_open,
        "buy_code":    buy_code,
        "entry_price": round(entry_price, 3) if not math.isnan(entry_price) else None,
        "exit_price":  round(exit_price,  3) if not math.isnan(exit_price)  else None,
        "pnl_pct":     round(pnl_pct, 2)     if not math.isnan(pnl_pct)     else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def chart_pair_full(
    result: dict,
    prices: pd.DataFrame,
    name_a: str, name_b: str,
    color_a: str, color_b: str,
    z_window: int,
) -> tuple[go.Figure, list]:
    """
    4-panel OOS chart for one pair. Returns (fig, trades).

    All four panels are drawn over the OOS valid_dates period only —
    no warm-up data bleeds into the visualisation.

    Row 1 — Normalised price (base=100 at OOS start)
    Row 2 — OOS spread with ±1σ/±2σ bands
    Row 3 — Rolling Z-score with trade entry/exit markers
    Row 4 — Rolling correlation of daily log-returns
    """
    a, b     = result["code_a"], result["code_b"]
    dates    = result["valid_dates"]
    spread   = result["spread"]
    z        = result["z_series"]
    z_now    = result["z_now"]
    beta_now = result["beta_now"]

    # Normalised prices indexed to OOS start
    p_a = prices.loc[dates, a].values
    p_b = prices.loc[dates, b].values
    na  = p_a / p_a[0] * 100
    nb  = p_b / p_b[0] * 100

    # Rolling return correlation over the OOS period
    log_a     = np.log(p_a)
    log_b     = np.log(p_b)
    ra        = pd.Series(np.diff(log_a), index=dates[1:])
    rb        = pd.Series(np.diff(log_b), index=dates[1:])
    roll_corr = ra.rolling(z_window).corr(rb)

    sp_mean = spread.mean()
    sp_std  = spread.std()

    # Build O(1) date→position lookup for trade marker indexing
    date_to_pos = {d: i for i, d in enumerate(dates)}

    trades = _detect_trades(z, dates, prices, a, b)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.22, 0.28, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            "① Normalised Price  (base = 100 at OOS start)",
            f"② Out-of-Sample Spread  (Current β: {beta_now:.3f}× {b})",
            f"③ Z-Score  ({z_window}-day rolling)  ·  Now: {z_now:+.2f}σ",
            f"④ Rolling {z_window}-day Return Correlation",
        ],
    )

    # ── Row 1: normalised prices ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=na, name=f"{a} {name_a}",
        line=dict(color=color_a, width=2),
        hovertemplate=f"{a}<br>%{{x|%Y-%m-%d}}<br>Index: %{{y:.1f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=nb, name=f"{b} {name_b}",
        line=dict(color=color_b, width=2),
        hovertemplate=f"{b}<br>%{{x|%Y-%m-%d}}<br>Index: %{{y:.1f}}<extra></extra>",
    ), row=1, col=1)

    # ── Row 2: OOS spread ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=spread, name="OOS Spread", showlegend=False,
        line=dict(color="#334155", width=1.5),
        hovertemplate="Spread<br>%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=sp_mean, line_dash="dot",
                  line_color="rgba(51,65,85,0.6)", row=2, col=1)
    for sig_v, clr in [(1, "#ca8a04"), (2, "#dc2626")]:
        for sign in [1, -1]:
            fig.add_hline(y=sp_mean + sign * sig_v * sp_std,
                          line_dash="dash", line_color=clr,
                          line_width=0.9, row=2, col=1)

    # ── Row 3: Z-score ────────────────────────────────────────────────────────
    fig.add_hrect(y0=2,  y1=5,  fillcolor="rgba(239,68,68,0.08)",  line_width=0, row=3, col=1)
    fig.add_hrect(y0=-5, y1=-2, fillcolor="rgba(16,185,129,0.08)", line_width=0, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=z, name="Z-score", showlegend=False,
        line=dict(color="#d97706", width=2),
        hovertemplate="Z<br>%{x|%Y-%m-%d}<br>%{y:.2f}σ<extra></extra>",
    ), row=3, col=1)
    for lv, clr, dash in [
        ( 2, "#dc2626", "dash"), (-2, "#059669", "dash"),
        ( 3, "#dc2626", "dot"),  (-3, "#059669", "dot"),
        ( 1, "#ca8a04", "dot"),  (-1, "#ca8a04", "dot"),
        ( 0, "#94a3b8", "dot"),
    ]:
        fig.add_hline(y=lv, line_dash=dash, line_color=clr,
                      line_width=1.0 if abs(lv) == 2 else 0.7, row=3, col=1)

    # ── Row 4: rolling correlation ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates[1:], y=roll_corr, name=f"{z_window}d Corr", showlegend=False,
        line=dict(color="#7c3aed", width=1.5),
        hovertemplate="Corr<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ), row=4, col=1)
    fig.add_hline(y=0.5, line_dash="dot",
                  line_color="rgba(100,116,139,0.5)", row=4, col=1)
    fig.add_hline(y=0, line_color="rgba(100,116,139,0.3)", row=4, col=1)

    # ── Trade overlays ────────────────────────────────────────────────────────
    for i, t in enumerate(trades, 1):
        is_open  = t.get("open", False)
        buy_a    = t["direction"] == "BUY_A"
        shade    = "rgba(16,185,129,0.09)"  if buy_a else "rgba(239,68,68,0.09)"
        lbl_clr  = "#059669"                if buy_a else "#dc2626"
        buy_code = t["buy_code"]
        buy_name = name_a if buy_a else name_b
        trade_lbl = f"#{i} BUY {buy_code} {buy_name}"

        for row in [1, 2, 3, 4]:
            fig.add_vrect(x0=t["entry"], x1=t["exit"],
                          fillcolor=shade, line_width=0, row=row, col=1)

        fig.add_trace(go.Scatter(
            x=[t["entry"]], y=[t["entry_z"]],
            mode="markers+text",
            marker=dict(symbol="circle", color=lbl_clr, size=11,
                        line=dict(color="white", width=1.5)),
            text=[f" #{i}"], textfont=dict(color=lbl_clr, size=9),
            textposition="top right" if i % 2 == 0 else "top left",
            name=trade_lbl, showlegend=True, legendgroup=f"trade{i}",
            hovertemplate=(f"BUY #{i}<br>%{{x|%Y-%m-%d}}<br>"
                           f"Z=%{{y:.2f}}σ<br>{trade_lbl}<extra></extra>"),
        ), row=3, col=1)

        pnl_str = f"  P&L: {t['pnl_pct']:+.1f}%" if t.get("pnl_pct") is not None else ""
        fig.add_trace(go.Scatter(
            x=[t["exit"]], y=[t["exit_z"]],
            mode="markers",
            marker=dict(
                symbol="x" if is_open else "square",
                color=lbl_clr, size=11,
                line=dict(color="#1e293b", width=1.5),
            ),
            showlegend=False, legendgroup=f"trade{i}",
            hovertemplate=(f"{'OPEN' if is_open else 'SELL'} #{i}{pnl_str}<br>"
                           f"%{{x|%Y-%m-%d}}<br>Z=%{{y:.2f}}σ<extra></extra>"),
        ), row=3, col=1)

        # Price panel markers — use O(1) dict lookup
        pidx_e = prices.index.get_indexer([t["entry"]], method="nearest")[0]
        pidx_x = prices.index.get_indexer([t["exit"]],  method="nearest")[0]
        y_series = na if buy_a else nb

        try:
            pos_e = date_to_pos[prices.index[pidx_e]]
            ep    = float(y_series[pos_e])
            fig.add_trace(go.Scatter(
                x=[t["entry"]], y=[ep], mode="markers",
                marker=dict(symbol="triangle-up", color=lbl_clr, size=10,
                            line=dict(color="white", width=1)),
                showlegend=False, legendgroup=f"trade{i}",
                hovertemplate=(f"BUY #{i}<br>%{{x|%Y-%m-%d}}<br>"
                               f"{buy_code} index: %{{y:.1f}}<extra></extra>"),
            ), row=1, col=1)
        except KeyError:
            pass

        if not is_open:
            try:
                pos_x = date_to_pos[prices.index[pidx_x]]
                xp    = float(y_series[pos_x])
                fig.add_trace(go.Scatter(
                    x=[t["exit"]], y=[xp], mode="markers",
                    marker=dict(symbol="square", color=lbl_clr, size=9,
                                line=dict(color="#1e293b", width=1.5)),
                    showlegend=False, legendgroup=f"trade{i}",
                    hovertemplate=(f"SELL #{i}{pnl_str}<br>%{{x|%Y-%m-%d}}<br>"
                                   f"{buy_code} index: %{{y:.1f}}<extra></extra>"),
                ), row=1, col=1)
            except KeyError:
                pass

    n_closed   = sum(1 for t in trades if not t.get("open"))
    n_open     = sum(1 for t in trades if t.get("open"))
    trade_note = f"  ·  {len(trades)} OOS trades ({n_closed} closed · {n_open} open)"

    fig.update_layout(
        height=860, template="plotly_dark",
        title=(f"{a} {name_a}  ↔  {b} {name_b}  |  OOS Walk-Forward Backtest  "
               f"· ADF p={result['adf_p']:.3f}  HL={result['half_life']:.0f}d  "
               f"Corr={result['corr']:.2f}{trade_note}"),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=-0.04, font=dict(size=10)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Index",  title_font_size=10, row=1, col=1)
    fig.update_yaxes(title_text="Spread", title_font_size=10, row=2, col=1)
    fig.update_yaxes(title_text="Z (σ)",  title_font_size=10, row=3, col=1)
    fig.update_yaxes(title_text="Corr",   title_font_size=10, row=4, col=1)
    return fig, trades


def chart_scatter_regression(
    result: dict,
    prices: pd.DataFrame,
    name_a: str, name_b: str,
    color_a: str, color_b: str,
) -> go.Figure:
    """
    Log-price scatter over the OOS period with a static OLS regression line.

    Note: The regression line here uses a static (full-period) OLS slope for
    visualisation clarity. The actual spread and Z-score in the main chart use
    the rolling OLS hedge ratio. A tighter cluster = more stable relationship.
    """
    a, b  = result["code_a"], result["code_b"]
    dates = result["valid_dates"]

    xa = np.log(prices.loc[dates, b].values)
    ya = np.log(prices.loc[dates, a].values)

    slope, intercept, _, _, _ = stats.linregress(xa, ya)
    x_line = np.linspace(xa.min(), xa.max(), 100)
    y_line = slope * x_line + intercept

    n          = len(xa)
    idx_colors = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xa, y=ya, mode="markers",
        marker=dict(
            color=idx_colors, colorscale="Viridis", size=5, opacity=0.7,
            colorbar=dict(title="Day #", thickness=10, len=0.6),
            showscale=True,
        ),
        text=[d.strftime("%Y-%m-%d") for d in dates],
        hovertemplate=f"log({b}): %{{x:.3f}}<br>log({a}): %{{y:.3f}}<br>%{{text}}<extra></extra>",
        name="Daily observations (OOS period)",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#334155", width=1.5, dash="dash"),
        name=f"Static OLS  slope={slope:.3f}  (visualisation only)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=(f"Log-Price Scatter: {b} (x) vs {a} (y)  —  OOS period<br>"
               f"<sup>Static slope={slope:.3f}  ·  tighter cluster = more stable relationship</sup>"),
        xaxis_title=f"log price of {b} {name_b}",
        yaxis_title=f"log price of {a} {name_a}",
        height=400, template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def chart_rolling_beta(result: dict, name_a: str, name_b: str) -> go.Figure:
    """
    Rolling hedge ratio (β) over the OOS period.

    A stable, near-horizontal β → the relative sizing of positions is consistent.
    A strongly trending β → the fundamental relationship is shifting; use caution.
    """
    a, b  = result["code_a"], result["code_b"]
    dates = result["valid_dates"]
    beta  = result["beta_series"]

    mean_b = float(np.nanmean(beta))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=beta, mode="lines",
        line=dict(color="#3b82f6", width=2),
        name="Rolling β (hedge ratio)",
        hovertemplate="β<br>%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=mean_b, line_dash="dot",
                  line_color="rgba(148,163,184,0.6)",
                  annotation_text=f"mean β={mean_b:.3f}",
                  annotation_position="top right")
    fig.update_layout(
        title=(f"Structural Drift Monitor: Rolling Hedge Ratio (β)<br>"
               f"<sup>How many units of {name_b} hedge 1 unit of {name_a} — "
               f"flat line = stable relationship</sup>"),
        yaxis_title="Hedge Ratio (β)",
        height=400, template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TRADE P&L TABLE
# ──────────────────────────────────────────────────────────────────────────────

def render_trade_table(
    trades: list, name_a: str, name_b: str,
    code_a: str, code_b: str,
) -> None:
    if not trades:
        st.caption("No historical trades detected in this OOS period.")
        return

    rows = []
    for i, t in enumerate(trades, 1):
        buy_code   = t["buy_code"]
        buy_name   = name_a if buy_code == code_a else name_b
        side_label = f"BUY {buy_code} {buy_name}"

        duration = (t["exit"] - t["entry"]).days
        if t.get("open"):
            pnl_str = "⏳ Open"
        else:
            p = t.get("pnl_pct")
            pnl_str = f"{p:+.2f}%" if p is not None else "—"

        rows.append({
            "#":           i,
            "Action":      side_label,
            "Entry Date":  t["entry"].strftime("%Y-%m-%d"),
            "Entry Z":     f"{t['entry_z']:+.2f}σ",
            "Entry Price": f"{t['entry_price']}" if t.get("entry_price") else "—",
            "Exit Date":   t["exit"].strftime("%Y-%m-%d"),
            "Exit Z":      f"{t['exit_z']:+.2f}σ",
            "Exit Price":  f"{t['exit_price']}"  if t.get("exit_price")  else "—",
            "Duration":    f"{duration}d",
            "P&L %":       pnl_str,
            "Status":      "Open" if t.get("open") else "Closed",
        })

    df = pd.DataFrame(rows)

    def colour_pnl(val):
        if val == "⏳ Open":
            return "color: #d97706; font-weight:600"
        try:
            v = float(val.replace("%", "").replace("+", ""))
            if v > 0: return "color: #059669; font-weight:600"
            if v < 0: return "color: #dc2626; font-weight:600"
        except Exception:
            pass
        return ""

    def colour_row(row):
        if row["Status"] == "Open":
            return ["background-color: rgba(217,119,6,0.08)"] * len(row)
        try:
            p = float(row["P&L %"].replace("%", "").replace("+", ""))
            if p > 0: return ["background-color: rgba(5,150,105,0.08)"] * len(row)
            if p < 0: return ["background-color: rgba(220,38,38,0.08)"] * len(row)
        except Exception:
            pass
        return [""] * len(row)

    styled = (df.style
              .apply(colour_row, axis=1)
              .map(colour_pnl, subset=["P&L %"])
              .hide(axis="index"))

    st.dataframe(styled, use_container_width=True,
                 height=min(500, 45 + 36 * len(rows)))

    closed = [t for t in trades if not t.get("open") and t.get("pnl_pct") is not None]
    if closed:
        wins      = [t["pnl_pct"] for t in closed if t["pnl_pct"] > 0]
        losses    = [t["pnl_pct"] for t in closed if t["pnl_pct"] <= 0]
        win_rate  = len(wins) / len(closed) * 100
        avg_win   = sum(wins)   / len(wins)   if wins   else 0.0
        avg_loss  = sum(losses) / len(losses) if losses else 0.0
        total_pnl = sum(t["pnl_pct"] for t in closed)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate",  f"{win_rate:.0f}%",
                  help="% of closed OOS trades that were profitable")
        c2.metric("Avg Win",   f"{avg_win:+.2f}%",
                  help="Average return on winning trades (buy leg only)")
        c3.metric("Avg Loss",  f"{avg_loss:+.2f}%",
                  help="Average return on losing trades (buy leg only)")
        c4.metric("Total P&L", f"{total_pnl:+.2f}%",
                  help="Sum of all closed trade returns — not compounded, no transaction costs")


# ──────────────────────────────────────────────────────────────────────────────
# EXPANDER EXPLANATIONS
# ──────────────────────────────────────────────────────────────────────────────

def expander_how_it_works():
    with st.expander("📖 How Pairs Trading Works (A-Share Edition)  配对交易原理", expanded=False):
        st.markdown("""
**Core idea:** Two stocks in the same sector tend to move together over time.
When one temporarily underperforms the other, the gap (called the *spread*) is
expected to close — you profit by buying the laggard.

**A-share implementation (buy-only):**
- Stock A drops relative to B (Z ≤ −2σ) → **Buy A** (the laggard), wait for it to catch up
- Stock B drops relative to A (Z ≥ +2σ) → **Buy B** (the laggard), wait for it to catch up
- Exit when spread reverts to Z ≈ 0 (they've converged)

**Why not short?**
Full short-selling of A-shares requires margin accounts and securities lending (融券),
which is costly, restricted, and unavailable to most retail traders.
The buy-only version captures the mean-reversion on the underperforming leg only.

**Limitation vs. true pairs:**
A classic pairs trade is market-neutral (long one, short the other simultaneously).
This buy-only version retains directional market exposure — if the whole sector
falls, you lose even if the spread converges. Always consider sector/market direction.
        """)


def expander_oos():
    with st.expander("🔬 V3 Walk-Forward Engine: What Changed & Why  滚动最小二乘法", expanded=False):
        st.markdown(r"""
**The V2 lookahead bias problem:**
V2 calculated the hedge ratio (β) using a static OLS fit over the *entire* dataset.
This means a trade in 2024 was priced using β that incorporated 2025–2026 data —
the algorithm knew the future. Historical P&L was an in-sample fit, not a real backtest.

**The V3 fix — Rolling OLS with shift(1):**

$$\hat{\beta}_t = \text{OLS}\bigl(\log P_A, \log P_B,\ [t - W,\ t-1]\bigr)$$

$$\text{spread}_t = \log P_A^t - \hat{\beta}_t \cdot \log P_B^t - \hat{\alpha}_t$$

On day $t$, the hedge ratio $\hat{\beta}_t$ is estimated using **only the preceding W days**.
The `.shift(1)` call ensures data up to $t-1$ is used — never day $t$ itself.

**What this means for you:**
- Every Z-score, every trade entry, every P&L figure in the trade table is a **true
  out-of-sample simulation** — the algorithm had no access to future prices
- The warm-up period (first W days) is discarded entirely from charts and statistics
- Statistical tests (ADF, Hurst, cointegration) are run on the OOS spread — equally honest

**Tradeoff:**
Rolling OLS requires more history. Set the data start at least `OLS window + 1 year`
before today. The default is 3 years back to ensure a robust OOS period.
        """)


def expander_cointegration():
    with st.expander("① Engle-Granger Cointegration Test  协整检验", expanded=False):
        st.markdown(r"""
**What it tests:** Whether two price series share a long-run equilibrium relationship.

Two stocks are *cointegrated* if there exists a linear combination of them
that is stationary (doesn't drift away forever):

$$\text{spread}_t = \log P_A - \beta \cdot \log P_B - \alpha$$

| p-value | Interpretation |
|---|---|
| < 0.05 | ✅ Strong cointegration — spread will revert |
| 0.05–0.10 | ⚠️ Moderate — tradeable but watch closely |
| > 0.10 | ❌ No confirmed cointegration — spread may drift |
        """)


def expander_adf():
    with st.expander("② ADF Stationarity Test  单位根检验", expanded=False):
        st.markdown(r"""
**What it tests:** Whether the spread series itself is stationary.

The Augmented Dickey-Fuller (ADF) test checks for a *unit root* in the spread.
If the spread has a unit root, it wanders randomly — not tradeable.
If it's stationary, it oscillates around a mean.

$$H_0: \text{spread has a unit root (non-stationary)} $$
$$H_1: \text{spread is stationary}$$
        """)


def expander_hurst():
    with st.expander("③ Hurst Exponent  赫斯特指数", expanded=False):
        st.markdown(r"""
**What it measures:** The persistence of the spread series.

Computed via R/S analysis directly on spread levels (not log-returns):

$$H = \frac{\log(R/S)}{\log(n)}$$

| H value | Behaviour | For pairs trading |
|---|---|---|
| < 0.45 | Mean-reverting | ✅ Ideal |
| 0.45–0.55 | Random walk | ⚠️ Marginal |
| > 0.55 | Trending | ❌ Spread drifts |
        """)


def expander_halflife():
    with st.expander("④ Half-Life  均值回归半衰期", expanded=False):
        st.markdown(r"""
**What it measures:** How quickly the spread reverts halfway back to its mean.

Estimated via the Ornstein-Uhlenbeck regression:

$$\Delta S_t = \lambda \cdot S_{t-1} + \mu + \varepsilon_t$$

$$\text{half-life} = \frac{-\ln(2)}{\lambda}$$

λ must be negative (mean-reverting) and > −2 (no explosive oscillation).

| Half-life | Implication |
|---|---|
| < 5 days | Very fast — slippage eats profit |
| 5–30 days | ✅ Ideal range for active trading |
| > 30 days | Slow — capital tied up too long per trade |
        """)


def expander_zscore():
    with st.expander("⑤ Z-Score & Entry/Exit Logic  Z分数与交易信号", expanded=False):
        st.markdown(r"""
**What it measures:** How far the current spread is from its recent average, in standard deviations.

$$Z_t = \frac{\text{spread}_t - \mu_{\text{rolling}}}{\sigma_{\text{rolling}}}$$

| Z-score | A-share Action |
|---|---|
| ≤ −2σ | ✅ BUY A — A is cheap relative to B |
| ≥ +2σ | ✅ BUY B — B is cheap relative to A |
| −1.5 to −2σ | 👁 WATCH — approaching entry |
| Around 0 | SELL — spread has converged, take profit |
| ≤ −3σ or ≥ +3σ | 🚨 STOP — relationship may have broken down |
        """)


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_raw_prices(tickers: tuple, start_date: str) -> dict:
    """
    Fetch daily qfq-adjusted close prices for all tickers.

    Uses forward-fill (limit=5) to handle short trading halts, which are
    common in A-shares. Rows where more than 30% of stocks have missing data
    are dropped, and a warning is shown if data is significantly trimmed.
    """
    frames, names_map, errors = {}, {}, []

    for code in tickers:
        try:
            df = dm.get_ohlcv_for_wave(code, granularity="Daily", start_date=start_date)
            if df is not None and not df.empty:
                frames[code] = df["Close"].rename(code)
                try:
                    names_map[code] = dm.get_stock_name_wave(code)
                except Exception:
                    names_map[code] = code
        except Exception as e:
            errors.append((code, str(e)))

    if not frames:
        return {"prices": None, "names": names_map, "errors": errors}

    prices_raw = pd.concat(frames.values(), axis=1)
    original_len = len(prices_raw)

    # Forward-fill short halts (≤5 consecutive days) common in A-shares
    prices_raw = prices_raw.ffill(limit=5)

    # Drop rows where more than 30% of stocks are missing (long suspensions)
    threshold = max(1, int(0.7 * len(prices_raw.columns)))
    prices_raw = prices_raw.dropna(thresh=threshold)
    prices_raw = prices_raw.dropna()

    trimmed = original_len - len(prices_raw)

    return {
        "prices":  prices_raw,
        "names":   names_map,
        "errors":  errors,
        "trimmed": trimmed,
    }


def run_all_pairs(
    prices: pd.DataFrame,
    names: dict,
    z_window: int,
    ols_window: int,
) -> tuple[list, list]:
    """
    Run analyse_pair for every unique combination.
    Returns (results, warnings) where warnings are pairs that failed.
    """
    log_p    = np.log(prices)
    results  = []
    warnings = []

    for a, b in itertools.combinations(list(prices.columns), 2):
        try:
            r = analyse_pair(a, b, log_p, z_window=z_window, ols_window=ols_window)
            results.append(r)
        except Exception as e:
            warnings.append(f"{a}/{b}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results, warnings


# ──────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

if "pt_preload_tickers" in st.session_state:
    st.session_state["pt_ticker_input"] = st.session_state.pop("pt_preload_tickers")

if "pt_ticker_input" not in st.session_state:
    st.session_state["pt_ticker_input"] = "600312\n600406\n603556\n002270\n002028\n600089"

if st.session_state.pop("pt_from_sector", False):
    sector_name = st.session_state.pop("pt_from_sector_name", "Sector Stock Selector")
    st.info(
        f"📊 Stocks pre-loaded from **{sector_name}** — "
        "review the list below and click **🔍 Run Walk-Forward Analysis**."
    )

st.title("🔗 Pair Trader  配对交易 (OOS Engine V3)")
st.caption(
    "Walk-forward out-of-sample backtest engine for A-share mean reversion. "
    "Hedge ratios are estimated using only past data — no lookahead bias."
)

col_in, col_cfg = st.columns([2, 1])

with col_in:
    st.subheader("📥 Stock Input  股票输入")
    ticker_raw = st.text_area(
        "Enter up to 10 A-share codes",
        key="pt_ticker_input",
        height=160,
        help="6-digit codes only. One per line or comma-separated. Max 10 stocks.",
    )

with col_cfg:
    st.subheader("⚙️ Settings  参数")

    # Default: 3 years back to ensure adequate warm-up + OOS period
    if "pt_start_date" not in st.session_state:
        st.session_state["pt_start_date"] = _beijing_today() - _dt.timedelta(days=365 * 3)

    start_date_val = st.date_input(
        "Data start (includes OLS warm-up)",
        value=st.session_state["pt_start_date"],
        min_value=_dt.date(2015, 1, 1),
        max_value=_beijing_today() - _dt.timedelta(days=60),
        help="Set at least OLS window + 1 year before today. Default is 3 years.",
    )
    if start_date_val != st.session_state["pt_start_date"]:
        st.session_state["pt_start_date"] = start_date_val

    ols_window = st.select_slider(
        "Rolling OLS Window (trading days)",
        options=[126, 252, 504],
        value=252,
        help="Days of history used to estimate β. 252 ≈ 1 year is standard.",
    )
    z_window = st.select_slider(
        "Z-score window (trading days)",
        options=[20, 40, 60, 90, 120],
        value=60,
        help="Rolling window for Z-score. 60d ≈ 3 months is standard.",
    )
    top_n = st.radio(
        "Show top N pairs",
        options=[1, 2, 3],
        index=1,
        horizontal=True,
    )

run_btn = st.button(
    "🔍 Run Walk-Forward Analysis  开始分析",
    type="primary",
    use_container_width=True,
)

if run_btn:
    raw = [t.strip().replace(",", "").replace("，", "")
           for t in ticker_raw.replace(",", "\n").splitlines()]
    tickers_clean = [t for t in raw if t.isdigit() and len(t) == 6][:10]

    if len(tickers_clean) < 2:
        st.error("Need at least 2 valid 6-digit codes.")
        st.stop()

    start_str = st.session_state["pt_start_date"].strftime("%Y%m%d")
    n_combos  = len(list(itertools.combinations(tickers_clean, 2)))
    with st.spinner(f"Fetching data for {len(tickers_clean)} stocks ({n_combos} pairs)…"):
        st.session_state["pt_raw"] = fetch_raw_prices(tuple(tickers_clean), start_str)

if "pt_raw" not in st.session_state:
    st.info("Press **Run Walk-Forward Analysis** to start.")
    st.stop()

_raw   = st.session_state["pt_raw"]
prices = _raw["prices"]
names  = _raw["names"]
errors = _raw["errors"]
trimmed = _raw.get("trimmed", 0)

if errors:
    with st.expander(f"⚠️ {len(errors)} fetch error(s)"):
        for code, msg in errors:
            st.write(f"`{code}`: {msg}")

if trimmed > 10:
    st.warning(
        f"⚠️ {trimmed} rows were trimmed from the dataset due to missing data "
        "(trading halts, suspensions). This is normal for A-shares."
    )

if prices is None or prices.empty:
    st.error("Could not fetch data. Check codes and try again.")
    st.stop()

pairs, pair_warnings = run_all_pairs(prices, names, z_window, ols_window)

if pair_warnings:
    with st.expander(f"⚠️ {len(pair_warnings)} pair(s) failed analysis"):
        for w in pair_warnings:
            st.caption(w)

if len(pairs) == 0:
    st.error(
        "No valid pairs found. This usually means the dataset is too short "
        "to satisfy the OLS warm-up period. Move the start date further back "
        "or reduce the OLS window."
    )
    st.stop()

bj_today  = _beijing_today()
oos_start = prices.index[ols_window] if len(prices) > ols_window else prices.index[0]
st.markdown(
    f"Analysed **{len(pairs)} pairs** from **{len(prices.columns)} stocks**  "
    f"· Raw data: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}  "
    f"· OOS period starts: **{oos_start.strftime('%Y-%m-%d')}**  "
    f"· Beijing date: **{bj_today}**"
)

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 All Pairs Ranked  全部配对排名")

with st.expander("What do these columns mean?  各列含义", expanded=False):
    st.markdown("""
| Column | Meaning |
|---|---|
| **EG p** | Engle-Granger cointegration p-value. < 0.10 = cointegrated ✓ |
| **ADF p** | ADF test on the OOS spread. < 0.10 = stationary ✓ |
| **Hurst** | R/S Hurst exponent on OOS spread levels. < 0.45 = mean-reverting ✓ |
| **HL (days)** | OU half-life: expected days for spread to revert halfway to mean |
| **Z now** | Current OOS Z-score. ±2σ = buy signal |
| **Corr** | Pearson correlation of daily log-returns over OOS period |
| **Score** | Composite quality score (max ~11). Higher = better pair |
| **β now** | Today's rolling hedge ratio (units of B per unit of A) |
    """)

table_rows = []
for r in pairs:
    a, b   = r["code_a"], r["code_b"]
    na, nb = names.get(a, a), names.get(b, b)
    sig, buy_c, reduce_c = signal_for_pair(r)
    if sig == "BUY_A":
        sig_txt = f"✅ BUY {a} {na}"
    elif sig == "BUY_B":
        sig_txt = f"✅ BUY {b} {nb}"
    elif sig == "WATCH":
        sig_txt = "👁 Watch (approaching ±2σ)"
    else:
        sig_txt = "⏸ Neutral"

    flags = (("✓coint " if r["coint_ok"] else "       ") +
             ("✓adf "   if r["adf_ok"]   else "     ") +
             ("✓H "     if r["hurst_ok"] else "   ") +
             ("✓HL"     if r["hl_ok"]    else "   "))

    table_rows.append({
        "Pair":   f"{a}/{b}",
        "Names":  f"{na} / {nb}",
        "EG p":   r["eg_p"],
        "ADF p":  r["adf_p"],
        "Hurst":  r["hurst"],
        "HL (d)": r["half_life"],
        "β now":  r["beta_now"],
        "Z now":  r["z_now"],
        "Corr":   r["corr"],
        "Score":  r["score"],
        "Signal": sig_txt,
        "Flags":  flags,
    })

df_table = pd.DataFrame(table_rows)

def colour_row(row):
    score = row["Score"]
    if score >= 7:   bg = "background-color: rgba(5,150,105,0.10)"
    elif score >= 4: bg = "background-color: rgba(217,119,6,0.08)"
    else:            bg = ""
    return [bg] * len(row)

def colour_z(val):
    try:
        v = float(val)
        if abs(v) >= 2.0: return "color: #059669; font-weight:600"
        if abs(v) >= 1.5: return "color: #d97706"
    except Exception:
        pass
    return ""

def colour_p(val):
    try:
        v = float(val)
        if v < 0.05: return "color: #059669; font-weight:600"
        if v < 0.10: return "color: #d97706"
        return "color: #dc2626"
    except Exception:
        pass
    return ""

styled = (
    df_table.style
    .apply(colour_row, axis=1)
    .map(colour_z, subset=["Z now"])
    .map(colour_p, subset=["EG p", "ADF p"])
    .format({
        "EG p": "{:.4f}", "ADF p": "{:.4f}",
        "Hurst": "{:.3f}", "HL (d)": "{:.1f}",
        "β now": "{:.3f}", "Z now": "{:.2f}",
        "Corr": "{:.3f}", "Score": "{:.1f}",
    })
)

st.dataframe(styled, use_container_width=True, height=min(400, 60 + 35 * len(pairs)))

# ──────────────────────────────────────────────────────────────────────────────
# VERDICT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🏆 Verdict  交易结论")

# With this:
top_pairs = pairs[:top_n]   # already sorted by score descending

# Optionally warn if the best pair has weak stats
best = top_pairs[0]
if not best["adf_ok"]:
    st.warning(
        "⚠️ Top pairs did not pass ADF stationarity (p < 0.10). "
        "The OOS period may be too short for reliable tests. "
        "Consider moving the start date further back."
    )

for rank, r in enumerate(top_pairs, 1):
    a, b   = r["code_a"], r["code_b"]
    na, nb = names.get(a, a), names.get(b, b)
    sig, buy_c, reduce_c = signal_for_pair(r)
    buy_name    = names.get(buy_c, buy_c)
    reduce_name = names.get(reduce_c, reduce_c)

    if sig in ("BUY_A", "BUY_B"):
        action_md = (
            f"### ✅ BUY NOW — {buy_c} {buy_name}\n"
            f"Z = **{r['z_now']:+.2f}σ** — {buy_c} is cheap relative to {reduce_c}.\n\n"
            f"**Action:** Buy **{buy_c} {buy_name}**  "
            f"(optionally reduce exposure to {reduce_c} {reduce_name})\n\n"
            f"**Position sizing:** β = {r['beta_now']:.3f} — "
            f"for every 1 unit of {buy_c}, you are implicitly referencing {r['beta_now']:.3f} units of {reduce_c}\n\n"
            f"**Exit:** When Z reverts to 0 · Expected: ~{r['half_life']:.0f} trading days\n\n"
            f"**Stop:** If Z widens beyond ±3σ — relationship may be breaking down"
        )
        action_color = "#059669"
    elif sig == "WATCH":
        action_md = (
            f"### 👁 WATCH — Approaching Entry\n"
            f"Z = **{r['z_now']:+.2f}σ** — almost at the ±2σ trigger.\n\n"
            f"Prepare to buy the laggard when Z crosses ±2σ. "
            f"Half-life: ~{r['half_life']:.0f} days once entered."
        )
        action_color = "#d97706"
    else:
        action_md = (
            f"### ⏸ NEUTRAL — Wait\n"
            f"Z = **{r['z_now']:+.2f}σ** — spread is near its mean.\n\n"
            f"Good statistical structure. Wait for Z to reach ±2σ before buying. "
            f"Half-life: ~{r['half_life']:.0f} days."
        )
        action_color = "#94a3b8"

    with st.container():
        st.markdown(
            f"<div style='padding:16px; border-radius:10px; border:1px solid {action_color}44; "
            f"background:{action_color}11; margin-bottom:12px;'>"
            f"<b>#{rank} &nbsp; {a} {na}  ↔  {b} {nb}</b><br>"
            f"EG p={r['eg_p']:.4f} · ADF p={r['adf_p']:.4f} · "
            f"H={r['hurst']:.3f} · HL={r['half_life']:.0f}d · "
            f"β={r['beta_now']:.3f} · Score={r['score']:.1f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(action_md)
        st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# DETAILED CHARTS
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Detailed Pair Charts  详细配对图表")

expander_how_it_works()
expander_oos()
expander_cointegration()
expander_adf()
expander_hurst()
expander_halflife()
expander_zscore()

for rank, r in enumerate(top_pairs, 1):
    a, b   = r["code_a"], r["code_b"]
    na, nb = names.get(a, a), names.get(b, b)
    clr_a, clr_b = PAIR_COLORS[(rank - 1) % len(PAIR_COLORS)]

    st.markdown(f"### Pair #{rank}  ·  {a} {na}  ↔  {b} {nb}")

    tab_main, tab_scatter, tab_drift, tab_trades = st.tabs([
        "📊 4-Panel Overview",
        "🔵 Log-Price Scatter",
        "🌊 Structural Drift (β)",
        "📋 Trade History & P&L",
    ])

    with tab_main:
        fig, trades = chart_pair_full(r, prices, na, nb, clr_a, clr_b, z_window)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_main_{rank}")
        with st.expander("How to read this chart  如何解读"):
            st.markdown(f"""
**① Normalised Price:** Both stocks indexed to 100 at the OOS start date.
Divergence between the lines = the trading opportunity. Triangles = buy entries, squares = exits.

**② OOS Spread:** The residual after removing the rolling hedge ratio relationship.
Dashed lines = ±1σ and ±2σ bands. Entry when spread touches ±2σ.
This spread is truly out-of-sample — β was never estimated with future data.

**③ Z-Score:** The spread normalised to standard deviations.
- ≤ **−2σ**: BUY {a} — {a} is cheap relative to {b}
- ≥ **+2σ**: BUY {b} — {b} is cheap relative to {a}
- Returns to **0**: exit and take profit
- Current Z = **{r['z_now']:+.2f}σ**

**④ Rolling {z_window}-day Correlation:** Below 0.3 for extended periods = relationship weakening.
            """)

    with tab_scatter:
        st.plotly_chart(
            chart_scatter_regression(r, prices, na, nb, clr_a, clr_b),
            use_container_width=True, key=f"chart_scatter_{rank}",
        )
        with st.expander("How to read this chart  如何解读"):
            st.markdown(f"""
Each dot is one OOS trading day. Colour progresses from dark (early) to bright (recent) via Viridis scale.

The dashed line is a **static OLS fit for visualisation only** — the actual spread
and Z-score use the rolling β from the main chart.

- **Tight linear cluster** = stable relationship over time ✅
- **Dots drifting away from the line over time** = structural shift ⚠️
- **Outlier dots far from the line** = high Z-score days — the entry opportunities
            """)

    with tab_drift:
        st.plotly_chart(
            chart_rolling_beta(r, na, nb),
            use_container_width=True, key=f"chart_drift_{rank}",
        )
        st.info(
            "A flat or gently oscillating β line means the relative sizing of "
            "the two stocks has been consistent — the pair is structurally stable. "
            "A strongly trending β (up or down) means one stock is gaining or losing "
            "relative importance — the pair may be breaking down."
        )

    with tab_trades:
        st.markdown(f"#### OOS Historical Trades — {a} {na} ↔ {b} {nb}")
        st.caption(
            "All entries and exits are **out-of-sample** — β was estimated "
            "from past data only at each point in time. "
            "P&L is on the bought stock (buy leg) only. No transaction costs included."
        )
        trades_tab = _detect_trades(r["z_series"], r["valid_dates"], prices, a, b)
        render_trade_table(trades_tab, na, nb, a, b)

    st.markdown("---")
