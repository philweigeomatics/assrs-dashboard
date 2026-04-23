"""
Pair Trader  —  ASSRS V2
────────────────────────
Enter up to 10 A-share codes. The engine tests every unique pair for
cointegration, stationarity, and mean-reversion quality, then surfaces
the top tradeable pairs with full charts and plain-English explanations.

All statistical logic is explained inline — no black boxes.

A-SHARE NOTE: Short-selling is restricted in A-shares.
This page uses a BUY-ONLY interpretation:
  Z ≤ −2σ  →  BUY the laggard (Stock A is cheap relative to B — buy A, reduce/avoid B)
  Z ≥ +2σ  →  BUY the laggard (Stock B is cheap relative to A — buy B, reduce/avoid A)
The "reduce/avoid" leg is NOT a short — it just means you hold more of the cheap one.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import data_manager as dm
from datetime import datetime, timedelta
import datetime as _dt

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
PAIR_COLORS = [
    ("#2dd4bf", "#f59e0b"),   # pair 1: teal / amber
    ("#a78bfa", "#60a5fa"),   # pair 2: violet / blue
    ("#f87171", "#34d399"),   # pair 3: red / green (if ever needed)
]

# ──────────────────────────────────────────────────────────────────────────────
# BEIJING TIME helper — Fix 3
# ──────────────────────────────────────────────────────────────────────────────
def _beijing_today() -> _dt.date:
    """Return today's date in Asia/Shanghai timezone."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime as _datetime
        return _datetime.now(ZoneInfo("Asia/Shanghai")).date()
    except Exception:
        # fallback: UTC+8 offset
        return (_dt.datetime.utcnow() + _dt.timedelta(hours=8)).date()

# ──────────────────────────────────────────────────────────────────────────────
# PURE MATH  (no Streamlit — safe to cache)
# ──────────────────────────────────────────────────────────────────────────────

def _hurst_rs(series: np.ndarray) -> float:
    """
    R/S Hurst exponent on a price series.
    H < 0.5  →  mean-reverting spread (ideal for pairs)
    H = 0.5  →  random walk
    H > 0.5  →  trending
    """
    n = len(series)
    if n < 20:
        return 0.5
    lr = np.diff(np.log(np.maximum(series, 1e-9)))
    mean_r = lr.mean()
    cumdev = np.cumsum(lr - mean_r)
    R = cumdev.max() - cumdev.min()
    S = lr.std()
    return math.log(R / S) / math.log(n) if S > 1e-9 else 0.5


def _zscore_series(spread: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Rolling Z-score of the spread.
    Z = (spread − rolling_mean) / rolling_std
    Entry signal when |Z| > 2.
    """
    s = pd.Series(spread)
    rm = s.rolling(window).mean()
    rs = s.rolling(window).std()
    return ((s - rm) / rs).values


def _half_life(spread: np.ndarray) -> float:
    """
    Half-life of mean reversion via AR(1).
    Answers: 'if we enter now, how many days until the spread closes halfway?'
    """
    ar = pd.Series(spread).autocorr(1)
    if 0 < abs(ar) < 1:
        return -math.log(2) / math.log(abs(ar))
    return 999.0


def analyse_pair(
    code_a: str, code_b: str,
    log_prices: pd.DataFrame,
    z_window: int = 60,
) -> dict:
    """
    Full statistical analysis for one pair.
    Returns a dict with all metrics and series needed for charts.
    """
    from statsmodels.tsa.stattools import adfuller, coint

    ya = log_prices[code_a].values
    xb = log_prices[code_b].values

    # ── Engle-Granger cointegration test ──────────────────────────────────────
    # Tests whether two log-price series share a common stochastic trend.
    # p < 0.10 means the pair is cointegrated — spreads must revert.
    eg_stat, eg_p, _ = coint(ya, xb)

    # ── OLS hedge ratio ───────────────────────────────────────────────────────
    # How many units of B per unit of A to keep the position dollar-neutral.
    slope, intercept, _, _, _ = stats.linregress(xb, ya)
    spread = ya - (slope * xb + intercept)

    # ── ADF stationarity test on spread ───────────────────────────────────────
    # Confirms the spread itself is stationary (mean-reverting), not just the stocks.
    adf_stat, adf_p, *_ = adfuller(spread, maxlag=10, autolag="AIC")

    # ── Hurst exponent on spread ──────────────────────────────────────────────
    proxy = np.exp(spread - spread.mean()) + 1
    h = _hurst_rs(proxy)

    # ── Pearson correlation of daily log returns ───────────────────────────────
    ra = np.diff(ya)
    rb = np.diff(xb)
    corr = float(np.corrcoef(ra, rb)[0, 1])

    # ── Half-life ─────────────────────────────────────────────────────────────
    hl = _half_life(spread)

    # ── Rolling Z-score ───────────────────────────────────────────────────────
    z = _zscore_series(spread, window=z_window)
    z_now = float(z[-1]) if not np.isnan(z[-1]) else 0.0

    # ── Composite quality score ───────────────────────────────────────────────
    score = (
        (3 if eg_p  < 0.10 else 1 if eg_p  < 0.20 else 0) +
        (2 if adf_p < 0.10 else 1 if adf_p < 0.15 else 0) +
        (2 if h     < 0.45 else 1 if h     < 0.50 else 0) +
        (1 if 5 <= hl <= 30 else 0) +
        (2 if abs(z_now) >= 2.0 else 1 if abs(z_now) >= 1.5 else 0) +
        min(corr, 1.0)
    )

    return {
        "code_a": code_a, "code_b": code_b,
        "eg_p":     round(eg_p,  4),
        "adf_p":    round(adf_p, 4),
        "hurst":    round(h,     3),
        "corr":     round(corr,  3),
        "half_life": round(hl,   1),
        "hedge_ratio": round(slope, 3),
        "z_now":    round(z_now, 2),
        "score":    round(score, 2),
        "spread":   spread,
        "z_series": z,
        "coint_ok": eg_p  < 0.10,
        "adf_ok":   adf_p < 0.10,
        "hurst_ok": h     < 0.45,
        "hl_ok":    5 <= hl <= 30,
        "tradeable": abs(z_now) >= 1.5,
    }


def signal_for_pair(result: dict) -> tuple[str, str, str]:
    """
    A-share buy-only interpretation.
    Returns (signal, buy_code, reduce_code).
    signal: 'BUY_A'  — A is cheap (Z ≤ −2): buy A, reduce B
            'BUY_B'  — B is cheap (Z ≥ +2): buy B, reduce A
            'WATCH'  — approaching trigger
            'NEUTRAL'
    """
    z = result["z_now"]
    a, b = result["code_a"], result["code_b"]
    if z <= -2.0:
        return "BUY_A", a, b    # A underperformed → buy A
    elif z >= 2.0:
        return "BUY_B", b, a    # B underperformed → buy B
    elif abs(z) >= 1.5:
        return "WATCH", a, b
    return "NEUTRAL", a, b


# ──────────────────────────────────────────────────────────────────────────────
# TRADE DETECTION + P&L
# ──────────────────────────────────────────────────────────────────────────────

def _detect_trades(z: np.ndarray, dates, prices: pd.DataFrame,
                   code_a: str, code_b: str,
                   entry_thresh: float = 2.0) -> list:
    """
    Scan Z-score series and return all historical trade intervals.

    Entry:  Z crosses ±entry_thresh
    Exit:   Z crosses 0 (convergence)

    A-share interpretation:
      direction='BUY_A'  Z ≤ −2 → buy A (A was cheap)
      direction='BUY_B'  Z ≥ +2 → buy B (B was cheap)

    P&L is computed on the BUY leg only (the stock you actually buy),
    from entry close price to exit close price.
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
                in_trade = True; entry_date = dt
                entry_z  = zc;   direction  = "BUY_A"
            elif zp < entry_thresh and zc >= entry_thresh:
                in_trade = True; entry_date = dt
                entry_z  = zc;   direction  = "BUY_B"
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


def _make_trade(entry_date, exit_date, entry_z, exit_z,
                direction, is_open, prices, code_a, code_b) -> dict:
    """Build a trade dict with P&L on the bought stock."""
    buy_code = code_a if direction == "BUY_A" else code_b

    # Get prices at entry and exit for the bought stock
    try:
        col = buy_code
        ei = prices.index.get_indexer([entry_date], method="nearest")[0]
        xi = prices.index.get_indexer([exit_date],  method="nearest")[0]
        entry_price = float(prices[col].iloc[ei])
        exit_price  = float(prices[col].iloc[xi])
        pnl_pct = (exit_price / entry_price - 1) * 100.0
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
        "pnl_pct":     round(pnl_pct, 2)    if not math.isnan(pnl_pct)     else None,
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
    4-panel chart for one pair. Returns (fig, trades).
      Row 1 — Normalised price (base=100): visual divergence
      Row 2 — Log-price spread with ±1σ/±2σ bands
      Row 3 — Rolling Z-score with entry/exit markers
      Row 4 — Rolling N-day correlation of daily returns
    """
    a, b    = result["code_a"], result["code_b"]
    dates   = prices.index
    spread  = result["spread"]
    z       = result["z_series"]
    z_now   = result["z_now"]

    na = prices[a].values / prices[a].values[0] * 100
    nb = prices[b].values / prices[b].values[0] * 100

    log_a = np.log(prices[a].values)
    log_b = np.log(prices[b].values)
    ra = pd.Series(np.diff(log_a), index=dates[1:])
    rb = pd.Series(np.diff(log_b), index=dates[1:])
    roll_corr = ra.rolling(z_window).corr(rb)

    sp_mean = spread.mean()
    sp_std  = spread.std()

    # Detect trades (needs raw prices for P&L)
    trades = _detect_trades(z, dates, prices, a, b)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.22, 0.28, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            "① Normalised Price  (base = 100 at start)",
            f"② Log-Price Spread  (ratio: 1× {a} vs {result['hedge_ratio']:.3f}× {b})",
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

    # ── Row 2: spread — Fix 1: use dark slate instead of "white" ─────────────
    fig.add_trace(go.Scatter(
        x=dates, y=spread, name="Spread", showlegend=False,
        line=dict(color="#334155", width=1.5),          # dark slate — visible on both themes
        hovertemplate="Spread<br>%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=sp_mean, line_dash="dot",
                  line_color="rgba(51,65,85,0.6)", row=2, col=1)
    for sig_v, clr in [(1, "#ca8a04"), (2, "#dc2626")]:   # darker yellow/red
        for sign in [1, -1]:
            fig.add_hline(y=sp_mean + sign * sig_v * sp_std,
                          line_dash="dash", line_color=clr,
                          line_width=0.9, row=2, col=1)

    # ── Row 3: Z-score ────────────────────────────────────────────────────────
    fig.add_hrect(y0=2,  y1=5,  fillcolor="rgba(239,68,68,0.08)",  line_width=0, row=3, col=1)
    fig.add_hrect(y0=-5, y1=-2, fillcolor="rgba(16,185,129,0.08)", line_width=0, row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=z, name="Z-score", showlegend=False,
        line=dict(color="#d97706", width=2),            # amber — visible on both themes
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
        line=dict(color="#7c3aed", width=1.5),          # violet — visible on both
        hovertemplate="Corr<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ), row=4, col=1)
    fig.add_hline(y=0.5, line_dash="dot",
                  line_color="rgba(100,116,139,0.5)", row=4, col=1)
    fig.add_hline(y=0,   line_color="rgba(100,116,139,0.3)", row=4, col=1)

    # ── Trade overlays ────────────────────────────────────────────────────────
    for i, t in enumerate(trades, 1):
        is_open  = t.get("open", False)
        buy_a    = t["direction"] == "BUY_A"
        shade    = "rgba(16,185,129,0.09)"  if buy_a else "rgba(239,68,68,0.09)"
        lbl_clr  = "#059669"                if buy_a else "#dc2626"
        buy_code = t["buy_code"]
        buy_name = name_a if buy_a else name_b
        trade_lbl = f"#{i} BUY {buy_code} {buy_name}"

        # Shade all 4 rows
        for row in [1, 2, 3, 4]:
            fig.add_vrect(x0=t["entry"], x1=t["exit"],
                          fillcolor=shade, line_width=0,
                          row=row, col=1)

        # Entry circle on Z panel
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

        # Exit marker on Z panel
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

        # Entry triangle on price panel
        pidx_e = prices.index.get_indexer([t["entry"]], method="nearest")[0]
        pidx_x = prices.index.get_indexer([t["exit"]],  method="nearest")[0]
        # Show triangle on the stock that was bought
        y_series = na if buy_a else nb
        ep = float(y_series[pidx_e])
        xp = float(y_series[pidx_x])

        fig.add_trace(go.Scatter(
            x=[t["entry"]], y=[ep],
            mode="markers",
            marker=dict(symbol="triangle-up", color=lbl_clr, size=10,
                        line=dict(color="white", width=1)),
            showlegend=False, legendgroup=f"trade{i}",
            hovertemplate=(f"BUY #{i}<br>%{{x|%Y-%m-%d}}<br>"
                           f"{buy_code} index: %{{y:.1f}}<extra></extra>"),
        ), row=1, col=1)

        if not is_open:
            fig.add_trace(go.Scatter(
                x=[t["exit"]], y=[xp],
                mode="markers",
                marker=dict(symbol="square", color=lbl_clr, size=9,
                            line=dict(color="#1e293b", width=1.5)),
                showlegend=False, legendgroup=f"trade{i}",
                hovertemplate=(f"SELL #{i}{pnl_str}<br>%{{x|%Y-%m-%d}}<br>"
                               f"{buy_code} index: %{{y:.1f}}<extra></extra>"),
            ), row=1, col=1)

    n_closed = sum(1 for t in trades if not t.get("open"))
    n_open   = sum(1 for t in trades if t.get("open"))
    trade_note = (f"  ·  {len(trades)} trades "
                  f"({n_closed} closed · {n_open} open)")

    fig.update_layout(
        height=860, template="plotly_dark",
        title=(f"{a} {name_a}  ↔  {b} {name_b}  |  "
               f"ADF p={result['adf_p']:.3f}  HL={result['half_life']:.0f}d  "
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
    Log-price scatter of A vs B with OLS regression line.
    The slope is the hedge ratio. Tight cluster = stable relationship.
    """
    a, b = result["code_a"], result["code_b"]
    xa = np.log(prices[b].values)
    ya = np.log(prices[a].values)
    dates = prices.index

    slope, intercept, _, _, _ = stats.linregress(xa, ya)
    x_line = np.linspace(xa.min(), xa.max(), 100)
    y_line = slope * x_line + intercept

    n = len(xa)
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
        name="Daily observations",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#334155", width=1.5, dash="dash"),
        name=f"OLS  slope={slope:.3f}",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"Log-Price Scatter: {b} (x) vs {a} (y)<br>"
              f"<sup>Ratio = {slope:.3f}  ·  tighter cluster = more stable relationship</sup>",
        xaxis_title=f"log price of {b} {name_b}",
        yaxis_title=f"log price of {a} {name_a}",
        height=400, template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# TRADE P&L TABLE
# ──────────────────────────────────────────────────────────────────────────────

def render_trade_table(trades: list, name_a: str, name_b: str,
                       code_a: str, code_b: str) -> None:
    """
    Render a styled per-trade P&L table under each pair chart.
    Closed trades show actual P&L%; open trade shows 'Open'.
    Summary row: Win rate, avg win, avg loss, total.
    """
    if not trades:
        st.caption("No historical trades detected in this period.")
        return

    rows = []
    for i, t in enumerate(trades, 1):
        buy_code = t["buy_code"]
        buy_name = name_a if buy_code == code_a else name_b
        side_label = f"BUY {buy_code} {buy_name}"

        if t.get("open"):
            pnl_str = "⏳ Open"
            duration = (t["exit"] - t["entry"]).days
        else:
            p = t.get("pnl_pct")
            pnl_str = f"{p:+.2f}%" if p is not None else "—"
            duration = (t["exit"] - t["entry"]).days

        rows.append({
            "#": i,
            "Action": side_label,
            "Entry Date": t["entry"].strftime("%Y-%m-%d"),
            "Entry Z": f"{t['entry_z']:+.2f}σ",
            "Entry Price": f"{t['entry_price']}" if t.get("entry_price") else "—",
            "Exit Date": t["exit"].strftime("%Y-%m-%d"),
            "Exit Z": f"{t['exit_z']:+.2f}σ",
            "Exit Price": f"{t['exit_price']}" if t.get("exit_price") else "—",
            "Duration": f"{duration}d",
            "P&L %": pnl_str,
            "Status": "Open" if t.get("open") else "Closed",
        })

    df = pd.DataFrame(rows)

    # Colour P&L column
    def colour_pnl(val):
        if val == "⏳ Open":
            return "color: #d97706; font-weight:600"
        try:
            v = float(val.replace("%", "").replace("+", ""))
            if v > 0:  return "color: #059669; font-weight:600"
            if v < 0:  return "color: #dc2626; font-weight:600"
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
              .applymap(colour_pnl, subset=["P&L %"])
              .hide(axis="index"))

    st.dataframe(styled, use_container_width=True,
                 height=min(500, 45 + 36 * len(rows)))

    # ── Summary stats ──────────────────────────────────────────────────────
    closed = [t for t in trades if not t.get("open") and t.get("pnl_pct") is not None]
    if closed:
        wins   = [t["pnl_pct"] for t in closed if t["pnl_pct"] > 0]
        losses = [t["pnl_pct"] for t in closed if t["pnl_pct"] <= 0]
        win_rate  = len(wins) / len(closed) * 100
        avg_win   = sum(wins)   / len(wins)   if wins   else 0.0
        avg_loss  = sum(losses) / len(losses) if losses else 0.0
        total_pnl = sum(t["pnl_pct"] for t in closed)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate",  f"{win_rate:.0f}%",
                  help="% of closed trades that were profitable")
        c2.metric("Avg Win",   f"{avg_win:+.2f}%",
                  help="Average return on winning trades")
        c3.metric("Avg Loss",  f"{avg_loss:+.2f}%",
                  help="Average return on losing trades")
        c4.metric("Total P&L", f"{total_pnl:+.2f}%",
                  help="Sum of all closed trade returns (not compounded)")


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

**Why it works in A-shares:**
Same-sector stocks share policy tailwinds, institutional investors, and macro exposure.
When retail flow or short-term news pushes one stock disproportionately, the
fundamental anchor tends to pull them back together.
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

**Important:** Cointegration is a necessary but not sufficient condition.
You also need the spread to revert *quickly enough* (see Half-Life).
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

Reject $H_0$ (p < 0.10) → spread is stationary → mean reversion is real.
        """)


def expander_hurst():
    with st.expander("③ Hurst Exponent  赫斯特指数", expanded=False):
        st.markdown(r"""
**What it measures:** The persistence of the spread series.

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

$$\text{half-life} = \frac{-\ln(2)}{\ln(|\phi_1|)}$$

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

**Exit rule:** Sell when Z reverts to 0 (spread back to mean).
**Stop rule:** If spread widens further to ±3σ, cut the position — the relationship may be breaking down.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCH  (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: tuple, start_date: str) -> dict:
    """
    Fetch daily qfq-adjusted close prices for all tickers.
    end_date is determined by the data_manager (Beijing time).
    Returns {'prices': DataFrame, 'names': dict, 'errors': list}
    """
    frames, names_map, errors = {}, {}, []

    for code in tickers:
        try:
            df = dm.get_ohlcv_for_wave(code, granularity="Daily", start_date=start_date)
            if df is None or df.empty:
                errors.append((code, "No data returned"))
                continue
            frames[code] = df["Close"].rename(code)
            try:
                names_map[code] = dm.get_stock_name_wave(code)
            except Exception:
                names_map[code] = code
        except Exception as e:
            errors.append((code, str(e)))

    if not frames:
        return {"prices": None, "names": names_map, "errors": errors}

    prices = pd.concat(frames.values(), axis=1).dropna()
    return {"prices": prices, "names": names_map, "errors": errors}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_raw_prices(tickers: tuple, start_date: str) -> dict:
    """
    Fetch prices only — no z_window dependency so this stays cached
    when the user adjusts the Z-score slider or top-N radio.
    Returns {'prices': DataFrame, 'names': dict, 'errors': list}
    """
    return fetch_prices(tickers, start_date)


def run_all_pairs(prices: pd.DataFrame, names: dict, z_window: int) -> list:
    """
    Pure computation — runs on already-fetched prices.
    Called live on every z_window change (cheap, no API calls).
    """
    log_p   = np.log(prices)
    results = []
    for a, b in itertools.combinations(list(prices.columns), 2):
        try:
            r = analyse_pair(a, b, log_p, z_window=z_window)
            results.append(r)
        except Exception:
            pass
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

st.title("🔗 Pair Trader  配对交易")
st.caption(
    "Buy the laggard. Enter up to 10 same-sector A-share codes — the engine "
    "identifies the best cointegrated pairs and tells you which stock to buy now."
)

# ── Input panel ───────────────────────────────────────────────────────────────
with st.container():
    col_in, col_cfg = st.columns([2, 1])

    with col_in:
        st.subheader("📥 Stock Input  股票输入")
        ticker_raw = st.text_area(
            "Enter up to 10 A-share codes (one per line or comma-separated)",
            value="600312\n600406\n603556\n002270\n002028\n600089",
            height=160,
            placeholder="600312\n600406\n...",
            help="6-digit codes only. Exchange suffix added automatically. Max 10 stocks = max 45 pairs.",
        )

    with col_cfg:
        st.subheader("⚙️ Settings  参数")

        # Fix 2: persist start_date in session state so Z-window changes don't reset it
        if "pt_start_date" not in st.session_state:
            st.session_state["pt_start_date"] = (
                _beijing_today() - _dt.timedelta(days=365 * 2)
            )

        start_date_val = st.date_input(
            "Data start  数据起始",
            value=st.session_state["pt_start_date"],
            min_value=_dt.date(2015, 1, 1),
            max_value=_beijing_today() - _dt.timedelta(days=60),
            key="pt_start_date_input",
            help="At least 1 year recommended for reliable cointegration tests.",
        )
        # Sync back to session state only when it actually changes
        if start_date_val != st.session_state["pt_start_date"]:
            st.session_state["pt_start_date"] = start_date_val

        z_window = st.select_slider(
            "Z-score window (trading days)",
            options=[20, 40, 60, 90, 120],
            value=60,
            help="Rolling window for Z-score calculation. 60d ≈ 3 months is standard.",
        )
        top_n = st.radio(
            "Show top N pairs",
            options=[1, 2, 3],
            index=1,
            horizontal=True,
            help="How many of the best pairs to display in full detail.",
        )

    run_btn = st.button("🔍 Find Best Pairs  开始分析", type="primary", use_container_width=True)

# ── Session state strategy ────────────────────────────────────────────────────
# "pt_raw"       : fetched prices DataFrame — only refreshed on button press
# "pt_start_date": the start date used for the last fetch (for display)
# pairs/top_n    : recomputed live on every widget change (no API call, cheap)
if run_btn:
    raw = [t.strip().replace(",", "").replace("，", "")
           for t in ticker_raw.replace(",", "\n").splitlines()]
    tickers_clean = [t for t in raw if t.isdigit() and len(t) == 6][:10]

    if len(tickers_clean) < 2:
        st.error("Need at least 2 valid 6-digit codes.")
        st.stop()

    # Use the persisted start date
    start_str = st.session_state["pt_start_date"].strftime("%Y%m%d")
    n_combos  = len(list(itertools.combinations(tickers_clean, 2)))
    with st.spinner(f"Fetching daily data for {len(tickers_clean)} stocks ({n_combos} pairs)…"):
        raw_payload = fetch_raw_prices(tuple(tickers_clean), start_str)
        st.session_state["pt_raw"]        = raw_payload
        st.session_state["pt_tickers"]    = tickers_clean
        st.session_state["pt_fetch_start"] = start_str   # record what was fetched

if "pt_raw" not in st.session_state:
    st.info("Press **Find Best Pairs** to run the analysis.")
    st.stop()

# Always recompute pairs live using current widget values
_raw   = st.session_state["pt_raw"]
prices = _raw["prices"]
names  = _raw["names"]
errors = _raw["errors"]

if prices is None:
    st.error("Could not fetch data. Check codes and try again.")
    st.stop()

# Pure math — runs on every slider/radio change with no API calls
pairs = run_all_pairs(prices, names, z_window)

if errors:
    with st.expander(f"⚠️ {len(errors)} fetch error(s)"):
        for code, msg in errors:
            st.write(f"`{code}`: {msg}")

if prices is None or len(pairs) == 0:
    st.error("Could not fetch data for any tickers. Check codes and try again.")
    st.stop()

n_tickers = len(prices.columns)
n_pairs   = len(pairs)

# Fix 3: display uses Beijing today for "as of" context
bj_today = _beijing_today()
st.markdown(
    f"Analysed **{n_pairs} pairs** from **{n_tickers} stocks** "
    f"· {len(prices)} daily bars "
    f"· {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')} "
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
| **ADF p** | ADF test on the spread. < 0.10 = stationary spread ✓ |
| **Hurst** | Hurst exponent of the spread. < 0.45 = mean-reverting ✓ |
| **HL (days)** | Half-life: expected days for spread to revert halfway |
| **Z now** | Current Z-score. ±2σ = buy signal |
| **Corr** | Pearson correlation of daily returns |
| **Score** | Composite quality score (max ~10). Higher = better pair |
| **Signal** | What to do now based on current Z |
    """)

table_rows = []
for r in pairs:
    a, b   = r["code_a"], r["code_b"]
    na, nb = names.get(a, a), names.get(b, b)
    sig, buy_c, reduce_c = signal_for_pair(r)
    buy_name   = names.get(buy_c, buy_c)
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
        "Pair":    f"{a}/{b}",
        "Names":   f"{na} / {nb}",
        "EG p":    r["eg_p"],
        "ADF p":   r["adf_p"],
        "Hurst":   r["hurst"],
        "HL (d)":  r["half_life"],
        "Z now":   r["z_now"],
        "Corr":    r["corr"],
        "Score":   r["score"],
        "Signal":  sig_txt,
        "Flags":   flags,
    })

df_table = pd.DataFrame(table_rows)

def colour_row(row):
    score = row["Score"]
    if score >= 7:
        bg = "background-color: rgba(5,150,105,0.10)"
    elif score >= 4:
        bg = "background-color: rgba(217,119,6,0.08)"
    else:
        bg = ""
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

styled = (df_table.style
          .apply(colour_row, axis=1)
          .applymap(colour_z,  subset=["Z now"])
          .applymap(colour_p,  subset=["EG p", "ADF p"])
          .format({"EG p": "{:.4f}", "ADF p": "{:.4f}",
                   "Hurst": "{:.3f}", "HL (d)": "{:.1f}",
                   "Z now": "{:+.2f}", "Corr": "{:.3f}", "Score": "{:.1f}"}))

st.dataframe(styled, use_container_width=True, height=min(400, 60 + 35 * len(pairs)))

# ──────────────────────────────────────────────────────────────────────────────
# VERDICT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🏆 Verdict  交易结论")

# A pair qualifies if it passes at least ADF + one of (EG or Hurst)
qualified = [r for r in pairs if r["adf_ok"] and (r["coint_ok"] or r["hurst_ok"])]
top_pairs  = qualified[:top_n] if qualified else []

if not qualified:
    st.error(
        "**No tradeable pairs found.** None of the tested pairs passed the minimum "
        "statistical criteria (ADF p < 0.10 AND cointegration or H < 0.45). "
        "Try adding more stocks from the same sub-sector, or extend the date range."
    )
    st.stop()

for rank, r in enumerate(top_pairs, 1):
    a, b   = r["code_a"], r["code_b"]
    na, nb = names.get(a, a), names.get(b, b)
    sig, buy_c, reduce_c = signal_for_pair(r)
    buy_name    = names.get(buy_c, buy_c)
    reduce_name = names.get(reduce_c, reduce_c)
    clr_a, clr_b = PAIR_COLORS[(rank - 1) % len(PAIR_COLORS)]

    if sig in ("BUY_A", "BUY_B"):
        action_md = (
            f"### ✅ BUY NOW — {buy_c} {buy_name}\n"
            f"Z = **{r['z_now']:+.2f}σ** — {buy_c} is cheap relative to {reduce_c}.\n\n"
            f"**Action:** Buy **{buy_c} {buy_name}**  "
            f"(optionally reduce exposure to {reduce_c} {reduce_name})\n\n"
            f"**Exit:** When Z reverts to 0 (spread converges) "
            f"· Expected: ~{r['half_life']:.0f} trading days\n\n"
            f"**Stop:** If Z widens beyond ±3σ — relationship may be breaking down"
        )
        action_color = "#059669"
    elif sig == "WATCH":
        action_md = (
            f"### 👁 WATCH — Approaching Entry\n"
            f"Z = **{r['z_now']:+.2f}σ** — almost at the ±2σ trigger.\n\n"
            f"Prepare to buy the laggard when Z crosses ±2σ.\n"
            f"Half-life: ~{r['half_life']:.0f} days once entered."
        )
        action_color = "#d97706"
    else:
        action_md = (
            f"### ⏸ NEUTRAL — Wait\n"
            f"Z = **{r['z_now']:+.2f}σ** — spread is near its mean.\n\n"
            f"Good statistical structure. Wait for Z to reach ±2σ before buying.\n"
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
            f"Corr={r['corr']:.3f} · Score={r['score']:.1f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(action_md)
        st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# DETAILED CHARTS PER PAIR
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Detailed Pair Charts  详细配对图表")

# Methodology expanders
expander_how_it_works()
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

    tab_main, tab_scatter, tab_trades = st.tabs([
        "📊 4-Panel Overview", "🔵 Log-Price Scatter", "📋 Trade History & P&L"
    ])

    with tab_main:
        fig, trades = chart_pair_full(r, prices, na, nb, clr_a, clr_b, z_window)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"chart_pair_full_{rank}",
        )
        with st.expander("How to read this chart  如何解读"):
            st.markdown(f"""
**① Normalised Price:** Both stocks rescaled to 100 at the start date.
Divergence between the two lines = the trading opportunity.
Triangles mark where you buy the laggard; squares mark the exit.

**② Log-Price Spread:** The residual after removing the linear relationship.
Dashed lines = ±1σ and ±2σ bands. When spread touches ±2σ, it's an entry signal.
Ratio here is **{r['hedge_ratio']:.3f}×** — the relative size of each position.

**③ Z-Score:** The spread in standard deviations.
- ≤ **−2σ**: BUY {a} — {a} is relatively cheap
- ≥ **+2σ**: BUY {b} — {b} is relatively cheap
- Returns to **0**: sell, take profit
- Current Z = **{r['z_now']:+.2f}σ**

**④ Rolling {z_window}-day Correlation:** If correlation drops below 0.3 and stays there,
the statistical relationship may be breaking down — treat as a warning.
            """)

    with tab_scatter:
        st.plotly_chart(
            chart_scatter_regression(r, prices, na, nb, clr_a, clr_b),
            use_container_width=True,
            key=f"chart_scatter_{rank}",
        )
        with st.expander("How to read this chart  如何解读"):
            st.markdown(f"""
Each dot is one trading day plotted as log({b}) on x-axis vs log({a}) on y-axis.
Dots are coloured by time (lighter = more recent, via Viridis colour scale).

- **Tight linear cluster** = stable relationship. The ratio is reliable.
- **Scatter drifting over time** = the relationship is shifting.
- **Slope = {r['hedge_ratio']:.3f}** — for every 1% move in log({b}), log({a}) moves ~{r['hedge_ratio']:.2f}%.
- **Outlier dots far from the line** = high Z-score days — those are the entry opportunities.
            """)

    with tab_trades:
        st.markdown(f"#### Historical Trades — {a} {na} ↔ {b} {nb}")
        st.caption(
            "**BUY** = enter the position (buy the laggard). "
            "**SELL** = exit when Z reverts to 0. "
            "P&L % is on the bought stock only (buy leg)."
        )
        # Need to re-detect trades here since we only have them inside chart_pair_full
        z_arr  = r["z_series"]
        trades_tab = _detect_trades(z_arr, prices.index, prices, a, b)
        render_trade_table(trades_tab, na, nb, a, b)

    st.markdown("---")
