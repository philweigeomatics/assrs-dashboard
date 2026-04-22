"""
Wave Trader  —  ASSRS V2
────────────────────────
Enter up to 10 A-share codes. The engine fetches Daily/Weekly/Monthly OHLCV via Tushare,
runs the wave-structure analysis, scores every stock 0-100, detects phase
offsets between stocks, and recommends the best rotation trio.

All scoring logic is explained inline in the UI — no black boxes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import data_manager as dm
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (consistent across all charts)
# ──────────────────────────────────────────────────────────────────────────────
WAVE_COLORS = [
    "#2dd4bf",  # teal
    "#a78bfa",  # violet
    "#f59e0b",  # amber
    "#60a5fa",  # blue
    "#f87171",  # red
    "#34d399",  # green
    "#fb923c",  # orange
    "#e879f9",  # pink
    "#facc15",  # yellow
    "#94a3b8",  # slate
]

SIGNAL_COLORS = {
    "BUY":   ("#10b981", "✅"),
    "HOLD":  ("#94a3b8", "⏸️"),
    "WATCH": ("#f59e0b", "👁️"),
    "EXIT":  ("#ef4444", "🚨"),
}

PRESETS = {
    "我的三股 My Trio":   ["002080", "600522", "688008"],
    "军工 Defense":       ["002414", "688122", "000768"],
    "新能源电池 EV/Battery": ["300014", "002466", "002594"],
    "电网电缆 Cable/Grid":  ["600522", "600089", "002532"],
}

# ──────────────────────────────────────────────────────────────────────────────
# PURE MATH FUNCTIONS  (no Streamlit calls — safe to cache)
# ──────────────────────────────────────────────────────────────────────────────

def _hurst(closes: list[float]) -> float:
    """
    Simplified R/S Hurst exponent.
    H < 0.5  →  mean-reverting  (ideal for wave trading)
    H = 0.5  →  random walk
    H > 0.5  →  trending
    """
    n = len(closes)
    if n < 6:
        return 0.5
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, n)]
    mean_r = sum(log_rets) / len(log_rets)
    cumdev = []
    s = 0.0
    for r in log_rets:
        s += r - mean_r
        cumdev.append(s)
    R = max(cumdev) - min(cumdev)
    std = math.sqrt(sum((r - mean_r) ** 2 for r in log_rets) / len(log_rets))
    return math.log(R / std) / math.log(n) if std > 1e-9 else 0.5


def _atr_pct(highs: list[float], lows: list[float], closes: list[float],
             period: int = 5) -> list[float]:
    """
    Compute ATR as % of closing price (EWM, span=period).
    Returns a list aligned to the input bars (first few values are approximate).
    """
    n = len(closes)
    tr_vals = []
    for i in range(n):
        h, l, c = highs[i], lows[i], closes[i]
        pc = closes[i - 1] if i > 0 else c
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_vals.append(tr)

    # EWM ATR
    alpha = 2.0 / (period + 1)
    atr = [tr_vals[0]]
    for i in range(1, n):
        atr.append(alpha * tr_vals[i] + (1 - alpha) * atr[-1])

    # Convert to % of price, floor at a small value to avoid div/0
    atr_pct = [atr[i] / max(closes[i], 0.01) * 100 for i in range(n)]
    return atr_pct


def _peaks_troughs(closes: list[float],
                   highs:  list[float] = None,
                   lows:   list[float] = None,
                   n_atr:  float = 1.0,
                   min_pct: float = 5.0,
                   atr_period: int = 5) -> tuple[list[int], list[int]]:
    """
    ATR-filtered ZigZag peak/trough detector.

    A new pivot is only registered when price has moved at least
      threshold = max(n_atr × ATR%, min_pct)
    from the LAST CONFIRMED pivot (not just the adjacent bar).

    This eliminates noise, enforces alternation (peak→trough→peak),
    and adapts the threshold to current volatility regime via ATR%.

    Args:
        closes     : closing price list
        highs/lows : for ATR computation (falls back to closes if None)
        n_atr      : ATR multiplier (1.0 = use 1× current ATR% as threshold)
        min_pct    : minimum % move floor regardless of ATR (default 5%)
        atr_period : EWM span for ATR smoothing (default 5 bars)

    Returns:
        (peaks_indices, troughs_indices) — indices into the closes list
    """
    n = len(closes)
    if n < 10:
        return [], []

    # Fall back to close-only if H/L not provided
    h = highs  if highs  is not None else closes
    l = lows   if lows   is not None else closes
    atr_pct_vals = _atr_pct(h, l, closes, period=atr_period)

    pivots    = []   # (index, price, 'P'|'T')
    start     = atr_period
    last_idx  = start
    last_price= closes[start]
    direction = None   # 'up' → looking for peak, 'down' → looking for trough

    for i in range(start + 1, n):
        thresh = max(n_atr * atr_pct_vals[i], min_pct) / 100.0
        move   = (closes[i] - last_price) / last_price

        if direction != 'up' and move >= thresh:
            # Enough upward move — confirm prior extreme as trough
            pivots.append((last_idx, last_price, 'T'))
            direction  = 'up'
            last_idx   = i
            last_price = closes[i]

        elif direction != 'down' and move <= -thresh:
            # Enough downward move — confirm prior extreme as peak
            pivots.append((last_idx, last_price, 'P'))
            direction  = 'down'
            last_idx   = i
            last_price = closes[i]

        else:
            # Still moving in the same direction — extend to more extreme price
            if direction == 'up'   and closes[i] > last_price:
                last_idx, last_price = i, closes[i]
            elif direction == 'down' and closes[i] < last_price:
                last_idx, last_price = i, closes[i]
            elif direction is None and abs(move) > 0.01:
                direction = 'up' if move > 0 else 'down'

    # Append the final open pivot
    if direction == 'up':
        pivots.append((last_idx, last_price, 'P'))
    elif direction == 'down':
        pivots.append((last_idx, last_price, 'T'))

    peaks   = [idx for idx, _, t in pivots if t == 'P']
    troughs = [idx for idx, _, t in pivots if t == 'T']
    return peaks, troughs


def _monthly_returns(closes: list[float]) -> list[float]:
    if len(closes) < 2:
        return []
    return [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]


def _concurrent_corr(r1: list[float], r2: list[float]) -> float:
    """Pearson correlation at lag-0 between two return series."""
    n = min(len(r1), len(r2))
    if n < 4:
        return 0.0
    a, b = r1[:n], r2[:n]
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da  = math.sqrt(sum((x - ma) ** 2 for x in a))
    db  = math.sqrt(sum((x - mb) ** 2 for x in b))
    return round(num / (da * db), 4) if da * db > 1e-9 else 0.0


def compute_wave_analysis(ticker: str, monthly_df: pd.DataFrame, granularity: str = 'Weekly') -> dict:
    """
    Core analysis function. Takes OHLCV at any granularity and returns a full dict of
    metrics, scores, and the trading signal.

    Score components (max 100):
      s_vol   (20)  Annualised volatility  — bigger moves = more to trade
      s_swing (20)  Avg monthly High-Low   — intra-wave amplitude
      s_rev   (20)  Reversal count         — structured waves, not noise
      s_surge (15)  Volume surge ratio     — institutional/retail confirmation
      s_range (15)  Range position (inv.)  — entry quality (near low = better)
      s_mom   (10)  3-month momentum       — confirms upswing phase
    """
    closes  = monthly_df["Close"].tolist()
    highs   = monthly_df["High"].tolist()
    lows    = monthly_df["Low"].tolist()
    vols    = monthly_df["Volume"].tolist()
    # Format dates at the right resolution for the granularity
    _dfmt = {"Daily": "%Y-%m-%d", "Weekly": "%Y-%m-%d", "Monthly": "%Y-%m"}.get(granularity, "%Y-%m-%d")
    dates   = [d.strftime(_dfmt) for d in monthly_df.index]
    n = len(closes)

    rets           = _monthly_returns(closes)
    # ATR ZigZag: N_ATR=1.0, 5% floor, ATR period scales with granularity
    _atr_per = {"Daily": 10, "Weekly": 5, "Monthly": 3}.get(granularity, 5)
    peaks, troughs = _peaks_troughs(
        closes, highs=highs, lows=lows,
        n_atr=1.0, min_pct=5.0, atr_period=_atr_per
    )
    hurst       = _hurst(closes)

    # ── Annualised volatility (std of period returns × √periods_per_year) ────
    _scale = {"Daily": 252, "Weekly": 52, "Monthly": 12}.get(granularity, 52)
    if len(rets) > 1:
        mean_r  = sum(rets) / len(rets)
        var_r   = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        vol_ann = math.sqrt(var_r) * math.sqrt(_scale) * 100
    else:
        vol_ann = 0.0

    # ── Average monthly swing amplitude ────────────────────────────────────
    avg_swing = sum(
        (highs[i] - lows[i]) / ((highs[i] + lows[i]) / 2) * 100
        for i in range(n)
    ) / n

    # ── Volume surge ratio ──────────────────────────────────────────────────
    avg_vol   = sum(vols) / len(vols) if vols else 1
    vol_surge = max(vols) / avg_vol  if avg_vol > 0 else 1.0

    # ── 52-week range position ──────────────────────────────────────────────
    year_high  = max(highs)
    year_low   = min(lows)
    current    = closes[-1]
    range_pct  = (current - year_low) / (year_high - year_low) * 100 \
                 if year_high != year_low else 50.0
    drawdown   = (current - year_high) / year_high * 100

    # ── 3-month momentum — bars-back depends on granularity ────────────────
    _mom_bars = {"Daily": 63, "Weekly": 13, "Monthly": 3}.get(granularity, 13)
    mom_3m = (closes[-1] - closes[-1 - _mom_bars]) / closes[-1 - _mom_bars] * 100              if n > _mom_bars else 0.0

    # ── 1-year return ───────────────────────────────────────────────────────
    yr_return = (closes[-1] - closes[0]) / closes[0] * 100

    # ──────────────────────────────────────────────────────────────────────
    # WAVE SCORE  (0 – 100)
    # ──────────────────────────────────────────────────────────────────────
    n_rev   = len(peaks) + len(troughs)

    s_vol   = min(vol_ann  / 70, 1.0) * 20           # cap at 70% annual vol
    s_swing = min(avg_swing / 7,  1.0) * 20           # cap at 7% monthly swing
    # Ideal reversal count: roughly 1 full wave per 8 bars
    # ATR ZigZag on real data typically yields 6-14 pivots per 100 weekly bars
    # Score peaks at 8-12 pivots, tapers off smoothly on either side
    _ideal_rev = max(4, n // 8)  # n = number of bars
    _rev_tol   = max(3, _ideal_rev // 2)  # tolerance band
    s_rev   = max(0, (_rev_tol - abs(n_rev - _ideal_rev)) / _rev_tol) * 20
    s_surge = min(vol_surge / 4,  1.0) * 15           # cap at 4× surge
    s_range = (1 - range_pct / 100) * 15              # inverted: near low = high score
    s_mom   = min(max(mom_3m, 0) / 30, 1.0) * 10     # cap at +30% 3M momentum

    wave_score = max(0.0, s_vol + s_swing + s_rev + s_surge + s_range + s_mom)

    # ──────────────────────────────────────────────────────────────────────
    # SIGNAL  (rule-based, runs in order)
    # ──────────────────────────────────────────────────────────────────────
    if range_pct > 85 and drawdown > -8:
        signal = "EXIT"
    elif range_pct < 40 and mom_3m > -5:
        signal = "BUY"
    elif mom_3m > 10 and range_pct < 75:
        signal = "BUY"
    elif mom_3m < -15:
        signal = "WATCH"
    else:
        signal = "HOLD"

    return {
        "ticker":         ticker,
        "granularity":    granularity,
        "dates":          dates,
        "closes":         closes,
        "highs":          highs,
        "lows":           lows,
        "volumes":        vols,
        "monthly_returns": [round(r * 100, 2) for r in rets],
        # metrics
        "hurst":          round(hurst,    3),
        "vol_annual_pct": round(vol_ann,  1),
        "avg_swing_pct":  round(avg_swing, 2),
        "vol_surge":      round(vol_surge, 2),
        "n_reversals":    n_rev,
        "peak_months":    peaks,
        "trough_months":  troughs,
        "n_peaks":        len(peaks),
        "n_troughs":      len(troughs),
        "atr_period":     _atr_per,
        "year_high":      year_high,
        "year_low":       year_low,
        "current_price":  current,
        "range_pct":      round(range_pct,  1),
        "drawdown_pct":   round(drawdown,   1),
        "mom_3m_pct":     round(mom_3m,     1),
        "yr_return_pct":  round(yr_return,  1),
        # score breakdown
        "s_vol":          round(s_vol,   1),
        "s_swing":        round(s_swing, 1),
        "s_rev":          round(s_rev,   1),
        "s_surge":        round(s_surge, 1),
        "s_range":        round(s_range, 1),
        "s_mom":          round(s_mom,   1),
        "wave_score":     round(wave_score, 1),
        # signal
        "signal":         signal,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCH  (cached so repeated runs don't re-hit Tushare)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_analyse(tickers: tuple[str, ...], granularity: str, start_date: str) -> dict:
    """
    Fetch OHLCV at the chosen granularity for each ticker and run wave analysis.
    granularity : "Daily" | "Weekly" | "Monthly"
    start_date  : "YYYYMMDD" string
    Returns {'results': [...], 'errors': [...], 'raw': {ticker: DataFrame}}
    Cached for 1 hour so re-running the page is instant.
    """
    results, errors, raw_frames = [], [], {}
    for t in tickers:
        try:
            df = dm.get_ohlcv_for_wave(t, granularity=granularity, start_date=start_date)
            min_bars = {"Daily": 20, "Weekly": 12, "Monthly": 6}[granularity]
            if df is None or len(df) < min_bars:
                errors.append((t, f"Not enough {granularity.lower()} bars (need ≥ {min_bars})"))
                continue
            raw_frames[t] = df
            res = compute_wave_analysis(t, df, granularity=granularity)
            try:
                res["name"] = dm.get_stock_name_wave(t)
            except Exception:
                res["name"] = t
            results.append(res)
        except Exception as e:
            errors.append((t, str(e)))
    return {"results": results, "errors": errors, "raw": raw_frames}


# ──────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def chart_price_with_peaks(r: dict, color: str) -> go.Figure:
    """
    Price line + ATR ZigZag overlay.
    - Grey line  : raw close prices
    - Dotted line: ZigZag connecting confirmed pivots
    - Red ▲      : peaks (exit zones)
    - Green ▽    : troughs (entry zones)
    """
    dates   = r["dates"]
    closes  = r["closes"]
    peaks   = r["peak_months"]
    troughs = r["trough_months"]
    gran    = r.get("granularity", "Weekly")
    n_atr_p = r.get("atr_period", 5)

    # Build sorted pivot list for ZigZag line
    all_pivots = sorted(
        [(i, closes[i], 'P') for i in peaks  if i < len(closes)] +
        [(i, closes[i], 'T') for i in troughs if i < len(closes)],
        key=lambda x: x[0]
    )
    zz_dates  = [dates[i]  for i, _, _ in all_pivots if i < len(dates)]
    zz_prices = [closes[i] for i, _, _ in all_pivots]
    zz_types  = [t          for _, _, t  in all_pivots]

    peak_dates    = [dates[i]  for i in peaks   if i < len(dates)]
    peak_prices   = [closes[i] for i in peaks   if i < len(closes)]
    trough_dates  = [dates[i]  for i in troughs if i < len(dates)]
    trough_prices = [closes[i] for i in troughs if i < len(closes)]

    fig = go.Figure()

    # Raw price line
    fig.add_trace(go.Scatter(
        x=dates, y=closes,
        mode="lines",
        name="Close",
        line=dict(color=color, width=2),
        hovertemplate="%{x}<br>¥%{y:.2f}<extra>Close</extra>",
    ))

    # ZigZag connecting pivots
    if zz_dates:
        fig.add_trace(go.Scatter(
            x=zz_dates, y=zz_prices,
            mode="lines",
            name="ZigZag (ATR filtered)",
            line=dict(color="rgba(255,255,255,0.35)", width=1.2, dash="dot"),
            hoverinfo="skip",
        ))

    # Peak markers
    if peak_dates:
        fig.add_trace(go.Scatter(
            x=peak_dates, y=peak_prices,
            mode="markers+text",
            name=f"▲ Peak — exit zone ({len(peaks)})",
            marker=dict(symbol="triangle-up", color="#ef4444", size=13,
                        line=dict(color="white", width=1)),
            text=["▲"] * len(peak_dates),
            textposition="top center",
            textfont=dict(color="#ef4444", size=9),
            hovertemplate="▲ PEAK<br>%{x}<br>¥%{y:.2f}<extra></extra>",
        ))

    # Trough markers
    if trough_dates:
        fig.add_trace(go.Scatter(
            x=trough_dates, y=trough_prices,
            mode="markers+text",
            name=f"▽ Trough — entry zone ({len(troughs)})",
            marker=dict(symbol="triangle-down", color="#10b981", size=13,
                        line=dict(color="white", width=1)),
            text=["▽"] * len(trough_dates),
            textposition="bottom center",
            textfont=dict(color="#10b981", size=9),
            hovertemplate="▽ TROUGH<br>%{x}<br>¥%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=(f"{r['ticker']} {r['name']} — ATR ZigZag Peaks & Troughs ({gran})  "
               f"· N_ATR=1.0 · ATR period={n_atr_p} bars · floor=5%  "
               f"· {len(peaks)} peaks · {len(troughs)} troughs"),
        yaxis_title="Price ¥ (qfq adjusted)",
        height=420, template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
        hovermode="x unified",
    )
    return fig

def chart_normalized_performance(results: list[dict]) -> go.Figure:
    """All stocks normalised to 100 at the start of the window — shows relative strength."""
    fig = go.Figure()
    for i, r in enumerate(results):
        base  = r["closes"][0]
        normd = [c / base * 100 for c in r["closes"]]
        label = f"{r['ticker']} {r['name']}"
        fig.add_trace(go.Scatter(
            x=r["dates"], y=normd,
            mode="lines+markers",
            name=label,
            line=dict(color=WAVE_COLORS[i % len(WAVE_COLORS)], width=2),
            marker=dict(size=5),
            hovertemplate="%{x}<br>%{y:.1f}<extra>" + label + "</extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(
        title="Normalised Performance (start of window = 100)",
        xaxis_title=None, yaxis_title="Index (100 = start)",
        legend=dict(orientation="h", y=-0.15),
        height=380, template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def chart_monthly_returns(r: dict, color: str) -> go.Figure:
    """Bar chart of monthly returns for a single stock."""
    rets  = r["monthly_returns"]
    dates = r["dates"][1:]  # one fewer than closes
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in rets]
    fig = go.Figure(go.Bar(
        x=dates, y=rets,
        marker_color=colors,
        hovertemplate="%{x}<br>%{y:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=f"{r['ticker']} {r['name']} — Monthly Returns",
        yaxis_title="%", height=220,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=36, b=0),
        showlegend=False,
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
    return fig


def chart_phase_heatmap(results: list[dict]) -> go.Figure:
    """
    Phase heatmap at the actual data granularity (Daily / Weekly / Monthly).

    Each column = one bar at the chosen frequency.
    Each cell   = that bar's period-over-period return, coloured green/red.
    ▲ = ATR ZigZag peak at that bar  ▽ = trough.

    For Daily (100+ cols): text labels are hidden to avoid crowding;
    hover shows the value. For Weekly/Monthly: labels shown.
    """
    gran = results[0].get("granularity", "Weekly")

    # ── date labels ──────────────────────────────────────────────────────────
    # dates[0] is the base bar; returns are dates[1:]
    all_dates = results[0]["dates"][1:]   # already formatted strings from compute_wave_analysis
    n_cols    = len(all_dates)

    # ── build matrix ─────────────────────────────────────────────────────────
    z_matrix, text_matrix, hover_matrix = [], [], []
    y_labels = []

    for r in results:
        rets      = r["monthly_returns"]   # period returns regardless of gran name
        peaks_set   = set(r["peak_months"])
        troughs_set = set(r["trough_months"])

        row_z, row_t, row_h = [], [], []
        for mi in range(n_cols):
            bar_idx = mi + 1   # offset: dates[0] has no return
            if mi < len(rets):
                val = rets[mi]
                row_z.append(val)
                # peak/trough marker
                if bar_idx in peaks_set:
                    marker = "▲"
                elif bar_idx in troughs_set:
                    marker = "▽"
                else:
                    marker = ""
                row_t.append(f"{val:+.1f}%{marker}" if marker else f"{val:+.1f}%")
                row_h.append(f"{all_dates[mi]}<br>{val:+.1f}%{(' '+marker) if marker else ''}")
            else:
                row_z.append(None)
                row_t.append("")
                row_h.append("")

        z_matrix.append(row_z)
        text_matrix.append(row_t)
        hover_matrix.append(row_h)
        y_labels.append(f"{r['ticker']} {r['name']}")

    # ── label visibility: hide text when too many columns ────────────────────
    # Daily: typically 200-500 cols → no text (hover only)
    # Weekly: ~50-100 cols → small text
    # Monthly: ~12-36 cols → full text
    # Always hover-only — text is unreadable when squeezed on daily/weekly
    texttemplate = ""
    textfont_sz  = 7

    # ── x-axis tick thinning for dense granularities ─────────────────────────
    # Show every Nth label so the axis isn't unreadable
    if n_cols > 200:
        tick_every = 20
    elif n_cols > 80:
        tick_every = 8
    elif n_cols > 40:
        tick_every = 4
    else:
        tick_every = 1

    tick_vals = [all_dates[i] for i in range(0, n_cols, tick_every)]

    # ── chart height: taller rows when fewer stocks ───────────────────────────
    row_h_px = max(40, min(80, 200 // len(results)))
    chart_h  = max(200, row_h_px * len(results) + 120)

    # ── column width: narrower when more bars ────────────────────────────────
    col_w = max(4, min(30, 1400 // max(n_cols, 1)))

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=all_dates,
        y=y_labels,
        text=text_matrix,
        customdata=hover_matrix,
        texttemplate=texttemplate,
        textfont=dict(size=textfont_sz),
        colorscale=[
            [0.0,  "#10b981"],   # negative = green (A-share: green = down)
            [0.45, "#1e293b"],
            [0.55, "#1e293b"],
            [1.0,  "#ef4444"],   # positive = red   (A-share: red = up)
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(title=f"{gran[:1]}收益%", thickness=12),
        xgap=1, ygap=2,
        hovertemplate="%{y}<br>%{customdata}<extra></extra>",
    ))

    gran_label = {"Daily": "日线", "Weekly": "周线", "Monthly": "月线"}.get(gran, gran)
    fig.update_layout(
        title=f"Phase Heatmap ({gran_label}) — ▲ Peak  ▽ Trough  · {n_cols} bars · each cell = one {gran.lower()} return",
        height=chart_h,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=60),
        xaxis=dict(
            side="top",
            tickvals=tick_vals,
            tickangle=-45,
            tickfont=dict(size=8),
        ),
    )
    return fig


def chart_score_breakdown(results: list[dict]) -> go.Figure:
    """Stacked horizontal bar showing how each stock's wave score is built."""
    components = ["s_vol", "s_swing", "s_rev", "s_surge", "s_range", "s_mom"]
    labels     = ["Volatility (20)", "Avg Swing (20)", "Reversals (20)",
                  "Vol Surge (15)", "Range Pos (15)", "3M Momentum (10)"]
    comp_colors = ["#2dd4bf", "#a78bfa", "#f59e0b", "#60a5fa", "#34d399", "#f87171"]

    fig = go.Figure()
    for ci, (comp, lbl, col) in enumerate(zip(components, labels, comp_colors)):
        vals = [r[comp] for r in results]
        names = [f"{r['ticker']}" for r in results]
        fig.add_trace(go.Bar(
            name=lbl,
            x=names,
            y=vals,
            marker_color=col,
            hovertemplate=f"{lbl}<br>%{{y:.1f}} pts<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title="Wave Score Breakdown by Component",
        yaxis_title="Score (0–100)", height=320,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    # Add max line
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Max 100", annotation_position="top right")
    return fig


def chart_correlation_matrix(results: list[dict]) -> go.Figure:
    """Concurrent correlation heatmap between all pairs."""
    n = len(results)
    tickers = [r["ticker"] for r in results]
    matrix  = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                r1 = results[i]["monthly_returns"]
                r2 = results[j]["monthly_returns"]
                matrix[i][j] = _concurrent_corr(r1, r2)

    fig = go.Figure(go.Heatmap(
        z=matrix, x=tickers, y=tickers,
        text=[[f"{matrix[i][j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        colorscale="RdYlGn_r",
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(title="Corr", thickness=12),
        hovertemplate="%{y} vs %{x}<br>Corr: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Concurrent Correlation Matrix (lower = better phase independence)",
        height=max(260, 60 * n + 80),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def chart_52wk_range(results: list[dict]) -> go.Figure:
    """Bullet chart showing each stock's position in its 52-week range."""
    fig = go.Figure()
    for i, r in enumerate(results):
        label = r["ticker"]
        fig.add_trace(go.Scatter(
            x=[r["year_low"], r["year_high"]],
            y=[label, label],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=10),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[r["current_price"]],
            y=[label],
            mode="markers",
            marker=dict(
                color=WAVE_COLORS[i % len(WAVE_COLORS)],
                size=14, symbol="diamond",
                line=dict(color="white", width=1.5),
            ),
            name=label,
            hovertemplate=f"{label}<br>Current: ¥{{x:.2f}}<br>Range: ¥{r['year_low']:.2f}–¥{r['year_high']:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title="52-Week Range Position  (diamond = current price)",
        height=max(160, 50 * len(results) + 80),
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# EXPLANATION EXPANDERS  (educational content baked into the UI)
# ──────────────────────────────────────────────────────────────────────────────

def show_methodology_expander():
    with st.expander("📖 How Wave Trading Works — Methodology  波段交易原理", expanded=False):
        st.markdown("""
**Wave trading** exploits the natural rhythm of A-share price cycles — typically **40–70 trading days** per cycle — driven by retail participation patterns, margin flows, and policy-driven sector rotations.

The core idea: buy near the trough, sell near the peak. Repeat. The engine does three things:

1. **Scores** each stock on how wave-tradeable its price structure is (0–100)
2. **Signals** whether *right now* is a good entry, hold, watch, or exit point
3. **Finds** the best rotation partner — a stock whose waves are *phase-offset* from yours, so you're always in something rising

**Why A-shares specifically?**
A-shares have ~80% retail participation. Retail investors chase momentum in herds, creating predictable boom-bust cycles that are more exploitable than institutional-dominated markets. The Hurst exponent (explained below) quantifies this.
        """)


def show_hurst_expander():
    with st.expander("📐 Hurst Exponent — Is this stock mean-reverting?  赫斯特指数", expanded=False):
        st.markdown("""
**What it is:** A measure of how "memory" a price series has.

| Value | Meaning | Implication for trading |
|---|---|---|
| **H < 0.5** | Mean-reverting | Prices overshoot then snap back — ideal for wave trading |
| **H = 0.5** | Random walk | No exploitable structure |
| **H > 0.5** | Trending | Prices persist in one direction — trend follow instead |

**How we calculate it (R/S method):**
1. Compute log returns: `ln(close[t] / close[t-1])` for each month
2. Compute cumulative deviation from mean return
3. R = range of that cumulative deviation (max − min)
4. S = standard deviation of the log returns
5. H = log(R/S) / log(n)

**For wave trading, you want H < 0.45.** This tells you the stock systematically overshoots and corrects — the wave structure is exploitable, not random.

> 💡 None of your candidates need to be perfectly mean-reverting. Even H = 0.50–0.55 is fine if the other score components are strong.
        """)


def show_score_expander():
    with st.expander("🎯 Wave Score (0–100) — Full Component Breakdown  评分细则", expanded=False):
        st.markdown("""
The Wave Score is the sum of 6 components. Each one answers a specific question about the stock's wave-tradability.

---

### Component 1 — Annualised Volatility  (max **20 pts**)
```
s_vol = min(vol_annual / 70%, 1) × 20
```
**Question:** How big are the moves?

Computed from the standard deviation of monthly returns, scaled up by √12 to express on an annual basis.
Anything ≥ 70% annual volatility gets the full 20 pts. Below that it scales linearly.

*Why 70%?* A-share mid-caps with strong wave behaviour typically run 50–90% annual vol. 70% is the sweet spot.

---

### Component 2 — Average Monthly Swing Amplitude  (max **20 pts**)
```
s_swing = min(avg_swing / 7%, 1) × 20
where: avg_swing[month] = (High − Low) / midpoint × 100
```
**Question:** Within each month, how wide is the High–Low band?

This is different from volatility — a stock can have high month-to-month vol (big jumps between months) but narrow intra-month swings. For wave trading you want BOTH: big inter-month moves *and* wide intra-month ranges to trade the full wave.

*Why 7%?* Liquid A-share mid-caps typically swing 4–8% intra-month. 7% is the upper end of normal.

---

### Component 3 — Wave Reversal Count  (max **20 pts**)
```
ideal = 5 reversals over 13 months
s_rev = max(0, (10 − |n_reversals − 5|) / 10) × 20
```
**Question:** Does the stock have structured wave cycles — not too few, not too many?

Reversals are detected using an **ATR-filtered ZigZag** — not a simple bar-by-bar comparison.

A new pivot only registers when price has moved at least:
```
threshold = max(N_ATR × ATR%, floor)
          = max(1.0 × ATR-as-%-of-price, 5%)
```
from the **last confirmed pivot**. This means:
- Minor weekly wiggles of 1–3% are ignored — they don't cross the threshold
- The threshold adapts to volatility: when the stock is volatile (high ATR%), it takes a bigger move to qualify
- Peaks and troughs strictly alternate (can't get two peaks in a row)

**Ideal count is data-driven:**
```
ideal = max(4, n_bars / 8)      # roughly one reversal per 8 bars
tolerance = max(3, ideal / 2)   # score tapers off within this band
s_rev = max(0, (tolerance - |n_reversals - ideal|) / tolerance) × 20
```
For 103 weekly bars: ideal ≈ 13, tolerance ≈ 6 — so counts of 7–19 all score well.

**Why penalise both extremes:**
- Too few (< ideal−tol): Stock trended without reversing — not a wave structure, wrong strategy
- Too many (> ideal+tol): Threshold was too tight, capturing noise — reconsider the ATR floor
- At ideal: Clean tradeable wave cycles with meaningful swing sizes

---

### Component 4 — Volume Surge Ratio  (max **15 pts**)
```
vol_surge = max_monthly_volume / avg_monthly_volume
s_surge = min(vol_surge / 4×, 1) × 15
```
**Question:** Do big moves come with volume confirmation?

In A-shares, institutional rotation triggers retail chasing. The volume surge ratio tells you whether the wave's upswing is backed by real participation or is a low-volume head-fake.

*Why 4×?* A clean A-share wave typically shows 3–5× average volume at the surge point.

---

### Component 5 — 52-Week Range Position  (max **15 pts**, inverted)
```
range_pct = (current − year_low) / (year_high − year_low) × 100
s_range = (1 − range_pct / 100) × 15
```
**Question:** Is there room for the stock to move up, or is it already near the top?

This component is **inverted** — being at the bottom of the range scores more than being at the top.

| Range Position | Score | Interpretation |
|---|---|---|
| 10% (near low) | 13.5 pts | Great entry potential |
| 50% (mid-range) | 7.5 pts | Neutral |
| 90% (near high) | 1.5 pts | Poor entry — wave mostly done |
| 100% (at high) | 0 pts | Exit zone |

---

### Component 6 — 3-Month Momentum  (max **10 pts**)
```
mom_3m = (close_now − close_3mo_ago) / close_3mo_ago × 100
s_mom = min(max(mom_3m, 0) / 30%, 1) × 10
```
**Question:** Is the stock currently in its rising phase?

Negative momentum → 0 pts (not penalised, just no bonus).
+30% over 3 months → full 10 pts.

This is the *smallest* component because momentum is a confirmation signal, not a structural quality measure. A stock can have excellent wave structure but be temporarily in a downswing — you'd still want to know about it, just enter later.

---

### Total
| Component | Max pts |
|---|---|
| Annualised Volatility | 20 |
| Avg Monthly Swing | 20 |
| Reversal Count | 20 |
| Volume Surge | 15 |
| Range Position (inverted) | 15 |
| 3M Momentum | 10 |
| **Total** | **100** |
        """)


def show_signal_expander():
    with st.expander("🚦 Signal Logic — BUY / HOLD / WATCH / EXIT  信号逻辑", expanded=False):
        st.markdown("""
The signal is **separate from the wave score** — it tells you where the stock is *right now* in its cycle, not whether it's a good wave-trading candidate overall.

Rules run in priority order. The first match wins:

```
Rule 1 → EXIT   : range_position > 85%  AND  drawdown_from_peak > −8%
Rule 2 → BUY    : range_position < 40%  AND  3M_momentum > −5%
Rule 3 → BUY    : 3M_momentum > +10%   AND  range_position < 75%
Rule 4 → WATCH  : 3M_momentum < −15%
Rule 5 → HOLD   : (everything else)
```

**EXIT** — You're near the 52-week high and haven't pulled back meaningfully yet. The wave has likely peaked. Sell to capture most of the upswing before mean-reversion.

**BUY (Rule 2)** — Near the low, not in freefall. This is the trough entry zone. Classic wave buy.

**BUY (Rule 3)** — Momentum breakout mid-range. Stock has turned off the bottom and is in the rising phase but still has room to run.

**WATCH** — Falling fast (−15%+ over 3 months). Could be a genuine breakdown, not just a trough. Wait for stabilisation before entering.

**HOLD** — No strong signal in either direction. Stay if you're already in; don't chase if you're not.

> ⚠️ A stock can score 80/100 on wave quality but show EXIT — that means it's a great wave-trading stock whose *current entry timing is poor*. Wait for the next trough.
        """)


def show_rotation_expander():
    with st.expander("🔄 Rotation Logic — Phase Offset & Seamless Capital Flow  轮动逻辑", expanded=False):
        st.markdown("""
**The goal:** Never have capital sitting idle. When Stock A is in its downswing/consolidation, Stock B or C should be in its upswing — so you rotate capital and stay productive.

**How phase offset is measured:**

For each pair of stocks, we compute the **concurrent correlation** (Pearson R at lag = 0):
```
corr(A, B) = how similar are A's monthly returns to B's monthly returns
             measured in the same months
```

| Correlation | Meaning |
|---|---|
| > 0.7 | Move together — poor rotation pair, they peak/trough at the same time |
| 0.4–0.7 | Partial overlap — acceptable |
| < 0.4 | Move independently — ideal rotation pair |
| Negative | Move opposite — perfect hedge but not typical in same-sector plays |

**Peak offset** counts how many months apart the detected peaks are between two stocks. A 2–3 month peak offset with a 50-day cycle = one stock is just entering its upswing as the other is topping out.

**The rotation table** ranks all input stocks by wave score. The recommendation is to flow capital from whichever stock just triggered EXIT → into the next highest-scored stock that shows BUY or HOLD.
        """)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────────────────────────────────────

st.title("🌊 Wave Trader  波段交易系统")
st.caption("Enter A-share codes → fetch live monthly data → score wave-tradability → find the best rotation trio")

# ── methodology upfront ───────────────────────────────────────────────────────
show_methodology_expander()

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# INPUT PANEL
# ──────────────────────────────────────────────────────────────────────────────
col_input, col_preset = st.columns([3, 2])

with col_input:
    st.subheader("📥 Ticker Input  股票代码输入")
    ticker_raw = st.text_area(
        "Enter A-share codes (one per line, or comma-separated). 6-digit codes only.",
        value="002080\n600522\n688008",
        height=120,
        help="e.g. 002080, 600522, 688008.  Exchange suffix (.SZ/.SS) is added automatically.",
        placeholder="002080\n600522\n688008",
    )
    g_col, d_col = st.columns([1, 2])
    with g_col:
        granularity = st.radio(
            "Bar size  频率",
            options=["Daily", "Weekly", "Monthly"],
            index=1,
            help=(
                "Daily: 60–500 bars, intraday noise but most detail.\n"
                "Weekly: 50–130 bars — best for 4–12 week A-share cycles.\n"
                "Monthly: 12–36 bars — trend view, hides swing detail."
            ),
            horizontal=False,
        )
    with d_col:
        import datetime as _dt
        default_start = _dt.date.today() - _dt.timedelta(days=365 * 2)
        start_date_val = st.date_input(
            "Data start date  数据起始日",
            value=default_start,
            min_value=_dt.date(2015, 1, 1),
            max_value=_dt.date.today() - _dt.timedelta(days=30),
            help="Choose how far back to fetch. More history = more wave cycles but older data.",
        )
        start_date_str = start_date_val.strftime("%Y%m%d")

with col_preset:
    st.subheader("⚡ Presets  预设组合")
    preset_choice = st.radio(
        "Load a preset:", options=["(none)"] + list(PRESETS.keys()), label_visibility="collapsed"
    )
    if preset_choice != "(none)":
        preset_tickers = PRESETS[preset_choice]
        st.info(f"Will load: {', '.join(preset_tickers)}")

st.divider()

# ── Parse tickers ─────────────────────────────────────────────────────────────
if preset_choice != "(none)":
    raw_list = PRESETS[preset_choice]
else:
    raw_list = [t.strip().replace(" ", "") for t in ticker_raw.replace(",", "\n").split("\n") if t.strip()]

# Normalise: add exchange suffix if missing
def normalise(t: str) -> str:
    t = t.upper().strip()
    if "." in t:
        return t.split(".")[0]   # strip suffix — data_manager handles it
    return t

tickers_clean = list(dict.fromkeys(normalise(t) for t in raw_list if t))[:10]

if not tickers_clean:
    st.warning("Add at least one ticker code above.")
    st.stop()

st.markdown(f"**Selected ({len(tickers_clean)}):** " + "  ·  ".join(f"`{t}`" for t in tickers_clean))

run_btn = st.button("🚀 Run Wave Analysis  开始分析", type="primary", use_container_width=True)

# ── session_state: persist payload so UI controls (sliders, selectors) don't ──
# ── reset the analysis when Streamlit reruns on widget interaction            ──
if run_btn:
    with st.spinner(f"Fetching {granularity.lower()} data from Tushare and running wave engine…"):
        st.session_state["wt_payload"]     = fetch_and_analyse(
            tuple(tickers_clean), granularity=granularity, start_date=start_date_str
        )
        st.session_state["wt_granularity"] = granularity

if "wt_payload" not in st.session_state:
    st.info("Press **Run Wave Analysis** to fetch live data and compute scores.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# FETCH + ANALYSE
# ──────────────────────────────────────────────────────────────────────────────
payload    = st.session_state["wt_payload"]
results    : list[dict]        = payload["results"]
errors     : list[tuple]       = payload["errors"]
raw_frames : dict[str, object] = payload.get("raw", {})

# Show errors
if errors:
    with st.expander(f"⚠️ {len(errors)} ticker(s) failed", expanded=True):
        for t, msg in errors:
            st.error(f"`{t}` — {msg}")

if not results:
    st.error("No valid results. Check your ticker codes and Tushare connection.")
    st.stop()

# Sort by wave score descending
results_sorted = sorted(results, key=lambda r: r["wave_score"], reverse=True)

st.success(f"✅ Analysed {len(results)} stock(s) · Data as of {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL SUMMARY BAR  (top of page — quick read)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🚦 Current Signals  当前信号")
sig_cols = st.columns(len(results_sorted))
for ci, (col, r) in enumerate(zip(sig_cols, results_sorted)):
    sig    = r["signal"]
    color, icon = SIGNAL_COLORS[sig]
    with col:
        st.markdown(
            f"""
            <div style="border:1px solid {color}40; border-radius:10px; padding:14px 10px; text-align:center;
                        background:{color}12;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-family:monospace; font-size:1.1rem; font-weight:700; color:{color};">{sig}</div>
                <div style="font-size:1.05rem; font-weight:700;">{r['ticker']}</div>
                <div style="font-size:0.78rem; color:#94a3b8;">{r['name']}</div>
                <div style="font-family:monospace; font-size:0.85rem; margin-top:6px;">
                    ¥{r['current_price']:.2f}
                </div>
                <div style="font-size:1.4rem; font-weight:800; color:{color}; margin-top:4px;">
                    {r['wave_score']:.0f}<span style="font-size:0.75rem; font-weight:400; color:#94a3b8;">/100</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

show_signal_expander()
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_overview, tab_scores, tab_charts, tab_phase, tab_corr, tab_rotation = st.tabs([
    "📊 Overview 总览",
    "🎯 Score Breakdown 评分",
    "📈 Charts 图表",
    "🗓️ Phase Heatmap 相位热图",
    "🔗 Correlation 相关性",
    "🔄 Rotation 轮动",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Stock-by-Stock Metrics  个股指标")

    # Metrics table
    rows = []
    for r in results_sorted:
        sig, (col, icon) = r["signal"], SIGNAL_COLORS[r["signal"]]
        rows.append({
            "Ticker": r["ticker"],
            "Name": r["name"],
            "Signal": f"{icon} {sig}",
            "Wave Score": r["wave_score"],
            "Hurst": r["hurst"],
            "Ann. Vol %": r["vol_annual_pct"],
            "Avg Swing %": r["avg_swing_pct"],
            "Reversals": r["n_reversals"],
            "Vol Surge ×": r["vol_surge"],
            "Range Pos %": r["range_pct"],
            "3M Mom %": r["mom_3m_pct"],
            "1Y Return %": r["yr_return_pct"],
            "Current ¥": r["current_price"],
            "52wk Low ¥": r["year_low"],
            "52wk High ¥": r["year_high"],
        })

    df_table = pd.DataFrame(rows)
    st.dataframe(
        df_table.style
            .background_gradient(subset=["Wave Score"], cmap="RdYlGn", vmin=0, vmax=100)
            .background_gradient(subset=["3M Mom %"], cmap="RdYlGn", vmin=-30, vmax=30)
            .background_gradient(subset=["Range Pos %"], cmap="RdYlGn_r", vmin=0, vmax=100)
            .format({
                "Wave Score": "{:.0f}", "Hurst": "{:.3f}",
                "Ann. Vol %": "{:.1f}", "Avg Swing %": "{:.1f}",
                "Vol Surge ×": "{:.2f}", "Range Pos %": "{:.0f}",
                "3M Mom %": "{:+.1f}", "1Y Return %": "{:+.1f}",
                "Current ¥": "{:.2f}", "52wk Low ¥": "{:.2f}", "52wk High ¥": "{:.2f}",
            }),
        use_container_width=True, hide_index=True,
    )

    st.plotly_chart(chart_52wk_range(results_sorted), use_container_width=True, key="chart_52wk")

    show_hurst_expander()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCORE BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════════
with tab_scores:
    st.subheader("Wave Score Components  评分明细")
    show_score_expander()

    st.plotly_chart(chart_score_breakdown(results_sorted), use_container_width=True, key="chart_score_breakdown")

    # Detailed per-stock breakdown
    for i, r in enumerate(results_sorted):
        color = WAVE_COLORS[i % len(WAVE_COLORS)]
        with st.expander(
            f"**{r['ticker']} {r['name']}** — Score: {r['wave_score']:.0f}/100  · Signal: {r['signal']}",
            expanded=(i == 0),
        ):
            c1, c2, c3 = st.columns(3)
            metrics = [
                ("Volatility (max 20)", r["s_vol"],   f"Ann. vol = {r['vol_annual_pct']:.1f}%  (target ≥ 70%)"),
                ("Avg Swing (max 20)", r["s_swing"],  f"Monthly H-L swing = {r['avg_swing_pct']:.1f}%  (target ≥ 7%)"),
                ("Reversals (max 20)", r["s_rev"],    f"{r['n_reversals']} pivots  ({r.get('n_peaks',0)} peaks · {r.get('n_troughs',0)} troughs · ideal ≈ {max(4, len(r['closes'])//8)})"),
                ("Vol Surge (max 15)", r["s_surge"],  f"Peak vol / avg vol = {r['vol_surge']:.1f}×  (target ≥ 4×)"),
                ("Range Pos (max 15)", r["s_range"],  f"At {r['range_pct']:.0f}% of 52-week range  (lower = better)"),
                ("3M Mom (max 10)",    r["s_mom"],    f"3-month return = {r['mom_3m_pct']:+.1f}%  (target ≥ +30%)"),
            ]
            for idx, (label, score, note) in enumerate(metrics):
                col = [c1, c2, c3][idx % 3]
                with col:
                    st.metric(label=label, value=f"{score:.1f} pts")
                    st.caption(note)

            # Mini score bar
            components = [r["s_vol"], r["s_swing"], r["s_rev"], r["s_surge"], r["s_range"], r["s_mom"]]
            comp_labels = ["Vol", "Swing", "Rev", "Surge", "Range", "Mom"]
            fig_mini = go.Figure(go.Bar(
                x=comp_labels, y=components,
                marker_color=["#2dd4bf","#a78bfa","#f59e0b","#60a5fa","#34d399","#f87171"],
                text=[f"{v:.1f}" for v in components],
                textposition="outside",
            ))
            fig_mini.update_layout(
                height=200, template="plotly_dark",
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False, yaxis=dict(range=[0, 22]),
            )
            st.plotly_chart(fig_mini, use_container_width=True, key=f"chart_mini_{i}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.plotly_chart(chart_normalized_performance(results_sorted), use_container_width=True, key="chart_norm_perf")

    st.markdown("---")
    st.subheader("Price with Wave Peaks & Troughs  价格与波峰波谷")
    st.caption(
        "▲ red = detected peak (potential EXIT zone)  "
        "▽ green = detected trough (potential BUY zone)  "
        "Detection: 3-point local max/min on closing prices."
    )
    for i, r in enumerate(results_sorted):
        color = WAVE_COLORS[i % len(WAVE_COLORS)]
        st.plotly_chart(chart_price_with_peaks(r, color), use_container_width=True, key=f"chart_peaks_{i}")

    st.markdown("---")
    st.subheader("Period Returns by Stock  各股周期收益")
    for i, r in enumerate(results_sorted):
        color = WAVE_COLORS[i % len(WAVE_COLORS)]
        st.plotly_chart(chart_monthly_returns(r, color), use_container_width=True, key=f"chart_returns_{i}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PHASE HEATMAP
# ════════════════════════════════════════════════════════════════════════════════
with tab_phase:
    st.subheader("Phase Heatmap  相位热图")
    st.caption(
        "Each row = one stock. Each cell = that month's return. "
        "▲ = detected peak (sell zone). ▽ = detected trough (buy zone). "
        "Read vertically to see whether stocks peak/trough at the same time."
    )
    st.plotly_chart(chart_phase_heatmap(results_sorted), use_container_width=True, key="chart_phase_heatmap")

    if len(results_sorted) >= 2:
        st.subheader("Concurrent Correlation Matrix  同期相关矩阵")
        st.caption(
            "Lower correlation = stocks move more independently = better rotation pair. "
            "Ideal < 0.4 between rotation partners."
        )
        st.plotly_chart(chart_correlation_matrix(results_sorted), use_container_width=True, key="chart_corr_matrix_phase")
        show_rotation_expander()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — CORRELATION  (comprehensive per-pair analysis)
# ════════════════════════════════════════════════════════════════════════════════
with tab_corr:
    st.subheader("Pairwise Correlation Analysis  两两相关性分析")
    st.caption(
        "For each pair of stocks: (1) overlaid normalised price series so you can see divergence moment by moment, "
        "(2) a rolling correlation window to see whether co-movement is stable or regime-shifting, "
        "(3) a return scatter to visualise the linear relationship."
    )

    if len(results_sorted) < 2:
        st.info("Add at least 2 tickers to see pairwise analysis.")
    else:
        # ── rolling window selector ────────────────────────────────────────────
        _win_options = {"Daily": [10, 20, 40, 60], "Weekly": [4, 8, 13, 26], "Monthly": [3, 6, 9, 12]}
        _win_labels  = {"Daily": "days", "Weekly": "weeks", "Monthly": "months"}
        _gran = results_sorted[0].get("granularity", "Weekly")
        _wins = _win_options.get(_gran, [4, 8, 13, 26])
        _unit = _win_labels.get(_gran, "bars")

        roll_win = st.select_slider(
            f"Rolling correlation window ({_unit})",
            options=_wins,
            value=_wins[1],
            help=f"Shorter = more reactive to regime changes. Longer = smoother, shows structural relationship.",
        )

        # Generate all unique pairs
        import itertools
        pairs = list(itertools.combinations(results_sorted, 2))

        for (rA, rB) in pairs:
            tA, tB = rA["ticker"], rA["name"], 
            tA, nameA = rA["ticker"], rA["name"]
            tB, nameB = rB["ticker"], rB["name"]
            colorA = WAVE_COLORS[results_sorted.index(rA) % len(WAVE_COLORS)]
            colorB = WAVE_COLORS[results_sorted.index(rB) % len(WAVE_COLORS)]

            st.markdown(f"---")
            st.markdown(f"### {tA} {nameA}  ↔  {tB} {nameB}")

            # Get aligned raw DataFrames from the cache payload
            dfA = raw_frames.get(tA)
            dfB = raw_frames.get(tB)

            if dfA is None or dfB is None:
                st.warning(f"Raw data missing for {tA} or {tB}.")
                continue

            # Align on common index
            common_idx = dfA.index.intersection(dfB.index)
            if len(common_idx) < 6:
                st.warning(f"Only {len(common_idx)} common bars between {tA} and {tB} — not enough to plot.")
                continue

            dfA_al = dfA.loc[common_idx, "Close"]
            dfB_al = dfB.loc[common_idx, "Close"]

            # Normalise to 100 at first common bar
            normA = dfA_al / dfA_al.iloc[0] * 100
            normB = dfB_al / dfB_al.iloc[0] * 100
            dates_common = common_idx

            # Period-over-period returns
            retA = dfA_al.pct_change().dropna()
            retB = dfB_al.pct_change().dropna()
            ret_idx = retA.index.intersection(retB.index)
            retA = retA.loc[ret_idx]
            retB = retB.loc[ret_idx]

            # Rolling correlation
            import numpy as _np
            def rolling_corr(s1, s2, w):
                out = []
                for i in range(len(s1)):
                    if i < w - 1:
                        out.append(None)
                    else:
                        a = s1.iloc[i - w + 1: i + 1].values
                        b = s2.iloc[i - w + 1: i + 1].values
                        if _np.std(a) < 1e-9 or _np.std(b) < 1e-9:
                            out.append(0.0)
                        else:
                            out.append(float(_np.corrcoef(a, b)[0, 1]))
                return out

            roll_corr_vals = rolling_corr(retA, retB, roll_win)
            full_corr_val  = _concurrent_corr(retA.tolist(), retB.tolist())

            # ── CHART 1: Overlaid normalised price ───────────────────────────
            fig_price = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.65, 0.35],
                vertical_spacing=0.05,
                subplot_titles=["Normalised Price (base = 100 at start)  归一化价格", f"Rolling {roll_win}-bar Correlation  滚动相关系数"],
            )

            fig_price.add_trace(go.Scatter(
                x=dates_common, y=normA.values,
                name=tA, line=dict(color=colorA, width=2),
                hovertemplate=f"{tA}<br>%{{x}}<br>Index: %{{y:.1f}}<extra></extra>",
            ), row=1, col=1)

            fig_price.add_trace(go.Scatter(
                x=dates_common, y=normB.values,
                name=tB, line=dict(color=colorB, width=2),
                hovertemplate=f"{tB}<br>%{{x}}<br>Index: %{{y:.1f}}<extra></extra>",
            ), row=1, col=1)

            # Divergence shading: fill between the two normalised series
            # Green where A > B, red where B > A
            yA = normA.values.tolist()
            yB = normB.values.tolist()
            x_all = list(dates_common)
            fig_price.add_trace(go.Scatter(
                x=x_all + x_all[::-1],
                y=yA + yB[::-1],
                fill="toself",
                fillcolor=f"rgba({int(colorA[1:3],16)},{int(colorA[3:5],16)},{int(colorA[5:7],16)},0.13)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name=f"{tA} ahead",
            ), row=1, col=1)

            # Rolling correlation line
            valid_dates = [dates_common[i] for i in range(len(roll_corr_vals)) if roll_corr_vals[i] is not None]
            valid_corr  = [v for v in roll_corr_vals if v is not None]

            if valid_corr:
                fig_price.add_trace(go.Scatter(
                    x=valid_dates, y=valid_corr,
                    name=f"Rolling {roll_win}-bar corr",
                    line=dict(color="#facc15", width=2),
                    hovertemplate="Corr: %{y:.2f}<extra></extra>",
                ), row=2, col=1)

                # Colour bands
                fig_price.add_hrect(y0=0.7,  y1=1.0,  fillcolor="rgba(239,68,68,0.08)",  line_width=0, row=2, col=1)
                fig_price.add_hrect(y0=-1.0, y1=-0.4, fillcolor="rgba(16,185,129,0.08)", line_width=0, row=2, col=1)
                fig_price.add_hline(y=full_corr_val, line_dash="dot", line_color="rgba(255,255,255,0.4)",
                                    annotation_text=f"Full-period avg: {full_corr_val:.2f}",
                                    annotation_position="top left", row=2, col=1)
                fig_price.add_hline(y=0, line_color="rgba(255,255,255,0.2)", row=2, col=1)

            fig_price.update_yaxes(title_text="Index", row=1, col=1)
            fig_price.update_yaxes(title_text="Corr", range=[-1.05, 1.05], row=2, col=1)
            fig_price.update_layout(
                height=520, template="plotly_dark",
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(orientation="h", y=-0.08, font=dict(size=11)),
                hovermode="x unified",
            )
            st.plotly_chart(fig_price, use_container_width=True, key=f"chart_price_{tA}_{tB}")

            # ── CHART 2: Return scatter ───────────────────────────────────────
            c_left, c_right = st.columns([2, 1])
            with c_left:
                retA_pct = (retA * 100).values.tolist()
                retB_pct = (retB * 100).values.tolist()
                dates_lbl = [str(d.date()) for d in ret_idx]

                # Linear regression line
                if len(retA_pct) >= 4:
                    xa = _np.array(retA_pct)
                    ya = _np.array(retB_pct)
                    coef = _np.polyfit(xa, ya, 1)
                    x_line = [min(xa), max(xa)]
                    y_line = [coef[0] * x + coef[1] for x in x_line]
                else:
                    x_line, y_line = [], []

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=retA_pct, y=retB_pct,
                    mode="markers",
                    marker=dict(
                        color=[i for i in range(len(retA_pct))],
                        colorscale="Viridis",
                        size=8, opacity=0.8,
                        colorbar=dict(title="Bar #", thickness=10, len=0.6),
                        showscale=True,
                    ),
                    text=dates_lbl,
                    hovertemplate=f"{tA}: %{{x:+.2f}}%<br>{tB}: %{{y:+.2f}}%<br>%{{text}}<extra></extra>",
                    name="Returns",
                ))
                if x_line:
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode="lines",
                        line=dict(color="rgba(255,255,255,0.5)", dash="dash", width=1.5),
                        name="Regression",
                        hoverinfo="skip",
                    ))
                fig_scatter.add_vline(x=0, line_color="rgba(255,255,255,0.2)")
                fig_scatter.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
                fig_scatter.update_layout(
                    title=f"Return Scatter: {tA} (x) vs {tB} (y)  — each dot = one {_gran.lower()} bar",
                    xaxis_title=f"{tA} return (%)",
                    yaxis_title=f"{tB} return (%)",
                    height=380, template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False,
                )
                st.plotly_chart(fig_scatter, use_container_width=True, key=f"chart_scatter_{tA}_{tB}")

            with c_right:
                # Stats summary
                corr_series = [v for v in roll_corr_vals if v is not None]
                corr_min = min(corr_series) if corr_series else 0
                corr_max = max(corr_series) if corr_series else 0
                corr_std = float(_np.std(corr_series)) if corr_series else 0

                # Count bars where stocks diverged (opposite signs)
                opposite_bars = sum(1 for a, b in zip(retA_pct, retB_pct) if a * b < 0)
                opp_pct = opposite_bars / len(retA_pct) * 100 if retA_pct else 0

                # Lag-1 correlation (B leads A by 1 bar)
                if len(retA.values) > 2:
                    lag1_corr = float(_np.corrcoef(retA.values[1:], retB.values[:-1])[0, 1])
                    lag1_corr_rev = float(_np.corrcoef(retA.values[:-1], retB.values[1:])[0, 1])
                else:
                    lag1_corr, lag1_corr_rev = 0.0, 0.0

                st.markdown("#### Pair Stats  配对统计")
                st.metric("Full-period Correlation", f"{full_corr_val:+.2f}")
                st.metric(f"Rolling Corr Range ({roll_win}-bar)", f"{corr_min:+.2f} → {corr_max:+.2f}")
                st.metric("Corr Stability (std)", f"{corr_std:.2f}",
                          help="Lower = correlation is stable. Higher = regime shifts between correlated and independent.")
                st.metric("Bars Moving Opposite", f"{opp_pct:.0f}%",
                          help=f"% of {_gran.lower()} bars where one stock went up while the other went down. Higher = better rotation pair.")
                st.metric(f"Lag +1 ({tB} leads {tA})", f"{lag1_corr:+.2f}",
                          help=f"Does {tB} moving today predict {tA} moving next bar?")
                st.metric(f"Lag +1 ({tA} leads {tB})", f"{lag1_corr_rev:+.2f}",
                          help=f"Does {tA} moving today predict {tB} moving next bar?")

                # Qualitative verdict
                if full_corr_val < 0.35:
                    verdict, vcolor = "✅ Independent — excellent rotation pair", "#10b981"
                elif full_corr_val < 0.60:
                    verdict, vcolor = "⚠️ Partial overlap — acceptable pair", "#f59e0b"
                else:
                    verdict, vcolor = "❌ High co-movement — poor rotation pair", "#ef4444"
                st.markdown(f"<div style='margin-top:12px; padding:10px; border-radius:8px; background:{vcolor}22; "
                            f"border:1px solid {vcolor}55; color:{vcolor}; font-weight:600; font-size:0.9rem;'>"
                            f"{verdict}</div>", unsafe_allow_html=True)

        # ── Correlation matrix (summary) ───────────────────────────────────
        st.markdown("---")
        st.subheader("Correlation Matrix Summary  相关矩阵汇总")
        st.plotly_chart(chart_correlation_matrix(results_sorted), use_container_width=True, key="chart_corr_matrix_summary")
        show_rotation_expander()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROTATION
# ════════════════════════════════════════════════════════════════════════════════
with tab_rotation:
    st.subheader("Rotation Ranking & Strategy  轮动策略")
    show_rotation_expander()

    # ── Rotation rank table ────────────────────────────────────────────────────
    st.markdown("#### Stocks Ranked by Wave Score  按波段评分排名")
    rank_rows = []
    for rank, r in enumerate(results_sorted, 1):
        sig, (sc, icon) = r["signal"], SIGNAL_COLORS[r["signal"]]
        corrs = []
        for other in results_sorted:
            if other["ticker"] != r["ticker"]:
                c = _concurrent_corr(r["monthly_returns"], other["monthly_returns"])
                corrs.append(f"{other['ticker']}: {c:+.2f}")
        rank_rows.append({
            "Rank": rank,
            "Ticker": r["ticker"],
            "Name": r["name"],
            "Wave Score": r["wave_score"],
            "Signal": f"{icon} {sig}",
            "Hurst": r["hurst"],
            "Range Pos %": r["range_pct"],
            "3M Mom %": r["mom_3m_pct"],
            "Peer Correlations": " | ".join(corrs),
        })
    df_rank = pd.DataFrame(rank_rows)
    st.dataframe(
        df_rank.style
            .background_gradient(subset=["Wave Score"], cmap="RdYlGn", vmin=0, vmax=100)
            .background_gradient(subset=["3M Mom %"], cmap="RdYlGn", vmin=-30, vmax=30)
            .background_gradient(subset=["Range Pos %"], cmap="RdYlGn_r", vmin=0, vmax=100)
            .format({
                "Wave Score": "{:.0f}", "Hurst": "{:.3f}",
                "Range Pos %": "{:.0f}", "3M Mom %": "{:+.1f}",
            }),
        use_container_width=True, hide_index=True,
    )

    # ── Rotation flow diagram ──────────────────────────────────────────────────
    st.markdown("#### Capital Flow Order  资金轮动顺序")
    st.caption(
        "Exit the stock showing EXIT first. Rotate capital into the next highest-scored "
        "stock showing BUY or HOLD. Repeat when each stock tops out."
    )

    top3 = results_sorted[:3]
    flow_cols = st.columns(len(top3) * 2 - 1)
    for i, r in enumerate(top3):
        sig, (sc, icon) = r["signal"], SIGNAL_COLORS[r["signal"]]
        color = WAVE_COLORS[i % len(WAVE_COLORS)]
        ci = i * 2
        with flow_cols[ci]:
            st.markdown(
                f"""
                <div style="border:1px solid {color}50; border-radius:10px; padding:14px; text-align:center;
                            background:{color}10;">
                    <div style="font-family:monospace; font-weight:800; font-size:1.1rem; color:{color};">
                        #{i+1} &nbsp; {r['ticker']}
                    </div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin:2px 0 8px;">{r['name']}</div>
                    <div style="font-size:1.6rem; font-weight:900; color:{color};">{r['wave_score']:.0f}</div>
                    <div style="font-size:0.7rem; color:#64748b;">Wave Score</div>
                    <div style="margin-top:8px; padding:4px 8px; border-radius:20px;
                                background:{sc}20; color:{sc}; font-weight:700; font-size:0.8rem;">
                        {icon} {sig}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if i < len(top3) - 1:
            with flow_cols[ci + 1]:
                st.markdown(
                    "<div style='text-align:center; font-size:2rem; color:#475569; "
                    "padding-top:28px;'>→</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Current tactical summary ───────────────────────────────────────────────
    st.markdown("#### Tactical Reading  当前战术解读")
    for r in results_sorted:
        sig = r["signal"]
        sc, icon = SIGNAL_COLORS[sig]
        if sig == "EXIT":
            advice = (
                f"**{r['ticker']}** is at {r['range_pct']:.0f}% of its 52-week range "
                f"with only {r['drawdown_pct']:.1f}% off the peak. "
                "The wave has likely crested — trim or exit before mean-reversion."
            )
        elif sig == "BUY":
            advice = (
                f"**{r['ticker']}** is showing a buy signal: "
                f"range position {r['range_pct']:.0f}%, 3M momentum {r['mom_3m_pct']:+.1f}%. "
                "Price is in an attractive entry zone for the next upleg."
            )
        elif sig == "WATCH":
            advice = (
                f"**{r['ticker']}** is falling fast ({r['mom_3m_pct']:+.1f}% over 3 months). "
                "Wait for the bleeding to stop — don't catch the knife."
            )
        else:
            advice = (
                f"**{r['ticker']}** is in a neutral zone (range pos {r['range_pct']:.0f}%, "
                f"3M mom {r['mom_3m_pct']:+.1f}%). "
                "Hold existing positions; no strong new entry or exit signal."
            )
        st.markdown(
            f"<div style='border-left:3px solid {sc}; padding:8px 14px; margin-bottom:8px; "
            f"background:{sc}0d; border-radius:0 6px 6px 0;'>{icon} {advice}</div>",
            unsafe_allow_html=True,
        )

    # ── MPMR regime reminder ───────────────────────────────────────────────────
    with st.expander("⚠️ MPMR Regime Warning  宏观制度提醒", expanded=False):
        st.warning("""
**This rotation strategy works in Risk-On and Choppy Sideways regimes.**

If your MPMR / NLI model is currently reading a ResMgmt (tariff shock) or QT regime:
- All three stocks can correlate to 1.0 on the downside regardless of their phase relationship
- The wave structure temporarily breaks down under systemic stress
- **Action:** Reduce position size across all three. Use signal-based entries only, not calendar-based timing.
- If NLI slope turns sharply negative for 3+ consecutive weeks → consider going flat across all positions.
        """)
