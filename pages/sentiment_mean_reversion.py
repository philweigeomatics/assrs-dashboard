"""
Sentiment-Driven Mean Reversion Scanner  ·  反转候选筛选

Scans the current user's watchlist for A-share stocks that retail has
hammered down too far / too fast on pure sentiment — where a snapback
within 2–5 trading days is statistically likely.

STRATEGY (3 layers — all hard rules must pass for a STRONG candidate)

  Layer 1 — Price Oversold Exhaustion
    • 1-day return z-score < -2.5         (price abnormally far below mean)
    • Consecutive down days ≥ 4           (panic cascade)
    • 1–2 跌停 in the last 5 days (bonus) (maximum-fear capitulation)

  Layer 2 — Volume / Sellers Exhausted
    • Volume declining on the down days   (sellers running out of inventory)

  Layer 3 — Confirmation
    • RSI(14) < 25                        (deeply oversold)
    • Sector peers NOT equally down       (isolated panic, not systemic)

REJECT
  • ST / *ST stocks                       (delisting risk)

Each watchlist ticker gets a verdict:
  🟢 STRONG     — all 5 hard rules pass
  🟡 WATCH      — 3-4 hard rules pass
  ⚪ Not now    — < 3 rules pass
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import auth_manager
import data_manager

auth_manager.require_login()

st.set_page_config(
    page_title="Mean Reversion Scanner | 反转候选",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Sentiment-Driven Mean Reversion · 反转候选")
st.caption(
    "Find stocks in your watchlist that retail has panic-sold too far, where "
    "selling pressure is exhausting and a 2–5 day snapback is likely."
)

# ── Strategy explainer ────────────────────────────────────────────────────────
with st.expander("📖 What this scanner looks for", expanded=False):
    st.markdown("""
**The setup**: a stock retail has hammered down on pure sentiment / panic —
*not* on a fundamental change — where selling is exhausting itself and a
snapback is mechanically likely within 2–5 days.

**The 3 layers (all must pass for STRONG):**

| Layer | Signal | Threshold |
|---|---|---|
| **1 · Price oversold** | 1-day return z-score | `< -2.5` |
| **1 · Price oversold** | Consecutive down days | `≥ 4` |
| **2 · Sellers exhausted** | Volume declining on down days | True |
| **3 · Confirmation** | RSI(14) | `< 25` |
| **3 · Confirmation** | Stock weaker than sector median 5-day return | by ≥ 5 pp |

**Bonus signals (informational, not gating):**
- 1–2 consecutive 跌停 (limit-down) hits → maximum-fear capitulation
- Stock is near MA60 / MA120 (a structural support level)
- Volume spike then contraction on the down days

**Automatically rejected:**
- ST / *ST stocks (delisting risk)

**Action template for a STRONG candidate:**
- Buy at next-day open (T+1)
- Target: MA20 reversion or z-score back above -0.5
- Stop: new 跌停 the next day OR fundamental bad news confirmed
""")

# ── Tunable parameters ───────────────────────────────────────────────────────
with st.expander("⚙️ Parameters (defaults match the spec)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    z_thresh         = c1.number_input("Z-score threshold (≤)",    value=-2.5, step=0.1, max_value=0.0)
    rsi_thresh       = c2.number_input("RSI threshold (<)",         value=25,   step=1,   min_value=1, max_value=50)
    down_days_min    = c3.number_input("Min consecutive down days", value=4,    step=1,   min_value=1, max_value=10)
    sector_div_pp    = c4.number_input("Sector divergence (pp)",    value=5.0,  step=0.5, min_value=0.0,
                                       help="Stock's 5-day return must be at least this many pp weaker than its sector median.")

# ── Computation helpers ──────────────────────────────────────────────────────

def _zscore_of_latest_return(close: pd.Series, window: int = 20) -> float | None:
    """Z-score of the most recent daily return vs the prior `window` returns."""
    rets = close.pct_change().dropna()
    if len(rets) < window + 1:
        return None
    sample = rets.iloc[-(window + 1):-1]
    mu, sd = sample.mean(), sample.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return None
    return float((rets.iloc[-1] - mu) / sd)


def _rsi(close: pd.Series, window: int = 14) -> float | None:
    """Classic Wilder RSI on the close series. Returns last RSI value."""
    if len(close) < window + 1:
        return None
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(window).mean()
    loss   = (-delta.clip(upper=0)).rolling(window).mean()
    if loss.iloc[-1] == 0 or pd.isna(loss.iloc[-1]):
        return 100.0
    rs  = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def _consecutive_down_days(close: pd.Series) -> int:
    """Trailing run of negative-return days ending on the latest bar."""
    rets = close.pct_change().dropna()
    n = 0
    for r in reversed(rets.tolist()):
        if r < 0:
            n += 1
        else:
            break
    return n


def _volume_exhaustion(close: pd.Series, volume: pd.Series, lookback: int = 5) -> bool:
    """
    Returns True if, over the last `lookback` down days, average volume is
    LOWER than over the prior `lookback` down days — i.e. selling pressure
    is fading even as price keeps dropping.
    """
    rets = close.pct_change()
    df = pd.concat([rets.rename("ret"), volume.rename("vol")], axis=1).dropna()
    down = df[df["ret"] < 0]
    if len(down) < 2 * lookback:
        # Fall back to a simpler check: latest down-day volume below the
        # mean of the prior 5 down-day volumes.
        if len(down) < 3:
            return False
        return down["vol"].iloc[-1] < down["vol"].iloc[-min(6, len(down)):-1].mean()
    recent_avg = down["vol"].iloc[-lookback:].mean()
    prior_avg  = down["vol"].iloc[-2 * lookback:-lookback].mean()
    return bool(recent_avg < prior_avg)


def _limit_down_streak(ts_code: str, lookback_days: int = 5) -> int:
    """
    Count limit-down days for `ts_code` in the most recent `lookback_days`
    trading days, using Tushare's stk_limit table. Returns 0 if the API
    is unavailable or no data.
    """
    try:
        data_manager.init_tushare()
        if data_manager.TUSHARE_API is None:
            return 0
        end   = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=lookback_days * 2 + 5)).strftime("%Y%m%d")
        # stk_limit returns up_limit / down_limit for the trade_date
        limits = data_manager.TUSHARE_API.stk_limit(
            ts_code=ts_code, start_date=start, end_date=end,
        )
        if limits is None or limits.empty:
            return 0
        # Pull close prices for the same range
        daily = data_manager.TUSHARE_API.daily(
            ts_code=ts_code, start_date=start, end_date=end,
        )
        if daily is None or daily.empty:
            return 0
        merged = limits.merge(
            daily[["trade_date", "close"]], on="trade_date", how="inner",
        ).sort_values("trade_date").tail(lookback_days)
        # close <= down_limit (within float tolerance) → limit-down hit
        ld = (merged["close"] <= merged["down_limit"] + 1e-4)
        return int(ld.sum())
    except Exception:
        return 0


def _is_st_stock(name: str | None) -> bool:
    if not name:
        return False
    return name.upper().startswith(("ST", "*ST"))


def _to_ts_code(ticker6: str) -> str:
    """Convert a 6-digit code to a Tushare ts_code with the correct suffix."""
    try:
        return data_manager.get_tushare_ticker(ticker6)
    except Exception:
        # Fallback heuristic
        if ticker6.startswith(("6", "9")):
            return f"{ticker6}.SH"
        return f"{ticker6}.SZ"


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_prices(ticker: str) -> pd.DataFrame | None:
    """Pull ≈1y of qfq OHLCV for a ticker. Cached for 15 minutes."""
    try:
        return data_manager.get_single_stock_data_live(ticker, lookback_years=1)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _reverse_sector_map() -> dict[str, str]:
    """ticker → sector_name. Built once per hour from the sector map DB."""
    out: dict[str, str] = {}
    try:
        smap = data_manager.get_sector_stock_map()
        for sector, tickers in smap.items():
            for t in tickers:
                out[t] = sector
    except Exception:
        pass
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _sector_median_5d_return(sector_name: str) -> float | None:
    """
    Median 5-trading-day return across all stocks in `sector_name`, using
    the per-sector PPI table — far faster than fetching every constituent.
    Falls back to None if the PPI table is unavailable.
    """
    try:
        df = data_manager.db.read_table(
            f"PPI_{sector_name}",
            columns="Date, Close",
            order_by="-Date",
            limit=10,
        )
        if df is None or len(df) < 6:
            return None
        df = df.sort_values("Date")
        return float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100
    except Exception:
        return None


# ── Scan ────────────────────────────────────────────────────────────────────

scan_col, _ = st.columns([1, 4])
scan = scan_col.button("🔍 Scan watchlist now", type="primary", use_container_width=True)

if not scan:
    st.info("Click **Scan watchlist now** to evaluate your watchlist against the rules above. "
            "Results cache for 15 minutes — re-scan after market close for fresh data.")
    st.stop()

watchlist = data_manager.get_watchlist()
if not watchlist:
    st.warning("📭 Your watchlist is empty. Add stocks in the Watchlist page first.")
    st.stop()

st.markdown(f"**Scanning {len(watchlist)} stocks…**")
progress = st.progress(0.0, text="Starting scan…")
status   = st.empty()

reverse_sector = _reverse_sector_map()
rows: list[dict] = []

for idx, item in enumerate(watchlist, start=1):
    ticker = item["ticker"]
    name   = item.get("stock_name") or data_manager.get_stock_name_from_db(ticker) or ticker
    progress.progress(idx / len(watchlist), text=f"{idx}/{len(watchlist)} · {ticker} {name}")

    # ── Reject 1: ST / *ST ─────────────────────────────────────────────────
    if _is_st_stock(name):
        rows.append({
            "_status":      "⛔",
            "_status_rank": 3,
            "Ticker":       ticker,
            "Name":         name,
            "Verdict":      "Skipped — ST/*ST",
            "Z-score":      None, "RSI": None, "Down days": None,
            "Vol exhaust": None, "跌停 streak": None,
            "5d ret %":     None, "Sector 5d %": None, "vs sector": None,
            "Rules passed": "—",
        })
        continue

    df = _fetch_prices(ticker)
    if df is None or df.empty or len(df) < 30:
        rows.append({
            "_status":      "⚠️",
            "_status_rank": 2,
            "Ticker":       ticker,
            "Name":         name,
            "Verdict":      "No price data",
            "Z-score":      None, "RSI": None, "Down days": None,
            "Vol exhaust": None, "跌停 streak": None,
            "5d ret %":     None, "Sector 5d %": None, "vs sector": None,
            "Rules passed": "—",
        })
        continue

    close  = df["Close"]
    volume = df["Volume"]

    # ── Metrics ────────────────────────────────────────────────────────────
    z         = _zscore_of_latest_return(close, window=20)
    rsi       = _rsi(close, window=14)
    down_run  = _consecutive_down_days(close)
    vol_ex    = _volume_exhaustion(close, volume, lookback=5)
    ts_code   = _to_ts_code(ticker)
    ld_streak = _limit_down_streak(ts_code, lookback_days=5)

    if len(close) >= 6:
        stock_5d_ret = float(close.iloc[-1] / close.iloc[-6] - 1) * 100
    else:
        stock_5d_ret = None

    sector       = reverse_sector.get(ticker)
    sector_5d    = _sector_median_5d_return(sector) if sector else None
    vs_sector_pp = (stock_5d_ret - sector_5d) if (stock_5d_ret is not None and sector_5d is not None) else None

    # ── Hard rules ─────────────────────────────────────────────────────────
    r_z       = z is not None and z <= z_thresh
    r_down    = down_run >= down_days_min
    r_vol     = vol_ex
    r_rsi     = rsi is not None and rsi < rsi_thresh
    # Sector divergence: stock must be at least `sector_div_pp` pp weaker.
    # If we have no sector data, we DON'T fail it — we mark as N/A and count
    # as a "soft pass" (so a watchlist ticker outside any tracked sector
    # isn't punished for the missing data).
    if vs_sector_pp is None:
        r_sector = None
    else:
        r_sector = vs_sector_pp <= -sector_div_pp

    flags = [r_z, r_down, r_vol, r_rsi, r_sector]
    hard_passed = sum(1 for f in flags if f is True)
    # Treat N/A as half-credit so it doesn't dominate the verdict either way
    soft_passed = hard_passed + (0.5 if r_sector is None else 0)

    if all(f is True for f in flags if f is not None) and hard_passed >= 4:
        verdict, status_emoji, rank = "🟢 STRONG candidate", "🟢", 0
    elif soft_passed >= 3:
        verdict, status_emoji, rank = "🟡 Watch", "🟡", 1
    else:
        verdict, status_emoji, rank = "⚪ Not now", "⚪", 2

    rules_str = (
        f"{'✓' if r_z       else '✗'} Z  "
        f"{'✓' if r_down    else '✗'} Down  "
        f"{'✓' if r_vol     else '✗'} Vol  "
        f"{'✓' if r_rsi     else '✗'} RSI  "
        f"{'✓' if r_sector is True else ('—' if r_sector is None else '✗')} Sector"
    )

    rows.append({
        "_status":      status_emoji,
        "_status_rank": rank,
        "Ticker":       ticker,
        "Name":         name,
        "Verdict":      verdict,
        "Z-score":      round(z, 2) if z is not None else None,
        "RSI":          round(rsi, 1) if rsi is not None else None,
        "Down days":    down_run,
        "Vol exhaust":  "✓" if vol_ex else "✗",
        "跌停 streak":  ld_streak,
        "5d ret %":     round(stock_5d_ret, 2) if stock_5d_ret is not None else None,
        "Sector 5d %":  round(sector_5d, 2) if sector_5d is not None else None,
        "vs sector":    round(vs_sector_pp, 2) if vs_sector_pp is not None else None,
        "Rules passed": rules_str,
    })

progress.empty()
status.empty()

# ── Display ─────────────────────────────────────────────────────────────────
results = pd.DataFrame(rows).sort_values(["_status_rank", "Z-score"], na_position="last")
n_strong = int((results["Verdict"].str.startswith("🟢", na=False)).sum())
n_watch  = int((results["Verdict"].str.startswith("🟡", na=False)).sum())
n_skip   = int(results["Verdict"].str.startswith("Skipped", na=False).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("🟢 Strong candidates", n_strong)
m2.metric("🟡 Watch",             n_watch)
m3.metric("⚪ Not now",           len(results) - n_strong - n_watch - n_skip)
m4.metric("⛔ Skipped (ST etc.)", n_skip)

if n_strong > 0:
    st.success(
        f"**{n_strong} strong candidate{'s' if n_strong != 1 else ''}** met all rules. "
        "Suggested action: buy at next open, target MA20 reversion, stop on new 跌停 or bad-news confirmation."
    )

# Drop the internal sort key before showing
display = results.drop(columns=["_status", "_status_rank"])
st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Verdict":       st.column_config.TextColumn(width="medium"),
        "Z-score":       st.column_config.NumberColumn(format="%.2f", help="Z-score of latest daily return vs prior 20 days"),
        "RSI":           st.column_config.NumberColumn(format="%.1f"),
        "Down days":     st.column_config.NumberColumn(help="Consecutive down days ending today"),
        "Vol exhaust":   st.column_config.TextColumn(help="Volume declining on recent down days (sellers exhausting)"),
        "跌停 streak":   st.column_config.NumberColumn(help="Limit-down hits in last 5 trading days"),
        "5d ret %":      st.column_config.NumberColumn(format="%.2f"),
        "Sector 5d %":   st.column_config.NumberColumn(format="%.2f", help="Sector PPI 5-day return"),
        "vs sector":     st.column_config.NumberColumn(format="%+.2f", help="Stock 5d return minus sector 5d return (negative = stock weaker than peers = isolated panic)"),
        "Rules passed":  st.column_config.TextColumn(width="medium"),
    },
)

st.caption(
    "Z-score = standardised distance of the latest 1-day return from the prior 20-day distribution. "
    "RSI = 14-day Wilder's. Vol exhaust = last 5 down-day average volume < prior 5 down-day average. "
    "Sector divergence uses each sector's PPI as the peer benchmark. "
    "Data cached for 15 minutes — click scan again to refresh."
)
