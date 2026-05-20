"""
T-Trading Scanner  ·  做T候选筛选

Ranks the current user's watchlist by suitability for intraday-T trading
(trading around an existing base position 底仓 — sell rally / buy back dip,
or vice-versa — to harvest intraday range without violating A-share T+1).

The composite T-score has five components plus three hard-fail gates.
See the in-page "📖 What this scanner looks for" expander for the full
breakdown and how to interpret each verdict.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import auth_manager
import data_manager

auth_manager.require_login()

st.set_page_config(
    page_title="T-Trading Scanner | 做T候选",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ T-Trading Scanner · 做T候选")
st.caption(
    "Find stocks in your watchlist that are STRUCTURALLY well-suited to "
    "intraday做T around a base position — large enough intraday range to "
    "matter, mean-reverting enough to harvest, liquid enough to actually fill."
)

# ── Strategy & scoring explainer ─────────────────────────────────────────────
with st.expander("📖 What this scanner looks for · 评分组成", expanded=False):
    st.markdown("""
**Reminder — what "做T" means on A-shares**

Because A-shares are T+1, you can't intraday-flip *new* shares. So 做T means
trading around an **existing base position (底仓)**: you sell some on an
intraday rally and buy back on a dip (正T), or buy a dip first then sell into
a rally (倒T). Net position size unchanged at end of day, but realised P&L
captures the intraday swing. The T+1 lock only applies to the *net new*
shares you bought today.

So a good T candidate needs **intraday range + mean-reversion + liquidity** —
*not* just any volatile name.

---

**The composite T-score (0–100)**

| # | Component | Weight | What it measures |
|---|---|---|---|
| 1 | **Intraday Range** | 30 % | 20-day avg `(High − Low) / Open` — has to be big enough to clear costs. Target ≥ 4 %. |
| 2 | **Liquidity** | 25 % | 20-day avg `turnover_rate` — tight spreads + ability to enter/exit at size. Target ≥ 8 %. |
| 3 | **Mean-Reversion Bias** | 25 % | 20-day avg `\|Close − Open\| / (High − Low)` — lower is better. Below 0.4 = Close lands in middle of day's range (oscillator, perfect for T). Above 0.7 = trending intraday (the dip you sell into keeps falling). |
| 4 | **ADX Regime** | 10 % | ADX(14) in [15, 35] = 1.0. Below 15 → no movement; above 35 → strong directional trend (limit-day risk). |
| 5 | **Range Position** | 10 % | Distance from 60-day high/low. Middle of range = 1.0; within ±5 % of an extreme = 0. |

**Hard-fail gates** (one trip → automatically rejected, regardless of score):
- 20-day avg turnover rate **< 2 %** (illiquid)
- 20-day avg intraday range **< 1 %** (no movement to harvest)
- Any 跌停 or 涨停 in the last 5 trading days (position-freeze risk)

---

**Verdicts**

| Verdict | Score range | Meaning |
|---|---|---|
| 🟢 STRONG | ≥ 75 | Best candidates — structurally suited to T trading right now |
| 🟡 OK | 55–74 | Workable but compromised on one dimension (lower range or less mean-reverting) |
| ⚪ Not now | < 55 | Score too low to be worth the friction |
| ⛔ Skip | (hard fail) | Disqualified by liquidity / range / limit-event gate |

**How to use the result table**
- Sort defaults to highest score first
- Hover over a column header for a one-line definition
- Click ticker → opens Single Stock Analysis for that ticker
- Re-scan to refresh (data cached 15 min)

---

**Action template for a 🟢 STRONG candidate**
1. Confirm you already hold a 底仓 in the name (or open one in a separate trade).
2. Define an intraday band around your cost basis: e.g. sell 30 % of position at +1.5 %, buy back at -0.5 %.
3. Use the day's opening 30 minutes to estimate the range; place limit orders, not market.
4. End the day at original size — don't get caught with a doubled position into close (you'd be T+1-locked).
""")

# ── Parameters ───────────────────────────────────────────────────────────────
with st.expander("⚙️ Parameters (defaults match the spec above)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    range_target    = c1.number_input("Range target (%)",    value=4.0,  step=0.5, min_value=1.0, max_value=10.0,
                                       help="Intraday range above which Component 1 saturates to 1.0.")
    turnover_target = c2.number_input("Turnover target (%)", value=8.0,  step=0.5, min_value=2.0, max_value=20.0,
                                       help="20d avg turnover rate above which Component 2 saturates to 1.0.")
    adx_band_lo     = c3.number_input("ADX band low",        value=15,   step=1,   min_value=5,  max_value=30,
                                       help="Below this ADX, Component 4 starts decaying toward 0.")
    adx_band_hi     = c4.number_input("ADX band high",       value=35,   step=1,   min_value=20, max_value=60,
                                       help="Above this ADX, Component 4 starts decaying toward 0 (strong directional trend).")
    extreme_pct     = c5.number_input("Extreme zone (%)",    value=5.0,  step=0.5, min_value=1.0, max_value=15.0,
                                       help="If price is within ±this % of the 60d high or low, Component 5 = 0.")
    st.caption(
        "Hard-fail thresholds: turnover < 2 %, range < 1 %, recent 跌停/涨停 in last 5 days. "
        "These are intentionally not exposed — they're disqualifiers, not tunable preferences."
    )

# ── Computation helpers ──────────────────────────────────────────────────────

def _intraday_range_pct(df: pd.DataFrame, window: int = 20) -> float | None:
    """20-day mean of (High - Low) / Open, in percent."""
    if len(df) < window:
        return None
    r = ((df['High'] - df['Low']) / df['Open']).tail(window)
    return float(r.mean() * 100)


def _mean_reversion_bias(df: pd.DataFrame, window: int = 20) -> float | None:
    """20-day mean of |Close - Open| / (High - Low). 0 = perfect mean-revert, 1 = perfect trend."""
    if len(df) < window:
        return None
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    bias = (df['Close'] - df['Open']).abs() / rng
    bias = bias.dropna().tail(window)
    if bias.empty:
        return None
    return float(bias.mean())


def _rsi_independent_adx(df: pd.DataFrame, window: int = 14) -> float | None:
    """
    Compute ADX(14) directly here so we don't depend on the analysis_engine
    having been run on this dataframe. Same Wilder formulation.
    """
    if len(df) < window * 3:
        return None
    high, low, close = df['High'], df['Low'], df['Close']
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm  = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr      = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = dx.ewm(alpha=1/window, adjust=False).mean()
    val = adx.iloc[-1]
    return float(val) if not pd.isna(val) else None


def _distance_from_extreme_score(df: pd.DataFrame, lookback: int = 60, extreme_pct: float = 5.0) -> float | None:
    """
    1.0 when price sits in the middle of its 60d range, 0.0 when within
    extreme_pct of the 60d high or low.
    """
    if len(df) < lookback:
        return None
    window = df.tail(lookback)
    hi, lo, px = float(window['High'].max()), float(window['Low'].min()), float(df['Close'].iloc[-1])
    if hi <= lo:
        return None
    pct_to_hi = (hi - px) / px * 100   # how far below the high
    pct_to_lo = (px - lo) / px * 100   # how far above the low
    nearest = min(pct_to_hi, pct_to_lo)
    if nearest <= 0:
        return 0.0
    if nearest >= extreme_pct * 3:     # comfortably in the middle
        return 1.0
    # Linear ramp from 0 (at extreme_pct away) to 1 (at 3*extreme_pct away)
    return float(min(max((nearest - extreme_pct) / (2 * extreme_pct), 0.0), 1.0))


def _adx_band_score(adx: float | None, lo: float, hi: float) -> float | None:
    """Tent function: 1.0 inside [lo, hi], linear ramp down outside."""
    if adx is None:
        return None
    if lo <= adx <= hi:
        return 1.0
    # Ramp distance — use 10 ADX units of decay outside the band
    if adx < lo:
        return float(max(0.0, 1.0 - (lo - adx) / 10.0))
    return float(max(0.0, 1.0 - (adx - hi) / 10.0))


def _saturating_score(value: float | None, target: float) -> float | None:
    """Linear 0→1 from 0 to target, capped at 1.0."""
    if value is None:
        return None
    return float(min(max(value / target, 0.0), 1.0))


def _recent_limit_event(ts_code: str, lookback_days: int = 5) -> bool:
    """
    True if `ts_code` hit either 涨停 OR 跌停 in the last `lookback_days`
    trading days. Uses Tushare's stk_limit + daily endpoints; returns False
    on any API failure (gracefully degrades — don't punish a stock just
    because the API is flaky).
    """
    try:
        data_manager.init_tushare()
        if data_manager.TUSHARE_API is None:
            return False
        end   = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=lookback_days * 2 + 5)).strftime("%Y%m%d")
        limits = data_manager.TUSHARE_API.stk_limit(ts_code=ts_code, start_date=start, end_date=end)
        if limits is None or limits.empty:
            return False
        daily = data_manager.TUSHARE_API.daily(ts_code=ts_code, start_date=start, end_date=end)
        if daily is None or daily.empty:
            return False
        m = limits.merge(daily[["trade_date", "close"]], on="trade_date", how="inner") \
                  .sort_values("trade_date").tail(lookback_days)
        hit_up   = (m["close"] >= m["up_limit"]   - 1e-4)
        hit_down = (m["close"] <= m["down_limit"] + 1e-4)
        return bool(hit_up.any() or hit_down.any())
    except Exception:
        return False


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_prices(ticker: str) -> pd.DataFrame | None:
    """≈1y of qfq OHLCV — cached 15 min."""
    try:
        return data_manager.get_single_stock_data_live(ticker, lookback_years=1)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_turnover_20d_avg(ts_code: str) -> float | None:
    """
    Pull last ~30 calendar days of daily_basic for one ticker and return the
    20-trading-day mean turnover_rate (in %). Cached 30 min — turnover is a
    relatively stable metric so we don't need to re-pull every scan.
    """
    try:
        data_manager.init_tushare()
        if data_manager.TUSHARE_API is None:
            return None
        end   = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=45)).strftime("%Y%m%d")
        df = data_manager.TUSHARE_API.daily_basic(
            ts_code=ts_code, start_date=start, end_date=end,
            fields="ts_code,trade_date,turnover_rate",
        )
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["turnover_rate"]).sort_values("trade_date").tail(20)
        if df.empty:
            return None
        return float(df["turnover_rate"].mean())
    except Exception:
        return None


def _to_ts_code(ticker6: str) -> str:
    try:
        return data_manager.get_tushare_ticker(ticker6)
    except Exception:
        return f"{ticker6}.SH" if ticker6.startswith(("6", "9")) else f"{ticker6}.SZ"


# ── Scan ─────────────────────────────────────────────────────────────────────
scan_col, _ = st.columns([1, 4])
scan = scan_col.button("⚡ Scan watchlist now", type="primary", use_container_width=True)

if not scan:
    st.info(
        "Click **Scan watchlist now** to score every stock in your watchlist on "
        "the 5-component T-tradability score. Results cache for 15 minutes."
    )
    st.stop()

watchlist = data_manager.get_watchlist()
if not watchlist:
    st.warning("📭 Your watchlist is empty. Add stocks in the Watchlist page first.")
    st.stop()

st.markdown(f"**Scanning {len(watchlist)} stocks…**")
progress = st.progress(0.0, text="Starting scan…")

WEIGHTS = {"range": 0.30, "turnover": 0.25, "meanrev": 0.25, "adx": 0.10, "extreme": 0.10}

rows: list[dict] = []
for idx, item in enumerate(watchlist, start=1):
    ticker = item["ticker"]
    name   = item.get("stock_name") or data_manager.get_stock_name_from_db(ticker) or ticker
    progress.progress(idx / len(watchlist), text=f"{idx}/{len(watchlist)} · {ticker} {name}")

    df = _fetch_prices(ticker)
    if df is None or df.empty or len(df) < 60:
        rows.append({
            "_rank": -1, "Ticker": ticker, "Name": name,
            "T-Score": None, "Verdict": "No data",
            "Range %": None, "Turnover %": None, "MeanRev bias": None,
            "ADX": None, "Range pos": None, "Limit event?": "—", "Why": "Insufficient price history",
        })
        continue

    ts_code = _to_ts_code(ticker)

    # Component metrics
    intraday_pct = _intraday_range_pct(df, window=20)
    mr_bias      = _mean_reversion_bias(df, window=20)
    adx_val      = _rsi_independent_adx(df, window=14)
    range_pos    = _distance_from_extreme_score(df, lookback=60, extreme_pct=extreme_pct)
    turnover_pct = _fetch_turnover_20d_avg(ts_code)
    limit_event  = _recent_limit_event(ts_code, lookback_days=5)

    # Hard-fail gates
    fail_reason = None
    if limit_event:
        fail_reason = "跌停/涨停 in last 5 days"
    elif turnover_pct is not None and turnover_pct < 2.0:
        fail_reason = f"Illiquid (turnover {turnover_pct:.1f}% < 2%)"
    elif intraday_pct is not None and intraday_pct < 1.0:
        fail_reason = f"No range (intraday {intraday_pct:.2f}% < 1%)"

    if fail_reason:
        rows.append({
            "_rank": 4, "Ticker": ticker, "Name": name,
            "T-Score": 0, "Verdict": "⛔ Skip",
            "Range %": round(intraday_pct, 2) if intraday_pct is not None else None,
            "Turnover %": round(turnover_pct, 2) if turnover_pct is not None else None,
            "MeanRev bias": round(mr_bias, 3) if mr_bias is not None else None,
            "ADX": round(adx_val, 1) if adx_val is not None else None,
            "Range pos": round(range_pos, 2) if range_pos is not None else None,
            "Limit event?": "⚠️ Yes" if limit_event else "—",
            "Why": fail_reason,
        })
        continue

    # Component sub-scores (each [0, 1])
    s_range    = _saturating_score(intraday_pct, range_target)
    s_turnover = _saturating_score(turnover_pct, turnover_target)
    s_meanrev  = None if mr_bias is None else float(max(0.0, 1.0 - mr_bias))
    s_adx      = _adx_band_score(adx_val, adx_band_lo, adx_band_hi)
    s_extreme  = range_pos  # already normalised

    parts = {
        "range":    s_range,
        "turnover": s_turnover,
        "meanrev":  s_meanrev,
        "adx":      s_adx,
        "extreme":  s_extreme,
    }
    # If any component is None (missing data) → exclude its weight and renormalise
    used_weight = sum(WEIGHTS[k] for k, v in parts.items() if v is not None)
    score = 0.0
    if used_weight > 0:
        score = sum(WEIGHTS[k] * v for k, v in parts.items() if v is not None) / used_weight
    score_pct = round(score * 100, 1)

    if score_pct >= 75:
        verdict, rank = "🟢 STRONG", 0
    elif score_pct >= 55:
        verdict, rank = "🟡 OK", 1
    else:
        verdict, rank = "⚪ Not now", 2

    rows.append({
        "_rank":  rank,
        "Ticker": ticker,
        "Name":   name,
        "T-Score": score_pct,
        "Verdict": verdict,
        "Range %": round(intraday_pct, 2) if intraday_pct is not None else None,
        "Turnover %": round(turnover_pct, 2) if turnover_pct is not None else None,
        "MeanRev bias": round(mr_bias, 3) if mr_bias is not None else None,
        "ADX": round(adx_val, 1) if adx_val is not None else None,
        "Range pos": round(range_pos, 2) if range_pos is not None else None,
        "Limit event?": "—",
        "Why": "",
    })

progress.empty()

# ── Display ─────────────────────────────────────────────────────────────────
results = pd.DataFrame(rows).sort_values(["_rank", "T-Score"], ascending=[True, False], na_position="last")
n_strong = int((results["Verdict"] == "🟢 STRONG").sum())
n_ok     = int((results["Verdict"] == "🟡 OK").sum())
n_skip   = int((results["Verdict"] == "⛔ Skip").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("🟢 Strong",        n_strong)
m2.metric("🟡 OK",            n_ok)
m3.metric("⚪ Not now",       len(results) - n_strong - n_ok - n_skip)
m4.metric("⛔ Skipped",       n_skip)

if n_strong > 0:
    st.success(
        f"**{n_strong} STRONG T-trading candidate{'s' if n_strong != 1 else ''}.** "
        "Confirm you hold a base position (底仓), then trade around it within the day's range."
    )

display = results.drop(columns=["_rank"])
st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "T-Score":       st.column_config.NumberColumn(format="%.1f",
                            help="Composite 0–100 score across the 5 components"),
        "Range %":       st.column_config.NumberColumn(format="%.2f",
                            help="20-day avg intraday range (High−Low)/Open in %"),
        "Turnover %":    st.column_config.NumberColumn(format="%.2f",
                            help="20-day avg turnover rate from daily_basic"),
        "MeanRev bias":  st.column_config.NumberColumn(format="%.3f",
                            help="20-day avg |Close−Open|/(High−Low). Lower=better. 0=mean-reverting, 1=trending."),
        "ADX":           st.column_config.NumberColumn(format="%.1f",
                            help="ADX(14). Sweet spot 15–35."),
        "Range pos":     st.column_config.NumberColumn(format="%.2f",
                            help="Distance from 60-day extreme. 1=middle of range, 0=at extreme."),
        "Limit event?":  st.column_config.TextColumn(
                            help="⚠️ Yes = 跌停 or 涨停 hit in last 5 days → automatic skip"),
        "Why":           st.column_config.TextColumn(width="medium",
                            help="Hard-fail reason if Skipped"),
    },
)

st.caption(
    "Data cached for 15 minutes (prices) / 30 minutes (turnover) — re-click scan to refresh. "
    "T-Score weights default to: Range 30 % · Turnover 25 % · MeanRev 25 % · ADX 10 % · Range-pos 10 %."
)
