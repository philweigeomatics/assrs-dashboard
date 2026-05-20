"""
T-Trading Scanner + Per-Stock Plan  ·  做T候选 + 单股方案

Two-in-one page:
  1. SCANNER — ranks the current user's watchlist by structural suitability for
     intraday做T (5-component composite score, 0-100). Results are PERSISTED
     in the t_trading_scans table so reopens are instant, with an age badge
     and warning when the scan is > 1 day old.
  2. PER-STOCK PLAN — click any row in the scan results (or search for any
     A-share via the picker below the table) to see a concrete trade plan
     for that single stock: buy/sell zones, sizing, risk rules.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth_manager
import data_manager

auth_manager.require_login()
data_manager.ensure_t_trading_scan_table()

st.set_page_config(
    page_title="T-Trading Scanner | 做T候选",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ T-Trading Scanner · 做T候选")
st.caption(
    "Find stocks structurally suited to intraday做T around a base position, "
    "then drill into any one for a concrete trade plan."
)

USER_ID = auth_manager.get_current_user_id()

# ── Strategy explainer ───────────────────────────────────────────────────────
with st.expander("📖 What this scanner looks for · 评分组成", expanded=False):
    st.markdown("""
**Reminder — what "做T" means on A-shares**

Because A-shares are T+1, you can't intraday-flip *new* shares. So 做T means
trading around an **existing base position (底仓)**: you sell some on an
intraday rally and buy back on a dip (正T), or buy a dip first then sell into
a rally (倒T). Net position size unchanged at end of day, but realised P&L
captures the intraday swing.

---

**The composite T-score (0–100)**

| # | Component | Weight | What it measures |
|---|---|---|---|
| 1 | **Intraday Range** | 30 % | 20-day avg `(High − Low) / Open` — has to be big enough to clear costs. Target ≥ 4 %. |
| 2 | **Liquidity** | 25 % | 20-day avg `turnover_rate` — tight spreads + ability to fill at size. Target ≥ 8 %. |
| 3 | **Mean-Reversion Bias** | 25 % | 20-day avg `\|Close − Open\| / (High − Low)` — lower is better. Below 0.4 = Close lands in middle of day's range (oscillator). |
| 4 | **ADX Regime** | 10 % | ADX(14) in [15, 35] = 1.0. Below 15 → no movement; above 35 → strong trend (limit-day risk). |
| 5 | **Range Position** | 10 % | Distance from 60-day high/low. Middle of range = 1.0; within ±5 % of an extreme = 0. |

**Hard-fail gates** (auto-reject regardless of score):
- 20d avg turnover **< 2 %**
- 20d avg intraday range **< 1 %**
- Any 跌停 or 涨停 in the last 5 trading days

**Verdicts:** 🟢 STRONG ≥ 75 · 🟡 OK 55–74 · ⚪ Not now < 55 · ⛔ Skip (hard fail)
""")

# ── Parameters ───────────────────────────────────────────────────────────────
with st.expander("⚙️ Parameters (defaults match the spec)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    range_target    = c1.number_input("Range target (%)",    value=4.0,  step=0.5, min_value=1.0, max_value=10.0)
    turnover_target = c2.number_input("Turnover target (%)", value=8.0,  step=0.5, min_value=2.0, max_value=20.0)
    adx_band_lo     = c3.number_input("ADX band low",        value=15,   step=1,   min_value=5,  max_value=30)
    adx_band_hi     = c4.number_input("ADX band high",       value=35,   step=1,   min_value=20, max_value=60)
    extreme_pct     = c5.number_input("Extreme zone (%)",    value=5.0,  step=0.5, min_value=1.0, max_value=15.0)

# ════════════════════════════════════════════════════════════════════════════
# COMPUTATION HELPERS — shared between the scanner and per-stock plan
# ════════════════════════════════════════════════════════════════════════════

def _intraday_range_pct(df, window=20):
    if len(df) < window: return None
    r = ((df['High'] - df['Low']) / df['Open']).tail(window)
    return float(r.mean() * 100)

def _mean_reversion_bias(df, window=20):
    if len(df) < window: return None
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    bias = ((df['Close'] - df['Open']).abs() / rng).dropna().tail(window)
    return float(bias.mean()) if not bias.empty else None

def _adx_14(df, window=14):
    if len(df) < window * 3: return None
    high, low, close = df['High'], df['Low'], df['Close']
    up   = high.diff();   down = -low.diff()
    plus_dm  = ((up > down) & (up > 0)).astype(float) * up
    minus_dm = ((down > up) & (down > 0)).astype(float) * down
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr      = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    val = dx.ewm(alpha=1/window, adjust=False).mean().iloc[-1]
    return float(val) if not pd.isna(val) else None

def _distance_from_extreme(df, lookback=60, extreme_pct=5.0):
    if len(df) < lookback: return None
    w = df.tail(lookback)
    hi, lo, px = float(w['High'].max()), float(w['Low'].min()), float(df['Close'].iloc[-1])
    if hi <= lo: return None
    nearest = min((hi - px) / px * 100, (px - lo) / px * 100)
    if nearest <= 0: return 0.0
    if nearest >= extreme_pct * 3: return 1.0
    return float(min(max((nearest - extreme_pct) / (2 * extreme_pct), 0.0), 1.0))

def _adx_band_score(adx, lo, hi):
    if adx is None: return None
    if lo <= adx <= hi: return 1.0
    if adx < lo: return float(max(0.0, 1.0 - (lo - adx) / 10.0))
    return float(max(0.0, 1.0 - (adx - hi) / 10.0))

def _saturating(value, target):
    return None if value is None else float(min(max(value / target, 0.0), 1.0))

def _recent_limit_event(ts_code, lookback_days=5):
    try:
        data_manager.init_tushare()
        if data_manager.TUSHARE_API is None: return False
        end   = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=lookback_days * 2 + 5)).strftime("%Y%m%d")
        limits = data_manager.TUSHARE_API.stk_limit(ts_code=ts_code, start_date=start, end_date=end)
        if limits is None or limits.empty: return False
        daily = data_manager.TUSHARE_API.daily(ts_code=ts_code, start_date=start, end_date=end)
        if daily is None or daily.empty: return False
        m = limits.merge(daily[["trade_date", "close"]], on="trade_date", how="inner") \
                  .sort_values("trade_date").tail(lookback_days)
        return bool(((m["close"] >= m["up_limit"] - 1e-4) | (m["close"] <= m["down_limit"] + 1e-4)).any())
    except Exception:
        return False

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_prices(ticker):
    try:
        return data_manager.get_single_stock_data_live(ticker, lookback_years=1)
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_turnover_20d_avg(ts_code):
    try:
        data_manager.init_tushare()
        if data_manager.TUSHARE_API is None: return None
        end   = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=45)).strftime("%Y%m%d")
        df = data_manager.TUSHARE_API.daily_basic(
            ts_code=ts_code, start_date=start, end_date=end,
            fields="ts_code,trade_date,turnover_rate")
        if df is None or df.empty: return None
        df = df.dropna(subset=["turnover_rate"]).sort_values("trade_date").tail(20)
        return float(df["turnover_rate"].mean()) if not df.empty else None
    except Exception:
        return None

def _to_ts_code(ticker6):
    try:
        return data_manager.get_tushare_ticker(ticker6)
    except Exception:
        return f"{ticker6}.SH" if ticker6.startswith(("6", "9")) else f"{ticker6}.SZ"


def _backtest_t_trading(df, mode, buy_pct, sell_pct, cost_round_trip_pct, lookback_days):
    """
    Simulate the T-trading plan on historical OHLC.

    Limitation: we only have daily bars, so we INFER the intraday path:
      - Close >= Open  → path: Open → Low  → High → Close  (down-first day)
      - Close <  Open  → path: Open → High → Low  → Close  (up-first day)
    Limit orders fill at their TARGET price (not the day's extreme), which is
    the realistic outcome — buy at `buy_target` if `Low <= buy_target`,
    sell at `sell_target` if `High >= sell_target`.

    Three per-day outcomes:
      - SUCCESS  — both legs filled in the right order; realised P&L locked in.
      - STUCK    — first leg filled but second didn't; position force-closed at
                   Close (this is the realistic loss scenario — you wouldn't
                   carry the extra inventory overnight given T+1 lock).
      - NO FILL  — neither leg's limit was touched; no trade, no impact.
    """
    if df is None or df.empty:
        return None
    bt = df.tail(int(lookback_days))[['Open', 'High', 'Low', 'Close']].dropna().copy()
    if bt.empty:
        return None

    bt['buy_target']   = bt['Open'] * (1 + buy_pct / 100)
    bt['sell_target']  = bt['Open'] * (1 + sell_pct / 100)
    bt['down_first']   = bt['Close'] >= bt['Open']
    bt['buy_touched']  = bt['Low']  <= bt['buy_target']
    bt['sell_touched'] = bt['High'] >= bt['sell_target']

    bt['outcome'] = 'no fill'
    bt['pnl_pct'] = 0.0
    cost_frac = cost_round_trip_pct / 100  # express as fraction

    # Actual fill prices — for success they equal the targets; for stuck the
    # second leg is the force-flatten at Close, not the unfilled target.
    bt['buy_fill']  = np.nan
    bt['sell_fill'] = np.nan

    if mode == '正T':
        # Success: rally day where Low touched buy zone AND High touched sell zone
        ok = bt['down_first'] & bt['buy_touched'] & bt['sell_touched']
        bt.loc[ok, 'outcome']   = 'success'
        bt.loc[ok, 'pnl_pct']   = (sell_pct - buy_pct) / 100 - cost_frac
        bt.loc[ok, 'buy_fill']  = bt.loc[ok, 'buy_target']
        bt.loc[ok, 'sell_fill'] = bt.loc[ok, 'sell_target']

        # Stuck: bought at buy_target but never reached sell_target → flatten at Close
        stuck = (~ok) & bt['buy_touched']
        bt.loc[stuck, 'outcome']   = 'stuck'
        stuck_pnl = (bt['Close'] - bt['buy_target']) / bt['Open'] - cost_frac
        bt.loc[stuck, 'pnl_pct']   = stuck_pnl[stuck]
        bt.loc[stuck, 'buy_fill']  = bt.loc[stuck, 'buy_target']
        bt.loc[stuck, 'sell_fill'] = bt.loc[stuck, 'Close']
    else:  # 倒T
        ok = (~bt['down_first']) & bt['sell_touched'] & bt['buy_touched']
        bt.loc[ok, 'outcome']   = 'success'
        bt.loc[ok, 'pnl_pct']   = (sell_pct - buy_pct) / 100 - cost_frac
        bt.loc[ok, 'buy_fill']  = bt.loc[ok, 'buy_target']
        bt.loc[ok, 'sell_fill'] = bt.loc[ok, 'sell_target']

        # Stuck: sold at sell_target but never reached buy_target → buy back at Close
        stuck = (~ok) & bt['sell_touched']
        bt.loc[stuck, 'outcome']   = 'stuck'
        stuck_pnl = (bt['sell_target'] - bt['Close']) / bt['Open'] - cost_frac
        bt.loc[stuck, 'pnl_pct']   = stuck_pnl[stuck]
        bt.loc[stuck, 'sell_fill'] = bt.loc[stuck, 'sell_target']
        bt.loc[stuck, 'buy_fill']  = bt.loc[stuck, 'Close']

    bt['cum_pnl_pct'] = bt['pnl_pct'].cumsum() * 100
    return bt


def _scan_one(ticker, name):
    """Compute all metrics + final score for one ticker. Returns the row dict."""
    df = _fetch_prices(ticker)
    if df is None or df.empty or len(df) < 60:
        return {"_rank": -1, "Ticker": ticker, "Name": name, "T-Score": None,
                "Verdict": "No data", "Range %": None, "Turnover %": None,
                "MeanRev bias": None, "ADX": None, "Range pos": None,
                "Limit event?": "—", "Why": "Insufficient price history"}

    ts_code = _to_ts_code(ticker)
    intraday_pct = _intraday_range_pct(df, 20)
    mr_bias      = _mean_reversion_bias(df, 20)
    adx_val      = _adx_14(df, 14)
    range_pos    = _distance_from_extreme(df, 60, extreme_pct)
    turnover_pct = _fetch_turnover_20d_avg(ts_code)
    limit_event  = _recent_limit_event(ts_code, 5)

    fail_reason = None
    if limit_event:
        fail_reason = "跌停/涨停 in last 5 days"
    elif turnover_pct is not None and turnover_pct < 2.0:
        fail_reason = f"Illiquid (turnover {turnover_pct:.1f}% < 2%)"
    elif intraday_pct is not None and intraday_pct < 1.0:
        fail_reason = f"No range (intraday {intraday_pct:.2f}% < 1%)"

    if fail_reason:
        return {"_rank": 4, "Ticker": ticker, "Name": name, "T-Score": 0,
                "Verdict": "⛔ Skip",
                "Range %": round(intraday_pct, 2) if intraday_pct is not None else None,
                "Turnover %": round(turnover_pct, 2) if turnover_pct is not None else None,
                "MeanRev bias": round(mr_bias, 3) if mr_bias is not None else None,
                "ADX": round(adx_val, 1) if adx_val is not None else None,
                "Range pos": round(range_pos, 2) if range_pos is not None else None,
                "Limit event?": "⚠️ Yes" if limit_event else "—",
                "Why": fail_reason}

    parts = {
        "range":    _saturating(intraday_pct, range_target),
        "turnover": _saturating(turnover_pct, turnover_target),
        "meanrev":  None if mr_bias is None else float(max(0.0, 1.0 - mr_bias)),
        "adx":      _adx_band_score(adx_val, adx_band_lo, adx_band_hi),
        "extreme":  range_pos,
    }
    WEIGHTS = {"range": 0.30, "turnover": 0.25, "meanrev": 0.25, "adx": 0.10, "extreme": 0.10}
    used = sum(WEIGHTS[k] for k, v in parts.items() if v is not None)
    score_pct = round(
        (sum(WEIGHTS[k] * v for k, v in parts.items() if v is not None) / used) * 100, 1
    ) if used > 0 else 0.0

    if score_pct >= 75:
        verdict, rank = "🟢 STRONG", 0
    elif score_pct >= 55:
        verdict, rank = "🟡 OK", 1
    else:
        verdict, rank = "⚪ Not now", 2

    return {"_rank": rank, "Ticker": ticker, "Name": name, "T-Score": score_pct,
            "Verdict": verdict,
            "Range %": round(intraday_pct, 2) if intraday_pct is not None else None,
            "Turnover %": round(turnover_pct, 2) if turnover_pct is not None else None,
            "MeanRev bias": round(mr_bias, 3) if mr_bias is not None else None,
            "ADX": round(adx_val, 1) if adx_val is not None else None,
            "Range pos": round(range_pos, 2) if range_pos is not None else None,
            "Limit event?": "—", "Why": ""}


# ════════════════════════════════════════════════════════════════════════════
# SCAN — load from DB on entry; scan-button wipes & re-saves
# ════════════════════════════════════════════════════════════════════════════

def _age_display(iso):
    try:
        scanned = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if scanned.tzinfo is None:
            scanned = scanned.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - scanned
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)} min ago", "ok", scanned
        if hours < 24:
            return f"{int(hours)}h ago", "ok", scanned
        days = int(hours / 24)
        sev = "warn" if days <= 3 else "stale"
        return (f"1 day ago" if days == 1 else f"{days} days ago"), sev, scanned
    except Exception:
        return iso, "ok", None

# Load any saved scan
saved_df, saved_at = data_manager.load_t_trading_scan(USER_ID)

# Scan control row
scan_l, scan_r = st.columns([1, 4])
do_scan = scan_l.button(
    "⚡ Re-scan watchlist" if saved_df is not None else "⚡ Scan watchlist now",
    type="primary", use_container_width=True,
    help="Wipes the saved scan and re-runs against your current watchlist. "
         "Takes ~10–30 seconds depending on watchlist size.",
)

# Age / status badge
if saved_df is not None and saved_at:
    age_str, severity, scanned_dt = _age_display(saved_at)
    if severity == "stale":
        scan_r.error(f"🔴 Last scan **{age_str}** — values are likely stale, re-scan recommended.")
    elif severity == "warn":
        scan_r.warning(f"🟡 Last scan **{age_str}** — over 1 day old, consider re-scanning.")
    else:
        scan_r.caption(f"💾 Loaded {len(saved_df)} saved row{'s' if len(saved_df) != 1 else ''} from "
                       f"last scan ({age_str}).")

# Run scan (wipes existing first)
if do_scan:
    watchlist = data_manager.get_watchlist()
    if not watchlist:
        st.warning("📭 Your watchlist is empty. Add stocks in the Watchlist page first.")
        st.stop()
    st.markdown(f"**Scanning {len(watchlist)} stocks…**")
    prog = st.progress(0.0, text="Starting…")
    fresh = []
    for idx, item in enumerate(watchlist, start=1):
        ticker = item["ticker"]
        name   = item.get("stock_name") or data_manager.get_stock_name_from_db(ticker) or ticker
        prog.progress(idx / len(watchlist), text=f"{idx}/{len(watchlist)} · {ticker} {name}")
        fresh.append(_scan_one(ticker, name))
    prog.empty()
    data_manager.save_t_trading_scan(USER_ID, fresh)
    st.success(f"✅ Scanned {len(fresh)} stocks, saved to database.")
    st.rerun()

# If still nothing, prompt
if saved_df is None and not do_scan:
    st.info("No saved scan yet. Click **Scan watchlist now** to run your first scan. "
            "Results will be saved and reload instantly next time.")
    st.stop()

# ── Render saved (or just-completed) table with row-selection ────────────────
results = saved_df.copy()
# Ensure sort order even when loaded from DB
def _row_rank(v):
    if v == "🟢 STRONG":  return 0
    if v == "🟡 OK":       return 1
    if v == "⛔ Skip":     return 3
    return 2
results["_rank"] = results["Verdict"].map(_row_rank)
results = results.sort_values(["_rank", "T-Score"], ascending=[True, False], na_position="last")

n_strong = int((results["Verdict"] == "🟢 STRONG").sum())
n_ok     = int((results["Verdict"] == "🟡 OK").sum())
n_skip   = int((results["Verdict"] == "⛔ Skip").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("🟢 Strong",  n_strong)
m2.metric("🟡 OK",      n_ok)
m3.metric("⚪ Not now", len(results) - n_strong - n_ok - n_skip)
m4.metric("⛔ Skipped", n_skip)

display = results.drop(columns=["_rank"])
selection = st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "T-Score":       st.column_config.NumberColumn(format="%.1f"),
        "Range %":       st.column_config.NumberColumn(format="%.2f"),
        "Turnover %":    st.column_config.NumberColumn(format="%.2f"),
        "MeanRev bias":  st.column_config.NumberColumn(format="%.3f"),
        "ADX":           st.column_config.NumberColumn(format="%.1f"),
        "Range pos":     st.column_config.NumberColumn(format="%.2f"),
        "Why":           st.column_config.TextColumn(width="medium"),
    },
)

# Determine selected ticker from row click
selected_ticker = None
selected_name   = None
if selection and getattr(selection, "selection", None):
    rows = selection.selection.rows
    if rows:
        sel_row = display.iloc[rows[0]]
        selected_ticker = sel_row["Ticker"]
        selected_name   = sel_row.get("Name") or selected_ticker

# ── Manual ticker picker (works for any A-share, not just watchlist) ─────────
st.markdown("---")
st.subheader("📋 Single-Stock T-Trading Plan")
st.caption("Click a row above to load that stock's plan, OR pick any A-share below.")

@st.cache_data(ttl=3600, show_spinner=False)
def _all_stock_options():
    stocks = data_manager.get_all_stock_basic()
    return [""] + [f"{s['ticker']} · {s['name']}" for s in stocks]

picker_pick = st.selectbox(
    "Stock code or name 股票代码或名称",
    options=_all_stock_options(),
    key="tt_plan_picker",
    format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
)
if picker_pick:
    selected_ticker = picker_pick.split(" · ")[0].strip()
    try:
        selected_name = picker_pick.split(" · ")[1].strip()
    except Exception:
        selected_name = selected_ticker

if not selected_ticker:
    st.info("Click any row in the scan table above, or type a ticker / name to load a plan.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# PER-STOCK PLAN  (loaded for `selected_ticker`)
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"### {selected_ticker} · {selected_name}")

plan_df = _fetch_prices(selected_ticker)
if plan_df is None or plan_df.empty or len(plan_df) < 30:
    st.error("Insufficient price history to build a plan for this ticker.")
    st.stop()

# Re-compute the same metrics for the plan card (live, not from DB — so a
# manually-picked ticker not in the scan still works).
plan_intraday = _intraday_range_pct(plan_df, 20)
plan_bias     = _mean_reversion_bias(plan_df, 20)
plan_adx      = _adx_14(plan_df, 14)
plan_rangepos = _distance_from_extreme(plan_df, 60, extreme_pct)
plan_ts_code  = _to_ts_code(selected_ticker)
plan_turnover = _fetch_turnover_20d_avg(plan_ts_code)

# ── 1) Suitability cards ────────────────────────────────────────────────────
st.markdown("#### 1 · Suitability diagnostic")
c1, c2, c3, c4, c5 = st.columns(5)
def _card(col, label, val, fmt, target, interp):
    if val is None:
        col.metric(label, "—", help="Insufficient data")
    else:
        col.metric(label, fmt.format(val), help=interp)
        col.caption(f"Target ≥ {target}" if "≥" in str(target) or "target" in str(target).lower() else f"({target})")

_card(c1, "Intraday Range %", plan_intraday, "{:.2f}", f"≥ {range_target:.1f}%",
      "20-day average (High − Low) / Open. Bigger = more harvestable distance per T.")
_card(c2, "Turnover %",       plan_turnover, "{:.2f}", f"≥ {turnover_target:.1f}%",
      "20-day average turnover rate. Higher = tighter spreads + better fills.")
_card(c3, "MeanRev bias",     plan_bias,     "{:.3f}", "lower is better",
      "20-day avg |Close − Open|/(High − Low). 0 = oscillator. 1 = trending intraday.")
_card(c4, "ADX(14)",          plan_adx,      "{:.1f}", f"in [{adx_band_lo}, {adx_band_hi}]",
      "Lower = no movement. Higher = strong trend (risk of limit day).")
_card(c5, "Range position",   plan_rangepos, "{:.2f}", "≥ 0.5 ideal",
      "1.0 = middle of 60d range. 0 = at extreme.")

# ── 2) Historical intraday behavior ─────────────────────────────────────────
st.markdown("#### 2 · Historical intraday behaviour (last 20 trading days)")
tail = plan_df.tail(20)
upside_pct   = ((tail['High']  - tail['Open']) / tail['Open'] * 100).clip(lower=0)
downside_pct = ((tail['Open']  - tail['Low'])  / tail['Open'] * 100).clip(lower=0)
co_drift_pct = ((tail['Close'] - tail['Open']) / tail['Open'] * 100)
# Loose round-trip indicator: Low < Open < High AND Close within 0.3% of Open
roundtrip_mask = (tail['Low'] < tail['Open']) & (tail['Open'] < tail['High']) & \
                 ((tail['Close'] - tail['Open']).abs() / tail['Open'] < 0.003)
roundtrip_pct = float(roundtrip_mask.sum()) / len(tail) * 100

up_p25, up_p50, up_p75 = np.percentile(upside_pct,   [25, 50, 75])
dn_p25, dn_p50, dn_p75 = np.percentile(downside_pct, [25, 50, 75])
co_mean = float(co_drift_pct.mean())

stat_l, stat_r = st.columns(2)
stat_l.markdown(f"""
**Upside excursion** _(High − Open) / Open_

| Pctile | Move |
|---|---|
| 25th | +{up_p25:.2f} % |
| 50th (median) | **+{up_p50:.2f} %** |
| 75th | +{up_p75:.2f} % |
""")
stat_r.markdown(f"""
**Downside excursion** _(Open − Low) / Open_

| Pctile | Move |
|---|---|
| 25th | −{dn_p25:.2f} % |
| 50th (median) | **−{dn_p50:.2f} %** |
| 75th | −{dn_p75:.2f} % |
""")
st.markdown(
    f"**Open-to-close drift** (mean): `{co_mean:+.2f} %`  ·  "
    f"**Round-trip days** (Close ≈ Open within 0.3 %): `{roundtrip_pct:.0f} %` of last 20 days"
)

# ── 3) Recommended trade plan ───────────────────────────────────────────────
st.markdown("#### 3 · Recommended trade plan")
# Mode pick
if co_mean > 0.10:
    mode, mode_label = "正T", "正T (buy dip → sell rally)"
    mode_why = f"Open-to-close drift {co_mean:+.2f}% leans bullish, so the rally is more reliable than the dip."
elif co_mean < -0.10:
    mode, mode_label = "倒T", "倒T (sell rally → buy dip)"
    mode_why = f"Open-to-close drift {co_mean:+.2f}% leans bearish, so the rally is the first thing to fade."
else:
    mode, mode_label = "正T", "正T (default — neutral drift)"
    mode_why = f"Drift is ≈ flat ({co_mean:+.2f}%); either direction works."

# Sizing rule
if plan_intraday is None:
    size_pct = 30
elif plan_intraday < 2.0:
    size_pct = 20
elif plan_intraday < 4.0:
    size_pct = 30
else:
    size_pct = 40

sell_lo = up_p25
sell_hi = up_p75
buy_lo  = -dn_p75
buy_hi  = -dn_p25
hard_stop = max(2.5, max(up_p75, dn_p75) * 1.4)

st.info(f"**Mode:** {mode_label}  ·  {mode_why}")

plan_l, plan_r = st.columns([3, 2])
with plan_l:
    # Plotly horizontal-band visualization. A-share convention: red = up, green = down.
    fig = go.Figure()
    # Sell zone (red — bullish-side action: sell into rally)
    fig.add_shape(type="rect", x0=0, x1=1, y0=sell_lo, y1=sell_hi,
                  fillcolor="rgba(239, 68, 68, 0.22)", line=dict(width=0))
    fig.add_annotation(x=0.5, y=(sell_lo + sell_hi) / 2,
                       text=f"<b>SELL ZONE</b><br>+{sell_lo:.2f} %  to  +{sell_hi:.2f} %",
                       showarrow=False, font=dict(color="#b91c1c", size=14))
    # Buy zone (green — bearish-side action: buy the dip)
    fig.add_shape(type="rect", x0=0, x1=1, y0=buy_lo, y1=buy_hi,
                  fillcolor="rgba(34, 197, 94, 0.22)", line=dict(width=0))
    fig.add_annotation(x=0.5, y=(buy_lo + buy_hi) / 2,
                       text=f"<b>BUY ZONE</b><br>{buy_lo:.2f} %  to  {buy_hi:.2f} %",
                       showarrow=False, font=dict(color="#15803d", size=14))
    # Open line + hard stops
    fig.add_hline(y=0,           line=dict(color="#475569", width=2, dash="solid"),
                  annotation_text="OPEN (today)", annotation_position="right",
                  annotation_font=dict(size=11))
    fig.add_hline(y=hard_stop,   line=dict(color="#b91c1c", width=1, dash="dash"),
                  annotation_text=f"hard stop +{hard_stop:.1f} %", annotation_position="right",
                  annotation_font=dict(size=10, color="#b91c1c"))
    fig.add_hline(y=-hard_stop,  line=dict(color="#15803d", width=1, dash="dash"),
                  annotation_text=f"hard stop −{hard_stop:.1f} %", annotation_position="right",
                  annotation_font=dict(size=10, color="#15803d"))
    fig.update_layout(
        height=380, margin=dict(l=20, r=120, t=20, b=20),
        template="plotly_white", showlegend=False,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(title="% deviation from today's open",
                   range=[-hard_stop * 1.35, hard_stop * 1.35],
                   zerolinewidth=2, zerolinecolor="#475569"),
    )
    st.plotly_chart(fig, use_container_width=True)

with plan_r:
    st.markdown(f"""
**Trade plan (assuming base position = N shares):**

- **Trade fraction:** {size_pct} % of base
- **First leg:**
{"  Buy at " + f"`{buy_lo:.2f} %` to `{buy_hi:.2f} %`" if mode == "正T" else "  Sell at " + f"`+{sell_lo:.2f} %` to `+{sell_hi:.2f} %`"}
- **Second leg (round-trip):**
{"  Sell same quantity at " + f"`+{sell_lo:.2f} %` to `+{sell_hi:.2f} %`" if mode == "正T" else "  Buy same quantity back at " + f"`{buy_lo:.2f} %` to `{buy_hi:.2f} %`"}
- **End-of-day target:** original base size
- **Order type:** LIMIT only (no market orders)
- **Round-trip deadline:** 14:45 — don't risk holding extra into close
""")

# ── 4) Risk rules ───────────────────────────────────────────────────────────
st.markdown("#### 4 · Risk rules & invalidation triggers")
st.markdown(f"""
| Trigger | Action |
|---|---|
| Open gap > ±2 % vs prev close | Wait until 10:00, re-evaluate the band on the new effective open |
| Price moves > **{hard_stop:.1f} %** from open without retracing | Cancel today's T — outside historical envelope |
| Stock hits 跌停 or 涨停 today | Plan invalidated — position is frozen, hold base only |
| ADX(14) crosses above 40 | Strong trend regime — pause T strategy for this stock |
| Range position drops below 0.3 (near 60d extreme) | Breakout/breakdown regime — pause T |
| 龙虎榜 institutional buy/sell shows up tomorrow | Directional bias may have changed — re-scan |
""")

st.caption(
    "ⓘ Trade band derived from the 25th–75th percentile of the last 20 days' intraday "
    "excursions. This is a heuristic envelope — *not* a guarantee. The day's actual high/low "
    "can land anywhere; the percentiles just tell you the historically typical zone."
)

# ── 5) Backtest ─────────────────────────────────────────────────────────────
st.markdown("#### 5 · Backtest the plan on historical days")
st.caption(
    "Tests the trade plan against the last N days using OHLC-inferred intraday paths. "
    "**Limit orders fill at their TARGET price** (not the day's extreme — that would be unrealistic). "
    "On days where only one leg fills, the position is force-flattened at the Close to model "
    "the realistic 'don't carry overnight' rule."
)

# Common parameters (lookback + cost)
bt_lb_col, bt_cost_col = st.columns(2)
bt_lookback = bt_lb_col.number_input(
    "Lookback days",
    value=60, step=10, min_value=20, max_value=250,
    key="tt_bt_lookback",
)
bt_cost = bt_cost_col.number_input(
    "Round-trip cost (%)",
    value=0.15, step=0.01, min_value=0.0, max_value=1.0, format="%.2f",
    key="tt_bt_cost",
    help="Commission (~0.05%) + stamp duty on sell (~0.05%) + slippage (~0.05%). "
         "Default 0.15% is typical for A-share retail.",
)

# Zone-strategy selector
st.markdown("**Zone strategy**")
bt_mode_col, bt_param_col = st.columns([1, 2])
bt_zone_mode = bt_mode_col.radio(
    "Zone strategy",
    options=["Range-scaled (recommended)", "Fixed target"],
    index=0,
    key="tt_bt_zone_mode",
    label_visibility="collapsed",
    help=(
        "**Range-scaled**: zones are derived from THIS stock's actual historical "
        "intraday excursions. A high-vol stock gets wider zones; a quiet stock "
        "gets tighter zones — automatically.\n\n"
        "**Fixed target**: symmetric zones around open at a user-set net profit "
        "(e.g. 1 %). One-size-fits-all — usually too shallow for high-range "
        "stocks and too aggressive for quiet ones."
    ),
)

if bt_zone_mode.startswith("Range-scaled"):
    bt_pctile = bt_param_col.select_slider(
        "Aggressiveness",
        options=[
            "P25 — fills often, smaller wins",
            "P50 — balanced (median)",
            "P75 — rarer fills, bigger wins",
        ],
        value="P50 — balanced (median)",
        key="tt_bt_pctile",
        help="Percentile of the last 20 days' upside/downside excursions used "
             "as the zone width. P25 = 25th percentile (close to open); "
             "P75 = 75th percentile (further out).",
    )
    if "P25" in bt_pctile:
        bt_buy_pct, bt_sell_pct, _label = -float(dn_p25), +float(up_p25), "P25"
    elif "P75" in bt_pctile:
        bt_buy_pct, bt_sell_pct, _label = -float(dn_p75), +float(up_p75), "P75"
    else:
        bt_buy_pct, bt_sell_pct, _label = -float(dn_p50), +float(up_p50), "P50"
    _zone_explainer = (
        f"**Zones at {_label} of this stock's historical excursions** · "
        f"buy `{bt_buy_pct:.3f} %` · sell `{bt_sell_pct:+.3f} %`"
    )
else:  # Fixed target
    bt_target_profit = bt_param_col.number_input(
        "Target net profit per trade (%)",
        value=1.0, step=0.1, min_value=0.2, max_value=5.0,
        key="tt_bt_target",
        help="Symmetric zones around open. Same target on every stock.",
    )
    _half_span = (bt_target_profit + bt_cost) / 2
    bt_buy_pct, bt_sell_pct = -_half_span, +_half_span
    _zone_explainer = (
        f"**Fixed symmetric zones for {bt_target_profit:.2f} % net target** · "
        f"buy `{bt_buy_pct:.3f} %` · sell `{bt_sell_pct:+.3f} %`"
    )

_implied_gross = bt_sell_pct - bt_buy_pct
_implied_net   = _implied_gross - bt_cost
st.markdown(
    f"{_zone_explainer}  ·  gross `{_implied_gross:.2f} %`  →  after `{bt_cost:.2f} %` cost  →  "
    f"**net `{_implied_net:+.2f} %` per success**"
)
if _implied_net <= 0:
    st.error(
        f"❌ Implied net profit is `{_implied_net:.2f} %` — zones are too tight to "
        "clear costs. Pick a higher percentile (P50 / P75) or use a fixed target."
    )
elif _implied_net < 0.20:
    st.warning(
        f"⚠️ Net profit per trade is only `{_implied_net:.2f} %` — close to noise. "
        "Consider P50 / P75 for wider zones."
    )

bt_df = _backtest_t_trading(plan_df, mode, bt_buy_pct, bt_sell_pct, bt_cost, bt_lookback)

if bt_df is None or bt_df.empty:
    st.warning("Insufficient history for backtest.")
else:
    n_total_days = len(bt_df)
    n_traded     = int((bt_df['outcome'] != 'no fill').sum())
    n_success    = int((bt_df['outcome'] == 'success').sum())
    n_stuck      = int((bt_df['outcome'] == 'stuck').sum())
    n_noFill     = int((bt_df['outcome'] == 'no fill').sum())
    total_pnl    = float(bt_df['pnl_pct'].sum() * 100)
    success_only_pnl = float(bt_df.loc[bt_df['outcome'] == 'success', 'pnl_pct'].sum() * 100)
    stuck_only_pnl   = float(bt_df.loc[bt_df['outcome'] == 'stuck',   'pnl_pct'].sum() * 100)
    avg_pnl_per_trade = float(bt_df.loc[bt_df['outcome'] != 'no fill', 'pnl_pct'].mean() * 100) if n_traded else 0.0
    win_rate     = (n_success / n_traded * 100) if n_traded else 0.0
    cum          = bt_df['cum_pnl_pct']
    max_dd       = float((cum - cum.cummax()).min()) if not cum.empty else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Trade days",      f"{n_traded} / {n_total_days}",
              help="Days where at least one leg's limit was touched (vs. days with no fills)")
    k2.metric("Round-trip wins", f"{n_success}  ({win_rate:.0f}%)",
              help="Days where both buy AND sell legs filled in the correct order")
    k3.metric("Stuck days",      f"{n_stuck}",
              help="Days where one leg filled but the other didn't — force-closed at Close")
    k4.metric("Avg P&L / trade", f"{avg_pnl_per_trade:+.3f} %")
    k5.metric("Cumulative P&L",  f"{total_pnl:+.2f} %",
              delta=f"max DD {max_dd:+.2f} %", delta_color="inverse")

    # Cumulative P&L chart with per-trade markers.
    # A-share colours: red = wins / positive, green = losses / negative.
    line_color = "#dc2626" if total_pnl >= 0 else "#16a34a"
    fill_color = "rgba(220, 38, 38, 0.12)" if total_pnl >= 0 else "rgba(22, 163, 74, 0.12)"
    fig_bt = go.Figure()
    # Base cumulative-P&L area
    fig_bt.add_trace(go.Scatter(
        x=bt_df.index.strftime('%Y-%m-%d'),
        y=bt_df['cum_pnl_pct'],
        mode='lines',
        line=dict(color=line_color, width=2),
        name='Cumulative P&L %',
        fill='tozeroy', fillcolor=fill_color,
        hovertemplate='%{x}<br>Cumulative: %{y:.2f}%<extra></extra>',
    ))
    fig_bt.add_hline(y=0, line_color='#94a3b8', line_width=1, line_dash='dot')

    # Success markers — small red circles on the line (A-share red = win)
    success_only = bt_df[bt_df['outcome'] == 'success']
    if not success_only.empty:
        fig_bt.add_trace(go.Scatter(
            x=success_only.index.strftime('%Y-%m-%d'),
            y=success_only['cum_pnl_pct'],
            mode='markers',
            name='✓ Round-trip win',
            marker=dict(color='#dc2626', size=7, symbol='circle',
                        line=dict(color='#7f1d1d', width=1)),
            customdata=np.stack([
                success_only['buy_fill'].values,
                success_only['sell_fill'].values,
                (success_only['pnl_pct'].values * 100),
            ], axis=-1),
            hovertemplate=(
                '%{x}<br>'
                '<b>✓ Success</b><br>'
                'Bought  ¥%{customdata[0]:.2f}<br>'
                'Sold    ¥%{customdata[1]:.2f}<br>'
                'P&L     %{customdata[2]:+.3f}%<extra></extra>'
            ),
        ))

    # Stuck markers — bigger green triangles (A-share green = loss / warning)
    stuck_only = bt_df[bt_df['outcome'] == 'stuck']
    if not stuck_only.empty:
        # Hover annotates which leg was the force-flatten
        if mode == '正T':
            stuck_annot = ['sell force-closed at Close'] * len(stuck_only)
        else:
            stuck_annot = ['buy force-replaced at Close'] * len(stuck_only)
        fig_bt.add_trace(go.Scatter(
            x=stuck_only.index.strftime('%Y-%m-%d'),
            y=stuck_only['cum_pnl_pct'],
            mode='markers',
            name='🛑 Stuck (force-flatten)',
            marker=dict(color='#16a34a', size=11, symbol='triangle-down',
                        line=dict(color='black', width=1)),
            customdata=np.stack([
                stuck_only['buy_fill'].values,
                stuck_only['sell_fill'].values,
                (stuck_only['pnl_pct'].values * 100),
                stuck_annot,
            ], axis=-1),
            hovertemplate=(
                '%{x}<br>'
                '<b>🛑 Stuck — %{customdata[3]}</b><br>'
                'Bought  ¥%{customdata[0]:.2f}<br>'
                'Sold    ¥%{customdata[1]:.2f}<br>'
                'P&L     %{customdata[2]:+.3f}%<extra></extra>'
            ),
        ))

    fig_bt.update_layout(
        title=f"Cumulative P&L over last {n_total_days} trading days",
        height=360, template='plotly_white',
        xaxis=dict(tickangle=-45),
        yaxis_title='Cumulative %',
        margin=dict(t=50, l=50, r=30, b=50),
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Breakdown caption
    st.caption(
        f"**Breakdown:** {n_success} winning round-trips contributed `{success_only_pnl:+.2f} %`. "
        f"{n_stuck} stuck-day flattens contributed `{stuck_only_pnl:+.2f} %`. "
        f"{n_noFill} days had no fill (no impact). "
        f"Average per traded day: `{avg_pnl_per_trade:+.3f} %`."
    )

    # Trade-by-trade log — actual realised fill prices.
    with st.expander("📋 Trade log (last 30 trade days)", expanded=False):
        log = bt_df[bt_df['outcome'] != 'no fill'].copy()
        log['Date']  = log.index.strftime('%Y-%m-%d')
        log['P&L %'] = (log['pnl_pct'] * 100).round(3)
        log = log[['Date', 'outcome', 'Open', 'buy_fill', 'sell_fill', 'Close', 'P&L %']]
        log = log.rename(columns={
            'outcome':   'Outcome',
            'buy_fill':  'Buy fill ¥',
            'sell_fill': 'Sell fill ¥',
        })
        log = log.tail(30).iloc[::-1]  # most recent first
        st.dataframe(
            log,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Open":         st.column_config.NumberColumn(format="¥%.2f"),
                "Buy fill ¥":   st.column_config.NumberColumn(format="¥%.2f"),
                "Sell fill ¥":  st.column_config.NumberColumn(format="¥%.2f"),
                "Close":        st.column_config.NumberColumn(format="¥%.2f"),
                "P&L %":        st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        st.caption(
            "**Buy fill ¥** and **Sell fill ¥** are the *actual* execution prices. "
            "Hover the markers on the cumulative-P&L chart above to see which leg "
            "was force-flattened on stuck rows."
        )

    st.caption(
        "ⓘ The backtest uses an OHLC-inferred intraday path (Close ≥ Open → "
        "Low-then-High; Close < Open → High-then-Low). Real intraday paths zigzag — "
        "this approximation tends to be *optimistic* for the 'success' count and *honest* "
        "for the 'stuck' count, because if neither leg can fill under the simplification, "
        "neither would in reality."
    )
