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

# Load any saved scan from the DB. If that comes back empty but we have an
# in-session result (e.g. the Supabase t_trading_scans table doesn't exist yet,
# so the save silently failed), fall back to the session copy so the user still
# sees the results they just waited for.
saved_df, saved_at = data_manager.load_t_trading_scan(USER_ID)
_persist_failed = False
if (saved_df is None or saved_df.empty) and "tt_results_df" in st.session_state:
    saved_df = st.session_state["tt_results_df"]
    saved_at = st.session_state.get("tt_results_at")
    _persist_failed = True

# Scan control row
scan_l, scan_r = st.columns([1, 4])
do_scan = scan_l.button(
    "⚡ Re-scan watchlist" if saved_df is not None else "⚡ Scan watchlist now",
    type="primary", use_container_width=True,
    help="Wipes the saved scan and re-runs against your current watchlist. "
         "Takes ~10–30 seconds depending on watchlist size.",
)

# Age / status badge
if saved_df is not None and not saved_df.empty and saved_at:
    age_str, severity, scanned_dt = _age_display(saved_at)
    if _persist_failed:
        scan_r.warning(
            "⚠️ Results are showing for this session but could **not be saved** to "
            "the database — they'll be lost on reload. Create the `t_trading_scans` "
            "table in Supabase (SQL in the page's docstring) to enable persistence."
        )
    elif severity == "stale":
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
    # Keep an in-session copy so the results render even if the DB save fails
    # (e.g. Supabase table not yet created). This is the source of truth for
    # the current session; the DB is for cross-session persistence + age.
    st.session_state["tt_results_df"] = pd.DataFrame(fresh)
    st.session_state["tt_results_at"] = datetime.now(timezone.utc).isoformat()
    data_manager.save_t_trading_scan(USER_ID, fresh)
    st.success(f"✅ Scanned {len(fresh)} stocks.")
    st.rerun()

# If still nothing, prompt
if saved_df is None or saved_df.empty:
    st.info("No scan results yet. Click **Scan watchlist now** to run your first scan.")
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

# ════════════════════════════════════════════════════════════════════════════
# 5) 做T SCENARIO LAB — dual-limit exact backtest + OHLC-constrained scenarios
# ════════════════════════════════════════════════════════════════════════════
# Why this backtest is legitimate on daily bars (unlike the sequential one we
# removed): the strategy places BOTH limit orders at the open, simultaneously —
# buy N at (open − x%) and sell N base shares at (open + y%). With both orders
# standing, the ORDER of fills doesn't change P&L; the only questions are
# "did the day touch each level?", and High/Low answer that EXACTLY. Every
# number in the core backtest is provable from OHLC — no path inference.
# What OHLC can't see is multi-oscillation (band crossed several times), so
# the core result is a CONSERVATIVE floor; the Monte Carlo layer estimates the
# multi-trip upside using synthetic paths CONSTRAINED to each day's real OHLC
# (start at Open, end at Close, touch the actual High and Low).

st.markdown("#### 5 · 做T Scenario Lab — dual-limit backtest")
st.markdown(
    "**Strategy tested:** at each open, place BOTH orders — buy `N` shares at "
    "`open − x%` AND sell `N` base shares at `open + y%`. Both fill → round-trip "
    "profit locked (order of fills irrelevant). One fills → the imbalance is "
    "marked at that day's close. P&L is **excess over pure buy-and-hold** of "
    "the base. T+1-legal by construction: the sell leg always comes from "
    "base shares held overnight."
)

_PEN = 0.0005  # fill requires 0.05% penetration past the level (queue realism)

lab_c1, lab_c2, lab_c3, lab_c4 = st.columns(4)
lab_shares = lab_c1.number_input("T shares per leg (N)", value=300, step=100,
                                 min_value=100, max_value=100000, key="tt_lab_shares",
                                 help="Shares per leg. Your base position must be ≥ N "
                                      "so the sell leg always comes from held shares.")
lab_buy_frac = lab_c2.number_input("Buy at ___ % of avg range below open",
                                   value=40, step=5, min_value=10, max_value=100,
                                   key="tt_lab_buy")
lab_sell_frac = lab_c3.number_input("Sell at ___ % of avg range above open",
                                    value=40, step=5, min_value=10, max_value=100,
                                    key="tt_lab_sell")
lab_lookback = lab_c4.number_input("Lookback days", value=120, step=20,
                                   min_value=30, max_value=250, key="tt_lab_lb")

_avg_rng = (plan_intraday or 0.0) / 100.0   # fraction of open
_b_off = (lab_buy_frac  / 100.0) * _avg_rng
_s_off = (lab_sell_frac / 100.0) * _avg_rng
st.caption(
    f"20d avg intraday range `{(plan_intraday or 0):.2f}%` → buy level = "
    f"`open − {_b_off*100:.2f}%`, sell level = `open + {_s_off*100:.2f}%`. "
    f"Fills require {_PEN*100:.2f}% penetration past the level."
)


def _dual_limit_backtest(df: pd.DataFrame, b_off: float, s_off: float,
                         n_shares: int, lookback: int) -> pd.DataFrame | None:
    """Exact dual-limit 做T backtest from OHLC. Returns per-day frame."""
    if df is None or df.empty or b_off <= 0 or s_off <= 0:
        return None
    d = df.tail(int(lookback))[['Open', 'High', 'Low', 'Close']].dropna().copy()
    if d.empty:
        return None
    d['buy_lv']  = d['Open'] * (1 - b_off)
    d['sell_lv'] = d['Open'] * (1 + s_off)
    d['buy_fill']  = d['Low']  <= d['buy_lv']  * (1 - _PEN)
    d['sell_fill'] = d['High'] >= d['sell_lv'] * (1 + _PEN)

    both      = d['buy_fill'] & d['sell_fill']
    buy_only  = d['buy_fill'] & ~d['sell_fill']
    sell_only = ~d['buy_fill'] & d['sell_fill']

    d['locked'] = 0.0
    d.loc[both, 'locked'] = (d['sell_lv'] - d['buy_lv']) * n_shares
    d['mtm'] = 0.0
    d.loc[buy_only,  'mtm'] = (d['Close'] - d['buy_lv'])  * n_shares
    d.loc[sell_only, 'mtm'] = (d['sell_lv'] - d['Close']) * n_shares
    d['pnl'] = d['locked'] + d['mtm']
    d['outcome'] = np.select([both, buy_only, sell_only],
                             ['both', 'buy_only', 'sell_only'], default='none')
    d['cum_pnl'] = d['pnl'].cumsum()
    return d


lab = _dual_limit_backtest(plan_df, _b_off, _s_off, int(lab_shares), int(lab_lookback))

if lab is None or lab.empty:
    st.warning("Not enough data (or zero avg range) to run the dual-limit backtest.")
else:
    n_days   = len(lab)
    n_both   = int((lab['outcome'] == 'both').sum())
    n_bonly  = int((lab['outcome'] == 'buy_only').sum())
    n_sonly  = int((lab['outcome'] == 'sell_only').sum())
    notional = float((lab['Open'] * lab_shares).mean())
    tot_pnl  = float(lab['pnl'].sum())
    tot_lock = float(lab['locked'].sum())
    tot_mtm  = float(lab['mtm'].sum())
    avg_bp   = (lab['pnl'] / (lab['Open'] * lab_shares)).mean() * 1e4  # bp/day

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Round-trip days", f"{n_both} / {n_days}",
              help="Days where BOTH levels were touched — profit locked, provable from High/Low.")
    k2.metric("One-sided days", f"{n_bonly}▼ / {n_sonly}▲",
              help="buy-only ▼ (holding extra into close) / sell-only ▲ (sold base, price ran).")
    k3.metric("Locked P&L", f"¥{tot_lock:+,.0f}")
    k4.metric("MTM drift P&L", f"¥{tot_mtm:+,.0f}",
              help="Mark-to-close of one-sided days. Negative = the unfilled leg cost you.")
    k5.metric("Total · per-day", f"¥{tot_pnl:+,.0f}",
              delta=f"{avg_bp:+.1f} bp/day on ¥{notional:,.0f}", delta_color="inverse")

    # Equity curve — A-share colours (red = profit)
    _lc = "#dc2626" if tot_pnl >= 0 else "#16a34a"
    fig_lab = go.Figure()
    fig_lab.add_trace(go.Scatter(
        x=lab.index.strftime('%Y-%m-%d'), y=lab['locked'].cumsum(),
        mode='lines', name='Locked only (floor)',
        line=dict(color='#94a3b8', width=1.5, dash='dot'),
        hovertemplate='%{x}<br>Locked cum: ¥%{y:,.0f}<extra></extra>'))
    fig_lab.add_trace(go.Scatter(
        x=lab.index.strftime('%Y-%m-%d'), y=lab['cum_pnl'],
        mode='lines', name='Total (locked + MTM)',
        line=dict(color=_lc, width=2.2),
        fill='tozeroy',
        fillcolor='rgba(220,38,38,0.08)' if tot_pnl >= 0 else 'rgba(22,163,74,0.08)',
        hovertemplate='%{x}<br>Total cum: ¥%{y:,.0f}<extra></extra>'))
    fig_lab.add_hline(y=0, line_color='#94a3b8', line_width=1, line_dash='dot')
    fig_lab.update_layout(
        title=f"Cumulative 做T excess P&L · {n_days} days · N={lab_shares}",
        height=330, template='plotly_white', yaxis_title='¥',
        margin=dict(t=45, l=50, r=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_lab, use_container_width=True)
    st.caption(
        "ⓘ Every number above is exact from daily OHLC — zero path assumptions. "
        "It is a **conservative floor**: days the band was crossed multiple times "
        "are credited only one round-trip (see Monte Carlo below for that upside)."
    )

    # ── Parameter heatmap ────────────────────────────────────────────────────
    with st.expander("🔥 Parameter heatmap — buy × sell offsets", expanded=False):
        fracs = list(range(20, 90, 10))  # % of avg range
        z, txt = [], []
        for bf in fracs:
            zrow, trow = [], []
            for sf in fracs:
                r = _dual_limit_backtest(plan_df, (bf/100)*_avg_rng,
                                         (sf/100)*_avg_rng, 100, int(lab_lookback))
                if r is None or r.empty:
                    zrow.append(np.nan); trow.append("")
                    continue
                bp = (r['pnl'] / (r['Open'] * 100)).mean() * 1e4
                fill = (r['outcome'] == 'both').mean() * 100
                zrow.append(bp); trow.append(f"{bp:+.0f}bp<br>{fill:.0f}%RT")
            z.append(zrow); txt.append(trow)
        _zmax = np.nanmax(np.abs(z)) or 1.0
        fig_hm = go.Figure(go.Heatmap(
            z=z, x=[f"sell {f}%" for f in fracs], y=[f"buy {f}%" for f in fracs],
            text=txt, texttemplate="%{text}", textfont=dict(size=10),
            colorscale=[[0, "#16a34a"], [0.5, "#f1f5f9"], [1, "#dc2626"]],
            zmid=0, zmin=-_zmax, zmax=_zmax,
            colorbar=dict(title="bp/day"),
            hovertemplate="buy %{y} · sell %{x}<br>%{z:+.1f} bp/day<extra></extra>"))
        fig_hm.update_layout(
            title="Avg daily excess P&L (bp of notional) · %RT = round-trip fill rate",
            height=430, template='plotly_white',
            xaxis_title="Sell offset (% of avg range)",
            yaxis_title="Buy offset (% of avg range)",
            margin=dict(t=50, l=60, r=30, b=50))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Offsets in % of the 20-day avg intraday range. Tight offsets fill "
                   "often but earn little per trip and suffer more one-sided drift; "
                   "wide offsets rarely fill. Red = positive (A-share convention).")

    # ── Monte Carlo multi-trip scenarios (OHLC-constrained paths) ───────────
    with st.expander("🎲 Monte Carlo — multi-trip upside (OHLC-constrained paths)",
                     expanded=False):
        st.markdown(
            "Paths are **not random prices** — each synthetic path is pinned to "
            "that day's real bar: starts at **Open**, ends at **Close**, touches "
            "the actual **High** and **Low**, never exceeds them. Only the "
            "*sequencing/oscillation between those anchors* is simulated, which is "
            "exactly the one thing daily data doesn't record. This estimates how "
            "many EXTRA round-trips (beyond the floor's one) the band typically "
            "allowed."
        )

        @st.cache_data(ttl=900, show_spinner=False)
        def _mc_multitrip(ticker, b_off, s_off, lookback, n_paths=200, n_steps=120):
            dfp = _fetch_prices(ticker)
            base = _dual_limit_backtest(dfp, b_off, s_off, 100, lookback)
            if base is None or base.empty:
                return None
            rng = np.random.default_rng(42)
            extra_by_day = []
            both_days = base[base['outcome'] == 'both']
            for dt, row in both_days.iterrows():
                O, H, L, C = row['Open'], row['High'], row['Low'], row['Close']
                blv, slv = row['buy_lv'], row['sell_lv']
                if H <= L:
                    extra_by_day.append(0.0); continue
                # Which extreme first: weight by close position (rally days more
                # likely dipped first). Waypoints: O → E1 → E2 → C.
                p_low_first = 0.6 if C >= O else 0.4
                lows_first = rng.random(n_paths) < p_low_first
                t1 = rng.integers(int(n_steps*0.15), int(n_steps*0.5),  n_paths)
                t2 = rng.integers(int(n_steps*0.5),  int(n_steps*0.85), n_paths)
                sigma = (H - L) / np.sqrt(n_steps) * 0.7
                trips = np.zeros(n_paths)
                for p in range(n_paths):
                    e1, e2 = (L, H) if lows_first[p] else (H, L)
                    ts = [0, t1[p], t2[p], n_steps - 1]
                    ws = [O, e1, e2, C]
                    path = np.empty(n_steps)
                    for s0 in range(3):  # 3 Brownian-bridge segments
                        a, b_ = ts[s0], ts[s0 + 1]
                        m = b_ - a
                        if m <= 0:
                            path[a] = ws[s0]; continue
                        steps_ = rng.normal(0, sigma, m)
                        steps_[0] = 0.0                             # pin start
                        noise = steps_.cumsum()
                        noise -= np.linspace(0, noise[-1], m)       # pin end
                        path[a:b_] = np.linspace(ws[s0], ws[s0+1], m) + noise
                    path[-1] = C
                    np.clip(path, L, H, out=path)
                    # count alternating touches of the two levels
                    state = np.zeros(n_steps, dtype=int)
                    state[path <= blv] = 1
                    state[path >= slv] = -1
                    seq = state[state != 0]
                    if seq.size:
                        alt = 1 + int((np.diff(seq) != 0).sum())
                        trips[p] = alt // 2
                extra_by_day.append(max(float(np.mean(trips)) - 1.0, 0.0))
            return {
                'n_both': len(both_days),
                'avg_extra': float(np.mean(extra_by_day)) if extra_by_day else 0.0,
                'spread_per_trip': float(((both_days['sell_lv'] - both_days['buy_lv'])).mean())
                                   if len(both_days) else 0.0,
            }

        if st.button("Run Monte Carlo (200 paths/day)", key="tt_lab_mc"):
            with st.spinner("Simulating OHLC-constrained paths…"):
                mc = _mc_multitrip(selected_ticker, _b_off, _s_off, int(lab_lookback))
            if not mc or mc['n_both'] == 0:
                st.info("No round-trip days in the window — nothing to simulate.")
            else:
                extra_pnl = mc['avg_extra'] * mc['spread_per_trip'] * lab_shares * mc['n_both']
                m1, m2, m3 = st.columns(3)
                m1.metric("Avg EXTRA trips on round-trip days", f"{mc['avg_extra']:.2f}",
                          help="Simulated mean of additional band crossings beyond the "
                               "one trip the exact floor already counts.")
                m2.metric("Est. extra P&L over window", f"¥{extra_pnl:+,.0f}")
                m3.metric("Optimistic total", f"¥{tot_pnl + extra_pnl:+,.0f}",
                          help="Floor (exact) + simulated multi-trip upside. Treat as "
                               "a ceiling estimate, not a promise.")
                st.caption(
                    "ⓘ Assumes both orders are re-armed immediately after each "
                    "round-trip (needs base ≥ N and cash for one extra leg). Dip-first "
                    "probability set to 60% on up-close days / 40% on down-close days."
                )

