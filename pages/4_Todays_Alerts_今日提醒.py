"""
Today's Opportunities & Alerts - 今日提醒
Scans all stocks and shows buy/sell signals in a clean table
扫描所有股票并显示买卖信号
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import data_manager

# Import from shared engine
from analysis_engine import run_single_stock_analysis

import auth_manager
auth_manager.require_login()

# Initialize tables
data_manager.create_watchlist_table()
MY_WATCHLIST = data_manager.get_watchlist_tickers()


if 'force_rescan' not in st.session_state:
    st.session_state.force_rescan = False

# ==================== SIGNAL CRITERIA ====================
# These mirror EXACTLY the discrete markers drawn on the Technical Analysis
# chart, so an alert means the same thing as the chart marker. The ADX-based
# buy/sell signals come via Entry_Candidate / Exit_Candidate, which the engine
# direction-gates by +DI vs −DI (same gating as the chart) — NOT the old
# 5-day-EMA price-trend gate, which produced contradictory calls.

# Bullish: (engine column or Entry_Candidate tier) → display label
_BULL_BOOL = {
    'Squeeze_Fired_Bullish':  'Bullish Squeeze Breakout 🚀',
    'Signal_Accumulation':    'Phase 1: Accumulation',
    'MACD_Bottoming':         'MACD Bottoming',
    'MACD_ClassicCrossover':  'MACD Bullish Crossover',
    'RSI_Bottoming':          'RSI Bottoming',
}
_BULL_ENTRY = {  # Entry_Candidate value → label
    'Strength Returning':     'Strength Returning (ADX)',
    'Trend Accelerating':     'Trend Accelerating (ADX)',
    'Screaming Buy':          'DI Screaming Buy 🚀',
}

# Bearish
_BEAR_BOOL = {
    'Squeeze_Fired_Bearish':  'Bearish Squeeze Drop 🩸',
    'Exit_MACD_Lead':         'MACD Exit Signal',
    'MACD_Peaking':           'MACD Peaking',
    'MACD_BearishCrossover':  'MACD Bearish Crossover',
    'RSI_Peaking':            'RSI Peaking',
}
_BEAR_EXIT = {  # Exit_Candidate value → label
    'Trend Topping':          'Trend Topping (ADX)',
    'Trend Collapsing':       'Trend Collapsing (ADX)',
    'Screaming Sell':         'DI Screaming Sell 🛑',
}


def get_beijing_date():
    """Get current date in Beijing timezone"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(beijing_tz).date()


def init_signals_tables():
    """Initialize the signals cache tables"""
    data_manager.create_signals_tables()


def _extract_signals(latest):
    """
    Return (bullish_labels, bearish_labels) for one stock's latest bar,
    using the exact same signal set the Technical Analysis chart plots.
    """
    bull, bear = [], []

    for col, label in _BULL_BOOL.items():
        if col in latest.index and bool(latest[col]):
            bull.append(label)
    for col, label in _BEAR_BOOL.items():
        if col in latest.index and bool(latest[col]):
            bear.append(label)

    # Entry/Exit candidates (ADX lifecycle + DI, direction-gated by +DI/−DI)
    ec = str(latest.get('Entry_Candidate', '') or '')
    if ec in _BULL_ENTRY:
        bull.append(_BULL_ENTRY[ec])
    xc = str(latest.get('Exit_Candidate', '') or '')
    if xc in _BEAR_EXIT:
        bear.append(_BEAR_EXIT[xc])

    return bull, bear


def scan_my_watchlist():
    """
    Scan the user's watchlist with LIVE Tushare data. Produces ONE ROW PER
    STOCK with its bullish + bearish signal sets and a net bias.
    """
    import time
    start_time = time.time()

    if not MY_WATCHLIST:
        st.error("❌ Your watchlist is empty! Add stocks on the Watchlist page.")
        return None, 0

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_stocks = len(MY_WATCHLIST)
    status_text.text(f"📡 正在扫描您的观察列表 ({total_stocks} 只股票)...")

    for idx, ticker in enumerate(MY_WATCHLIST, 1):
        progress = idx / total_stocks
        progress_bar.progress(progress)
        status_text.text(f"🔍 调取并分析 {idx}/{total_stocks}: {ticker} - {progress*100:.1f}%")

        try:
            stock_df = data_manager.get_single_stock_data_live(ticker, lookback_years=1)
            if stock_df is None or len(stock_df) < 100:
                st.warning(f"⚠️ {ticker}: 数据不足 (需要至少100天)")
                continue

            analysis_df = run_single_stock_analysis(stock_df)
            if analysis_df is None or analysis_df.empty:
                continue

            latest = analysis_df.iloc[-1]
            stock_name = data_manager.get_stock_name_from_db(ticker) or ticker

            bull, bear = _extract_signals(latest)
            if not bull and not bear:
                continue   # no signals → not in the table at all

            # Net bias for the stock
            if bull and not bear:
                bias = '🚀 Bullish'
            elif bear and not bull:
                bias = '⚠️ Bearish'
            else:
                bias = '⚖️ Mixed'

            # Combined directional signal string (▲ bull / ▼ bear)
            parts = [f"▲ {s}" for s in bull] + [f"▼ {s}" for s in bear]
            signals_str = "  ·  ".join(parts)

            # NOTE: only the 10 cache-schema columns are stored. Bull/bear
            # counts are derived in the display from the ▲/▼ markers in
            # `Signals`, so they survive a cache round-trip without a schema
            # change.
            results.append({
                'Type':         bias,                 # cache 'type' column = bias
                'Ticker':       ticker,
                'Name':         stock_name,
                'Signals':      signals_str,
                'Signal_Count': len(bull) + len(bear),
                'Price':        float(latest.get('Close', 0)),
                'RSI':          float(latest.get('RSI_14', 0)),
                'ADX':          float(latest.get('ADX', 0)),
                'MACD':         float(latest.get('MACD', 0)),
                'Volume':       float(latest.get('Volume', 0)),
            })

        except Exception as e:
            st.warning(f"⚠️ {ticker} 分析失败: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()
    scan_duration = time.time() - start_time

    if results:
        df = pd.DataFrame(results)
        # Bullish first, then by signal count
        bias_rank = {'🚀 Bullish': 0, '⚖️ Mixed': 1, '⚠️ Bearish': 2}
        df['_rank'] = df['Type'].map(bias_rank).fillna(3)
        df = df.sort_values(['_rank', 'Signal_Count'], ascending=[True, False]).drop(columns=['_rank'])
        return df, scan_duration
    return pd.DataFrame(), scan_duration


# ==================== MAIN PAGE ====================
st.set_page_config(page_title="Today's Alerts | 今日提醒", page_icon="🎯", layout="wide")

st.title("🎯 Today's Opportunities & Alerts | 今日提醒")

# Initialize tables in existing database
# init_signals_tables()

# Get today's date in Beijing time
today_beijing = get_beijing_date()
today_str = today_beijing.strftime('%Y-%m-%d')

st.markdown(f"**Beijing Date:** {today_str} {today_beijing.strftime('%A')}")

# Check if we have cached data for today
# When checking cache
cached_df = data_manager.get_cached_signals(today_str)
metadata = data_manager.get_scan_metadata(today_str)

if st.session_state.force_rescan:
    cached_df = None  # Ignore cache if force rescan flag is set
    metadata = None

# Cache version guard: old snapshots are one-row-PER-ALERT (Type values
# '🚀 Opportunity' / '⚠️ Alert'). The new layout is one-row-PER-STOCK with a
# Bias Type ('🚀 Bullish' / '⚠️ Bearish' / '⚖️ Mixed'). If we loaded an old
# snapshot, discard it so the page re-scans into the new schema.
if cached_df is not None and 'Type' in cached_df.columns:
    _old_labels = {'🚀 Opportunity', '⚠️ Alert'}
    if cached_df['Type'].astype(str).isin(_old_labels).any():
        cached_df = None
        metadata = None


# Show cache status
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    if cached_df is not None:
        st.success("✅ Using cached data")
    else:
        st.info("🔄 Need to scan")

with col2:
    if st.button("🔄 Force Rescan", type="secondary"):
        st.session_state.force_rescan = True
        st.rerun()

with col3:
    filter_type = st.selectbox(
        "Filter", ["All", "🚀 Bullish", "⚠️ Bearish", "⚖️ Mixed"])

with col4:
    min_signals = st.selectbox("Min Signals", [1, 2, 3, 4], index=0)

# Show scan metadata if available
if metadata:
    with st.expander("📊 Scan Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stocks Scanned", metadata['total_stocks_scanned'])
        with col2:
            st.metric("Scan Duration", f"{metadata['scan_duration_seconds']:.1f}s")
        with col3:
            st.metric("Opportunities", metadata['opportunities_found'])
        with col4:
            st.metric("Alerts", metadata['alerts_found'])
        st.caption(f"Last scanned: {metadata['created_at']}")

st.markdown("---")

# ── Decide: use cache or scan ─────────────────────────────────────
# Load or scan data
if cached_df is not None:
    # Use cached data
    df = cached_df
    st.info(f"📦 Showing today's snapshot ({today_str})")
else:
    # Need to scan
    st.info(f"🎯 Scanning your watchlist with ({len(MY_WATCHLIST)} stocks. )")
    with st.spinner("🔍 Scanning all stocks for signals... This may take a few minutes."):
        df, scan_duration = scan_my_watchlist()
        
    if df is None:
        st.error("❌ Scanning failed! ")
        st.stop()
    
    # Save as today's snapshot for this user
    if not df.empty:
        save_success = data_manager.save_signals_to_cache(df, today_str, scan_duration)
        if save_success:
            st.success(f"✅ Scan complete in {scan_duration:.1f}s — snapshot saved.")
        else:
            st.warning("⚠️ Scan complete but failed to save snapshot.")
    else:
        st.success("✨ No signals detected in your watchlist today.")
        st.info("💡 Market may be consolidating, or no strong trends detected.")
        st.stop()
        
    # Reset flag after successful scan
    st.session_state.force_rescan = False

if df.empty:
    st.success("✨ No signals detected today. Market is quiet!")
    st.info("💡 This could mean:\n- All stocks are in neutral zones\n- No strong trends detected\n- Market is consolidating")
    st.stop()

# Derive bull/bear counts from the ▲/▼ markers in the Signals string so the
# breakdown survives a cache round-trip (the DB stores only the 10-col schema).
def _count_dir(sig, mark):
    return str(sig).count(mark)
df['Bull'] = df['Signals'].apply(lambda s: _count_dir(s, '▲'))
df['Bear'] = df['Signals'].apply(lambda s: _count_dir(s, '▼'))

# Apply filters
filtered_df = df.copy()
if filter_type == "🚀 Bullish":
    filtered_df = filtered_df[filtered_df['Type'] == '🚀 Bullish']
elif filter_type == "⚠️ Bearish":
    filtered_df = filtered_df[filtered_df['Type'] == '⚠️ Bearish']
elif filter_type == "⚖️ Mixed":
    filtered_df = filtered_df[filtered_df['Type'] == '⚖️ Mixed']

filtered_df = filtered_df[filtered_df['Signal_Count'] >= min_signals]

# ==================== SUMMARY STATS ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🚀 Bullish", int((df['Type'] == '🚀 Bullish').sum()))
with col2:
    st.metric("⚠️ Bearish", int((df['Type'] == '⚠️ Bearish').sum()))
with col3:
    st.metric("⚖️ Mixed", int((df['Type'] == '⚖️ Mixed').sum()))
with col4:
    st.metric("📊 Stocks with signals", len(df))

st.markdown("---")

# ==================== DISPLAY TABLE ====================
if filtered_df.empty:
    st.warning(f"No results match your filters (Filter: {filter_type}, Min Signals: {min_signals})")
else:
    st.subheader(f"Found {len(filtered_df)} stocks with signals")

    display_df = filtered_df.copy()
    display_df['Price']  = display_df['Price'].apply(lambda x: f"¥{x:.2f}")
    display_df['RSI']    = display_df['RSI'].apply(lambda x: f"{x:.1f}")
    display_df['ADX']    = display_df['ADX'].apply(lambda x: f"{x:.1f}")
    display_df['MACD']   = display_df['MACD'].apply(lambda x: f"{x:.4f}")
    display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")

    # Code → deep link into Technical Analysis (?ticker= seeds active_ticker).
    display_df['Ticker'] = display_df['Ticker'].apply(
        lambda t: f"/single-stock-analysis?ticker={t}")

    # One row per stock: Bias · Code · Name · ▲ · ▼ · Signals · indicators
    display_df = display_df[['Type', 'Ticker', 'Name', 'Bull', 'Bear',
                             'Signals', 'Price', 'RSI', 'ADX', 'MACD', 'Volume']]
    display_df.columns = ['Bias', 'Code', 'Stock Name', '▲', '▼',
                          'Signal Details', 'Price', 'RSI', 'ADX', 'MACD', 'Volume']

    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            "Bias": st.column_config.TextColumn(
                "Bias", width="small",
                help="Net bias: 🚀 Bullish (only bull signals) · ⚠️ Bearish (only bear) · ⚖️ Mixed (both)"),
            "Code": st.column_config.LinkColumn(
                "Code", width="small",
                display_text=r"ticker=(.+)$",
                help="Click to open this stock in Technical Analysis 技术分析"),
            "Stock Name": st.column_config.TextColumn("Stock Name", width="medium"),
            "▲": st.column_config.NumberColumn("▲", width="small", help="Bullish signal count"),
            "▼": st.column_config.NumberColumn("▼", width="small", help="Bearish signal count"),
            "Signal Details": st.column_config.TextColumn(
                "Signal Details", width="large",
                help="▲ = bullish marker, ▼ = bearish marker — same set as the Technical Analysis chart"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "RSI": st.column_config.TextColumn("RSI", width="small"),
            "ADX": st.column_config.TextColumn("ADX", width="small"),
            "MACD": st.column_config.TextColumn("MACD", width="small"),
            "Volume": st.column_config.TextColumn("Volume", width="small"),
        }
    )
    
    # ==================== DOWNLOAD BUTTON ====================
    st.markdown("---")
    
    csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"signals_{today_str}.csv",
        mime="text/csv"
    )

# ==================== LEGEND ====================
st.markdown("---")
st.markdown("### 📖 Signal Definitions")
st.caption(
    "These are the **exact same discrete signals plotted on the Technical "
    "Analysis chart** — an alert here means the same marker appears there. "
    "ADX-based buy/sell signals are direction-gated by +DI vs −DI (the chart's "
    "Entry/Exit candidates), not a separate price-trend rule."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**▲ Bullish markers**")
    st.markdown("""
    - **Strength Returning (ADX)** — ADX Bottoming / Reversing Up while +DI dominant
    - **Trend Accelerating (ADX)** — ADX Accelerating Up while +DI dominant
    - **DI Screaming Buy 🚀** — fresh +DI bullish cross with momentum blow-out
    - **Bullish Squeeze Breakout 🚀** — Bollinger squeeze fired upward
    - **Phase 1: Accumulation** — OBV-divergence accumulation phase
    - **MACD Bottoming** — MACD stopped falling, turning up
    - **MACD Bullish Crossover** — MACD crossed above its signal line
    - **RSI Bottoming** — RSI in the bottom decile, turning up
    """)

with col2:
    st.markdown("**▼ Bearish markers**")
    st.markdown("""
    - **Trend Topping (ADX)** — ADX Peaking / Reversing Down while +DI dominant
    - **Trend Collapsing (ADX)** — ADX Accelerating Down while +DI dominant
    - **DI Screaming Sell 🛑** — fresh −DI bearish cross with momentum blow-out
    - **Bearish Squeeze Drop 🩸** — Bollinger squeeze fired downward
    - **MACD Exit Signal** — MACD-lead exit (bearish cross / MA cross-down)
    - **MACD Peaking** — MACD stopped rising, turning down
    - **MACD Bearish Crossover** — MACD crossed below its signal line
    - **RSI Peaking** — RSI in the top decile, turning down
    """)

st.markdown("---")
st.caption(
    "💡 One row per stock · **Bias** = 🚀 Bullish (only ▲) / ⚠️ Bearish (only ▼) / "
    "⚖️ Mixed (both). Results are cached daily; click 'Force Rescan' for fresh data."
)
