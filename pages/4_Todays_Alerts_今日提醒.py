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
# Boolean column signals - ONLY THE ONES YOU WANT
BULLISH_SIGNALS = {
    'MACD_Bottoming': 'MACD Bottoming',
    'MACD_ClassicCrossover': 'MACD Positive Crossover',
    'RSI_Bottoming': 'RSI Bottoming',
    'DI_Screaming_Buy': 'DI Screaming Breakout 🚀'  # <-- Add this line!
}

BEARISH_SIGNALS = {
    'MACD_Peaking': 'MACD Peaking',
    'MACD_BearishCrossover': 'MACD Bearish Crossover',
    'RSI_Peaking': 'RSI Peaking'
}

# ==========================================
# ADX PATTERN SIGNALS (WITH PRICE CONTEXT)
# ==========================================

# # BULLISH: ADX patterns that signal BUY opportunities
# # Only reversal patterns after price decline
# ADX_BULLISH_PATTERNS = {
#     'Bottoming + Downtrend': 'ADX Bottoming (after decline)',
#     'Reversing Up + Downtrend': 'ADX Reversing Up (after decline)',
# }

# # BEARISH: ADX patterns that signal SELL alerts
# # Only exhaustion patterns after price rally
# ADX_BEARISH_PATTERNS = {
#     'Peaking + Uptrend': 'ADX Peaking (after rally)',
#     'Reversing Down + Uptrend': 'ADX Reversing Down (after rally)',
# }

def get_beijing_date():
    """Get current date in Beijing timezone"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(beijing_tz).date()


def init_signals_tables():
    """Initialize the signals cache tables"""
    data_manager.create_signals_tables()


def check_adx_signals(latest, price_trend):
    """
    Check ADX pattern and return appropriate signal based on price trend context.

    BULLISH (Downtrend + ANY turning point):
    - ADX Bottoming + Downtrend → "ADX End (Bottoming) after decline"
    - ADX Peaking + Downtrend → "ADX End (Peaking) after decline"
    - ADX Reversing Up + Downtrend → "ADX Reversing after decline"
    - ADX Reversing Down + Downtrend → "ADX Reversing after decline"

    BEARISH (Uptrend + ANY turning point):
    - ADX Bottoming + Uptrend → "ADX End (Bottoming) after rally"
    - ADX Peaking + Uptrend → "ADX End (Peaking) after rally"
    - ADX Reversing Up + Uptrend → "ADX Reversing after rally"
    - ADX Reversing Down + Uptrend → "ADX Reversing after rally"

    Returns: (signal_name, signal_type) or (None, None)
    """
    if 'ADX_Pattern' not in latest.index:
        return None, None

    adx_pattern = str(latest['ADX_Pattern'])

    # Define ADX turning points (only these count as signals)
    adx_extremes = ['Bottoming', 'Peaking']  # ADX at extremes
    adx_reversals = ['Reversing Up', 'Reversing Down']  # ADX direction changes

    # BULLISH: Downtrend + ANY ADX turning point
    if price_trend == 'downtrend':
        if adx_pattern in adx_extremes:
            return f"ADX End ({adx_pattern}) after decline", 'bullish'
        elif adx_pattern in adx_reversals:
            return "ADX Reversing after decline", 'bullish'

    # BEARISH: Uptrend + ANY ADX turning point
    elif price_trend == 'uptrend':
        if adx_pattern in adx_extremes:
            return f"ADX End ({adx_pattern}) after rally", 'bearish'
        elif adx_pattern in adx_reversals:
            return "ADX Reversing after rally", 'bearish'

    # Ignore all other patterns:
    # - Neutral trends (no signal)
    # - Non-turning-point patterns (Strong Trend, Losing Steam, Slowing Down, etc.)
    return None, None



# this is the old scan_all_stocks, that fetches from the database.
def scan_my_watchlist():
    """
    Scan YOUR custom watchlist for signals using LIVE Tushare data.
    No more sector map dependency!
    """
    import time
    start_time = time.time()
    
    if not MY_WATCHLIST:
        st.error("❌ Your watchlist is empty! Please add stocks to MY_WATCHLIST at the top of this file.")
        return None, 0
    
    results = []
    
    # Create progress bars
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_stocks = len(MY_WATCHLIST)
    status_text.text(f"📡 正在扫描您的观察列表 ({total_stocks} 只股票)...")
    
    for idx, ticker in enumerate(MY_WATCHLIST, 1):
        # Update progress
        progress = idx / total_stocks
        progress_bar.progress(progress)
        status_text.text(f"🔍 调取并分析 {idx}/{total_stocks}: {ticker} - {progress*100:.1f}%")
        
        try:
            # Fetch LIVE data from Tushare (qfq)
            stock_df = data_manager.get_single_stock_data_live(ticker, lookback_years=1)
            
            if stock_df is None or len(stock_df) < 100:
                st.warning(f"⚠️ {ticker}: 数据不足 (需要至少100天)")
                continue
            
            # Run technical analysis
            analysis_df = run_single_stock_analysis(stock_df)
            
            if analysis_df is None or analysis_df.empty:
                continue
            
            # Get latest row (today's signals)
            latest = analysis_df.iloc[-1]
            
            # Get stock name
            stock_name = data_manager.get_stock_name_from_db(ticker)
            if not stock_name:
                stock_name = ticker
            
            # Calculate 5-day EMA for trend
            ema_5d = analysis_df['Close'].ewm(span=5, adjust=False).mean()
            current_price = latest['Close']
            current_ema = ema_5d.iloc[-1]
            previous_ema = ema_5d.iloc[-2] if len(ema_5d) >= 2 else current_ema
            
            # Determine trend
            if current_price > current_ema and current_ema > previous_ema:
                price_trend = 'uptrend'
            elif current_price < current_ema and current_ema < previous_ema:
                price_trend = 'downtrend'
            else:
                price_trend = 'neutral'
            
            # --- CHECK BULLISH SIGNALS ---
            bullish_signals_found = []
            
            for signal_col, signal_name in BULLISH_SIGNALS.items():
                if signal_col in latest.index and latest[signal_col] == True:
                    bullish_signals_found.append(signal_name)
            
            # Check ADX signals (downtrend + turning point)
            adx_signal, adx_type = check_adx_signals(latest, price_trend)
            if adx_signal and adx_type == 'bullish':
                bullish_signals_found.append(adx_signal)
            
            if bullish_signals_found:
                results.append({
                    'Type': '🚀 Opportunity',
                    'Ticker': ticker,
                    'Name': stock_name,
                    'Signals': ', '.join(bullish_signals_found),
                    'Signal_Count': len(bullish_signals_found),
                    'Price': float(latest.get('Close', 0)),
                    'RSI': float(latest.get('RSI_14', 0)),
                    'ADX': float(latest.get('ADX', 0)),
                    'MACD': float(latest.get('MACD', 0)),
                    'Volume': float(latest.get('Volume', 0))
                })
            
            # --- CHECK BEARISH SIGNALS ---
            bearish_signals_found = []
            
            for signal_col, signal_name in BEARISH_SIGNALS.items():
                if signal_col in latest.index and latest[signal_col] == True:
                    bearish_signals_found.append(signal_name)
            
             # Check ADX signals (uptrend + turning point)
            if adx_signal and adx_type == 'bearish':
                bearish_signals_found.append(adx_signal)
            
            if bearish_signals_found:
                results.append({
                    'Type': '⚠️ Alert',
                    'Ticker': ticker,
                    'Name': stock_name,
                    'Signals': ', '.join(bearish_signals_found),
                    'Signal_Count': len(bearish_signals_found),
                    'Price': float(latest.get('Close', 0)),
                    'RSI': float(latest.get('RSI_14', 0)),
                    'ADX': float(latest.get('ADX', 0)),
                    'MACD': float(latest.get('MACD', 0)),
                    'Volume': float(latest.get('Volume', 0))
                })
                
        except Exception as e:
            st.warning(f"⚠️ {ticker} 分析失败: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Calculate scan duration
    scan_duration = time.time() - start_time
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(['Type', 'Signal_Count'], ascending=[True, False])
        return df, scan_duration
    else:
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
    filter_type = st.selectbox("Filter", ["All", "🚀 Opportunities Only", "⚠️ Alerts Only"])

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

# ✅ COMPUTE CONFLICT INDICATOR (for both cached and fresh data)
if 'Conflict' not in df.columns:
    opportunity_tickers = set(df[df['Type'] == '🚀 Opportunity']['Ticker'])
    alert_tickers = set(df[df['Type'] == '⚠️ Alert']['Ticker'])
    conflict_tickers = opportunity_tickers & alert_tickers
    df['Conflict'] = df['Ticker'].apply(lambda x: '⚠️' if x in conflict_tickers else '')



# Apply filters
filtered_df = df.copy()

if filter_type == "🚀 Opportunities Only":
    filtered_df = filtered_df[filtered_df['Type'] == '🚀 Opportunity']
elif filter_type == "⚠️ Alerts Only":
    filtered_df = filtered_df[filtered_df['Type'] == '⚠️ Alert']

filtered_df = filtered_df[filtered_df['Signal_Count'] >= min_signals]

# ==================== SUMMARY STATS ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    opportunities_count = len(df[df['Type'] == '🚀 Opportunity'])
    st.metric("🚀 Opportunities", opportunities_count)

with col2:
    alerts_count = len(df[df['Type'] == '⚠️ Alert'])
    st.metric("⚠️ Alerts", alerts_count)

with col3:
    multi_signal = len(df[df['Signal_Count'] >= 2])
    st.metric("🔥 Strong Signals (2+)", multi_signal)

with col4:
    total_signals = df['Signal_Count'].sum()
    st.metric("📊 Total Signals", int(total_signals))

st.markdown("---")

# ==================== DISPLAY TABLE ====================
if filtered_df.empty:
    st.warning(f"No results match your filters (Type: {filter_type}, Min Signals: {min_signals})")
else:
    st.subheader(f"Found {len(filtered_df)} stocks with signals")
    
    # Format the display dataframe
    display_df = filtered_df.copy()
    
    # Format numeric columns
    display_df['Price'] = display_df['Price'].apply(lambda x: f"¥{x:.2f}")
    display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}")
    display_df['ADX'] = display_df['ADX'].apply(lambda x: f"{x:.1f}")
    display_df['MACD'] = display_df['MACD'].apply(lambda x: f"{x:.4f}")
    display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
    
    # Reorder columns
    # display_df = display_df[[
    #     'Type', 'Ticker', 'Name', 'Signal_Count', 'Signals', 
    #     'Price', 'RSI', 'ADX', 'MACD', 'Volume'
    # ]]
    
    # # Rename columns for display
    # display_df.columns = [
    #     'Type', 'Code', 'Stock Name', '# Signals', 'Signal Details',
    #     'Price', 'RSI', 'ADX', 'MACD', 'Volume'
    # ]

    # Reorder columns for display
    display_df = display_df[['Conflict', 'Type', 'Ticker', 'Name', 'Signal_Count', 'Signals', 'Price', 'RSI', 'ADX', 'MACD', 'Volume']]
    
    # Rename columns for display
    display_df.columns = ['Conflict', 'Type', 'Code', 'Stock Name', '# Signals', 'Signal Details', 'Price', 'RSI', 'ADX', 'MACD', 'Volume']
    
    # Display with color coding
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            'Conflict': st.column_config.TextColumn(
                        'Conflict',
                        width='small',
                        help='Conflict indicator - stock has both bullish and bearish signals'
                    ),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Code": st.column_config.TextColumn("Code", width="small"),
            "Stock Name": st.column_config.TextColumn("Stock Name", width="medium"),
            "# Signals": st.column_config.NumberColumn("# Signals", width="small"),
            "Signal Details": st.column_config.TextColumn("Signal Details", width="large"),
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

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🚀 Bullish Signals (Opportunities)**")
    st.markdown("""
    - **MACD Bottoming** - MACD stopped falling, reversal detected
    - **MACD Positive Crossover** - MACD crossed above Signal line
    - **RSI Bottoming** - RSI in bottom 10%, oversold
    - **DI Screaming Breakout** - Explosive surge in institutional buying pressure

    **ADX Signals (Downtrend + Turning Point):**
    - **ADX End (Bottoming) after decline** - Low ADX turning after downtrend
    - **ADX End (Peaking) after decline** - High ADX peaking after downtrend
    - **ADX Reversing after decline** - ADX direction change after downtrend
    """)


with col2:
    st.markdown("**⚠️ Bearish Signals (Alerts)**")
    st.markdown("""
    - **MACD Peaking** - MACD stopped rising, exhaustion detected
    - **MACD Bearish Crossover** - MACD crossed below Signal line
    - **RSI Peaking** - RSI in top 10%, overbought

    **ADX Signals (Uptrend + Turning Point):**
    - **ADX End (Bottoming) after rally** - Low ADX turning after uptrend
    - **ADX End (Peaking) after rally** - High ADX peaking after uptrend
    - **ADX Reversing after rally** - ADX direction change after uptrend
    """)

st.markdown("---")
st.caption("💡 Tip: Results are cached daily in the main database. Click 'Force Rescan' if you want fresh data.")
