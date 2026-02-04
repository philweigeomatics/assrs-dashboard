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
import sqlite3
import data_manager

# Import from shared engine
from analysis_engine import run_single_stock_analysis

if 'force_rescan' not in st.session_state:
    st.session_state.force_rescan = False

# ==================== SIGNAL CRITERIA ====================
# Boolean column signals - ONLY THE ONES YOU WANT
BULLISH_SIGNALS = {
    'MACD_Bottoming': 'MACD Bottoming',
    'MACD_ClassicCrossover': 'MACD Positive Crossover',
    'RSI_Bottoming': 'RSI Bottoming'
}

BEARISH_SIGNALS = {
    'MACD_Peaking': 'MACD Peaking',
    'MACD_BearishCrossover': 'MACD Bearish Crossover',
    'RSI_Peaking': 'RSI Peaking'
}

# ==========================================
# ADX PATTERN SIGNALS (WITH PRICE CONTEXT)
# ==========================================

# BULLISH: ADX patterns that signal BUY opportunities
# Only reversal patterns after price decline
ADX_BULLISH_PATTERNS = {
    'Bottoming + Downtrend': 'ADX Bottoming (after decline)',
    'Reversing Up + Downtrend': 'ADX Reversing Up (after decline)',
}

# BEARISH: ADX patterns that signal SELL alerts
# Only exhaustion patterns after price rally
ADX_BEARISH_PATTERNS = {
    'Peaking + Uptrend': 'ADX Peaking (after rally)',
    'Reversing Down + Uptrend': 'ADX Reversing Down (after rally)',
}

def get_beijing_date():
    """Get current date in Beijing timezone"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(beijing_tz).date()


def init_signals_tables():
    """Initialize the signals cache tables"""
    data_manager.create_signals_tables()



def scan_all_stocks():
    """
    Scan all stocks in the database for signals.
    Returns: combined_df with opportunities and alerts
    """
    import time
    start_time = time.time()
    
    # Load all stock data
    all_stock_data = data_manager.get_all_stock_data_from_db()
    
    if not all_stock_data:
        return None, 0
    
    results = []
    total_stocks = len(all_stock_data)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (ticker, stock_df) in enumerate(all_stock_data.items(), 1):
        # Update progress
        progress = idx / total_stocks
        progress_bar.progress(progress)
        status_text.text(f"Scanning: {idx}/{total_stocks} stocks ({progress*100:.1f}%)")
        
        try:
            # Skip if not enough data
            if stock_df is None or len(stock_df) < 100:
                continue
            
            # Run technical analysis
            analysis_df = run_single_stock_analysis(stock_df)
            
            if analysis_df is None or analysis_df.empty:
                continue
            
            # Get latest row (today's signals)
            latest = analysis_df.iloc[-1]
            
            # Get stock name from database
            stock_name = data_manager.get_stock_name_from_db(ticker)
            if not stock_name:
                stock_name = ticker

            # ==========================================
            # CALCULATE PRICE TREND (using EMA)
            # ==========================================
            
            # Calculate 5-day EMA of closing prices
            ema_5d = analysis_df['Close'].ewm(span=5, adjust=False).mean()
            
            # Get current price and current EMA
            current_price = latest['Close']
            current_ema = ema_5d.iloc[-1]
            
            # Determine trend: price above EMA = uptrend, below = downtrend
            # Use 2% threshold to avoid noise
            if current_price > current_ema * 1.02:
                price_trend = 'uptrend'
            elif current_price < current_ema * 0.98:
                price_trend = 'downtrend'
            else:
                price_trend = 'neutral'
            
            # ==================== CHECK BULLISH SIGNALS ====================
            bullish_signals_found = []
            
            # Check boolean columns
            for signal_col, signal_name in BULLISH_SIGNALS.items():
                if signal_col in latest.index and latest[signal_col] == True:
                    bullish_signals_found.append(signal_name)
            
            # Check ADX Pattern with PRICE CONTEXT
            if 'ADX_Pattern' in latest.index:
                adx_pattern = str(latest['ADX_Pattern'])
                
                # Bottoming: Only bullish if price is in downtrend (reversal setup)
                if adx_pattern == 'Bottoming' and price_trend == 'downtrend':
                    bullish_signals_found.append(ADX_BULLISH_PATTERNS['Bottoming + Downtrend'])
                
                # Reversing Up: Only bullish if price is in downtrend (catching reversal)
                elif adx_pattern == 'Reversing Up' and price_trend == 'downtrend':
                    bullish_signals_found.append(ADX_BULLISH_PATTERNS['Reversing Up + Downtrend'])
            
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
            
            # ==================== CHECK BEARISH SIGNALS ====================
            bearish_signals_found = []
            
            # Check boolean columns
            for signal_col, signal_name in BEARISH_SIGNALS.items():
                if signal_col in latest.index and latest[signal_col] == True:
                    bearish_signals_found.append(signal_name)
            
            # Check ADX Pattern with PRICE CONTEXT
            if 'ADX_Pattern' in latest.index:
                adx_pattern = str(latest['ADX_Pattern'])
                
                # Peaking: Only bearish if price is in uptrend (exhaustion at top)
                if adx_pattern == 'Peaking' and price_trend == 'uptrend':
                    bearish_signals_found.append(ADX_BEARISH_PATTERNS['Peaking + Uptrend'])
                
                # Reversing Down: Only bearish if price is in uptrend (trend breaking)
                elif adx_pattern == 'Reversing Down' and price_trend == 'uptrend':
                    bearish_signals_found.append(ADX_BEARISH_PATTERNS['Reversing Down + Uptrend'])
            
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
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Calculate scan duration
    scan_duration = time.time() - start_time
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        # Sort by type (Opportunities first), then by signal count
        df = df.sort_values(['Type', 'Signal_Count'], ascending=[True, False])
        return df, scan_duration
    else:
        return pd.DataFrame(), scan_duration


# ==================== MAIN PAGE ====================
st.set_page_config(page_title="Today's Alerts | 今日提醒", page_icon="🎯", layout="wide")

st.title("🎯 Today's Opportunities & Alerts | 今日提醒")

# Initialize tables in existing database
init_signals_tables()

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

# Load or scan data
if cached_df is not None:
    # Use cached data
    df = cached_df
    st.info(f"📦 Loaded from cache (scanned earlier today)")
else:
    # Need to scan
    with st.spinner("🔍 Scanning all stocks for signals... This may take a few minutes."):
        df, scan_duration = scan_all_stocks()
        
        if df is None:
            st.error("❌ No stock data found in database. Please run data sync first.")
            st.stop()
        
        # Save to cache
        if not df.empty:
            save_success = data_manager.save_signals_to_cache(df, today_str, scan_duration)
            if save_success:
                st.success(f"✅ Scan complete in {scan_duration:.1f}s. Results cached for today.")
                st.session_state.force_rescan = False #reset flag
                cached_df = df
            else:
                st.warning("⚠️ Scan complete but failed to cache results.")

if df.empty:
    st.success("✨ No signals detected today. Market is quiet!")
    st.info("💡 This could mean:\n- All stocks are in neutral zones\n- No strong trends detected\n- Market is consolidating")
    st.stop()

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
    display_df = display_df[[
        'Type', 'Ticker', 'Name', 'Signal_Count', 'Signals', 
        'Price', 'RSI', 'ADX', 'MACD', 'Volume'
    ]]
    
    # Rename columns for display
    display_df.columns = [
        'Type', 'Code', 'Stock Name', '# Signals', 'Signal Details',
        'Price', 'RSI', 'ADX', 'MACD', 'Volume'
    ]
    
    # Display with color coding
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
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
    - **ADX Bottoming** - Trend strength bottoming, reversal setup
    - **RSI Bottoming** - RSI in bottom 10%, oversold
    """)

with col2:
    st.markdown("**⚠️ Bearish Signals (Alerts)**")
    st.markdown("""
    - **MACD Peaking** - MACD stopped rising, exhaustion detected
    - **MACD Bearish Crossover** - MACD crossed below Signal line
    - **ADX Peaking** - Trend strength peaking, momentum slowing
    - **ADX Reversing** - Trend strength falling, trend breaking
    - **RSI Peaking** - RSI in top 10%, overbought
    """)

st.markdown("---")
st.caption("💡 Tip: Results are cached daily in the main database. Click 'Force Rescan' if you want fresh data.")
