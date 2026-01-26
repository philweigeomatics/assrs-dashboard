"""
Single Stock Analysis Page
Advanced 3-Phase Trading System with Block Detection & Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import data_manager

st.set_page_config(
    page_title="📈 Single Stock | 个股分析",
    page_icon="📈",
    layout="wide"
)

# Check if statsmodels available
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ==========================================
# HELPER FUNCTIONS
# ==========================================

@st.cache_data(ttl=600)
def load_single_stock(ticker):
    """Load stock data using data_manager."""
    return data_manager.get_single_stock_data(ticker, use_data_start_date=True)


def run_single_stock_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced 3-Phase Analysis:
    Phase 1: Accumulation (OBV rising)
    Phase 2: Squeeze (BB width contraction)
    Phase 3: Golden Launch (ADX + MACD trigger)
    """
    df_analysis = df.copy()
    
    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        df_analysis.index = pd.to_datetime(df_analysis.index)
    df_analysis = df_analysis.sort_index()
    
    # Moving Averages
    df_analysis['MA20'] = ta.trend.sma_indicator(df_analysis['Close'], window=20)
    df_analysis['MA50'] = ta.trend.sma_indicator(df_analysis['Close'], window=50)
    df_analysis['MA200'] = ta.trend.sma_indicator(df_analysis['Close'], window=200)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df_analysis['Close'], window=20, window_dev=2)
    df_analysis['BB_Upper'] = bb.bollinger_hband()
    df_analysis['BB_Lower'] = bb.bollinger_lband()
    df_analysis['BB_Width'] = (df_analysis['BB_Upper'] - df_analysis['BB_Lower']) / df_analysis['Close']
    
    # MACD
    macd = ta.trend.MACD(df_analysis['Close'])
    df_analysis['MACD'] = macd.macd()
    df_analysis['MACD_Signal'] = macd.macd_signal()
    df_analysis['MACD_Hist'] = macd.macd_diff()
    
    # RSI
    df_analysis['RSI_14'] = ta.momentum.RSIIndicator(df_analysis['Close'], window=14).rsi()
    
    # ADX
    adx = ta.trend.ADXIndicator(df_analysis['High'], df_analysis['Low'], df_analysis['Close'], window=14)
    df_analysis['ADX'] = adx.adx()
    
    # OBV (On-Balance Volume)
    df_analysis['OBV'] = ta.volume.on_balance_volume(df_analysis['Close'], df_analysis['Volume'])
    
    # Price & OBV changes (20-day)
    df_analysis['PriceChg_20d'] = df_analysis['Close'].pct_change(periods=20)
    df_analysis['OBVChg_20d'] = df_analysis['OBV'].pct_change(periods=20)
    
    # Volume Stats
    df_analysis['VolMean_100d'] = df_analysis['Volume'].rolling(window=100, min_periods=20).mean()
    df_analysis['VolStd_100d'] = df_analysis['Volume'].rolling(window=100, min_periods=20).std()
    df_analysis['Volume_ZScore'] = (df_analysis['Volume'] - df_analysis['VolMean_100d']) / df_analysis['VolStd_100d']
    
    # Initialize signal columns
    df_analysis['Signal_Accumulation'] = False
    df_analysis['Signal_Squeeze'] = False
    df_analysis['Signal_GoldenLaunch'] = False
    df_analysis['Exit_MACDLead'] = False
    
    # Phase 1: Accumulation (OPTIMAL)
    # OBV rising while price is relatively flat (classic smart money accumulation)

    df_analysis['PriceChg_20d'] = df_analysis['Close'].pct_change(periods=20)
    df_analysis['OBVChg_20d'] = df_analysis['OBV'].pct_change(periods=20)

    accumulation = (
        (df_analysis['OBVChg_20d'] > 0.05) &  # OBV up 5% (volume accumulating)
        (df_analysis['PriceChg_20d'].abs() < 0.05) &  # Price relatively flat (5% not 10%)
        (df_analysis['RSI_14'] < 60)  # Not overbought (60 not 50 - allows early stage)
    )
    df_analysis.loc[accumulation, 'Signal_Accumulation'] = True

    
    # Phase 2: Squeeze
    # BB width contraction
    df_analysis['Min_Width_120d'] = df_analysis['BB_Width'].rolling(window=120, min_periods=20).min()
    # Squeeze = BB width is near its 120-day low (within 25%)
    squeeze = df_analysis['BB_Width'] <= (df_analysis['Min_Width_120d'] * 1.20)

    df_analysis.loc[squeeze, 'Signal_Squeeze'] = True
    
    # Phase 3: Golden Launch
    # Strong trend + MACD CROSSOVER (not just above)

    # Detect MACD crossover (just happened)
    macd_cross_up = (
        (df_analysis['MACD'].shift(1) <= df_analysis['MACD_Signal'].shift(1)) &  # Was below/at yesterday
        (df_analysis['MACD'] > df_analysis['MACD_Signal'])  # Is above today
    )

    # Golden Launch = Fresh MACD cross + confirming conditions
    launch = (
        macd_cross_up &  # ← KEY: Only on the crossover day itself
        (df_analysis['MA20'] > df_analysis['MA50']) &  # Bullish MA alignment
        (df_analysis['ADX'] > 25) &  # Strong trend
        (df_analysis['RSI_14'] > 50) &  # Momentum
        (df_analysis['RSI_14'] < 70)  # Not overextended
    )
    df_analysis.loc[launch, 'Signal_GoldenLaunch'] = True

    
    # Exit Signal: Multiple conditions (any one triggers)

    # ==========================================
    # EXIT SIGNAL (Improved - Less Noise)
    # ==========================================

    # Condition 1: MACD crosses down from bullish zone WITH confirmation
    macd_bearish_cross = (
        (df_analysis['MACD'].shift(1) > df_analysis['MACD_Signal'].shift(1)) &  # Was above
        (df_analysis['MACD'] < df_analysis['MACD_Signal']) &  # Crossed down
        (df_analysis['MACD'].shift(1) > 0) &  # Was in bullish territory
        (df_analysis['MACD_Hist'] < df_analysis['MACD_Hist'].shift(1))  # Histogram weakening
    )

    # Condition 2: MA20 crosses below MA50 (major trend reversal)
    ma_cross_down = (
        (df_analysis['MA20'].shift(1) > df_analysis['MA50'].shift(1)) &
        (df_analysis['MA20'] < df_analysis['MA50']) &
        (df_analysis['ADX'] > 20)  # ← ADD: Only in trending conditions (not noise)
    )

    # Condition 3: RSI extreme exhaustion (remove this - too frequent)
    # rsi_exhaustion causes too many false exits - REMOVED

    # Combine conditions (both must be stronger signals now)
    exit_signal = macd_bearish_cross | ma_cross_down

    df_analysis.loc[exit_signal, 'Exit_MACDLead'] = True
    
    return df_analysis


def calculate_multiple_blocks(df, lookback=60):

    if len(df) < 20:
        return []
    
    subset = df.tail(lookback).copy()
    all_dates = subset.index.tolist()
    
    # 1. Detect Breakout Indices (high vol + big price move)
    subset['PctChange'] = subset['Close'].pct_change().abs()
    subset['VolRatio'] = subset['Volume'] / subset['Volume'].rolling(20).mean().shift(1)
    
    breakout_mask = (subset['PctChange'] > 0.03) & (subset['VolRatio'] > 1.5)
    breakout_dates = subset.index[breakout_mask].tolist()
    
    # 2. Create segment boundaries
    boundary_indices = [0]
    for d in breakout_dates:
        if d in all_dates:
            boundary_indices.append(all_dates.index(d))
    boundary_indices.append(len(all_dates))
    boundary_indices = sorted(list(set(boundary_indices)))
    
    blocks = []
    
    # 3. Iterate through segments
    for i in range(len(boundary_indices) - 1):
        idx_start = boundary_indices[i]
        idx_end = boundary_indices[i + 1]
        
        if i == len(boundary_indices) - 2:  # Last block (current/active)
            seg_df = subset.iloc[idx_start:]
            is_active = True
        else:  # Historical block
            seg_df = subset.iloc[idx_start:idx_end]
            is_active = False
        
        if len(seg_df) < 3:
            continue  # Skip noise
        
        # 4. Find the "box" - price range with most volume
        price_min = seg_df['Low'].min()
        price_max = seg_df['High'].max()
        
        if price_min == price_max:
            price_max = price_min + 0.01
        
        # Volume profile: bin prices and sum volume in each bin
        bins = np.linspace(price_min, price_max, 21)
        indices = np.digitize(seg_df['Close'], bins)
        
        bin_volumes = {}
        for j, vol in zip(indices, seg_df['Volume']):
            bin_volumes[j] = bin_volumes.get(j, 0) + vol
        
        # Find bins containing 70% of volume (the core trading range)
        sorted_bins = sorted(bin_volumes.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(bin_volumes.values())
        
        current_vol = 0
        value_bins = []
        for bin_idx, vol in sorted_bins:
            current_vol += vol
            value_bins.append(bin_idx)
            if current_vol >= 0.7 * total_volume:
                break
        
        valid_indices = [v for v in value_bins if 1 <= v < len(bins)]
        if not valid_indices:
            continue
        
        # Define box range
        top = bins[max(valid_indices)]
        bot = bins[min(valid_indices) - 1]
        
        if top <= bot:
            top = bot + 0.01
        
        # Determine status
        current_price = df['Close'].iloc[-1]
        
        if is_active:
            if current_price > top:
                status = 'BREAKOUT'
            elif current_price < bot:
                status = 'BREAKDOWN'
            else:
                status = 'INSIDE'
        else:
            if current_price > top:
                status = 'SUPPORT_BELOW'
            elif current_price < bot:
                status = 'RESISTANCE_ABOVE'
            else:
                status = 'INSIDE_OLD_RANGE'
        
        blocks.append({
            'start': seg_df.index[0],
            'end': seg_df.index[-1],
            'top': top,
            'bot': bot,
            'status': status,
            'is_active': is_active
        })
    
    return blocks


def calculate_trend_forecast(df: pd.DataFrame, lookback: int = 60, forecast_days: int = 30, 
                             degree: int = 1, model_type: str = 'Linear') -> tuple:
    """
    Calculate trend forecast using multiple methods.
    Returns (forecast_series, upper_bound, lower_bound).
    """
    if df.empty or len(df) < lookback:
        return None, None, None
    
    recent = df.tail(lookback).copy()
    
    # Use VWAP instead of Close for volume-weighted price
    vwap = (recent['Close'] * recent['Volume']).sum() / recent['Volume'].sum()
    prices = recent['Close'].values
    
    try:
        if model_type == 'Linear' or model_type == 'Quadratic':
            # Polynomial regression
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, deg=degree)
            poly = np.poly1d(coeffs)
            
            # Extend to forecast
            x_future = np.arange(len(prices) + forecast_days)
            forecast = poly(x_future)
            
            # Estimate bounds based on residual std
            residuals = prices - poly(x)
            std = np.std(residuals)
            
            forecast_only = forecast[-forecast_days:]
            upper = forecast_only + 2 * std
            lower = forecast_only - 2 * std
            
            return forecast_only, upper, lower
        
        elif model_type == 'Holt-Winters' and HAS_STATSMODELS:
            # Exponential Smoothing
            model = ExponentialSmoothing(
                prices,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            fit = model.fit()
            forecast = fit.forecast(steps=forecast_days)
            
            # Simple bounds
            std = np.std(prices)
            upper = forecast + 2 * std
            lower = forecast - 2 * std
            
            return forecast, upper, lower
        
        elif model_type == 'ARIMA' and HAS_STATSMODELS:
            # ARIMA with volume as exogenous variable
            vol_norm = (recent['Volume'] - recent['Volume'].mean()) / recent['Volume'].std()
            
            model = ARIMA(prices, exog=vol_norm, order=(1, 1, 1))
            fit = model.fit()
            
            # Forecast (need future exog - use mean)
            exog_forecast = np.zeros(forecast_days)
            forecast_result = fit.forecast(steps=forecast_days, exog=exog_forecast)
            
            # Get confidence intervals
            forecast_obj = fit.get_forecast(steps=forecast_days, exog=exog_forecast)
            conf_int = forecast_obj.conf_int()
            
            return forecast_result, conf_int[:, 1], conf_int[:, 0]
        
        else:
            return None, None, None
    
    except Exception as e:
        st.warning(f"Forecast failed: {str(e)}")
        return None, None, None


def create_single_stock_chart_analysis(df: pd.DataFrame, blocks: list = None) -> go.Figure:
    """
    Create 4-panel chart with trading blocks.
    Blocks drawn as horizontal rectangles during their active period.
    """
    df = df.tail(250).sort_index()
    dates = df.index.strftime('%Y-%m-%d')
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('Price + Trading Blocks + Signals', 'Volume + OBV', 'MACD', 'RSI + ADX'),
        row_heights=[0.5, 0.15, 0.15, 0.15]
    )
    
    # ==========================================
    # PANEL 1: PRICE
    # ==========================================
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=dates, y=df['BB_Upper'],
        line=dict(color='rgba(147,197,253,0.5)', width=1),
        name='BB Upper',
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=df['BB_Lower'],
        line=dict(color='rgba(147,197,253,0.5)', width=1),
        fill='tonexty',
        fillcolor='rgba(59,130,246,0.05)',
        name='BB Lower',
        showlegend=True
    ), row=1, col=1)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        showlegend=True,
        increasing=dict(line=dict(color='#ef4444')),
        decreasing=dict(line=dict(color='#22c55e'))
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=dates, y=df['MA20'], 
        name='MA20',
        line=dict(color='#fbbf24', dash='dot', width=1.5),
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=df['MA50'], 
        name='MA50',
        line=dict(color='#3b82f6', width=2),
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=df['MA200'], 
        name='MA200',
        line=dict(color='#374151', width=2.5),
        showlegend=True
    ), row=1, col=1)
    
    # ==========================================
    # TRADING BLOCKS (Original Simple Version)
    # ==========================================
    
    if blocks:
        colors = [
            'rgba(255, 99, 71, 0.2)',   # Red
            'rgba(255, 165, 0, 0.2)',    # Orange
            'rgba(255, 215, 0, 0.2)'     # Yellow
        ]
        
        for idx, block in enumerate(blocks[:3]):
            color = colors[idx % len(colors)]
            
            start_date = block['start'].strftime('%Y-%m-%d')
            end_date = block['end'].strftime('%Y-%m-%d')
            
            # Draw rectangle for the block
            fig.add_shape(
                type="rect",
                x0=start_date,
                x1=end_date,
                y0=block['bot'],
                y1=block['top'],
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'),
                row=1, col=1
            )
            
            # Add label
            fig.add_annotation(
                x=end_date,
                y=(block['top'] + block['bot']) / 2,
                text=f"Box {idx+1}<br>¥{block['bot']:.2f}-¥{block['top']:.2f}<br>{block['status']}",
                showarrow=False,
                font=dict(size=9, color='black'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1,
                borderpad=3,
                row=1, col=1
            )
    
    # ==========================================
    # PHASE SIGNALS
    # ==========================================
    
    acc = df[df['Signal_Accumulation']]
    if not acc.empty:
        fig.add_trace(go.Scatter(
            x=acc.index.strftime('%Y-%m-%d'),
            y=acc['Low'] * 0.98,
            mode='markers', 
            name='Phase 1: Accumulation',
            marker=dict(color='#eab308', size=10, symbol='circle'),
            showlegend=True
        ), row=1, col=1)
    
    sqz = df[df['Signal_Squeeze']]
    if not sqz.empty:
        fig.add_trace(go.Scatter(
            x=sqz.index.strftime('%Y-%m-%d'),
            y=sqz['High'] * 1.02,
            mode='markers', 
            name='Phase 2: Squeeze',
            marker=dict(color='#64748b', size=8, symbol='square'),
            showlegend=True
        ), row=1, col=1)
    
    launch = df[df['Signal_GoldenLaunch']]
    if not launch.empty:
        fig.add_trace(go.Scatter(
            x=launch.index.strftime('%Y-%m-%d'),
            y=launch['High'] * 1.05,
            mode='markers', 
            name='🚀 GOLDEN LAUNCH',
            marker=dict(color='#ef4444', size=16, symbol='star',
                       line=dict(width=2, color='black')),
            showlegend=True
        ), row=1, col=1)
    
    exits = df[df['Exit_MACDLead']]
    if not exits.empty:
        fig.add_trace(go.Scatter(
            x=exits.index.strftime('%Y-%m-%d'),
            y=exits['High'] * 1.01,
            mode='markers',
            name='Exit Signal',
            marker=dict(color='#22c55e', size=10, symbol='x'),
            showlegend=True
        ), row=1, col=1)
    
    # ==========================================
    # PANEL 2: VOLUME + OBV
    # ==========================================
    
    fig.add_trace(go.Bar(
        x=dates, 
        y=df['Volume'],
        name='Volume',
        marker=dict(color='rgba(209, 213, 219, 0.5)'),
        showlegend=True
    ), row=2, col=1)
    
    if 'OBV' in df.columns:
        # Normalize OBV to volume scale
        obv_normalized = df['OBV'] / df['OBV'].max() * df['Volume'].max()
        
        fig.add_trace(go.Scatter(
            x=dates, 
            y=obv_normalized,
            name='OBV (scaled)',
            line=dict(color='#f97316', width=3),
            showlegend=True
        ), row=2, col=1)
    
    # ==========================================
    # PANEL 3: MACD
    # ==========================================
    
    colors = ['#ef4444' if val > 0 else '#22c55e' for val in df['MACD_Hist']]
    
    fig.add_trace(go.Bar(
        x=dates, 
        y=df['MACD_Hist'],
        name='MACD Histogram',
        marker=dict(color=colors),
        showlegend=True
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, 
        y=df['MACD'],
        name='MACD',
        line=dict(color='#2563eb', width=2),
        showlegend=True
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, 
        y=df['MACD_Signal'],
        name='MACD Signal',
        line=dict(color='#f97316', width=2),
        showlegend=True
    ), row=3, col=1)
    
    # ==========================================
    # PANEL 4: RSI + ADX
    # ==========================================
    
    fig.add_trace(go.Scatter(
        x=dates, 
        y=df['RSI_14'],
        name='RSI (14)',
        line=dict(color='#8b5cf6', width=2.5),
        showlegend=True
    ), row=4, col=1)
    
    fig.add_hline(y=70, line_dash='dot', line_color='#dc2626', 
                  annotation_text="Overbought (70)", row=4, col=1)
    fig.add_hline(y=30, line_dash='dot', line_color='#16a34a', 
                  annotation_text="Oversold (30)", row=4, col=1)
    
    if 'ADX' in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, 
            y=df['ADX'],
            name='ADX (Trend Strength)',
            line=dict(color='#10b981', width=2.5, dash='dash'),
            showlegend=True
        ), row=4, col=1)
        
        fig.add_hline(y=25, line_dash='dot', line_color='#6b7280',
                     annotation_text="Strong Trend (25+)", row=4, col=1)
    
    # ==========================================
    # LAYOUT
    # ==========================================
    
    fig.update_layout(
        height=1100,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        xaxis4_title='Date',
        yaxis1_title='Price (¥)',
        yaxis2_title='Volume',
        yaxis3_title='MACD',
        yaxis4_title='RSI / ADX',
        yaxis4_range=[0, 100],
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    return fig


def create_forecast_chart(df: pd.DataFrame, forecast: np.ndarray, upper: np.ndarray, 
                         lower: np.ndarray, model_name: str = 'Linear') -> go.Figure:
    """Create forecast chart with confidence bands."""
    
    recent = df.tail(60).copy()
    dates_hist = recent.index
    last_date = dates_hist[-1]
    
    # Generate future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast), freq='B')
    
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=dates_hist,
        y=recent['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines',
        name=f'{model_name} Forecast',
        line=dict(color='#ef4444', width=2, dash='dash')
    ))
    
    # Confidence bands
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        mode='lines',
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title=f'{model_name} Trend Forecast (30 Days)',
        height=400,
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    return fig


# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================

if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = None

def set_active_ticker(ticker: str):
    """Set active ticker and sync input."""
    st.session_state.active_ticker = ticker
    st.session_state.ticker_input = ticker

def analyze_ticker():
    """Analyze the ticker from input box."""
    ticker_val = st.session_state.get('ticker_input')
    if ticker_val:
        st.session_state.active_ticker = ticker_val


# ==========================================
# MAIN APP
# ==========================================

st.title("📈 Single Stock Analysis")
st.markdown("**3-Phase Trading System**: Accumulation → Squeeze → Golden Launch")

# Input section
c1, c2 = st.columns([3, 1])

with c1:
    ticker_input = st.text_input("Enter Stock Code (e.g., 600760):", key='ticker_input')

with c2:
    st.button("Analyze", key='analyze_btn', on_click=analyze_ticker)

# Search history
history = data_manager.get_search_history()
if history:
    st.caption("Recent searches (click to load):")
    hist_cols = st.columns(min(len(history), 5), gap='small')
    for idx, item in enumerate(history):
        col = hist_cols[idx % len(hist_cols)]
        with col:
            st.button(
                item['ticker'],
                key=f'history_{idx}',
                on_click=set_active_ticker,
                args=(item['ticker'],)
            )

# Main analysis
if st.session_state.active_ticker:
    ticker = st.session_state.active_ticker.strip()
    
    with st.spinner(f"Loading {ticker}..."):
        stock_df = load_single_stock(ticker)
    
    if stock_df is None or stock_df.empty:
        st.error(f"No data found for {ticker}. Check ticker is valid.")
    else:
        # Run analysis
        analysis_df = run_single_stock_analysis(stock_df)
        
        if analysis_df.empty or len(analysis_df) < 50:
            st.error("Not enough data to compute signals.")
        else:
            # Detect trading blocks
            blocks = calculate_multiple_blocks(analysis_df, lookback=60)
            
            # Display chart
            fig_stock = create_single_stock_chart_analysis(analysis_df, blocks=blocks)
            st.plotly_chart(fig_stock, use_container_width=True)
            
            # Latest status cards
            latest_row = analysis_df.iloc[-1]
            
            st.subheader("Latest Status")
            cola, colb, colc, cold = st.columns(4)
            
            accum = bool(latest_row.get('Signal_Accumulation', False))
            squeeze = bool(latest_row.get('Signal_Squeeze', False))
            launch = bool(latest_row.get('Signal_GoldenLaunch', False))
            adx_val = float(latest_row.get('ADX', 0.0))
            
            with cola:
                st.markdown("**Phase 1: Accumulation**")
                st.markdown(f":{('orange' if accum else 'grey')}[{'ACTIVE' if accum else 'INACTIVE'}]")
            
            with colb:
                st.markdown("**Phase 2: Squeeze**")
                st.markdown(f":{('grey')}[{'TIGHT' if squeeze else 'LOOSE'}]")
            
            with colc:
                st.markdown("**Phase 3: LAUNCH**")
                st.markdown(f":{('red' if launch else 'grey')}[{'TRIGGERED' if launch else 'WAITING'}]")
            
            with cold:
                st.markdown("**Trading Block (Latest)**")
                if blocks:
                    block = blocks[0]
                    status = block['status']
                    color = 'red' if status == 'BREAKOUT' else 'green' if status == 'BREAKDOWN' else 'grey'
                    st.markdown(f":{color}[{status}]")
                    st.caption(f"Range: {block['bot']:.2f} - {block['top']:.2f}")
                else:
                    st.markdown(":grey[NO BLOCKS]")
            
            # Metrics table
            st.markdown("---")
            st.subheader("Key Metrics")
            
            table_cols = ['Close', 'MA20', 'MA50', 'MA200', 'RSI_14', 'ADX',
                         'Signal_Accumulation', 'Signal_Squeeze', 'Signal_GoldenLaunch']
            
            metrics_df = analysis_df[table_cols].tail(5)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Trend Forecast Section
            st.markdown("---")
            st.subheader("Trend Forecast (30 Days)")
            
            col_model, col_lookback, col_info = st.columns([1, 1, 2])
            
            with col_lookback:
                lookback_option = st.selectbox("Lookback Period:", [30, 60, 90, 120], index=1)
            
            with col_model:
                model_options = ['Linear (Straight)', 'Quadratic (Curved)']
                if HAS_STATSMODELS:
                    model_options.extend(['Holt-Winters (Exp. Smoothing)', 'ARIMA (AutoRegressive)'])
                model_option = st.radio("Trend Model:", model_options, index=0)
            
            with col_info:
                degree = 1
                if 'Linear' in model_option:
                    st.info(f"Linear Regression: Projects trend as straight line based on last {lookback_option} days.")
                elif 'Quadratic' in model_option:
                    st.info(f"Polynomial: Fits curve to detect acceleration/deceleration.")
                    degree = 2
                elif 'Holt-Winters' in model_option:
                    st.info(f"Exponential Smoothing: Weights recent data heavily.")
                elif 'ARIMA' in model_option:
                    st.info(f"ARIMA: Uses volume as exogenous variable to weight predictions.")
            
            # Calculate forecast
            model_key = model_option.split()[0]  # Extract: Linear, Quadratic, Holt-Winters, ARIMA
            
            forecast, upper, lower = calculate_trend_forecast(
                analysis_df,
                lookback=lookback_option,
                forecast_days=30,
                degree=degree,
                model_type=model_key
            )
            
            if forecast is not None:
                fig_fc = create_forecast_chart(analysis_df, forecast, upper, lower, model_name=model_key)
                st.plotly_chart(fig_fc, use_container_width=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Projected Target (30d)", f"{forecast[-1]:.2f}")
                with c2:
                    st.metric("Upside Resistance", f"{upper[-1]:.2f}")
                with c3:
                    st.metric("Downside Support", f"{lower[-1]:.2f}")
            else:
                st.warning("Not enough data or model failed to converge.")

else:
    st.info("👆 Enter a stock code above to begin analysis")
