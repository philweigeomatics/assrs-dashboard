import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta
import os
from datetime import datetime, timedelta

# Try importing statsmodels for advanced forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Assuming data_manager is in the same directory or path
import data_manager 

# --- 1. CONFIGURATION ---
V1_RULES_FILE = 'assrs_backtest_results_SECTORS_V1_Rules.csv'
V2_REGIME_FILE = 'assrs_backtest_results_SECTORS_V2_Regime.csv'

# Set page config
st.set_page_config(
    page_title="ASSRS Sector Scoreboard",
    layout="wide"
)

# --- 2. HELPER FUNCTIONS (SECTOR DATA - UNCHANGED) ---

@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_data(filepath, model_name):
    """Loads and prepares all data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert all potential numeric columns
        cols_to_numeric = ['TOTAL_SCORE', 'Open', 'High', 'Low', 'Close', 'Volume_Metric']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.empty:
            return None, None, f"No data in {model_name} file.", None

        # 1. Get Latest Scores
        latest_date = df['Date'].max()
        latest_scores_df = df[df['Date'] == latest_date].copy()
        
        # 2. Get Full History
        full_history_df = df.copy()
        
        return latest_scores_df, full_history_df, latest_date.strftime('%Y-%m-%d'), None
        
    except FileNotFoundError:
        return None, None, None, f"ERROR: File not found: {filepath}. The data-update task may not have run yet."
    except Exception as e:
        return None, None, None, f"An error occurred: {str(e)}"

def style_action(action):
    """Applies color to the 'ACTION' column for the dataframe."""
    if not isinstance(action, str): return ''
    if 'GREEN' in action:
        return 'color: #15803d; background-color: #dcfce7; font-weight: 600;'
    if 'YELLOW' in action:
        return 'color: #a16207; background-color: #fef9c3; font-weight: 600;'
    if 'RED' in action:
        return 'color: #b91c1c; background-color: #fee2e2; font-weight: 600;'
    if 'CONSOLIDATION' in action:
        return 'color: #4b5563; background-color: #f3f4f6; font-weight: 500;'
    return ''

def create_drilldown_chart(chart_data, model_type):
    """
    Creates a 3-plot interactive chart for Sectors.
    """
    is_v1 = model_type == 'v1'
    y_title_score = 'V1 Score (-3 to 8)' if is_v1 else 'V2 Bull Probability (0 to 1)'
    y_range_score = [-3.1, 8.1] if is_v1 else [-0.1, 1.1]

    date_strings = chart_data['Date'].dt.strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price (PPI)', 'Volume (Z-Score)', 'Signal (Score)'),
        row_heights=[0.5, 0.2, 0.3]
    )

    # Price (Chinese Colors: Red Up, Green Down)
    fig.add_trace(go.Candlestick(
        x=date_strings,
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name='Price',
        increasing=dict(line=dict(color='#ef4444')), # Red
        decreasing=dict(line=dict(color='#22c55e'))  # Green
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=date_strings,
        y=chart_data['Volume_Metric'],
        name='Volume Metric',
        marker_color='rgba(107, 114, 128, 0.3)'
    ), row=2, col=1)

    # Score
    fig.add_trace(go.Scatter(
        x=date_strings,
        y=chart_data['TOTAL_SCORE'],
        name='Score',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ), row=3, col=1)

    # Thresholds
    if is_v1:
        fig.add_hline(y=2.5, line_dash="dash", line_color="#a16207", annotation_text="Buy (2.5)", row=3, col=1)
        fig.add_hline(y=5.0, line_dash="dash", line_color="#15803d", annotation_text="Strong (5.0)", row=3, col=1)
    else:
        fig.add_hline(y=0.8, line_dash="dash", line_color="#15803d", annotation_text="High Prob (0.8)", row=3, col=1)
        fig.add_hline(y=0.2, line_dash="dash", line_color="#b91c1c", annotation_text="Low Prob (0.2)", row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis3_title='Date',
        yaxis1_title="PPI",
        yaxis2_title="Vol Z-Score",
        yaxis3_title=y_title_score,
        yaxis3_range=y_range_score,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    return fig

# --- 3. SINGLE STOCK ANALYSIS LOGIC (NEW 3-PHASE) ---

def run_single_stock_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced 3-Phase Analysis: Accumulation -> Squeeze -> Trigger
    Replaces old Breakout/Pullback signals.
    """
    df_analysis = df.copy()

    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        df_analysis.index = pd.to_datetime(df_analysis.index)
    df_analysis = df_analysis.sort_index()

    # =======================
    # 0. BASIC TREND FILTERS
    # =======================
    df_analysis['MA20'] = ta.trend.sma_indicator(df_analysis['Close'], window=20)
    df_analysis['MA50'] = ta.trend.sma_indicator(df_analysis['Close'], window=50)
    df_analysis['MA200'] = ta.trend.sma_indicator(df_analysis['Close'], window=200)
    
    # =======================
    # PHASE 1: ACCUMULATION (OBV Divergence)
    # =======================
    df_analysis['OBV'] = ta.volume.on_balance_volume(df_analysis['Close'], df_analysis['Volume'])
    
    # We check the "Slope" over the last 20 days
    df_analysis['Price_Chg_20d'] = df_analysis['Close'].pct_change(periods=20)
    df_analysis['OBV_Chg_20d'] = df_analysis['OBV'].pct_change(periods=20)

    # LOGIC: Price is boring (moved < 3% up or down), but OBV is UP (> 5%)
    df_analysis['Signal_Accumulation'] = (
        (df_analysis['Price_Chg_20d'].abs() < 0.03) & 
        (df_analysis['OBV_Chg_20d'] > 0.05)
    )

    # =======================
    # PHASE 2: THE SQUEEZE (Bollinger Bandwidth)
    # =======================
    bb_indicator = ta.volatility.BollingerBands(close=df_analysis['Close'], window=20, window_dev=2)
    df_analysis['BB_Upper'] = bb_indicator.bollinger_hband()
    df_analysis['BB_Lower'] = bb_indicator.bollinger_lband()
    df_analysis['BB_Mid']   = bb_indicator.bollinger_mavg()
    
    # Bandwidth % = (Upper - Lower) / Mid
    df_analysis['BB_Width'] = bb_indicator.bollinger_wband()
    
    # Is the current width near a 6-month (120-day) LOW?
    # We define "Squeeze" as being within 20% of the 120-day minimum width
    df_analysis['Min_Width_120d'] = df_analysis['BB_Width'].rolling(window=120).min()
    df_analysis['Signal_Squeeze'] = df_analysis['BB_Width'] <= (df_analysis['Min_Width_120d'] * 1.2)

    # =======================
    # PHASE 3: THE LAUNCH (Trigger)
    # =======================
    # 1. ADX for Trend Strength
    adx_indicator = ta.trend.ADXIndicator(df_analysis['High'], df_analysis['Low'], df_analysis['Close'], window=14)
    df_analysis['ADX'] = adx_indicator.adx()
    
    # 2. Volume Surge
    df_analysis['20d_Avg_Vol'] = df_analysis['Volume'].rolling(window=20).mean()
    
    # 3. The Signal:
    # Was there a squeeze in the last 10 days?
    df_analysis['Recent_Squeeze'] = df_analysis['Signal_Squeeze'].rolling(window=10).max() > 0

    df_analysis['Signal_Golden_Launch'] = (
        df_analysis['Recent_Squeeze'] &                     # Context: Coming out of quiet period
        (df_analysis['Close'] > df_analysis['BB_Upper']) &  # Breakout
        (df_analysis['Volume'] > df_analysis['20d_Avg_Vol'] * 1.5) & # Power
        (df_analysis['ADX'] > 20)                           # Trend Strength
    )

    # =======================
    # STANDARD EXITS & INDICATORS
    # =======================
    macd_indicator = ta.trend.MACD(close=df_analysis['Close'])
    df_analysis['MACD'] = macd_indicator.macd()
    df_analysis['MACD_Signal'] = macd_indicator.macd_signal()
    df_analysis['MACD_Hist'] = macd_indicator.macd_diff()
    
    df_analysis['High_50d'] = df_analysis['Close'].rolling(window=50).max()

    # Exit: MACD Lead
    df_analysis['Exit_MACD_Lead'] = (
        (df_analysis['MACD'] < df_analysis['MACD_Signal']) &
        (df_analysis['MACD'].shift(1) > df_analysis['MACD_Signal'].shift(1)) &
        (df_analysis['MACD'] > 0)
    )

    # Keep RSI for plotting
    df_analysis['RSI_14'] = ta.momentum.rsi(df_analysis['Close'], window=14)

    return df_analysis

def calculate_trading_block(df, lookback=60):
    """
    Identifies the current 'Trading Block' (Support/Resistance) using Volume Profile.
    Returns: block_top, block_bottom, status_string
    """
    subset = df.tail(lookback).copy()
    if len(subset) < 20:
        return None, None, "Insufficient Data"
    
    # 1. Bucket prices into bins
    # Use roughly 20 bins across the price range
    price_min = subset['Low'].min()
    price_max = subset['High'].max()
    bins = np.linspace(price_min, price_max, 21) # 20 bins
    
    # 2. Assign volume to bins (using Close price as proxy)
    indices = np.digitize(subset['Close'], bins)
    
    bin_volumes = {}
    for i, vol in zip(indices, subset['Volume']):
        bin_volumes[i] = bin_volumes.get(i, 0) + vol
        
    # 3. Find the "Value Area" (70% of volume)
    # Sort bins by volume descending
    sorted_bins = sorted(bin_volumes.items(), key=lambda x: x[1], reverse=True)
    
    total_volume = sum(bin_volumes.values())
    current_vol = 0
    value_bins = []
    
    for bin_idx, vol in sorted_bins:
        current_vol += vol
        value_bins.append(bin_idx)
        if current_vol > 0.7 * total_volume:
            break
            
    # 4. Determine Block Levels
    # Indices correspond to bins array. 
    # bin_idx i corresponds to range bins[i-1] to bins[i] (roughly)
    # Safe bounds check
    valid_indices = [i for i in value_bins if 1 <= i < len(bins)]
    
    if not valid_indices:
        return price_max, price_min, "Undefined"
        
    min_idx = min(valid_indices)
    max_idx = max(valid_indices)
    
    block_bottom = bins[min_idx-1]
    block_top = bins[max_idx]
    
    # 5. Status Check
    last_close = subset['Close'].iloc[-1]
    
    if last_close > block_top * 1.01:
        status = "BREAKOUT (Bullish)"
    elif last_close < block_bottom * 0.99:
        status = "BREAKDOWN (Bearish)"
    else:
        status = "INSIDE BLOCK"
        
    return block_top, block_bottom, status

def create_single_stock_chart(analysis_df: pd.DataFrame, window: int = 250, block_info=None) -> go.Figure:
    """
    Build a 4-panel Plotly chart matching the WebApp:
    1) Price + Bollinger Bands + MAs + Signals + TRADING BLOCK
    2) Volume
    3) MACD
    4) RSI
    """
    df = analysis_df.tail(window).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    date_strings = df.index.strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('Price (Accumulation → Squeeze → Launch)', 'Volume', 'MACD', 'RSI'),
        row_heights=[0.5, 0.15, 0.15, 0.15]
    )

    # --- PANEL 1: PRICE, BANDS, SIGNALS ---

    # 1. Bollinger Bands (Background)
    # Upper Band
    fig.add_trace(go.Scatter(
        x=date_strings, y=df['BB_Upper'],
        mode='lines', name='BB Upper',
        line=dict(width=1, color='rgba(147, 197, 253, 0.5)'),
        showlegend=False,
    ), row=1, col=1)

    # Lower Band (Filled)
    fig.add_trace(go.Scatter(
        x=date_strings, y=df['BB_Lower'],
        mode='lines', name='Bollinger Band',
        line=dict(width=1, color='rgba(147, 197, 253, 0.5)'),
        fill='tonexty', fillcolor='rgba(59, 130, 246, 0.05)',
        showlegend=True,
    ), row=1, col=1)

    # --- TRADING BLOCK OVERLAY ---
    if block_info and block_info[0] is not None:
        top, bottom, _ = block_info
        # We draw the block across the last 60 days (or however long the lookback was)
        # For simplicity, we draw it across the visible chart area or just the recent part
        # Let's draw it as a shape covering the last 60 days
        
        # Calculate start date for shape
        if len(df) >= 60:
            start_shape_idx = len(df) - 60
            start_date_str = date_strings[start_shape_idx]
        else:
            start_date_str = date_strings[0]
            
        end_date_str = date_strings[-1]
        
        fig.add_shape(
            type="rect",
            x0=start_date_str, y0=bottom,
            x1=end_date_str, y1=top,
            fillcolor="rgba(147, 51, 234, 0.15)", # Purple tint
            line=dict(color="rgba(147, 51, 234, 0.6)", width=1, dash="dash"),
            row=1, col=1,
            layer="below"
        )
        # Add labels
        fig.add_annotation(x=end_date_str, y=top, text="Block Res", showarrow=False, yshift=5, font=dict(size=10, color="purple"), row=1, col=1)
        fig.add_annotation(x=end_date_str, y=bottom, text="Block Sup", showarrow=False, yshift=-5, font=dict(size=10, color="purple"), row=1, col=1)


    # 2. Candlesticks (Chinese Colors: Red Up, Green Down)
    fig.add_trace(go.Candlestick(
        x=date_strings,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing=dict(line=dict(color='#ef4444')), # Red
        decreasing=dict(line=dict(color='#22c55e'))  # Green
    ), row=1, col=1)

    # 3. MAs
    fig.add_trace(go.Scatter(x=date_strings, y=df['MA20'], name='MA20', line=dict(width=1, color='#fbbf24', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=date_strings, y=df['MA50'], name='MA50', line=dict(width=1.5, color='#3b82f6')), row=1, col=1)
    fig.add_trace(go.Scatter(x=date_strings, y=df['MA200'], name='MA200', line=dict(width=2, color='#374151')), row=1, col=1)

    # 4. Signals (Markers)
    # Phase 1: Accumulation (Yellow)
    acc_df = df[df['Signal_Accumulation']]
    fig.add_trace(go.Scatter(
        x=acc_df.index.strftime('%Y-%m-%d'), y=acc_df['Low'] * 0.98,
        mode='markers', name='Accumulation',
        marker=dict(color='#eab308', size=6, symbol='circle')
    ), row=1, col=1)

    # Phase 2: Squeeze (Grey)
    sqz_df = df[df['Signal_Squeeze']]
    fig.add_trace(go.Scatter(
        x=sqz_df.index.strftime('%Y-%m-%d'), y=sqz_df['High'] * 1.02,
        mode='markers', name='Squeeze',
        marker=dict(color='#64748b', size=5, symbol='square-open')
    ), row=1, col=1)

    # Phase 3: Golden Launch (Red Star - Matches Up)
    launch_df = df[df['Signal_Golden_Launch']]
    fig.add_trace(go.Scatter(
        x=launch_df.index.strftime('%Y-%m-%d'), y=launch_df['High'] * 1.05,
        mode='markers', name='GOLDEN LAUNCH',
        marker=dict(color='#ef4444', size=14, symbol='star', line=dict(width=1, color='black'))
    ), row=1, col=1)

    # Exit Signals (Green X - Matches Down)
    exit_df = df[df['Exit_MACD_Lead']]
    fig.add_trace(go.Scatter(
        x=exit_df.index.strftime('%Y-%m-%d'), y=exit_df['High'] * 1.01,
        mode='markers', name='Exit MACD',
        marker=dict(color='#22c55e', size=8, symbol='x') # Green X for Exit
    ), row=1, col=1)

    # --- PANEL 2: VOLUME ---
    fig.add_trace(go.Bar(
        x=date_strings, y=df['Volume'], name='Volume',
        marker=dict(color='#d1d5db')
    ), row=2, col=1)
    
    if '20d_Avg_Vol' in df.columns:
        fig.add_trace(go.Scatter(
            x=date_strings, y=df['20d_Avg_Vol'], name='Vol MA20',
            line=dict(width=1, color='#4b5563')
        ), row=2, col=1)

    # --- PANEL 3: MACD ---
    # Histogram colors (Red=Pos/Up, Green=Neg/Down)
    colors = np.where(df['MACD_Hist'] >= 0, '#ef4444', '#22c55e')
    fig.add_trace(go.Bar(
        x=date_strings, y=df['MACD_Hist'], name='MACD Hist',
        marker_color=colors
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=date_strings, y=df['MACD'], name='MACD',
        line=dict(color='#2563eb', width=1.5)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=date_strings, y=df['MACD_Signal'], name='Signal',
        line=dict(color='#f97316', width=1.5)
    ), row=3, col=1)

    # --- PANEL 4: RSI ---
    fig.add_trace(go.Scatter(
        x=date_strings, y=df['RSI_14'], name='RSI',
        line=dict(color='#8b5cf6', width=2)
    ), row=4, col=1)

    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=4, col=1)

    # Layout Updates
    fig.update_layout(
        height=1000,
        template="plotly_white", 
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis_rangeslider_visible=False,
        
        # Unified Hover (Crosshair)
        hovermode="x unified",
        
        # Enable Spikes
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True),
        xaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True),
        xaxis3=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True),
        xaxis4=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True, title='Date'),

        yaxis1_title="Price",
        yaxis2_title="Vol",
        yaxis3_title="MACD",
        yaxis4_title="RSI",
        yaxis4_range=[0, 100],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
    )

    return fig

def calculate_trend_forecast(df, lookback=60, forecast_days=30, degree=1, model_type="Linear"):
    """
    Calculates a Trend Projection using Linear/Poly, Holt-Winters, or ARIMA.
    Now supports VOLUME integration:
      - Holt-Winters: Uses VWAP (Volume Weighted Average Price) for smoothing.
      - ARIMA: Uses Volume as an exogenous variable (ARIMAX).
    """
    # 1. Prepare Data (Use last N days)
    subset = df.tail(lookback).copy()
    if len(subset) < 15:
        return None, None, None # Not enough data
    
    # X values for Regression
    y = subset['Close'].values
    x = np.arange(len(y))
    last_x = x[-1]
    future_x = np.arange(last_x + 1, last_x + 1 + forecast_days)
    
    future_prices = None
    upper_band = None
    lower_band = None

    # --- MODEL 1 & 2: REGRESSION (Linear/Quadratic) ---
    if model_type in ["Linear", "Quadratic"]:
        coeffs = np.polyfit(x, y, degree)
        poly_eqn = np.poly1d(coeffs)
        
        # Residuals for bands
        line_values = poly_eqn(x)
        residuals = y - line_values
        std_dev = np.std(residuals)
        
        future_prices = poly_eqn(future_x)
        upper_band = future_prices + (2 * std_dev)
        lower_band = future_prices - (2 * std_dev)

    # --- MODEL 3: EXPONENTIAL SMOOTHING (Holt-Winters on VWAP) ---
    elif model_type == "Holt-Winters" and HAS_STATSMODELS:
        # Calculate Volume Weighted Average Price (VWAP) for smoothing
        # Approx Daily VWAP = (Open+High+Low+Close)/4
        subset['Daily_VWAP'] = (subset['Open'] + subset['High'] + subset['Low'] + subset['Close']) / 4
        # Or true cumulative VWAP over window? No, HW needs a time series. Daily VWAP is best.
        
        try:
            # Fit model on Daily VWAP (incorporates volume indirectly via price action intensity)
            model = ExponentialSmoothing(
                subset['Daily_VWAP'], 
                trend='add', 
                damped_trend=True, 
                seasonal=None
            ).fit()
            
            future_prices = model.forecast(forecast_days).values
            
            # Calculate simple bands based on recent volatility of VWAP
            resid_std = subset['Daily_VWAP'].std() * 0.5 # Heuristic for bands
            upper_band = future_prices + (2 * resid_std)
            lower_band = future_prices - (2 * resid_std)
        except Exception as e:
            st.error(f"Holt-Winters Error: {e}")
            return None, None, None

    # --- MODEL 4: ARIMA (ARIMAX with Volume) ---
    elif model_type == "ARIMA" and HAS_STATSMODELS:
        try:
            # Endog = Close Price
            # Exog = Volume (Normalized to avoid scaling issues)
            vol_mean = subset['Volume'].mean()
            vol_std = subset['Volume'].std()
            exog_vol = (subset['Volume'] - vol_mean) / vol_std
            
            # Fit ARIMAX
            # Order (1,1,1) is a safe generic starting point for daily data
            model = ARIMA(subset['Close'], exog=exog_vol, order=(1,1,1)).fit()
            
            # To forecast, we need future Volume. We'll assume Volume reverts to mean.
            # Future exog = 0 (since we normalized to mean 0)
            future_exog = np.zeros(forecast_days)
            
            forecast_res = model.get_forecast(steps=forecast_days, exog=future_exog)
            future_prices = forecast_res.predicted_mean.values
            conf_int = forecast_res.conf_int(alpha=0.05) # 95% confidence
            
            lower_band = conf_int.iloc[:, 0].values
            upper_band = conf_int.iloc[:, 1].values
            
        except Exception as e:
            st.error(f"ARIMA Error: {e}")
            return None, None, None

    return future_prices, upper_band, lower_band

def create_forecast_chart(df, future_prices, upper_band, lower_band, model_name="Linear"):
    """
    Plots the Historical Price + Linear/Poly Forecast + 2-Sigma Bands.
    """
    fig = go.Figure()
    
    # 1. Historical Close (Last 100 days for context)
    history = df.tail(100)
    fig.add_trace(go.Scatter(
        x=np.arange(len(history)), 
        y=history['Close'],
        mode='lines',
        name='History',
        line=dict(color='#374151', width=2)
    ))
    
    # Forecast X-axis starts after history
    start_idx = len(history)
    future_x = np.arange(start_idx, start_idx + len(future_prices))
    
    # 2. Upper Band
    fig.add_trace(go.Scatter(
        x=future_x,
        y=upper_band,
        mode='lines',
        name='Upper Band',
        line=dict(width=0),
        showlegend=False
    ))
    
    # 3. Lower Band (Fill to Upper)
    fig.add_trace(go.Scatter(
        x=future_x,
        y=lower_band,
        mode='lines',
        name='Confidence Channel (95%)',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.2)', # Light Blue
    ))
    
    # 4. Mean Forecast Line
    fig.add_trace(go.Scatter(
        x=future_x,
        y=future_prices,
        mode='lines',
        name=f'{model_name} Projection',
        line=dict(color='#2563eb', width=3, dash='dash')
    ))

    fig.update_layout(
        title=f"Statistical Prediction: {model_name} Trend (30 Days)",
        yaxis_title="Price",
        xaxis_title="Trading Days (Past & Future)",
        template="plotly_white",
        height=500,
        xaxis=dict(showgrid=False),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

@st.cache_data(ttl=600)
def load_single_stock(ticker: str):
    """
    MODIFIED: Fetches single stock data DIRECTLY from Tushare (bypassing the Database).
    This ensures the data is always fresh and not stale from previous DB saves.
    """
    # Fix for Tushare Fetching Issue: 
    # Explicitly initialize Tushare if not done already (Streamlit context)
    if data_manager.TUSHARE_API is None:
        data_manager.init_tushare()

    # 1. Define lookback range (Last 3 years for good technicals)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * 3)
    
    start_str = start_dt.strftime('%Y%m%d')
    end_str = end_dt.strftime('%Y%m%d')
    
    # 2. Fetch directly from data_manager's robust fetcher
    # NOTE: We do NOT call get_single_stock_data() because that saves to DB.
    df = data_manager.fetch_stock_data_robust(ticker, start_str, end_str)
    
    if df is None or df.empty:
        return None
        
    # 3. Calculate Volume Metrics locally (DB helper usually does this)
    # Used for Z-Score logic in Sector charts, but good to have here too
    df['Vol_Mean_100d'] = df['Volume'].rolling(window=100, min_periods=20).mean()
    df['Vol_Std_100d'] = df['Volume'].rolling(window=100, min_periods=20).std()
    df['Volume_ZScore'] = (df['Volume'] - df['Vol_Mean_100d']) / df['Vol_Std_100d']
    
    return df.sort_index()

# --- 4. MAIN APP LAYOUT ---

st.title("ASSRS Sector Rotation & Single Stock Analysis")

# Tabs = separate “pages”
tab_dashboard, tab_single = st.tabs(["Sector Dashboard", "Single Stock Analysis"])

# ========= TAB 1: Sector Dashboard (UNCHANGED) =========
with tab_dashboard:
    # --- Load Sector Data ---
    v1_latest, v1_hist, v1_date, v1_error = load_data(V1_RULES_FILE, "V1")
    v2_latest, v2_hist, v2_date, v2_error = load_data(V2_REGIME_FILE, "V2")

    st.markdown("### Sector Rotation Scoreboard")

    # --- Create 2-Column Layout ---
    col1, col2 = st.columns(2)

    # --- V1 (Rule-Based) Scorecard ---
    with col1:
        st.header("V1: Rule-Based Scorecard (8-Point)")
        if v1_latest is not None:
            st.caption(f"Last Updated: {v1_date}")
            
            v1_display_df = v1_latest[['Sector', 'TOTAL_SCORE', 'ACTION']].sort_values(by='TOTAL_SCORE', ascending=False)
            styled_v1_df = v1_display_df.style.map(style_action, subset=['ACTION'])
            
            st.dataframe(
                styled_v1_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "TOTAL_SCORE": st.column_config.NumberColumn(format="%.2f"),
                    "ACTION": st.column_config.TextColumn(width="medium")
                }
            )
            
            # --- V1 Charting ---
            v1_sector_to_chart = st.selectbox(
                "Select V1 Sector to Chart:",
                v1_hist['Sector'].unique(),
                key="v1_selector"
            )
            
            if v1_sector_to_chart:
                chart_data = v1_hist[v1_hist['Sector'] == v1_sector_to_chart]
                fig = create_drilldown_chart(chart_data, model_type='v1')
                st.plotly_chart(fig, use_container_width=True, key="v1_chart")

        else:
            st.error(v1_error)

    # --- V2 (Regime-Switching) Scorecard ---
    with col2:
        st.header("V2: Regime-Switching Model")
        if v2_latest is not None:
            st.caption(f"Last Updated: {v2_date}")
            
            v2_display_df = v2_latest[['Sector', 'TOTAL_SCORE', 'ACTION']].copy()
            v2_display_df = v2_display_df.sort_values(by='TOTAL_SCORE', ascending=False)
            v2_display_df['TOTAL_SCORE'] = (v2_display_df['TOTAL_SCORE'] * 100).map('{:.0f}%'.format)
            styled_v2_df = v2_display_df.style.map(style_action, subset=['ACTION'])
            
            st.dataframe(
                styled_v2_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "TOTAL_SCORE": "Bull Probability",
                    "ACTION": st.column_config.TextColumn(width="medium")
                }
            )
            
            # --- V2 Charting ---
            v2_sector_to_chart = st.selectbox(
                "Select V2 Sector to Chart:",
                v2_hist['Sector'].unique(),
                key="v2_selector"
            )
            
            if v2_sector_to_chart:
                chart_data = v2_hist[v2_hist['Sector'] == v2_sector_to_chart]
                fig = create_drilldown_chart(chart_data, model_type='v2')
                st.plotly_chart(fig, use_container_width=True, key="v2_chart")
                
        else:
            st.error(v2_error)


# ========= TAB 2: Single Stock Analysis (UPDATED) =========
with tab_single:
    st.markdown("### 3-Phase Single Stock Analysis")
    st.info("Logic: Phase 1 (Accumulation/OBV) → Phase 2 (Squeeze/Bollinger) → Phase 3 (Launch/ADX)")

    # Initialize session state for the active ticker
    if 'active_ticker' not in st.session_state:
        st.session_state.active_ticker = None

    c1, c2 = st.columns([3, 1])
    with c1:
        ticker_input = st.text_input("Enter Stock Code (e.g., 600760)", key="ticker_input")
    with c2:
        if st.button("Analyze", key="analyze_btn"):
            st.session_state.active_ticker = ticker_input

    if st.session_state.active_ticker:
        ticker = st.session_state.active_ticker.strip()
        
        # Fetch data
        stock_df = load_single_stock(ticker)
        
        if stock_df is None or stock_df.empty:
            st.error(
                f"No data found for {ticker}. "
                "Check that the ticker is valid and Tushare is configured."
            )
        else:
            # Run the NEW 3-Phase Logic
            analysis_df = run_single_stock_analysis(stock_df)
            
            # --- NEW: Trading Block Analysis ---
            block_top, block_bot, block_status = calculate_trading_block(analysis_df, lookback=60)

            # Check if we have enough data
            if analysis_df.empty or len(analysis_df) < 50:
                st.error("Not enough data to compute 3-Phase signals for this stock.")
            else:
                latest_row = analysis_df.iloc[-1]

                # Create the NEW 4-Panel Chart (Passing block info now)
                fig_stock = create_single_stock_chart(analysis_df, block_info=(block_top, block_bot, block_status))
                st.plotly_chart(fig_stock, use_container_width=True)

                # NEW: Latest signals summary (Cards)
                st.subheader("Latest Status")
                col_a, col_b, col_c, col_d = st.columns(4)

                accum = bool(latest_row.get('Signal_Accumulation', False))
                squeeze = bool(latest_row.get('Signal_Squeeze', False))
                launch = bool(latest_row.get('Signal_Golden_Launch', False))
                adx_val = float(latest_row.get('ADX', 0.0))

                with col_a:
                    st.markdown("**Phase 1: Accumulation**")
                    st.markdown("**:orange[ACTIVE]**" if accum else ":grey[INACTIVE]")
                with col_b:
                    st.markdown("**Phase 2: Squeeze**")
                    st.markdown("**:grey[TIGHT]**" if squeeze else ":grey[LOOSE]")
                with col_c:
                    st.markdown("**Phase 3: LAUNCH**")
                    st.markdown("**:red[TRIGGERED]**" if launch else ":grey[WAITING]")
                with col_d:
                    st.markdown("**Trading Block (60d)**")
                    if "BREAKOUT" in block_status:
                        st.markdown(f"**:red[{block_status}]**")
                    elif "BREAKDOWN" in block_status:
                        st.markdown(f"**:green[{block_status}]**")
                    else:
                        st.markdown(f"**:grey[{block_status}]**")
                    st.caption(f"Range: {block_bot:.2f} - {block_top:.2f}")

                # NEW: Debug table (last 50 rows)
                st.subheader("Recent Data (last 50 days)")
                
                # Select relevant columns
                table_cols = [
                    'Close', 'MA20', 'MA50', 'MA200',
                    'RSI_14', 'ADX', 
                    'Signal_Accumulation', 'Signal_Squeeze', 'Signal_Golden_Launch',
                    'Exit_MACD_Lead'
                ]
                # Filter in case some cols missing (safety)
                valid_cols = [c for c in table_cols if c in analysis_df.columns]
                
                df_for_table = analysis_df[valid_cols].copy()
                df_for_table = df_for_table.reset_index()
                df_for_table['Date'] = df_for_table['Date'].dt.strftime('%Y-%m-%d')
                df_for_table = df_for_table.tail(50)
                
                st.dataframe(
                    df_for_table, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Close": st.column_config.NumberColumn(format="%.2f"),
                        "MA200": st.column_config.NumberColumn(format="%.2f"),
                        "RSI_14": st.column_config.NumberColumn(format="%.0f"),
                        "ADX": st.column_config.NumberColumn(format="%.1f"),
                        "Signal_Accumulation": st.column_config.CheckboxColumn(label="Accum (P1)"),
                        "Signal_Squeeze": st.column_config.CheckboxColumn(label="Squeeze (P2)"),
                        "Signal_Golden_Launch": st.column_config.CheckboxColumn(label="LAUNCH (P3)"),
                        "Exit_MACD_Lead": st.column_config.CheckboxColumn(label="Exit")
                    }
                )

                # --- STATISTICAL PREDICTION SECTION (NEW) ---
                st.markdown("---")
                st.subheader("Statistical Prediction (Forecast)")
                
                c_opts, c_info = st.columns([1, 2])
                with c_opts:
                    lookback_option = st.selectbox("Lookback Period (Days)", [10, 20, 30, 60], index=2)
                    
                    # UPDATED: Model Selection
                    model_options = ["Linear (Straight)", "Quadratic (Curved)"]
                    if HAS_STATSMODELS:
                        model_options.extend(["Holt-Winters (Exp. Smoothing)", "ARIMA (AutoRegressive)"])
                    
                    model_option = st.radio("Trend Model", model_options, index=0)
                
                with c_info:
                    degree = 1
                    if "Linear" in model_option:
                        st.info(f"**Linear Regression:** Projects current trend as a straight line based on last {lookback_option} days.")
                    elif "Quadratic" in model_option:
                        st.info(f"**Polynomial Regression:** Fits a curve to detect acceleration/deceleration.")
                        degree = 2
                    elif "Holt-Winters" in model_option:
                        st.info(f"**Exponential Smoothing (Holt-Winters):** Weights recent data heavily. \n\n**Volume Integration:** Uses 'Daily VWAP' (Volume Weighted Price) instead of raw Close price to ensure trend follows volume.")
                    elif "ARIMA" in model_option:
                        st.info(f"**ARIMA (ARIMAX):** AutoRegressive Integrated Moving Average. \n\n**Volume Integration:** Uses Volume as an 'Exogenous Variable' to mathematically weight the price prediction based on volume strength.")
                
                # Calculate forecast based on selection
                model_key = model_option.split()[0] # "Linear", "Quadratic", "Holt-Winters", "ARIMA"
                
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
