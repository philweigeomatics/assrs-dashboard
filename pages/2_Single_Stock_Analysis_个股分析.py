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
from datetime import date
from scipy import stats
from scipy.signal import find_peaks
from streamlit_plotly_events import plotly_events

from analysis_engine import run_single_stock_analysis, simulate_next_day_indicators

import auth_manager
auth_manager.require_login()


# ==================== SIGNAL DEFINITIONS ( should be the same as Todays Alerts ) ====================

BULLISH_SIGNALS = {
    'MACD_Bottoming': 'MACD Bottoming',
    'MACD_ClassicCrossover': 'MACD Positive Crossover',
    'RSI_Bottoming': 'RSI Bottoming',
    'Squeeze_Fired_Bullish': 'Bullish Squeeze Breakout'  # <--- ADD THIS
}

BEARISH_SIGNALS = {
    'MACD_Peaking': 'MACD Peaking',
    'MACD_BearishCrossover': 'MACD Bearish Crossover',
    'RSI_Peaking': 'RSI Peaking',
    'Squeeze_Fired_Bearish': 'Bearish Squeeze Drop'     # <--- ADD THIS
}

ADX_BULLISH_PATTERNS = {
    'Bottoming + Downtrend': 'ADX Bottoming (after decline)',
    'Reversing Up + Downtrend': 'ADX Reversing Up (after decline)',
}

ADX_BEARISH_PATTERNS = {
    'Peaking + Uptrend': 'ADX Peaking (after rally)',
    'Reversing Down + Uptrend': 'ADX Reversing Down (after rally)',
}



st.set_page_config(
    page_title="📈 Single Stock | 个股分析",
    page_icon="📈",
    layout="wide"
)

# # ensure the stock basic is updated daily.
# try:
#     data_manager.ensure_stock_basic_updated()
# except Exception as e:
#     pass

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

@st.cache_data(ttl=60)
def load_single_stock(ticker, cache_date):
    """Load stock data LIVE from Tushare API (qfq adjusted, no database)."""
    return data_manager.get_single_stock_data_live(ticker, lookback_years=3)



def backtest_signal_expectancy(analysis_df, buy_signal_type, sell_signal_type):
    """
    Backtest a buy/sell signal strategy:
    - Buy 100 shares next day at open when buy signal detected
    - Sell ALL shares next day at open when sell signal detected
    - If already holding, buy 100 more shares on new buy signal
    - If no position, ignore sell signals
    
    Handles both boolean signals (MACD, RSI) and ADX pattern signals with price context
    
    Returns:
    - trades_df: DataFrame with all transactions
    - summary: Dictionary with performance metrics
    """
    df = analysis_df.copy()
    
    # Add next day's open for execution
    df['Next_Open'] = df['Open'].shift(-1)
    
    # Calculate price trend for ADX patterns
    ema_5d = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_5d'] = ema_5d
    df['EMA_5d_prev'] = df['EMA_5d'].shift(1)
    
    # Determine price trend
    df['Price_Trend'] = 'neutral'
    df.loc[(df['Close'] > df['EMA_5d']) & (df['EMA_5d'] > df['EMA_5d_prev']), 'Price_Trend'] = 'uptrend'
    df.loc[(df['Close'] < df['EMA_5d']) & (df['EMA_5d'] < df['EMA_5d_prev']), 'Price_Trend'] = 'downtrend'
    
    # Initialize
    trades = []
    position = 0  # shares held
    total_cost = 0.0  # total cost basis
    
    for idx, row in df.iterrows():
        # Check buy signal
        buy_signal = False
        
        if buy_signal_type in BULLISH_SIGNALS:
            # Boolean signal (MACD, RSI)
            buy_signal = row.get(buy_signal_type, False)
        elif buy_signal_type in ADX_BULLISH_PATTERNS:
            # ADX pattern with price context
            adx_pattern = str(row.get('ADX_Pattern', ''))
            price_trend = row.get('Price_Trend', 'neutral')
            
            if buy_signal_type == 'Bottoming + Downtrend':
                buy_signal = (adx_pattern == 'Bottoming' and price_trend == 'downtrend')
            elif buy_signal_type == 'Reversing Up + Downtrend':
                buy_signal = (adx_pattern == 'Reversing Up' and price_trend == 'downtrend')
        
        # Check sell signal
        sell_signal = False
        
        if sell_signal_type in BEARISH_SIGNALS:
            # Boolean signal (MACD, RSI)
            sell_signal = row.get(sell_signal_type, False)
        elif sell_signal_type in ADX_BEARISH_PATTERNS:
            # ADX pattern with price context
            adx_pattern = str(row.get('ADX_Pattern', ''))
            price_trend = row.get('Price_Trend', 'neutral')
            
            if sell_signal_type == 'Peaking + Uptrend':
                sell_signal = (adx_pattern == 'Peaking' and price_trend == 'uptrend')
            elif sell_signal_type == 'Reversing Down + Uptrend':
                sell_signal = (adx_pattern == 'Reversing Down' and price_trend == 'uptrend')
        
        next_open = row['Next_Open']
        
        if pd.isna(next_open):
            continue
        
        # BUY LOGIC: Always buy 100 shares if signal triggered
        if buy_signal:
            shares_bought = 100
            cost = shares_bought * next_open
            position += shares_bought
            total_cost += cost
            
            trades.append({
                'Date': idx,
                'Action': 'BUY',
                'Shares': shares_bought,
                'Price': next_open,
                'Amount': cost,
                'Position': position,
                'Total_Cost': total_cost
            })
        
        # SELL LOGIC: Sell ALL shares if signal triggered AND holding position
        if sell_signal and position > 0:
            shares_sold = position
            proceeds = shares_sold * next_open
            
            # Calculate profit for this complete transaction
            avg_cost = total_cost / position
            profit = proceeds - total_cost
            profit_pct = (next_open - avg_cost) / avg_cost * 100
            
            trades.append({
                'Date': idx,
                'Action': 'SELL ALL',
                'Shares': shares_sold,
                'Price': next_open,
                'Amount': proceeds,
                'Position': 0,
                'Total_Cost': 0,
                'Profit': profit,
                'Profit_Pct': profit_pct,
                'Avg_Cost': avg_cost
            })
            
            # Reset position
            position = 0
            total_cost = 0.0
    
    if not trades:
        return None, None
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate complete transactions (from first BUY to SELL ALL)
    complete_transactions = []
    buy_group = []
    
    for _, trade in trades_df.iterrows():
        if trade['Action'] == 'BUY':
            buy_group.append(trade)
        elif trade['Action'] == 'SELL ALL':
            if buy_group:
                # This is a complete transaction
                first_buy_date = buy_group[0]['Date']
                sell_date = trade['Date']
                total_invested = sum(b['Amount'] for b in buy_group)
                total_shares = sum(b['Shares'] for b in buy_group)
                avg_buy_price = total_invested / total_shares
                sell_price = trade['Price']
                profit = trade['Profit']
                profit_pct = trade['Profit_Pct']
                
                complete_transactions.append({
                    'Entry_Date': first_buy_date,
                    'Exit_Date': sell_date,
                    'Shares': total_shares,
                    'Avg_Buy_Price': avg_buy_price,
                    'Sell_Price': sell_price,
                    'Profit': profit,
                    'Profit_Pct': profit_pct
                })
                
                buy_group = []  # Reset for next transaction
    
    if not complete_transactions:
        return trades_df, None
    
    trans_df = pd.DataFrame(complete_transactions)
    
    # Summary metrics
    summary = {
        'Total_Transactions': len(trans_df),
        'Winning_Trades': (trans_df['Profit'] > 0).sum(),
        'Losing_Trades': (trans_df['Profit'] < 0).sum(),
        'Win_Rate': (trans_df['Profit'] > 0).mean() * 100,
        'Avg_Profit_Pct': trans_df['Profit_Pct'].mean(),
        'Total_Profit': trans_df['Profit'].sum(),
        'Best_Trade_Pct': trans_df['Profit_Pct'].max(),
        'Worst_Trade_Pct': trans_df['Profit_Pct'].min(),
        'Median_Profit_Pct': trans_df['Profit_Pct'].median()
    }
    
    return trades_df, summary


def create_backtest_chart(trades_df, analysis_df):
    """
    Create a narrow chart showing price + buy/sell actions
    Uses FULL analysis_df period (not just last 250 days)
    """
    import plotly.graph_objects as go
    
    # Use FULL data period (no tail slicing)
    df = analysis_df.sort_index()
    dates = df.index.strftime('%Y-%m-%d').tolist()
    
    # Smart date ticks (show ~10-15 dates max for readability)
    total_dates = len(dates)
    if total_dates <= 50:
        tick_interval = 5
    elif total_dates <= 250:
        tick_interval = total_dates // 10
    elif total_dates <= 500:
        tick_interval = total_dates // 12
    else:
        tick_interval = total_dates // 15
    
    tick_vals = dates[::tick_interval]
    
    fig = go.Figure()
    
    # Price line (simple)
    fig.add_trace(go.Scatter(
        x=dates,
        y=df['Close'],
        name='Price',
        line=dict(color='#3b82f6', width=2),
        mode='lines'
    ))
    
    # Mark BUY signals
    buys = trades_df[trades_df['Action'] == 'BUY']
    if not buys.empty:
        buy_dates = buys['Date'].dt.strftime('%Y-%m-%d').tolist()
        buy_prices = buys['Price'].tolist()
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            name='BUY (100 shares)',
            mode='markers+text',
            marker=dict(color='#22c55e', size=12, symbol='triangle-up'),
            text=['▲'] * len(buys),
            textposition='bottom center',
            textfont=dict(size=14, color='#15803d')
        ))
    
    # Mark SELL signals
    sells = trades_df[trades_df['Action'] == 'SELL ALL']
    if not sells.empty:
        sell_dates = sells['Date'].dt.strftime('%Y-%m-%d').tolist()
        sell_prices = sells['Price'].tolist()
        sell_labels = [f"▼ {row['Profit_Pct']:.1f}%" for _, row in sells.iterrows()]
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            name='SELL ALL',
            mode='markers+text',
            marker=dict(color='#ef4444', size=12, symbol='triangle-down'),
            text=sell_labels,
            textposition='top center',
            textfont=dict(size=10, color='#991b1b')
        ))
    
    fig.update_layout(
        title=f'Signal-Based Trading Actions ({len(df)} days)',
        height=250,  # Narrow chart
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Price (¥)',
        hovermode='x unified',
        margin=dict(l=50, r=50, t=40, b=40),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Apply smart ticks
    fig.update_xaxes(
        type='category',
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_vals,
        tickangle=0  # Horizontal
    )
    
    return fig


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


def analyze_stock_personality(df: pd.DataFrame) -> dict:
    """
    Compute stock personality metrics:
      - Hurst Exponent (mean-reverting vs trending vs random walk)
      - Monthly seasonality: 10-day forward return by calendar month
      - Optimal dip-reversal setup: best drop threshold + avoid-months, ranked by Sharpe
    Expects df with columns: close, vol (Tushare naming).
    """
    import itertools

    results = {}

    # ── 1. Hurst Exponent (DFA — Detrended Fluctuation Analysis) ───────────────
    # DFA on price levels is the correct method for financial time series.
    # R/S on raw price levels (common mistake) always returns H≈1.0 regardless
    # of the true personality. DFA detrends each window before measuring fluctuation.
    prices = df['Close'].dropna().values
    min_lag, max_lag, n_lags = 10, max(20, len(prices) // 4), 20
    dfa_lags = np.unique(
        np.logspace(np.log10(min_lag), np.log10(max_lag), n_lags).astype(int)
    )
    dfa_lags = dfa_lags[dfa_lags >= min_lag]
    flucts, valid_lags = [], []
    for lag in dfa_lags:
        n_chunks = len(prices) // lag
        if n_chunks < 2:
            continue
        f_list = []
        for j in range(n_chunks):
            chunk = prices[j * lag:(j + 1) * lag]
            x = np.arange(len(chunk))
            trend = np.polyval(np.polyfit(x, chunk, 1), x)
            resid = chunk - trend
            f_list.append(np.sqrt(np.mean(resid ** 2)))
        flucts.append(np.mean(f_list))
        valid_lags.append(lag)
    if len(valid_lags) >= 4:
        hurst = float(np.polyfit(np.log(valid_lags), np.log(flucts), 1)[0])
    else:
        hurst = 0.5
    results['hurst'] = round(hurst, 4)
    if hurst < 0.45:
        results['hurst_label'] = 'Mean-Reverting'
        results['hurst_color'] = '#22c55e'
        results['hurst_advice'] = 'Dip-reversal strategies work best. Avoid trend-following.'
    elif hurst > 0.55:
        results['hurst_label'] = 'Trending / Momentum'
        results['hurst_color'] = '#3b82f6'
        results['hurst_advice'] = 'Momentum & breakout strategies work best. Avoid mean-reversion fades.'
    else:
        results['hurst_label'] = 'Random Walk'
        results['hurst_color'] = '#f59e0b'
        results['hurst_advice'] = 'Mixed personality — use tight risk controls. Both reversal and momentum have limited edge.'

    # ── 2. Monthly Seasonality (10-day forward return) ─────────────────────────
    df2 = df.copy()
    df2.index = pd.to_datetime(df2.index)
    df2['fwd10'] = df2['Close'].shift(-10) / df2['Close'] - 1
    df2['month'] = df2.index.month
    monthly = df2.groupby('month')['fwd10'].mean().dropna()
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    results['monthly_fwd'] = {month_names[m]: round(v * 100, 2) for m, v in monthly.items()}
    avoid_months_nums = monthly[monthly < 0].index.tolist()
    results['avoid_months'] = [month_names[m] for m in avoid_months_nums]
    strong_months_nums = monthly.nlargest(3).index.tolist()
    results['strong_months'] = [month_names[m] for m in strong_months_nums]

    # ── 3. Grid Search: best dip threshold + seasonal filter ───────────────────
    df3 = df.copy()
    df3.index = pd.to_datetime(df3.index)
    df3['ret']    = df3['Close'].pct_change()
    df3['month']  = df3.index.month

    # RSI-14 helper
    delta  = df3['Close'].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs_    = gain / loss.replace(0, np.nan)
    df3['rsi'] = 100 - 100 / (1 + rs_)

    best_sharpe = -np.inf
    best_params = {}
    best_trades = []

    drop_thresholds = [-0.02, -0.025, -0.03, -0.035, -0.04]
    avoid_combos    = [avoid_months_nums]  # use the negative-seasonality months

    for drop_thr, avoid_m in itertools.product(drop_thresholds, avoid_combos):
        trades = []
        in_trade = False
        entry_price = 0.0
        entry_idx   = 0
        commission  = 0.001
        slippage    = 0.002

        for i in range(20, len(df3) - 11):
            row = df3.iloc[i]
            if not in_trade:
                if (row['ret'] <= drop_thr and
                        row['rsi'] > 20 and
                        row['month'] not in avoid_m):
                    entry_price = row['Close'] * (1 + slippage)
                    in_trade    = True
                    entry_idx   = i
            else:
                days_held = i - entry_idx
                pnl = (row['Close'] / entry_price) - 1
                exit_reason = None
                if pnl >= 0.08:
                    exit_reason = 'profit'
                elif pnl <= -0.05:
                    exit_reason = 'stop'
                elif days_held >= 10:
                    exit_reason = 'time'
                elif row.get('rsi', 50) > 68:
                    exit_reason = 'rsi_exit'
                if exit_reason:
                    net = pnl - commission * 2 - slippage
                    trades.append(net)
                    in_trade = False

        if len(trades) >= 8:
            arr    = np.array(trades)
            sharpe = arr.mean() / arr.std() * np.sqrt(252 / 10) if arr.std() > 0 else 0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'drop_threshold': drop_thr, 'avoid_months': [month_names[m] for m in avoid_m]}
                best_trades = trades

    if best_trades:
        arr = np.array(best_trades)
        n   = len(arr)
        total_days = (df3.index[-1] - df3.index[0]).days
        years = total_days / 365.25
        cagr  = (1 + arr.sum()) ** (1 / years) - 1 if years > 0 else 0
        results['best_strategy'] = {
            **best_params,
            'sharpe':   round(best_sharpe, 2),
            'cagr_pct': round(cagr * 100, 1),
            'win_rate': round(np.sum(arr > 0) / n * 100, 1),
            'n_trades': n,
            'max_dd':   round(float(np.min(arr)) * 100, 1),
        }
    else:
        results['best_strategy'] = None

    return results


def create_single_stock_chart_analysis(
    df: pd.DataFrame,
    fundamentals_df: pd.DataFrame = None,
    blocks: list = None,
    comp_df: pd.DataFrame = None,
    comp_name: str = "Comparison",
    scale_mode: str = "pct",  # "pct" = Same % Scale  |  "new" = New Price Scale
) -> go.Figure:

    """
    Create 6-panel (or 7-panel when a comparison stock is active) chart.

    comp_df    : Optional price DataFrame for a second stock ('Close' col + DatetimeIndex).
    comp_name  : Legend label for the comparison stock.
    scale_mode : "pct" — both stocks rebased to the main stock's first price so they share
                          the same ¥ axis (TradingView "Same % Scale" equivalent).
                 "new" — comparison stock plotted on its own secondary y-axis with absolute
                          price (TradingView "New Price Scale").
    When a comparison stock is active a 7th panel is added showing relative performance
    (comparison % return minus main % return) as a green/red filled area.
    """
    df = df.tail(250).sort_index()
    dates = df.index.strftime('%Y-%m-%d').tolist()

    # Pre-align comparison data so has_comp is stable before make_subplots
    has_comp = False
    comp_aligned = pd.DataFrame()
    if comp_df is not None and not comp_df.empty and 'Close' in comp_df.columns:
        _cdf = comp_df.copy()
        _cdf.index = pd.to_datetime(_cdf.index)
        _aligned = _cdf[['Close']].reindex(df.index, method='ffill').dropna()
        if not _aligned.empty and _aligned['Close'].iloc[0] != 0:
            has_comp = True
            comp_aligned = _aligned

    # Extract the dynamic MACD parameters
    p_fast = int(df['MACD_Fast_Param'].iloc[-1]) if 'MACD_Fast_Param' in df.columns else 12
    p_slow = int(df['MACD_Slow_Param'].iloc[-1]) if 'MACD_Slow_Param' in df.columns else 26
    p_sign = int(df['MACD_Sign_Param'].iloc[-1]) if 'MACD_Sign_Param' in df.columns else 9

    n_rows   = 7 if has_comp else 6
    comp_row = 7  # relative-performance panel lives here when active

    _titles_base = (
        'Price & Trading Blocks + Signals',
        'Volume & OBV',
        f'MACD ({p_fast}, {p_slow}, {p_sign})',
        'RSI',
        'ADX Trend Analysis',
        'P/E Ratio',
    )
    _subplot_titles = _titles_base + (f'Relative Performance vs {comp_name}',) if has_comp else _titles_base

    _heights_base  = [0.40, 0.10, 0.18, 0.10, 0.18, 0.08]
    _row_heights   = _heights_base + [0.14] if has_comp else _heights_base[:-1] + [0.10]

    _specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_rows - 1)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=_subplot_titles,
        row_heights=_row_heights,
        specs=_specs,
    )

    # Build custom hover text once — used by the MACD trace in ROW 3
    macd_hover_text = [
        f"MACD ({fast}, {slow}, {sign}): {macd:.3f}"
        for fast, slow, sign, macd in zip(
            df['MACD_Fast_Param'], df['MACD_Slow_Param'],
            df['MACD_Sign_Param'], df['MACD']
        )
    ]

    # ====== ADD REGIME SHADING HERE (right after make_subplots, before any traces) ======
    
    regime_colors = {
        'Low Volatility': 'rgba(34, 197, 94, 0.08)',
        'Normal Volatility': 'rgba(59, 130, 246, 0.05)',
        'High Volatility': 'rgba(255, 110, 0, 0.11)',
        'Extreme Volatility': 'rgba(239, 68, 68, 0.16)'
    }
    
    if 'Market_Regime' in df.columns:
        
        # Remove NaN regimes first
        df_clean = df.dropna(subset=['Market_Regime'])

        
        if not df_clean.empty:
            # Find regime changes
            regime_changes = (df_clean['Market_Regime'] != df_clean['Market_Regime'].shift(1))
            change_indices = df_clean[regime_changes].index.tolist()

            # Add first index if not already there
            if len(change_indices) == 0 or change_indices[0] != df_clean.index[0]:
                change_indices.insert(0, df_clean.index[0])

            for i in range(len(change_indices)):
                start_idx = change_indices[i]
                end_idx = change_indices[i+1] if i+1 < len(change_indices) else df_clean.index[-1]
                
                regime = df_clean.loc[start_idx, 'Market_Regime']
                start_date = start_idx.strftime('%Y-%m-%d')
                end_date = end_idx.strftime('%Y-%m-%d')
                
                # Only draw if regime is valid
                if regime in regime_colors:
                    # Get the y-axis range for the price chart
                    y_min = df_clean['Low'].min() * 0.98
                    y_max = df_clean['High'].max() * 1.02
                    
                    # Add shape instead of vrect
                    fig.add_shape(
                        type="rect",
                        x0=start_date,
                        x1=end_date,
                        y0=y_min,
                        y1=y_max,
                        fillcolor=regime_colors[regime],
                        line=dict(width=0),
                        layer="below",
                        row=1, col=1
                    )
                    
                    # Add label logic - FIXED
                    should_label = False
                    if i == 0:
                        # First segment - always label it
                        should_label = True
                    elif i > 0 and regime != df_clean.loc[change_indices[i-1], 'Market_Regime']:
                        # Not first segment - only label if regime actually changed
                        should_label = True
                    
                    if should_label:
                        # Get y position for label
                        # Add label for each regime segment
                        segment_data = df_clean.loc[start_idx:end_idx]
                        y_label = df_clean['High'].max() * 1.01  # Even higher above the highest price

                        fig.add_annotation(
                            x=start_date,
                            y=y_label,
                            text=regime,
                            showarrow=False,
                            font=dict(size=9, color='black'),
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='gray',
                            borderwidth=1,
                            borderpad=2,
                            xanchor='left',
                            yanchor='bottom',  # Changed from top
                            row=1, col=1
                        )



    # ====== END OF REGIME SHADING ======
    
    # ==================== ROW 1: Price Chart (no changes) ====================
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=dates, y=df['BB_Upper'],
        line=dict(color='rgba(147,197,253,0.5)', width=1),
        name='BB Upper', showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=df['BB_Lower'],
        line=dict(color='rgba(147,197,253,0.5)', width=1),
        fill='tonexty', fillcolor='rgba(59,130,246,0.05)',
        name='BB Lower', showlegend=True
    ), row=1, col=1)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', showlegend=True,
        increasing=dict(line=dict(color='#ef4444')),
        decreasing=dict(line=dict(color='#22c55e'))
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=dates, y=df['EMA5'],
        name='EMA5',
        line=dict(color='#a855f7', width=1.5),
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=df['MA20'], name='MA20',
        line=dict(color='#fbbf24', dash='dot', width=1.5),
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=df['MA50'], name='MA50',
        line=dict(color='#3b82f6', width=2),
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates, y=df['MA200'], name='MA200',
        line=dict(color='#374151', width=2.5),
        showlegend=True
    ), row=1, col=1)

    
    
    # Trading Blocks
    if blocks:
        colors = ['rgba(255, 99, 71, 0.2)', 'rgba(255, 165, 0, 0.2)', 'rgba(255, 215, 0, 0.2)']
        recent_blocks = blocks[-3:]  # Get last 3 blocks
    
        for idx, block in enumerate(recent_blocks):
            color = colors[idx % len(colors)]
            color = colors[idx % len(colors)]
            start_date = block['start'].strftime('%Y-%m-%d')
            end_date = block['end'].strftime('%Y-%m-%d')
            
            fig.add_shape(
                type='rect', x0=start_date, x1=end_date,
                y0=block['bot'], y1=block['top'],
                fillcolor=color,
                line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'),
                row=1, col=1
            )
            
            fig.add_annotation(
                x=end_date, y=block['top'] * 1.02,
                text=f"Box {idx+1}<br>{block['bot']:.2f}-{block['top']:.2f}<br>{block['status']}",
                showarrow=True, arrowhead=2, arrowsize=1, ay=-30,
                font=dict(size=9, color='black'),
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='gray', borderwidth=1, borderpad=3,
                row=1, col=1
            )
    
    # Signal markers
    acc = df[df['Signal_Accumulation']]
    if not acc.empty:
        fig.add_trace(go.Scatter(
            x=acc.index.strftime('%Y-%m-%d'), y=acc['Low'] * 0.98,
            mode='markers', name='Phase 1: Accumulation',
            marker=dict(color='#eab308', size=10, symbol='circle'),
            showlegend=True
        ), row=1, col=1)
    
    sqz = df[df['Signal_Squeeze']]
    if not sqz.empty:
        fig.add_trace(go.Scatter(
            x=sqz.index.strftime('%Y-%m-%d'), y=sqz['High'] * 1.02,
            mode='markers', name='Phase 2: Squeeze',
            marker=dict(color='#64748b', size=8, symbol='square'),
            showlegend=True
        ), row=1, col=1)


    # NEW: Bullish Squeeze Breakout (Fired Up)
    bull_sqz = df[df.get('Squeeze_Fired_Bullish', False)]
    if not bull_sqz.empty:
        fig.add_trace(go.Scatter(
            x=bull_sqz.index.strftime('%Y-%m-%d'), 
            y=bull_sqz['Low'] * 0.95,  # Placed slightly below the candle
            mode='markers', 
            name='🚀 Bullish Squeeze Breakout',
            marker=dict(color='#10b981', size=14, symbol='triangle-up', line=dict(width=1, color='black')),
            showlegend=True
        ), row=1, col=1)

    # NEW: Bearish Squeeze Drop (Fired Down)
    bear_sqz = df[df.get('Squeeze_Fired_Bearish', False)]
    if not bear_sqz.empty:
        fig.add_trace(go.Scatter(
            x=bear_sqz.index.strftime('%Y-%m-%d'), 
            y=bear_sqz['High'] * 1.05,  # Placed slightly above the candle
            mode='markers', 
            name='🩸 Bearish Squeeze Drop',
            marker=dict(color='#ef4444', size=14, symbol='triangle-down', line=dict(width=1, color='black')),
            showlegend=True
        ), row=1, col=1)
    
    # launch = df[df['Signal_Golden_Launch']]
    # if not launch.empty:
    #     fig.add_trace(go.Scatter(
    #         x=launch.index.strftime('%Y-%m-%d'), y=launch['High'] * 1.05,
    #         mode='markers', name='⭐ GOLDEN LAUNCH',
    #         marker=dict(color='#ef4444', size=16, symbol='star',
    #                    line=dict(width=2, color='black')),
    #         showlegend=True
    #     ), row=1, col=1)

    ## ==================== VISUALIZE DI SCREAMING BREAKOUT ====================
    # if 'DI_Screaming_Buy' in df.columns:
    #     # Filter the dataframe for only the days where the signal fired
    #     screaming_buys = df[df['DI_Screaming_Buy'] == True]
        
    #     if not screaming_buys.empty:
    #         # Get the string formatted dates to match the x-axis
    #         buy_dates = screaming_buys.index.strftime('%Y-%m-%d').tolist()
            
    #         # Place the marker 2% below the low of the candle so it doesn't overlap the wick
    #         buy_prices = screaming_buys['Low'] * 0.98  
            
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=buy_dates,
    #                 y=buy_prices,
    #                 mode='markers+text',
    #                 marker=dict(
    #                     symbol='triangle-up', 
    #                     size=16,              
    #                     color='#FFD700',      # Bright Gold color
    #                     line=dict(color='black', width=1)
    #                 ),
    #                 name='DI Screaming Breakout',
    #                 text='🚀',                # Optional: Adds a rocket emoji right next to the arrow
    #                 textposition='bottom center',
    #                 hovertext='🚀 DI Screaming Breakout (Violent Buyer Expansion)',
    #                 hoverinfo='text'
    #             ),
    #             row=1, col=1
    #         )
    
    exits = df[df['Exit_MACD_Lead']]
    if not exits.empty:
        fig.add_trace(go.Scatter(
            x=exits.index.strftime('%Y-%m-%d'), y=exits['High'] * 1.01,
            mode='markers', name='Exit Signal',
            marker=dict(color='#22c55e', size=10, symbol='x'),
            showlegend=True
        ), row=1, col=1)
    
    # ==================== ROW 2: VOLUME WITH NORMALIZED OBV ====================
    # Volume bars
    colors_volume = ['#ef4444' if df.loc[idx, 'Close'] > df.loc[idx, 'Open'] 
                    else '#22c55e' for idx in df.index]

    fig.add_trace(go.Bar(
        x=dates, y=df['Volume'], name='Volume',
        marker=dict(color=colors_volume, opacity=0.7),
        showlegend=True
    ), row=2, col=1)

    # Volume-Scaled OBV - Normalized to Volume Scale
    if 'Volume_Scaled_OBV' in df.columns:
        # Scale OBV to match volume range for visibility
        obv_min = df['Volume_Scaled_OBV'].min()
        obv_max = df['Volume_Scaled_OBV'].max()
        vol_min = df['Volume'].min()
        vol_max = df['Volume'].max()
        
        # Normalize OBV to volume scale
        df['OBV_Scaled_Display'] = (
            (df['Volume_Scaled_OBV'] - obv_min) / (obv_max - obv_min) * 
            (vol_max - vol_min) + vol_min
        )
        
        fig.add_trace(go.Scatter(
            x=dates, y=df['OBV_Scaled_Display'], 
            name='Vol-Scaled OBV',
            line=dict(color='#f59e0b', width=3),
            mode='lines',
            showlegend=True,
            hovertemplate='OBV: %{customdata:.2f}<extra></extra>',  # Show real OBV value
            customdata=df['Volume_Scaled_OBV']  # Original values for hover
        ), row=2, col=1)



    # ==================== ROW 3: MACD WITH SCENARIO MARKERS ====================
    # MACD Histogram (Chinese colors: red=positive, green=negative)
    colors = ['#ef4444' if val > 0 else '#22c55e' for val in df['MACD_Hist']]

    fig.add_trace(go.Bar(
        x=dates, 
        y=df['MACD_Hist'] * 2.5,  # Scale up for visibility
        name='MACD Histogram (2.5x)',
        marker=dict(color=colors),
        showlegend=True
    ), row=3, col=1)

    # MACD Line (with dynamic param hover)
    fig.add_trace(go.Scatter(
        x=dates, y=df['MACD'], name='MACD',
        line=dict(color='#2563eb', width=2),
        showlegend=True,
        hoverinfo='text',
        hovertext=macd_hover_text,
    ), row=3, col=1)

    # Signal Line
    fig.add_trace(go.Scatter(
        x=dates, y=df['MACD_Signal'], name='MACD Signal',
        line=dict(color='#f97316', width=2),
        showlegend=True
    ), row=3, col=1)


    # Zero line
    fig.add_hline(y=0, line_dash='solid', line_color='gray', line_width=1, row=3, col=1)

    # ==================== MACD SCENARIO MARKERS (NEW) ====================
    # We'll add markers for ALL MACD triggers, labeled with the scenario

    # Create a combined scenario label
    def get_macd_scenario_label(row):
        """Returns the scenario name for labeling"""
        if row.get('MACD_ClassicCrossover', False):
            return 'Crossover'
        elif row.get('MACD_Approaching', False):
            return 'Approaching'
        elif row.get('MACD_Bottoming', False):
            return 'Bottoming'
        elif row.get('MACD_MomentumBuilding', False):
            return 'Momentum'
        elif row.get('MACD_Peaking', False):
            return 'Peaking'
        elif row.get('MACD_BearishCrossover', False):
            return 'Bear Cross'
        else:
            return None

    # Apply labeling
    if 'MACD_Trigger' in df.columns or 'MACD_Peaking' in df.columns or 'MACD_BearishCrossover' in df.columns:
        df['MACD_Scenario_Label'] = df.apply(get_macd_scenario_label, axis=1)
        
        # BULLISH signals
        macd_signals = df[df.get('MACD_Trigger', False) == True].copy()
        
        # BEARISH warnings
        macd_peaking = df[df.get('MACD_Peaking', False) == True].copy()
        macd_bearish = df[df.get('MACD_BearishCrossover', False) == True].copy()  # NEW
        
        # Plot bullish signals (existing code)
        if not macd_signals.empty:
            colors_map = {
                'Crossover': '#10b981',
                'Approaching': '#3b82f6',
                'Bottoming': '#f59e0b',
                'Momentum': '#ef4444'
            }
            marker_colors = [colors_map.get(label, '#6b7280') for label in macd_signals['MACD_Scenario_Label']]
            
            fig.add_trace(go.Scatter(
                x=macd_signals.index.strftime('%Y-%m-%d'),
                y=macd_signals['MACD'] * 1.12,
                mode='markers+text',
                name='MACD Triggers',
                marker=dict(color=marker_colors, size=12, symbol='circle', line=dict(width=2, color='white')),
                text=macd_signals['MACD_Scenario_Label'],
                textposition='top center',
                textfont=dict(size=9, color='black'),
                showlegend=True,
                hovertemplate='%{text}<br>MACD: %{y:.4f}<extra></extra>'
            ), row=3, col=1)
        
        # Plot PEAKING warnings (existing code)
        if not macd_peaking.empty:
            fig.add_trace(go.Scatter(
                x=macd_peaking.index.strftime('%Y-%m-%d'),
                y=macd_peaking['MACD'] * 1.15,
                mode='markers+text',
                name='MACD Peaking',
                marker=dict(color="#ffee00", size=12, symbol='triangle-down', line=dict(width=2, color="#000000")),
                text='',
                textposition='top center',
                textfont=dict(size=12, color='#dc2626'),
                showlegend=True,
                hovertemplate='<b>Peaking Warning</b><br>MACD: %{y:.4f}<extra></extra>'
            ), row=3, col=1)
        
        # ==================== NEW: Plot BEARISH CROSSOVER (strong exit signal) ====================
        if not macd_bearish.empty:
            fig.add_trace(go.Scatter(
                x=macd_bearish.index.strftime('%Y-%m-%d'),
                y=macd_bearish['MACD'] * 0.85,  # Position below MACD line
                mode='markers+text',
                name='Bearish Cross',
                marker=dict(color="#f30000", size=12, symbol='triangle-down', line=dict(width=2, color='#7f1d1d')),
                text='',
                textposition='top center',
                textfont=dict(size=14, color='#991b1b'),
                showlegend=True,
                hovertemplate='<b>Bearish Crossover</b><br>MACD: %{y:.4f}<extra></extra>'
            ), row=3, col=1)

    # ==================== BACKGROUND SHADING: LARGE TRENDS ONLY ====================
    if 'Large_Uptrend' in df.columns:
        # Group consecutive uptrend days
        uptrend_groups = (df['Large_Uptrend'] != df['Large_Uptrend'].shift()).cumsum()
        uptrend_df = df[df['Large_Uptrend']].copy()
        
        if not uptrend_df.empty:
            for group_id, group_data in uptrend_df.groupby(uptrend_groups[df['Large_Uptrend']]):
                if len(group_data) >= 5:  # Only shade sustained trends (5+ days)
                    start_date = group_data.index[0].strftime('%Y-%m-%d')
                    end_date = group_data.index[-1].strftime('%Y-%m-%d')
                    fig.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor="rgba(239, 68, 68, 0.08)",  # RED for uptrend
                        layer="below",
                        line_width=0,
                        row=3, col=1
                    )

    if 'Large_Downtrend' in df.columns:
        # Group consecutive downtrend days
        downtrend_groups = (df['Large_Downtrend'] != df['Large_Downtrend'].shift()).cumsum()
        downtrend_df = df[df['Large_Downtrend']].copy()
        
        if not downtrend_df.empty:
            for group_id, group_data in downtrend_df.groupby(downtrend_groups[df['Large_Downtrend']]):
                if len(group_data) >= 5:  # Only shade sustained trends (5+ days)
                    start_date = group_data.index[0].strftime('%Y-%m-%d')
                    end_date = group_data.index[-1].strftime('%Y-%m-%d')
                    fig.add_vrect(
                        x0=start_date, x1=end_date,
                        fillcolor="rgba(34, 197, 94, 0.08)",  # GREEN for downtrend
                        layer="below",
                        line_width=0,
                        row=3, col=1
                    )


    # ==================== ROW 4: RSI WITH DYNAMIC PERCENTILE ZONES (CHINESE STYLE) ====================

    
    # Main RSI line
    fig.add_trace(go.Scatter(
        x=dates, 
        y=df['RSI_14'],
        name='RSI(14)',
        line=dict(color='#1f2937', width=2.5),
        showlegend=True
    ), row=4, col=1)
        
    # ==================== MARK EXTREME RSI EVENTS ====================
    # Bottom 10% - Potential buying opportunity (green markers below)
    if 'RSI_Bottoming' in df.columns:
        rsi_bottoming = df[df['RSI_Bottoming'] == True].copy()
        if not rsi_bottoming.empty:
            fig.add_trace(go.Scatter(
                x=rsi_bottoming.index.strftime('%Y-%m-%d'),
                y=rsi_bottoming['RSI_14'] * 0.92,  # Position below RSI line
                mode='markers',
                name='🔵 RSI 极低 (P10)',
                marker=dict(color='#16a34a', size=10, symbol='triangle-up', 
                        line=dict(width=1, color='#15803d')),
                showlegend=True,
                hovertemplate='<b>RSI Bottoming</b><br>RSI: %{y:.1f}<extra></extra>'
            ), row=4, col=1)
    
    # Top 10% - Potential selling opportunity (red markers above)
    if 'RSI_Peaking' in df.columns:
        rsi_peaking = df[df['RSI_Peaking'] == True].copy()
        if not rsi_peaking.empty:
            fig.add_trace(go.Scatter(
                x=rsi_peaking.index.strftime('%Y-%m-%d'),
                y=rsi_peaking['RSI_14'] * 1.08,  # Position above RSI line
                mode='markers',
                name='🔴 RSI 极高 (P90)',
                marker=dict(color='#dc2626', size=10, symbol='triangle-down',
                        line=dict(width=1, color='#991b1b')),
                showlegend=True,
                hovertemplate='<b>RSI Peaking</b><br>RSI: %{y:.1f}<extra></extra>'
            ), row=4, col=1)
    
    
    # Reference lines
    fig.add_hline(y=50, line_dash='solid', line_color='#6b7280', line_width=2,
                annotation_text='中性 50', annotation_position='right',
                row=4, col=1)

    # 70 line (Overbought)
    fig.add_hline(
        y=70, 
        line_dash='dash', 
        line_color='#dc2626', 
        line_width=1.5,
        annotation_text='超买 70',
        annotation_position='right',
        row=4, col=1
    )

    # 30 line (Oversold)
    fig.add_hline(
        y=30, 
        line_dash='dash', 
        line_color='#16a34a', 
        line_width=1.5,
        annotation_text='超卖 30',
        annotation_position='right',
        row=4, col=1
    )

    
    # Update y-axis
    fig.update_yaxes(title_text='RSI', range=[0, 100], row=4, col=1)


    
    # ==================== ROW 5: NEW ADX PANEL WITH LABELS ====================
    # Raw ADX
    fig.add_trace(go.Scatter(
        x=dates, y=df['ADX'], name='ADX (Raw)',
        line=dict(color='rgba(100,116,139,0.6)', width=2),
        mode='lines+markers',
        marker=dict(size=3, color='rgba(100,116,139,0.6)'),
        showlegend=True
    ), row=5, col=1)

        # Add +DI line (green)
    if 'DI_Plus' in df.columns:
        fig.add_trace(go.Scatter(
            x=dates,
            y=df['DI_Plus'],
            name='+DI',
            line=dict(color='red', width=1.5)
        ), row=5, col=1)

    # Add -DI line (red)
    if 'DI_Minus' in df.columns:
        fig.add_trace(go.Scatter(
            x=dates,
            y=df['DI_Minus'],
            name='-DI',
            line=dict(color='green', width=1.5)
        ), row=5, col=1)

    # ==================== VISUALIZE DI SCREAMING BREAKOUT ON ADX ====================
    if 'DI_Screaming_Buy' in df.columns:
        screaming_buys = df[df['DI_Screaming_Buy'] == True]
        
        if not screaming_buys.empty:
            buy_dates = screaming_buys.index.strftime('%Y-%m-%d').tolist()
            
            # Place the marker exactly on the DI+ line so you see the buyer surge!
            buy_di_plus = screaming_buys['DI_Plus'] 
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_di_plus,
                    mode='markers+text',
                    marker=dict(
                        symbol='star',        # Changed to a star to highlight the indicator 
                        size=14,              
                        color='#FFD700',      # Bright Gold
                        line=dict(color='black', width=1)
                    ),
                    name='DI Screaming Breakout',
                    text='🚀',                
                    textposition='top center',
                    hovertext='🚀 DI Screaming Breakout (Violent Buyer Expansion)',
                    hoverinfo='text'
                ),
                row=5, col=1  # <--- Places it on the ADX panel!
            )

    # Add threshold line at 25
    fig.add_hline(y=25, line_dash="dot", line_color="gray", 
                annotation_text="Trend Threshold (25)", row=5, col=1)
    

    # ADX LOWESS (smooth)
    if 'ADX_LOWESS' in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df['ADX_LOWESS'], name='ADX LOWESS',
            line=dict(color='#10b981', width=3),
            showlegend=True
        ), row=5, col=1)

    # ADX Bollinger Bands
    if 'ADX_BB_Upper' in df.columns and 'ADX_BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df['ADX_BB_Upper'], name='ADX BB Upper',
            line=dict(color='#ef4444', width=1.5, dash='dash'),
            showlegend=True
        ), row=5, col=1)
        
        fig.add_trace(go.Scatter(
            x=dates, y=df['ADX_BB_Lower'], name='ADX BB Lower',
            line=dict(color='#ef4444', width=1.5, dash='dash'),
            fill='tonexty', fillcolor='rgba(239,68,68,0.1)',
            showlegend=True
        ), row=5, col=1)

    # ==================== ADD PATTERN LABELS ====================
    if 'ADX_Pattern' in df.columns:
        # Define specific colors for the continuous states
        # Red/Orange = Trend is building/strong. Green = Trend is dying.
        state_colors = {
            'Accelerating Up': '#ef4444',     # Bright Red (Explosive momentum)
            'Strong Trend': '#991b1b',        # Dark Red (Sustained momentum)
            'Losing Steam': '#f59e0b',        # Orange (Warning)
            'Accelerating Down': '#22c55e',   # Bright Green (Trend dying fast)
            'Slowing Down': "#62C7F7"         # Dark Green (Dead trend)
        }
        
        # Filter for only the continuous states
        ribbon_df = df[df['ADX_Pattern'].isin(state_colors.keys())].copy()
        
        if not ribbon_df.empty:
            ribbon_colors = [state_colors[val] for val in ribbon_df['ADX_Pattern']]
            
            fig.add_trace(go.Scatter(
                x=ribbon_df.index.strftime('%Y-%m-%d'),
                y=[2] * len(ribbon_df),  # Place firmly at the bottom (y=2) of the ADX panel
                mode='markers',
                marker=dict(
                    symbol='square',
                    size=8,              # Small, unobtrusive squares
                    color=ribbon_colors,
                    line=dict(width=0)
                ),
                name='Trend State Ribbon',
                hoverinfo='text',
                hovertext=[f"Trend State: {state}" for state in ribbon_df['ADX_Pattern']],
                showlegend=False
            ), row=5, col=1)


        # Mark Bottoming
        bottoming = df[df['ADX_Pattern'] == 'Bottoming']
        if not bottoming.empty:
            fig.add_trace(go.Scatter(
                x=bottoming.index.strftime('%Y-%m-%d'),
                y=bottoming['ADX'] * 0.95,
                mode='markers+text',
                name='🔄 Bottoming',
                marker=dict(color='#f59e0b', size=12, symbol='triangle-up'),
                text='🔄',
                textposition='bottom center',
                showlegend=True
            ), row=5, col=1)

        reversing_up = df[df['ADX_Pattern'] == 'Reversing Up']
        if not reversing_up.empty:
            fig.add_trace(go.Scatter(
                x=reversing_up.index.strftime('%Y-%m-%d'),
                y=reversing_up['ADX'] * 1.05,
                mode='markers+text',
                name='Reversing Up',
                marker=dict(color='#22c55e', size=12, symbol='triangle-up'),
                text='🔺',
                textposition='bottom center',
                showlegend=True
            ), row=5, col=1)
        
        # # Mark Accelerating Up
        # accelerating_up = df[df['ADX_Pattern'] == 'Accelerating Up']
        # if not accelerating_up.empty:
        #     fig.add_trace(go.Scatter(
        #         x=accelerating_up.index.strftime('%Y-%m-%d'),
        #         y=accelerating_up['ADX'] * 1.05,
        #         mode='markers+text',
        #         name='🚀 Accelerating',
        #         marker=dict(color='#ef4444', size=10, symbol='triangle-up'),
        #         text='🚀',
        #         textposition='top center',
        #         showlegend=True
        #     ), row=5, col=1)
        
        # # Mark Strong Trend
        # strong = df[df['ADX_Pattern'] == 'Strong Trend']
        # if not strong.empty:
        #     fig.add_trace(go.Scatter(
        #         x=strong.index.strftime('%Y-%m-%d'),
        #         y=strong['ADX'],
        #         mode='markers',
        #         name='💪 Strong',
        #         marker=dict(color="#ff0000", size=9, symbol='diamond', opacity=0.5),
        #         showlegend=True
        #     ), row=5, col=1)

        # # Losing Steam
        # losing_steam = df[df['ADX_Pattern'] == 'Losing Steam']
        # if not losing_steam.empty:
        #     fig.add_trace(go.Scatter(
        #         x=losing_steam.index.strftime('%Y-%m-%d'),
        #         y=losing_steam['ADX'],
        #         mode='markers',
        #         name='⚠️ Losing Steam',
        #         marker=dict(color='#f59e0b', size=9, symbol='diamond', opacity=0.5),
        #         showlegend=True
        #     ), row=5, col=1)
        
        # Mark Peaking (NEW - Warning!)
        peaking = df[df['ADX_Pattern'] == 'Peaking']
        if not peaking.empty:
            fig.add_trace(go.Scatter(
                x=peaking.index.strftime('%Y-%m-%d'),
                y=peaking['ADX'] * 1.08,
                mode='markers+text',
                name='🔴 Peaking',
                marker=dict(color="#ff1f01", size=14, symbol='triangle-down'),
                text='🔴',
                textposition='top center',
                showlegend=True
            ), row=5, col=1)
        
        # Mark Reversing Down (NEW - Danger!)
        reversing_down = df[df['ADX_Pattern'] == 'Reversing Down']
        if not reversing_down.empty:
            fig.add_trace(go.Scatter(
                x=reversing_down.index.strftime('%Y-%m-%d'),
                y=reversing_down['ADX'] * 1.05,
                mode='markers+text',
                name='🔻 Reversing',
                marker=dict(color="#00f85b", size=14, symbol='triangle-down'),
                text='🔻',
                textposition='top center',
                showlegend=True
            ), row=5, col=1)

        # accelerating_down = df[df['ADX_Pattern'] == 'Accelerating Down']
        # if not accelerating_down.empty:
        #     fig.add_trace(go.Scatter(
        #         x=accelerating_down.index.strftime('%Y-%m-%d'),
        #         y=accelerating_down['ADX'] * 1.05,
        #         mode='markers+text',
        #         name='📉 Accelerating Down',
        #         marker=dict(color="#025721", size=14, symbol='triangle-down'),
        #         text='📉',
        #         textposition='top center',
        #         showlegend=True
        #     ), row=5, col=1)

        # slowing_down = df[df['ADX_Pattern'] == 'Slowing Down']
        # if not slowing_down.empty:
        #     fig.add_trace(go.Scatter(
        #         x=slowing_down.index.strftime('%Y-%m-%d'),
        #         y=slowing_down['ADX'] * 1.05,
        #         mode='markers+text',
        #         name='🔽 Slowing Down',
        #         marker=dict(color="#3b82f6", size=14, symbol='triangle-down'),
        #         text='🔽',
        #         textposition='top center',
        #         showlegend=True
        #     ), row=5, col=1)



    # ── Comparison stock overlay + Relative Performance panel ───────────────────
    if has_comp:
        comp_dates  = comp_aligned.index.strftime('%Y-%m-%d').tolist()
        main_first  = df['Close'].iloc[0]
        comp_first  = comp_aligned['Close'].iloc[0]

        if scale_mode == "pct":
            # ── Same % Scale ──────────────────────────────────────────────────
            # Rebase comparison to main stock's first price so both sit on the
            # same ¥ axis.  The gap between the candlestick and this line is the
            # pure relative performance (same as TradingView "Same % Scale").
            comp_rebased = comp_aligned['Close'] / comp_first * main_first
            fig.add_trace(go.Scatter(
                x=comp_dates,
                y=comp_rebased.tolist(),
                name=f'{comp_name} (same % scale)',
                line=dict(color='#f97316', width=2),
                opacity=0.85,
                hovertemplate='%{x}<br>' + comp_name + ' (rebased ¥): %{y:.2f}<extra></extra>',
            ), row=1, col=1, secondary_y=False)
            # Hide unused secondary axis
            fig.update_yaxes(visible=False, secondary_y=True, row=1, col=1)

        else:
            # ── New Price Scale ───────────────────────────────────────────────
            # Comparison stock's actual price on its own right-hand y-axis.
            fig.add_trace(go.Scatter(
                x=comp_dates,
                y=comp_aligned['Close'].tolist(),
                name=f'{comp_name} (¥)',
                line=dict(color='#f97316', width=2),
                opacity=0.85,
                hovertemplate='%{x}<br>' + comp_name + ': ¥%{y:.2f}<extra></extra>',
            ), row=1, col=1, secondary_y=True)
            fig.update_yaxes(
                title_text=f'{comp_name} (¥)',
                secondary_y=True, row=1, col=1,
                showgrid=False,
                tickprefix='¥',
            )

        # ── Row 7: Relative Performance (comp % − main %) ─────────────────
        # Green = comparison outperforming, Red = main outperforming
        main_pct = (df['Close'] / main_first - 1) * 100
        comp_pct = (comp_aligned['Close'] / comp_first - 1) * 100
        rel      = comp_pct.reindex(main_pct.index).fillna(method='ffill') - main_pct

        rel_vals = rel.fillna(0)
        pos_vals = rel_vals.clip(lower=0)
        neg_vals = rel_vals.clip(upper=0)

        fig.add_trace(go.Scatter(
            x=dates, y=pos_vals.tolist(),
            fill='tozeroy', fillcolor='rgba(34,197,94,0.35)',
            line=dict(width=0), showlegend=False,
            hovertemplate='%{x}<br>+%{y:.1f} pp (comp ahead)<extra></extra>',
        ), row=comp_row, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=neg_vals.tolist(),
            fill='tozeroy', fillcolor='rgba(239,68,68,0.35)',
            line=dict(width=0), showlegend=False,
            hovertemplate='%{x}<br>%{y:.1f} pp (main ahead)<extra></extra>',
        ), row=comp_row, col=1)
        fig.add_hline(y=0, line_color='#9ca3af', line_dash='dash', row=comp_row, col=1)
        fig.update_yaxes(
            title_text='Outperf (pp)',
            ticksuffix=' pp',
            zeroline=False,
            showgrid=True,
            row=comp_row, col=1,
        )

    # Reference lines
    fig.add_hline(
        y=25, line_dash='dot', line_color='#6b7280',
        annotation_text='Strong Trend (25)',
        row=5, col=1
    )

    fig.add_hline(
        y=20, line_dash='dot', line_color='#d1d5db',
        annotation_text='Weak Trend (20)',
        row=5, col=1
    )


        # === ROW 6: P/E RATIO ===
    if fundamentals_df is not None and not fundamentals_df.empty:
        # Merge fundamentals with analysis dates
        fund_aligned = fundamentals_df.reindex(df.index, method='ffill')
        
        if 'PE_TTM' in fund_aligned.columns:
            pe_data = fund_aligned['PE_TTM'].dropna()
            if not pe_data.empty:
                pe_dates = pe_data.index.strftime('%Y-%m-%d').tolist()
                
                fig.add_trace(go.Scatter(
                    x=pe_dates, y=pe_data,
                    name='P/E (TTM)',
                    line=dict(color='#8b5cf6', width=2.5),
                    showlegend=True
                ), row=6, col=1)
                
                # Add reference lines
                pe_median = pe_data.median()
                fig.add_hline(
                    y=pe_median,
                    line_dash='dash',
                    line_color='gray',
                    annotation_text=f'Median: {pe_median:.1f}',
                    row=6, col=1
                )
    
    fig.update_yaxes(title_text='P/E', row=6, col=1)
    
    
    # ==================== UPDATE LAYOUT ====================
    # Y-axis index note: secondary_y=True on row 1 allocates yaxis2 for the
    # comparison stock's right-hand axis, pushing every subsequent panel up by 1:
    #   yaxis1  = Row 1 primary (Price)
    #   yaxis2  = Row 1 secondary (comparison stock — managed separately)
    #   yaxis3  = Row 2 (Volume)
    #   yaxis4  = Row 3 (MACD)
    #   yaxis5  = Row 4 (RSI)
    #   yaxis6  = Row 5 (ADX)
    #   yaxis7  = Row 6 (P/E  — managed by update_yaxes above)
    #   yaxis8  = Row 7 (Rel Perf, if has_comp — managed by update_yaxes above)
    _bottom_xaxis = f'xaxis{n_rows}_title'
    fig.update_layout(
        height=1500 + (150 if has_comp else 0),
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        **{_bottom_xaxis: 'Date'},
        yaxis1_title='Price (¥)',
        yaxis3_title='Volume',
        yaxis4_title='MACD',
        yaxis5_title='RSI',
        yaxis5_range=[0, 100],
        yaxis6_title='ADX',
        yaxis6_range=[0, 60],
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

    # Smart tick selection
    total_dates = len(dates)
    if total_dates <= 30:
        tick_interval = 1
    elif total_dates <= 60:
        tick_interval = 3
    elif total_dates <= 120:
        tick_interval = 5
    else:
        tick_interval = max(5, total_dates // 20)

    tick_vals = dates[::tick_interval]

    # Bottom row shows x-axis labels; all rows above it are hidden
    fig.update_xaxes(
        type='category',
        tickangle=-45,
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_vals,
        row=n_rows, col=1,
    )
    for row in range(1, n_rows):
        fig.update_xaxes(type='category', showticklabels=False, row=row, col=1)

    return fig


# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================

if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = None

def set_active_ticker(ticker: str):
    """Set active ticker (called from history or external links)."""
    st.session_state.active_ticker = ticker

def analyze_ticker():
    """Trigger analysis from the Analyze button — reads the combobox selection."""
    pick = (st.session_state.get("ssa_stock_pick") or "").strip()
    if pick:
        st.session_state.active_ticker = pick.split(" · ")[0].strip()


def analyze_return_distribution(df, ticker_name="Stock"):
    """
    Analyze return distribution for T+1 trading (buy today, sell tomorrow).

    Returns comprehensive risk metrics and visualizations including:
    - Return distribution histogram with normal curve overlay
    - Q-Q plot to assess normality
    - VaR and CVaR at 95% and 99% confidence levels
    - Distribution statistics (skewness, kurtosis, fat tail analysis)

    Args:
        df: DataFrame with 'Close' column and DatetimeIndex
        ticker_name: Name of the stock for display

    Returns:
        fig: Plotly figure with visualizations
        metrics_df: DataFrame with risk metrics
    """

    # Calculate daily returns
    returns = df['Close'].pct_change().dropna()

    if len(returns) < 30:
        return None, None

    # ========================================
    # RISK METRICS CALCULATION
    # ========================================

    # Basic statistics
    mean_return = returns.mean()
    std_return = returns.std()
    median_return = returns.median()

    # Distribution shape
    skewness = returns.skew()
    kurtosis = returns.kurtosis()  # Excess kurtosis (normal = 0)

    # VaR (Value at Risk) - Loss threshold at confidence level
    var_95 = returns.quantile(0.05)  # 5th percentile (95% confident loss won't exceed this)
    var_99 = returns.quantile(0.01)  # 1st percentile (99% confident)

    # CVaR (Conditional VaR / Expected Shortfall) - Average loss beyond VaR
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()

    # Upside potential
    upside_95 = returns.quantile(0.95)  # 95th percentile gain
    upside_99 = returns.quantile(0.99)  # 99th percentile gain

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    # Jarque-Bera test for normality (p < 0.05 means NOT normal)
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    is_normal = jb_pvalue > 0.05

    # Fat tail indicator (kurtosis > 3 indicates fat tails)
    is_fat_tail = kurtosis > 3

    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

    # Max single-day gain/loss
    max_gain = returns.max()
    max_loss = returns.min()

    # ========================================
    # CREATE METRICS DATAFRAME
    # ========================================

    metrics = {
        'Metric': [
            'Mean Daily Return',
            'Median Daily Return',
            'Std Dev (Daily)',
            'Annualized Volatility',
            'Sharpe Ratio (Annual)',
            '',  # Separator
            'Win Rate',
            'Max Single-Day Gain',
            'Max Single-Day Loss',
            '',  # Separator
            'VaR 95% (Daily)',
            'VaR 99% (Daily)',
            'CVaR 95% (Daily)',
            'CVaR 99% (Daily)',
            '',  # Separator
            'Upside 95th %tile',
            'Upside 99th %tile',
            '',  # Separator
            'Skewness',
            'Kurtosis (Excess)',
            'Distribution Type',
            'Fat Tail?',
            'Jarque-Bera p-value'
        ],
        'Value': [
            f"{mean_return*100:.3f}%",
            f"{median_return*100:.3f}%",
            f"{std_return*100:.3f}%",
            f"{std_return*np.sqrt(252)*100:.2f}%",
            f"{sharpe:.2f}",
            '',
            f"{win_rate*100:.1f}%",
            f"{max_gain*100:.2f}%",
            f"{max_loss*100:.2f}%",
            '',
            f"{var_95*100:.2f}%",
            f"{var_99*100:.2f}%",
            f"{cvar_95*100:.2f}%",
            f"{cvar_99*100:.2f}%",
            '',
            f"{upside_95*100:.2f}%",
            f"{upside_99*100:.2f}%",
            '',
            f"{skewness:.2f}",
            f"{kurtosis:.2f}",
            'Normal' if is_normal else 'Non-Normal',
            'Yes' if is_fat_tail else 'No',
            f"{jb_pvalue:.4f}"
        ],
        'Interpretation': [
            'Average T+1 return',
            '50th percentile return',
            'Daily volatility',
            'Annual volatility',
            'Risk-adjusted return',
            '',
            'Probability of profit',
            'Best case (historical)',
            'Worst case (historical)',
            '',
            '95% confidence max loss',
            '99% confidence max loss',
            'Avg loss when VaR95 breached',
            'Avg loss when VaR99 breached',
            '',
            'Top 5% gain threshold',
            'Top 1% gain threshold',
            '',
            'Neg=left skew, Pos=right skew',
            '>3 = fat tails, <0 = thin tails',
            'Based on Jarque-Bera test',
            'Extreme events more likely',
            'p<0.05 = Non-normal'
        ]
    }

    metrics_df = pd.DataFrame(metrics)

    # ========================================
    # CREATE VISUALIZATIONS
    # ========================================

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Return Distribution & Normal Curve',
            'Q-Q Plot (Normality Check)',
            'Return Time Series',
            'Risk Metrics Visualization'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # --- PLOT 1: Histogram with Normal Curve ---
    hist_counts, bin_edges = np.histogram(returns, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Actual distribution
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist_counts,
            name='Actual',
            marker=dict(color='rgba(59, 130, 246, 0.6)', line=dict(width=0)),
            showlegend=True
        ),
        row=1, col=1
    )

    # Fitted normal distribution
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_curve = stats.norm.pdf(x_range, mean_return, std_return)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_curve,
            name='Normal Fit',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )

    # Add VaR lines
    fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                  annotation_text="VaR 95%", row=1, col=1)
    fig.add_vline(x=var_99, line_dash="dash", line_color="red", 
                  annotation_text="VaR 99%", row=1, col=1)

    # --- PLOT 2: Q-Q Plot ---
    qq = stats.probplot(returns, dist="norm")

    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode='markers',
            name='Q-Q',
            marker=dict(color='rgba(59, 130, 246, 0.6)', size=4),
            showlegend=False
        ),
        row=1, col=2
    )

    # Reference line
    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=qq[1][1] + qq[1][0] * qq[0][0],
            mode='lines',
            name='Normal Line',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )

    # --- PLOT 3: Return Time Series ---
    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns * 100,
            mode='lines',
            name='Daily Return',
            line=dict(color='rgba(59, 130, 246, 0.8)', width=1),
            showlegend=False
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  opacity=0.5, row=2, col=1)

    # Highlight extreme losses
    extreme_losses = returns[returns <= var_99]
    fig.add_trace(
        go.Scatter(
            x=extreme_losses.index,
            y=extreme_losses * 100,
            mode='markers',
            name='Extreme Loss (>VaR99)',
            marker=dict(color='red', size=6),
            showlegend=True
        ),
        row=2, col=1
    )

    # --- PLOT 4: Risk Metrics Bar Chart ---
    risk_metrics_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%', 
                           'Max Loss', 'Mean', 'Upside 95%', 'Upside 99%', 'Max Gain']
    risk_metrics_values = [var_95*100, var_99*100, cvar_95*100, cvar_99*100,
                          max_loss*100, mean_return*100, upside_95*100, upside_99*100, max_gain*100]
    risk_colors = ['orange', 'red', 'darkred', 'darkred', 
                   'crimson', 'blue', 'green', 'darkgreen', 'limegreen']

    fig.add_trace(
        go.Bar(
            x=risk_metrics_labels,
            y=risk_metrics_values,
            marker=dict(color=risk_colors),
            showlegend=False,
            text=[f"{v:.2f}%" for v in risk_metrics_values],
            textposition='outside'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Daily Return", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)

    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)

    fig.update_xaxes(title_text="Metric", tickangle=-45, row=2, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title=dict(
            text=f"{ticker_name} - T+1 Return Distribution & Risk Analysis<br><sub>Data: {len(returns)} trading days</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        )
    )

    return fig, metrics_df


def analyze_realistic_t1_trading(df, ticker_name="Stock"):
    """
    Realistic T+1 trading analysis considering intraday prices.

    Scenarios analyzed:
    1. Buy at today's close, sell at tomorrow's HIGH (best case)
    2. Buy at today's close, sell at tomorrow's close (baseline)
    3. Buy at today's LOW (if you can catch dip), sell at tomorrow's HIGH (optimal)
    4. Buy at today's LOW, sell at tomorrow's close (realistic best)

    Args:
        df: DataFrame with OHLC data
        ticker_name: Stock name

    Returns:
        fig: Plotly figure
        scenarios_df: DataFrame with scenario analysis
        entry_signals_df: Conditional entry signals with realistic returns
    """

    # Calculate returns for different scenarios
    df_analysis = df.copy()

    # Scenario 1: Close-to-Close (traditional)
    df_analysis['T0_Close'] = df_analysis['Close']
    df_analysis['T1_Close'] = df_analysis['Close'].shift(-1)
    df_analysis['Return_Close_Close'] = (df_analysis['T1_Close'] / df_analysis['T0_Close']) - 1

    # Scenario 2: Close-to-High (sell at best intraday price tomorrow)
    df_analysis['T1_High'] = df_analysis['High'].shift(-1)
    df_analysis['Return_Close_High'] = (df_analysis['T1_High'] / df_analysis['T0_Close']) - 1

    # Scenario 3: Low-to-High (optimal - catch today's dip, sell tomorrow's peak)
    df_analysis['T0_Low'] = df_analysis['Low']
    df_analysis['Return_Low_High'] = (df_analysis['T1_High'] / df_analysis['T0_Low']) - 1

    # Scenario 4: Low-to-Close (realistic best - catch dip, sell tomorrow close)
    df_analysis['Return_Low_Close'] = (df_analysis['T1_Close'] / df_analysis['T0_Low']) - 1

    # Today's intraday opportunity (how much cheaper can you buy vs close?)
    df_analysis['T0_Discount'] = (df_analysis['T0_Close'] / df_analysis['T0_Low']) - 1

    # Tomorrow's intraday opportunity (how much higher vs close?)
    df_analysis['T1_Premium'] = (df_analysis['T1_High'] / df_analysis['T1_Close']) - 1

    # Today's return (for conditional analysis)
    df_analysis['Today_Return'] = df_analysis['Close'].pct_change()

    # Drop rows with missing data
    df_analysis = df_analysis.dropna()

    if len(df_analysis) < 50:
        return None, None, None

    # ========================================
    # SCENARIO COMPARISON
    # ========================================

    scenarios = {
        'Scenario': [
            'Close → Close',
            'Close → High (T+1)',
            'Low (T+0) → High (T+1)',
            'Low (T+0) → Close (T+1)'
        ],
        'Description': [
            'Traditional (close to close)',
            'Exit at best intraday price',
            'Perfect timing (buy dip, sell peak)',
            'Realistic best (buy dip, exit normal)'
        ],
        'Win Rate': [
            (df_analysis['Return_Close_Close'] > 0).mean(),
            (df_analysis['Return_Close_High'] > 0).mean(),
            (df_analysis['Return_Low_High'] > 0).mean(),
            (df_analysis['Return_Low_Close'] > 0).mean()
        ],
        'Avg Return': [
            df_analysis['Return_Close_Close'].mean(),
            df_analysis['Return_Close_High'].mean(),
            df_analysis['Return_Low_High'].mean(),
            df_analysis['Return_Low_Close'].mean()
        ],
        'Median Return': [
            df_analysis['Return_Close_Close'].median(),
            df_analysis['Return_Close_High'].median(),
            df_analysis['Return_Low_High'].median(),
            df_analysis['Return_Low_Close'].median()
        ],
        'Best Case': [
            df_analysis['Return_Close_Close'].max(),
            df_analysis['Return_Close_High'].max(),
            df_analysis['Return_Low_High'].max(),
            df_analysis['Return_Low_Close'].max()
        ],
        'Worst Case': [
            df_analysis['Return_Close_Close'].min(),
            df_analysis['Return_Close_High'].min(),
            df_analysis['Return_Low_High'].min(),
            df_analysis['Return_Low_Close'].min()
        ]
    }

    scenarios_df = pd.DataFrame(scenarios)

    # Calculate improvement vs baseline
    baseline_avg = scenarios_df.iloc[0]['Avg Return']
    scenarios_df['Improvement vs Baseline'] = scenarios_df['Avg Return'] - baseline_avg

    # ========================================
    # CONDITIONAL ENTRY WITH INTRADAY PRICES
    # ========================================

    drop_buckets = [
        (-0.01, 0.00, "0% to -1%"),
        (-0.02, -0.01, "-1% to -2%"),
        (-0.03, -0.02, "-2% to -3%"),
        (-0.04, -0.03, "-3% to -4%"),
        (-0.05, -0.04, "-4% to -5%"),
        (-1.00, -0.05, "< -5%"),
    ]

    entry_results = []

    for lower, upper, label in drop_buckets:
        mask = (df_analysis['Today_Return'] >= lower) & (df_analysis['Today_Return'] < upper)
        bucket_data = df_analysis[mask]

        if len(bucket_data) < 3:
            continue

        # Calculate returns for each scenario
        close_close = bucket_data['Return_Close_Close']
        close_high = bucket_data['Return_Close_High']
        low_high = bucket_data['Return_Low_High']
        low_close = bucket_data['Return_Low_Close']

        # Average intraday opportunities
        avg_t0_discount = bucket_data['T0_Discount'].mean()  # How much cheaper vs close
        avg_t1_premium = bucket_data['T1_Premium'].mean()    # How much higher vs close

        entry_results.append({
            'Entry Signal': label,
            'Sample Size': len(bucket_data),
            'Avg T0 Discount': avg_t0_discount,
            'Avg T1 Premium': avg_t1_premium,
            'Close→Close Return': close_close.mean(),
            'Close→High Return': close_high.mean(),
            'Low→High Return': low_high.mean(),
            'Low→Close Return': low_close.mean(),
            'Close→Close Win%': (close_close > 0).mean(),
            'Close→High Win%': (close_high > 0).mean(),
            'Low→High Win%': (low_high > 0).mean(),
            'Low→Close Win%': (low_close > 0).mean()
        })

    entry_signals_df = pd.DataFrame(entry_results)

    # ========================================
    # VISUALIZATIONS
    # ========================================

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Scenario Comparison: Average Returns',
            'Scenario Comparison: Win Rates',
            'Intraday Opportunity: T+0 Entry Discount',
            'Intraday Opportunity: T+1 Exit Premium',
            'Conditional Entry: Close→High Returns',
            'Conditional Entry: Low→Close Returns'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "box"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    # Plot 1: Average Returns by Scenario
    colors_scenarios = ['blue', 'green', 'orange', 'purple']
    fig.add_trace(
        go.Bar(
            x=scenarios_df['Scenario'],
            y=scenarios_df['Avg Return'] * 100,
            marker=dict(color=colors_scenarios),
            text=[f"{v:.2f}%" for v in scenarios_df['Avg Return'] * 100],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )

    # Plot 2: Win Rates by Scenario
    fig.add_trace(
        go.Bar(
            x=scenarios_df['Scenario'],
            y=scenarios_df['Win Rate'] * 100,
            marker=dict(color=colors_scenarios),
            text=[f"{v:.1f}%" for v in scenarios_df['Win Rate'] * 100],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=2)

    # Plot 3: T+0 Discount Distribution
    fig.add_trace(
        go.Box(
            y=df_analysis['T0_Discount'] * 100,
            name='T+0 Discount',
            marker=dict(color='rgba(59, 130, 246, 0.6)'),
            showlegend=False
        ),
        row=2, col=1
    )

    # Plot 4: T+1 Premium Distribution
    fig.add_trace(
        go.Box(
            y=df_analysis['T1_Premium'] * 100,
            name='T+1 Premium',
            marker=dict(color='rgba(34, 197, 94, 0.6)'),
            showlegend=False
        ),
        row=2, col=2
    )

    # Plot 5: Conditional Entry - Close to High
    if not entry_signals_df.empty:
        fig.add_trace(
            go.Bar(
                x=entry_signals_df['Entry Signal'],
                y=entry_signals_df['Close→High Return'] * 100,
                marker=dict(color='green'),
                text=[f"{v:.2f}%" for v in entry_signals_df['Close→High Return'] * 100],
                textposition='outside',
                showlegend=False
            ),
            row=3, col=1
        )

        # Plot 6: Conditional Entry - Low to Close
        fig.add_trace(
            go.Bar(
                x=entry_signals_df['Entry Signal'],
                y=entry_signals_df['Low→Close Return'] * 100,
                marker=dict(color='purple'),
                text=[f"{v:.2f}%" for v in entry_signals_df['Low→Close Return'] * 100],
                textposition='outside',
                showlegend=False
            ),
            row=3, col=2
        )

    # Update axes
    fig.update_yaxes(title_text="Avg Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Discount (%)", row=2, col=1)
    fig.update_yaxes(title_text="Premium (%)", row=2, col=2)
    fig.update_yaxes(title_text="Return (%)", row=3, col=1)
    fig.update_yaxes(title_text="Return (%)", row=3, col=2)

    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_xaxes(tickangle=-45, row=3, col=1)
    fig.update_xaxes(tickangle=-45, row=3, col=2)

    fig.update_layout(
        height=1000,
        showlegend=False,
        template='plotly_white',
        title=dict(
            text=f"{ticker_name} - Realistic T+1 Trading Analysis (Intraday Prices)<br><sub>Close vs Intraday Entry/Exit Comparison</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        )
    )

    return fig, scenarios_df, entry_signals_df


def calculate_intraday_stats(df):
    """Calculate summary statistics for intraday opportunities."""

    # T+0 opportunities (today's low vs close)
    t0_discount = (df['Close'] / df['Low']) - 1

    # T+1 opportunities (tomorrow's high vs close)
    t1_premium = (df['High'].shift(-1) / df['Close'].shift(-1)) - 1

    stats = {
        'T0_Avg_Discount': t0_discount.mean(),
        'T0_Median_Discount': t0_discount.median(),
        'T0_Max_Discount': t0_discount.max(),
        'T0_Days_Discount_1pct': (t0_discount > 0.01).sum(),
        'T0_Days_Discount_2pct': (t0_discount > 0.02).sum(),
        'T1_Avg_Premium': t1_premium.mean(),
        'T1_Median_Premium': t1_premium.median(),
        'T1_Max_Premium': t1_premium.max(),
        'T1_Days_Premium_1pct': (t1_premium > 0.01).sum(),
        'T1_Days_Premium_2pct': (t1_premium > 0.02).sum(),
        'Total_Days': len(df)
    }

    return stats



def analyze_down_day_bounce_probability(df, ticker_name="Stock"):
    """
    The missing piece: Analyze bounce probability and magnitude after down days.

    Answers:
    1. Given today is down X%, what's the probability tomorrow is UP?
    2. What's the expected magnitude of tomorrow's move?
    3. Does bigger drop = bigger bounce? (Mean reversion analysis)
    4. Risk/Reward: Is the expected gain worth the downside risk?

    Args:
        df: DataFrame with OHLC data
        ticker_name: Stock name

    Returns:
        fig: Plotly figure
        analysis_df: Detailed analysis by drop magnitude
        recommendation: Trading recommendation dict
    """

    # Calculate returns
    df_analysis = df.copy()
    df_analysis['Today_Return'] = df_analysis['Close'].pct_change()
    df_analysis['Tomorrow_Return'] = df_analysis['Today_Return'].shift(-1)

    # Also calculate realistic returns (considering intraday)
    df_analysis['Tomorrow_Close'] = df_analysis['Close'].shift(-1)
    df_analysis['Tomorrow_High'] = df_analysis['High'].shift(-1)
    df_analysis['Today_Low'] = df_analysis['Low']

    # Best case tomorrow return (if you sell at high)
    df_analysis['Tomorrow_Return_to_High'] = (df_analysis['Tomorrow_High'] / df_analysis['Close']) - 1

    # Best entry today (if you buy at low)
    df_analysis['Tomorrow_Return_from_Low'] = (df_analysis['Tomorrow_Close'] / df_analysis['Today_Low']) - 1

    df_analysis = df_analysis.dropna()

    if len(df_analysis) < 50:
        return None, None, None

    # ========================================
    # ANALYZE DOWN DAYS ONLY
    # ========================================

    # Filter only down days
    down_days = df_analysis[df_analysis['Today_Return'] < 0].copy()

    if len(down_days) < 20:
        return None, None, None

    # Categorize down days by magnitude
    down_buckets = [
        (0.00, -0.01, "0% to -1%"),
        (-0.01, -0.02, "-1% to -2%"),
        (-0.02, -0.03, "-2% to -3%"),
        (-0.03, -0.04, "-3% to -4%"),
        (-0.04, -0.05, "-4% to -5%"),
        (-0.05, -1.00, "< -5%"),
    ]

    results = []

    for upper, lower, label in down_buckets:  # Note: reversed order for down days
        mask = (down_days['Today_Return'] <= upper) & (down_days['Today_Return'] > lower)
        bucket_data = down_days[mask]

        if len(bucket_data) < 3:
            continue

        # Tomorrow's returns
        tmr_returns = bucket_data['Tomorrow_Return']
        tmr_returns_high = bucket_data['Tomorrow_Return_to_High']
        tmr_returns_low = bucket_data['Tomorrow_Return_from_Low']

        # Key metrics
        bounce_probability = (tmr_returns > 0).mean()  # Prob tomorrow is UP
        continuation_probability = (tmr_returns < 0).mean()  # Prob tomorrow is also DOWN

        # Expected returns
        avg_tmr_return = tmr_returns.mean()
        median_tmr_return = tmr_returns.median()

        # Conditional expected returns
        avg_if_bounce = tmr_returns[tmr_returns > 0].mean() if (tmr_returns > 0).any() else 0
        avg_if_continue = tmr_returns[tmr_returns < 0].mean() if (tmr_returns < 0).any() else 0

        # Realistic returns (with intraday)
        avg_tmr_return_high = tmr_returns_high.mean()
        avg_tmr_return_low = tmr_returns_low.mean()

        # Risk metrics
        worst_case = tmr_returns.min()
        best_case = tmr_returns.max()

        # Expected value calculation
        expected_value = avg_tmr_return

        # Risk/Reward ratio
        # If EV is positive: reward = avg_if_bounce, risk = abs(avg_if_continue)
        # Risk/Reward > 1 means good opportunity
        if avg_if_continue != 0:
            risk_reward = abs(avg_if_bounce / avg_if_continue)
        else:
            risk_reward = 0

        # Sharpe-like score (return / volatility)
        sharpe = avg_tmr_return / tmr_returns.std() if tmr_returns.std() > 0 else 0

        # Kelly Criterion (optimal position size)
        # Kelly = (p * b - q) / b, where p=win_prob, q=lose_prob, b=win/loss ratio
        if continuation_probability > 0 and avg_if_continue != 0:
            # this line below caused RunTimeError: divide by zero encountered in scalar divide
            kelly_pct = (bounce_probability * abs(avg_if_bounce/avg_if_continue) - continuation_probability) / abs(avg_if_bounce/avg_if_continue)  
            kelly_pct = max(0, min(kelly_pct, 1))  # Clamp between 0-100%
        else:
            kelly_pct = 0

        results.append({
            'Drop Magnitude': label,
            'Sample Size': len(bucket_data),
            'Bounce Probability': bounce_probability,
            'Continue Down Probability': continuation_probability,
            'Avg Tomorrow Return': avg_tmr_return,
            'Median Tomorrow Return': median_tmr_return,
            'Avg If Bounce': avg_if_bounce,
            'Avg If Continue': avg_if_continue,
            'Best Case Tomorrow': best_case,
            'Worst Case Tomorrow': worst_case,
            'Risk/Reward Ratio': risk_reward,
            'Sharpe-like Score': sharpe,
            'Expected Value': expected_value,
            'Kelly % (Position Size)': kelly_pct,
            'Avg Return (Close→High)': avg_tmr_return_high,
            'Avg Return (Low→Close)': avg_tmr_return_low
        })

    if not results:
        return None, None, None

    analysis_df = pd.DataFrame(results)

    # ========================================
    # MEAN REVERSION ANALYSIS
    # ========================================

    # Test: Does bigger drop lead to bigger bounce?
    correlation = down_days['Today_Return'].corr(down_days['Tomorrow_Return'])

    # Negative correlation = mean reversion (big drop → big bounce)
    # Positive correlation = momentum (big drop → continue down)

    mean_reversion_strength = abs(correlation) if correlation < 0 else 0
    momentum_strength = correlation if correlation > 0 else 0

    # ========================================
    # GENERATE RECOMMENDATION
    # ========================================

    # Find best opportunity (highest expected value with reasonable sample size)
    valid_entries = analysis_df[
        (analysis_df['Sample Size'] >= 5) &
        (analysis_df['Bounce Probability'] > 0.5) &
        (analysis_df['Expected Value'] > 0)
    ]

    if len(valid_entries) > 0:
        best_entry = valid_entries.loc[valid_entries['Expected Value'].idxmax()]

        recommendation = {
            'has_opportunity': True,
            'best_drop_range': best_entry['Drop Magnitude'],
            'bounce_prob': best_entry['Bounce Probability'],
            'expected_return': best_entry['Avg Tomorrow Return'],
            'risk_reward': best_entry['Risk/Reward Ratio'],
            'position_size': best_entry['Kelly % (Position Size)'],
            'avg_if_win': best_entry['Avg If Bounce'],
            'avg_if_lose': best_entry['Avg If Continue'],
            'sample_size': best_entry['Sample Size'],
            'mean_reversion': mean_reversion_strength,
            'momentum': momentum_strength
        }
    else:
        recommendation = {
            'has_opportunity': False,
            'mean_reversion': mean_reversion_strength,
            'momentum': momentum_strength
        }

    # ========================================
    # VISUALIZATIONS
    # ========================================

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Bounce Probability After Down Days',
            'Expected Tomorrow Return',
            'Risk/Reward Ratio by Drop Size',
            'Mean Reversion Analysis',
            'Kelly % Position Size',
            'Avg Bounce vs Avg Continue'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    # Plot 1: Bounce Probability
    colors_bounce = ['green' if p > 0.5 else 'red' for p in analysis_df['Bounce Probability']]
    fig.add_trace(
        go.Bar(
            x=analysis_df['Drop Magnitude'],
            y=analysis_df['Bounce Probability'] * 100,
            marker=dict(color=colors_bounce),
            text=[f"{v:.1f}%" for v in analysis_df['Bounce Probability'] * 100],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                  annotation_text="50% (Random)", row=1, col=1)

    # Plot 2: Expected Return
    colors_return = ['green' if r > 0 else 'red' for r in analysis_df['Avg Tomorrow Return']]
    fig.add_trace(
        go.Bar(
            x=analysis_df['Drop Magnitude'],
            y=analysis_df['Avg Tomorrow Return'] * 100,
            marker=dict(color=colors_return),
            text=[f"{v:.2f}%" for v in analysis_df['Avg Tomorrow Return'] * 100],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # Plot 3: Risk/Reward
    colors_rr = ['green' if rr > 1 else 'orange' if rr > 0.7 else 'red' 
                 for rr in analysis_df['Risk/Reward Ratio']]
    fig.add_trace(
        go.Bar(
            x=analysis_df['Drop Magnitude'],
            y=analysis_df['Risk/Reward Ratio'],
            marker=dict(color=colors_rr),
            text=[f"{v:.2f}" for v in analysis_df['Risk/Reward Ratio']],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray",
                  annotation_text="1:1 Break-even", row=2, col=1)

    # Plot 4: Mean Reversion Scatter
    fig.add_trace(
        go.Scatter(
            x=down_days['Today_Return'] * 100,
            y=down_days['Tomorrow_Return'] * 100,
            mode='markers',
            marker=dict(
                color='rgba(59, 130, 246, 0.4)',
                size=5
            ),
            name='Data Points',
            showlegend=False
        ),
        row=2, col=2
    )

    # Add trend line
    z = np.polyfit(down_days['Today_Return'], down_days['Tomorrow_Return'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(down_days['Today_Return'].min(), down_days['Today_Return'].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_line * 100,
            y=p(x_line) * 100,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Trend (ρ={correlation:.2f})',
            showlegend=True
        ),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", row=2, col=2)

    # Plot 5: Kelly Position Size
    fig.add_trace(
        go.Bar(
            x=analysis_df['Drop Magnitude'],
            y=analysis_df['Kelly % (Position Size)'] * 100,
            marker=dict(color='rgba(16, 185, 129, 0.7)'),
            text=[f"{v:.1f}%" for v in analysis_df['Kelly % (Position Size)'] * 100],
            textposition='outside',
            showlegend=False
        ),
        row=3, col=1
    )

    # Plot 6: Bounce vs Continue magnitudes
    x_labels = analysis_df['Drop Magnitude']

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=analysis_df['Avg If Bounce'] * 100,
            name='Avg Bounce',
            marker=dict(color='green'),
            text=[f"+{v:.2f}%" for v in analysis_df['Avg If Bounce'] * 100],
            textposition='outside'
        ),
        row=3, col=2
    )

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=analysis_df['Avg If Continue'] * 100,
            name='Avg Continue',
            marker=dict(color='red'),
            text=[f"{v:.2f}%" for v in analysis_df['Avg If Continue'] * 100],
            textposition='outside'
        ),
        row=3, col=2
    )

    # Update axes
    fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Avg Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Today's Drop (%)", row=2, col=2)
    fig.update_yaxes(title_text="Tomorrow's Return (%)", row=2, col=2)
    fig.update_yaxes(title_text="Position Size (%)", row=3, col=1)
    fig.update_yaxes(title_text="Return (%)", row=3, col=2)

    for i in range(1, 4):
        for j in range(1, 3):
            if not (i == 2 and j == 2):  # Skip scatter plot
                fig.update_xaxes(tickangle=-45, row=i, col=j)

    fig.update_layout(
        height=1100,
        showlegend=True,
        template='plotly_white',
        title=dict(
            text=f"{ticker_name} - Down Day Bounce Analysis<br><sub>Mean Reversion vs Momentum | Sample: {len(down_days)} down days</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        )
    )

    return fig, analysis_df, recommendation




# ==========================================
# MAIN APP
# ==========================================

st.subheader("📈 Single Stock Analysis")

# ── Load full stock list once per hour (client-side filtering = no reruns) ────
@st.cache_data(ttl=3600, show_spinner=False)
def _all_stock_options():
    stocks = data_manager.get_all_stock_basic()
    # Format: "688041 · 海光信息" — searchable by either part
    return [""] + [f"{s['ticker']} · {s['name']}" for s in stocks]

# ── Auto-analyze when a stock is picked from the combobox ─────────────────────
def _on_stock_pick():
    pick = (st.session_state.get("ssa_stock_pick") or "").strip()
    if pick:
        st.session_state.active_ticker = pick.split(" · ")[0].strip()

# ── Single combobox — type to filter, click to select ─────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.selectbox(
        "Stock code or name 股票代码或名称",
        options=_all_stock_options(),
        key="ssa_stock_pick",
        format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
        on_change=_on_stock_pick,
    )
with c2:
    st.write(""); st.write("")
    st.button("Analyze 分析", key="analyze_btn", type="primary",
              on_click=analyze_ticker, use_container_width=True)

# ── Recent searches ───────────────────────────────────────────────────────────
history = data_manager.get_search_history()
if history:
    history_options = [""] + [item["display"] for item in history[:10]]

    def _on_history_change():
        selected = st.session_state.get("history_select", "")
        if selected:
            for item in history:
                if item["display"] == selected:
                    st.session_state.active_ticker = item["ticker"]
                    st.session_state["history_select"] = ""
                    break

    st.selectbox(
        "📜 Recent searches:",
        options=history_options,
        format_func=lambda x: "Select a recent search…" if x == "" else x,
        key="history_select",
        on_change=_on_history_change,
    )



# Main analysis
if st.session_state.active_ticker:
    ticker = st.session_state.active_ticker.strip()
    
    with st.spinner(f"Loading {ticker}...前复权数据"):
        stock_df = load_single_stock(ticker,date.today())
    
    if stock_df is None or stock_df.empty:
        st.error(f"No data found for {ticker}. Check ticker is valid.")
    else:
        company_name = data_manager.update_search_history(ticker)

        # Fetch fundamental data
        start_date_str = stock_df.index.min().strftime('%Y%m%d')
        end_date_str = stock_df.index.max().strftime('%Y%m%d')
        fundamentals_df = data_manager.get_stock_fundamentals_live(ticker, start_date_str, end_date_str)
        
        # Get latest fundamentals for display
        latest_fund = {}
        if fundamentals_df is not None and not fundamentals_df.empty:
            latest_row = fundamentals_df.iloc[-1]
            latest_fund = {
                'PE_TTM': latest_row.get('PE_TTM', 'N/A'),
                'PB': latest_row.get('PB', 'N/A'),
                'Total_MV_Yi': latest_row.get('Total_MV_Yi', 'N/A'),
                'Circ_MV_Yi': latest_row.get('Circ_MV_Yi', 'N/A'),
                'Turnover_Rate': latest_row.get('Turnover_Rate', 'N/A')
            }

        # Format fundamentals for display
        pe_str = f"{latest_fund['PE_TTM']:.2f}" if isinstance(latest_fund.get('PE_TTM'), (int, float)) else 'N/A'
        pb_str = f"{latest_fund['PB']:.2f}" if isinstance(latest_fund.get('PB'), (int, float)) else 'N/A'
        mv_str = f"{latest_fund['Total_MV_Yi']:.0f}" if isinstance(latest_fund.get('Total_MV_Yi'), (int, float)) else 'N/A'
        circ_mv_str = f"{latest_fund['Circ_MV_Yi']:.0f}" if isinstance(latest_fund.get('Circ_MV_Yi'), (int, float)) else 'N/A'
        turnover_str = f"{latest_fund['Turnover_Rate']:.2f}" if isinstance(latest_fund.get('Turnover_Rate'), (int, float)) else 'N/A'

        with st.spinner("Running analysis...计算技术指标和交易信号"):
            # Run analysis
            analysis_df = run_single_stock_analysis(stock_df)

        
        if analysis_df.empty or len(analysis_df) < 50:
            st.error("Not enough data to compute signals.")
        else:
            with st.spinner("Calculating trading blocks...检测交易区间"):
                # Detect trading blocks
                blocks = calculate_multiple_blocks(analysis_df, lookback=60)

            # ==================== COMPANY INFO HEADER (NARROW) ====================
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 20px; border-radius: 10px; margin-bottom: 15px;">
                <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr; gap: 20px; align-items: center;">
                    <div>
                        <h3 style="color: white; margin: 0; font-size: 20px;">{company_name}</h3>
                        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 13px;">{ticker}</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: rgba(255,255,255,0.7); font-size: 11px;">昨日停盘价</div>
                        <div style="color: white; font-size: 16px; font-weight: bold;">¥{stock_df['Close'].iloc[-1]:.2f}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: rgba(255,255,255,0.7); font-size: 11px;">P/E</div>
                        <div style="color: white; font-size: 14px; font-weight: bold;">{pe_str}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: rgba(255,255,255,0.7); font-size: 11px;">P/B</div>
                        <div style="color: white; font-size: 14px; font-weight: bold;">{pb_str}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: rgba(255,255,255,0.7); font-size: 11px;">总市值</div>
                        <div style="color: white; font-size: 14px; font-weight: bold;">{mv_str}亿</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: rgba(255,255,255,0.7); font-size: 11px;">换手率</div>
                        <div style="color: white; font-size: 14px; font-weight: bold;">{turnover_str}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ==================== MARKET STATUS (STREAMLIT NATIVE) ====================
            # Get latest row for status display
            latest_row = analysis_df.iloc[-1]
            close_price = latest_row['Close']
            prev_close = analysis_df.iloc[-2]['Close'] if len(analysis_df) > 1 else close_price

            # Get all signals
            accumulation = bool(latest_row.get('Signal_Accumulation', False))
            squeeze = bool(latest_row.get('Signal_Squeeze', False))

            # Bull signals
            macd_bottoming = bool(latest_row.get('MACD_Bottoming', False))
            macd_crossover = bool(latest_row.get('MACD_Classic_Crossover', False))
            rsi_bottoming = bool(latest_row.get('RSI_Bottoming', False))

            # Bear signals
            macd_peaking = bool(latest_row.get('MACD_Peaking', False))
            macd_bearish = bool(latest_row.get('MACD_Bearish_Crossover', False))
            rsi_peaking = bool(latest_row.get('RSI_Peaking', False))

            # ADX info
            adx_val = float(latest_row.get('ADX', 0.0))
            adx_pattern = str(latest_row.get('ADX_Pattern', ''))

            # ========================== Market Status =================== #
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.markdown("**🔄 Squeeze**")
                if squeeze:
                    st.success("TIGHT")
                    st.caption("低波动")
                else:
                    st.info("LOOSE")
                    st.caption("正常波动")

            with col2:
                st.markdown("**📥 Accumulation**")
                if accumulation:
                    st.success("ACTIVE")
                    st.caption("建仓阶段")
                else:
                    st.info("INACTIVE")
                    st.caption("等待中")

            with col3:
                st.markdown("**🐂 Bull Signals**")
                bull_signals = []
                if macd_bottoming:
                    bull_signals.append("MACD底")
                if macd_crossover:
                    bull_signals.append("MACD金叉")
                if rsi_bottoming:
                    bull_signals.append("RSI底")
                
                if bull_signals:
                    st.success(" | ".join(bull_signals))
                    st.caption(f"{len(bull_signals)}个信号")
                else:
                    st.info("无")
                    st.caption("等待信号")

            with col4:
                st.markdown("**🐻 Bear Signals**")
                bear_signals = []
                if macd_peaking:
                    bear_signals.append("MACD顶")
                if macd_bearish:
                    bear_signals.append("MACD死叉")
                if rsi_peaking:
                    bear_signals.append("RSI顶")
                
                if bear_signals:
                    st.error(" | ".join(bear_signals))
                    st.caption(f"{len(bear_signals)}个信号")
                else:
                    st.info("无")
                    st.caption("等待信号")

            with col5:
                st.markdown("**📦 Trading Block**")
                if blocks:
                    block = blocks[-1]
                    status = block['status']
                    
                    # Only show if it's the active block
                    if block.get('is_active', False):
                        if status == 'BREAKOUT':
                            st.error("BREAKOUT")
                        elif status == 'BREAKDOWN':
                            st.success("BREAKDOWN")
                        elif status == 'INSIDE':
                            st.warning("INSIDE")
                        else:
                            st.info(status)
                        
                        st.caption(f"¥{block['bot']:.2f}-¥{block['top']:.2f}")
                    else:
                        st.info("NO ACTIVE BLOCK")
                        st.caption("Historical block only")
                else:
                    st.info("无")
                    st.caption("无明确区间")

            with col6:
                st.markdown("**📊 Market Regime**")
                regime = latest_row.get('Market_Regime', 'Normal Volatility')
                
                if regime == 'High Volatility':
                    st.error("HIGH")
                    st.caption("高波动环境")
                elif regime == 'Low Volatility':
                    st.success("LOW")
                    st.caption("低波动环境")
                else:
                    st.info("NORMAL")
                    st.caption("正常环境")

            st.caption(f"📈 ADX: **{adx_val:.1f}** | Pattern: **{adx_pattern if adx_pattern else 'None'}**")
            # st.markdown("---")
            # ==================== END MARKET STATUS ====================

            
            # ── Chart section wrapped in a fragment so the comparison picker
            #    only rerenders the chart — NOT the analysis above it.
            @st.fragment
            def chart_with_comparison(analysis_df, fundamentals_df, blocks):
                # ── Controls row ─────────────────────────────────────────────
                ctrl_l, ctrl_r = st.columns([3, 2])
                with ctrl_l:
                    comp_input = st.text_input(
                        "📊 Compare with another stock (optional — 6-digit code)",
                        value="",
                        placeholder="e.g. 600036",
                        key="comp_ticker_input",
                        help="Overlay a second stock on the price chart. "
                             "Changing the ticker or scale mode only rerenders the chart.",
                    )
                with ctrl_r:
                    scale_choice = st.radio(
                        "Price scale",
                        options=["Same % Scale", "New Price Scale"],
                        index=0,
                        horizontal=True,
                        key="comp_scale_mode",
                        help=(
                            "**Same % Scale**: comparison stock rebased to the main stock's "
                            "first price — both share the left ¥ axis. The gap between the "
                            "lines is pure relative performance.\n\n"
                            "**New Price Scale**: comparison stock plotted on its own right-hand "
                            "axis at actual price."
                        ),
                    )
                scale_mode = "pct" if scale_choice == "Same % Scale" else "new"

                # ── Load comparison data ──────────────────────────────────────
                comp_df_overlay   = None
                comp_name_overlay = "Comparison"
                _comp = comp_input.strip().split()[0] if comp_input.strip() else ""
                if _comp and _comp.isdigit() and len(_comp) == 6:
                    try:
                        with st.spinner(f"Loading {_comp}…"):
                            comp_df_overlay = load_single_stock(_comp, date.today())
                        try:
                            _ts = data_manager.get_tushare_ticker(_comp)
                            _sb = data_manager.db.read_table(
                                'stock_basic', filters={'ts_code': _ts}, columns='name', limit=1)
                            _cname = _sb.iloc[0]['name'] if (_sb is not None and not _sb.empty) else _comp
                        except Exception:
                            _cname = _comp
                        comp_name_overlay = f"{_cname} ({_comp})"
                    except Exception as _e:
                        st.warning(f"Could not load {_comp}: {_e}")

                # ── Render ────────────────────────────────────────────────────
                with st.spinner("Generating chart..."):
                    fig_stock = create_single_stock_chart_analysis(
                        analysis_df,
                        fundamentals_df=fundamentals_df,
                        blocks=blocks,
                        comp_df=comp_df_overlay,
                        comp_name=comp_name_overlay,
                        scale_mode=scale_mode,
                    )
                st.plotly_chart(fig_stock, use_container_width=True)

            chart_with_comparison(analysis_df, fundamentals_df, blocks)

            st.markdown("---")

            @st.fragment
            def simulator_section():
                """Isolated simulator - only this reruns on slider changes"""

                st.subheader("🎮 What-If Simulator (明日指标模拟器)")
                st.caption("输入明日价格变化和成交量，实时计算MACD、RSI、ADX指标值 | Input tomorrow's price change and volume to see indicator values")

                # Get reference values from parent scope - CORRECT VARIABLE NAME: analysis_df
                latest = analysis_df.iloc[-1]
                vol_10d_avg = analysis_df['Volume'].rolling(10).mean().iloc[-1]
                vol_yesterday = latest['Volume']
                close_yesterday = latest['Close']


                # Calculate dynamic bounds
                vol_10d_avg_millions = vol_10d_avg / 1e6  # Convert to millions
                vol_min = max(0.01, vol_10d_avg_millions * 0.1)  # Min: 10% of avg, or 0.01M
                vol_max = vol_10d_avg_millions * 10  # Max: 10x average
                vol_step = max(0.001, vol_10d_avg_millions * 0.05)  # Step: 5% of avg, floor at 0.001M

                # Input section
                col_input1, col_input2 = st.columns(2)

                with col_input1:
                    st.markdown("**📈 明日价格变化 (%)**")
                    price_change = st.slider(
                        "Price Change",
                        min_value=-10.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.1,
                        format="%.1f%%",
                        help="正数=上涨，负数=下跌",
                        key="frag_sim_price",
                        label_visibility="collapsed"
                    )
                    target_price = close_yesterday * (1 + price_change / 100)
                    st.caption(f"昨收: ¥{close_yesterday:.2f} → 明日: ¥{target_price:.2f}")

                with col_input2:
                    st.markdown("**📦 明日成交量 (M)**")
                    vol_input_millions = st.number_input(
                        "Volume",
                        min_value=vol_min,
                        max_value=vol_max,
                        value=vol_10d_avg / 1e6,
                        step=vol_step,
                        format="%.3f",
                        help="以百万为单位",
                        key="frag_sim_volume",
                        label_visibility="collapsed"
                    )
                    volume_tomorrow = vol_input_millions * 1e6
                    st.caption(f"昨日: {vol_yesterday/1e6:.1f}M | 10日均: {vol_10d_avg/1e6:.1f}M")

                # Calculate simulation - uses analysis_df (not analysisdf)
                sim_result = simulate_next_day_indicators(analysis_df, price_change, volume_tomorrow)

                if sim_result:
                    st.markdown("---")

                    # Display results
                    col_macd, col_rsi, col_adx, col_obv = st.columns(4)

                    with col_macd:
                        st.markdown("**📊 MACD**")
                        macd_tmr = sim_result['macd_tomorrow']
                        macd_delta = macd_tmr - sim_result['macd_today']

                        st.metric("MACD Line", f"{macd_tmr:.4f}", delta=f"{macd_delta:.4f}", delta_color="normal")
                        st.metric("MACD Signal", f"{sim_result['macd_signal_tomorrow']:.4f}", 
                                delta=f"{sim_result['macd_signal_tomorrow'] - sim_result['macd_signal_today']:.4f}")
                        st.metric("Gap", f"{sim_result['macd_gap_tomorrow']:.4f}", 
                                delta=f"{sim_result['macd_gap_tomorrow'] - sim_result['macd_gap_today']:.4f}")

                        if sim_result['signals']['MACD_Bottoming']:
                            st.success("✅ MACD Bottoming")
                        if sim_result['signals']['MACD_Bullish_Cross']:
                            st.success("✅ 金叉 Bullish Cross")
                        if sim_result['signals']['MACD_Bearish_Cross']:
                            st.error("⚠️ 死叉 Bearish Cross")

                        with st.expander("📊 MACD 参考区间"):
                            # Use the corrected 10-day low that includes the simulated tomorrow
                            low_tmr = sim_result.get('macd_10d_low_tomorrow', sim_result['macd_10d_low'])
                            
                            st.caption(f"10日低点: {low_tmr:.4f}")
                            st.caption(f"10日高点: {sim_result['macd_10d_high']:.4f}")
                            
                            # Show the exact narrow band for the bottoming bounce
                            st.caption(f"Bottoming区间: {low_tmr:.4f} ~ {low_tmr * 0.95:.4f}")

                    with col_rsi:
                        st.markdown("**📈 RSI**")
                        rsi_tmr = sim_result['rsi_tomorrow']
                        rsi_delta = rsi_tmr - sim_result['rsi_today']

                        st.metric("RSI(14)", f"{rsi_tmr:.1f}", delta=f"{rsi_delta:.1f}", delta_color="normal")

                        if rsi_tmr < 30:
                            st.error("🔴 超卖 Oversold")
                        elif rsi_tmr > 70:
                            st.warning("🔴 超买 Overbought")
                        elif 40 <= rsi_tmr <= 60:
                            st.info("⚪ 中性 Neutral")
                        else:
                            st.success("🟢 正常 Normal")

                        if sim_result['signals']['RSI_Bottoming']:
                            st.success("✅ RSI Bottoming")
                        if sim_result['signals']['RSI_Peaking']:
                            st.error("⚠️ RSI Peaking")

                        with st.expander("📈 RSI 参考线"):
                            st.caption(f"底部10%线: {sim_result['rsi_p10']:.1f}")
                            st.caption(f"顶部90%线: {sim_result['rsi_p90']:.1f}")

                    with col_adx:
                        st.markdown("**📉 ADX (趋势强度)**")
                        adx_tmr = sim_result['adx_tomorrow']
                        adx_delta = adx_tmr - sim_result['adx_today']

                        st.metric("ADX", f"{adx_tmr:.1f}", delta=f"{adx_delta:.1f}", delta_color="normal")

                        if adx_tmr < 20:
                            st.info("📊 弱趋势 Weak")
                        elif adx_tmr < 25:
                            st.info("📊 无趋势 No Trend")
                        elif adx_tmr < 40:
                            st.success("📊 强趋势 Strong")
                        else:
                            st.warning("📊 极强趋势 Very Strong")

                        # ADX Pattern - Only 4 patterns shown as informational
                        adx_pattern = sim_result['adx_pattern']
                        if adx_pattern:
                            if adx_pattern == "Bottoming":
                                st.info("ℹ️ **ADX Bottoming** 趋势即将启动")
                            elif adx_pattern == "Reversing Up":
                                st.success("📈 **ADX Reversing Up**趋势开始增强")
                            elif adx_pattern == "Peaking":
                                st.warning("⚠️ **ADX Peaking**趋势可能见顶")
                            elif adx_pattern == "Reversing Down":
                                st.error("📉 **ADX Reversing Down**趋势开始减弱")

                        st.caption("*ADX仅为近似估算")

                    with col_obv:
                        st.markdown("**📦 OBV (Volume-Scaled)**")
                        obv_scaled_tmr = sim_result['obv_scaled_tomorrow']
                        obv_delta = obv_scaled_tmr - sim_result['obv_scaled_today']

                        st.metric("Vol-Scaled OBV", f"{obv_scaled_tmr:.2f}", delta=f"{obv_delta:.2f}", delta_color="normal")

                        obv_3d_scaled = sim_result['obv_scaled_3d_ago']
                        if obv_scaled_tmr > obv_3d_scaled:
                            st.success("✅ 高于3日前")
                            st.caption(f"3日前: {obv_3d_scaled:.2f}")
                        else:
                            st.error("⚠️ 低于3日前")
                            st.caption(f"需增加: {obv_3d_scaled - obv_scaled_tmr:.2f}")

                        vol_ratio = volume_tomorrow / vol_10d_avg
                        if vol_ratio > 1.5:
                            st.warning("📦 放量 High")
                        elif vol_ratio > 1.2:
                            st.info("📦 温和放量")
                        elif vol_ratio < 0.8:
                            st.info("📦 缩量 Low")
                        else:
                            st.success("📦 正常量")

                    # MACD Bottoming condition details
                    with st.expander("🔍 MACD Bottoming 条件详情"):
                        st.markdown(f"**需满足≥2个条件**: {sim_result['conditions_met']}/4")

                        cond_col1, cond_col2 = st.columns(2)

                        with cond_col1:
                            macd_stopped = sim_result['macd_tomorrow'] >= sim_result['macd_today']
                            
                            # Use the corrected 10d low that includes tomorrow's simulation
                            macd_10d_low_tmr = sim_result['macd_10d_low_tomorrow']
                            macd_in_zone = (sim_result['macd_tomorrow'] >= macd_10d_low_tmr) and (sim_result['macd_tomorrow'] <= macd_10d_low_tmr * 0.95)

                            st.markdown("**1. MACD停止下跌**")
                            if macd_stopped:
                                st.success(f"✅ {sim_result['macd_tomorrow']:.4f} ≥ {sim_result['macd_today']:.4f}")
                            else:
                                st.error(f"❌ {sim_result['macd_tomorrow']:.4f} < {sim_result['macd_today']:.4f}")

                            st.markdown("**2. MACD在底部区间**")
                            if macd_in_zone:
                                st.success(f"✅ 在底部区间内")
                            else:
                                st.error(f"❌ 不在底部区间")
                                st.caption(f"目标区间: {macd_10d_low_tmr:.4f} ~ {macd_10d_low_tmr * 0.95:.4f}")

                        with cond_col2:
                            # gap_narrowing = sim_result['macd_gap_tomorrow'] > sim_result['macd_gap_today']

                            gap_narrowing = abs(sim_result['macd_gap_tomorrow']) < abs(sim_result['macd_gap_today'])
                            
                            # Grab the raw boolean evaluated by the engine
                            obv_rising = sim_result['obv_rising']

                            st.markdown("**3. MACD缺口收窄**")
                            if gap_narrowing:
                                st.success(f"✅ 缺口收窄")
                            else:
                                st.error(f"❌ 缺口未收窄")

                            st.markdown("**4. OBV上升 (Raw未缩放)**")
                            if obv_rising:
                                st.success(f"✅ OBV上升")
                                st.caption(f"{sim_result['obv_raw_tomorrow']:,.0f} > {sim_result['obv_raw_3d_ago']:,.0f}")
                            else:
                                st.error(f"❌ OBV未上升")
                                st.caption(f"需增加: {sim_result['obv_raw_3d_ago'] - sim_result['obv_raw_tomorrow']:,.0f}")

                else:
                    st.error("无法计算模拟结果")

            simulator_section()
            st.markdown("---")

            st.subheader("🎯 Setup-Conditioned Expectancy")
            st.markdown("Test your strategy: Buy on bullish signals, sell on bearish signals.")
            
            # ADD THIS: Date Range Selector
            col_date, col_buy, col_sell = st.columns([1, 1, 1])

            with col_date:
                # Get available date range from analysis data
                min_date = analysis_df.index.min().date()
                max_date = analysis_df.index.max().date()
                
                backtest_start = st.date_input(
                    "Backtest Start Date",
                    value=min_date,  # Default to earliest date
                    min_value=min_date,
                    max_value=max_date,
                    key='backtest_start_date'
                )
                
                st.caption(f"Available: {min_date} to {max_date}")

            with col_buy:
                # Your existing buy signal selector
                buy_options = list(BULLISH_SIGNALS.keys()) + list(ADX_BULLISH_PATTERNS.keys())
                buy_display = {**BULLISH_SIGNALS, **ADX_BULLISH_PATTERNS}
                buy_signal = st.selectbox(
                    "Buy Signal (Entry)", 
                    buy_options, 
                    format_func=lambda x: buy_display[x]
                )

            with col_sell:
                # Your existing sell signal selector
                sell_options = list(BEARISH_SIGNALS.keys()) + list(ADX_BEARISH_PATTERNS.keys())
                sell_display = {**BEARISH_SIGNALS, **ADX_BEARISH_PATTERNS}
                sell_signal = st.selectbox(
                    "Sell Signal (Exit)", 
                    sell_options, 
                    format_func=lambda x: sell_display[x]
                )

            # Filter analysis data based on selected start date
            if buy_signal and sell_signal:
                with st.spinner("Running backtest..."):
                    # Filter data from selected start date
                    backtest_df = analysis_df[analysis_df.index >= pd.Timestamp(backtest_start)]
                    
                    if len(backtest_df) < 10:
                        st.warning(f"⚠️ Not enough data from {backtest_start}. Need at least 10 days.")
                    else:
                        # Run backtest on filtered data
                        trades_df, summary = backtest_signal_expectancy(
                            backtest_df,  # Use filtered data
                            buy_signal, 
                            sell_signal
                        )
                        
                        if summary is None:
                            st.warning("No complete transactions found in selected period. Try different signals or earlier start date.")
                        else:
                            # Show backtest period info
                            st.info(f"📊 Backtesting from **{backtest_start}** to **{max_date}** ({len(backtest_df)} days)")
                            
                            # Display chart (pass filtered data)
                            fig_backtest = create_backtest_chart(trades_df, backtest_df)
                            st.plotly_chart(fig_backtest, use_container_width=True)
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Total Transactions", int(summary['Total_Transactions']))
                                st.caption(f"{int(summary['Winning_Trades'])} wins / {int(summary['Losing_Trades'])} losses")

                            with col2:
                                win_rate = summary['Win_Rate']
                                st.metric("Win Rate", f"{win_rate:.1f}%")
                                if win_rate >= 60:
                                    st.success("Strong")
                                elif win_rate >= 50:
                                    st.info("Decent")
                                else:
                                    st.warning("Weak")

                            with col3:
                                avg_profit = summary['Avg_Profit_Pct']
                                st.metric("Avg Profit/Trade", f"{avg_profit:.2f}%")
                                total_profit_dollars = summary['Total_Profit']
                                if avg_profit > 0:
                                    st.success(f"¥{total_profit_dollars:.2f} total")  # Display as currency, not %
                                else:
                                    st.error(f"¥{total_profit_dollars:.2f} total")


                            with col4:
                                st.metric("Best Trade", f"{summary['Best_Trade_Pct']:.2f}%")
                                st.caption(f"Worst: {summary['Worst_Trade_Pct']:.2f}%")




            # Return Risk Analysis Section
            st.markdown("---")
            st.subheader("📊 Return Risk Analysis | 交易风险分析")

            # Generate analysis
            fig_dist, metrics_df = analyze_return_distribution(stock_df, ticker_name=ticker)

            if fig_dist is not None:
                # Display metrics table and chart
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("#### Risk Metrics | 风险指标")
                    st.dataframe(
                        metrics_df,
                        hide_index=True,
                        height=700,
                        use_container_width=True
                    )

                with col2:
                    st.markdown("#### Visual Analysis | 可视化分析")
                    st.plotly_chart(fig_dist, use_container_width=True)

                # Trading insights
                st.markdown("### 💡 Key Insights | 关键洞察")

                returns = stock_df['Close'].pct_change().dropna()  # ← CORRECT
                var_95 = returns.quantile(0.05)
                cvar_95 = returns[returns <= var_95].mean()
                win_rate = (returns > 0).sum() / len(returns)
                kurtosis = returns.kurtosis()
                mean_return = returns.mean()

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Win Rate | 胜率", f"{win_rate*100:.1f}%")
                    if win_rate > 0.55:
                        st.success("✅ Positive edge | 正向优势")
                    elif win_rate > 0.45:
                        st.info("⚖️ Neutral | 中性")
                    else:
                        st.warning("⚠️ Negative edge | 负向优势")

                with col_b:
                    st.metric("Distribution | 分布类型", 
                            "Fat Tail" if kurtosis > 3 else "Normal-like")
                    if kurtosis > 3:
                        st.warning(f"⚠️ Kurtosis = {kurtosis:.1f}")
                        st.caption("Extreme events more likely | 极端事件更可能")
                    else:
                        st.success("✅ Normal distribution | 正态分布")

                with col_c:
                    st.metric("CVaR 95% (Daily) | 条件风险", f"{cvar_95*100:.2f}%")
                    st.caption("Avg loss beyond VaR | 超VaR平均损失")
                    if cvar_95 < -0.05:
                        st.error("⚠️ High tail risk | 高尾部风险")

                # Recommendation box
                st.info(f"""
                **💼 T+1 Risk Summary:**

                - **Expected daily return:** {mean_return*100:.3f}% per trade
                - **Worst-case 1-day loss (95% CVaR):** {cvar_95*100:.2f}%
                {f"⚠️ **High tail risk** — CVaR exceeds 5%. Reduce position size accordingly." if abs(cvar_95) > 0.05 else ""}
                {f"⚠️ **Fat tails detected** (kurtosis {kurtosis:.1f}) — widen stop losses beyond normal ATR estimates." if kurtosis > 3 else ""}

                📌 **Position sizing:** Use the Kelly Criterion in the Down Day Bounce section below — it is calibrated to this stock's actual bounce probabilities and win/loss ratios.
                """)

            else:
                st.warning("Not enough data for distribution analysis (need 30+ days) | 数据不足（需要30天以上）")

            
            
            # ========================================
            # REPLACE THE PREVIOUS CONDITIONAL ENTRY SECTION WITH THIS
            # ========================================

            st.markdown("---")
            st.subheader("🎯 Realistic T+1 Trading Analysis | 真实T+1交易分析")
            st.caption(
                "Close-to-close ignores intraday opportunity. "
                "T+0 dips let you buy cheaper; T+1 spikes let you sell higher. "
                "Compares 4 realistic entry/exit scenarios using this stock's actual intraday history."
            )

            # Generate analysis
            fig_realistic, scenarios_df, entry_df = analyze_realistic_t1_trading(stock_df, ticker_name=ticker)

            if fig_realistic is not None:
                # Show visualizations
                st.plotly_chart(fig_realistic, use_container_width=True)

                # Scenario comparison table
                st.markdown("#### 📊 Scenario Comparison | 场景对比")

                display_scenarios = scenarios_df.copy()
                display_scenarios['Win Rate'] = (display_scenarios['Win Rate'] * 100).map('{:.1f}%'.format)
                display_scenarios['Avg Return'] = (display_scenarios['Avg Return'] * 100).map('{:.2f}%'.format)
                display_scenarios['Median Return'] = (display_scenarios['Median Return'] * 100).map('{:.2f}%'.format)
                display_scenarios['Best Case'] = (display_scenarios['Best Case'] * 100).map('{:.2f}%'.format)
                display_scenarios['Worst Case'] = (display_scenarios['Worst Case'] * 100).map('{:.2f}%'.format)
                display_scenarios['Improvement vs Baseline'] = (display_scenarios['Improvement vs Baseline'] * 100).map('{:.2f}%'.format)

                st.dataframe(display_scenarios, use_container_width=True, hide_index=True)

                # Intraday statistics
                intraday_stats = calculate_intraday_stats(stock_df)

                st.markdown("#### 📈 Intraday Opportunities | 盘中机会")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Avg T+0 Discount | 今日折扣",
                        f"{intraday_stats['T0_Avg_Discount']*100:.2f}%"
                    )
                    st.caption(f"Low vs Close | 最低价 vs 收盘")
                    st.caption(f">{1}% discount on {intraday_stats['T0_Days_Discount_1pct']} days")

                with col2:
                    st.metric(
                        "Avg T+1 Premium | 明日溢价",
                        f"{intraday_stats['T1_Avg_Premium']*100:.2f}%"
                    )
                    st.caption(f"High vs Close | 最高价 vs 收盘")
                    st.caption(f">{1}% premium on {intraday_stats['T1_Days_Premium_1pct']} days")

                with col3:
                    improvement = (scenarios_df.iloc[3]['Avg Return'] - scenarios_df.iloc[0]['Avg Return']) * 100
                    st.metric(
                        "Potential Improvement | 潜在提升",
                        f"{improvement:.2f}%"
                    )
                    st.caption("Low→Close vs Close→Close")

                st.caption(
                    "💡 For per-drop-bucket win rates and Kelly-optimal position sizing, "
                    "see the **Down Day Bounce Analysis** section below."
                )

            else:
                st.warning("Not enough data for realistic T+1 analysis (need 50+ days)")

            st.markdown("---")
            st.subheader("📉➡️📈 Down Day Bounce Analysis | 下跌反弹分析")
            st.caption(
                "Given today is a down day — what happens tomorrow? "
                "Bounce probability, expected magnitude, mean-reversion vs momentum behaviour, "
                "and Kelly-optimal position sizing, all bucketed by today's drop size."
            )

            # Generate bounce analysis
            fig_bounce, bounce_df, recommendation = analyze_down_day_bounce_probability(stock_df, ticker_name=ticker)

            if fig_bounce is not None:
                # Show visualizations
                st.plotly_chart(fig_bounce, use_container_width=True)

                # Display analysis table
                st.markdown("#### 📊 Detailed Bounce Analysis | 详细反弹分析")

                display_bounce = bounce_df.copy()

                # Format columns
                pct_cols = ['Bounce Probability', 'Continue Down Probability', 
                            'Avg Tomorrow Return', 'Median Tomorrow Return',
                            'Avg If Bounce', 'Avg If Continue', 
                            'Best Case Tomorrow', 'Worst Case Tomorrow',
                            'Expected Value', 'Kelly % (Position Size)',
                            'Avg Return (Close→High)', 'Avg Return (Low→Close)']

                for col in pct_cols:
                    if col in display_bounce.columns:
                        display_bounce[col] = (display_bounce[col] * 100).map('{:.2f}%'.format)

                if 'Risk/Reward Ratio' in display_bounce.columns:
                    display_bounce['Risk/Reward Ratio'] = display_bounce['Risk/Reward Ratio'].map('{:.2f}'.format)

                if 'Sharpe-like Score' in display_bounce.columns:
                    display_bounce['Sharpe-like Score'] = display_bounce['Sharpe-like Score'].map('{:.2f}'.format)

                st.dataframe(display_bounce, use_container_width=True, hide_index=True)

                # Trading recommendation
                st.markdown("### 💡 Complete Trading Strategy | 完整交易策略")

                if recommendation['has_opportunity']:
                    st.success("✅ Profitable Bounce Opportunity Detected | 检测到盈利反弹机会")

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Best Entry Signal",
                            recommendation['best_drop_range']
                        )
                        st.caption(f"Sample: {recommendation['sample_size']} occurrences")

                    with col2:
                        st.metric(
                            "Bounce Probability",
                            f"{recommendation['bounce_prob']*100:.1f}%"
                        )
                        if recommendation['bounce_prob'] > 0.6:
                            st.success("High confidence ✓")
                        else:
                            st.info("Moderate confidence")

                    with col3:
                        st.metric(
                            "Expected Return",
                            f"{recommendation['expected_return']*100:.2f}%"
                        )
                        st.caption("Average outcome tomorrow")

                    with col4:
                        st.metric(
                            "Risk/Reward",
                            f"{recommendation['risk_reward']:.2f}"
                        )
                        if recommendation['risk_reward'] > 1.5:
                            st.success("Excellent R/R ✓")
                        elif recommendation['risk_reward'] > 1:
                            st.info("Acceptable R/R")
                        else:
                            st.warning("Poor R/R")

                    # Market behavior analysis
                    st.markdown("#### 📈 Market Behavior | 市场行为")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        if recommendation['mean_reversion'] > 0.3:
                            st.success(f"**Mean Reversion Detected** | **均值回归**")
                            st.markdown(f"""
                            **Correlation: {-recommendation['mean_reversion']:.2f}**

                            ✅ This stock exhibits **mean reversion** behavior:
                            - Bigger drops tend to lead to bigger bounces
                            - "Buy the dip" strategy works well
                            - Oversold conditions typically reverse
                            """)
                        elif recommendation['momentum'] > 0.3:
                            st.warning(f"**Momentum Detected** | **动量效应**")
                            st.markdown(f"""
                            **Correlation: {recommendation['momentum']:.2f}**

                            ⚠️ This stock exhibits **momentum** behavior:
                            - Bigger drops tend to continue falling
                            - "Catch falling knife" is dangerous
                            - Wait for trend reversal confirmation
                            """)
                        else:
                            st.info("**Random Walk** | **随机游走**")
                            st.markdown("""
                            Correlation near zero - no clear pattern.
                            Tomorrow's direction is largely unpredictable.
                            """)

                    with col_b:
                        st.markdown("**Win/Loss Breakdown | 盈亏分解**")
                        st.markdown(f"""
                        - **If tomorrow bounces ({recommendation['bounce_prob']*100:.0f}% chance):**
                        - Average gain: **{recommendation['avg_if_win']*100:.2f}%**

                        - **If tomorrow continues down ({(1-recommendation['bounce_prob'])*100:.0f}% chance):**
                        - Average loss: **{recommendation['avg_if_lose']*100:.2f}%**

                        **Expected Value:** {recommendation['expected_return']*100:.2f}%
                        """)

                    # Complete trading plan
                    st.markdown("#### 🎯 Complete Trading Plan | 完整交易计划")

                    kelly_pct = recommendation['position_size']
                    half_kelly = kelly_pct / 2

                    st.info(f"""
                    **📋 Step-by-Step Trading Plan:**

                    **1. WAIT FOR ENTRY SIGNAL**
                    - Watch for drop in range: **{recommendation['best_drop_range']}**
                    - Historical occurrence: {recommendation['sample_size']} times in dataset

                    **2. POSITION SIZING**
                    - Kelly Optimal: **{kelly_pct*100:.1f}%** of capital
                    - Conservative (Half-Kelly): **{half_kelly*100:.1f}%** of capital
                    - Max risk per trade: **1-2%** of portfolio (recommended)

                    **3. ENTRY EXECUTION**
                    - **Option A (Conservative):** Buy at close today
                    - **Option B (Better):** Set limit order at today's low (typically {bounce_df[bounce_df['Drop Magnitude']==recommendation['best_drop_range']]['Avg Return (Low→Close)'].values[0] if len(bounce_df[bounce_df['Drop Magnitude']==recommendation['best_drop_range']]) > 0 else 'N/A'} better return)

                    **4. EXIT STRATEGY**
                    - **Target:** +{recommendation['avg_if_win']*100:.2f}% (sell at close tomorrow)
                    - **Aggressive:** Set limit at tomorrow's expected high for extra {bounce_df[bounce_df['Drop Magnitude']==recommendation['best_drop_range']]['Avg Return (Close→High)'].values[0] if len(bounce_df[bounce_df['Drop Magnitude']==recommendation['best_drop_range']]) > 0 else 'N/A'}
                    - **Stop Loss:** {recommendation['avg_if_lose']*100:.2f}% (if tomorrow continues down)

                    **5. EXPECTED OUTCOME**
                    - Win rate: {recommendation['bounce_prob']*100:.0f}%
                    - Avg profit per trade: {recommendation['expected_return']*100:.2f}%
                    - Risk/Reward ratio: {recommendation['risk_reward']:.2f}:1

                    ---

                    **⚠️ Risk Management:**
                    - Never risk more than 2% of portfolio on single trade
                    - Use stop loss religiously
                    - Past performance doesn't guarantee future results
                    - Consider overall market conditions
                    """)

                else:
                    st.warning("⚠️ No Clear Bounce Opportunity Detected | 未检测到明确反弹机会")

                    st.markdown(f"""
                    Based on historical data:
                    - Most down days do NOT lead to profitable bounces
                    - Mean reversion strength: {recommendation['mean_reversion']:.2f}
                    - Momentum strength: {recommendation['momentum']:.2f}

                    **Recommendation:** Avoid "buy the dip" strategy for this stock.
                    Consider:
                    - Trend-following strategies instead
                    - Longer time horizons (T+5, T+10)
                    - Only trade with strong overall market confirmation
                    """)

            else:
                st.warning("Not enough data for down day bounce analysis (need 50+ days with 20+ down days)")


            # ── Stock Personality & Strategy Analysis ─────────────────────────────
            st.markdown("---")
            st.subheader("股票个性 & 策略分析 | Stock Personality & Strategy Analysis")
            st.caption(
                "Analyses this stock's unique behaviour to discover which strategy fits it best "
                "— not copied from another stock. | 探索本股独有个性，发现最适合的交易策略。"
            )

            with st.spinner("计算中… Hurst / seasonality / grid search..."):
                personality = analyze_stock_personality(analysis_df)

            # ── Hurst Exponent ────────────────────────────────────────────
            hurst_val    = personality['hurst']
            hurst_label  = personality['hurst_label']
            hurst_color  = personality['hurst_color']
            hurst_advice = personality['hurst_advice']

            st.markdown("#### 🧬 Hurst Exponent — What Kind of Stock Is This? | \u8fd9\u652f\u80a1\u7968\u662f\u4ec0\u4e48\u4e2a\u6027\uff1f")
            h1, h2, h3 = st.columns([1, 2, 3])
            with h1:
                st.metric("Hurst Exponent", f"{hurst_val:.4f}")
            with h2:
                st.markdown(
                    f"<span style='font-size:18px; font-weight:bold; color:{hurst_color}'>"
                    f"{hurst_label}</span>",
                    unsafe_allow_html=True
                )
            with h3:
                st.info(hurst_advice)

            st.markdown(
                """
| Range | Personality | Best Strategy |
|-------|-------------|---------------|
| H < 0.45 | 🟢 Mean-Reverting 均值回归 | Dip-buy on drops, fade extremes |
| H ≈ 0.5  | 🟡 Random Walk 随机游走 | No strong directional edge |
| H > 0.55 | 🔵 Trending / Momentum 趋势 | Follow breakouts & MACD crossovers |
                """
            )

            # ── Monthly Seasonality ──────────────────────────────────
            st.markdown("#### 📅 Monthly Seasonality — 10-Day Forward Return by Month | 月度季节性")
            monthly_fwd  = personality['monthly_fwd']
            month_order  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            months_p     = [m for m in month_order if m in monthly_fwd]
            fwd_vals     = [monthly_fwd[m] for m in months_p]
            bar_colors   = ['#22c55e' if v >= 0 else '#ef4444' for v in fwd_vals]

            fig_season = go.Figure(go.Bar(
                x=months_p,
                y=fwd_vals,
                marker_color=bar_colors,
                text=[f"{v:+.2f}%" for v in fwd_vals],
                textposition='outside'
            ))
            fig_season.update_layout(
                title="各月\u5165\u573a\u540e10\u4ea4\u6613\u65e5\u5e73\u5747\u6536\u76ca | Avg 10-Day Return by Entry Month",
                yaxis_title="Avg 10d Return (%)",
                height=320,
                template='plotly_white',
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_season, use_container_width=True)

            sc1, sc2 = st.columns(2)
            with sc1:
                strong_m = personality.get('strong_months', [])
                st.success(f"🟢 Best months to enter: **{', '.join(strong_m)}**")
            with sc2:
                avoid_m = personality.get('avoid_months', [])
                if avoid_m:
                    st.error(f"🔴 Avoid entering in: **{', '.join(avoid_m)}**")
                else:
                    st.info("No consistently negative months found.")

            # ── Auto-Discovered Optimal Setup ───────────────────────────
            st.markdown("#### 🎯 Optimal Dip-Reversal Setup (Auto-Discovered) | 自动发现最优参数")
            best = personality.get('best_strategy')
            if best:
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("跌幅闽值 | Drop Thr.",  f"{best['drop_threshold']*100:.1f}%")
                b2.metric("CAGR",             f"{best['cagr_pct']:.1f}%")
                b3.metric("Sharpe",           f"{best['sharpe']:.2f}")
                b4.metric("胜率 | Win Rate",  f"{best['win_rate']:.1f}%")
                b5.metric("交易次数 | Trades", str(best['n_trades']))

                avoid_str = ', '.join(best['avoid_months']) if best['avoid_months'] else 'None'
                st.markdown(
                    f"""
**入场条件 Entry rule:** Daily return \u2264 `{best['drop_threshold']*100:.1f}%` AND RSI > 20  
**外场条件 Exit rules:** +8% profit target / \u22125% stop loss / 10-day time stop / RSI > 68  
**过滤月份 Seasonal filter (skip in):** `{avoid_str}`  
**单笔最大亏损 Max single-trade loss:** `{best['max_dd']:.1f}%`
                    """
                )
                st.caption(
                    "⚠\ufe0f 以上为历史数据回测\u7ed3\u679c\uff0c\u4e0d\u4ee3\u8868\u672a\u6765\u8868\u73b0\u3002"
                    " Commission 0.1% + slippage 0.2% included. Past performance \u2260 future results."
                )
            else:
                st.warning("未找到足够交易次数 (need 8+ trades) to auto-discover a reliable setup.")
                st.info(
                    f"个性\u5df2\u8bc6\u522b: Hurst = {hurst_val:.3f} \u2192 {hurst_label}. "
                    "数据窗口可能\u8fc7\u77ed\uff0c\u5efa\u8bae\u83b7\u53d6\u66f4\u591a\u5386\u53f2\u6570\u636e\u518d\u8fd0\u884c\u3002"
                )


            # ── Sector Affinity Analysis ──────────────────────────────────────────
            st.markdown("---")

            @st.fragment
            def sector_affinity_section(analysis_df):
                st.subheader("🧲 Sector Affinity Analysis | 板块相关性分析")
                st.caption(
                    "Rolling Pearson correlation between this stock's daily returns and each sector's "
                    "market-cap weighted index (PPI). Higher = this stock moves more like that sector's basket. "
                    "Useful even if the stock doesn't directly supply that sector."
                )

                window = st.select_slider(
                    "Rolling window (trading days)",
                    options=[5, 10, 20, 30, 60],
                    value=20,
                    key="sector_affinity_window",
                )

                # ── Target stock daily returns ────────────────────────────────────
                stock_ret = analysis_df["Close"].pct_change().dropna()

                # ── Load PPI sector indices from DB ───────────────────────────────
                sector_returns: dict = {}
                for sector_name in data_manager.get_sector_stock_map().keys():
                    table_name = f"PPI_{sector_name}"
                    try:
                        if not data_manager.db.table_exists(table_name):
                            continue
                        ppi_df = data_manager.db.read_table(
                            table_name, columns="Date,Close", order_by="Date"
                        )
                        if ppi_df is None or ppi_df.empty:
                            continue
                        ppi_df["Date"] = pd.to_datetime(ppi_df["Date"])
                        ppi_df = ppi_df.set_index("Date").sort_index()
                        sec_ret = ppi_df["Close"].pct_change().dropna()
                        if len(sec_ret) > 10:
                            sector_returns[sector_name] = sec_ret
                    except Exception:
                        continue

                if not sector_returns:
                    st.warning("No PPI sector data found in database. Run main.py to build sector indices first.")
                    return

                # ── Rolling Pearson correlation per sector ────────────────────────
                all_rolling: dict = {}
                for sector, sec_ret in sector_returns.items():
                    aligned = pd.concat(
                        [stock_ret.rename("stock"), sec_ret.rename(sector)], axis=1
                    ).dropna()
                    if len(aligned) < window + 5:
                        continue
                    rolling_corr = aligned["stock"].rolling(window=window).corr(aligned[sector])
                    all_rolling[sector] = rolling_corr.reindex(stock_ret.index)

                if not all_rolling:
                    st.warning("Not enough overlapping data to compute correlations.")
                    return

                # ── CSI 300 broad-market benchmark ────────────────────────────
                _CSI300_LABEL = "CSI 300 (沪深300)"
                try:
                    @st.cache_data(ttl=3600, show_spinner=False)
                    def _load_csi300_ret(lb):
                        _df = data_manager.get_index_data_live("000300.SH", lookback_days=lb)
                        if _df is None or _df.empty:
                            return None
                        return _df["Close"].pct_change().dropna()

                    _csi_ret = _load_csi300_ret(900)
                    if _csi_ret is not None:
                        _csi_aln = pd.concat(
                            [stock_ret.rename("stock"), _csi_ret.rename("csi")], axis=1
                        ).dropna()
                        if len(_csi_aln) >= window + 5:
                            _csi_roll = (
                                _csi_aln["stock"].rolling(window=window).corr(_csi_aln["csi"])
                            )
                            all_rolling[_CSI300_LABEL] = _csi_roll.reindex(stock_ret.index)
                except Exception:
                    pass

                # Current correlations — last non-NaN value in each series
                current_corr: dict = {}
                for s, series in all_rolling.items():
                    valid = series.dropna()
                    if not valid.empty:
                        current_corr[s] = valid.iloc[-1]

                if not current_corr:
                    st.warning("Could not compute current correlations.")
                    return

                corr_series = pd.Series(current_corr).sort_values(ascending=True)

                # ── Bar chart + side panel ────────────────────────────────────────
                col_bar, col_info = st.columns([3, 1])

                with col_bar:
                    _CSI300_BAR_COLOR = "rgba(251,191,36,0.95)"
                    bar_colors = [
                        _CSI300_BAR_COLOR if s == _CSI300_LABEL
                        else ("rgba(34,197,94,0.85)" if v >= 0 else "rgba(239,68,68,0.85)")
                        for s, v in zip(corr_series.index, corr_series.values)
                    ]
                    fig_bar = go.Figure(go.Bar(
                        x=corr_series.values,
                        y=corr_series.index.tolist(),
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:+.3f}" for v in corr_series.values],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        title=f"Current Sector Correlations (last {window}-day window)",
                        xaxis=dict(
                            range=[-1.15, 1.15],
                            zeroline=True,
                            zerolinecolor="rgba(200,200,200,0.4)",
                            title="Pearson r",
                        ),
                        yaxis=dict(title=""),
                        height=530,
                        margin=dict(l=10, r=90, t=40, b=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_info:
                    top3 = corr_series.sort_values(ascending=False).head(3)
                    bot3 = corr_series.sort_values(ascending=True).head(3)
                    st.markdown("**🔝 Most correlated**")
                    for s, v in top3.items():
                        st.markdown(f"- {s} `{v:+.3f}`")
                    st.markdown("")
                    st.markdown("**🔻 Least / negative**")
                    for s, v in bot3.items():
                        st.markdown(f"- {s} `{v:+.3f}`")

                # ── Rolling heatmap — last 252 trading days ───────────────────────
                rolling_df = pd.DataFrame(all_rolling)
                sector_order = corr_series.sort_values(ascending=False).index.tolist()
                valid_cols = [c for c in sector_order if c in rolling_df.columns]
                rolling_df = rolling_df[valid_cols].dropna(how="all").tail(252)

                date_labels = [
                    d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    for d in rolling_df.index
                ]

                fig_heat = go.Figure(go.Heatmap(
                    z=rolling_df.values.T,
                    x=date_labels,
                    y=rolling_df.columns.tolist(),
                    colorscale="RdYlGn",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="r", tickvals=[-1, -0.5, 0, 0.5, 1]),
                ))
                fig_heat.update_layout(
                    title=f"Rolling {window}-Day Correlation Heatmap (last 252 trading days)",
                    height=540,
                    margin=dict(l=10, r=20, t=40, b=70),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    xaxis=dict(showticklabels=True, tickangle=45, nticks=12),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # ── Trend insight: is top-sector affinity strengthening? ───────────
                if current_corr:
                    top_sector = corr_series.sort_values(ascending=False).index[0]
                    if top_sector in all_rolling:
                        top_series = all_rolling[top_sector].dropna().tail(60)
                        if len(top_series) >= 10:
                            recent_5  = top_series.tail(5).mean()
                            recent_20 = top_series.tail(20).mean()
                            trend_str = (
                                "📈 strengthening"
                                if recent_5 > recent_20 + 0.05
                                else ("📉 weakening" if recent_5 < recent_20 - 0.05 else "➡️ stable")
                            )
                            st.info(
                                f"**Dominant sector**: {top_sector} "
                                f"(r = {current_corr[top_sector]:+.3f})  "
                                f"— affinity is **{trend_str}** over the past 5 vs 20 days."
                            )

                # ── Sector Rotation Analysis ──────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 🔄 Sector Rotation | 板块轮动")
                st.caption(
                    "**Top panel**: rolling correlation of the 5 highest-affinity sectors "
                    "(grey = remaining sectors).  "
                    "**Bottom strip**: which sector 'owns' this stock's price action each day — "
                    "a colour change = a rotation event."
                )

                rolling_df_rot = pd.DataFrame(all_rolling).dropna(how="all").tail(252)

                if len(rolling_df_rot.columns) >= 2 and len(rolling_df_rot) >= window + 5:
                    rolling_filled = rolling_df_rot.ffill()
                    dominant_series = rolling_filled.idxmax(axis=1).dropna()

                    avg_corr_rot = rolling_filled.mean()
                    top5 = avg_corr_rot.sort_values(ascending=False).head(5).index.tolist()
                    sector_all = rolling_df_rot.columns.tolist()

                    _PALETTE = [
                        "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
                        "#8b5cf6", "#06b6d4", "#f97316", "#84cc16",
                        "#ec4899", "#64748b",
                    ]
                    color_map = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(sector_all)}
                    color_map[_CSI300_LABEL] = "rgba(251,191,36,0.95)"  # amber — distinct from sector lines

                    fig_rot = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.82, 0.18],
                        vertical_spacing=0.02,
                    )

                    # Gray background lines for non-top sectors
                    for _s in [s for s in sector_all if s not in top5]:
                        if _s in rolling_filled.columns:
                            fig_rot.add_trace(go.Scatter(
                                x=rolling_filled.index,
                                y=rolling_filled[_s],
                                mode="lines",
                                line=dict(color="rgba(130,130,130,0.18)", width=0.8),
                                showlegend=False,
                                hoverinfo="skip",
                            ), row=1, col=1)

                    # Top-5 sector lines
                    for _s in top5:
                        if _s in rolling_filled.columns:
                            fig_rot.add_trace(go.Scatter(
                                x=rolling_filled.index,
                                y=rolling_filled[_s],
                                name=_s,
                                mode="lines",
                                line=dict(color=color_map[_s], width=2),
                                legendgroup=_s,
                            ), row=1, col=1)

                    fig_rot.add_hline(
                        y=0, line_dash="dash",
                        line_color="rgba(200,200,200,0.25)",
                        row=1, col=1,
                    )

                    # Dominant-sector strip — run-length encode consecutive same-sector spans
                    if not dominant_series.empty:
                        _runs: list = []
                        _prev_sec = dominant_series.iloc[0]
                        _t0       = dominant_series.index[0]
                        for _t, _sec in dominant_series.items():
                            if _sec != _prev_sec:
                                _runs.append((_t0, _t, _prev_sec))
                                _t0, _prev_sec = _t, _sec
                        _runs.append((_t0, dominant_series.index[-1], _prev_sec))

                        _legend_shown: set = set()
                        for _t0_r, _t1_r, _sector in _runs:
                            # Invisible scatter so the sector appears in the legend strip
                            _show_leg = _sector not in top5 and _sector not in _legend_shown
                            if _show_leg:
                                _legend_shown.add(_sector)
                                fig_rot.add_trace(go.Scatter(
                                    x=[None], y=[None],
                                    mode="markers",
                                    marker=dict(color=color_map.get(_sector, "#64748b"), size=10, symbol="square"),
                                    name=_sector,
                                    legendgroup=_sector,
                                ), row=1, col=1)
                            fig_rot.add_shape(
                                type="rect",
                                x0=_t0_r, x1=_t1_r,
                                y0=0, y1=1,
                                fillcolor=color_map.get(_sector, "#64748b"),
                                line_width=0,
                                row=2, col=1,
                            )

                    fig_rot.update_layout(
                        height=540,
                        margin=dict(l=10, r=20, t=30, b=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.01, x=0,
                            font=dict(size=11),
                        ),
                        hovermode="x unified",
                    )
                    fig_rot.update_yaxes(
                        title_text="Pearson r", row=1, col=1,
                        range=[-1.05, 1.05],
                    )
                    fig_rot.update_yaxes(
                        showticklabels=False, row=2, col=1, range=[0, 1],
                    )
                    fig_rot.update_xaxes(showgrid=False, row=2, col=1)

                    st.plotly_chart(fig_rot, use_container_width=True)

                    # ── Rotation summary ──────────────────────────────────────────
                    if not dominant_series.empty:
                        _n_rot = int((dominant_series != dominant_series.shift(1)).sum()) - 1
                        _dom_counts = dominant_series.value_counts()

                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.metric("Distinct sectors that led", dominant_series.nunique())
                            st.markdown("**Days in the lead:**")
                            for _s, _cnt in _dom_counts.head(5).items():
                                _pct = _cnt / len(dominant_series) * 100
                                _bar = "█" * max(1, int(_pct / 5))
                                st.markdown(
                                    f"<div style='margin:3px 0;font-size:13px'>"
                                    f"<span style='color:{color_map.get(_s,'#64748b')}"
                                    f";font-weight:700'>{_s}</span>&nbsp;&nbsp;"
                                    f"{_bar}&nbsp;{_cnt}d&nbsp;({_pct:.0f}%)</div>",
                                    unsafe_allow_html=True,
                                )
                        with rc2:
                            st.metric("Rotation events (sector switches)", _n_rot)
                            if _n_rot > 40:
                                st.warning(
                                    "⚠️ High rotation — this stock shifts themes frequently; "
                                    "treat sector affinity as short-lived signals."
                                )
                            elif _n_rot < 8:
                                st.success(
                                    "✅ Stable sector identity — price action is consistently "
                                    "explained by one dominant sector."
                                )
                            else:
                                st.info(
                                    "ℹ️ Moderate rotation — anchored to a primary sector "
                                    "with occasional theme-driven shifts."
                                )
                else:
                    st.info("Not enough overlapping data for rotation analysis at the current window size.")

            sector_affinity_section(analysis_df)

else:
    st.info("👆 Enter a stock code above to begin analysis")
