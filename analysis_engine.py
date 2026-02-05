"""
Shared Technical Analysis Engine
Contains core analysis functions used across multiple pages
"""

import pandas as pd
import numpy as np
import ta
from scipy import stats

def run_single_stock_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced 3-Phase Trading System Analysis
    Phase 1: Accumulation (OBV rising while price flat - smart money)
    Phase 2: Squeeze (BB width contraction - energy building)
    Phase 3: Golden Launch (ADX + MACD trigger - breakout)
    
    Args:
        df: DataFrame with OHLC data (columns: Open, High, Low, Close, Volume)
    
    Returns:
        DataFrame with all technical indicators and signal columns
    """
    df_analysis = df.copy()
    
    # Ensure datetime index
    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        df_analysis.index = pd.to_datetime(df_analysis.index)
    df_analysis = df_analysis.sort_index()
    
    # ==================== MOVING AVERAGES ====================
    df_analysis['MA5'] = ta.trend.sma_indicator(df_analysis['Close'], window=5)
    df_analysis['EMA5'] = ta.trend.ema_indicator(df_analysis['Close'], window=5)
    df_analysis['MA20'] = ta.trend.sma_indicator(df_analysis['Close'], window=20)
    df_analysis['MA50'] = ta.trend.sma_indicator(df_analysis['Close'], window=50)
    df_analysis['MA200'] = ta.trend.sma_indicator(df_analysis['Close'], window=200)
    
    # ==================== BOLLINGER BANDS ====================
    bb = ta.volatility.BollingerBands(df_analysis['Close'], window=20, window_dev=2)
    df_analysis['BB_Upper'] = bb.bollinger_hband()
    df_analysis['BB_Lower'] = bb.bollinger_lband()
    df_analysis['BB_Width'] = (df_analysis['BB_Upper'] - df_analysis['BB_Lower']) / df_analysis['Close']
    
    # ==================== MACD ====================
    macd = ta.trend.MACD(df_analysis['Close'])
    df_analysis['MACD'] = macd.macd()
    df_analysis['MACD_Signal'] = macd.macd_signal()
    df_analysis['MACD_Hist'] = macd.macd_diff()
    
    # ==================== RSI ====================
    df_analysis['RSI_14'] = ta.momentum.RSIIndicator(df_analysis['Close'], window=14).rsi()
    
    # ==================== ADX ====================
    adx = ta.trend.ADXIndicator(df_analysis['High'], df_analysis['Low'], df_analysis['Close'], window=14)
    df_analysis['ADX'] = adx.adx()
    
    # ADX LOWESS Smoothing
    try:
        from scipy.signal import savgol_filter
        df_analysis['ADX_LOWESS'] = savgol_filter(
            df_analysis['ADX'].fillna(method='ffill'),
            window_length=11,
            polyorder=3
        )
    except:
        df_analysis['ADX_LOWESS'] = df_analysis['ADX'].ewm(span=9, adjust=False).mean()
    
    # ADX Bollinger Bands
    df_analysis['ADX_BB_Middle'] = df_analysis['ADX'].rolling(window=20).mean()
    df_analysis['ADX_BB_Std'] = df_analysis['ADX'].rolling(window=20).std()
    df_analysis['ADX_BB_Upper'] = df_analysis['ADX_BB_Middle'] + 2 * df_analysis['ADX_BB_Std']
    df_analysis['ADX_BB_Lower'] = df_analysis['ADX_BB_Middle'] - 2 * df_analysis['ADX_BB_Std']
    
    # ==================== OBV ====================
    df_analysis['OBV'] = ta.volume.on_balance_volume(df_analysis['Close'], df_analysis['Volume'])
    df_analysis['Volume_Scaled_OBV'] = df_analysis['OBV'] / df_analysis['Volume'].rolling(window=20).mean()
    
    # ==================== VOLUME STATS ====================
    df_analysis['VolMean_100d'] = df_analysis['Volume'].rolling(window=100, min_periods=20).mean()
    df_analysis['VolStd_100d'] = df_analysis['Volume'].rolling(window=100, min_periods=20).std()
    df_analysis['Volume_ZScore'] = (df_analysis['Volume'] - df_analysis['VolMean_100d']) / df_analysis['VolStd_100d']
    
    # ==================== ADAPTIVE PARAMETERS ====================
    def get_adaptive_lookback(df_length, min_days=60, max_days=250):
        if df_length < 120:
            return max(min_days, int(df_length * 0.5))
        elif df_length < 250:
            return 120
        else:
            return max_days
    
    lookback_period = get_adaptive_lookback(len(df_analysis))
    params = calculate_adaptive_parameters_percentile(df, lookback_days=30)
    lookback = params['obv_lookback']
    
    # ==================== INITIALIZE SIGNAL COLUMNS ====================
    df_analysis['Signal_Accumulation'] = False
    df_analysis['Signal_Squeeze'] = False
    df_analysis['Signal_Golden_Launch'] = False
    df_analysis['Exit_MACD_Lead'] = False
    
    # ==================== PHASE 1: ACCUMULATION ====================
    df_analysis['PriceChg'] = df_analysis['Close'].pct_change(periods=lookback)
    df_analysis[f'OBVChg{lookback}d'] = df_analysis['OBV'].pct_change(periods=lookback)
    
    accumulation = (
        (df_analysis[f'OBVChg{lookback}d'] > params['obv_threshold']) &  # OBV up
        (df_analysis['PriceChg'].abs() < params['price_flat_threshold']) &  # Price flat
        (df_analysis['RSI_14'] < 60)  # Not overbought
    )
    df_analysis.loc[accumulation, 'Signal_Accumulation'] = True
    
    # ==================== PHASE 2: SQUEEZE ====================
    df_analysis['BB_Width_Percentile'] = df_analysis['BB_Width'].rolling(
        window=lookback_period, min_periods=20
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan)
    
    df_analysis['Squeeze_Raw'] = df_analysis['BB_Width_Percentile'] <= 0.10
    
    def consecutive_days_filter(series, min_days=3):
        if series.sum() == 0:
            return pd.Series(False, index=series.index)
        groups = (series != series.shift()).cumsum()
        count = series.groupby(groups).transform('size')
        return (series) & (count >= min_days)
    
    df_analysis['Signal_Squeeze'] = consecutive_days_filter(df_analysis['Squeeze_Raw'], min_days=3)
    
    squeeze_groups = (df_analysis['Signal_Squeeze'] != df_analysis['Signal_Squeeze'].shift()).cumsum()
    df_analysis['Squeeze_Age'] = df_analysis.groupby(squeeze_groups)['Signal_Squeeze'].cumsum()
    df_analysis['Squeeze_Mature'] = (df_analysis['Signal_Squeeze']) & (df_analysis['Squeeze_Age'] >= 5)
    
    # Keep legacy column for backward compatibility
    df_analysis['Min_Width_120d'] = df_analysis['BB_Width'].rolling(window=120, min_periods=20).min()
    
    # ==================== ADX TREND & ACCELERATION ====================
    def calculate_adx_trend(series, window=5):
        def get_slope(y):
            if len(y) < 3 or y.isna().any():
                return 0
            x = np.arange(len(y))
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        return series.rolling(window=window, min_periods=3).apply(get_slope, raw=False)
    
    df_analysis['ADX_Slope'] = calculate_adx_trend(df_analysis['ADX_LOWESS'], window=5)
    df_analysis['ADX_Acceleration'] = df_analysis['ADX_Slope'] - df_analysis['ADX_Slope'].shift(1)
    
    # ==================== ADX PATTERN DETECTION (COMPLETE - 9 PATTERNS) ====================
    
    # Pattern 1: BOTTOMING (Low ADX, falling but slowing down)
    adx_bottoming = (
        (df_analysis['ADX'] < 25) &
        (df_analysis['ADX_Slope'] < 0) &
        (df_analysis['ADX_Acceleration'] > 0.15) &
        (df_analysis['ADX_Slope'] > df_analysis['ADX_Slope'].shift(1))
    )
    
    # Pattern 2: REVERSING UP (ADX just turned from down to up)
    adx_reversing_up = (
        (df_analysis['ADX'] < 30) &
        (df_analysis['ADX_Slope'] > 0) &
        (df_analysis['ADX_Slope'].shift(1) <= 0)
    )
    
    # Pattern 3: ACCELERATING UP (ADX rising and speeding up)
    adx_accelerating_up = (
        (df_analysis['ADX_Slope'] > 0.5) &
        (df_analysis['ADX_Acceleration'] > 0.1)
    )
    
    # Pattern 4: STRONG & STABLE (High ADX, relatively stable)
    adx_strong_stable = (
        (df_analysis['ADX'] >= 30) &
        (df_analysis['ADX_Slope'] >= -0.3) &
        (df_analysis['ADX_Slope'] <= 0.3)
    )
    
    # Pattern 5: DECELERATING UP (ADX rising but slowing down)
    adx_decelerating_up = (
        (df_analysis['ADX'] > 25) &
        (df_analysis['ADX_Slope'] > 0) &
        (df_analysis['ADX_Acceleration'] < -0.1)
    )
    
    # Pattern 6: PEAKING (High ADX, still rising but losing steam)
    adx_peaking = (
        (df_analysis['ADX'] >= 30) &
        (df_analysis['ADX_Slope'] > 0) &
        (df_analysis['ADX_Acceleration'] < -0.15) &
        (df_analysis['ADX_Slope'] < df_analysis['ADX_Slope'].shift(1))
    )
    
    # Pattern 7: REVERSING DOWN (ADX just turned from up to down)
    adx_reversing_down = (
        (df_analysis['ADX'] >= 25) &
        (df_analysis['ADX_Slope'] < 0) &
        (df_analysis['ADX_Slope'].shift(1) > 0)
    )
    
    # Pattern 8: ACCELERATING DOWN (ADX falling and speeding up downward)
    adx_accelerating_down = (
        (df_analysis['ADX_Slope'] < -0.5) &
        (df_analysis['ADX_Acceleration'] < -0.1)
    )
    
    # Pattern 9: DECELERATING DOWN (ADX falling but slowing)
    adx_decelerating_down = (
        (df_analysis['ADX'] < 30) &
        (df_analysis['ADX_Slope'] < 0) &
        (df_analysis['ADX_Acceleration'] > 0.1)
    )
    
    # Assign patterns (priority order matters!)
    df_analysis['ADX_Pattern'] = 'Neutral'
    
    # Low ADX states (potential trend starting)
    df_analysis.loc[adx_bottoming, 'ADX_Pattern'] = 'Bottoming'
    df_analysis.loc[adx_reversing_up, 'ADX_Pattern'] = 'Reversing Up'
    df_analysis.loc[adx_accelerating_up, 'ADX_Pattern'] = 'Accelerating Up'
    
    # High ADX states (strong trend or exhaustion)
    df_analysis.loc[adx_strong_stable, 'ADX_Pattern'] = 'Strong Trend'
    df_analysis.loc[adx_decelerating_up, 'ADX_Pattern'] = 'Losing Steam'
    df_analysis.loc[adx_peaking, 'ADX_Pattern'] = 'Peaking'
    
    # Falling ADX states (trend weakening)
    df_analysis.loc[adx_reversing_down, 'ADX_Pattern'] = 'Reversing Down'
    df_analysis.loc[adx_accelerating_down, 'ADX_Pattern'] = 'Accelerating Down'
    df_analysis.loc[adx_decelerating_down, 'ADX_Pattern'] = 'Slowing Down'
    
    # Create flags for healthy vs warning conditions
    adx_healthy = adx_bottoming | adx_reversing_up | adx_accelerating_up | adx_strong_stable
    adx_warning = adx_peaking | adx_reversing_down | adx_accelerating_down
    
    # ==================== MACD SCENARIOS ====================
    df_analysis['MACD_Gap'] = df_analysis['MACD'] - df_analysis['MACD_Signal']
    df_analysis['MACD_Momentum'] = df_analysis['MACD'] - df_analysis['MACD'].shift(1)
    df_analysis['MACD_Momentum_Pct'] = df_analysis['MACD'].pct_change()
    df_analysis['MACD_Acceleration'] = df_analysis['MACD_Momentum_Pct'] - df_analysis['MACD_Momentum_Pct'].shift(1)
    
    # Scenario 1: Classic crossover
    classic_crossover = (
        (df_analysis['MACD_Gap'].shift(1) <= 0) &
        (df_analysis['MACD_Gap'] > 0)
    )
    
    # Scenario 2: Approaching from below
    approaching_from_below = (
        (df_analysis['MACD_Gap'] < 0) &
        (df_analysis['MACD_Gap'] > -0.5 * df_analysis['MACD_Signal'].abs()) &
        (df_analysis['MACD_Gap'] > df_analysis['MACD_Gap'].shift(1)) &
        (df_analysis['MACD_Gap'].shift(1) > df_analysis['MACD_Gap'].shift(2)) &
        (df_analysis['MACD_Momentum_Pct'] > 0) &
        (
            (df_analysis['MACD_Acceleration'] > 0.03) |
            (
                (df_analysis['MACD_Momentum_Pct'].shift(2) < -0.05) &
                (df_analysis['MACD_Momentum_Pct'] > -0.02)
            )
        )
    )
    
    # Scenario 3: Bottoming
    bottoming = (
        (df_analysis['MACD_Gap'] < 0) &
        (df_analysis['MACD_Gap'].shift(1) < df_analysis['MACD_Gap'].shift(2)) &
        (df_analysis['MACD'].shift(1) < df_analysis['MACD'].shift(2)) &
        (df_analysis['MACD'] > df_analysis['MACD'].shift(1)) &
        (df_analysis['MACD_Gap'] > df_analysis['MACD_Gap'].shift(1))
    )
    
    # Scenario 4: Momentum building (2-day acceleration)
    df_analysis['MACD_Gap_Ratio'] = df_analysis['MACD_Gap'] / df_analysis['MACD_Signal'].abs()
    df_analysis['MACD_Gap_Ratio_Change'] = (
        df_analysis['MACD_Gap_Ratio'] - df_analysis['MACD_Gap_Ratio'].shift(1)
    )
    
    momentum_building = (
        (df_analysis['MACD_Gap'] > 0) &
        (df_analysis['MACD_Gap'].shift(1) > 0) &
        (df_analysis['MACD_Gap'].shift(2) > 0) &
        (df_analysis['MACD_Gap'] > df_analysis['MACD_Gap'].shift(1)) &
        (df_analysis['MACD_Gap_Ratio'] > df_analysis['MACD_Gap_Ratio'].shift(1)) &
        (df_analysis['MACD_Gap_Ratio'].shift(1) > df_analysis['MACD_Gap_Ratio'].shift(2)) &
        (df_analysis['MACD_Gap_Ratio_Change'] > 0.10) &
        (df_analysis['MACD_Gap_Ratio_Change'].shift(1) > 0.10) &
        (df_analysis['MACD'] > df_analysis['MACD'].shift(1))
    )
    
    macd_trigger = classic_crossover | approaching_from_below | bottoming | momentum_building
    
    # Scenario 5: Peaking (BEARISH)
    peaking = (
        (df_analysis['MACD_Gap'] > 0) &
        (df_analysis['MACD'].shift(1) > df_analysis['MACD'].shift(2)) &
        (df_analysis['MACD'] < df_analysis['MACD'].shift(1)) &
        (df_analysis['MACD_Momentum_Pct'] < df_analysis['MACD_Momentum_Pct'].shift(1))
    )
    
    # Scenario 6: Bearish crossover (BEARISH)
    bearish_crossover = (
        (df_analysis['MACD_Gap'].shift(1) > 0) &
        (df_analysis['MACD_Gap'] <= 0) &
        (df_analysis['MACD'].shift(1) > 0)
    )
    
    # Store MACD scenario columns
    df_analysis['MACD_ClassicCrossover'] = classic_crossover
    df_analysis['MACD_Approaching'] = approaching_from_below
    df_analysis['MACD_Bottoming'] = bottoming
    df_analysis['MACD_MomentumBuilding'] = momentum_building
    df_analysis['MACD_Trigger'] = macd_trigger
    df_analysis['MACD_Peaking'] = peaking
    df_analysis['MACD_BearishCrossover'] = bearish_crossover
    
    # ==================== MACD SIGNAL LINE TREND ====================
    df_analysis['MACD_Signal_Slope'] = df_analysis['MACD_Signal'] - df_analysis['MACD_Signal'].shift(2)
    df_analysis['Large_Uptrend'] = df_analysis['MACD_Signal_Slope'] > 0
    df_analysis['Large_Downtrend'] = df_analysis['MACD_Signal_Slope'] < 0
    
    # ==================== RSI DYNAMIC PERCENTILE THRESHOLDS ====================
    lookback_window = min(252, len(df_analysis))  # Use 1 year or available data
    
    if 'RSI_14' in df_analysis.columns:
        df_analysis['RSI_P10'] = df_analysis['RSI_14'].rolling(
            window=lookback_window, min_periods=60
        ).quantile(0.10)
        
        df_analysis['RSI_P90'] = df_analysis['RSI_14'].rolling(
            window=lookback_window, min_periods=60
        ).quantile(0.90)
        
        # Bottoming: Must be BOTH in bottom 10% AND <= 30
        df_analysis['RSI_Bottoming'] = (
            (df_analysis['RSI_14'] <= df_analysis['RSI_P10']) &
            (df_analysis['RSI_14'] <= 30)
        )
        
        # Peaking: Must be BOTH in top 10% AND >= 70
        df_analysis['RSI_Peaking'] = (
            (df_analysis['RSI_14'] >= df_analysis['RSI_P90']) &
            (df_analysis['RSI_14'] >= 70)
        )
    
    # ==================== PHASE 3: GOLDEN LAUNCH ====================
    launch = (
        macd_trigger &
        (df_analysis['MA20'] > df_analysis['MA50']) &
        adx_healthy &
        (df_analysis['RSI_14'] <= 70) &
        (df_analysis['Volume'] > df_analysis['Volume'].rolling(5).mean())
    )
    df_analysis.loc[launch, 'Signal_Golden_Launch'] = True
    
    # ==================== EXIT SIGNALS ====================
    macd_bearish_cross = (
        (df_analysis['MACD'].shift(1) > df_analysis['MACD_Signal'].shift(1)) &
        (df_analysis['MACD'] < df_analysis['MACD_Signal']) &
        (df_analysis['MACD'].shift(1) > 0) &
        (df_analysis['MACD_Hist'] < df_analysis['MACD_Hist'].shift(1))
    )
    
    ma_cross_down = (
        (df_analysis['MA20'].shift(1) > df_analysis['MA50'].shift(1)) &
        (df_analysis['MA20'] < df_analysis['MA50']) &
        (df_analysis['ADX'] > 20)
    )
    
    exit_signal = macd_bearish_cross | ma_cross_down
    df_analysis.loc[exit_signal, 'Exit_MACD_Lead'] = True
    
    # ==================== SIGNAL SCORE ====================
    df_analysis['Signal_Score'] = 0
    
    # POSITIVE INDICATORS (+1 each)
    if 'ADX_Pattern' in df_analysis.columns:
        df_analysis.loc[df_analysis['ADX_Pattern'] == 'Bottoming', 'Signal_Score'] += 1
        df_analysis.loc[df_analysis['ADX_Pattern'] == 'Reversing Up', 'Signal_Score'] += 1
        df_analysis.loc[df_analysis['ADX_Pattern'] == 'Accelerating Up', 'Signal_Score'] += 1
        df_analysis.loc[df_analysis['ADX_Pattern'] == 'Strong Trend', 'Signal_Score'] += 1
    
    if 'RSI_Bottoming' in df_analysis.columns:
        df_analysis.loc[df_analysis['RSI_Bottoming'], 'Signal_Score'] += 1
    
    if 'MACD_Bottoming' in df_analysis.columns:
        df_analysis.loc[df_analysis['MACD_Bottoming'], 'Signal_Score'] += 1
    
    if 'MACD_Approaching' in df_analysis.columns:
        df_analysis.loc[df_analysis['MACD_Approaching'], 'Signal_Score'] += 1
    
    if 'MACD_ClassicCrossover' in df_analysis.columns:
        df_analysis.loc[df_analysis['MACD_ClassicCrossover'], 'Signal_Score'] += 1
    
    if 'MACD_MomentumBuilding' in df_analysis.columns:
        df_analysis.loc[df_analysis['MACD_MomentumBuilding'], 'Signal_Score'] += 1
    
    if 'Signal_Accumulation' in df_analysis.columns:
        df_analysis.loc[df_analysis['Signal_Accumulation'], 'Signal_Score'] += 1
    
    # NEGATIVE INDICATORS (-1 each)
    if 'MACD_Peaking' in df_analysis.columns:
        df_analysis.loc[df_analysis['MACD_Peaking'], 'Signal_Score'] -= 1
    
    if 'MACD_BearishCrossover' in df_analysis.columns:
        df_analysis.loc[df_analysis['MACD_BearishCrossover'], 'Signal_Score'] -= 1
    
    if 'RSI_Peaking' in df_analysis.columns:
        df_analysis.loc[df_analysis['RSI_Peaking'], 'Signal_Score'] -= 1
    
    if 'ADX_Pattern' in df_analysis.columns:
        df_analysis.loc[df_analysis['ADX_Pattern'] == 'Peaking', 'Signal_Score'] -= 1
        df_analysis.loc[df_analysis['ADX_Pattern'] == 'Reversing Down', 'Signal_Score'] -= 1
        df_analysis.loc[df_analysis['ADX_Pattern'] == 'Accelerating Down', 'Signal_Score'] -= 1
    
    # Calculate cumulative score
    df_analysis['Cumulative_Score'] = df_analysis['Signal_Score'].cumsum()
    
    return df_analysis


def calculate_adaptive_parameters_percentile(df, lookback_days=30):
    """
    Calculate adaptive parameters based on recent volatility.
    
    Args:
        df: DataFrame with OHLC data
        lookback_days: Number of days to look back for volatility calculation
    
    Returns:
        Dictionary with adaptive parameters
    """
    df_temp = df.copy()
    
    # Calculate volatility windows
    for window in [10, 15, 20, 25, 30]:
        df_temp[f'vol_{window}d'] = df_temp['Close'].pct_change().rolling(window).std()
    
    current_vol_10d = df_temp['vol_10d'].iloc[-1]
    vol_5days_ago = df_temp['vol_10d'].iloc[-5] if len(df_temp) >= 5 else current_vol_10d
    
    # Determine volatility trend
    vol_trend = 'rising' if current_vol_10d > vol_5days_ago * 1.2 else \
                'falling' if current_vol_10d < vol_5days_ago * 0.8 else 'stable'
    
    recent_vol = df_temp['vol_10d'].iloc[-lookback_days:].dropna()
    
    if len(recent_vol) < 10:
        return {
            'vol_window': 20,
            'obv_lookback': 10,
            'obv_threshold': 0.025,
            'current_vol': current_vol_10d,
            'vol_regime': 'insufficient_data',
            'price_flat_threshold': 0.05
        }
    
    p25, p50, p75, p90 = recent_vol.quantile([0.25, 0.50, 0.75, 0.90])
    
    # Determine regime based on percentile
    if current_vol_10d >= p90:
        vol_regime, percentile, vol_window, obv_lookback = 'very_high', 90, 10, 5
    elif current_vol_10d >= p75:
        vol_regime, percentile, vol_window, obv_lookback = 'high', 75, 15, 7
    elif current_vol_10d >= p50:
        vol_regime, percentile, vol_window, obv_lookback = 'medium_high', 60, 20, 10
    elif current_vol_10d >= p25:
        vol_regime, percentile, vol_window, obv_lookback = 'medium_low', 40, 25, 12
    else:
        vol_regime, percentile, vol_window, obv_lookback = 'low', 20, 30, 15
    
    # Adjust for trend
    if vol_trend == 'rising' and vol_regime in ['high', 'very_high']:
        vol_window = max(10, vol_window - 5)
        obv_lookback = max(5, obv_lookback - 2)
    
    current_vol = df_temp[f'vol_{min(vol_window, 30)}d'].iloc[-1]
    obv_threshold = max(0.008, min(0.08, current_vol * obv_lookback * 0.35))
    price_flat_threshold = max(0.03, min(0.08, current_vol * obv_lookback * 0.5))
    
    return {
        'vol_window': vol_window,
        'current_vol': current_vol,
        'vol_percentile': percentile,
        'vol_regime': vol_regime,
        'vol_trend': vol_trend,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'obv_lookback': obv_lookback,
        'obv_threshold': obv_threshold,
        'price_flat_threshold': price_flat_threshold,
        'cycle_estimate': obv_lookback * 2
    }
