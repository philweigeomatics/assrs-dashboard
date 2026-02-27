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
    


    # ==================== MARKET REGIME DETECTION ====================
    # MUST be called before MACD so we have historical regimes day-by-day
    df_analysis = detect_market_regime(df_analysis)

    # ==================== POINT-IN-TIME DYNAMIC MACD ====================
    # 1. Calculate all three speeds for the entire history
    macd_fast = ta.trend.MACD(df_analysis['Close'], window_fast=8, window_slow=21, window_sign=5)
    macd_norm = ta.trend.MACD(df_analysis['Close'], window_fast=12, window_slow=26, window_sign=9)
    macd_slow = ta.trend.MACD(df_analysis['Close'], window_fast=15, window_slow=30, window_sign=9)

    # 2. Stitch them together day-by-day based on the historical regime
    import numpy as np
    
    cond_high = df_analysis['Market_Regime'] == 'High Volatility'
    cond_low = df_analysis['Market_Regime'] == 'Low Volatility'
    
    # Select MACD Line
    df_analysis['MACD'] = np.where(cond_high, macd_fast.macd(),
                          np.where(cond_low, macd_slow.macd(),
                                   macd_norm.macd()))
                                   
    # Select Signal Line
    df_analysis['MACD_Signal'] = np.where(cond_high, macd_fast.macd_signal(),
                                 np.where(cond_low, macd_slow.macd_signal(),
                                          macd_norm.macd_signal()))
                                          
    # Calculate final Histogram
    df_analysis['MACD_Hist'] = df_analysis['MACD'] - df_analysis['MACD_Signal']
    
    # # Save today's active parameters for the UI Simulator to use
    # latest_regime = df_analysis['Market_Regime'].iloc[-1]
    # df_analysis['MACD_Fast_Param'] = 8 if latest_regime == 'High Volatility' else 15 if latest_regime == 'Low Volatility' else 12
    # df_analysis['MACD_Slow_Param'] = 21 if latest_regime == 'High Volatility' else 30 if latest_regime == 'Low Volatility' else 26
    # df_analysis['MACD_Sign_Param'] = 5 if latest_regime == 'High Volatility' else 9
    # =========================================================
    
    # ==================== MACD ====================
    # macd = ta.trend.MACD(df_analysis['Close'])
    # df_analysis['MACD'] = macd.macd()
    # df_analysis['MACD_Signal'] = macd.macd_signal()
    # df_analysis['MACD_Hist'] = macd.macd_diff()

    # 3. Save the active parameters day-by-day so the UI can prove the gears are shifting!
    df_analysis['MACD_Fast_Param'] = np.where(cond_high, 8, np.where(cond_low, 15, 12))
    df_analysis['MACD_Slow_Param'] = np.where(cond_high, 21, np.where(cond_low, 30, 26))
    df_analysis['MACD_Sign_Param'] = np.where(cond_high, 5, np.where(cond_low, 9, 9))
    
    # (Do NOT put the standard ta.trend.MACD calculation here anymore!)
    
    # ==================== RSI ====================
    df_analysis['RSI_14'] = ta.momentum.RSIIndicator(df_analysis['Close'], window=14).rsi()
    
    # ==================== ADX ====================
    adx = ta.trend.ADXIndicator(df_analysis['High'], df_analysis['Low'], df_analysis['Close'], window=14)
    df_analysis['ADX'] = adx.adx()
    df_analysis['DI_Plus'] = adx.adx_pos()    # ← ADD THIS
    df_analysis['DI_Minus'] = adx.adx_neg()   # ← ADD THIS

    # DI Crossover signals
    df_analysis['DI_Bullish_Cross'] = (
        (df_analysis['DI_Plus'] > df_analysis['DI_Minus']) &
        (df_analysis['DI_Plus'].shift(1) <= df_analysis['DI_Minus'].shift(1))
    )
    df_analysis['DI_Bearish_Cross'] = (
        (df_analysis['DI_Minus'] > df_analysis['DI_Plus']) &
        (df_analysis['DI_Minus'].shift(1) <= df_analysis['DI_Plus'].shift(1))
    )
    
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
        (df_analysis['ADX'] < 22) &
        (df_analysis['ADX_Slope'] < 0) &
        (df_analysis['ADX_Slope'] > -0.15) &  # But BARELY falling (almost flat)
        (df_analysis['ADX_Acceleration'] > 0.05)
    )
    
    # Pattern 2: REVERSING UP (ADX just turned from down to up)
    adx_reversing_up = (
        (df_analysis['ADX'] < 25 ) &
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
        (df_analysis['ADX_Slope'] < -0.3 ) &
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
    
    # Scenario 3: Bottoming (MACD in bottom zone with recovery signs)
    # Calculate rolling stats for context
    df_analysis['MACD_5d_Low'] = df_analysis['MACD'].rolling(5).min()
    df_analysis['MACD_10d_Low'] = df_analysis['MACD'].rolling(10).min()

    bottoming = (
        (df_analysis['MACD'] < 0) &
        # Tighter window: within 5% of low instead of 15%
        (df_analysis['MACD'] >= df_analysis['MACD_10d_Low']) &
        (df_analysis['MACD'] <= df_analysis['MACD_10d_Low'] * 0.95) &  # Changed from 0.85 to 0.95
        
        # Require at least 2 out of 4 signals:
        (
            (
                # first detect that MACD is coming back from yesterday, meaning that yesterday is even lower.
                # 2nd is that the MACD today is higher than the lowest of the last 5 days, showing that it's trying to come off the bottom.
                # 3rd is that the GAP between MACD and signal line is narrowing, showing that the signal line hasn't picked up yet.
                # 4th is that OBV is rising, showing that volume is supporting the move.
                (df_analysis['MACD'] >= df_analysis['MACD'].shift(1)).astype(int) +
                (df_analysis['MACD'] > df_analysis['MACD_5d_Low']).astype(int) +
                (df_analysis['MACD_Gap'] > df_analysis['MACD_Gap'].shift(1)).astype(int) +
                (df_analysis['OBV'] > df_analysis['OBV'].shift(3)).astype(int)
            ) >= 4  # Need at least 2 signals
        )
    )

    def debug_macd_bottoming(df_analysis, target_date='2025-09-10'):
  
        try:
            import pandas as pd
            target = pd.to_datetime(target_date)
            
            # Get 10 days leading up to and including target
            mask = (df_analysis.index >= target - pd.Timedelta(days=15)) & \
                (df_analysis.index <= target + pd.Timedelta(days=2))
            
            if not mask.any():
                print(f"❌ Date {target_date} not found in data")
                return
            
            print("\n" + "="*120)
            print(f"🔍 MACD BOTTOMING DEBUG - Last 10 Days Around {target_date}")
            print("="*120)
            
            debug_rows = df_analysis[mask].tail(10).copy()
            
            print("\n📊 RAW VALUES (Last 10 Days):")
            print("-"*120)
            raw_cols = ['MACD', 'MACD_Signal', 'MACD_Gap', 'OBV', 'MACD_5d_Low', 'MACD_10d_Low']
            raw_df = debug_rows[raw_cols].copy()
            raw_df.index = raw_df.index.strftime('%Y-%m-%d')
            print(raw_df.to_string())
            
            print("\n" + "="*120)
            print("🔬 CONDITION CHECKS (Last 10 Days):")
            print("-"*120)
            
            # Now check conditions for each row
            debug_data = []
            for idx in debug_rows.index:
                row = df_analysis.loc[idx]
                idx_loc = df_analysis.index.get_loc(idx)
                
                # Get previous values safely
                macd_prev = df_analysis.iloc[idx_loc-1]['MACD'] if idx_loc > 0 else None
                gap_prev = df_analysis.iloc[idx_loc-1]['MACD_Gap'] if idx_loc > 0 else None
                obv_prev3 = df_analysis.iloc[idx_loc-3]['OBV'] if idx_loc >= 3 else None
                
                # Calculate conditions
                cond1_macd_neg = row['MACD'] < 0
                cond2_within_15pct = row['MACD'] <= row['MACD_10d_Low'] * 1.15
                
                signal1_stopped = (row['MACD'] >= macd_prev) if macd_prev is not None else False
                signal2_off_low = row['MACD'] > row['MACD_5d_Low']
                signal3_gap_narrow = (row['MACD_Gap'] > gap_prev) if gap_prev is not None else False
                signal4_obv_rising = (row['OBV'] > obv_prev3) if obv_prev3 is not None else False
                
                any_signal = signal1_stopped or signal2_off_low or signal3_gap_narrow or signal4_obv_rising
                bottoming_detected = cond1_macd_neg and cond2_within_15pct and any_signal
                
                debug_data.append({
                    'Date': idx.strftime('%m-%d'),
                    'MACD<0': '✓' if cond1_macd_neg else '✗',
                    'Within15%': '✓' if cond2_within_15pct else '✗',
                    '├─10d_Low*1.15': f"{row['MACD_10d_Low']*1.15:.4f}",
                    'Stopped↑': '✓' if signal1_stopped else '✗',
                    '├─Δ': f"{row['MACD']-macd_prev:.4f}" if macd_prev else 'N/A',
                    'Off_5d_Low': '✓' if signal2_off_low else '✗',
                    'Gap_Narrow': '✓' if signal3_gap_narrow else '✗',
                    '├─Gap_Δ': f"{row['MACD_Gap']-gap_prev:.4f}" if gap_prev else 'N/A',
                    'OBV_Up3d': '✓' if signal4_obv_rising else '✗',
                    '├─OBV_Δ': f"{(row['OBV']-obv_prev3)/obv_prev3*100:.1f}%" if obv_prev3 else 'N/A',
                    'ANY': '✓' if any_signal else '✗',
                    '🎯': '✅' if bottoming_detected else '❌'
                })
            
            df_debug = pd.DataFrame(debug_data)
            print(df_debug.to_string(index=False))
            
            print("\n" + "="*120)
            print("📋 WHAT TO LOOK FOR:")
            print("  1. Check RAW VALUES: Is MACD actually at/near bottom on Sep 10?")
            print("  2. Check MACD vs 10d_Low*1.15: Is MACD within the 15% threshold?")
            print("  3. Check Δ (delta) values: Did MACD stop falling? Did gap narrow?")
            print("  4. If all look correct but still ❌, the threshold (15%, 3d, etc.) is too strict")
            print("\n💡 ADJUSTMENTS:")
            print("  - If MACD outside 15%: Increase to 1.20 or 1.25")
            print("  - If Stopped always ✗: MACD is still falling (expected at true bottom)")
            print("  - If Gap_Narrow always ✗: Gap still widening (classic late detection)")
            print("  - If OBV_Up3d ✗: Try OBV.shift(5) or remove OBV requirement")
            print("="*120 + "\n")
            
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            import traceback
            traceback.print_exc()

    # Call it
    # debug_macd_bottoming(df_analysis, target_date='2025-09-10')
    
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
        (df_analysis['MACD'] > 0 ) &
        (df_analysis['MACD_Signal'] > 0 ) &
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
    
    # ==================== PHASE 3: GOLDEN LAUNCH (Old Code) ====================
    # launch = (
    #     macd_trigger &
    #     (df_analysis['MA20'] > df_analysis['MA50']) &
    #     adx_healthy &
    #     (df_analysis['RSI_14'] <= 70) &
    #     (df_analysis['Volume'] > df_analysis['Volume'].rolling(5).mean())
    # )
    # df_analysis.loc[launch, 'Signal_Golden_Launch'] = True
    
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


    # ==================== MARKET REGIME DETECTION ====================
    # df_analysis = detect_market_regime(df_analysis)
    # this was moved up so it detect regime before MACD.
    
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



from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy import stats

# INTERNAL TOGGLE: Set this to 'HMM', 'JUMP', or 'ATR'
REGIME_METHOD = 'HMM' 

def detect_market_regime(
    df: pd.DataFrame,
    freq: str = "daily",
    method: str | None = None,
    fallback_method: str | None = "JUMP",
    window: int | None = None,
    refit_every: int | None = None,
    min_obs: int | None = None,
    z_clip: float = 3.0,
):
    """
    Volatility/Liquidity regime detection.

    Backward compatible:
      - existing calls: detect_market_regime(df) still work (defaults to daily + global REGIME_METHOD)
    Robust:
      - auto fallback from HMM -> JUMP if insufficient data or fit issues
      - weekly-safe (won't crash due to uninitialized model/rank_map)
    """

    if df is None or df.empty:
        return df

    # Resolve method locally (do NOT mutate global REGIME_METHOD)
    m = (method or REGIME_METHOD).upper()

    tf = (freq or "daily").lower()
    is_weekly = ("week" in tf) or tf.startswith("w") or ("周" in tf)

    # Defaults scale with timeframe
    if window is None:
        window = 26 if is_weekly else 90
    if refit_every is None:
        refit_every = 2 if is_weekly else 5
    if min_obs is None:
        # enough for rolling z-score + stable HMM
        min_obs = 70 if is_weekly else 120

    # If not enough data, fallback if HMM was requested
    if len(df) < min_obs:
        if m == "HMM" and fallback_method:
            return detect_market_regime(df, freq=freq, method=fallback_method, fallback_method=None)
        df = df.copy()
        df["Market_Regime"] = "Normal Volatility"
        df["ATR_Percentile"] = 0.5
        return df

    # =========================
    # HMM MODE
    # =========================
    if m == "HMM":
        df = df.copy()

        vol_ma_len = 10 if is_weekly else 20
        ewm_span = 2 if is_weekly else 3

        # True range & NATR
        df["prev_close"] = df["Close"].shift(1)
        df["h_l"] = df["High"] - df["Low"]
        df["h_pc"] = (df["High"] - df["prev_close"]).abs()
        df["l_pc"] = (df["Low"] - df["prev_close"]).abs()
        df["TR"] = df[["h_l", "h_pc", "l_pc"]].max(axis=1)
        df["NATR"] = (df["TR"] / df["Close"]) * 100

        # RVOL (log)
        df["Vol_MA"] = df["Volume"].rolling(vol_ma_len).mean()
        df["RVOL"] = df["Volume"] / (df["Vol_MA"] + 1)
        df["Log_RVOL"] = np.log(df["RVOL"] + 0.1)

        # Smooth then rolling z-score
        feat_df = df[["NATR", "Log_RVOL"]].copy()
        feat_df["NATR"] = feat_df["NATR"].ewm(span=ewm_span).mean()
        feat_df["Log_RVOL"] = feat_df["Log_RVOL"].ewm(span=ewm_span).mean()

        # raw=True => x is ndarray, x[-1] is always "last"
        zfn = lambda x: (x[-1] - x.mean()) / (x.std() + 1e-6)

        feat_df["scaled_natr"] = feat_df["NATR"].rolling(window=window).apply(zfn, raw=True)
        feat_df["scaled_rvol"] = feat_df["Log_RVOL"].rolling(window=window).apply(zfn, raw=True)

        features = feat_df[["scaled_natr", "scaled_rvol"]].dropna()
        if features.empty or len(features) < (window + 5):
            if fallback_method:
                return detect_market_regime(df, freq=freq, method=fallback_method, fallback_method=None)
            df["Market_Regime"] = "Normal Volatility"
            df["ATR_Percentile"] = 0.5
            return df

        # clamp extreme spikes (A-share theme bursts / limit behavior)
        features = features.clip(-z_clip, z_clip)
        X = features.values

        # Rolling refit HMM + robust fallback
        final_regimes = [1] * window  # 1 = Normal baseline

        model = None
        rank_map = None
        last_good_model = None
        last_good_rank_map = None

        for i in range(window, len(X)):
            need_refit = (model is None) or (i % refit_every == 0)

            if need_refit:
                X_train = X[i - window : i]
                try:
                    mm = GaussianHMM(
                        n_components=3,
                        covariance_type="diag",
                        n_iter=200,
                        random_state=42
                    )
                    mm.fit(X_train)

                    state_energy = [float(mm.means_[j][0] + mm.means_[j][1]) for j in range(3)]
                    ordered = np.argsort(state_energy)
                    rm = {int(ordered[0]): 0, int(ordered[1]): 1, int(ordered[2]): 2}

                    model = mm
                    rank_map = rm
                    last_good_model = mm
                    last_good_rank_map = rm

                except Exception:
                    # fallback to last good model OR fallback method
                    if last_good_model is not None:
                        model = last_good_model
                        rank_map = last_good_rank_map
                    elif fallback_method:
                        return detect_market_regime(df, freq=freq, method=fallback_method, fallback_method=None)
                    else:
                        final_regimes.append(1)
                        continue

            # decode sequence (less jumpy than single-point predict)
            seq_start = max(0, i - window)
            decoded = model.predict(X[seq_start : i + 1])
            current_state = int(decoded[-1])
            final_regimes.append(int(rank_map.get(current_state, 1)))

        # Smart persistence (keep your original intent)
        smoothed_final = []
        curr_fixed = final_regimes[0]
        strength = 5.0
        natr_series = features["scaled_natr"].values

        for i in range(len(final_regimes)):
            s = final_regimes[i]
            is_wide_candle = natr_series[i] > 0.5
            target_state = 1 if (s == 0 and is_wide_candle) else s

            if target_state == 2 and target_state != curr_fixed:
                strength = 0
                curr_fixed = 2
            elif target_state != curr_fixed:
                strength -= 1.0
            else:
                strength = min(10.0, strength + 1.0)

            if strength <= 0:
                curr_fixed = target_state
                strength = 4.0

            smoothed_final.append(curr_fixed)

        smoothed_series = pd.Series(smoothed_final, index=features.index)
        label_map = {0: "Low Volatility", 1: "Normal Volatility", 2: "High Volatility"}

        df["Market_Regime"] = smoothed_series.map(label_map).reindex(df.index, method="ffill")
        df["ATR_Percentile"] = smoothed_series.map({0: 0.2, 1: 0.5, 2: 0.8}).reindex(df.index, method="ffill")
        return df

    # =========================
    # JUMP MODE
    # =========================
    if m == "JUMP":
        df = df.copy()
        vol_win = 10 if is_weekly else 20
        df["vol_rolling"] = df["Close"].pct_change().rolling(vol_win).std()

        thresh_hi = df["vol_rolling"].rolling(252, min_periods=60).quantile(0.8)
        thresh_lo = df["vol_rolling"].rolling(252, min_periods=60).quantile(0.2)

        regimes = []
        curr = "Normal Volatility"
        for v in df["vol_rolling"]:
            if pd.isna(v):
                regimes.append(curr)
                continue
            if v > thresh_hi:
                curr = "High Volatility"
            elif v < thresh_lo:
                curr = "Low Volatility"
            else:
                curr = "Normal Volatility"
            regimes.append(curr)

        df["Market_Regime"] = regimes
        df["ATR_Percentile"] = pd.Series(regimes).map(
            {"Low Volatility": 0.2, "Normal Volatility": 0.5, "High Volatility": 0.8}
        ).values
        return df

    # Fallback
    df = df.copy()
    df["Market_Regime"] = "Normal Volatility"
    df["ATR_Percentile"] = 0.5
    return df



def simulate_next_day_indicators(df, price_change_pct, volume):
    """
    Simulate what technical indicators would be tomorrow
    Uses Volume_Scaled_OBV to match chart display

    Args:
        df: DataFrame with technical indicators (correct column names with underscores)
        price_change_pct: Tomorrow's price change percentage
        volume: Tomorrow's volume

    Returns:
        Dictionary with simulated indicator values
    """
    if len(df) < 30:
        return None

    latest = df.iloc[-1]

    # Calculate tomorrow's price
    close_today = latest['Close']
    close_tomorrow = close_today * (1 + price_change_pct / 100)

    # Old MACD calculator using static MACD parameters
    # # Get current EMA values for MACD calculation
    # ema12_today = df['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
    # ema26_today = df['Close'].ewm(span=26, adjust=False).mean().iloc[-1]

    # # Calculate tomorrow's EMAs
    # alpha12 = 2 / (12 + 1)
    # alpha26 = 2 / (26 + 1)

    # ema12_tomorrow = alpha12 * close_tomorrow + (1 - alpha12) * ema12_today
    # ema26_tomorrow = alpha26 * close_tomorrow + (1 - alpha26) * ema26_today

    # # MACD
    # macd_tomorrow = ema12_tomorrow - ema26_tomorrow
    # macd_signal_today = latest['MACD_Signal']
    # alpha_signal = 2 / (9 + 1)
    # macd_signal_tomorrow = alpha_signal * macd_tomorrow + (1 - alpha_signal) * macd_signal_today
    # macd_hist_tomorrow = macd_tomorrow - macd_signal_tomorrow

    # ==================== DYNAMIC SIMULATOR MACD ====================
    # Grab today's active dynamic parameters
    p_fast = int(latest.get('MACD_Fast_Param', 12))
    p_slow = int(latest.get('MACD_Slow_Param', 26))
    p_sign = int(latest.get('MACD_Sign_Param', 9))

    # Get current EMA values USING DYNAMIC PARAMS
    ema_fast_today = df['Close'].ewm(span=p_fast, adjust=False).mean().iloc[-1]
    ema_slow_today = df['Close'].ewm(span=p_slow, adjust=False).mean().iloc[-1]

    # Calculate tomorrow's EMAs
    alpha_fast = 2 / (p_fast + 1)
    alpha_slow = 2 / (p_slow + 1)

    ema_fast_tomorrow = alpha_fast * close_tomorrow + (1 - alpha_fast) * ema_fast_today
    ema_slow_tomorrow = alpha_slow * close_tomorrow + (1 - alpha_slow) * ema_slow_today

    # MACD Tomorrow
    macd_tomorrow = ema_fast_tomorrow - ema_slow_tomorrow
    macd_signal_today = latest['MACD_Signal']
    alpha_signal = 2 / (p_sign + 1)
    macd_signal_tomorrow = alpha_signal * macd_tomorrow + (1 - alpha_signal) * macd_signal_today
    macd_hist_tomorrow = macd_tomorrow - macd_signal_tomorrow
    # ================================================================

    # RSI
    price_change = close_tomorrow - close_today
    returns = df['Close'].diff()
    gains = returns.where(returns > 0, 0)
    losses = -returns.where(returns < 0, 0)

    avg_gain_today = gains.rolling(14).mean().iloc[-1]
    avg_loss_today = losses.rolling(14).mean().iloc[-1]

    if price_change > 0:
        avg_gain_tomorrow = (avg_gain_today * 13 + price_change) / 14
        avg_loss_tomorrow = (avg_loss_today * 13) / 14
    else:
        avg_gain_tomorrow = (avg_gain_today * 13) / 14
        avg_loss_tomorrow = (avg_loss_today * 13 + abs(price_change)) / 14

    if avg_loss_tomorrow == 0:
        rsi_tomorrow = 100
    else:
        rs = avg_gain_tomorrow / avg_loss_tomorrow
        rsi_tomorrow = 100 - (100 / (1 + rs))

    # OBV - use Volume_Scaled_OBV to match chart
    obv_today = latest['OBV']
    if close_tomorrow > close_today:
        obv_tomorrow = obv_today + volume
    elif close_tomorrow < close_today:
        obv_tomorrow = obv_today - volume
    else:
        obv_tomorrow = obv_today

    # 1. FIX: Calculate tomorrow's true 20-day volume average
    if len(df) >= 19:
        vol_20d_avg_tomorrow = (df['Volume'].iloc[-19:].sum() + volume) / 20
    else:
        vol_20d_avg_tomorrow = (df['Volume'].sum() + volume) / (len(df) + 1)

    vol_20d_avg_today = df['Volume'].rolling(20).mean().iloc[-1]

    # Convert to Volume_Scaled_OBV
    obv_scaled_today = latest.get('Volume_Scaled_OBV', obv_today / vol_20d_avg_today)
    obv_scaled_tomorrow = obv_tomorrow / vol_20d_avg_tomorrow

    # Convert to Volume_Scaled_OBV (matches chart display)
    vol_20d_avg = df['Volume'].rolling(20).mean().iloc[-1]
    obv_scaled_today = latest.get('Volume_Scaled_OBV', obv_today / vol_20d_avg)
    obv_scaled_tomorrow = obv_tomorrow / vol_20d_avg
   # Fixed index to -3 (T-2 days ago relative to tomorrow)
    obv_scaled_3d_ago = df['Volume_Scaled_OBV'].iloc[-3] if len(df) >= 3 and 'Volume_Scaled_OBV' in df.columns else obv_scaled_today
    obv_3d_ago_raw = df['OBV'].iloc[-3] if len(df) >= 3 else obv_today

    # ADX - simplified approximation
    adx_today = latest['ADX']
    price_move_magnitude = abs(price_change_pct)

    if price_move_magnitude > 2.0:
        adx_tomorrow = min(100, adx_today + 1.0)
    elif price_move_magnitude > 1.0:
        adx_tomorrow = adx_today + 0.3
    else:
        adx_tomorrow = max(0, adx_today - 0.3)

    # Determine ADX pattern (only 4 patterns: Bottoming, Reversing Up, Peaking, Reversing Down)
    adx_pattern = None
    adx_slope_today = latest.get('ADX_Slope', 0)

    if adx_tomorrow < 22:
        adx_pattern = "Bottoming"
    elif adx_today < 25 and adx_tomorrow >= adx_today and adx_slope_today <= 0:
        adx_pattern = "Reversing Up"
    elif adx_tomorrow > 30 and adx_tomorrow < adx_today and adx_slope_today > 0:
        adx_pattern = "Peaking"
    elif adx_today > 25 and adx_tomorrow < adx_today and adx_slope_today >= 0:
        adx_pattern = "Reversing Down"

    # ==========================================
    # THRESHOLDS & SIGNALS
    # ==========================================
    # Calculate tomorrow's TRUE 10-day low (including tomorrow's MACD)
    macd_9d_low = df['MACD'].rolling(9).min().iloc[-1]
    macd_10d_low_tomorrow = min(macd_9d_low, macd_tomorrow)
    macd_10d_high = df['MACD'].rolling(10).max().iloc[-1]

    # Get thresholds
    macd_10d_low = df['MACD'].rolling(10).min().iloc[-1]
    macd_10d_high = df['MACD'].rolling(10).max().iloc[-1]
    rsi_p10 = df['RSI_14'].rolling(60).quantile(0.10).iloc[-1] if 'RSI_14' in df.columns else 30
    rsi_p90 = df['RSI_14'].rolling(60).quantile(0.90).iloc[-1] if 'RSI_14' in df.columns else 70

    # Check signals
    signals = {}

    # MACD signals (directional)
    macd_stopped_falling = macd_tomorrow >= latest['MACD']
    macd_in_bottom_zone = (macd_tomorrow >= macd_10d_low_tomorrow) and (macd_tomorrow <= macd_10d_low_tomorrow * 0.95)
    macd_gap_narrowing = (macd_tomorrow - macd_signal_tomorrow) > (latest['MACD'] - latest['MACD_Signal'])
    # Use raw OBV to perfectly match the `df['OBV'] > df['OBV'].shift(3)` logic in the main engine
    obv_rising = obv_tomorrow > obv_3d_ago_raw

    conditions_met = sum([macd_stopped_falling, macd_in_bottom_zone, macd_gap_narrowing, obv_rising])
    signals['MACD_Bottoming'] = macd_tomorrow < 0 and macd_in_bottom_zone and conditions_met >= 2
    signals['MACD_Bullish_Cross'] = (latest['MACD'] < latest['MACD_Signal']) and (macd_tomorrow > macd_signal_tomorrow)
    signals['MACD_Bearish_Cross'] = (latest['MACD'] > latest['MACD_Signal']) and (macd_tomorrow < macd_signal_tomorrow)

    # RSI signals (directional)
    signals['RSI_Bottoming'] = rsi_tomorrow < 30 and rsi_tomorrow <= rsi_p10
    signals['RSI_Peaking'] = rsi_tomorrow > 70 and rsi_tomorrow >= rsi_p90

    # ADX pattern (informational, not directional)
    signals['ADX_Pattern'] = adx_pattern

    return {
        'input_price_change_pct': price_change_pct,
        'input_volume': volume,
        'close_today': close_today,
        'close_tomorrow': close_tomorrow,

        'macd_today': latest['MACD'],
        'macd_tomorrow': macd_tomorrow,
        'macd_signal_today': macd_signal_today,
        'macd_signal_tomorrow': macd_signal_tomorrow,
        'macd_hist_today': latest['MACD_Hist'],
        'macd_hist_tomorrow': macd_hist_tomorrow,
        'macd_gap_today': latest['MACD'] - latest['MACD_Signal'],
        'macd_gap_tomorrow': macd_tomorrow - macd_signal_tomorrow,
        'macd_10d_low_tomorrow': macd_10d_low_tomorrow,
        'macd_10d_high': macd_10d_high,
        'macd_10d_low': macd_10d_low,
        'rsi_today': latest['RSI_14'],
        'rsi_tomorrow': rsi_tomorrow,
        'rsi_p10': rsi_p10,
        'rsi_p90': rsi_p90,

        'adx_today': adx_today,
        'adx_tomorrow': adx_tomorrow,
        'adx_pattern': adx_pattern,

        'obv_scaled_today': obv_scaled_today,
        'obv_scaled_tomorrow': obv_scaled_tomorrow,
        'obv_scaled_3d_ago': obv_scaled_3d_ago,
        'obv_rising': obv_rising,

        'obv_raw_today': obv_today,
        'obv_raw_tomorrow': obv_tomorrow,
        'obv_raw_3d_ago': obv_3d_ago_raw,

        'volume_today': latest['Volume'],
        'volume_tomorrow': volume,
        'volume_10d_avg': df['Volume'].rolling(10).mean().iloc[-1],
        'volume_20d_avg': vol_20d_avg,

        'signals': signals,
        'conditions_met': conditions_met
    }


# def detect_market_regime(df):
#     """
#     Advanced Regime Detection with Noise Filtering.
#     """
#     if df.empty or len(df) < 100:
#         df['Market_Regime'] = 'Normal Volatility'
#         df['ATR_Percentile'] = 0.5
#         return df

#     # ==========================================================
#     # OPTION 3: DENOISED HIDDEN MARKOV MODEL (HMM)
#     # ==========================================================
#     if REGIME_METHOD == 'HMM':
#         # 1. PURE ENERGY FEATURES
#         df['prev_close'] = df['Close'].shift(1)
#         df['h_l'] = df['High'] - df['Low']
#         df['h_pc'] = (df['High'] - df['prev_close']).abs()
#         df['l_pc'] = (df['Low'] - df['prev_close']).abs()
#         df['TR'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
#         df['NATR'] = (df['TR'] / df['Close']) * 100
        
#         df['Vol_MA'] = df['Volume'].rolling(20).mean()
#         df['RVOL'] = df['Volume'] / (df['Vol_MA'] + 1)
#         df['Log_RVOL'] = np.log(df['RVOL'] + 0.1) 

#         # 2. FEATURE ENGINEERING
#         window = 90
#         feat_df = df[['NATR', 'Log_RVOL']].copy()
#         feat_df['NATR'] = feat_df['NATR'].ewm(span=3).mean()
#         feat_df['Log_RVOL'] = feat_df['Log_RVOL'].ewm(span=3).mean()
        
#         feat_df['scaled_natr'] = feat_df['NATR'].rolling(window=window).apply(
#             lambda x: (x.iloc[-1]  - x.mean()) / (x.std() + 1e-6)
#         )
#         feat_df['scaled_rvol'] = feat_df['Log_RVOL'].rolling(window=window).apply(
#             lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-6)
#         )
        
#         features = feat_df[['scaled_natr', 'scaled_rvol']].dropna()
#         X = features.values

#         # 3. ONLINE ROLLING HMM FIT
#         final_regimes = []
#         for _ in range(window): final_regimes.append(1)
            
#         for i in range(window, len(X)):
#             if i % 5 == 0:
#                 X_train = X[i-window:i]
#                 model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
#                 try: model.fit(X_train)
#                 except: pass
                
#                 # Sort by Energy (0=Low, 2=High)
#                 state_energy = [model.means_[j][0] + model.means_[j][1] for j in range(3)]
#                 ordered = np.argsort(state_energy)
#                 rank_map = {ordered[0]: 0, ordered[1]: 1, ordered[2]: 2}
            
#             states = model.predict(X[i-window:i+1])
#             current_state = states[-1]
#             final_regimes.append(rank_map[current_state])

#         # 4. SMART PERSISTENCE (The Fix)
#         smoothed_final = []
#         if final_regimes:
#             curr_fixed = final_regimes[0]
#             strength = 5.0
#             natr_series = features['scaled_natr'].values
            
#             for i in range(len(final_regimes)):
#                 s = final_regimes[i]
#                 # Raised threshold: Only consider candles "Wide" if Z-Score > 0.5
#                 is_wide_candle = natr_series[i] > 0.5
                
#                 # --- THE COMPROMISE LOGIC ---
#                 # If HMM wants 'Low Vol' (0) but candles are 'Wide', 
#                 # we force the target to 'Normal Vol' (1).
#                 # This prevents Green, but allows Red to decay to Blue.
#                 if s == 0 and is_wide_candle:
#                     target_state = 1 
#                 else:
#                     target_state = s
                
#                 # --- STANDARD DECAY LOGIC ---
#                 if target_state == 2 and target_state != curr_fixed:
#                     strength = 0 # Instant Red
#                     curr_fixed = 2
#                 elif target_state != curr_fixed:
#                     decay_rate = 1.0 # Standard decay speed
#                     strength -= decay_rate
#                 else:
#                     strength = min(10.0, strength + 1.0)
                
#                 if strength <= 0:
#                     curr_fixed = target_state
#                     strength = 4.0
                
#                 smoothed_final.append(curr_fixed)

#         smoothed_series = pd.Series(smoothed_final, index=features.index)
#         label_map = {0: 'Low Volatility', 1: 'Normal Volatility', 2: 'High Volatility'}
        
#         df['Market_Regime'] = smoothed_series.map(label_map).reindex(df.index, method='ffill')
#         df['ATR_Percentile'] = smoothed_series.map({0: 0.2, 1: 0.5, 2: 0.8}).reindex(df.index, method='ffill')
#     # ==========================================================
#     # OPTION 4: JUMP MODEL (Persistence-Penalized)
#     # ==========================================================
#     elif REGIME_METHOD == 'JUMP':
#         # Jump logic: Only switches if volatility crosses 80th/20th percentile
#         # and stays there (Sticky regimes)
#         df['vol_rolling'] = df['Close'].pct_change().rolling(20).std()
#         thresh_hi = df['vol_rolling'].quantile(0.8)
#         thresh_lo = df['vol_rolling'].quantile(0.2)
        
#         regimes = []
#         curr = 'Normal Volatility'
#         for v in df['vol_rolling']:
#             if pd.isna(v): regimes.append(curr); continue
#             if v > thresh_hi: curr = 'High Volatility'
#             elif v < thresh_lo: curr = 'Low Volatility'
#             # If in the middle, we don't 'jump' back to Normal immediately (Stickiness)
#             regimes.append(curr)
            
#         df['Market_Regime'] = regimes
#         df['ATR_Percentile'] = df['vol_rolling'].rank(pct=True)

#     # ==========================================================
#     # LEGACY: ORIGINAL ATR METHOD
#     # ==========================================================
#     else:
#         # ATR logic exactly as you have it now
#         df['ATR_20'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=20).average_true_range()
#         df['ATR_Pct'] = (df['ATR_20'] / df['Close']) * 100
#         df['ATR_Percentile'] = df['ATR_Pct'].rolling(window=252, min_periods=60).rank(pct=True)
        
#         df['Market_Regime'] = 'Normal Volatility'
#         df.loc[df['ATR_Percentile'] < 0.25, 'Market_Regime'] = 'Low Volatility'
#         df.loc[df['ATR_Percentile'] > 0.75, 'Market_Regime'] = 'High Volatility'
#         df.loc[df['ATR_Percentile'] > 0.90, 'Market_Regime'] = 'Extreme Volatility'


#     return df