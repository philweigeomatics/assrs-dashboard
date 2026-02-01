import pandas as pd
import numpy as np
import ta
import data_manager
from scipy import stats
from scipy.signal import savgol_filter

# Fetch data for stock
ticker = "600562"
stock_df = data_manager.get_single_stock_data(ticker, use_data_start_date=True)

if stock_df is not None and not stock_df.empty:
    # Calculate indicators (EXACTLY as your code)
    df_analysis = stock_df.copy()
    
    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        df_analysis.index = pd.to_datetime(df_analysis.index)
    df_analysis = df_analysis.sort_index()
    
    # Moving Averages
    df_analysis['MA20'] = ta.trend.sma_indicator(df_analysis['Close'], window=20)
    df_analysis['MA50'] = ta.trend.sma_indicator(df_analysis['Close'], window=50)
    
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
    
    # ==================== ADX SMOOTHING (EXACT MATCH) ====================
    try:
        df_analysis['ADX_LOWESS'] = savgol_filter(
            df_analysis['ADX'].fillna(method='ffill'),
            window_length=11,
            polyorder=3
        )
    except:
        df_analysis['ADX_LOWESS'] = df_analysis['ADX'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands for ADX
    df_analysis['ADX_BB_Middle'] = df_analysis['ADX'].rolling(window=20).mean()
    df_analysis['ADX_BB_Std'] = df_analysis['ADX'].rolling(window=20).std()
    df_analysis['ADX_BB_Upper'] = df_analysis['ADX_BB_Middle'] + (2 * df_analysis['ADX_BB_Std'])
    df_analysis['ADX_BB_Lower'] = df_analysis['ADX_BB_Middle'] - (2 * df_analysis['ADX_BB_Std'])
    
    # ==================== ADX TREND & ACCELERATION (5-DAY LINEAR REGRESSION) ====================
    def calculate_adx_trend(series, window=5):
        """Calculate ADX slope using linear regression"""
        def get_slope(y):
            if len(y) < 3 or y.isna().any():
                return 0
            x = np.arange(len(y))
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        return series.rolling(window=window, min_periods=3).apply(get_slope, raw=False)
    
    df_analysis['ADX_Slope'] = calculate_adx_trend(df_analysis['ADX'], window=5)
    df_analysis['ADX_Acceleration'] = df_analysis['ADX_Slope'] - df_analysis['ADX_Slope'].shift(1)
    
    # ==================== MACD GAP & MOMENTUM ====================
    df_analysis['MACD_Gap'] = df_analysis['MACD'] - df_analysis['MACD_Signal']
    df_analysis['MACD_Momentum'] = df_analysis['MACD'] - df_analysis['MACD'].shift(1)
    
    # Check specific dates
    target_dates = ['2025-09-05', '2025-09-08', '2025-09-09']
    
    for date_str in target_dates:
        try:
            date = pd.to_datetime(date_str)
            if date in df_analysis.index:
                row = df_analysis.loc[date]
                prev_date = df_analysis.index[df_analysis.index < date][-1]
                prev_row = df_analysis.loc[prev_date]
                
                prev2_date = df_analysis.index[df_analysis.index < prev_date][-1] if len(df_analysis.index[df_analysis.index < prev_date]) > 0 else None
                prev2_row = df_analysis.loc[prev2_date] if prev2_date else None
                
                print(f"\n{'='*70}")
                print(f"📅 Date: {date_str}")
                print(f"{'='*70}")
                
                # ==================== MACD DATA ====================
                print(f"\n📊 MACD VALUES:")
                print(f"  Current: MACD={row['MACD']:.4f}, Signal={row['MACD_Signal']:.4f}, Gap={row['MACD_Gap']:.4f}")
                print(f"  Previous: MACD={prev_row['MACD']:.4f}, Signal={prev_row['MACD_Signal']:.4f}, Gap={prev_row['MACD_Gap']:.4f}")
                if prev2_row is not None:
                    print(f"  2 Days Ago: Gap={prev2_row['MACD_Gap']:.4f}")
                print(f"  Momentum: Current={row['MACD_Momentum']:.4f}, Previous={prev_row['MACD_Momentum']:.4f}")
                
                # ==================== MACD SCENARIO CHECKS (EXACT FROM YOUR CODE) ====================
                print(f"\n🔍 MACD SCENARIO CHECKS:")
                
                # Scenario 1: Classic crossover
                classic_crossover = (prev_row['MACD_Gap'] <= 0) and (row['MACD_Gap'] > 0)
                print(f"  1️⃣  Classic Crossover: {classic_crossover}")
                if not classic_crossover:
                    print(f"     ↳ Gap was {prev_row['MACD_Gap']:.4f} yesterday, now {row['MACD_Gap']:.4f}")
                
                # Scenario 2: Approaching from below
                approaching = (
                    (row['MACD_Gap'] < 0)
                    and (row['MACD_Gap'] > -0.5)
                    and (row['MACD_Gap'] > prev_row['MACD_Gap'])
                    and (prev2_row is not None and prev_row['MACD_Gap'] > prev2_row['MACD_Gap'])
                    and (row['MACD_Momentum'] > 0)
                )
                print(f"  2️⃣  Approaching (Gap Narrowing): {approaching}")
                if not approaching and row['MACD_Gap'] < 0:
                    checks = []
                    if not (row['MACD_Gap'] > -0.5):
                        checks.append(f"Gap too wide: {row['MACD_Gap']:.4f}")
                    if not (row['MACD_Gap'] > prev_row['MACD_Gap']):
                        checks.append("Not narrowing today")
                    if prev2_row is not None and not (prev_row['MACD_Gap'] > prev2_row['MACD_Gap']):  # FIXED
                        checks.append("Not narrowing yesterday")
                    if not (row['MACD_Momentum'] > 0):
                        checks.append(f"Momentum not positive: {row['MACD_Momentum']:.4f}")
                    if checks:  # FIXED: Added this check
                        print(f"     ↳ Failed: {'; '.join(checks)}")

                
                # Scenario 3: Higher low bounce
                higher_low = (
                    (row['MACD_Gap'] > 0)
                    and (row['MACD_Gap'] < prev_row['MACD_Gap'])
                    and (row['MACD_Momentum'] > 0)
                    and (row['MACD_Momentum'] > prev_row['MACD_Momentum'])
                    and (row['MACD_Gap'] > 0.1)
                )
                print(f"  3️⃣  Higher Low Bounce: {higher_low}")
                if not higher_low and row['MACD_Gap'] > 0:
                    checks = []
                    if not (row['MACD_Gap'] < prev_row['MACD_Gap']):
                        checks.append("Gap not shrinking")
                    if not (row['MACD_Momentum'] > 0):
                        checks.append("Momentum not positive")
                    if not (row['MACD_Momentum'] > prev_row['MACD_Momentum']):
                        checks.append("Momentum not accelerating")
                    if not (row['MACD_Gap'] > 0.1):
                        checks.append(f"Gap too small: {row['MACD_Gap']:.4f}")
                    if checks:
                        print(f"     ↳ Failed: {'; '.join(checks)}")
                
                # Scenario 4: Momentum building
                momentum_building = (
                    (row['MACD_Gap'] > 0)
                    and (prev_row['MACD_Gap'] > 0)
                    and (prev2_row is not None and prev2_row['MACD_Gap'] > 0)  # This one is OK
                    and (row['MACD_Momentum'] > 0)
                    and (prev_row['MACD_Momentum'] <= 0.05)
                    and (row['MACD_Hist'] > prev_row['MACD_Hist'])
                )
                
                macd_trigger = classic_crossover or approaching or higher_low or momentum_building
                print(f"\n  ✅ MACD Trigger Active: {macd_trigger}")
                
                # ==================== ADX DATA ====================
                print(f"\n📈 ADX VALUES:")
                print(f"  Raw ADX: {row['ADX']:.2f}")
                print(f"  ADX Slope (5-day): {row['ADX_Slope']:.4f}")
                print(f"  ADX Acceleration: {row['ADX_Acceleration']:.4f}")
                print(f"  ADX BB: Lower={row['ADX_BB_Lower']:.2f}, Middle={row['ADX_BB_Middle']:.2f}, Upper={row['ADX_BB_Upper']:.2f}")
                
                # ==================== ADX SCENARIO CHECKS (EXACT FROM YOUR CODE) ====================
                print(f"\n🔍 ADX SCENARIO CHECKS:")
                
                # Pattern 1: ADX Bottoming
                adx_bottoming = (
                    (row['ADX'] < 25)
                    and (row['ADX_Slope'] < 0)
                    and (row['ADX_Acceleration'] > 0.1)
                    and (row['ADX_Slope'] > prev_row['ADX_Slope'])
                )
                print(f"  1️⃣  ADX Bottoming (Reversal Setup): {adx_bottoming}")
                if not adx_bottoming and row['ADX'] < 25:
                    checks = []
                    if not (row['ADX_Slope'] < 0):
                        checks.append(f"Slope not negative: {row['ADX_Slope']:.4f}")
                    if not (row['ADX_Acceleration'] > 0.1):
                        checks.append(f"Acceleration too low: {row['ADX_Acceleration']:.4f}")
                    if not (row['ADX_Slope'] > prev_row['ADX_Slope']):
                        checks.append("Slope not improving")
                    if checks:
                        print(f"     ↳ Failed: {'; '.join(checks)}")
                
                # Pattern 2: ADX Accelerating
                adx_accelerating = (
                    (row['ADX_Slope'] > 0.3)
                    and (row['ADX_Acceleration'] > 0.1)
                )
                print(f"  2️⃣  ADX Accelerating (Breakout Mode): {adx_accelerating}")
                if not adx_accelerating:
                    checks = []
                    if not (row['ADX_Slope'] > 0.3):
                        checks.append(f"Slope too low: {row['ADX_Slope']:.4f}")
                    if not (row['ADX_Acceleration'] > 0.1):
                        checks.append(f"Acceleration too low: {row['ADX_Acceleration']:.4f}")
                    print(f"     ↳ Failed: {'; '.join(checks)}")
                
                # Pattern 3: ADX Strong & Stable
                adx_strong_stable = (
                    (row['ADX'] >= 30)
                    and (row['ADX_Slope'] >= -0.3)
                    and (row['ADX_Slope'] <= 0.3)
                )
                print(f"  3️⃣  ADX Strong & Stable: {adx_strong_stable}")
                if not adx_strong_stable:
                    if row['ADX'] < 30:
                        print(f"     ↳ ADX too low: {row['ADX']:.2f}")
                    elif not (-0.3 <= row['ADX_Slope'] <= 0.3):
                        print(f"     ↳ Slope not stable: {row['ADX_Slope']:.4f}")
                
                adx_healthy = adx_bottoming or adx_accelerating or adx_strong_stable
                print(f"\n  ✅ ADX Healthy: {adx_healthy}")
                
                # ==================== OTHER CONDITIONS ====================
                print(f"\n📋 OTHER CONDITIONS:")
                ma_ok = row['MA20'] > row['MA50']
                rsi_ok = (row['RSI_14'] >= 50) and (row['RSI_14'] <= 70)
                vol_ok = row['Volume'] > df_analysis['Volume'].rolling(20).mean().loc[date]
                
                print(f"  MA20 > MA50: {ma_ok} (MA20={row['MA20']:.2f}, MA50={row['MA50']:.2f})")
                print(f"  RSI in range: {rsi_ok} (RSI={row['RSI_14']:.2f}, need 50-70)")
                print(f"  Volume > Avg: {vol_ok} (Vol={row['Volume']:.0f}, Avg={df_analysis['Volume'].rolling(20).mean().loc[date]:.0f})")
                
                # ==================== FINAL VERDICT ====================
                all_conditions = macd_trigger and adx_healthy and ma_ok and rsi_ok and vol_ok
                
                print(f"\n{'='*70}")
                if all_conditions:
                    print(f"✅ ✅ ✅ GOLDEN LAUNCH SHOULD TRIGGER! ✅ ✅ ✅")
                else:
                    print(f"❌ GOLDEN LAUNCH NOT TRIGGERED")
                    print(f"\n🚫 FAILED CONDITIONS:")
                    if not macd_trigger:
                        print(f"   ❌ MACD: No valid scenario detected")
                    if not adx_healthy:
                        print(f"   ❌ ADX: Not in healthy state (all patterns failed)")
                    if not ma_ok:
                        print(f"   ❌ MA: Bearish alignment (MA20 not above MA50)")
                    if not rsi_ok:
                        print(f"   ❌ RSI: Out of range (need 50-70, got {row['RSI_14']:.2f})")
                    if not vol_ok:
                        print(f"   ❌ Volume: Below average")
                print(f"{'='*70}")
                    
        except Exception as e:
            print(f"❌ Error for {date_str}: {e}")
            import traceback
            traceback.print_exc()
