import pandas as pd
import numpy as np
import ta
from datetime import datetime

# --- 1. NEW: TIGHTNESS PRESET DEFINITIONS ---
TIGHTNESS_PRESETS = {
    'INTENSE': {
        # --- ACCELERATION (Strictest) ---
        'check_breakout': True, 'breakout_period': 10,
        'check_adx': True,      'accel_adx_floor': 18,
        'check_rsi': True,      'accel_rsi_min': 45,
        'accel_rsi_max': 60, # <-- Key is present
        'check_vol': True,      'accel_vol_mult': 1.5,
        # --- CONTINUATION (Strict) ---
        'check_cont_adx': True, 'cont_adx_floor': 20,
        'check_cont_rsi': True, 'cont_rsi_min': 45,
        'cont_rsi_max': 70,
        # --- WEAKENING (Fastest Exit) ---
        'check_weak_rsi': True, 'weak_rsi_trigger': 70,
        'check_weak_macd': True,'weak_decline_days': 3
    },
    'MEDIUM': {
        # --- ACCELERATION (Relaxed 1: No ADX/Vol) ---
        'check_breakout': True, 'breakout_period': 7,
        'check_adx': False,     'accel_adx_floor': 18,
        'check_rsi': True,      'accel_rsi_min': 40,
        'accel_rsi_max': 65, # <--!!!--- FIX: Added missing key ---!!!---
        'check_vol': False,     'accel_vol_mult': 1.5,
        # --- CONTINUATION (Relaxed 1) ---
        'check_cont_adx': True, 'cont_adx_floor': 18,
        'check_cont_rsi': True, 'cont_rsi_min': 40,
        'cont_rsi_max': 75,
        # --- WEAKENING (Slower Exit) ---
        'check_weak_rsi': True, 'weak_rsi_trigger': 72,
        'check_weak_macd': True,'weak_decline_days': 4
    },
    'RELAXED': {
        # --- ACCELERATION (Relaxed 2: Easiest) ---
        'check_breakout': True, 'breakout_period': 4,
        'check_adx': False,     'accel_adx_floor': 15,
        'check_rsi': True,      'accel_rsi_min': 35,
        'accel_rsi_max': 70, # <--!!!--- FIX: Added missing key ---!!!---
        'check_vol': False,     'accel_vol_mult': 1.2,
        # --- CONTINUATION (Relaxed 2) ---
        'check_cont_adx': False,'cont_adx_floor': 18,
        'check_cont_rsi': True, 'cont_rsi_min': 35,
        'cont_rsi_max': 75,
        # --- WEAKENING (Slowest Exit) ---
        'check_weak_rsi': True, 'weak_rsi_trigger': 75,
        'check_weak_macd': True,'weak_decline_days': 4
    }
}


# --- 2. CORE STATE CALCULATION FUNCTION ---

def calculate_stock_state(all_stock_data, as_of_date,
                          # --- 0. Master Preset Control ---
                          tightness_preset='MEDIUM',
                          
                          # --- 1. Core Lookbacks (Tunable) ---
                          ema_fast_period=10,
                          ema_slow_period=20,
                          adx_period=14,
                          rsi_period=14,
                          macd_fast_period=12,
                          macd_slow_period=26,
                          macd_signal_period=9,
                          bb_period=20,
                          bb_std_dev=2,
                          atr_period=14,
                          vol_avg_period=20,
                          min_history_days=100,
                          
                          # --- 2. Specific Overrides (Optional) ---
                          **kwargs):
    """
    Calculates the 4-state timing model.
    Accepts a 'tightness_preset' string to load parameters.
    Also accepts **kwargs to override any single parameter.
    """
    
    daily_states = []
    
    # --- Load Parameters from Preset ---
    if tightness_preset not in TIGHTNESS_PRESETS:
        print(f"Warning: Tightness preset '{tightness_preset}' not found. Defaulting to MEDIUM.")
        tightness_preset = 'MEDIUM'
        
    params = TIGHTNESS_PRESETS[tightness_preset].copy()
    
    # --- Allow Overrides ---
    params.update(kwargs)

    # ---!!!--- DEFENSIVE CODING: Set defaults for ALL params ---!!!---
    # This prevents KeyErrors if main.py sends a bad dictionary
    p_breakout_period = int(params.get('breakout_period', 10))
    p_weak_decline_days = int(params.get('weak_decline_days', 3))

    # Dynamically determine the slowest lookback needed for TA
    slowest_lookback = max(ema_slow_period, adx_period, rsi_period, 
                           macd_slow_period, bb_period, vol_avg_period, 
                           p_breakout_period, min_history_days)
    
    for ticker, full_df in all_stock_data.items():
        # 1. Slice data to be point-in-time (prevents lookahead bias)
        df = full_df.loc[full_df.index <= as_of_date].copy()
        
        if len(df) < slowest_lookback + 5: 
            continue
            
        # --- 2. Calculate All Required Technical Indicators ---
        
        # Trend Backbone
        df['EMA_Fast'] = ta.trend.ema_indicator(df['Close'], window=ema_fast_period)
        df['EMA_Slow'] = ta.trend.ema_indicator(df['Close'], window=ema_slow_period)
        
        # ADX
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=adx_period)
        df['ADX'] = adx.adx()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
        
        # MACD
        macd = ta.trend.MACD(df['Close'], 
                              window_slow=macd_slow_period, 
                              window_fast=macd_fast_period, 
                              window_sign=macd_signal_period)
        df['MACD_Hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=bb_period, window_dev=bb_std_dev)
        df['BB_Upper'] = bb.bollinger_hband()
        
        # ATR 
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=atr_period)
        
        # Volume & Price High
        df['Volume_Avg'] = df['Volume'].rolling(window=vol_avg_period).mean()
        df['High_Breakout'] = df['High'].rolling(window=p_breakout_period).max().shift(1)
        
        df.ffill(inplace=True) 

        # --- 3. Get Last N Rows for State Logic ---
        if len(df) < p_weak_decline_days:
            continue
            
        last = df.iloc[-1]
        prev_1 = df.iloc[-2]
        
        rsi_slice = df['RSI'].iloc[-p_weak_decline_days:]
        macd_slice = df['MACD_Hist'].iloc[-p_weak_decline_days:]

        if last.isnull().any():
            continue
            
        # --- 4. Implement 4-State Logic ---
        
        trend_strength = (last['EMA_Fast'] - last['EMA_Slow']) / last['Close'] if last['Close'] != 0 else 0.0
        ema_slope = last['EMA_Fast'] - prev_1['EMA_Fast']
        
        # --- State 1: WEAKENING (Highest Priority) ---
        rsi_declining = (rsi_slice.diff().dropna() < 0).all()
        macd_shrinking = (macd_slice.diff().dropna() < 0).all()

        # Build the WEAKENING check dynamically based on switches
        # ---!!!--- DEFENSIVE .get() FIX ---!!!---
        check_rsi_weak = (last['RSI'] > params.get('weak_rsi_trigger', 70) and rsi_declining) if params.get('check_weak_rsi', True) else False
        check_macd_weak = (macd_shrinking) if params.get('check_weak_macd', True) else False
        
        is_weakening = check_rsi_weak or check_macd_weak
        
        if is_weakening:
            state = "WEAKENING"
        else:
            # --- Check Core Backbone ---
            backbone_holds = (trend_strength > 0) and (ema_slope > 0)
            
            if backbone_holds:
                # --- State 2: ACCELERATION (Second Priority) ---
                adx_rising = last['ADX'] > prev_1['ADX']
                
                # ---!!!--- DEFENSIVE .get() FIX ---!!!---
                check_breakout = (last['Close'] > last['High_Breakout']) if params.get('check_breakout', True) else True
                check_adx = (last['ADX'] > params.get('accel_adx_floor', 18) and adx_rising) if params.get('check_adx', True) else True
                check_rsi = (last['RSI'] >= params.get('accel_rsi_min', 45) and last['RSI'] <= params.get('accel_rsi_max', 60)) if params.get('check_rsi', True) else True
                check_vol = (last['Volume'] > (last['Volume_Avg'] * params.get('accel_vol_mult', 1.5))) if params.get('check_vol', True) else True
                # ---!!!--- END OF FIX ---!!!---

                is_accelerating = (
                    check_breakout and 
                    check_adx and 
                    check_rsi and
                    check_vol
                )
                
                if is_accelerating:
                    state = "ACCELERATION"
                else:
                    # --- State 3: CONTINUATION (Third Priority) ---
                    # ---!!!--- DEFENSIVE .get() FIX ---!!!---
                    check_cont_adx = (last['ADX'] > params.get('cont_adx_floor', 20)) if params.get('check_cont_adx', True) else True
                    check_cont_rsi = (last['RSI'] >= params.get('cont_rsi_min', 45) and last['RSI'] <= params.get('cont_rsi_max', 70)) if params.get('check_cont_rsi', True) else True
                    # ---!!!--- END OF FIX ---!!!---
                    
                    is_continuing = (
                        check_cont_adx and 
                        check_cont_rsi and 
                        (last['Close'] <= last['BB_Upper']) # Still check not to buy *above* BB
                    )
                    
                    if is_continuing:
                        state = "CONTINUATION"
                    else:
                        # --- State 4: CONSOLIDATION (Hold State) ---
                        state = "CONSOLIDATION"
            else:
                # --- State 5: STRUCTURAL_AVOID (Default for Broken Trend) ---
                state = "STRUCTURAL_AVOID"
                
        # --- 5. Append Results ---
        daily_states.append({
            'Date': as_of_date.strftime('%Y-%m-%d'),
            'Ticker': ticker,
            'State': state,
            'Open': f"{last['Open']:.2f}",
            'Close': f"{last['Close']:.2f}",
            'EMA_20_Slope': f"{ema_slope:.3f}", 
            'EMA_20': f"{last['EMA_Fast']:.2f}", 
            'EMA_60': f"{last['EMA_Slow']:.2f}", 
            'Trend_Strength': f"{trend_strength:.4f}",
            'ADX_14': f"{last['ADX']:.1f}",
            'RSI_14': f"{last['RSI']:.1f}",
            'MACD_Hist_Shrink_3D': macd_shrinking, 
            'RSI_Decline_3D': rsi_declining, 
            'MACD_Hist': f"{last['MACD_Hist']:.3f}",
            'High_10D_Breakout': f"{last['High_Breakout']:.2f}", 
            'Vol_Ratio_1.5x': last['Volume'] / last['Volume_Avg'] 
        })

    return pd.DataFrame(daily_states)

