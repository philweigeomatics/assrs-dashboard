import pandas as pd
import numpy as np
import ta
from datetime import datetime

# --- 1. SCORING CONSTANTS (FAST, RESPONSIVE MODEL) ---

# --- Lookback Periods ---
LOOKBACK_RS_VOLUME = 20     # 20-Day Lookback for RS & Volume Ranks
LOOKBACK_LONGTERM_SMA = 100   # 100-Day SMA for Health Filter & Long-Term Vol Avg
LOOKBACK_10D_RSI = 10         # 10-Day RSI for Reversal
LOOKBACK_10D_EMA = 10         # 10-Day EMA for Fast Trend
LOOKBACK_20D_EMA = 20         # 20-Day EMA for Base Trend
LOOKBACK_5D_ATR = 5           # 5-Day ATR for Short Vol
LOOKBACK_20D_ATR = 20         # 20-Day ATR for Base Vol

# --- Indicator Thresholds ---
RS_RANK_PERCENTILE = 70       # Top 30%
REVERSAL_RSI_LEVEL = 40.0     # RSI <= 40
VOLUME_RANK_PERCENTILE = 30 # Bottom 30% (for the new Quietness Ratio)
EMA_TARGET_GAP = 0.03         # 3% gap (EMA10 vs EMA20) for full 1.0 pt
ATR_TARGET_COMPRESSION = 0.8  # 5D ATR is 80% (or less) of 20D ATR for full 1.0 pt

# --- 2. CORE SCORING FUNCTION ---

def calculate_scorecard(all_sector_ppi_data, as_of_date):
    """
    Calculates the ASSRS scorecard for all PPIs at a specific point in time.
    Uses "point-in-time" data slicing to prevent lookahead bias.
    V1.1: Uses 'Volume_Metric' (pre-calculated Norm_Vol_Metric)
    """
    
    # --- Step 1: Calculate Base Indicators for ALL Sectors (Point-in-Time) ---
    sector_list = list(all_sector_ppi_data.keys())
    
    sector_ranked_returns = {}
    sector_ranked_quietness = {} # This will use the pre-calced metric
    
    calculated_data = {}

    for sector, full_df in all_sector_ppi_data.items():
        # Slice the DataFrame to only include data up to the 'as_of_date'
        df = full_df.loc[full_df.index <= as_of_date].copy()

        if len(df) < LOOKBACK_LONGTERM_SMA + 5: 
            continue
            
        # --- Calculate all Technical Indicators ---
        
        # 1. Return for RS Rank
        df['Return_20D'] = df['Close'].pct_change(periods=LOOKBACK_RS_VOLUME)
        
        # 2. Volume Metric (Normalized Z-Score)
        # The 'Volume_Metric' column is pre-calculated in data_manager.py
        # We just need to get its 20-day average for the "Quietness" signal
        df['Quietness_Metric'] = df['Volume_Metric'].rolling(window=LOOKBACK_RS_VOLUME).mean()
        
        # 4. Other TAs
        df['EMA_10'] = df['Close'].ewm(span=LOOKBACK_10D_EMA, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=LOOKBACK_20D_EMA, adjust=False).mean()
        df['SMA_100'] = df['Close'].rolling(window=LOOKBACK_LONGTERM_SMA).mean()
        
        df['ATR_5'] = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=LOOKBACK_5D_ATR
        ).average_true_range()
        
        df['ATR_20'] = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=LOOKBACK_20D_ATR
        ).average_true_range()
        
        df['RSI_10'] = ta.momentum.RSIIndicator(df['Close'], window=LOOKBACK_10D_RSI).rsi()

        # --- Store the *last* valid row for ranking ---
        last_valid_row = df.iloc[-1]
        if last_valid_row.isnull().any():
            continue
            
        sector_ranked_returns[sector] = last_valid_row['Return_20D']
        sector_ranked_quietness[sector] = last_valid_row['Quietness_Metric'] 
        
        calculated_data[sector] = df 


    # --- Step 2: Rank Sectors (Cross-Sectional Analysis) ---
    all_ranked_returns = [ret for ret in sector_ranked_returns.values() if pd.notna(ret)]
    all_ranked_quietness = [q for q in sector_ranked_quietness.values() if pd.notna(q)] 
    
    if not all_ranked_returns:
        return pd.DataFrame()
        
    return_rank_threshold = np.percentile(all_ranked_returns, RS_RANK_PERCENTILE)
    
    if not all_ranked_quietness:
        quietness_rank_threshold = -999 # Default to a value no sector will match
    else:
        # We rank the Z-scores. We want the *lowest* Z-scores (bottom 30%)
        quietness_rank_threshold = np.percentile(all_ranked_quietness, VOLUME_RANK_PERCENTILE) 


    # --- Step 3: Apply Final Scoring Logic (Iterate again) ---
    scorecard = []
    
    for sector in sector_list:
        if sector not in calculated_data:
            continue
            
        df = calculated_data[sector]
        
        if as_of_date not in df.index:
            continue 
            
        last = df.loc[as_of_date]
        
        prev_date_loc = df.index.get_loc(as_of_date) - 1
        if prev_date_loc < 0:
            continue 
        prev_last = df.iloc[prev_date_loc]
        
        if last.isnull().any():
            continue
            
        score = 0.0
        
        # --- INDICATOR 1: Relative Strength Rank (Max 2.0 pts) ---
        rs_score = 0.0
        if last['Return_20D'] >= return_rank_threshold:
            rs_score = 2.0
        
        # --- INDICATOR 2: Short-Term Reversal (Max 1.0 pt) ---
        reversal_score = 0.0
        daily_return = (last['Close'] - prev_last['Close']) / prev_last['Close']
        if last['RSI_10'] <= REVERSAL_RSI_LEVEL and daily_return > 0:
            reversal_score = 1.0

        # --- INDICATOR 3: Trend Strength (EMA Gap) (Max 1.0 pt) ---
        ema_gap = last['EMA_10'] - last['EMA_20']
        relative_gap_percent = last['EMA_20'] if last['EMA_20'] != 0 else 1 # Avoid divide by zero
        trend_score = 0.0
        if relative_gap_percent > 0:
             trend_score = min(1.0, (ema_gap / relative_gap_percent) / EMA_TARGET_GAP)
        
        # --- INDICATOR 4: Volatility Compression (ATR Ratio) (Max 1.0 pt) ---
        atr_ratio = last['ATR_5'] / last['ATR_20']
        
        atr_score = 0.0
        if atr_ratio <= 1.0 and atr_ratio > 0:
             atr_score = min(1.0, ATR_TARGET_COMPRESSION / atr_ratio)
            
        # --- INDICATOR 5: Contrarian Volume (Quietness Ratio) (Max 2.0 pts) ---
        contra_vol_score = 0.0
        if last['Quietness_Metric'] <= quietness_rank_threshold:
            contra_vol_score = 2.0
        
        # --- INDICATOR 6: Long-Term Health Filter (Penalty -3.0 pts) ---
        lt_filter = 0.0
        if last['Close'] < last['SMA_100']:
            lt_filter = -3.0

        total_score = rs_score + reversal_score + trend_score + atr_score + contra_vol_score + lt_filter

        # --- Traffic Light Decision ---
        traffic_light = 'RED (EXIT/AVOID)'
        if total_score >= 5.0:
            traffic_light = 'GREEN (BUY)'
        elif total_score >= 3.0:
            traffic_light = 'YELLOW (WATCH/HOLD)'
            
        # ---!!!--- NEW: ADD ALL DATA TO THE OUTPUT ---!!!---
        scorecard.append({
            'Sector': sector,
            'RS_Score': f"{rs_score:.1f}",
            'Reversal_Score': f"{reversal_score:.1f}",
            'Trend_Score': f"{trend_score:.1f}",
            'ATR_Score': f"{atr_score:.1f}",
            'Contra_Vol_Score': f"{contra_vol_score:.1f}", 
            'LT_Filter': f"{lt_filter:.1f}",
            'TOTAL_SCORE': total_score, 
            'ACTION': traffic_light,
            'Open': last['Open'],
            'High': last['High'], # <-- ADDED
            'Low': last['Low'],   # <-- ADDED
            'Close': last['Close'],
            'Volume_Metric': last['Volume_Metric'] # <-- ADDED
        })

    return pd.DataFrame(scorecard)