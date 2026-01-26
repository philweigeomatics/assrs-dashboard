import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels
from datetime import datetime
import warnings


# --- 1. MODEL CONSTANTS ---
K_REGIMES = 2
FIT_LOOKBACK_DAYS = 200

# REMOVED: Fixed thresholds (now adaptive)
# Will calculate dynamically based on recent score distribution


# --- 2. HELPER FUNCTIONS ---

def _fit_model(mod):
    """Internal helper to fit a model, suppressing all warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)
        warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.EstimationWarning)
        warnings.filterwarnings("ignore", category=statsmodels.tsa.base.tsa_model.ValueWarning)
        
        try:
            return mod.fit(method='nm', search_reps=10, maxiter=500, verbose=0)
        except Exception:
            return mod.fit(search_reps=10, maxiter=500, verbose=0)


def classify_market_regime(market_score):
    """
    Classify overall market regime from MARKET_PROXY score.
    Returns regime info dict with position sizing multipliers.
    """
    if market_score > 0.75:
        return {
            'regime': 'STRONG_BULL',
            'label': '🚀 Strong Bull',
            'confidence_multiplier': 1.0,
            'max_sectors': 5,
            'strategy': 'Aggressive sector rotation'
        }
    elif market_score > 0.60:
        return {
            'regime': 'BULL',
            'label': '📈 Bull',
            'confidence_multiplier': 0.8,
            'max_sectors': 3,
            'strategy': 'Selective sector bets'
        }
    elif market_score > 0.40:
        return {
            'regime': 'NEUTRAL',
            'label': '📊 Neutral',
            'confidence_multiplier': 0.5,
            'max_sectors': 2,
            'strategy': 'Defensive or market-neutral'
        }
    elif market_score > 0.25:
        return {
            'regime': 'BEAR',
            'label': '📉 Bear',
            'confidence_multiplier': 0.3,
            'max_sectors': 1,
            'strategy': 'Defensive only (Healthcare, Utilities)'
        }
    else:
        return {
            'regime': 'DEEP_BEAR',
            'label': '🔻 Deep Bear',
            'confidence_multiplier': 0.1,
            'max_sectors': 0,
            'strategy': 'Cash or short positions'
        }


def get_adaptive_thresholds(recent_scores, lookback=60):
    """
    Calculate dynamic BUY/SELL thresholds based on recent distribution.
    """
    if len(recent_scores) < lookback:
        # Fallback to conservative fixed thresholds
        return {
            'buy': 0.75,
            'sell': 0.25,
            'method': 'fallback'
        }
    
    recent = recent_scores.tail(lookback)
    
    buy_threshold = recent.quantile(0.70)  # Top 30%
    sell_threshold = recent.quantile(0.30)  # Bottom 30%
    
    # Ensure minimum spread
    if buy_threshold - sell_threshold < 0.25:
        buy_threshold = 0.70
        sell_threshold = 0.30
    
    return {
        'buy': buy_threshold,
        'sell': sell_threshold,
        'method': 'adaptive',
        'median': recent.median()
    }


def assign_sector_action(sector_score, market_regime_info, is_market_proxy=False):
    """
    Assign action label based on sector score AND market context.
    """
    if is_market_proxy:
        return 'BENCHMARK', 0.0
    
    market_regime = market_regime_info['regime']
    sector_bull = sector_score > 0.70
    sector_bear = sector_score < 0.30
    
    # Strong Bull Market
    if market_regime == 'STRONG_BULL':
        if sector_bull:
            return '🟢 STRONG BUY', 1.0
        elif sector_bear:
            return '🔴 AVOID', 0.0
        else:
            return '🟡 HOLD/WATCH', 0.5
    
    # Bull Market
    elif market_regime == 'BULL':
        if sector_bull:
            return '🟢 BUY', 0.8
        elif sector_bear:
            return '🔴 AVOID', 0.0
        else:
            return '🟡 NEUTRAL', 0.3
    
    # Neutral Market
    elif market_regime == 'NEUTRAL':
        if sector_bull:
            return '🟢 SELECTIVE BUY', 0.5
        else:
            return '⚪ CASH', 0.0
    
    # Bear Market
    elif market_regime == 'BEAR':
        if sector_bull:
            return '🟡 DEFENSIVE BUY', 0.3
        else:
            return '⚪ CASH / SHORT', 0.0
    
    # Deep Bear
    else:
        if sector_bull:
            return '🟡 WATCH', 0.1
        else:
            return '⚪ CASH / SHORT', 0.0


# --- 3. MAIN SCORING FUNCTION (ENHANCED V2.5) ---

def calculate_regime_scores(all_sector_ppi_data, as_of_date, historical_scores=None):
    """
    Enhanced V2.5: Calculates regime scores with:
    - Market context awareness (MARKET_PROXY as reference)
    - Volume confirmation adjustment
    - Momentum adjustment
    - Adaptive thresholds
    - Cross-sector validation
    
    Args:
        all_sector_ppi_data: Dict of {sector: DataFrame}
        as_of_date: datetime for point-in-time scoring
        historical_scores: Optional DataFrame with past scores for adaptive thresholds
    
    Returns:
        DataFrame with scores + market context metadata
    """
    
    sector_list = list(all_sector_ppi_data.keys())
    scorecard = []
    fit_errors = []
    
    # --- PHASE 1: Score All Sectors (Including MARKET_PROXY) ---
    
    for sector, full_df in all_sector_ppi_data.items():
        
        df = full_df.loc[full_df.index <= as_of_date].copy()

        if len(df) < FIT_LOOKBACK_DAYS:
            fit_errors.append((sector, 'Insufficient history'))
            continue
            
        df['returns'] = df['Close'].pct_change()
        df['returns'] = df['returns'].replace(0.0, 0.000001)
        
        model_data = df[['returns', 'Volume_Metric']].iloc[-FIT_LOOKBACK_DAYS:].dropna()

        if len(model_data) < (FIT_LOOKBACK_DAYS * 0.8):
            fit_errors.append((sector, 'Too many NaN rows'))
            continue
        
        res = None
        fit_type = "Full (Ret+Vol+Var)"
        
        try:
            mod = sm.tsa.MarkovRegression(
                endog=model_data['returns'], 
                k_regimes=K_REGIMES, 
                trend='c', 
                exog=model_data['Volume_Metric'], 
                switching_variance=True
            )
            res = _fit_model(mod)
            
        except Exception as e1:
            try:
                fit_type = "Fallback (Ret+Var)"
                mod_v21 = sm.tsa.MarkovRegression(
                    endog=model_data['returns'], 
                    k_regimes=K_REGIMES, 
                    trend='c', 
                    switching_variance=True
                )
                res = _fit_model(mod_v21)
            except Exception as e2:
                fit_errors.append((sector, f'Model fit failed: {str(e2)[:50]}'))
                continue

        if res is None:
            fit_errors.append((sector, 'Model returned None'))
            continue

        try:
            bull_regime = 0 if res.params['const[0]'] > res.params['const[1]'] else 1
            bear_regime = 1 - bull_regime
        except (KeyError, ValueError, np.linalg.LinAlgError) as e:
            fit_errors.append((sector, f'Regime detection failed: {str(e)[:30]}'))
            continue

        try:
            bull_regime_probability = res.smoothed_marginal_probabilities[bull_regime].iloc[-1]
        except Exception as e:
            fit_errors.append((sector, f'Probability extraction failed: {str(e)[:30]}'))
            continue
        
        # --- ENHANCEMENT 1: MOMENTUM ADJUSTMENT ---
        total_score = bull_regime_probability
        try:
            recent_probs = res.smoothed_marginal_probabilities[bull_regime].iloc[-5:]
            if len(recent_probs) >= 5:
                prob_momentum = (recent_probs.iloc[-1] - recent_probs.iloc[0]) / 5
                momentum_adj = np.clip(prob_momentum * 2, -0.10, 0.10)
                total_score = total_score + momentum_adj
        except:
            pass  # Fallback to raw probability
        
        # --- ENHANCEMENT 2: VOLUME CONFIRMATION ---
        latest_volume_z = model_data['Volume_Metric'].iloc[-1]
        
        # Penalize bull signals with declining volume (distribution)
        if total_score > 0.7 and latest_volume_z < -0.5:
            volume_penalty = min(0.15, abs(latest_volume_z) * 0.05)
            total_score = total_score - volume_penalty
        
        # Boost bull signals with surging volume (accumulation)
        elif total_score > 0.7 and latest_volume_z > 2.0:
            volume_boost = min(0.10, (latest_volume_z - 2) * 0.03)
            total_score = total_score + volume_boost
        
        # Ensure score stays in valid range
        total_score = np.clip(total_score, 0, 1)
        
        # Store results (no ACTION yet, will be assigned in Phase 2)
        last = df.iloc[-1]
        scorecard.append({
            'Sector': sector,
            'Date': as_of_date.strftime('%Y-%m-%d'),
            'TOTAL_SCORE': total_score,  # Float, not string!
            'Bull_Prob_Raw': bull_regime_probability,
            'Volume_Z': latest_volume_z,
            'Open': last['Open'],
            'High': last['High'],
            'Low': last['Low'],
            'Close': last['Close'],
            'Volume_Metric': last['Volume_Metric'],
            'Fit_Type': fit_type
        })

    # Convert to DataFrame for Phase 2
    df_scores = pd.DataFrame(scorecard)
    
    if df_scores.empty:
        print("WARNING: All sectors failed to score!")
        if fit_errors:
            print("Fit errors:")
            for sector, error in fit_errors[:5]:  # Show first 5
                print(f"  {sector}: {error}")
        return pd.DataFrame()
    
    # --- PHASE 2: Extract Market Regime ---
    
    market_row = df_scores[df_scores['Sector'] == 'MARKET_PROXY']
    if not market_row.empty:
        market_score = float(market_row.iloc[0]['TOTAL_SCORE'])
        market_regime_info = classify_market_regime(market_score)
    else:
        # Fallback: use median of all sectors
        market_score = df_scores['TOTAL_SCORE'].median()
        market_regime_info = classify_market_regime(market_score)
        print(f"WARNING: MARKET_PROXY not found, using median score {market_score:.2f}")
    
    # --- PHASE 3: Calculate Adaptive Thresholds ---
    
    if historical_scores is not None and 'TOTAL_SCORE' in historical_scores.columns:
        thresholds = get_adaptive_thresholds(historical_scores['TOTAL_SCORE'])
    else:
        thresholds = {'buy': 0.75, 'sell': 0.25, 'method': 'fallback'}
    
    # --- PHASE 4: Assign Context-Aware Actions ---
    
    def assign_action_row(row):
        is_market = (row['Sector'] == 'MARKET_PROXY')
        action, sizing = assign_sector_action(
            row['TOTAL_SCORE'],
            market_regime_info,
            is_market_proxy=is_market
        )
        return pd.Series({'ACTION': action, 'Position_Size': sizing})
    
    actions = df_scores.apply(assign_action_row, axis=1)
    df_scores['ACTION'] = actions['ACTION']
    df_scores['Position_Size'] = actions['Position_Size']
    
    # --- PHASE 5: Add Market Context Metadata ---
    
    df_scores['Market_Score'] = market_score
    df_scores['Market_Regime'] = market_regime_info['label']
    df_scores['Strategy'] = market_regime_info['strategy']
    df_scores['Threshold_Buy'] = thresholds['buy']
    df_scores['Threshold_Sell'] = thresholds['sell']
    
    # --- PHASE 6: Calculate Excess Probability (Sector Alpha) ---
    
    df_scores['Excess_Prob'] = df_scores.apply(
        lambda row: 0.0 if row['Sector'] == 'MARKET_PROXY' 
        else row['TOTAL_SCORE'] - market_score,
        axis=1
    )
    
    # --- PHASE 7: Cross-Sector Validation Metrics ---
    
    # Dispersion (rotation intensity)
    non_market = df_scores[df_scores['Sector'] != 'MARKET_PROXY']['TOTAL_SCORE']
    dispersion = non_market.std() if len(non_market) > 1 else 0.0
    
    if dispersion > 0.35:
        rotation_status = "HIGH ROTATION"
    elif dispersion < 0.20:
        rotation_status = "LOW ROTATION"
    else:
        rotation_status = "MODERATE ROTATION"
    
    df_scores['Rotation_Status'] = rotation_status
    df_scores['Dispersion'] = dispersion
    
    # Print summary for debugging
    print(f"\n=== V2 Scoring Summary ({as_of_date.strftime('%Y-%m-%d')}) ===")
    print(f"Market Regime: {market_regime_info['label']} (Score: {market_score:.2f})")
    print(f"Rotation Status: {rotation_status} (Dispersion: {dispersion:.2f})")
    print(f"Thresholds: BUY>{thresholds['buy']:.2f}, SELL<{thresholds['sell']:.2f} ({thresholds['method']})")
    print(f"Sectors Scored: {len(df_scores)} | Errors: {len(fit_errors)}")
    
    if fit_errors:
        print("\nFit Errors:")
        for sector, error in fit_errors[:3]:
            print(f"  {sector}: {error}")
    
    return df_scores


# --- 4. CONVENIENCE WRAPPER FOR BACKWARD COMPATIBILITY ---

def calculate_regime_scores_simple(all_sector_ppi_data, as_of_date):
    """
    Simplified wrapper that matches original function signature.
    For backward compatibility with existing code.
    """
    return calculate_regime_scores(all_sector_ppi_data, as_of_date, historical_scores=None)
