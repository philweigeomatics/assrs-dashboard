import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels
from datetime import datetime
import warnings # Import warnings to suppress them

# --- 1. MODEL CONSTANTS (Regime Switching Model) ---
K_REGIMES = 2 # 2 states: 0 = Bear, 1 = Bull
FIT_LOOKBACK_DAYS = 200 # Use ~200 days of history to fit the model

# --- 2. NEW STRATEGY THRESHOLDS ---
BUY_THRESHOLD_PROB = 0.8  # 80% probability of being in the "Bull" regime
SELL_THRESHOLD_PROB = 0.2 # 20% probability (i.e., 80% chance of "Bear")

# --- 3. CORE SCORING FUNCTION (REGIME SWITCHING V2.4) ---

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


def calculate_regime_scores(all_sector_ppi_data, as_of_date):
    """
    Calculates the ASSRS scorecard for all PPIs at a specific point in time
    using a 2-State Markov Regime Switching Model.
    """
    
    sector_list = list(all_sector_ppi_data.keys())
    scorecard = []
    
    for sector, full_df in all_sector_ppi_data.items():
        
        df = full_df.loc[full_df.index <= as_of_date].copy()

        if len(df) < FIT_LOOKBACK_DAYS:
            continue
            
        df['returns'] = df['Close'].pct_change()
        df['returns'] = df['returns'].replace(0.0, 0.000001) # Jitter
        
        model_data = df[['returns', 'Volume_Metric']].iloc[-FIT_LOOKBACK_DAYS:].dropna()

        if len(model_data) < (FIT_LOOKBACK_DAYS * 0.8): 
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
                continue 

        if res is None:
            continue

        try:
            bull_regime = 0 if res.params['const[0]'] > res.params['const[1]'] else 1
            bear_regime = 1 - bull_regime
        except (KeyError, ValueError, np.linalg.LinAlgError):
            continue

        try:
            bull_regime_probability = res.smoothed_marginal_probabilities[bull_regime].iloc[-1]
        except Exception:
            continue 
        
        total_score = bull_regime_probability
        
        traffic_light = 'YELLOW (HOLD/WATCH)'
        if total_score >= BUY_THRESHOLD_PROB:
            traffic_light = 'GREEN (BUY)'
        elif total_score < SELL_THRESHOLD_PROB:
            traffic_light = 'RED (EXIT/AVOID)'
            
        # 8. Append results
        # ---!!!--- NEW: ADD ALL DATA TO THE OUTPUT ---!!!---
        last = df.iloc[-1]
        scorecard.append({
            'Sector': sector,
            'Date': as_of_date.strftime('%Y-%m-%d'),
            'TOTAL_SCORE': f"{total_score:.2f}",
            'ACTION': traffic_light,
            'Open': last['Open'],
            'High': last['High'], # <-- ADDED
            'Low': last['Low'],   # <-- ADDED
            'Close': last['Close'],
            'Volume_Metric': last['Volume_Metric'], # <-- ADDED
            'Fit_Type': fit_type
        })

    return pd.DataFrame(scorecard)