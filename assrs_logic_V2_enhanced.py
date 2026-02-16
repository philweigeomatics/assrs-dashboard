import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels
from datetime import datetime
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress frequency warning for time series models
warnings.filterwarnings('ignore', category=ValueWarning, 
                       message='.*date index.*frequency.*')



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


def assign_sector_action(sector_score, market_regime_info, is_benchmark=False):
    """
    Assign action label based on sector score AND market context.
    """
    if is_benchmark:
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


def _score_single_series(df, as_of_date, sector_name, fit_lookback=FIT_LOOKBACK_DAYS):
    """
    Internal helper to score a single time series (PPI or CSI300).

    Args:
        df: DataFrame with OHLCV data (index = DatetimeIndex)
        as_of_date: Date to score
        sector_name: Name for logging
        fit_lookback: Number of days to use for model fitting

    Returns:
        dict with score and metadata, or None if failed
    """
    # Filter data up to as_of_date
    df_filtered = df.loc[df.index <= as_of_date].copy()

    if len(df_filtered) < fit_lookback:
        return None

    # Calculate returns
    df_filtered['returns'] = df_filtered['Close'].pct_change()
    df_filtered['returns'] = df_filtered['returns'].replace(0.0, 0.000001)

    # Prepare volume metric
    if 'Volume_Metric' not in df_filtered.columns:
        if 'Volume' in df_filtered.columns:
            # Create volume metric from raw volume
            df_filtered['Volume_Metric'] = (
                df_filtered['Volume'] / df_filtered['Volume'].rolling(window=100).mean()
            )
        else:
            # No volume available, use constant
            df_filtered['Volume_Metric'] = 1.0

    # Prepare model data
    model_data = df_filtered[['returns', 'Volume_Metric']].iloc[-fit_lookback:].dropna()

    if len(model_data) < (fit_lookback * 0.8):
        return None

    # Try to fit Markov model
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
        # Fallback: without volume
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
            return None

    if res is None:
        return None

    # Identify bull regime
    try:
        bull_regime = 0 if res.params['const[0]'] > res.params['const[1]'] else 1
        bear_regime = 1 - bull_regime
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return None

    # Extract bull probability
    try:
        bull_regime_probability = res.smoothed_marginal_probabilities[bull_regime].iloc[-1]
    except Exception:
        return None

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

    # Get latest OHLCV
    last = df_filtered.iloc[-1]

    return {
        'Sector': sector_name,
        'Date': as_of_date.strftime('%Y-%m-%d'),
        'TOTAL_SCORE': total_score,
        'Bull_Prob_Raw': bull_regime_probability,
        'Volume_Z': latest_volume_z,
        'Open': last.get('Open', np.nan),
        'High': last.get('High', np.nan),
        'Low': last.get('Low', np.nan),
        'Close': last['Close'],
        'Volume_Metric': last.get('Volume_Metric', 1.0),
        'Fit_Type': fit_type
    }

# --- 3. MAIN SCORING FUNCTION (ENHANCED V3 WITH CSI300 SUPPORT) ---
def calculate_regime_scores(all_sector_ppi_data, as_of_date, 
                           historical_scores=None,
                           market_index_df=None):
    """
    Enhanced V3: Calculates regime scores with CSI300 market benchmark.

    Key improvements:
    - Uses real CSI300 index for market regime (not synthetic MARKET_PROXY)
    - CSI300 and PPIs both use returns (unit-agnostic)
    - Fallback to sector median if CSI300 unavailable

    Args:
        all_sector_ppi_data: Dict of {sector: DataFrame} with PPI data
        as_of_date: datetime for point-in-time scoring
        historical_scores: Optional DataFrame with past scores for adaptive thresholds
        market_index_df: Optional CSI300 DataFrame (DatetimeIndex, OHLCV columns)

    Returns:
        DataFrame with scores + market context metadata
    """
    sector_list = list(all_sector_ppi_data.keys())
    scorecard = []
    fit_errors = []

    # --- PHASE 1: Score All Sectors (PPI-based) ---
    print(f"\n=== Scoring {len(sector_list)} sectors for {as_of_date.strftime('%Y-%m-%d')} ===")

    for sector, full_df in all_sector_ppi_data.items():
        result = _score_single_series(full_df, as_of_date, sector)

        if result is not None:
            scorecard.append(result)
        else:
            fit_errors.append((sector, 'Scoring failed'))

    # Convert to DataFrame
    df_scores = pd.DataFrame(scorecard)

    if df_scores.empty:
        print("⚠️  WARNING: All sectors failed to score!")
        if fit_errors:
            print("Fit errors:")
            for sector, error in fit_errors[:5]:
                print(f"  {sector}: {error}")
        return pd.DataFrame()

    # --- PHASE 2: Calculate Market Score (CSI300 or Fallback) ---
    market_score = None
    market_source = None

    # Try CSI300 first
    if market_index_df is not None and not market_index_df.empty:
        print("📊 Calculating market regime from CSI300 index...")

        result = _score_single_series(market_index_df, as_of_date, 'CSI300')

        if result is not None:
            market_score = result['TOTAL_SCORE']
            market_source = 'CSI300'
            print(f"✓ Market score from CSI300: {market_score:.3f}")
        else:
            print("⚠️  CSI300 scoring failed, using fallback...")

    # Fallback: use median of sector scores
    if market_score is None:
        market_score = df_scores['TOTAL_SCORE'].median()
        market_source = 'Sector Median'
        print(f"✓ Market score from sector median: {market_score:.3f}")

    # Classify market regime
    market_regime_info = classify_market_regime(market_score)

    # --- PHASE 3: Calculate Adaptive Thresholds ---
    if historical_scores is not None and 'TOTAL_SCORE' in historical_scores.columns:
        thresholds = get_adaptive_thresholds(historical_scores['TOTAL_SCORE'])
    else:
        thresholds = {'buy': 0.75, 'sell': 0.25, 'method': 'fallback'}

    # --- PHASE 4: Assign Context-Aware Actions ---
    def assign_action_row(row):
        # No benchmark sector in this version (CSI300 is external)
        action, sizing = assign_sector_action(
            row['TOTAL_SCORE'],
            market_regime_info,
            is_benchmark=False
        )
        return pd.Series({'ACTION': action, 'Position_Size': sizing})

    actions = df_scores.apply(assign_action_row, axis=1)
    df_scores['ACTION'] = actions['ACTION']
    df_scores['Position_Size'] = actions['Position_Size']

    # --- PHASE 5: Add Market Context Metadata ---
    df_scores['Market_Score'] = market_score
    df_scores['Market_Regime'] = market_regime_info['label']
    df_scores['Market_Source'] = market_source  # Track data source
    df_scores['Strategy'] = market_regime_info['strategy']
    df_scores['Threshold_Buy'] = thresholds['buy']
    df_scores['Threshold_Sell'] = thresholds['sell']

    # --- PHASE 6: Calculate Excess Probability (Sector Alpha) ---
    df_scores['Excess_Prob'] = df_scores['TOTAL_SCORE'] - market_score

    # --- PHASE 7: Cross-Sector Validation Metrics ---
    # Dispersion (rotation intensity)
    dispersion = df_scores['TOTAL_SCORE'].std() if len(df_scores) > 1 else 0.0

    if dispersion > 0.35:
        rotation_status = "HIGH ROTATION"
    elif dispersion < 0.20:
        rotation_status = "LOW ROTATION"
    else:
        rotation_status = "MODERATE ROTATION"

    df_scores['Rotation_Status'] = rotation_status
    df_scores['Dispersion'] = dispersion

    # --- Print Summary ---
    print(f"\n{'='*60}")
    print(f"SCORING SUMMARY - {as_of_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    print(f"Market Source:    {market_source}")
    print(f"Market Regime:    {market_regime_info['label']} (Score: {market_score:.3f})")
    print(f"Rotation Status:  {rotation_status} (Dispersion: {dispersion:.3f})")
    print(f"Thresholds:       BUY>{thresholds['buy']:.2f}, SELL<{thresholds['sell']:.2f} ({thresholds['method']})")
    print(f"Sectors Scored:   {len(df_scores)} | Errors: {len(fit_errors)}")

    if fit_errors and len(fit_errors) <= 3:
        print(f"\nFit Errors:")
        for sector, error in fit_errors:
            print(f"  {sector}: {error}")

    # Show top/bottom sectors
    top3 = df_scores.nlargest(3, 'TOTAL_SCORE')[['Sector', 'TOTAL_SCORE', 'ACTION']]
    bottom3 = df_scores.nsmallest(3, 'TOTAL_SCORE')[['Sector', 'TOTAL_SCORE', 'ACTION']]

    print(f"\nTop 3 Sectors:")
    for _, row in top3.iterrows():
        print(f"  {row['Sector']:20s} {row['TOTAL_SCORE']:.3f}  {row['ACTION']}")

    print(f"\nBottom 3 Sectors:")
    for _, row in bottom3.iterrows():
        print(f"  {row['Sector']:20s} {row['TOTAL_SCORE']:.3f}  {row['ACTION']}")

    print(f"{'='*60}\n")

    return df_scores


# --- 4. CONVENIENCE WRAPPER FOR BACKWARD COMPATIBILITY ---

def calculate_regime_scores_simple(all_sector_ppi_data, as_of_date):
    """
    Simplified wrapper that matches original function signature.
    For backward compatibility with existing code.
    """
    return calculate_regime_scores(all_sector_ppi_data, as_of_date, historical_scores=None)
