"""
Shared Technical Analysis Engine
Contains core analysis functions used across multiple pages.

All indicator parameters are centralised in the CONSTANTS block below.
Change a value once here and it propagates to every page automatically.
"""

import pandas as pd
import numpy as np
import ta
from scipy import stats
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ============================================================
#  INDICATOR CONSTANTS  — edit here, nowhere else
# ============================================================

# ── Moving Averages ─────────────────────────────────────────
MA_SHORT   = 5
MA_SHORT2  = 10    # 10-day SMA — the other short average A-share charts show
MA_MID     = 20
MA_QUARTER = 60    # 季线 — the quarter line, a standard A-share reference level
MA_LONG    = 50
MA_MACRO   = 200

# ── Bollinger Bands ──────────────────────────────────────────
BB_WINDOW  = 20
BB_STD     = 2

# ── RSI ──────────────────────────────────────────────────────
RSI_WINDOW      = 14
RSI_OVERSOLD    = 30
RSI_OVERBOUGHT  = 70
RSI_LOOKBACK    = 252   # rolling window for dynamic percentile bands

# ── ADX / DMI ────────────────────────────────────────────────
ADX_WINDOW          = 14   # standard Wilder period  (was 10 – too noisy)
# RETIRED: Savitzky-Golay is a CENTERED filter and leaked future bars into
# ADX_LOWESS. Kept only so single_debugger.py still imports; nothing in the app
# smooths with them any more. Do not reintroduce — see the note at ADX_LOWESS.
ADX_SG_WINDOW       = 7    # (unused)
ADX_SG_POLYORDER    = 2    # (unused)
ADX_EWM_SPAN        = 9    # EWM fallback span when SG is unavailable
ADX_SLOPE_WINDOW    = 5    # linear-regression window for slope
ADX_BB_WINDOW       = 20   # Bollinger-Band window on ADX itself
ADX_BB_STD          = 2

# ADX level thresholds (calibrated for window=14)
ADX_WEAK        = 20
ADX_EMERGING    = 25
ADX_STRONG      = 30
ADX_VERY_STRONG = 40

# ADX slope / acceleration thresholds
ADX_SLOPE_FLAT_LO = -0.15
ADX_SLOPE_FLAT_HI =  0.15
ADX_SLOPE_ACCEL   =  0.05
ADX_SLOPE_FAST    =  0.50
ADX_ACCEL_FAST    =  0.10
ADX_ACCEL_SLOW    = -0.10
ADX_ACCEL_PEAK    = -0.15

# DI Screaming Buy: crossover AND one-day spread widening > this value.
# (Still used by the what-if simulator's rough next-day forecast.)
DI_SCREAMING_BUY_MOMENTUM = 15

# Hardened Screaming Buy/Sell (historical signal): the fixed 15-point rule is
# replaced by a volatility-normalized threshold + a volume-confirmation gate.
DI_SCREAM_SIGMA        = 2.0   # spread jump must exceed this many σ of its own recent distribution
DI_SCREAM_SIGMA_WINDOW = 60    # window for the spread-momentum volatility scale
DI_SCREAM_MO_FLOOR     = 8.0   # absolute floor so a near-dead stock can't fire on noise
DI_SCREAM_VOL_MULT     = 1.5   # signal-bar volume must be ≥ this × its 20-day average

# ── MACD (three-speed dynamic set) ───────────────────────────
MACD_FAST_F, MACD_FAST_S, MACD_FAST_SIGN =  8, 21,  5
MACD_NORM_F, MACD_NORM_S, MACD_NORM_SIGN = 12, 26,  9
MACD_SLOW_F, MACD_SLOW_S, MACD_SLOW_SIGN = 15, 30,  9

# ── Volume ───────────────────────────────────────────────────
VOL_LOOKBACK    = 100
VOL_OBV_WINDOW  = 20

# ── Market Regime (HMM) ──────────────────────────────────────
REGIME_METHOD = 'HMM'   # 'HMM', 'JUMP', or 'ATR'

# ── Structural regime anchor ─────────────────────────────────
# China's "924" policy pivot (2024-09-24) triggered a structural regime
# change in A-shares. Statistical baselines (BB-width squeeze percentile,
# RSI dynamic percentile bands) must NOT span this break — a "bottom 10%"
# reading against the prior bear regime is not comparable to the same
# reading in the current regime. We anchor after the Oct 1–7 National Day
# holiday and its chaotic re-open week so percentile windows never reach
# before it. Update this if a future break occurs.
#
# SINGLE SOURCE OF TRUTH for the whole app — portfolio_fit imports this rather
# than keeping its own copy. It previously held 2024-10-14 here and 2024-10-10
# in portfolio_fit; two constants meaning the same thing drift apart silently.
# Settled on the 14th, the Monday a full week after the 8 Oct re-open, so the
# window starts clear of that week rather than two sessions into it.
REGIME_ANCHOR = pd.Timestamp("2024-10-14")


# ============================================================
#  REGIME-AWARE ROLLING STATISTICS
# ============================================================

# PERF: fully vectorised with native pandas rolling/expanding (C-level), no
# Python per-bar loop. Only bars in [anchor_pos, anchor_pos + base_window) have
# a plain trailing window that would reach before the regime anchor; for those
# the clamped window is an EXPANDING window from the anchor, computed in one
# shot with .expanding(). Every other bar uses the plain rolling window. This
# is exactly equivalent to the per-bar clamp (verified 0.0 diff) and ~70x
# faster than the loop it replaces.

def _regime_pct_rank(series: pd.Series, base_window: int, anchor_pos: int,
                     min_periods: int = 20) -> pd.Series:
    """Percentile rank (0–1) of the CURRENT value within a regime-clamped
    trailing window (never reaching before the anchor)."""
    out = series.rolling(base_window, min_periods=min_periods).rank(pct=True)
    n = len(series)
    if 0 < anchor_pos < n:
        hi = min(anchor_pos + base_window, n)
        seg = series.iloc[anchor_pos:hi]
        out.iloc[anchor_pos:hi] = seg.expanding(min_periods=min_periods).rank(pct=True).values
    return out


def _regime_quantile(series: pd.Series, base_window: int, anchor_pos: int,
                     q: float, min_periods: int = 60) -> pd.Series:
    """Rolling q-quantile VALUE over a regime-clamped trailing window."""
    out = series.rolling(base_window, min_periods=min_periods).quantile(q)
    n = len(series)
    if 0 < anchor_pos < n:
        hi = min(anchor_pos + base_window, n)
        seg = series.iloc[anchor_pos:hi]
        out.iloc[anchor_pos:hi] = seg.expanding(min_periods=min_periods).quantile(q).values
    return out


def _regime_anchor_pos(index) -> int:
    """First index position on/after REGIME_ANCHOR (n if all pre-anchor,
    0 if all post-anchor)."""
    try:
        return int((pd.DatetimeIndex(index) < REGIME_ANCHOR).sum())
    except Exception:
        return 0


# ============================================================
#  ADX PATTERN DETECTION  (9 phases)
# ============================================================

def apply_adx_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify every bar into one of 9 ADX lifecycle phases.
    Requires: ADX, ADX_Slope, ADX_Acceleration columns.
    All thresholds reference the CONSTANTS block above.
    """
    adx  = df['ADX']
    sl   = df['ADX_Slope']
    acc  = df['ADX_Acceleration']
    sl_1 = sl.shift(1)

    p_bottoming = (
        (adx < ADX_EMERGING) &
        (sl  < 0) &
        (sl  > ADX_SLOPE_FLAT_LO) &
        (acc > ADX_SLOPE_ACCEL)
    )
    p_reversing_up = (
        (adx < ADX_EMERGING) &
        (sl  > 0) &
        (sl_1 <= 0)
    )
    p_accel_up = (
        (sl  > ADX_SLOPE_FAST) &
        (acc > ADX_ACCEL_FAST)
    )
    p_strong_stable = (
        (adx >= ADX_STRONG) &
        (sl  >= ADX_SLOPE_FLAT_LO) &
        (sl  <=  ADX_SLOPE_FLAT_HI)
    )
    p_decelerating_up = (
        (adx > ADX_EMERGING) &
        (sl  > 0) &
        (acc < ADX_ACCEL_SLOW)
    )
    p_peaking = (
        (adx >= ADX_STRONG) &
        (sl  > 0) &
        (acc < ADX_ACCEL_PEAK) &
        (sl  < sl_1)
    )
    p_reversing_down = (
        (adx >= ADX_EMERGING) &
        (sl  < 0) &
        (sl_1 > 0)
    )
    p_accel_down = (
        (sl  < -ADX_SLOPE_FAST) &
        (acc < ADX_ACCEL_SLOW)
    )
    p_decel_down = (
        (adx < ADX_STRONG) &
        (sl  < -ADX_SLOPE_FLAT_HI) &
        (acc > ADX_ACCEL_FAST)
    )

    # Assign in PRIORITY ORDER — Peaking overwrites Losing Steam
    df['ADX_Pattern'] = 'Neutral'
    df.loc[p_bottoming,       'ADX_Pattern'] = 'Bottoming'
    df.loc[p_reversing_up,    'ADX_Pattern'] = 'Reversing Up'
    df.loc[p_accel_up,        'ADX_Pattern'] = 'Accelerating Up'
    df.loc[p_strong_stable,   'ADX_Pattern'] = 'Strong Trend'
    df.loc[p_decelerating_up, 'ADX_Pattern'] = 'Losing Steam'
    df.loc[p_decel_down,      'ADX_Pattern'] = 'Slowing Down'
    df.loc[p_accel_down,      'ADX_Pattern'] = 'Accelerating Down'
    df.loc[p_reversing_down,  'ADX_Pattern'] = 'Reversing Down'
    df.loc[p_peaking,         'ADX_Pattern'] = 'Peaking'   # highest priority

    return df


# ============================================================
#  MAIN ANALYSIS ENGINE
# ============================================================

def run_single_stock_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and signals for one stock.

    Args:
        df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume

    Returns:
        DataFrame with all indicator and signal columns appended.
    """
    df_analysis = df.copy()

    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        df_analysis.index = pd.to_datetime(df_analysis.index)
    df_analysis = df_analysis.sort_index()

    # ── Moving Averages ──────────────────────────────────────
    df_analysis['MA5']   = ta.trend.sma_indicator(df_analysis['Close'], window=MA_SHORT)
    df_analysis['MA10']  = ta.trend.sma_indicator(df_analysis['Close'], window=MA_SHORT2)
    df_analysis['EMA5']  = ta.trend.ema_indicator(df_analysis['Close'], window=MA_SHORT)
    df_analysis['MA20']  = ta.trend.sma_indicator(df_analysis['Close'], window=MA_MID)
    df_analysis['MA60']  = ta.trend.sma_indicator(df_analysis['Close'], window=MA_QUARTER)
    df_analysis['MA50']  = ta.trend.sma_indicator(df_analysis['Close'], window=MA_LONG)
    df_analysis['MA200'] = ta.trend.sma_indicator(df_analysis['Close'], window=MA_MACRO)

    # ── Bollinger Bands ──────────────────────────────────────
    bb = ta.volatility.BollingerBands(df_analysis['Close'], window=BB_WINDOW, window_dev=BB_STD)
    df_analysis['BB_Upper'] = bb.bollinger_hband()
    df_analysis['BB_Lower'] = bb.bollinger_lband()
    df_analysis['BB_Width'] = (df_analysis['BB_Upper'] - df_analysis['BB_Lower']) / df_analysis['Close']

    # ── Market Regime (must run before dynamic MACD) ─────────
    df_analysis = detect_market_regime(df_analysis)

    # ── Dynamic MACD (three-speed, regime-switched) ──────────
    macd_fast = ta.trend.MACD(df_analysis['Close'],
                               window_fast=MACD_FAST_F, window_slow=MACD_FAST_S, window_sign=MACD_FAST_SIGN)
    macd_norm = ta.trend.MACD(df_analysis['Close'],
                               window_fast=MACD_NORM_F, window_slow=MACD_NORM_S, window_sign=MACD_NORM_SIGN)
    macd_slow = ta.trend.MACD(df_analysis['Close'],
                               window_fast=MACD_SLOW_F, window_slow=MACD_SLOW_S, window_sign=MACD_SLOW_SIGN)

    cond_high = df_analysis['Market_Regime'] == 'High Volatility'
    cond_low  = df_analysis['Market_Regime'] == 'Low Volatility'

    df_analysis['MACD']        = np.where(cond_high, macd_fast.macd(),
                                  np.where(cond_low,  macd_slow.macd(),
                                                      macd_norm.macd()))
    df_analysis['MACD_Signal'] = np.where(cond_high, macd_fast.macd_signal(),
                                  np.where(cond_low,  macd_slow.macd_signal(),
                                                      macd_norm.macd_signal()))
    df_analysis['MACD_Hist']   = df_analysis['MACD'] - df_analysis['MACD_Signal']

    # Day-by-day active parameters (so UI simulator can read the correct gear)
    df_analysis['MACD_Fast_Param'] = np.where(cond_high, MACD_FAST_F, np.where(cond_low, MACD_SLOW_F, MACD_NORM_F))
    df_analysis['MACD_Slow_Param'] = np.where(cond_high, MACD_FAST_S, np.where(cond_low, MACD_SLOW_S, MACD_NORM_S))
    df_analysis['MACD_Sign_Param'] = np.where(cond_high, MACD_FAST_SIGN, np.where(cond_low, MACD_SLOW_SIGN, MACD_NORM_SIGN))

    # ── RSI ──────────────────────────────────────────────────
    df_analysis['RSI_14'] = ta.momentum.RSIIndicator(df_analysis['Close'], window=RSI_WINDOW).rsi()

    # ── ADX / DMI ────────────────────────────────────────────
    # FIX: Use ADX_WINDOW=14 (Wilder standard). window=10 produced a noisier ADX
    # whose volatility misaligned the slope/acceleration thresholds, firing
    # patterns that shouldn't trigger and creating a messy chart.
    adx_ind = ta.trend.ADXIndicator(
        df_analysis['High'], df_analysis['Low'], df_analysis['Close'],
        window=ADX_WINDOW
    )
    df_analysis['ADX']      = adx_ind.adx()
    df_analysis['DI_Plus']  = adx_ind.adx_pos()
    df_analysis['DI_Minus'] = adx_ind.adx_neg()

    # DI crossover signals
    df_analysis['DI_Bullish_Cross'] = (
        (df_analysis['DI_Plus']  > df_analysis['DI_Minus']) &
        (df_analysis['DI_Plus'].shift(1) <= df_analysis['DI_Minus'].shift(1))
    )
    df_analysis['DI_Bearish_Cross'] = (
        (df_analysis['DI_Minus'] > df_analysis['DI_Plus']) &
        (df_analysis['DI_Minus'].shift(1) <= df_analysis['DI_Plus'].shift(1))
    )

    # DI Screaming Buy / Sell — a fresh +DI/−DI crossover whose spread also
    # expands violently in one day. Two hardening filters over the raw cross:
    #
    #   1. Volatility-normalized threshold — the 1-day spread jump must exceed
    #      DI_SCREAM_SIGMA × the stock's OWN recent spread-momentum σ (floored
    #      at DI_SCREAM_MO_FLOOR so a near-dead stock can't fire on noise).
    #      This is fair across calm large-caps and jumpy small-caps, replacing
    #      the old fixed 15-point rule that over-fired on volatile names.
    #   2. Volume confirmation — the thrust must come with real participation:
    #      signal-bar volume ≥ DI_SCREAM_VOL_MULT × its 20-day average. A big
    #      directional bar on thin volume is a trap, not an ignition.
    #
    # (Location / trend context is intentionally NOT gated here — that's a
    #  read-the-chart judgement, not a mechanical filter.)
    _di_spread    = df_analysis['DI_Plus'] - df_analysis['DI_Minus']
    _di_spread_mo = _di_spread - _di_spread.shift(1)

    _spread_mo_sigma = _di_spread_mo.rolling(DI_SCREAM_SIGMA_WINDOW, min_periods=20).std()
    _dyn_thresh = np.maximum(DI_SCREAM_MO_FLOOR, DI_SCREAM_SIGMA * _spread_mo_sigma)

    _vol_avg20 = df_analysis['Volume'].rolling(20, min_periods=5).mean()
    _vol_ok    = df_analysis['Volume'] >= DI_SCREAM_VOL_MULT * _vol_avg20

    df_analysis['DI_Screaming_Buy'] = (
        df_analysis['DI_Bullish_Cross'] &
        (_di_spread_mo >  _dyn_thresh) &
        _vol_ok
    )
    df_analysis['DI_Screaming_Sell'] = (
        df_analysis['DI_Bearish_Cross'] &
        (_di_spread_mo < -_dyn_thresh) &
        _vol_ok
    )

    # ── Downtrend / Uptrend Reversal (early DMI reversal) ─────
    # A DI cross is only a REVERSAL when it flips the prevailing trend with
    # early momentum behind it — not a mid-trend cross or a dead bounce.
    #   Downtrend Reversal (bullish):
    #     • was a downtrend  — Close was below MA20 on the prior bar
    #     • trigger          — +DI crosses above −DI (DI_Bullish_Cross)
    #     • early confirm     — MACD histogram turning up (momentum, even if
    #                           MACD is still negative — a leading signal)
    #   Uptrend Reversal (bearish) is the exact mirror.
    _macd_hist_rising  = df_analysis['MACD_Hist'] > df_analysis['MACD_Hist'].shift(1)
    _macd_hist_falling = df_analysis['MACD_Hist'] < df_analysis['MACD_Hist'].shift(1)
    _was_below_ma20    = df_analysis['Close'].shift(1) < df_analysis['MA20'].shift(1)
    _was_above_ma20    = df_analysis['Close'].shift(1) > df_analysis['MA20'].shift(1)

    df_analysis['Downtrend_Reversal'] = (
        _was_below_ma20 &
        df_analysis['DI_Bullish_Cross'] &
        _macd_hist_rising
    )
    df_analysis['Uptrend_Reversal'] = (
        _was_above_ma20 &
        df_analysis['DI_Bearish_Cross'] &
        _macd_hist_falling
    )

    # ── ADX Smoothing ─────────────────────────────────────────
    # FIX: fillna(method='ffill') is deprecated — use .ffill() directly.
    adx_filled = df_analysis['ADX'].ffill()
    # CAUSAL smoothing. This was savgol_filter, which is a CENTERED window:
    # every point was computed from ~3 bars before AND 3 bars after it, so the
    # smoothed series could not be produced in real time and quietly rewrote
    # its own recent history as new bars arrived.
    #
    # Measured on a synthetic ADX series: the value at bar 200 read 18.07 when
    # computed on day 200 and 18.47 three days later — a +0.39 revision to a bar
    # already printed, settling only once the half-window had passed. The slope
    # sign, which is what ADX_Pattern keys off, differed from the causal version
    # on 26% of bars, so up to a quarter of pattern labels were provisional and
    # any expectancy measured on them was flattered by hindsight.
    #
    # EWM was already the fallback here and is now the only path: same span,
    # strictly backward looking, and a printed bar never changes again.
    df_analysis['ADX_LOWESS'] = adx_filled.ewm(span=ADX_EWM_SPAN, adjust=False).mean()

    # ADX Bollinger Bands
    # FIX: was computed on raw ADX while slope was computed on ADX_LOWESS, creating
    # a visual/logical mismatch. Both now use the same smoothed series.
    df_analysis['ADX_BB_Middle'] = df_analysis['ADX_LOWESS'].rolling(window=ADX_BB_WINDOW).mean()
    df_analysis['ADX_BB_Std']    = df_analysis['ADX_LOWESS'].rolling(window=ADX_BB_WINDOW).std()
    df_analysis['ADX_BB_Upper']  = df_analysis['ADX_BB_Middle'] + ADX_BB_STD * df_analysis['ADX_BB_Std']
    df_analysis['ADX_BB_Lower']  = df_analysis['ADX_BB_Middle'] - ADX_BB_STD * df_analysis['ADX_BB_Std']

    # ── ADX Slope & Acceleration ──────────────────────────────
    def _linreg_slope(y):
        if len(y) < 3 or np.isnan(y).any():
            return 0.0
        x = np.arange(len(y), dtype=float)
        slope, *_ = stats.linregress(x, y)
        return float(slope)

    df_analysis['ADX_Slope'] = (
        df_analysis['ADX_LOWESS']
        .rolling(window=ADX_SLOPE_WINDOW, min_periods=3)
        .apply(_linreg_slope, raw=True)
    )

    # FIX: raw diff(1) of slope is very spiky; a 3-day EWM smooths it while
    # preserving direction changes needed for pattern detection.
    df_analysis['ADX_Acceleration'] = (
        df_analysis['ADX_Slope'].diff(1).ewm(span=3, adjust=False).mean()
    )

    # ── ADX Pattern Detection ─────────────────────────────────
    df_analysis = apply_adx_patterns(df_analysis)

    # ── Direction-Gated Entry / Exit Candidates ──────────────
    # ADX patterns describe trend STRENGTH; they say nothing about
    # direction. Fusing them with +DI vs -DI turns them into actionable
    # buy/sell signals:
    #   • Bullish patterns (Bottoming, Reversing Up, Accelerating Up)
    #     are only constructive when +DI ≥ -DI (uptrend in place /
    #     re-emerging). Otherwise they mean the existing DOWNtrend is
    #     gaining strength — exact opposite of a buy signal.
    #   • Bearish patterns (Peaking, Reversing Down, Accelerating Down)
    #     are only sell signals when +DI > -DI (current uptrend losing
    #     strength). Otherwise they describe a downtrend topping out.
    _di_bullish = df_analysis['DI_Plus'] >= df_analysis['DI_Minus']

    df_analysis['Entry_Candidate'] = ''
    df_analysis.loc[
        _di_bullish & df_analysis['ADX_Pattern'].isin(['Bottoming', 'Reversing Up']),
        'Entry_Candidate'
    ] = 'Strength Returning'
    df_analysis.loc[
        _di_bullish & (df_analysis['ADX_Pattern'] == 'Accelerating Up'),
        'Entry_Candidate'
    ] = 'Trend Accelerating'
    # Screaming Buy is the strongest signal — it already encodes its own
    # direction (Bullish Cross), so overwrite anything else on the same bar.
    df_analysis.loc[df_analysis['DI_Screaming_Buy'], 'Entry_Candidate'] = 'Screaming Buy'

    df_analysis['Exit_Candidate'] = ''
    df_analysis.loc[
        _di_bullish & df_analysis['ADX_Pattern'].isin(['Peaking', 'Reversing Down']),
        'Exit_Candidate'
    ] = 'Trend Topping'
    df_analysis.loc[
        _di_bullish & (df_analysis['ADX_Pattern'] == 'Accelerating Down'),
        'Exit_Candidate'
    ] = 'Trend Collapsing'
    df_analysis.loc[df_analysis['DI_Screaming_Sell'], 'Exit_Candidate'] = 'Screaming Sell'

    # ── OBV ──────────────────────────────────────────────────
    df_analysis['OBV'] = ta.volume.on_balance_volume(df_analysis['Close'], df_analysis['Volume'])
    df_analysis['Volume_Scaled_OBV'] = (
        df_analysis['OBV'] / df_analysis['Volume'].rolling(window=VOL_OBV_WINDOW).mean()
    )

    # ── Volume Statistics ─────────────────────────────────────
    df_analysis['VolMean_100d']  = df_analysis['Volume'].rolling(window=VOL_LOOKBACK, min_periods=20).mean()
    df_analysis['VolStd_100d']   = df_analysis['Volume'].rolling(window=VOL_LOOKBACK, min_periods=20).std()
    df_analysis['Volume_ZScore'] = (
        (df_analysis['Volume'] - df_analysis['VolMean_100d']) / df_analysis['VolStd_100d']
    )

    # ── Adaptive Parameters ───────────────────────────────────
    def _adaptive_lookback(n, min_d=60, max_d=250):
        if n < 120:   return max(min_d, int(n * 0.5))
        elif n < 250: return 120
        else:         return max_d

    lookback_period = _adaptive_lookback(len(df_analysis))
    params   = calculate_adaptive_parameters_percentile(df, lookback_days=30)
    lookback = params['obv_lookback']

    # First bar on/after the 924 structural regime anchor — statistical
    # percentile baselines below must not reach before it.
    anchor_pos = _regime_anchor_pos(df_analysis.index)

    # ── Signal Column Initialisation ─────────────────────────
    df_analysis['Signal_Accumulation']  = False
    df_analysis['Signal_Squeeze']       = False
    df_analysis['Signal_Golden_Launch'] = False
    df_analysis['Exit_MACD_Lead'] = False
    
    # ==================== SHARED HELPERS ====================
    def consecutive_days_filter(series, min_days=3):
        if series.sum() == 0:
            return pd.Series(False, index=series.index)
        groups = (series != series.shift()).cumsum()
        count = series.groupby(groups).transform('size')
        return (series) & (count >= min_days)

    # ==================== PHASE 1: ACCUMULATION ====================
    df_analysis['PriceChg'] = df_analysis['Close'].pct_change(periods=lookback)

    # FIX: Use the Volume_Scaled_OBV difference instead of pct_change on raw OBV.
    # This prevents the negative-number math bug and normalizes the volume perfectly.
    df_analysis[f'OBV_Scaled_Diff'] = df_analysis['Volume_Scaled_OBV'] - df_analysis['Volume_Scaled_OBV'].shift(lookback)

    # OBV threshold scaled to lookback: lookback * 0.35 requires ~68% up-close days,
    # ensuring a clear buyer tilt rather than ~60% (lookback * 0.2) or near-noise 53% (fixed 0.5).
    obv_acc_threshold = lookback * 0.35

    accumulation_raw = (
        (df_analysis[f'OBV_Scaled_Diff'] > obv_acc_threshold) &  # ~68% up-close days required
        (df_analysis['PriceChg'].abs() < params['price_flat_threshold']) &  # Price genuinely trapped
        (df_analysis['RSI_14'] < 60) &  # Not overbought
        (df_analysis['Close'] > df_analysis['MA200'])  # Macro uptrend context
    )
    # Duration filter: all conditions must hold for 3 consecutive days (mirrors Squeeze filter).
    # Prevents single-day OBV spikes from firing the signal.
    df_analysis.loc[consecutive_days_filter(accumulation_raw, min_days=3), 'Signal_Accumulation'] = True
    
    # ==================== PHASE 2: SQUEEZE ====================
    # Regime-aware BB-width percentile: the trailing window never spans the
    # 924 anchor, so "bottom 10% of BB width" is measured within one regime.
    df_analysis['BB_Width_Percentile'] = _regime_pct_rank(
        df_analysis['BB_Width'], lookback_period, anchor_pos, min_periods=20
    )

    df_analysis['Squeeze_Raw'] = df_analysis['BB_Width_Percentile'] <= 0.10

    df_analysis['Signal_Squeeze'] = consecutive_days_filter(df_analysis['Squeeze_Raw'], min_days=3)
    
    squeeze_groups = (df_analysis['Signal_Squeeze'] != df_analysis['Signal_Squeeze'].shift()).cumsum()
    df_analysis['Squeeze_Age'] = df_analysis.groupby(squeeze_groups)['Signal_Squeeze'].cumsum()
    df_analysis['Squeeze_Mature'] = (df_analysis['Signal_Squeeze']) & (df_analysis['Squeeze_Age'] >= 5)
    
    # NEW: Directional Squeeze Breakout Triggers
    # A squeeze is just stored energy. These signals tell us which way the energy is releasing.
    df_analysis['Squeeze_Fired_Bullish'] = (
        df_analysis['Signal_Squeeze'].shift(1) &
        (df_analysis['Close'] > df_analysis['BB_Upper']) &
        (df_analysis['MACD']  > df_analysis['MACD_Signal'])
    )
    df_analysis['Squeeze_Fired_Bearish'] = (
        df_analysis['Signal_Squeeze'].shift(1) &
        (df_analysis['Close'] < df_analysis['BB_Lower']) &
        (df_analysis['MACD']  < df_analysis['MACD_Signal'])
    )

    # Legacy compatibility
    df_analysis['Min_Width_120d'] = df_analysis['BB_Width'].rolling(window=120, min_periods=20).min()

    # ── MACD Scenarios ────────────────────────────────────────
    df_analysis['MACD_Gap']          = df_analysis['MACD'] - df_analysis['MACD_Signal']
    df_analysis['MACD_Momentum']     = df_analysis['MACD'] - df_analysis['MACD'].shift(1)
    df_analysis['MACD_Momentum_Pct'] = df_analysis['MACD'].pct_change()
    df_analysis['MACD_Acceleration'] = (
        df_analysis['MACD_Momentum_Pct'] - df_analysis['MACD_Momentum_Pct'].shift(1)
    )

    # Scenario 1: Classic crossover
    classic_crossover = (
        (df_analysis['MACD_Gap'].shift(1) <= 0) &
        (df_analysis['MACD_Gap']          >  0)
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
                (df_analysis['MACD_Momentum_Pct']           > -0.02)
            )
        )
    )

    # Scenario 3: Bottoming
    df_analysis['MACD_5d_Low']  = df_analysis['MACD'].rolling(5).min()
    df_analysis['MACD_10d_Low'] = df_analysis['MACD'].rolling(10).min()

    # FIX: comment said "Need at least 2 signals" but threshold was >= 4 (all four).
    # Corrected to >= 2 to match stated design intent.
    bottoming = (
        (df_analysis['MACD'] < 0) &
        (df_analysis['MACD'] >= df_analysis['MACD_10d_Low']) &
        (df_analysis['MACD'] <= df_analysis['MACD_10d_Low'] * 0.95) &
        (
            (
                (df_analysis['MACD'] >= df_analysis['MACD'].shift(1)).astype(int) +
                (df_analysis['MACD'] >  df_analysis['MACD_5d_Low']).astype(int) +
                (df_analysis['MACD_Gap'] > df_analysis['MACD_Gap'].shift(1)).astype(int) +
                (df_analysis['OBV']  >  df_analysis['OBV'].shift(3)).astype(int)
            ) >= 2
        )
    )

    # Scenario 4: Momentum building
    df_analysis['MACD_Gap_Ratio']        = df_analysis['MACD_Gap'] / df_analysis['MACD_Signal'].abs()
    df_analysis['MACD_Gap_Ratio_Change'] = (
        df_analysis['MACD_Gap_Ratio'] - df_analysis['MACD_Gap_Ratio'].shift(1)
    )

    momentum_building = (
        (df_analysis['MACD_Gap']               > 0) &
        (df_analysis['MACD_Gap'].shift(1)       > 0) &
        (df_analysis['MACD_Gap'].shift(2)       > 0) &
        (df_analysis['MACD_Gap']               > df_analysis['MACD_Gap'].shift(1)) &
        (df_analysis['MACD_Gap_Ratio']          > df_analysis['MACD_Gap_Ratio'].shift(1)) &
        (df_analysis['MACD_Gap_Ratio'].shift(1) > df_analysis['MACD_Gap_Ratio'].shift(2)) &
        (df_analysis['MACD_Gap_Ratio_Change']           > 0.10) &
        (df_analysis['MACD_Gap_Ratio_Change'].shift(1)  > 0.10) &
        (df_analysis['MACD']                    > df_analysis['MACD'].shift(1))
    )

    macd_trigger = classic_crossover | approaching_from_below | bottoming | momentum_building

    # Scenario 5: Peaking (bearish)
    # FIX: the old condition fired on a SINGLE one-bar down-tick in MACD with no
    # requirement that momentum was actually rolling over. Right after a bullish
    # crossover (MACD climbing off zero), one noisy down-bar — while the
    # histogram was still EXPANDING — falsely flagged "Peaking" at the start of
    # an uptrend (the yellow triangle sitting on the crossover). A real peak is a
    # momentum rollover: the histogram (MACD_Gap) must CONTRACT for two
    # consecutive bars. That cannot happen right after a crossover (the histogram
    # is expanding there) and it filters single-bar MACD noise, while still
    # catching genuine tops (verified against synthetic crossover-vs-peak cases).
    peaking = (
        (df_analysis['MACD_Gap']              > 0) &
        (df_analysis['MACD']                  > 0) &
        (df_analysis['MACD_Signal']           > 0) &
        (df_analysis['MACD']                  < df_analysis['MACD'].shift(1)) &   # MACD turned down
        (df_analysis['MACD'].shift(1)         > df_analysis['MACD'].shift(2)) &   # was rising into the turn
        (df_analysis['MACD_Gap']              < df_analysis['MACD_Gap'].shift(1)) &        # histogram contracting…
        (df_analysis['MACD_Gap'].shift(1)     < df_analysis['MACD_Gap'].shift(2))          # …for a 2nd bar = confirmed rollover
    )

    # Scenario 6: Bearish crossover
    bearish_crossover = (
        (df_analysis['MACD_Gap'].shift(1) > 0) &
        (df_analysis['MACD_Gap']          <= 0) &
        (df_analysis['MACD'].shift(1)     > 0)
    )

    df_analysis['MACD_ClassicCrossover']  = classic_crossover
    df_analysis['MACD_Approaching']       = approaching_from_below
    df_analysis['MACD_Bottoming']         = bottoming
    df_analysis['MACD_MomentumBuilding']  = momentum_building
    df_analysis['MACD_Trigger']           = macd_trigger
    df_analysis['MACD_Peaking']           = peaking
    df_analysis['MACD_BearishCrossover']  = bearish_crossover

    # ── MACD Signal Line Trend ────────────────────────────────
    df_analysis['MACD_Signal_Slope'] = df_analysis['MACD_Signal'] - df_analysis['MACD_Signal'].shift(2)
    df_analysis['Large_Uptrend']     = df_analysis['MACD_Signal_Slope'] > 0
    df_analysis['Large_Downtrend']   = df_analysis['MACD_Signal_Slope'] < 0

    # ── RSI Dynamic Percentile Thresholds ────────────────────
    # Regime-aware: the P10/P90 bands are measured only within the current
    # regime (window never spans the 924 anchor), so "overbought/oversold
    # relative to recent history" doesn't blend the pre-924 bear regime in.
    lookback_window = min(RSI_LOOKBACK, len(df_analysis))
    df_analysis['RSI_P10'] = _regime_quantile(
        df_analysis['RSI_14'], lookback_window, anchor_pos, q=0.10, min_periods=60
    )
    df_analysis['RSI_P90'] = _regime_quantile(
        df_analysis['RSI_14'], lookback_window, anchor_pos, q=0.90, min_periods=60
    )

    df_analysis['RSI_Bottoming'] = (
        (df_analysis['RSI_14'] <= df_analysis['RSI_P10']) &
        (df_analysis['RSI_14'] <= RSI_OVERSOLD)
    )
    df_analysis['RSI_Peaking'] = (
        (df_analysis['RSI_14'] >= df_analysis['RSI_P90']) &
        (df_analysis['RSI_14'] >= RSI_OVERBOUGHT)
    )

    # ── Exit Signals ──────────────────────────────────────────
    macd_bearish_cross = (
        (df_analysis['MACD'].shift(1)     > df_analysis['MACD_Signal'].shift(1)) &
        (df_analysis['MACD']              < df_analysis['MACD_Signal']) &
        (df_analysis['MACD'].shift(1)     > 0) &
        (df_analysis['MACD_Hist']         < df_analysis['MACD_Hist'].shift(1))
    )
    ma_cross_down = (
        (df_analysis['MA20'].shift(1) > df_analysis['MA50'].shift(1)) &
        (df_analysis['MA20']          < df_analysis['MA50']) &
        (df_analysis['ADX']           > ADX_WEAK)
    )
    df_analysis.loc[macd_bearish_cross | ma_cross_down, 'Exit_MACD_Lead'] = True

    # ── Signal Score ──────────────────────────────────────────
    df_analysis['Signal_Score'] = 0

    for pat in ['Bottoming', 'Reversing Up', 'Accelerating Up', 'Strong Trend']:
        df_analysis.loc[df_analysis['ADX_Pattern'] == pat, 'Signal_Score'] += 1

    for col in ['RSI_Bottoming', 'MACD_Bottoming', 'MACD_Approaching',
                'MACD_ClassicCrossover', 'MACD_MomentumBuilding', 'Signal_Accumulation']:
        if col in df_analysis.columns:
            df_analysis.loc[df_analysis[col], 'Signal_Score'] += 1

    for col in ['MACD_Peaking', 'MACD_BearishCrossover', 'RSI_Peaking']:
        if col in df_analysis.columns:
            df_analysis.loc[df_analysis[col], 'Signal_Score'] -= 1

    for pat in ['Peaking', 'Reversing Down', 'Accelerating Down']:
        df_analysis.loc[df_analysis['ADX_Pattern'] == pat, 'Signal_Score'] -= 1

    df_analysis['Cumulative_Score'] = df_analysis['Signal_Score'].cumsum()

    return df_analysis


# ============================================================
#  ADAPTIVE PARAMETER CALCULATION
# ============================================================

def calculate_adaptive_parameters_percentile(df: pd.DataFrame, lookback_days: int = 30) -> dict:
    """
    Derive volatility-adaptive parameters based on recent volatility regime.
    """
    df_temp = df.copy()
    for w in [10, 15, 20, 25, 30]:
        df_temp[f'vol_{w}d'] = df_temp['Close'].pct_change().rolling(w).std()

    current_vol_10d = df_temp['vol_10d'].iloc[-1]
    vol_5days_ago   = df_temp['vol_10d'].iloc[-5] if len(df_temp) >= 5 else current_vol_10d

    vol_trend = (
        'rising'  if current_vol_10d > vol_5days_ago * 1.2 else
        'falling' if current_vol_10d < vol_5days_ago * 0.8 else
        'stable'
    )

    recent_vol = df_temp['vol_10d'].iloc[-lookback_days:].dropna()

    if len(recent_vol) < 10:
        return {
            'vol_window': 20,
            'obv_lookback': 10,
            'obv_threshold': 0.025,
            'current_vol': current_vol_10d,
            'vol_regime': 'insufficient_data',
            'price_flat_threshold': 0.04  # tight default for insufficient-data fallback
        }

    p25, p50, p75, p90 = recent_vol.quantile([0.25, 0.50, 0.75, 0.90])

    if   current_vol_10d >= p90: vol_regime, percentile, vol_window, obv_lookback = 'very_high', 90, 10,  5
    elif current_vol_10d >= p75: vol_regime, percentile, vol_window, obv_lookback = 'high',      75, 15,  7
    elif current_vol_10d >= p50: vol_regime, percentile, vol_window, obv_lookback = 'medium_high',60, 20, 10
    elif current_vol_10d >= p25: vol_regime, percentile, vol_window, obv_lookback = 'medium_low', 40, 25, 12
    else:                         vol_regime, percentile, vol_window, obv_lookback = 'low',        20, 30, 15

    if vol_trend == 'rising' and vol_regime in ['high', 'very_high']:
        vol_window = max(10, vol_window - 5)
        obv_lookback = max(5, obv_lookback - 2)
    
    current_vol = df_temp[f'vol_{min(vol_window, 30)}d'].iloc[-1]
    obv_threshold = max(0.008, min(0.08, current_vol * obv_lookback * 0.35))
    price_flat_threshold = max(0.03, min(0.05, current_vol * obv_lookback * 0.5))  # cap tightened 0.08→0.05
    
    return {
        'vol_window': vol_window, 'current_vol': current_vol,
        'vol_percentile': percentile, 'vol_regime': vol_regime, 'vol_trend': vol_trend,
        'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
        'obv_lookback': obv_lookback, 'obv_threshold': obv_threshold,
        'price_flat_threshold': price_flat_threshold,
        'cycle_estimate': obv_lookback * 2
    }


# ============================================================
#  MARKET REGIME DETECTION  (HMM / JUMP)
# ============================================================

def yang_zhang_vol(df: pd.DataFrame, window: int = 20,
                   trading_days: int = 242) -> pd.Series:
    """
    Rolling Yang-Zhang (2000) volatility, annualised.

    Chosen over Parkinson / Garman-Klass because it is the only common OHLC
    estimator that captures BOTH drift and the overnight gap. A-shares gap
    hard: policy lands outside session hours and holidays stack up news, so an
    estimator that ignores close-to-open throws away a large share of the real
    variance. It is also ~14x more efficient than close-to-close, which means a
    20-day window is about as stable as a year of close-to-close — the reason
    a regime read anchored inside the current market is even possible.

    σ²_YZ = σ²_overnight + k·σ²_open→close + (1−k)·σ²_RogersSatchell

    On 涨跌停 limit days this estimator behaves far better than the range-only
    alternatives, which is a second reason to prefer it here. When a stock gaps
    to the limit and locks, High == Low == Open == Close, so the intraday terms
    collapse to zero — Parkinson and Garman-Klass therefore read a locked limit
    day as almost perfectly calm. Yang-Zhang still sees it, because the move
    lands in the overnight close-to-open term instead. What remains lost is
    only the truncation: price cannot travel past the band, so a session that
    "wanted" to move further is recorded at the limit. Measured against an
    uncensored twin series at a realistic limit frequency (~2% of sessions),
    that understates the level by ~1%, versus ~2% for Parkinson.

    The one bias that does apply: high and low come from discrete trades rather
    than the continuous path, so the recorded range is always slightly narrow.
    Verified against simulated GBM — the error shrinks from ~16% at 24 intraday
    observations to ~1% at 8000, and sits near ~2% for liquid names. Always in
    the same direction, so treat readings as a mild floor.

    Returns an all-NaN series if OHLC isn't usable; callers should fall back.
    """
    need = {"Open", "High", "Low", "Close"}
    if df is None or df.empty or not need.issubset(df.columns):
        return pd.Series(np.nan, index=df.index if df is not None else None)

    o_, h_, l_, c_ = (pd.to_numeric(df[x], errors="coerce")
                      for x in ("Open", "High", "Low", "Close"))
    # Open is the weak column in this data set — zero/NaN opens appear on some
    # index series, and log(0) would silently poison the whole window.
    if (o_ <= 0).any() or o_.isna().all():
        return pd.Series(np.nan, index=df.index)

    prev_c = c_.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.log(o_ / prev_c)      # overnight jump
        c = np.log(c_ / o_)          # open → close
        hc, ho = np.log(h_ / c_), np.log(h_ / o_)
        lc, lo = np.log(l_ / c_), np.log(l_ / o_)
    rs = hc * ho + lc * lo           # Rogers-Satchell, drift-independent

    n = int(window)
    if n < 3:
        return pd.Series(np.nan, index=df.index)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var_yz = (o.rolling(n).var(ddof=1)
              + k * c.rolling(n).var(ddof=1)
              + (1.0 - k) * rs.rolling(n).mean())
    # Rogers-Satchell can go slightly negative on tiny samples; a negative
    # variance is meaningless, not informative.
    return np.sqrt(var_yz.clip(lower=0.0)) * np.sqrt(trading_days)


def compute_vol_regime(df: pd.DataFrame, window: int = 20,
                       anchor: "pd.Timestamp | str | None" = None,
                       hi_pct: float = 0.80, lo_pct: float = 0.20,
                       trading_days: int = 242, min_obs: int = 60) -> dict:
    """
    Volatility regime by percentile rank against a FIXED-ANCHOR history.

    Deliberately different from detect_market_regime's HMM mode, which rolling
    z-scores its features over a trailing window before the model sees them.
    That makes the HMM self-normalising: in a sustained high-volatility stretch
    the trailing mean rises to meet it, the z-score decays, and the label drifts
    back to "Normal" while volatility is still objectively high. It answers "is
    today unusual versus recent days" — a different question from "which regime
    are we in", and they diverge exactly when the answer matters.

    A fixed anchor fixes the ruler. Percentiles are taken against every session
    since the 2024-09-24 structural break, so "80th percentile" means high for
    THIS market rather than high relative to a window that follows you around.

    Returns a dict with ok/label/current/percentile/thresholds. `current` and
    the thresholds are annualised decimals (0.28 = 28%).
    """
    if df is None or df.empty or len(df) < window + 2:
        return {"ok": False, "reason": "not enough price history"}

    anchor = REGIME_ANCHOR if anchor is None else pd.Timestamp(anchor)

    vol = yang_zhang_vol(df, window=window, trading_days=trading_days)
    estimator = "Yang-Zhang"
    if vol.isna().all():
        # Fall back rather than fail: close-to-close is worse but always available.
        vol = (pd.to_numeric(df["Close"], errors="coerce").pct_change()
               .rolling(window).std() * np.sqrt(trading_days))
        estimator = "close-to-close"

    vol = vol.dropna()
    if vol.empty:
        return {"ok": False, "reason": "volatility could not be computed"}

    idx = pd.to_datetime(vol.index)
    hist = vol[idx >= anchor]
    anchored = True
    if len(hist) < min_obs:
        # Not enough post-anchor data to rank against — say so rather than
        # quietly ranking against a pre-break market that no longer exists.
        hist = vol
        anchored = False

    current = float(vol.iloc[-1])
    pct = float((hist < current).sum() / len(hist))

    if pct >= hi_pct:
        label, kind = "High Volatility", "warning"
    elif pct <= lo_pct:
        label, kind = "Low Volatility", "success"
    else:
        label, kind = "Normal Volatility", "info"

    return {
        "ok": True,
        "label": label,
        "kind": kind,
        "current": current,
        "percentile": pct,
        "hi_thresh": float(hist.quantile(hi_pct)),
        "lo_thresh": float(hist.quantile(lo_pct)),
        "median": float(hist.median()),
        "estimator": estimator,
        "window": int(window),
        "anchor": anchor.strftime("%Y-%m-%d"),
        "anchored": anchored,
        "n_obs": int(len(hist)),
        "hist_start": pd.Timestamp(hist.index[0]).strftime("%Y-%m-%d"),
        "as_of": pd.Timestamp(vol.index[-1]).strftime("%Y-%m-%d"),
        "series": vol,
    }


def detect_market_regime(
    df: pd.DataFrame,
    freq: str = "daily",
    method: str | None = None,
    fallback_method: str | None = "JUMP",
    window: int | None = None,
    refit_every: int | None = None,
    min_obs: int | None = None,
    z_clip: float = 3.0,
) -> pd.DataFrame:
    """
    Volatility/Liquidity regime detection.

    Backward compatible: detect_market_regime(df) still works.
    Auto-falls back from HMM -> JUMP if insufficient data or fit fails.
    Weekly-safe: won't crash due to uninitialised model/rank_map.
    """
    if df is None or df.empty:
        return df

    m  = (method or REGIME_METHOD).upper()
    tf = (freq or "daily").lower()
    is_weekly = ("week" in tf) or tf.startswith("w") or ("周" in tf)

    if window      is None: window      = 26  if is_weekly else 90
    if refit_every is None: refit_every = 2   if is_weekly else 5
    if min_obs     is None: min_obs     = 70  if is_weekly else 120

    if len(df) < min_obs:
        if m == "HMM" and fallback_method:
            return detect_market_regime(df, freq=freq, method=fallback_method, fallback_method=None)
        df = df.copy()
        df["Market_Regime"]  = "Normal Volatility"
        df["ATR_Percentile"] = 0.5
        return df

    # ── HMM MODE ─────────────────────────────────────────────
    if m == "HMM":
        df = df.copy()
        vol_ma_len = 10 if is_weekly else 20
        ewm_span   = 2  if is_weekly else 3

        df["prev_close"] = df["Close"].shift(1)
        df["h_l"]  = df["High"] - df["Low"]
        df["h_pc"] = (df["High"] - df["prev_close"]).abs()
        df["l_pc"] = (df["Low"]  - df["prev_close"]).abs()
        df["TR"]   = df[["h_l", "h_pc", "l_pc"]].max(axis=1)
        df["NATR"] = (df["TR"] / df["Close"]) * 100

        df["Vol_MA"]   = df["Volume"].rolling(vol_ma_len).mean()
        df["RVOL"]     = df["Volume"] / (df["Vol_MA"] + 1)
        df["Log_RVOL"] = np.log(df["RVOL"] + 0.1)

        feat_df = df[["NATR", "Log_RVOL"]].copy()
        feat_df["NATR"]     = feat_df["NATR"].ewm(span=ewm_span).mean()
        feat_df["Log_RVOL"] = feat_df["Log_RVOL"].ewm(span=ewm_span).mean()

        zfn = lambda x: (x[-1] - x.mean()) / (x.std() + 1e-6)
        feat_df["scaled_natr"] = feat_df["NATR"].rolling(window=window).apply(zfn, raw=True)
        feat_df["scaled_rvol"] = feat_df["Log_RVOL"].rolling(window=window).apply(zfn, raw=True)

        features = feat_df[["scaled_natr", "scaled_rvol"]].dropna()
        if features.empty or len(features) < (window + 5):
            if fallback_method:
                return detect_market_regime(df, freq=freq, method=fallback_method, fallback_method=None)
            df["Market_Regime"]  = "Normal Volatility"
            df["ATR_Percentile"] = 0.5
            return df

        features = features.clip(-z_clip, z_clip)
        X = features.values

        final_regimes = [1] * window
        model = None
        rank_map = None
        last_good_model = None
        last_good_rank_map = None

        for i in range(window, len(X)):
            need_refit = (model is None) or (i % refit_every == 0)
            if need_refit:
                X_train = X[i - window : i]
                try:
                    mm = GaussianHMM(n_components=3, covariance_type="diag",
                                     n_iter=200, random_state=42)
                    mm.fit(X_train)
                    state_energy = [float(mm.means_[j][0] + mm.means_[j][1]) for j in range(3)]
                    ordered = np.argsort(state_energy)
                    rm = {int(ordered[0]): 0, int(ordered[1]): 1, int(ordered[2]): 2}
                    model = mm
                    rank_map = rm
                    last_good_model    = mm
                    last_good_rank_map = rm
                except Exception:
                    if last_good_model is not None:
                        model    = last_good_model
                        rank_map = last_good_rank_map
                    elif fallback_method:
                        return detect_market_regime(df, freq=freq, method=fallback_method, fallback_method=None)
                    else:
                        final_regimes.append(1)
                        continue

            seq_start    = max(0, i - window)
            decoded      = model.predict(X[seq_start : i + 1])
            current_state = int(decoded[-1])
            final_regimes.append(int(rank_map.get(current_state, 1)))

        # Smart persistence (damp single-bar spikes)
        smoothed_final = []
        curr_fixed = final_regimes[0]
        strength   = 5.0
        natr_series = features["scaled_natr"].values

        for i in range(len(final_regimes)):
            s = final_regimes[i]
            is_wide_candle = natr_series[i] > 0.5
            target_state   = 1 if (s == 0 and is_wide_candle) else s

            if target_state == 2 and target_state != curr_fixed:
                strength   = 0
                curr_fixed = 2
            elif target_state != curr_fixed:
                strength -= 1.0
            else:
                strength = min(10.0, strength + 1.0)

            if strength <= 0:
                curr_fixed = target_state
                strength   = 4.0

            smoothed_final.append(curr_fixed)

        smoothed_series = pd.Series(smoothed_final, index=features.index)
        label_map = {0: "Low Volatility", 1: "Normal Volatility", 2: "High Volatility"}

        df["Market_Regime"]  = smoothed_series.map(label_map).reindex(df.index, method="ffill")
        df["ATR_Percentile"] = (
            smoothed_series.map({0: 0.2, 1: 0.5, 2: 0.8}).reindex(df.index, method="ffill")
        )
        return df

    # ── JUMP MODE ────────────────────────────────────────────
    if m == "JUMP":
        df      = df.copy()
        vol_win = 10 if is_weekly else 20
        df["vol_rolling"] = df["Close"].pct_change().rolling(vol_win).std()

        thresh_hi = df["vol_rolling"].rolling(252, min_periods=60).quantile(0.8)
        thresh_lo = df["vol_rolling"].rolling(252, min_periods=60).quantile(0.2)

        regimes = []
        curr    = "Normal Volatility"
        for v, hi, lo in zip(df["vol_rolling"], thresh_hi, thresh_lo):
            if pd.isna(v):
                regimes.append(curr)
                continue
            if v > hi:   curr = "High Volatility"
            elif v < lo: curr = "Low Volatility"
            else:        curr = "Normal Volatility"
            regimes.append(curr)

        df["Market_Regime"]  = regimes
        df["ATR_Percentile"] = (
            pd.Series(regimes)
            .map({"Low Volatility": 0.2, "Normal Volatility": 0.5, "High Volatility": 0.8})
            .values
        )
        return df

    # ── Bare fallback ─────────────────────────────────────────
    df = df.copy()
    df["Market_Regime"]  = "Normal Volatility"
    df["ATR_Percentile"] = 0.5
    return df


# ============================================================
#  NEXT-DAY SIMULATOR
# ============================================================

def simulate_next_day_indicators(df: pd.DataFrame,
                                  price_change_pct: float,
                                  volume: float,
                                  open_tomorrow: float | None = None,
                                  high_tomorrow: float | None = None,
                                  low_tomorrow: float | None = None) -> dict | None:
    """
    Simulate what technical indicators would look like tomorrow.

    Uses dynamic MACD parameters stored in df columns (MACD_Fast/Slow/Sign_Param)
    so the simulation always matches the currently active gear.
    Uses a mini-df ADX re-calculation for an exact ADX/DI simulation.

    Args:
        df:                DataFrame produced by run_single_stock_analysis()
        price_change_pct:  Hypothetical tomorrow % price change
        volume:            Hypothetical tomorrow volume
        open/high/low_tomorrow: Optional explicit OHL for the simulated bar.
            When omitted, high and low are ESTIMATED as the close range padded
            by 0.2x the mean true range and open is assumed flat at today's
            close. That estimate is only a stand-in so ADX has something to
            chew on — supplying the real shape makes ADX and ±DI exact rather
            than approximate, and lets the ghost be drawn as a candle instead
            of a bare line.

    Returns:
        Dictionary with simulated indicator values and signals, or None if too short.
    """
    if len(df) < 30:
        return None

    latest      = df.iloc[-1]
    close_today = latest['Close']
    close_tomorrow = close_today * (1 + price_change_pct / 100)

    # ── Dynamic MACD simulation via EWM step ─────────────────
    p_fast = int(latest.get('MACD_Fast_Param', MACD_NORM_F))
    p_slow = int(latest.get('MACD_Slow_Param', MACD_NORM_S))
    p_sign = int(latest.get('MACD_Sign_Param', MACD_NORM_SIGN))

    ema_fast_today = df['Close'].ewm(span=p_fast, adjust=False).mean().iloc[-1]
    ema_slow_today = df['Close'].ewm(span=p_slow, adjust=False).mean().iloc[-1]

    alpha_fast = 2 / (p_fast + 1)
    alpha_slow = 2 / (p_slow + 1)
    alpha_sign = 2 / (p_sign + 1)

    ema_fast_tomorrow = alpha_fast * close_tomorrow + (1 - alpha_fast) * ema_fast_today
    ema_slow_tomorrow = alpha_slow * close_tomorrow + (1 - alpha_slow) * ema_slow_today

    macd_tomorrow        = ema_fast_tomorrow - ema_slow_tomorrow
    macd_signal_today    = latest['MACD_Signal']
    macd_signal_tomorrow = alpha_sign * macd_tomorrow + (1 - alpha_sign) * macd_signal_today
    macd_hist_tomorrow   = macd_tomorrow - macd_signal_tomorrow

    # ── RSI simulation via Wilder step ───────────────────────
    price_change = close_tomorrow - close_today
    returns      = df['Close'].diff()
    gains        = returns.where(returns > 0, 0)
    losses       = -returns.where(returns < 0, 0)

    avg_gain_today = gains.ewm(com=RSI_WINDOW - 1, adjust=False).mean().iloc[-1]
    avg_loss_today = losses.ewm(com=RSI_WINDOW - 1, adjust=False).mean().iloc[-1]

    if price_change > 0:
        avg_gain_tomorrow = (avg_gain_today * (RSI_WINDOW - 1) + price_change)  / RSI_WINDOW
        avg_loss_tomorrow = (avg_loss_today * (RSI_WINDOW - 1))                 / RSI_WINDOW
    else:
        avg_gain_tomorrow = (avg_gain_today * (RSI_WINDOW - 1))                 / RSI_WINDOW
        avg_loss_tomorrow = (avg_loss_today * (RSI_WINDOW - 1) + abs(price_change)) / RSI_WINDOW

    rsi_tomorrow = (100 if avg_loss_tomorrow == 0
                    else 100 - (100 / (1 + avg_gain_tomorrow / avg_loss_tomorrow)))

    # ── OBV simulation ────────────────────────────────────────
    obv_today = latest['OBV']
    obv_tomorrow = (obv_today + volume if close_tomorrow > close_today else
                    obv_today - volume if close_tomorrow < close_today else
                    obv_today)

    if len(df) >= 19:
        vol_20d_avg_tomorrow = (df['Volume'].iloc[-19:].sum() + volume) / 20
    else:
        vol_20d_avg_tomorrow = (df['Volume'].sum() + volume) / (len(df) + 1)

    vol_20d_avg_today   = df['Volume'].rolling(20).mean().iloc[-1]
    obv_scaled_today    = latest.get('Volume_Scaled_OBV', obv_today / vol_20d_avg_today)
    obv_scaled_tomorrow = obv_tomorrow / vol_20d_avg_tomorrow

    obv_scaled_3d_ago = (df['Volume_Scaled_OBV'].iloc[-3]
                         if len(df) >= 3 and 'Volume_Scaled_OBV' in df.columns
                         else obv_scaled_today)
    obv_3d_ago_raw    = df['OBV'].iloc[-3] if len(df) >= 3 else obv_today

    # ── Exact ADX simulation via mini-df ─────────────────────
    df_sim = df[['High', 'Low', 'Close']].tail(100).copy()

    recent_tr = (df_sim['High'] - df_sim['Low']).mean()
    _est_high = max(close_today, close_tomorrow) + (recent_tr * 0.2)
    _est_low  = min(close_today, close_tomorrow) - (recent_tr * 0.2)

    open_tmr = float(open_tomorrow) if open_tomorrow else float(close_today)
    high_tmr = float(high_tomorrow) if high_tomorrow else float(_est_high)
    low_tmr  = float(low_tomorrow)  if low_tomorrow  else float(_est_low)
    # A candle must contain its own open and close, whatever was typed.
    high_tmr = max(high_tmr, open_tmr, close_tomorrow)
    low_tmr  = min(low_tmr,  open_tmr, close_tomorrow)
    ohl_supplied = bool(high_tomorrow and low_tomorrow)

    high_tomorrow, low_tomorrow = high_tmr, low_tmr
    tomorrow_idx   = df_sim.index[-1] + pd.Timedelta(days=1)

    df_sim = pd.concat([df_sim, pd.DataFrame(
        {'High': [high_tomorrow], 'Low': [low_tomorrow], 'Close': [close_tomorrow]},
        index=[tomorrow_idx]
    )])

    adx_calc = ta.trend.ADXIndicator(
        df_sim['High'], df_sim['Low'], df_sim['Close'], window=ADX_WINDOW
    )
    df_sim['ADX']      = adx_calc.adx()
    df_sim['DI_Plus']  = adx_calc.adx_pos()
    df_sim['DI_Minus'] = adx_calc.adx_neg()

    # Reproduce the same smoothing pipeline as the main engine
    adx_filled_sim = df_sim['ADX'].ffill()
    # Same causal smoothing as the main engine — see the note there. It matters
    # doubly in the simulator: a centered filter would let the hypothetical
    # next-day bar reach backwards and alter the ADX of days already closed,
    # so the "what if tomorrow" panel would silently restate yesterday.
    df_sim['ADX_LOWESS'] = adx_filled_sim.ewm(span=ADX_EWM_SPAN, adjust=False).mean()

    def _sim_slope(y):
        if len(y) < 3 or np.isnan(y).any(): return 0.0
        x = np.arange(len(y), dtype=float)
        sl, *_ = stats.linregress(x, y)
        return float(sl)

    df_sim['ADX_Slope'] = (
        df_sim['ADX_LOWESS']
        .rolling(window=ADX_SLOPE_WINDOW, min_periods=3)
        .apply(_sim_slope, raw=True)
    )
    df_sim['ADX_Acceleration'] = df_sim['ADX_Slope'].diff(1).ewm(span=3, adjust=False).mean()
    df_sim = apply_adx_patterns(df_sim)

    tmr  = df_sim.iloc[-1]
    yest = df_sim.iloc[-2]

    adx_today    = float(yest['ADX'])
    adx_tomorrow = float(tmr['ADX'])
    adx_pattern  = str(tmr['ADX_Pattern'])

    di_plus_today  = float(yest['DI_Plus'])
    di_minus_today = float(yest['DI_Minus'])
    di_plus_tmr    = float(tmr['DI_Plus'])
    di_minus_tmr   = float(tmr['DI_Minus'])

    di_spread_today    = di_plus_today  - di_minus_today
    di_spread_tmr      = di_plus_tmr    - di_minus_tmr
    di_spread_momentum = di_spread_tmr  - di_spread_today

    di_bullish_cross_tmr = (di_plus_tmr > di_minus_tmr) and (di_plus_today <= di_minus_today)
    di_screaming_buy     = di_bullish_cross_tmr and (di_spread_momentum > DI_SCREAMING_BUY_MOMENTUM)

    # ── Thresholds ────────────────────────────────────────────
    macd_9d_low          = df['MACD'].rolling(9).min().iloc[-1]
    macd_10d_low_tomorrow= min(macd_9d_low, macd_tomorrow)
    macd_10d_high        = df['MACD'].rolling(10).max().iloc[-1]
    macd_10d_low         = df['MACD'].rolling(10).min().iloc[-1]

    rsi_p10 = (df['RSI_14'].rolling(60).quantile(0.10).iloc[-1]
               if 'RSI_14' in df.columns else RSI_OVERSOLD)
    rsi_p90 = (df['RSI_14'].rolling(60).quantile(0.90).iloc[-1]
               if 'RSI_14' in df.columns else RSI_OVERBOUGHT)

    # ── Signal evaluation ─────────────────────────────────────
    macd_stopped_falling = macd_tomorrow >= latest['MACD']
    macd_in_bottom_zone  = (macd_tomorrow >= macd_10d_low_tomorrow and
                            macd_tomorrow <= macd_10d_low_tomorrow * 0.95)
    macd_gap_narrowing   = ((macd_tomorrow - macd_signal_tomorrow) >
                            (latest['MACD'] - latest['MACD_Signal']))
    obv_rising           = obv_tomorrow > obv_3d_ago_raw

    conditions_met = sum([macd_stopped_falling, macd_in_bottom_zone,
                          macd_gap_narrowing, obv_rising])

    signals = {
        'MACD_Bottoming':    macd_tomorrow < 0 and macd_in_bottom_zone and conditions_met >= 2,
        'MACD_Bullish_Cross':(latest['MACD'] < latest['MACD_Signal']) and
                             (macd_tomorrow  > macd_signal_tomorrow),
        'MACD_Bearish_Cross':(latest['MACD'] > latest['MACD_Signal']) and
                             (macd_tomorrow  < macd_signal_tomorrow),
        'RSI_Bottoming':     rsi_tomorrow < RSI_OVERSOLD  and rsi_tomorrow <= rsi_p10,
        'RSI_Peaking':       rsi_tomorrow > RSI_OVERBOUGHT and rsi_tomorrow >= rsi_p90,
        'ADX_Pattern':       adx_pattern,
        'DI_Screaming_Buy':  di_screaming_buy,
    }

    # ── Moving averages / Bollinger / OBV momentum for the ghost ──────────
    # A rolling mean at a NEW bar is the last N-1 real closes plus tomorrow's,
    # so each of these is exact rather than an approximation.
    _closes = df['Close']

    def _ma_tmr(n: int):
        if len(_closes) < n - 1:
            return None
        return float((_closes.iloc[-(n - 1):].sum() + close_tomorrow) / n)

    ma_tomorrow = {f'MA{n}': _ma_tmr(n) for n in (5, 10, 20, 50, 60, 200)}
    ma_today = {f'MA{n}': (float(latest[f'MA{n}'])
                           if f'MA{n}' in df.columns and pd.notna(latest.get(f'MA{n}'))
                           else None)
                for n in (5, 10, 20, 50, 60, 200)}

    _a5 = 2 / (MA_SHORT + 1)
    ema5_today = float(latest['EMA5']) if 'EMA5' in df.columns else float(close_today)
    ema5_tomorrow = _a5 * close_tomorrow + (1 - _a5) * ema5_today

    _bbwin = df['Close'].iloc[-(BB_WINDOW - 1):].tolist() + [close_tomorrow]
    _bbmid = float(np.mean(_bbwin))
    _bbsd = float(np.std(_bbwin, ddof=0))
    bb_tomorrow = {'mid': _bbmid,
                   'upper': _bbmid + BB_STD * _bbsd,
                   'lower': _bbmid - BB_STD * _bbsd}

    # OBV动能 = net signed volume over 20 sessions / average daily volume,
    # both including the simulated bar.
    obv_mom_tomorrow = None
    if 'OBV' in df.columns and len(df) >= 20:
        _obv_20_ago = float(df['OBV'].iloc[-20])
        if vol_20d_avg_tomorrow:
            obv_mom_tomorrow = (obv_tomorrow - _obv_20_ago) / vol_20d_avg_tomorrow
    obv_mom_today = None
    if 'OBV' in df.columns and len(df) >= 21:
        _avg_t = float(df['Volume'].rolling(20, min_periods=1).mean().iloc[-1])
        if _avg_t:
            obv_mom_today = (obv_today - float(df['OBV'].iloc[-21])) / _avg_t

    return {
        'input_price_change_pct': price_change_pct,
        'input_volume':           volume,
        'close_today':            close_today,
        'close_tomorrow':         close_tomorrow,

        'macd_today':             latest['MACD'],
        'macd_tomorrow':          macd_tomorrow,
        'macd_signal_today':      macd_signal_today,
        'macd_signal_tomorrow':   macd_signal_tomorrow,
        'macd_hist_today':        latest['MACD_Hist'],
        'macd_hist_tomorrow':     macd_hist_tomorrow,
        'macd_gap_today':         latest['MACD'] - latest['MACD_Signal'],
        'macd_gap_tomorrow':      macd_tomorrow  - macd_signal_tomorrow,
        'macd_10d_low_tomorrow':  macd_10d_low_tomorrow,
        'macd_10d_high':          macd_10d_high,
        'macd_10d_low':           macd_10d_low,

        'rsi_today':              latest['RSI_14'],
        'rsi_tomorrow':           rsi_tomorrow,
        'rsi_p10':                rsi_p10,
        'rsi_p90':                rsi_p90,

        'adx_today':              adx_today,
        'adx_tomorrow':           adx_tomorrow,
        'adx_pattern':            adx_pattern,
        # ±DI were computed all along for the screaming-cross test but never
        # returned, so the ghost could not draw them alongside ADX.
        'di_plus_today':          di_plus_today,
        'di_plus_tomorrow':       di_plus_tmr,
        'di_minus_today':         di_minus_today,
        'di_minus_tomorrow':      di_minus_tmr,

        'open_tomorrow':          open_tmr,
        'high_tomorrow':          high_tmr,
        'low_tomorrow':           low_tmr,
        'ohl_supplied':           ohl_supplied,

        'ma_today':               ma_today,
        'ma_tomorrow':            ma_tomorrow,
        'ema5_today':             ema5_today,
        'ema5_tomorrow':          ema5_tomorrow,
        'bb_tomorrow':            bb_tomorrow,
        'bb_today': {'mid':   float(latest.get('BB_Upper', np.nan)) if False else None,
                     'upper': (float(latest['BB_Upper'])
                               if 'BB_Upper' in df.columns and pd.notna(latest.get('BB_Upper')) else None),
                     'lower': (float(latest['BB_Lower'])
                               if 'BB_Lower' in df.columns and pd.notna(latest.get('BB_Lower')) else None)},
        'obv_mom_today':          obv_mom_today,
        'obv_mom_tomorrow':       obv_mom_tomorrow,

        'obv_scaled_today':       obv_scaled_today,
        'obv_scaled_tomorrow':    obv_scaled_tomorrow,
        'obv_scaled_3d_ago':      obv_scaled_3d_ago,
        'obv_rising':             obv_rising,

        'obv_raw_today':          obv_today,
        'obv_raw_tomorrow':       obv_tomorrow,
        'obv_raw_3d_ago':         obv_3d_ago_raw,

        'volume_today':           latest['Volume'],
        'volume_tomorrow':        volume,
        'volume_10d_avg':         df['Volume'].rolling(10).mean().iloc[-1],
        'volume_20d_avg':         vol_20d_avg_today,

        'signals':                signals,
        'conditions_met':         conditions_met,
    }
