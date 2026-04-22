# strategy_engine.py

import pandas as pd
import numpy as np
from scipy import stats

def analyze_stock_personality(df: pd.DataFrame) -> dict:
    """
    Input:  df with columns: trade_date, open, high, low, close, vol
            (standard Tushare daily output, qfq adjusted)
    Output: personality dict + optimal strategy params
    """
    df = df.copy().sort_values('trade_date').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df['month'] = pd.to_datetime(df['trade_date']).dt.month

    # 1. Hurst exponent
    prices = df['close'].values
    lags = range(2, 50)
    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
    H = np.polyfit(np.log(lags), np.log(tau), 1)[0]

    # 2. Lag-1 autocorrelation
    ac1 = df['ret'].autocorr(lag=1)

    # 3. Forward returns after drops (the key signal test)
    df['fwd5']  = df['close'].pct_change(5).shift(-5)
    df['fwd10'] = df['close'].pct_change(10).shift(-10)

    drop_stats = {}
    for thresh in [-0.02, -0.03, -0.04, -0.05]:
        mask = df['ret'] <= thresh
        n = mask.sum()
        if n >= 3:
            drop_stats[thresh] = {
                'n': int(n),
                'fwd5_avg': float(df.loc[mask, 'fwd5'].mean() * 100),
                'fwd5_pos_pct': float((df.loc[mask, 'fwd5'] > 0).mean() * 100),
                'fwd10_avg': float(df.loc[mask, 'fwd10'].mean() * 100),
            }

    # 4. Monthly seasonality
    monthly_fwd10 = {}
    for m in range(1, 13):
        vals = df.loc[df['month'] == m, 'fwd10']
        if len(vals) >= 3:
            monthly_fwd10[m] = float(vals.mean() * 100)

    avoid_months = [m for m, v in monthly_fwd10.items() if v < -0.5]
    good_months  = [m for m, v in monthly_fwd10.items() if v > 1.5]

    # 5. Volatility profile
    ann_vol  = df['ret'].std() * np.sqrt(252) * 100
    skewness = float(stats.skew(df['ret'].dropna()))
    kurt     = float(stats.kurtosis(df['ret'].dropna()))

    # Determine personality label
    if H < 0.48 and ac1 < -0.03:
        personality = "mean_reverting"
        strategy_hint = "Dip Reversal is primary strategy"
    elif H > 0.52 or ac1 > 0.05:
        personality = "trending"
        strategy_hint = "MACD Crossover / Breakout preferred"
    else:
        personality = "mixed"
        strategy_hint = "Dip Reversal with seasonal filter"

    return {
        'hurst': round(H, 4),
        'ac_lag1': round(ac1, 4),
        'ann_vol': round(ann_vol, 1),
        'skewness': round(skewness, 3),
        'kurtosis': round(kurt, 3),
        'personality': personality,
        'strategy_hint': strategy_hint,
        'drop_stats': drop_stats,
        'monthly_fwd10': monthly_fwd10,
        'avoid_months': avoid_months,
        'good_months': good_months,
    }


def find_optimal_strategy(df: pd.DataFrame) -> dict:
    """
    Grid search: find best drop threshold + seasonal filter combo.
    Returns the best config by Sharpe ratio.
    """
    df = df.copy().sort_values('trade_date').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df['month'] = pd.to_datetime(df['trade_date']).dt.month

    # Compute RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss)

    COMMISSION = 0.001
    SLIPPAGE   = 0.002
    n_years    = (pd.to_datetime(df['trade_date'].iloc[-1]) -
                  pd.to_datetime(df['trade_date'].iloc[0])).days / 365.25

    # Detect avoid months from data
    df['fwd10'] = df['close'].pct_change(10).shift(-10)
    monthly_fwd = {m: df.loc[df['month']==m, 'fwd10'].mean()*100
                   for m in range(1, 13)
                   if len(df[df['month']==m]) >= 3}
    data_avoid = [m for m, v in monthly_fwd.items() if v < -0.5]

    best = {'sharpe': -99}

    for drop_thresh in [-0.025, -0.03, -0.035, -0.04]:
        for avoid in [[], data_avoid, [7, 11, 12], [7, 8, 11, 12]]:
            equity = 1.0; trades = []
            in_pos = False; ep = 0; ei = 0

            for i in range(1, len(df)):
                row = df.iloc[i]
                price = row['close']

                if not in_pos:
                    rsi_ok = not pd.isna(row['rsi']) and row['rsi'] > 20
                    season_ok = row['month'] not in avoid
                    if row['ret'] <= drop_thresh and rsi_ok and season_ok:
                        in_pos = True
                        ep = price * (1 + SLIPPAGE)
                        ei = i
                        equity *= (1 - COMMISSION)
                else:
                    days = i - ei
                    pnl = (price - ep) / ep
                    exit_now = (pnl >= 0.08 or pnl <= -0.05 or days >= 10 or
                                (not pd.isna(row['rsi']) and row['rsi'] > 68))
                    if exit_now:
                        net_pnl = (price*(1-SLIPPAGE) - ep) / ep
                        equity *= (1 + net_pnl) * (1 - COMMISSION)
                        trades.append({'pnl': net_pnl*100, 'days': days})
                        in_pos = False

            if len(trades) < 5:
                continue

            pnls = [t['pnl'] for t in trades]
            eq_r = np.diff([1.0] + list(np.cumprod(1 + np.array(pnls)/100)))
            eq_base = np.array([1.0] + list(np.cumprod(1 + np.array(pnls)/100)))[:-1]
            daily_r = eq_r / (eq_base + 1e-9)
            sharpe = np.mean(daily_r) / (np.std(daily_r) + 1e-9) * np.sqrt(252)
            cagr = (np.prod(1 + np.array(pnls)/100) ** (1/n_years) - 1) * 100
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100

            if sharpe > best['sharpe']:
                best = {
                    'sharpe': round(sharpe, 2),
                    'cagr': round(cagr, 1),
                    'win_rate': round(win_rate, 1),
                    'n_trades': len(trades),
                    'drop_thresh': drop_thresh,
                    'avoid_months': avoid,
                    'avg_win': round(np.mean([p for p in pnls if p > 0] or [0]), 2),
                    'avg_loss': round(np.mean([p for p in pnls if p <= 0] or [0]), 2),
                }

    return best