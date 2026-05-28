"""
Shared utilities for Sector Analysis pages
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import data_manager
from analysis_engine import detect_market_regime

_NUMERIC_COLS = ['Open', 'High', 'Low', 'Close', 'Volume_Metric']

def _coerce_numerics(df):
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data(ttl=600)
def load_v2_data():
    """Load sector PPI data from DB as long-format DataFrame."""
    try:
        ppi_data = data_manager.load_ppi_data_from_db()
        if not ppi_data:
            return None, None, None, "No PPI data found in database — run main.py first"

        rows = []
        for sector, df in ppi_data.items():
            for date, row in df.iterrows():
                rows.append({
                    'Date': date,
                    'Sector': sector,
                    'Open': row.get('Open', row.get('Close')),
                    'High': row.get('High', row.get('Close')),
                    'Low': row.get('Low', row.get('Close')),
                    'Close': row['Close'],
                    'Volume_Metric': row.get('Volume_Metric', 0.0),
                })

        if not rows:
            return None, None, None, "PPI tables found but contained no rows"

        v2hist = pd.DataFrame(rows)
        v2hist['Date'] = pd.to_datetime(v2hist['Date'])
        v2hist = _coerce_numerics(v2hist)
        v2hist = v2hist.sort_values('Date')

        latest_date = v2hist['Date'].max()
        v2latest = v2hist[v2hist['Date'] == latest_date].copy()

        return v2latest, v2hist, latest_date.strftime('%Y-%m-%d'), None

    except Exception as e:
        return None, None, None, str(e)

@st.cache_data(ttl=600)
def load_csi300_with_regime(freq_cn: str):
    """Fetch CSI300 with enough history to compute regimes, then return full df."""
    if freq_cn == "日线":
        rawdf = data_manager.get_index_data_live('000300.SH', lookback_days=365, freq='daily')
        if rawdf is None or rawdf.empty:
            return rawdf
        return detect_market_regime(rawdf, freq='daily')
    
    rawdf = data_manager.get_index_data_live('000300.SH', lookback_days=1825, freq='weekly')
    if rawdf is None or rawdf.empty:
        return rawdf
    return detect_market_regime(rawdf, freq='weekly')

def create_sector_chart(chart_data):
    """Create 3-panel sector chart."""
    date_strings = chart_data['Date'].dt.strftime('%Y-%m-%d')
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Price (PPI)", "Volume Z-Score", "Bull Probability"),
        row_heights=[0.5, 0.2, 0.3]
    )
    
    # Price
    fig.add_trace(go.Candlestick(
        x=date_strings,
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name='Price',
        increasing=dict(line=dict(color='#ef4444')),
        decreasing=dict(line=dict(color='#22c55e'))
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=date_strings,
        y=chart_data['Volume_Metric'],
        name='Volume',
        marker=dict(color='rgba(107, 114, 128, 0.3)')
    ), row=2, col=1)
    
    # Score
    fig.add_trace(go.Scatter(
        x=date_strings,
        y=chart_data['TOTAL_SCORE'],
        name='Score',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ), row=3, col=1)
    
    # Thresholds
    buy_thresh = chart_data['ThresholdBuy'].iloc[-1] if 'ThresholdBuy' in chart_data.columns else 0.75
    sell_thresh = chart_data['ThresholdSell'].iloc[-1] if 'ThresholdSell' in chart_data.columns else 0.25
    fig.add_hline(y=buy_thresh, line_dash='dash', line_color='#15803d', row=3, col=1)
    fig.add_hline(y=sell_thresh, line_dash='dash', line_color='#b91c1c', row=3, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        template='plotly_white',
        xaxis3_title='Date',
        yaxis1_title='PPI',
        yaxis2_title='Vol Z',
        yaxis3_title='Bull Prob',
        yaxis3_range=[-0.1, 1.1],
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_rolling_correlation_chart(ret_panel, reference_sector, compare_sectors, window=20):
    """
    Show rolling correlations between one reference sector and multiple others.
    
    Args:
        ret_panel: DataFrame with Date index and sector returns as columns
        reference_sector: The base sector to compare against (e.g., "Technology")
        compare_sectors: List of sectors to compare (e.g., ["Healthcare", "Finance", "Energy"])
        window: Rolling window size in days
    """
    fig = go.Figure()
    
    for sector in compare_sectors:
        if sector in ret_panel.columns and reference_sector in ret_panel.columns:
            # Calculate rolling correlation
            rolling_corr = ret_panel[reference_sector].rolling(window=window).corr(ret_panel[sector])
            
            fig.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                name=f"{reference_sector} ↔ {sector}",
                mode='lines',
                line=dict(width=2)
            ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(y=0.7, line_dash='dot', line_color='green', opacity=0.3)
    fig.add_hline(y=-0.7, line_dash='dot', line_color='red', opacity=0.3)
    
    fig.update_layout(
        title=f'Rolling {window}-Day Correlation: {reference_sector} vs Other Sectors',
        xaxis_title='Date',
        yaxis_title='Correlation Coefficient',
        yaxis_range=[-1, 1],
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def pivot_sector_series(hist_df, value_col):
    """Pivot to Date x Sector."""
    df = hist_df[['Date', 'Sector', value_col]].dropna().copy()
    wide = df.pivot_table(index='Date', columns='Sector', values=value_col, aggfunc='last')
    return wide.sort_index()


def build_sector_panels(hist_df, csi300_df=None, market_sector='MARKET_PROXY'):
    """
    Build close, ret, vol, exret panels.
    
    If csi300_df provided: exret = sector_return - csi300_daily_return  ← CORRECT
    If not:                exret = sector_return - MARKET_PROXY return  ← fallback only
    """
    close_panel = pivot_sector_series(hist_df, 'Close')
    vol_panel   = pivot_sector_series(hist_df, 'Volume_Metric')
    ret_panel   = close_panel.pct_change()

    exret_panel = ret_panel.copy()

    if csi300_df is not None and 'Close' in csi300_df.columns:
        # Use real CSI300 daily return as market benchmark
        csi300_ret = csi300_df['Close'].pct_change()
        # Align dates — only dates present in both
        common_dates = exret_panel.index.intersection(csi300_ret.index)
        exret_panel = exret_panel.loc[common_dates]
        csi300_ret_aligned = csi300_ret.loc[common_dates]
        exret_panel = exret_panel.sub(csi300_ret_aligned, axis=0)
    elif market_sector in ret_panel.columns:
        # Fallback: subtract MARKET_PROXY PPI return
        exret_panel = ret_panel.sub(ret_panel[market_sector], axis=0)
        exret_panel[market_sector] = 0.0

    return close_panel, ret_panel, vol_panel, exret_panel


def compute_market_gate(ret_panel, exret_panel, market_sector="MARKET_PROXY", lookback=60, mkt_down_thresh=-0.01):
    """Compute market gate metrics."""
    ret_lb = ret_panel.tail(lookback)
    ex_lb = exret_panel.tail(lookback)
    
    latest_dt = ret_lb.index.max()
    if latest_dt is None:
        return None
    
    # Market return: AVERAGE over the lookback period (or last day if you prefer)
    mkt_ret = ret_lb[market_sector].iloc[-1] if market_sector in ret_lb.columns else ret_lb.mean(axis=1).iloc[-1]
    
    # Dispersion: AVERAGE standard deviation across the lookback period
    dispersion = ex_lb.std(axis=1).mean()  # std across sectors for each day, then average
    
    # Latest day metrics (these should stay as last day)
    ret_last = ret_lb.iloc[-1].dropna()
    breadth_down = (ret_last < 0).sum() / len(ret_last) if len(ret_last) > 0 else 0
    
    regime = "DOWN" if mkt_ret < mkt_down_thresh else "UP" if mkt_ret > 0.01 else "FLAT"
    confidence = "HIGH" if dispersion > 0.015 else "LOW" if dispersion < 0.005 else "MODERATE"
    
    return {
        'market_return': mkt_ret,
        'dispersion': dispersion,
        'breadth_down': breadth_down,
        'regime': regime,
        'confidence': confidence
    }


def compute_market_gate_with_context(ret_panel, exret_panel, csi300_df=None,
                                      lookback=20, history_window=252):
    """
    Compute market gate metrics with historical context.
    Optimized for A-shares: 20-day period vs 252-day (1 year) history.
    
    Args:
        lookback: Current period to evaluate (default 20 days = 1 month)
        history_window: Historical comparison period (default 252 days = 1 year)
    """
    # Get current period
    ret_lb = ret_panel.tail(lookback)
    ex_lb = exret_panel.tail(lookback)
    
    # Get historical period for comparison (1 year)
    ex_hist = exret_panel.tail(history_window)
    
    latest_dt = ret_lb.index.max()
    if latest_dt is None or len(ex_lb) < lookback:
        return None
    
    # === CURRENT 20-DAY PERIOD METRICS ===
    # mkt_ret = ret_lb[market_sector].mean() if market_sector in ret_lb.columns else ret_lb.mean(axis=1).mean()
    # Use real CSI300 return for market_return metric
    if csi300_df is not None and 'Close' in csi300_df.columns:
        csi300_ret = csi300_df['Close'].pct_change().tail(lookback)
        mkt_ret = csi300_ret.mean()
    else:
        mkt_ret = ret_panel.tail(lookback).mean(axis=1).mean()  # fallback


    # Current 20-day average dispersion
    dispersion_current = ex_lb.std(axis=1).mean()
    
    # Current 20-day average breadth
    breadth_down_series = (ret_lb < 0).sum(axis=1) / ret_lb.count(axis=1)
    breadth_down = breadth_down_series.mean()
    
    # === HISTORICAL CONTEXT (1 YEAR) ===
    # Calculate dispersion for every day in the past year
    dispersion_history = ex_hist.std(axis=1)
    
    # Calculate rolling 20-day averages for the entire year
    dispersion_rolling = dispersion_history.rolling(lookback).mean().dropna()
    
    if len(dispersion_rolling) < 10:
        return None
    
    # Percentile ranking: Where does current 20-day period rank vs past year?
    dispersion_percentile = (dispersion_rolling < dispersion_current).sum() / len(dispersion_rolling)
    
    # === REGIME CLASSIFICATION (A-shares specific thresholds) ===
    if dispersion_percentile >= 0.85:
        regime_state = "EXTREME_ROTATION"
        regime_label = "🔥 极端轮动 (Extreme Rotation)"
        regime_color = "success"
    elif dispersion_percentile >= 0.70:
        regime_state = "STRONG_ROTATION"
        regime_label = "✅ 强势轮动 (Strong Rotation)"
        regime_color = "success"
    elif dispersion_percentile >= 0.50:
        regime_state = "MODERATE_ROTATION"
        regime_label = "⚪ 温和轮动 (Moderate Rotation)"
        regime_color = "info"
    elif dispersion_percentile >= 0.30:
        regime_state = "LOW_ROTATION"
        regime_label = "⚠️ 弱势轮动 (Weak Rotation)"
        regime_color = "warning"
    else:
        regime_state = "HIGH_CORRELATION"
        regime_label = "❌ 板块共振 (High Correlation)"
        regime_color = "error"
    
    # Traditional confidence
    confidence = "HIGH" if dispersion_current > 0.015 else "LOW" if dispersion_current < 0.005 else "MODERATE"
    
    # Recent trend (last 10 days vs prior 10 days)
    if len(dispersion_rolling) >= 20:
        recent_10d = dispersion_rolling.tail(10).mean()
        prior_10d = dispersion_rolling.tail(20).head(10).mean()
        trend_change = (recent_10d - prior_10d) / prior_10d if prior_10d != 0 else 0
        
        if trend_change > 0.10:  # 10% increase
            trend_label = "📈 加速轮动 (Accelerating)"
        elif trend_change < -0.10:  # 10% decrease
            trend_label = "📉 收敛中 (Converging)"
        else:
            trend_label = "➡️ 稳定 (Stable)"
    else:
        trend_label = "➡️ 数据不足"
    
    # Calculate volatility of dispersion (regime stability)
    dispersion_volatility = dispersion_rolling.tail(60).std()
    regime_stability = "稳定" if dispersion_volatility < dispersion_rolling.mean() * 0.3 else "波动"
    
    return {
        'market_return': mkt_ret,
        'dispersion': dispersion_current,
        'dispersion_percentile': dispersion_percentile,
        'breadth_down': breadth_down,
        'confidence': confidence,
        'regime_state': regime_state,
        'regime_label': regime_label,
        'regime_color': regime_color,
        'trend_label': trend_label,
        'regime_stability': regime_stability,
        'history_p25': dispersion_rolling.quantile(0.25),
        'history_p50': dispersion_rolling.quantile(0.50),
        'history_p75': dispersion_rolling.quantile(0.75),
        'history_p85': dispersion_rolling.quantile(0.85),
        'days_in_current_regime': len(dispersion_rolling[dispersion_rolling >= dispersion_rolling.quantile(dispersion_percentile)].tail(60))
    }



def compute_transition_matrix(exret_panel, lookback=60, top_k=3):
    df = exret_panel.tail(lookback).copy()

    if len(df) < 10:
        return None, None

    def get_top_k(row):
        valid = row.dropna()
        if len(valid) < top_k:
            return set()
        return set(valid.nlargest(top_k).index.tolist())

    top_k_sets = df.apply(get_top_k, axis=1)
    all_sectors = df.columns.tolist()
    counts = pd.DataFrame(0, index=all_sectors, columns=all_sectors)

    for i in range(len(top_k_sets) - 1):
        today    = top_k_sets.iloc[i]
        tomorrow = top_k_sets.iloc[i + 1]
        if not today or not tomorrow:
            continue
        for leader in today:
            for follower in tomorrow:
                counts.loc[leader, follower] += 1

    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return probs, counts


def predict_tomorrow(probs, counts, today_leaders, top_n=8, min_samples=10):
    """
    Predict tomorrow's followers.
    - min_samples raised to 10 (was 3) for statistical reliability
    - Leaders weighted by their excess return magnitude, not equal weight
    """
    if probs is None or today_leaders.empty:
        return pd.DataFrame()

    follower_scores = pd.Series(0.0, index=probs.columns)
    total_weight    = 0.0

    for leader in today_leaders.index:
        if leader not in probs.index:
            continue
        if counts.loc[leader].sum() < min_samples:
            continue

        # Weight by absolute excess return magnitude
        weight = abs(today_leaders[leader])
        if weight == 0:
            weight = 1e-6  # avoid zero weight if return is exactly 0

        follower_scores += probs.loc[leader] * weight
        total_weight    += weight

    if total_weight == 0:
        return pd.DataFrame()

    follower_scores /= total_weight

    result = follower_scores.sort_values(ascending=False).head(top_n)
    return pd.DataFrame({
        'Sector':           result.index,
        'P(NextDay in Top-K)': result.values,
        'Min Samples Used': min_samples
    }).reset_index(drop=True)


def get_today_topk(exret_panel, top_k=3):
    """Get today's top leaders."""
    df = exret_panel.copy()
    
    latest_dt = df.index.max()
    latest_row = df.loc[latest_dt].dropna().sort_values(ascending=False)
    return latest_dt, latest_row.head(top_k)

# def predict_tomorrow(probs, counts, today_leaders, top_n=8, min_samples=3):
#     """Predict tomorrow's followers."""
#     if probs is None or today_leaders.empty:
#         return pd.DataFrame()
    
#     follower_scores = pd.Series(0.0, index=probs.columns)
    
#     for leader in today_leaders.index:
#         if leader in probs.index:
#             if counts.loc[leader].sum() >= min_samples:
#                 follower_scores += probs.loc[leader]
    
#     if follower_scores.sum() > 0:
#         follower_scores /= len(today_leaders)
    
#     result = follower_scores.sort_values(ascending=False).head(top_n)
#     return pd.DataFrame({
#         'Sector': result.index,
#         'P(NextDay in Top-K)': result.values
#     }).reset_index(drop=True)

def build_nextday_predictions(exret_panel, vol_panel, z_window=20, lookback=60):
    """Build state-based next-day predictions."""
    df_ex = exret_panel.tail(lookback).copy()
    df_vol = vol_panel.tail(lookback).copy()
    
    
    results = []
    
    for sector in df_ex.columns:
        ex = df_ex[sector].dropna()
        vol = df_vol[sector].dropna()
        
        common_idx = ex.index.intersection(vol.index)
        if len(common_idx) < z_window + 10:
            continue
        
        ex = ex.loc[common_idx]
        vol = vol.loc[common_idx]
        
        ex_z = (ex - ex.rolling(z_window).mean()) / ex.rolling(z_window).std()
        ex_z = ex_z.dropna()
        vol = vol.loc[ex_z.index]
        
        ex_bins = pd.cut(ex_z, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
        vol_bins = pd.cut(vol, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
        
        states = ex_bins.astype(str) + '|' + vol_bins.astype(str)
        # ✅ FIX: Remove any states that contain 'nan'
        valid_states = states[~states.str.contains('nan', na=False)]
        if len(valid_states) < 10:
            continue

        state_stats = {}
        for i in range(len(states) - 1):
            state = states.iloc[i]
            next_ret = ex.iloc[i + 1]
            
            if pd.isna(state):
                continue
            
            if state not in state_stats:
                state_stats[state] = {'wins': 0, 'total': 0, 'sum_ret': 0}
            
            state_stats[state]['total'] += 1
            state_stats[state]['sum_ret'] += next_ret
            if next_ret > 0:
                state_stats[state]['wins'] += 1
        
        if len(states) > 0:
            today_state = states.iloc[-1]
            
            if pd.notna(today_state) and today_state in state_stats:
                stats = state_stats[today_state]
                win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
                avg_ret = stats['sum_ret'] / stats['total'] if stats['total'] > 0 else 0
                
                results.append({
                    'Sector': sector,
                    'Current_State': today_state,
                    'P(NextDay Outperform)': win_rate,
                    'E[NextDay ExcessRet]': avg_ret,
                    'State_Samples': stats['total']
                })
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    return df.sort_values('P(NextDay Outperform)', ascending=False)



def build_state_stats(ex_series, vol_series, z_window=20):
    """Build state stats for single sector."""
    common_idx = ex_series.index.intersection(vol_series.index)
    if len(common_idx) < z_window + 10:
        return None, None
    
    ex = ex_series.loc[common_idx]
    vol = vol_series.loc[common_idx]
    
    ex_z = (ex - ex.rolling(z_window).mean()) / ex.rolling(z_window).std()
    ex_z = ex_z.dropna()
    vol = vol.loc[ex_z.index]
    ex = ex.loc[ex_z.index]
    
    
    ex_bins = pd.cut(ex_z, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
    vol_bins = pd.cut(vol, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
    
    states = ex_bins.astype(str) + '|' + vol_bins.astype(str)
    valid_states = states[~states.str.contains('nan', na=False)]
    
    if len(valid_states) < 10:
        return None, None
    
    state_stats = {}
    for i in range(len(states) - 1):
        state = states.iloc[i]
        next_ret = ex.iloc[i + 1]
        
        if pd.isna(state):
            continue
        
        if state not in state_stats:
            state_stats[state] = {'wins': 0, 'total': 0, 'sum_ret': 0}
        
        state_stats[state]['total'] += 1
        state_stats[state]['sum_ret'] += next_ret
        if next_ret > 0:
            state_stats[state]['wins'] += 1
    
    stats_list = []
    for state, data in state_stats.items():
        stats_list.append({
            'state': state,
            'count': data['total'],
            'win_rate': data['wins'] / data['total'] if data['total'] > 0 else 0,
            'avg_excess_ret': data['sum_ret'] / data['total'] if data['total'] > 0 else 0
        })
    
    df = pd.DataFrame(stats_list).set_index('state').sort_values('count', ascending=False)
    latest_state = states.iloc[-1] if len(states) > 0 else None
    
    return df, latest_state


def make_heatmap(probs, title="Transition Probabilities"):
    """Create transition heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=probs.values, x=probs.columns.tolist(), y=probs.index.tolist(),
        colorscale='Blues', text=probs.values.round(2), texttemplate='%{text}',
        textfont=dict(size=16), colorbar=dict(title="Probability")
    ))
    fig.update_layout(
        title=title, xaxis_title="Tomorrow", yaxis_title="Today",
        height=600, template='plotly_white',
        # ← ADD THESE LINES
        xaxis=dict(tickfont=dict(size=13)),  # x-axis labels
        yaxis=dict(tickfont=dict(size=13)),  # y-axis labels
        font=dict(size=14)  # title and other text
    )
    return fig

# Add all other shared functions from your original file:

# ========================================== 
# NEW: PERFORMANCE & ROTATION FUNCTIONS
# ========================================== 

def create_performance_comparison_chart(hist_df, lookback_days=60, selected_sectors=None):
    """Create normalized performance comparison chart (base 100)."""
    # Get data for the last N days based on actual dates
    latest_date = hist_df['Date'].max()
    start_date = latest_date - pd.Timedelta(days=lookback_days)
    recent = hist_df[hist_df['Date'] >= start_date].copy()
    
    # DEBUG: Print what we have
    print(f"Date range: {recent['Date'].min()} to {recent['Date'].max()}")
    print(f"Available sectors: {recent['Sector'].unique().tolist()}")
    print(f"Selected sectors: {selected_sectors}")
    
    # Pivot to get Close prices by sector
    pivot = recent.pivot_table(index='Date', columns='Sector', values='Close')
    
    # Filter selected sectors if provided
    if selected_sectors and len(selected_sectors) > 0:
        # Only keep sectors that exist in both selected list AND pivot columns
        available_sectors = [s for s in selected_sectors if s in pivot.columns]
        print(f"Filtered to: {available_sectors}")
        
        if available_sectors:
            pivot = pivot[available_sectors]
        else:
            # If no match, show all
            st.warning(f"Selected sectors not found in data. Showing all sectors.")
    
    # Remove MARKET_PROXY if it exists and wasn't explicitly selected
    if 'MARKET_PROXY' in pivot.columns:
        if not selected_sectors or 'MARKET_PROXY' not in selected_sectors:
            pivot = pivot.drop(columns=['MARKET_PROXY'])
    
    # Normalize to base 100
    first_valid_row = pivot.dropna(how='all').iloc[0]
    normalized = (pivot / first_valid_row * 100)
    
    # Create chart
    fig = go.Figure()
    
    # Add trace for each sector
    for sector in normalized.columns:
        sector_data = normalized[sector].dropna()
        
        fig.add_trace(go.Scatter(
            x=sector_data.index,
            y=sector_data.values,
            name=sector,
            mode='lines',
            line=dict(width=2),
            hovertemplate=f'<b>{sector}</b><br>Date: %{{x|%Y-%m-%d}}<br>Index: %{{y:.1f}}<extra></extra>'
        ))
    
    # Add horizontal line at 100
    fig.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.5)
    
    fig.update_layout(
        title=f'Sector Performance Comparison (Base 100) - Last {lookback_days} Days',
        xaxis_title='Date',
        yaxis_title='Normalized Index (Base 100)',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def compute_rrg_series(close_panel, csi300_df, sectors=None,
                       rs_window=20, mom_window=5):
    """
    Compute JdK-style Relative Rotation Graph series for each sector vs CSI300.

    For each sector:
        rs_raw[t]   = sector_close[t] / csi300_close[t]
        rs_ratio[t] = 100 + (rs_raw[t] - mean_W(rs_raw)) / std_W(rs_raw)
        rs_mom[t]   = 100 + (rs_ratio[t] - rs_ratio[t-mom_window])

    Both axes are centred at 100 so the cross (100,100) is the benchmark.

    Args:
        close_panel: Date x Sector close-price panel (trading days, sorted).
        csi300_df:   DataFrame with 'Close' column indexed by trading date.
        sectors:     subset of sectors to compute (defaults to all non-proxy).
        rs_window:   window for the RS z-score normalisation (trading days).
        mom_window:  lag for the momentum (rate-of-change) of RS-Ratio.

    Returns:
        rs_ratio_panel, rs_momentum_panel  (both Date x Sector, centred at 100)
        Returns (None, None) if CSI300 is missing.
    """
    if csi300_df is None or 'Close' not in csi300_df.columns:
        return None, None

    bench = csi300_df['Close'].copy()
    common = close_panel.index.intersection(bench.index)
    if len(common) < rs_window + mom_window + 5:
        return None, None
    close = close_panel.loc[common].sort_index()
    bench = bench.loc[common].sort_index()

    if sectors is None:
        sectors = [c for c in close.columns if c != 'MARKET_PROXY']

    min_periods = max(5, rs_window // 2)
    rs_ratio_dict = {}
    rs_mom_dict = {}

    for sector in sectors:
        if sector not in close.columns:
            continue
        sec = close[sector].astype(float)
        b = bench.reindex(sec.index).ffill()
        rs_raw = sec / b
        rs_raw = rs_raw.replace([np.inf, -np.inf], np.nan)

        roll_mean = rs_raw.rolling(rs_window, min_periods=min_periods).mean()
        roll_std = rs_raw.rolling(rs_window, min_periods=min_periods).std()
        rs_ratio = 100.0 + (rs_raw - roll_mean) / roll_std.replace(0, np.nan)

        rs_mom = 100.0 + (rs_ratio - rs_ratio.shift(mom_window))

        rs_ratio_dict[sector] = rs_ratio
        rs_mom_dict[sector] = rs_mom

    if not rs_ratio_dict:
        return None, None

    return pd.DataFrame(rs_ratio_dict), pd.DataFrame(rs_mom_dict)


def _quadrant_for(x, y):
    if pd.isna(x) or pd.isna(y):
        return None
    if x >= 100 and y >= 100:
        return 'Leading'
    if x < 100 and y >= 100:
        return 'Improving'
    if x < 100 and y < 100:
        return 'Lagging'
    return 'Weakening'


def create_rrg_chart(rs_ratio_panel, rs_mom_panel, sectors=None, tail_length=10):
    """
    Build an RRG scatter with a trajectory tail per sector.

    Returns (fig, rotation_df) where rotation_df has one row per sector with
    latest RS-Ratio / RS-Momentum, quadrant, and tail deltas (direction of travel).
    """
    if rs_ratio_panel is None or rs_mom_panel is None:
        return None, None

    if sectors is None:
        sectors = [c for c in rs_ratio_panel.columns if c in rs_mom_panel.columns]

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
               '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']

    fig = go.Figure()
    rows = []

    for i, sector in enumerate(sectors):
        if sector not in rs_ratio_panel.columns or sector not in rs_mom_panel.columns:
            continue
        x_full = rs_ratio_panel[sector]
        y_full = rs_mom_panel[sector]
        df_xy = pd.concat([x_full, y_full], axis=1, keys=['x', 'y']).dropna()
        if len(df_xy) < 2:
            continue
        tail = df_xy.tail(tail_length)
        if len(tail) < 2:
            continue

        color = palette[i % len(palette)]
        n = len(tail)
        # marker sizes ramp up so the latest point is the largest
        sizes = [4 + 10 * (j / (n - 1)) for j in range(n)]
        opacities = [0.25 + 0.75 * (j / (n - 1)) for j in range(n)]

        # Trajectory line + fading markers
        fig.add_trace(go.Scatter(
            x=tail['x'].values,
            y=tail['y'].values,
            mode='lines+markers',
            name=sector,
            legendgroup=sector,
            line=dict(color=color, width=1.5),
            marker=dict(size=sizes, color=color, opacity=opacities,
                        line=dict(width=0)),
            hovertemplate=(f'<b>{sector}</b><br>'
                           'Date: %{customdata}<br>'
                           'RS-Ratio: %{x:.2f}<br>'
                           'RS-Momentum: %{y:.2f}<extra></extra>'),
            customdata=[d.strftime('%Y-%m-%d') for d in tail.index],
            showlegend=True,
        ))

        # Labelled latest point
        x_last, y_last = tail['x'].iloc[-1], tail['y'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[x_last], y=[y_last],
            mode='markers+text',
            name=sector,
            legendgroup=sector,
            text=[sector],
            textposition='top center',
            textfont=dict(size=11, color=color),
            marker=dict(size=14, color=color,
                        line=dict(width=2, color='white')),
            showlegend=False,
            hoverinfo='skip',
        ))

        rows.append({
            'Sector': sector,
            'RS_Ratio': x_last,
            'RS_Momentum': y_last,
            'Quadrant': _quadrant_for(x_last, y_last),
            'ΔRS_Ratio_tail': x_last - tail['x'].iloc[0],
            'ΔRS_Momentum_tail': y_last - tail['y'].iloc[0],
            'Tail_Days': n,
        })

    fig.add_vline(x=100, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(y=100, line_dash='dash', line_color='gray', opacity=0.5)

    fig.update_layout(
        title=f'Relative Rotation Graph vs CSI300 — tail = last {tail_length} trading days',
        xaxis_title='RS-Ratio (level, 100 = in line with CSI300)',
        yaxis_title='RS-Momentum (100 = flat, >100 rising)',
        height=720,
        template='plotly_white',
        hovermode='closest',
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
        annotations=[
            dict(x=0.99, y=0.99, xref='paper', yref='paper',
                 text='🟢 LEADING<br>(buy / hold)', showarrow=False,
                 font=dict(size=12, color='#15803d'), opacity=0.45,
                 xanchor='right', yanchor='top'),
            dict(x=0.01, y=0.99, xref='paper', yref='paper',
                 text='🔵 IMPROVING<br>(early buy)', showarrow=False,
                 font=dict(size=12, color='#3b82f6'), opacity=0.45,
                 xanchor='left', yanchor='top'),
            dict(x=0.99, y=0.01, xref='paper', yref='paper',
                 text='🟡 WEAKENING<br>(take profits)', showarrow=False,
                 font=dict(size=12, color='#f59e0b'), opacity=0.45,
                 xanchor='right', yanchor='bottom'),
            dict(x=0.01, y=0.01, xref='paper', yref='paper',
                 text='🔴 LAGGING<br>(avoid)', showarrow=False,
                 font=dict(size=12, color='#dc2626'), opacity=0.45,
                 xanchor='left', yanchor='bottom'),
        ],
    )

    rotation_df = pd.DataFrame(rows)
    return fig, rotation_df


def compute_lead_lag_matrix(exret_panel, sectors, max_lag=5, lookback=120):
    """
    For each ordered pair (A, B), find the lag in [-max_lag, +max_lag]
    that maximises |corr(A_t, B_{t-lag})| on the last `lookback` trading days.

    Convention:
        lag > 0  => B leads A by `lag` trading days
        lag < 0  => A leads B
        lag == 0 => contemporaneous

    Returns (best_lag, best_corr) as DataFrames indexed [A] x columns [B].
    """
    sectors = [s for s in sectors if s in exret_panel.columns]
    if len(sectors) < 2:
        return None, None

    df = exret_panel[sectors].tail(lookback).dropna(how='all')
    if len(df) < max_lag * 4:
        return None, None

    best_lag = pd.DataFrame(0, index=sectors, columns=sectors, dtype=int)
    best_corr = pd.DataFrame(np.nan, index=sectors, columns=sectors, dtype=float)

    for a in sectors:
        for b in sectors:
            if a == b:
                best_corr.loc[a, b] = 1.0
                continue
            best_c = 0.0
            best_l = 0
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    c = df[a].corr(df[b].shift(lag))
                else:
                    c = df[a].shift(-lag).corr(df[b])
                if pd.notna(c) and abs(c) > abs(best_c):
                    best_c = c
                    best_l = lag
            best_lag.loc[a, b] = best_l
            best_corr.loc[a, b] = best_c

    return best_lag, best_corr


def create_sector_rotation_map(hist_df, lookback_short=5, lookback_long=20):
    """[DEPRECATED] Legacy quadrant chart. Use compute_rrg_series + create_rrg_chart."""
    # Get latest data
    latest_date = hist_df['Date'].max()
    recent = hist_df[hist_df['Date'] >= latest_date - pd.Timedelta(days=lookback_long*2)]
    
    # Pivot Close prices
    pivot = recent.pivot_table(index='Date', columns='Sector', values='Close').sort_index()
    
    if len(pivot) < lookback_long:
        return None
    
    sectors_data = []
    
    for sector in pivot.columns:
        if sector == 'MARKET_PROXY':
            continue
            
        prices = pivot[sector].dropna()
        if len(prices) < lookback_long:
            continue
        
        # Calculate relative strength (% change over long period)
        long_change = ((prices.iloc[-1] - prices.iloc[-lookback_long]) / prices.iloc[-lookback_long]) * 100
        
        # Calculate momentum (recent vs prior period)
        recent_avg = prices.iloc[-lookback_short:].mean()
        prior_avg = prices.iloc[-lookback_long:-lookback_short].mean()
        momentum = ((recent_avg - prior_avg) / prior_avg) * 100
        
        sectors_data.append({
            'Sector': sector,
            'Relative_Strength': long_change,
            'Momentum': momentum,
            'Current_Price': prices.iloc[-1]
        })
    
    if not sectors_data:
        return None
    
    df = pd.DataFrame(sectors_data)
    
    # Calculate average for quadrant lines
    avg_strength = df['Relative_Strength'].mean()
    avg_momentum = df['Momentum'].mean()
    
    # Assign quadrants
    def assign_quadrant(row):
        if row['Relative_Strength'] >= avg_strength and row['Momentum'] >= avg_momentum:
            return '🟢 Leading'
        elif row['Relative_Strength'] >= avg_strength and row['Momentum'] < avg_momentum:
            return '🟡 Weakening'
        elif row['Relative_Strength'] < avg_strength and row['Momentum'] < avg_momentum:
            return '🔴 Lagging'
        else:
            return '🔵 Improving'
    
    df['Quadrant'] = df.apply(assign_quadrant, axis=1)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Color mapping
    color_map = {
        '🟢 Leading': '#15803d',
        '🟡 Weakening': '#f59e0b',
        '🔴 Lagging': '#dc2626',
        '🔵 Improving': '#3b82f6'
    }
    
    for quadrant in df['Quadrant'].unique():
        quad_data = df[df['Quadrant'] == quadrant]
        fig.add_trace(go.Scatter(
            x=quad_data['Relative_Strength'],
            y=quad_data['Momentum'],
            mode='markers+text',
            name=quadrant,
            text=quad_data['Sector'],
            textposition='top center',
            marker=dict(
                size=15,
                color=color_map.get(quadrant, 'gray'),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         f'Relative Strength: %{{x:.1f}}%<br>' +
                         f'Momentum: %{{y:.1f}}%<br>' +
                         '<extra></extra>'
        ))
    
    # Add quadrant lines
    fig.add_vline(x=avg_strength, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(y=avg_momentum, line_dash='dash', line_color='gray', opacity=0.5)
    
    # Add quadrant labels
    max_x = df['Relative_Strength'].max()
    min_x = df['Relative_Strength'].min()
    max_y = df['Momentum'].max()
    min_y = df['Momentum'].min()
    
    annotations = [
        dict(x=max_x*0.8, y=max_y*0.8, text="🟢 LEADING<br>(Buy/Hold)", showarrow=False, 
             font=dict(size=12, color='#15803d'), opacity=0.3),
        dict(x=min_x*0.8, y=max_y*0.8, text="🔵 IMPROVING<br>(Watch/Early Buy)", showarrow=False,
             font=dict(size=12, color='#3b82f6'), opacity=0.3),
        dict(x=max_x*0.8, y=min_y*0.8, text="🟡 WEAKENING<br>(Take Profits)", showarrow=False,
             font=dict(size=12, color='#f59e0b'), opacity=0.3),
        dict(x=min_x*0.8, y=min_y*0.8, text="🔴 LAGGING<br>(Avoid)", showarrow=False,
             font=dict(size=12, color='#dc2626'), opacity=0.3)
    ]
    
    fig.update_layout(
        title=f'Sector Rotation Map (Strength: {lookback_long}d, Momentum: {lookback_short}d vs prior)',
        xaxis_title=f'Relative Strength (% Change, {lookback_long} days)',
        yaxis_title=f'Momentum (Recent {lookback_short}d vs Prior)',
        height=700,
        template='plotly_white',
        annotations=annotations,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig, df


# - build_state_stats
# - make_heatmap
# - create_performance_comparison_chart
# - create_sector_rotation_map
# ... (copy them all here)
