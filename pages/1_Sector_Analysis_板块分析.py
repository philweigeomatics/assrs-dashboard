"""
Sector Analysis Page
- V2 Enhanced Dashboard
- Sector Interaction Lab
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import data_manager
from analysis_engine import detect_market_regime


st.set_page_config(
    page_title="📊 Sector Analysis | 板块分析",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# DATA LOADING
# ==========================================

V2_REGIME_FILE = 'assrs_backtest_results_SECTORS_V2_Regime.csv'

@st.cache_data(ttl=600)
def load_v2_data():
    """Load V2 Enhanced data."""
    try:
        df = pd.read_csv(V2_REGIME_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        numeric_cols = ['TOTAL_SCORE', 'Open', 'High', 'Low', 'Close', 
                       'Volume_Metric', 'Market_Score', 'Excess_Prob', 
                       'Position_Size', 'Dispersion','Market_Breadth']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df.empty:
            return None, None, None, "No data"
        
        latest_date = df['Date'].max()
        latest = df[df['Date'] == latest_date].copy()
        history = df.copy()
        
        return latest, history, latest_date.strftime('%Y-%m-%d'), None
    
    except Exception as e:
        return None, None, None, str(e)
    
@st.cache_data(ttl=600)
def load_csi300_with_regime(freq_cn: str):
    """
    Fetch CSI300 with enough history to compute regimes, then return full df.
    The caller can tail() it for display.
    """
    if freq_cn == "日线":
        # fetch ~1y daily so HMM/JUMP/ATR all have enough bars
        raw_df = data_manager.get_index_data_live("000300.SH", lookback_days=365, freq="daily")
        if raw_df is None or raw_df.empty:
            return raw_df
        return detect_market_regime(raw_df, freq="daily")

    # 周线: fetch ~5y to have enough weekly bars for HMM
    raw_df = data_manager.get_index_data_live("000300.SH", lookback_days=1825, freq="weekly")
    if raw_df is None or raw_df.empty:
        return raw_df
    return detect_market_regime(raw_df, freq="weekly")



# ==========================================
# HELPER FUNCTIONS (Dashboard)
# ==========================================

def style_action(action):
    """Style ACTION column."""
    if not isinstance(action, str):
        return ""
    if '🟢' in action or 'BUY' in action:
        return 'color: #15803d; background-color: #dcfce7; font-weight: 600'
    if '🟡' in action:
        return 'color: #a16207; background-color: #fef9c3; font-weight: 600'
    if '🔴' in action or 'AVOID' in action:
        return 'color: #b91c1c; background-color: #fee2e2; font-weight: 600'
    return ""


def create_sector_chart(chartdata):
    """Create 3-panel sector chart."""
    date_strings = chartdata['Date'].dt.strftime('%Y-%m-%d')
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=('Price (PPI)', 'Volume Z-Score', 'Bull Probability'),
        row_heights=[0.5, 0.2, 0.3]
    )
    
    # Price
    fig.add_trace(go.Candlestick(
        x=date_strings, open=chartdata['Open'], high=chartdata['High'],
        low=chartdata['Low'], close=chartdata['Close'], name='Price',
        increasing=dict(line=dict(color='#ef4444')),
        decreasing=dict(line=dict(color='#22c55e'))
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=date_strings, y=chartdata['Volume_Metric'], name='Volume',
        marker=dict(color='rgba(107, 114, 128, 0.3)')
    ), row=2, col=1)
    
    # Score
    fig.add_trace(go.Scatter(
        x=date_strings, y=chartdata['TOTAL_SCORE'], name='Score',
        line=dict(color='#10b981', width=2), fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ), row=3, col=1)
    
    # Thresholds
    buy_thresh = chartdata['Threshold_Buy'].iloc[-1] if 'Threshold_Buy' in chartdata.columns else 0.75
    sell_thresh = chartdata['Threshold_Sell'].iloc[-1] if 'Threshold_Sell' in chartdata.columns else 0.25
    
    fig.add_hline(y=buy_thresh, line_dash='dash', line_color='#15803d', row=3, col=1)
    fig.add_hline(y=sell_thresh, line_dash='dash', line_color='#b91c1c', row=3, col=1)
    
    fig.update_layout(
        height=700, showlegend=False, template='plotly_white',
        xaxis3_title='Date', yaxis1_title='PPI', yaxis2_title='Vol Z',
        yaxis3_title='Bull Prob', yaxis3_range=[-0.1, 1.1],
        xaxis_rangeslider_visible=False
    )
    
    return fig


# ==========================================
# HELPER FUNCTIONS (Interaction Lab)
# ==========================================

def pivot_sector_series(hist_df, value_col):
    """Pivot to Date x Sector."""
    df = hist_df[['Date', 'Sector', value_col]].dropna().copy()
    wide = df.pivot_table(index='Date', columns='Sector', values=value_col, aggfunc='last')
    return wide.sort_index()


def build_sector_panels(hist_df, market_sector="MARKET_PROXY"):
    """Build close, ret, vol, exret panels."""
    close_panel = pivot_sector_series(hist_df, 'Close')
    vol_panel = pivot_sector_series(hist_df, 'Volume_Metric')
    ret_panel = close_panel.pct_change()
    
    exret_panel = ret_panel.copy()
    if market_sector in ret_panel.columns:
        exret_panel = ret_panel.sub(ret_panel[market_sector], axis=0)
        exret_panel[market_sector] = 0.0
    
    return close_panel, ret_panel, vol_panel, exret_panel


def compute_market_gate(ret_panel, exret_panel, market_sector="MARKET_PROXY", 
                       lookback=60, mkt_down_thresh=-0.01):
    """Compute market gate metrics."""
    ret_lb = ret_panel.tail(lookback)
    ex_lb = exret_panel.tail(lookback)
    
    latest_dt = ret_lb.index.max()
    if latest_dt is None:
        return None
    
    mkt_ret = ret_lb[market_sector].iloc[-1] if market_sector in ret_lb.columns else ret_lb.mean(axis=1).iloc[-1]
    
    ex_last = ex_lb.iloc[-1].dropna()
    dispersion = ex_last.std()
    
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


def compute_transition_matrix(exret_panel, lookback=60, top_k=3, market_sector="MARKET_PROXY"):
    """Compute sector transition probabilities."""
    df = exret_panel.tail(lookback).copy()
    if market_sector in df.columns:
        df = df.drop(columns=[market_sector])
    
    if len(df) < 10:
        return None, None
    
    def get_topk(row):
        return set(row.nlargest(top_k).index.tolist())
    
    topk_sets = df.apply(get_topk, axis=1)
    
    all_sectors = df.columns.tolist()
    counts = pd.DataFrame(0, index=all_sectors, columns=all_sectors)
    
    for i in range(len(topk_sets) - 1):
        today = topk_sets.iloc[i]
        tomorrow = topk_sets.iloc[i + 1]
        for leader in today:
            for follower in tomorrow:
                counts.loc[leader, follower] += 1
    
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return probs, counts


def get_today_topk(exret_panel, top_k=3, market_sector="MARKET_PROXY"):
    """Get today's top leaders."""
    df = exret_panel.copy()
    if market_sector in df.columns:
        df = df.drop(columns=[market_sector])
    
    latest_dt = df.index.max()
    latest_row = df.loc[latest_dt].dropna().sort_values(ascending=False)
    return latest_dt, latest_row.head(top_k)


def predict_tomorrow(probs, counts, today_leaders, top_n=8, min_samples=3):
    """Predict tomorrow's followers."""
    if probs is None or today_leaders.empty:
        return pd.DataFrame()
    
    follower_scores = pd.Series(0.0, index=probs.columns)
    
    for leader in today_leaders.index:
        if leader in probs.index:
            if counts.loc[leader].sum() >= min_samples:
                follower_scores += probs.loc[leader]
    
    if follower_scores.sum() > 0:
        follower_scores /= len(today_leaders)
    
    result = follower_scores.sort_values(ascending=False).head(top_n)
    return pd.DataFrame({
        'Sector': result.index,
        'P(NextDay in Top-K)': result.values
    }).reset_index(drop=True)


def build_nextday_predictions(exret_panel, vol_panel, z_window=20, lookback=60, market_sector="MARKET_PROXY"):
    """Build state-based next-day predictions."""
    df_ex = exret_panel.tail(lookback).copy()
    df_vol = vol_panel.tail(lookback).copy()
    
    if market_sector in df_ex.columns:
        df_ex = df_ex.drop(columns=[market_sector])
    if market_sector in df_vol.columns:
        df_vol = df_vol.drop(columns=[market_sector])
    
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
        
        ex_bins = pd.cut(ex_z, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
        vol_bins = pd.cut(vol, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
        
        states = ex_bins.astype(str) + '|' + vol_bins.astype(str)
        
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
    
    ex_bins = pd.cut(ex_z, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
    vol_bins = pd.cut(vol, bins=[-np.inf, -0.5, 0.5, np.inf], labels=['L', 'M', 'H'])
    
    states = ex_bins.astype(str) + '|' + vol_bins.astype(str)
    
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


def create_sector_rotation_map(hist_df, lookback_short=5, lookback_long=20):
    """Create sector rotation quadrant map."""
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


# ==========================================
# MAIN APP
# ==========================================

st.title("📊 Sector Analysis")

v2_latest, v2_hist, v2_date, v2_error = load_v2_data()

if v2_latest is None:
    st.error(f"❌ Error loading data: {v2_error}")
    st.stop()


# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Interaction Lab", "🎯 Performance & Rotation"])

# ==========================================
# TAB 1: DASHBOARD
# ==========================================

with tab1:
    
    # Market Banner
    if 'Market_Regime' in v2_latest.columns:
        market_row = v2_latest[v2_latest['Sector'] == 'MARKET_PROXY']
        
        if not market_row.empty:
            market_regime = market_row.iloc[0]['Market_Regime']
            market_score = market_row.iloc[0]['Market_Score']
            strategy = market_row.iloc[0].get('Strategy', 'N/A')
            rotation = v2_latest.iloc[0].get('Rotation_Status', 'N/A')
            dispersion = v2_latest.iloc[0].get('Dispersion', 0)
            
            if '🚀' in market_regime or 'Strong Bull' in market_regime:
                st.success(f"## {market_regime} Market")
            elif '📈' in market_regime:
                st.success(f"## {market_regime} Market")
            elif '📊' in market_regime:
                st.info(f"## {market_regime} Market")
            else:
                st.error(f"## {market_regime} Market")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CSI 300 Top 10", f"{market_score*100:.0f}%")
            m2.metric("Rotation", rotation)
            m3.metric("Dispersion", f"{dispersion:.2f}")
            m4.metric("Strategy", strategy)
            
            st.caption(f"📅 {v2_date}")
            st.markdown("---")
    
    # === TWO COLUMNS: SECTOR SIGNALS + CSI 300 CHART ===
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📊 Sector Signals")
        
        sectors_df = v2_latest[v2_latest['Sector'] != 'MARKET_PROXY'].copy()
        sectors_df = sectors_df.sort_values('TOTAL_SCORE', ascending=False)
        
        display_df = sectors_df[['Sector', 'TOTAL_SCORE', 'ACTION', 'Market_Breadth', 'Excess_Prob', 'Position_Size']].copy()
        display_df['TOTAL_SCORE'] = display_df['TOTAL_SCORE'].apply(lambda x: f"{x*100:.0f}%")
        display_df['Excess_Prob'] = display_df['Excess_Prob'].map(lambda x: f"{x:.2f}")
        display_df['Position_Size'] = (display_df['Position_Size'] * 100).map(lambda x: f"{x:.0f}%")
        display_df['Market_Breadth'] = (display_df['Market_Breadth'] * 100).map(lambda x: f"{x:.0f}%")
        
        def style_action(val):
            if 'BUY' in val:
                return 'color: #15803d; background-color: #dcfce7; font-weight: 600'
            elif 'AVOID' in val:
                return 'color: #b91c1c; background-color: #fee2e2; font-weight: 600'
            return ''
        
        def style_breadth(val):
            if isinstance(val, str) and '%' in val:
                pct = float(val.replace('%', ''))
                if pct > 50:
                    return 'color: #15803d; background-color: #dcfce7; font-weight: 600'
                else:
                    return 'color: #b91c1c; background-color: #fee2e2; font-weight: 600'
            return ''
        
        styled = display_df.style.map(style_action, subset=['ACTION']).map(style_breadth, subset=['Market_Breadth'])
        st.dataframe(styled, hide_index=True, use_container_width=True, height=400)
        
        # Summary
        st.markdown("**Summary**")
        buy = sectors_df[sectors_df['ACTION'].str.contains('BUY', na=False, case=False)]
        avoid = sectors_df[sectors_df['ACTION'].str.contains('AVOID', na=False, case=False)]
        
        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 BUY", len(buy))
        if not buy.empty:
            s1.caption(", ".join(buy['Sector'].head(3).tolist()))
        
        s2.metric("📊 Exposure", f"{sectors_df['Position_Size'].sum()*100:.0f}%")
        
        s3.metric("🔴 AVOID", len(avoid))
        if not avoid.empty:
            s3.caption(", ".join(avoid['Sector'].head(3).tolist()))
    
    with col_right:
        st.subheader("📈 沪深300 ")
        
        # Frequency selector (horizontal, compact)
        freq = st.radio("周期", ["日线", "周线"], key='csi300_freq', horizontal=True, label_visibility="collapsed")
        

        # Fetch CSI300 + compute regime (cached)
        with st.spinner("📡 加载中..."):
            raw_df = load_csi300_with_regime(freq)

        if raw_df is not None and not raw_df.empty:
            if freq == "日线":
                chart_df = raw_df.tail(180).copy()
                title = "CSI 300 - 日K线 (6个月)"
            else:
                chart_df = raw_df.tail(52).copy()
                title = "CSI 300 - 周K线 (1年)"
        else:
            chart_df = None

        
        if chart_df is not None and not chart_df.empty:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            if "Market_Regime" in chart_df.columns and chart_df["Market_Regime"].notna().any():
              st.caption(f"当前波动状态: **{chart_df['Market_Regime'].dropna().iloc[-1]}**")

            
            # Prepare dates as strings
            dates = chart_df.index.strftime('%Y-%m-%d').tolist()
            
            # === SHOW ONLY 5 DATES ===
            total_dates = len(dates)
            tick_interval = max(1, total_dates // 5)  # Divide into 5 segments
            tick_vals = dates[::tick_interval][:5]  # Take only first 5
            tick_text = tick_vals
            
            # Create compact candlestick chart with volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # ===== Regime shading =====
            regime_colors = {
                "Low Volatility":    "rgba(34, 197, 94, 0.08)",
                "Normal Volatility": "rgba(59, 130, 246, 0.05)",
                "High Volatility":   "rgba(255, 110, 0, 0.11)",
                "Extreme Volatility":"rgba(220, 38, 38, 0.12)",  # in case ATR fallback produces this label
            }

            if "Market_Regime" in chart_df.columns and chart_df["Market_Regime"].notna().any():
                df_clean = chart_df.dropna(subset=["Market_Regime"]).copy()

                # segment by regime changes
                changes = df_clean["Market_Regime"].ne(df_clean["Market_Regime"].shift(1))
                change_indices = df_clean.index[changes].tolist()
                if len(change_indices) == 0 or change_indices[0] != df_clean.index[0]:
                    change_indices.insert(0, df_clean.index[0])

                # y ranges for each subplot
                y_min_price = df_clean["Low"].min() * 0.98
                y_max_price = df_clean["High"].max() * 1.02
                y_max_vol = df_clean["Volume"].max() * 1.05

                for i in range(len(change_indices)):
                    start_idx = change_indices[i]
                    end_idx = change_indices[i + 1] if i + 1 < len(change_indices) else df_clean.index[-1]

                    regime = df_clean.loc[start_idx, "Market_Regime"]
                    if regime not in regime_colors:
                        continue

                    start_date = start_idx.strftime("%Y-%m-%d")
                    end_date = end_idx.strftime("%Y-%m-%d")

                    # Price panel shading
                    fig.add_shape(
                        type="rect",
                        x0=start_date, x1=end_date,
                        y0=y_min_price, y1=y_max_price,
                        fillcolor=regime_colors[regime],
                        line=dict(width=0),
                        layer="below",
                        row=1, col=1
                    )

                    # # Volume panel shading
                    # fig.add_shape(
                    #     type="rect",
                    #     x0=start_date, x1=end_date,
                    #     y0=0, y1=y_max_vol,
                    #     fillcolor=regime_colors[regime],
                    #     line=dict(width=0),
                    #     layer="below",
                    #     row=2, col=1
                    # )
            # ===== end shading =====

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=dates,
                open=chart_df['Open'],
                high=chart_df['High'],
                low=chart_df['Low'],
                close=chart_df['Close'],
                name='CSI 300',
                increasing_line_color='#ef4444',
                decreasing_line_color='#22c55e'
            ), row=1, col=1)
            
            # Volume bars
            colors = ['#ef4444' if chart_df['Close'].iloc[i] >= chart_df['Open'].iloc[i] 
                    else '#22c55e' for i in range(len(chart_df))]
            
            fig.add_trace(go.Bar(
                x=dates,
                y=chart_df['Volume'],
                name='成交量',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
            
            fig.update_layout(
                title=title,
                height=500,
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=40)
            )
            
            fig.update_yaxes(title_text="指数", row=1, col=1)
            fig.update_yaxes(title_text="成交量", row=2, col=1)
            
            # Show only 5 dates, horizontal
            fig.update_xaxes(
                type='category',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=0,  # Horizontal
                row=2, col=1
            )
            
            # Hide ticks on top panel
            fig.update_xaxes(
                type='category',
                showticklabels=False,
                row=1, col=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ 无法加载数据")

    

    # Add after the Summary section (around line 460)
    st.markdown("---")
    st.subheader("📊 Market Breadth History 市场宽度 - 超过20天均线")

    # Filter last 60 days of data
    breadth_history = v2_hist[v2_hist['Sector'] != 'MARKET_PROXY'].copy()
    breadth_history = breadth_history.sort_values('Date', ascending=False)

    # Get unique dates (most recent first)
    unique_dates = breadth_history['Date'].dt.date.unique()[:60]  # Last 60 days

    if len(unique_dates) == 0:
        st.warning("No historical breadth data available")
    else:
        # Pagination setup
        DAYS_PER_PAGE = 10
        total_pages = (len(unique_dates) + DAYS_PER_PAGE - 1) // DAYS_PER_PAGE
        
        # Initialize page state
        if 'breadth_page' not in st.session_state:
            st.session_state.breadth_page = 0
        
        # Page navigation buttons
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.button("◀ Previous 10 Days", disabled=(st.session_state.breadth_page >= total_pages - 1)):
                st.session_state.breadth_page += 1
                st.rerun()
        
        with col2:
            start_idx = st.session_state.breadth_page * DAYS_PER_PAGE
            page_end = min(start_idx + DAYS_PER_PAGE, len(unique_dates))
            date_range = f"{unique_dates[page_end-1]} to {unique_dates[start_idx]}"
            st.markdown(f"<center><b>Page {st.session_state.breadth_page + 1} of {total_pages}</b><br>{date_range}</center>", 
                    unsafe_allow_html=True)
        
        with col3:
            if st.button("Next 10 Days ▶", disabled=(st.session_state.breadth_page == 0)):
                st.session_state.breadth_page -= 1
                st.rerun()
        
        # Get dates for current page
        end_idx = start_idx + DAYS_PER_PAGE
        page_dates = unique_dates[start_idx:end_idx]
        
        # REVERSE dates so newest is on the right
        page_dates = page_dates[::-1]
        
        # Build the breadth table
        breadth_data = []
        
        for sector in sorted(breadth_history['Sector'].unique()):
            row = {'Sector': sector}
            for date in page_dates:
                # Get breadth value for this sector on this date
                sector_date_data = breadth_history[
                    (breadth_history['Sector'] == sector) & 
                    (breadth_history['Date'].dt.date == date)
                ]
                
                if not sector_date_data.empty:
                    breadth_val = sector_date_data.iloc[0]['Market_Breadth']
                    row[date.strftime('%m/%d')] = breadth_val  # Keep as number
                else:
                    row[date.strftime('%m/%d')] = None
            
            breadth_data.append(row)
        
        breadth_df = pd.DataFrame(breadth_data)
        
        # Style function for breadth values
        def style_breadth_cell(val):
            """Style breadth: <50% green (opportunity), >=50% red (overextended)"""
            if pd.isna(val):
                return ''
            pct = val * 100
            if pct < 50:
                return 'background-color: #dcfce7; color: #15803d; font-weight: 600'
            else:
                return 'background-color: #fee2e2; color: #b91c1c; font-weight: 600'
        
        # Format function for display (separate from styling)
        def format_breadth(val):
            """Format as percentage"""
            if pd.isna(val):
                return "—"
            return f"{val*100:.0f}%"
        
        # Apply styling FIRST (on numeric values)
        date_cols = [col for col in breadth_df.columns if col != 'Sector']
        styled = breadth_df.style.map(style_breadth_cell, subset=date_cols)
        
        # THEN format for display
        styled = styled.format(format_breadth, subset=date_cols)
        
        st.dataframe(styled, hide_index=True, use_container_width=True, height=500)
        
        st.caption("💡 Green (<50%): Most stocks below MA20 (opportunity). Red (≥50%): Most stocks above MA20 (extended).")


    # Chart
    st.markdown("---")
    st.subheader("🔍 Sector Deep Dive")
    
    sector = st.selectbox("Select Sector:", sorted(v2_hist['Sector'].unique()))
    if sector:
        data = v2_hist[v2_hist['Sector'] == sector]
        fig = create_sector_chart(data)
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 2: INTERACTION LAB
# ==========================================

with tab2:
    st.subheader("🔬 Sector Interaction Lab")
    
    # Build panels
    close_panel, ret_panel, vol_panel, exret_panel = build_sector_panels(v2_hist)
    
    available_days = int(exret_panel.dropna(how="all").shape[0])
    
    if available_days < 30:
        st.warning(f"Not enough data ({available_days} days)")
        st.stop()
    
    max_lb = min(120, available_days)
    default_lb = min(60, max_lb)
    
    lookback = st.slider("Lookback (days)", 20, max_lb, default_lb, 5)
    
    # Market Gate
    gate = compute_market_gate(ret_panel, exret_panel, lookback=lookback)
    
    if gate:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Market Return", f"{gate['market_return']*100:.2f}%")
        g2.metric("Dispersion", f"{gate['dispersion']*100:.2f}%")
        g3.metric("Breadth Down", f"{gate['breadth_down']*100:.0f}%")
        g4.metric("Confidence", gate['confidence'])
        
        if gate['confidence'] == "LOW":
            st.warning(f"Low rotation confidence today")
        elif gate['confidence'] == "HIGH":
            st.success(f"High dispersion - rotation signals meaningful")
    
    st.markdown("---")
    
    # Sub-tabs
    t1, t2, t3 = st.tabs(["Transition Matrix", "Next-Day Odds", "Raw Panels"])
    
    # Transition Matrix
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            top_k = st.selectbox("Top-K", [2, 3, 4, 5], index=1)
        with c2:
            tm_lb = st.selectbox("Lookback", [20, 30, 40, 60, 90, 120], index=3)
        
        probs, counts = compute_transition_matrix(exret_panel, tm_lb, top_k)
        latest_dt, leaders = get_today_topk(exret_panel, top_k)
        
        st.markdown(f"**Today's Leaders ({latest_dt.strftime('%Y-%m-%d')})**")
        st.dataframe(leaders.reset_index().rename(columns={'index': 'Sector', latest_dt: 'ExcessRet'}), 
                    hide_index=True)
        
        st.markdown("**Predicted Followers**")
        pred = predict_tomorrow(probs, counts, leaders)
        st.dataframe(pred, hide_index=True)
        
        if probs is not None:
            fig = make_heatmap(probs, f"Top-{top_k} Transitions ({tm_lb}d)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Next-Day Odds
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            z_win = st.selectbox("Z-window", [10, 15, 20, 30], index=2)
        with c2:
            odds_lb = st.selectbox("Training lookback", [30, 40, 60, 90], index=2)
        
        preds = build_nextday_predictions(exret_panel, vol_panel, z_win, odds_lb)
        
        if preds is not None:
            st.dataframe(preds.head(20), hide_index=True)
            
            sector_pick = st.selectbox("Deep-dive", preds['Sector'].tolist())
            stats, state = build_state_stats(exret_panel[sector_pick], vol_panel[sector_pick], z_win)
            
            if stats is not None:
                st.caption(f"Current state: **{state}**")
                
                top = stats.head(12)
                fig = go.Figure(go.Bar(x=top.index.tolist(), y=top['win_rate'].values))
                fig.update_layout(title=f"Win Rate by State - {sector_pick}", 
                                 xaxis_title="State", yaxis_title="Win Rate", height=450)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(stats)
        else:
            st.warning("Not enough data")
    
    # Raw Panels
    with t3:
        sectors = [c for c in exret_panel.columns if c != 'MARKET_PROXY']
        pick = st.selectbox("Sector", sectors)
        
        df_view = pd.DataFrame({
            'ExcessRet': exret_panel[pick],
            'Volume': vol_panel[pick],
            'Close': close_panel[pick]
        }).dropna().tail(200)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                           subplot_titles=('Close', 'Excess Return', 'Volume Z'),
                           row_heights=[0.5, 0.25, 0.25])
        
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Close'], name='Close'), row=1, col=1)
        fig.add_trace(go.Bar(x=df_view.index, y=df_view['ExcessRet'], name='ExRet'), row=2, col=1)
        fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], name='Vol'), row=3, col=1)
        
        fig.update_layout(height=700, template='plotly_white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ========================================== 
# TAB 3: PERFORMANCE & ROTATION
# ========================================== 
with tab3:

    # ========================================== 
    # TAB 3: PERFORMANCE & ROTATION
    # ========================================== 
    st.header("📈 Sector Co-Movement Analysis")

    # Sector selection
    all_sectors = sorted([s for s in v2_hist['Sector'].unique() if s != 'MARKET_PROXY'])

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_sectors = st.multiselect(
            "Select sectors to compare (2-5 recommended)",
            options=all_sectors,
            default=all_sectors[:3] if len(all_sectors) >= 3 else all_sectors,
            key="sector_selector"
        )
    with col2:
        lookback = st.selectbox("Time Period", [30, 60, 90, 120, 180], index=1, key="lookback_selector")

    if not selected_sectors or len(selected_sectors) < 2:
        st.warning("Please select at least 2 sectors to compare")
        st.stop()

    # Get data
    latest_date = v2_hist['Date'].max()
    start_date = latest_date - pd.Timedelta(days=lookback)
    recent = v2_hist[v2_hist['Date'] >= start_date].copy()

    # Pivot and calculate daily returns
    pivot = recent.pivot_table(index='Date', columns='Sector', values='Close')
    pivot = pivot[selected_sectors]
    returns = pivot.pct_change() * 100  # Daily % returns

    # Drop first row (NaN) and remove days with missing data
    returns = returns.dropna()

    st.markdown("---")

    # ========================================== 
    # 1. DAILY RETURNS COMPARISON (Main View)
    # ========================================== 
    st.subheader("📊 Daily Returns Comparison")
    st.caption("Shows how much each sector moved (%) each day - easy to see which move together")

    # Create multi-line chart of daily returns
    fig_returns = go.Figure()

    for sector in returns.columns:
        fig_returns.add_trace(go.Scatter(
            x=returns.index,
            y=returns[sector],
            name=sector,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{sector}</b><br>Date: %{{x|%Y-%m-%d}}<br>Daily Return: %{{y:.2f}}%<extra></extra>'
        ))

    fig_returns.add_hline(y=0, line_dash='solid', line_color='gray', line_width=1, opacity=0.5)

    fig_returns.update_layout(
        xaxis_title='Date',
        yaxis_title='Daily Return (%)',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_returns, use_container_width=True)

    st.markdown("---")

    # ========================================== 
    # 2. CORRELATION HEATMAP
    # ========================================== 
    st.subheader("🔥 Correlation Matrix")
    st.caption("Shows how sectors move together: 1.0 = perfect sync, -1.0 = perfect opposite, 0 = independent")

    corr_matrix = returns.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdYlGn',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=14),
        colorbar=dict(title="Correlation"),
        zmin=-1,
        zmax=1
    ))

    fig_corr.update_layout(
        title=f'Daily Return Correlation - Last {lookback} Days',
        height=max(400, len(selected_sectors) * 60),
        template='plotly_white',
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12))
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.markdown("**Interpretation:**")
        
        # Find highest correlation pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
        
        # Only show if we have enough pairs
        if len(corr_df) > 3:
            st.markdown("**Highest Correlation:**")
            for _, row in corr_df.head(3).iterrows():
                corr_val = row['Correlation']
                if corr_val > 0.7:
                    emoji = "🟢"
                elif corr_val > 0.3:
                    emoji = "🟡"
                else:
                    emoji = "⚪"
                st.markdown(f"{emoji} {row['Pair']}: **{corr_val:.2f}**")
            
            st.markdown("**Lowest Correlation:**")
            for _, row in corr_df.tail(3).iterrows():
                corr_val = row['Correlation']
                if corr_val < -0.3:
                    emoji = "🔴"
                elif corr_val < 0.3:
                    emoji = "🟡"
                else:
                    emoji = "⚪"
                st.markdown(f"{emoji} {row['Pair']}: **{corr_val:.2f}**")
        else:
            # Just show all pairs when there are 3 or fewer
            st.markdown("**Sector Pairs:**")
            for _, row in corr_df.iterrows():
                corr_val = row['Correlation']
                if corr_val > 0.7:
                    emoji = "🟢 Strong"
                elif corr_val > 0.3:
                    emoji = "🟡 Moderate"
                elif corr_val > -0.3:
                    emoji = "⚪ Weak"
                else:
                    emoji = "🔴 Negative"
                st.markdown(f"{emoji} {row['Pair']}: **{corr_val:.2f}**")
            
            st.caption(f"Select more sectors to see top/bottom correlations")


    st.markdown("---")

    # ========================================== 
    # 3. SCATTER PLOT (Pairwise Comparison)
    # ========================================== 
    if len(selected_sectors) >= 2:
        st.subheader("📍 Pairwise Movement Scatter")
        st.caption("When Sector X moves +1%, what does Sector Y do? Each dot is one day.")
        
        col1, col2 = st.columns(2)
        with col1:
            sector_x = st.selectbox("X-axis (Reference Sector)", selected_sectors, index=0, key="scatter_x")
        with col2:
            sector_y = st.selectbox("Y-axis (Compare To)", 
                                    [s for s in selected_sectors if s != sector_x], 
                                    index=0, key="scatter_y")
        
        # Create scatter plot
        fig_scatter = go.Figure()
        date_numeric = (returns.index - returns.index.min()).days

        fig_scatter.add_trace(go.Scatter(
            x=returns[sector_x],
            y=returns[sector_y],
            mode='markers',
            marker=dict(
                size=8,
                color=date_numeric,  # ✅ Numbers work with colorscale
                colorscale='Viridis',
                showscale=False
            ),
            text=returns.index.strftime('%Y-%m-%d'),
            hovertemplate=f'<b>%{{text}}</b><br>{sector_x}: %{{x:.2f}}%<br>{sector_y}: %{{y:.2f}}%<extra></extra>'
        ))
        
        # Add diagonal line (perfect correlation)
        max_val = max(returns[sector_x].max(), returns[sector_y].max())
        min_val = min(returns[sector_x].min(), returns[sector_y].min())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Perfect Correlation',
            showlegend=True
        ))
        
        # Add zero lines
        fig_scatter.add_hline(y=0, line_dash='solid', line_color='lightgray', line_width=1)
        fig_scatter.add_vline(x=0, line_dash='solid', line_color='lightgray', line_width=1)
        
        # Calculate correlation
        correlation = returns[sector_x].corr(returns[sector_y])
        
        fig_scatter.update_layout(
            title=f'{sector_x} vs {sector_y} | Correlation: {correlation:.2f}',
            xaxis_title=f'{sector_x} Daily Return (%)',
            yaxis_title=f'{sector_y} Daily Return (%)',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Interpretation
        if correlation > 0.7:
            st.success(f"**Strong Positive Correlation ({correlation:.2f}):** These sectors move together. When {sector_x} goes up 1%, {sector_y} typically goes up too.")
        elif correlation > 0.3:
            st.info(f"**Moderate Positive Correlation ({correlation:.2f}):** These sectors somewhat move together.")
        elif correlation > -0.3:
            st.warning(f"**Low Correlation ({correlation:.2f}):** These sectors move independently. Good for diversification!")
        else:
            st.error(f"**Negative Correlation ({correlation:.2f}):** These sectors move opposite. When {sector_x} goes up, {sector_y} tends to go down.")

    st.markdown("---")

    # ========================================== 
    # 4. DAILY MOVEMENT TABLE
    # ========================================== 
    with st.expander("📋 Show Daily Movement Table", expanded=False):
        st.caption("Raw data: Daily % changes for each sector")
        
        display_returns = returns.tail(30).copy()  # Last 30 days
        display_returns = display_returns.sort_index(ascending=False)
        
        # Format with color
        def color_returns(val):
            if pd.isna(val):
                return ''
            color = '#dcfce7' if val > 0 else '#fee2e2' if val < 0 else ''
            return f'background-color: {color}'
        
        styled_returns = display_returns.style.format('{:+.2f}%').applymap(color_returns)
        
        st.dataframe(styled_returns, use_container_width=True, height=400)


    
    # Rotation Map
    st.header("🎯 Sector Rotation Map")
    
    col1, col2 = st.columns(2)
    with col1:
        short_period = st.slider("Momentum Period (days)", 3, 10, 5)
    with col2:
        long_period = st.slider("Strength Period (days)", 10, 30, 20)
    
    rotation_chart, rotation_df = create_sector_rotation_map(
        v2_hist,
        lookback_short=short_period,
        lookback_long=long_period
    )
    
    if rotation_chart:
        st.plotly_chart(rotation_chart, use_container_width=True)
        
        # Summary by quadrant
        st.subheader("📊 Quadrant Breakdown")
        
        quad_cols = st.columns(4)
        for idx, (quadrant, color) in enumerate([
            ('🟢 Leading', '#dcfce7'),
            ('🔵 Improving', '#dbeafe'),
            ('🟡 Weakening', '#fef9c3'),
            ('🔴 Lagging', '#fee2e2')
        ]):
            with quad_cols[idx]:
                sectors_in_quad = rotation_df[rotation_df['Quadrant'] == quadrant]['Sector'].tolist()
                st.markdown(f"**{quadrant}**")
                if sectors_in_quad:
                    for sector in sectors_in_quad:
                        st.markdown(f"• {sector}")
                else:
                    st.caption("None")
        
        # Detailed table
        st.subheader("📋 Detailed Metrics")
        display_rotation = rotation_df[['Sector', 'Quadrant', 'Relative_Strength', 'Momentum']].copy()
        display_rotation['Relative_Strength'] = display_rotation['Relative_Strength'].map('{:+.1f}%'.format)
        display_rotation['Momentum'] = display_rotation['Momentum'].map('{:+.1f}%'.format)
        display_rotation = display_rotation.sort_values('Quadrant')
        
        st.dataframe(display_rotation, hide_index=True, use_container_width=True)
        
        # Explanation
        with st.expander("💡 How to use the Rotation Map"):
            st.markdown("""
            **Quadrants explain sector lifecycle:**
            - **🟢 Leading (Top Right):** Strong AND accelerating → **BUY or HOLD**
            - **🔵 Improving (Top Left):** Weak but gaining momentum → **WATCH for entry**
            - **🟡 Weakening (Bottom Right):** Strong but slowing down → **TAKE PROFITS**
            - **🔴 Lagging (Bottom Left):** Weak AND declining → **AVOID**
            
            **Trading Strategy:**
            1. **Buy** when sectors move from Improving → Leading
            2. **Hold** while in Leading quadrant
            3. **Sell** when sectors move from Leading → Weakening
            4. **Avoid** sectors in Lagging quadrant
            """)
    else:
        st.warning("Not enough data to create rotation map")