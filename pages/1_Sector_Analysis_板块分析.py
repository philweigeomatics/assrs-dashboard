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
                       'Position_Size', 'Dispersion']
        
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
        colorbar=dict(title="Probability")
    ))
    fig.update_layout(
        title=title, xaxis_title="Tomorrow", yaxis_title="Today",
        height=600, template='plotly_white'
    )
    return fig


# ==========================================
# MAIN APP
# ==========================================

st.title("📊 Sector Analysis")

# Load data
v2_latest, v2_hist, v2_date, v2_error = load_v2_data()

if v2_latest is None:
    st.error(f"Error loading data: {v2_error}")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🔬 Interaction Lab"])

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
    
    # Sector Table
    st.subheader("🎯 Sector Signals")
    
    sectors_df = v2_latest[v2_latest['Sector'] != 'MARKET_PROXY'].copy()
    sectors_df = sectors_df.sort_values('TOTAL_SCORE', ascending=False)
    
    display_df = sectors_df[['Sector', 'TOTAL_SCORE', 'ACTION', 'Excess_Prob', 'Position_Size']].copy()
    display_df['TOTAL_SCORE'] = (display_df['TOTAL_SCORE'] * 100).map('{:.0f}%'.format)
    display_df['Excess_Prob'] = display_df['Excess_Prob'].map(lambda x: f"{x:+.2f}")
    display_df['Position_Size'] = (display_df['Position_Size'] * 100).map('{:.0f}%'.format)
    
    styled = display_df.style.map(style_action, subset=['ACTION'])
    st.dataframe(styled, hide_index=True, use_container_width=True)
    
    # Summary
    st.markdown("### 📊 Summary")
    buy = sectors_df[sectors_df['ACTION'].str.contains('BUY', na=False, case=False)]
    avoid = sectors_df[sectors_df['ACTION'].str.contains('AVOID', na=False, case=False)]
    
    s1, s2, s3 = st.columns(3)
    s1.metric("🟢 BUY", len(buy))
    if not buy.empty:
        s1.caption(", ".join(buy['Sector'].head(3).tolist()))
    
    s2.metric("💰 Exposure", f"{sectors_df['Position_Size'].sum()*100:.0f}%")
    
    s3.metric("🔴 AVOID", len(avoid))
    if not avoid.empty:
        s3.caption(", ".join(avoid['Sector'].head(3).tolist()))
    
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
