"""
Sector Analysis - Interaction Lab
Analyze sector correlations, transitions, and next-day odds
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sector_utils import (
    load_v2_data,
    build_sector_panels,
    compute_market_gate,
    compute_transition_matrix,
    get_today_topk,
    predict_tomorrow,
    build_nextday_predictions,
    build_state_stats,
    make_heatmap
)
from explanations import INTERACTION_LAB

st.title("🔬 Sector Interaction Lab")

# Load data
v2latest, v2hist, v2date, v2error = load_v2_data()
if v2latest is None:
    st.error(f"Error loading data: {v2error}")
    st.stop()

# Build panels
close_panel, ret_panel, vol_panel, exret_panel = build_sector_panels(v2hist)

# Check available days
available_days = int(exret_panel.dropna(how='all').shape[0])
if available_days < 30:
    st.warning(f"Not enough data: {available_days} days")
    st.stop()

max_lb = min(120, available_days)
default_lb = min(60, max_lb)
lookback = st.slider("Lookback days", 20, max_lb, default_lb, 5)

# Market Gate
gate = compute_market_gate(ret_panel, exret_panel, lookback=lookback)

if gate:
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Market Return", f"{gate['market_return']*100:.2f}%")
    g2.metric("Dispersion", f"{gate['dispersion']*100:.2f}%")
    g3.metric("Breadth Down", f"{gate['breadth_down']*100:.0f}%")
    g4.metric("Confidence", gate['confidence'])
    
    if gate['confidence'] == 'LOW':
        st.warning(f"⚠️ Low rotation confidence today")
    elif gate['confidence'] == 'HIGH':
        st.success(f"✅ High dispersion - rotation signals meaningful")

st.markdown("---")

# Create sub-tabs
t1, t2, t3 = st.tabs(["🔄 Transition Matrix", "🎲 Next-Day Odds", "📋 Raw Panels"])

with t1:
    st.subheader("🔄 Sector Transition Probabilities")
    
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.selectbox("Top-K", [2, 3, 4, 5], index=1)
    with c2:
        tm_lb = st.selectbox("Lookback", [20, 30, 40, 60, 90, 120], index=3)
    
    probs, counts = compute_transition_matrix(exret_panel, tm_lb, top_k)
    latest_dt, leaders = get_today_topk(exret_panel, top_k)
    
    st.markdown(f"**Today's Leaders** ({latest_dt.strftime('%Y-%m-%d')})")
    st.dataframe(leaders.reset_index().rename(columns={'index': 'Sector', latest_dt: 'ExcessRet%'}), hide_index=True)
    
    st.markdown("**Predicted Followers**")
    pred = predict_tomorrow(probs, counts, leaders)
    st.dataframe(pred, hide_index=True)
    
    if probs is not None:
        fig = make_heatmap(probs, f"Top-{top_k} Transitions ({tm_lb}d)")
        st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("🎲 Next-Day Odds (State-Based)")
    
    with st.expander("How does it work?", expanded=False):
    # ✅ ADD LANGUAGE TOGGLE
        col1, col2, col3 = st.columns([3, 1, 1])
        with col2:
            lang = st.radio(
                "Language",
                ["English", "中文"],
                horizontal=True,
                key="interaction_lab_lang"
            )
        # ✅ USE LANGUAGE-SPECIFIC CONTENT  
        st.markdown(INTERACTION_LAB[lang]["content"])
    
    
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
            
            fig = go.Figure(go.Bar(
                x=top.index.tolist(),
                y=top['win_rate'].values
            ))
            fig.update_layout(
                title=f"Win Rate by State - {sector_pick}",
                xaxis_title="State",
                yaxis_title="Win Rate",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(stats)
        else:
            st.warning("Not enough data")

with t3:
    st.subheader("📋 Raw Data Panels")
    
    sectors = [c for c in exret_panel.columns if c != "MARKET_PROXY"]
    pick = st.selectbox("Sector", sectors)
    
    df_view = pd.DataFrame({
        'ExcessRet': exret_panel[pick],
        'Volume': vol_panel[pick],
        'Close': close_panel[pick]
    }).dropna().tail(200)
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Close", "Excess Return", "Volume Z"),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Close'], name='Close'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_view.index, y=df_view['ExcessRet'], name='ExRet'), row=2, col=1)
    fig.add_trace(go.Bar(x=df_view.index, y=df_view['Volume'], name='Vol'), row=3, col=1)
    
    fig.update_layout(height=700, template='plotly_white', showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)




