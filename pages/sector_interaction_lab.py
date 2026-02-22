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
    make_heatmap,
    compute_market_gate_with_context
)
from explanations import INTERACTION_LAB

import auth_manager
auth_manager.require_login()


st.title("🔬 Sector Interaction Lab")
st.markdown("### Market Gate ###")

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

# Lookback slider
lookback = st.slider("Lookback days", 5, max_lb, default_lb, 5)

# Fixed history window (1 year)
HISTORY_WINDOW = 252

st.markdown(f"""
**分析周期 Analysis Period:** 当前 {lookback} 天 vs 过去 {HISTORY_WINDOW} 天 (1年)  
Current {lookback} days vs Past {HISTORY_WINDOW} days (1 year)
""")

# Market Gate with Context
gate = compute_market_gate_with_context(
    ret_panel, 
    exret_panel, 
    lookback=lookback,  # ✅ Use slider value
    history_window=HISTORY_WINDOW
)

if gate:
    # Regime banner with Chinese/English
    if gate['regime_color'] == 'success':
        st.success(f"""
        {gate['regime_label']}  
        百分位: {gate['dispersion_percentile']*100:.0f}% | 趋势: {gate['trend_label']} | 稳定性: {gate['regime_stability']}
        """)
    elif gate['regime_color'] == 'warning':
        st.warning(f"""
        {gate['regime_label']}  
        百分位: {gate['dispersion_percentile']*100:.0f}% | 趋势: {gate['trend_label']} | 稳定性: {gate['regime_stability']}
        """)
    elif gate['regime_color'] == 'error':
        st.error(f"""
        {gate['regime_label']}  
        百分位: {gate['dispersion_percentile']*100:.0f}% | 趋势: {gate['trend_label']} | 稳定性: {gate['regime_stability']}
        """)
    else:
        st.info(f"""
        {gate['regime_label']}  
        百分位: {gate['dispersion_percentile']*100:.0f}% | 趋势: {gate['trend_label']} | 稳定性: {gate['regime_stability']}
        """)
    
    # Metrics row
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "市场收益 Market Return", 
        f"{gate['market_return']*100:.2f}%"
    )
    g2.metric(
        "离散度 Dispersion", 
        f"{gate['dispersion']*100:.2f}%",
        delta=f"{(gate['dispersion'] - gate['history_p50'])*100:+.2f}% vs 中位数"
    )
    g3.metric(
        "下跌广度 Breadth Down", 
        f"{gate['breadth_down']*100:.0f}%"
    )
    g4.metric(
        "百分位排名 Percentile", 
        f"{gate['dispersion_percentile']*100:.0f}%"
    )
    
    # Trading advice based on regime
    st.markdown("---")
    st.subheader("📋 操作建议 Trading Recommendation")
    
    if gate['regime_state'] == "EXTREME_ROTATION":
        st.success(f"""
        **🔥 极端轮动市场 - 积极进行板块轮动**
        - ✅ 当前{lookback}天离散度处于年内前15%，板块分化极大
        - ✅ 强烈推荐使用转换矩阵和次日预测进行轮动交易
        - ✅ 加大仓位于强势板块，快速切换
        - ⚠️ 注意：极端轮动可能预示市场结构变化
        """)
    elif gate['regime_state'] == "STRONG_ROTATION":
        st.success(f"""
        **✅ 强势轮动市场 - 适合板块轮动**
        - ✅ 当前{lookback}天离散度处于年内前30%，板块差异明显
        - ✅ 推荐使用板块强度指标进行选股
        - ⚪ 可进行中短期轮动操作
        """)
    elif gate['regime_state'] == "MODERATE_ROTATION":
        st.info(f"""
        **⚪ 温和轮动市场 - 谨慎轮动**
        - ⚪ {lookback}天离散度处于中等水平
        - ⚠️ 轮动信号可信度一般，需结合其他指标
        - 建议持有强势板块，观察趋势变化
        """)
    elif gate['regime_state'] == "LOW_ROTATION":
        st.warning(f"""
        **⚠️ 弱势轮动市场 - 不建议轮动**
        - ❌ 板块分化不明显，轮动效果差
        - 建议降低换手率，持有核心仓位
        - 关注市场整体方向，而非板块选择
        """)
    else:  # HIGH_CORRELATION
        st.error(f"""
        **❌ 板块共振市场 - 停止轮动**
        - ❌ 所有板块高度相关，轮动无意义
        - ❌ 当前{lookback}天离散度处于年内后30%
        - 建议关注择时，暂停板块轮动策略
        - 等待市场结构分化后再操作
        """)
    
    # Historical context visualization
    with st.expander("📊 历史对比 Historical Context", expanded=False):
        st.markdown(f"""
        **当前{lookback}天离散度: {gate['dispersion']*100:.2f}%** 在过去{HISTORY_WINDOW}天中排名 **第{gate['dispersion_percentile']*100:.0f}百分位**
        
        **历史分位数 (过去1年 {lookback}天滚动平均):**
        - 25% 分位: {gate['history_p25']*100:.2f}%
        - 50% 分位 (中位数): {gate['history_p50']*100:.2f}%
        - 75% 分位: {gate['history_p75']*100:.2f}%
        - 85% 分位: {gate['history_p85']*100:.2f}%
        
        **解读:**
        - 当前值高于中位数 **{(gate['dispersion'] - gate['history_p50'])*100:+.2f}%**
        - 趋势: {gate['trend_label']}
        - 制度稳定性: {gate['regime_stability']}
        
        **如何使用滑块:**
        - **20天**: 适合短线轮动，捕捉快速变化
        - **40天**: 中线轮动，过滤短期噪音
        - **60天**: 长线趋势，识别持久性制度
        - **更长周期**: 战略性制度判断
        """)
    
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




