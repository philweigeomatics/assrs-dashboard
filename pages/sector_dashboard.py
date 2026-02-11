"""
Sector Analysis - Dashboard
Shows overview of all sectors with key metrics and CSI 300 chart
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sector_utils import (
    load_v2_data, 
    load_csi300_with_regime,
    create_sector_chart
)


# Load data
v2latest, v2hist, v2date, v2error = load_v2_data()
if v2latest is None:
    st.error(f"Error loading data: {v2error}")
    st.stop()

# Market Banner (if available)
if 'MarketRegime' in v2latest.columns:
    market_row = v2latest[v2latest['Sector'] == "MARKET_PROXY"]
    if not market_row.empty:
        market_regime = market_row.iloc[0]['MarketRegime']
        market_score = market_row.iloc[0]['MarketScore']
        strategy = market_row.iloc[0].get('Strategy', 'N/A')
        rotation = v2latest.iloc[0].get('RotationStatus', 'N/A')
        dispersion = v2latest.iloc[0].get('Dispersion', 0)
        
        if "🟢" in market_regime or "Strong Bull" in market_regime:
            st.success(f"🟢 {market_regime} Market")
        elif "🟡" in market_regime:
            st.success(f"🟡 {market_regime} Market")
        elif "🔴" in market_regime:
            st.info(f"🔴 {market_regime} Market")
        else:
            st.error(f"⚠️ {market_regime} Market")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CSI 300 (Top 10%)", f"{market_score * 100.0:.0f}%")
        m2.metric("Rotation", rotation)
        m3.metric("Dispersion", f"{dispersion:.2f}")
        m4.metric("Strategy", strategy)
        
        st.caption(f"📅 {v2date}")
        st.markdown("---")

# Two columns: Sector Signals | CSI 300 Chart
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📊 Sector Signals")
    
    # Filter out MARKET_PROXY
    sectors_df = v2latest[v2latest['Sector'] != "MARKET_PROXY"].copy()
    sectors_df = sectors_df.sort_values('TOTAL_SCORE', ascending=False)
    
    # Display table
    display_df = sectors_df[['Sector', 'TOTAL_SCORE', 'ACTION', 'Market_Breadth', 'Excess_Prob', 'Position_Size']].copy()
    display_df['TOTAL_SCORE'] = display_df['TOTAL_SCORE'].apply(lambda x: f"{x*100.0:.0f}%")
    display_df['Excess_Prob'] = display_df['Excess_Prob'].map(lambda x: f"{x:.2f}")
    display_df['Position_Size'] = (display_df['Position_Size'] * 100).map(lambda x: f"{x:.0f}%")
    display_df['Market_Breadth'] = (display_df['Market_Breadth'] * 100).map(lambda x: f"{x:.0f}%")
    
    def style_action(val):
        if "BUY" in val:
            return "color: #15803d; background-color: #dcfce7; font-weight: 600"
        elif "AVOID" in val:
            return "color: #b91c1c; background-color: #fee2e2; font-weight: 600"
        return ""
    
    def style_breadth(val):
        if isinstance(val, str) and "%" in val:
            pct = float(val.replace("%", ""))
            if pct < 50:
                return "color: #15803d; background-color: #dcfce7; font-weight: 600"
            else:
                return "color: #b91c1c; background-color: #fee2e2; font-weight: 600"
        return ""
    
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
    
    s2.metric("💼 Exposure", f"{sectors_df['Position_Size'].sum()*100.0:.0f}%")
    
    s3.metric("🔴 AVOID", len(avoid))
    if not avoid.empty:
        s3.caption(", ".join(avoid['Sector'].head(3).tolist()))

with col_right:
    st.subheader("📈 沪深300指数")
    
    # Frequency selector (horizontal, compact)
    freq = st.radio("", ["日", "周"], key="csi300_freq", horizontal=True, label_visibility="collapsed")
    
    with st.spinner("..."):
        rawdf = load_csi300_with_regime(freq)
    
    if rawdf is not None and not rawdf.empty:
        if freq == "日":
            chart_df = rawdf.tail(180).copy()
            title = "CSI 300 - 日K (6个月)"
        else:
            chart_df = rawdf.tail(52).copy()
            title = "CSI 300 - 周K (1年)"
    else:
        chart_df = None
    
    if chart_df is not None and not chart_df.empty:
        # Show current regime
        if 'Market_Regime' in chart_df.columns and chart_df['Market_Regime'].notna().any():
            st.caption(f"🔹 {chart_df['Market_Regime'].dropna().iloc[-1]}")
        
        # Prepare dates as strings
        dates = chart_df.index.strftime('%Y-%m-%d').tolist()
        
        # Show only 5 dates
        total_dates = len(dates)
        tick_interval = max(1, total_dates // 5)
        tick_vals = dates[::tick_interval][:5]
        tick_text = tick_vals
        
        # Create compact candlestick chart with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Regime shading
        regime_colors = {
            'Low Volatility': 'rgba(34, 197, 94, 0.08)',
            'Normal Volatility': 'rgba(59, 130, 246, 0.05)',
            'High Volatility': 'rgba(255, 110, 0, 0.11)',
            'Extreme Volatility': 'rgba(220, 38, 38, 0.12)'
        }
        
        if 'Market_Regime' in chart_df.columns and chart_df['Market_Regime'].notna().any():
            df_clean = chart_df.dropna(subset=['Market_Regime']).copy()
            
            # Segment by regime changes
            changes = df_clean['Market_Regime'].ne(df_clean['Market_Regime'].shift(1))
            change_indices = df_clean.index[changes].tolist()
            
            if len(change_indices) == 0 or change_indices[0] != df_clean.index[0]:
                change_indices.insert(0, df_clean.index[0])
            
            # Y ranges for each subplot
            y_min_price = df_clean['Low'].min() * 0.98
            y_max_price = df_clean['High'].max() * 1.02
            y_max_vol = df_clean['Volume'].max() * 1.05
            
            for i in range(len(change_indices)):
                start_idx = change_indices[i]
                end_idx = change_indices[i + 1] if i + 1 < len(change_indices) else df_clean.index[-1]
                regime = df_clean.loc[start_idx, 'Market_Regime']
                
                if regime not in regime_colors:
                    continue
                
                start_date = start_idx.strftime('%Y-%m-%d')
                end_date = end_idx.strftime('%Y-%m-%d')
                
                # Price panel shading
                fig.add_shape(
                    type='rect',
                    x0=start_date, x1=end_date,
                    y0=y_min_price, y1=y_max_price,
                    fillcolor=regime_colors[regime],
                    line=dict(width=0),
                    layer='below',
                    row=1, col=1
                )
                
                # Volume panel shading
                fig.add_shape(
                    type='rect',
                    x0=start_date, x1=end_date,
                    y0=0, y1=y_max_vol,
                    fillcolor=regime_colors[regime],
                    line=dict(width=0),
                    layer='below',
                    row=2, col=1
                )
        
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
        colors = ['#ef4444' if chart_df['Close'].iloc[i] >= chart_df['Open'].iloc[i] else '#22c55e' 
                  for i in range(len(chart_df))]
        
        fig.add_trace(go.Bar(
            x=dates,
            y=chart_df['Volume'],
            name='',
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
        
        fig.update_yaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="", row=2, col=1)
        
        # Show only 5 dates, horizontal
        fig.update_xaxes(
            type='category',
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=0,
            row=2, col=1
        )
        
        # Hide ticks on top panel
        fig.update_xaxes(type='category', showticklabels=False, row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ 无法加载 CSI 300 数据")

# Market Breadth History
st.markdown("---")
st.subheader("📊 Market Breadth History - 超20天均线")

breadth_history = v2hist[v2hist['Sector'] != "MARKET_PROXY"].copy()
breadth_history = breadth_history.sort_values('Date', ascending=False)

# Filter last 60 days of data
unique_dates = breadth_history['Date'].dt.date.unique()[:60]

if len(unique_dates) == 0:
    st.warning("No historical breadth data available")
else:
    # Pagination setup
    DAYS_PER_PAGE = 10
    total_pages = (len(unique_dates) + DAYS_PER_PAGE - 1) // DAYS_PER_PAGE
    
    if 'breadth_page' not in st.session_state:
        st.session_state.breadth_page = 0
    
    # Page navigation buttons
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("◀ Previous 10 Days", disabled=st.session_state.breadth_page >= total_pages - 1):
            st.session_state.breadth_page += 1
            st.rerun()
    
    with col2:
        start_idx = st.session_state.breadth_page * DAYS_PER_PAGE
        page_end = min(start_idx + DAYS_PER_PAGE, len(unique_dates))
        date_range = f"{unique_dates[page_end-1]} to {unique_dates[start_idx]}"
        st.markdown(f"<center><b>Page {st.session_state.breadth_page + 1} of {total_pages}</b><br>{date_range}</center>", 
                   unsafe_allow_html=True)
    
    with col3:
        if st.button("Next 10 Days ▶", disabled=st.session_state.breadth_page == 0):
            st.session_state.breadth_page -= 1
            st.rerun()
    
    # Get dates for current page
    end_idx = start_idx + DAYS_PER_PAGE
    page_dates = unique_dates[start_idx:end_idx]
    page_dates = page_dates[::-1]  # ✅ Reverse order: Latest dates on RIGHT

    
    # Build the breadth table
    breadth_data = []
    for sector in sorted(breadth_history['Sector'].unique()):
        row = {'Sector': sector}
        for date in page_dates:
            sector_date_data = breadth_history[(breadth_history['Sector'] == sector) & 
                                              (breadth_history['Date'].dt.date == date)]
            if not sector_date_data.empty:
                breadth_val = sector_date_data.iloc[0]['Market_Breadth']
                row[date.strftime('%m-%d')] = breadth_val
            else:
                row[date.strftime('%m-%d')] = None
        breadth_data.append(row)
    
    breadth_df = pd.DataFrame(breadth_data)
    
    # Apply styling
    def style_breadth_cell(val):
        if pd.isna(val):
            return ""
        if val < 0.5:
            return "color: #15803d; background-color: #dcfce7; font-weight: 600"
        else:
            return "color: #b91c1c; background-color: #fee2e2; font-weight: 600"
    
    date_cols = [col for col in breadth_df.columns if col != 'Sector']
    
    # Format breadth values as percentages
    def format_breadth(val):
        if pd.isna(val):
            return ""
        return f"{val*100:.0f}%"
    
    styled = breadth_df.style.map(style_breadth_cell, subset=date_cols).format(format_breadth, subset=date_cols)
    
    st.dataframe(styled, hide_index=True, use_container_width=True, height=600)
    st.caption("🟢 Green: <50% (Most stocks below MA20 = opportunity). 🔴 Red: >50% (Most stocks above MA20 = extended).")

# ============================================================================
# SECTOR ROTATION DETECTION MODULE (DAILY + ADJUSTABLE ROLLING WINDOW)
# Add this at the bottom of your sector_dashboard.py
# ============================================================================

st.markdown("---")
st.subheader("🔄 Sector Rotation Detection")

# Define rotation pairs
ROTATION_PAIRS = {
    'Cyclical vs Defensive': {
        'cyclical': '399395.SZ',  # 国证消费 CNI Consumer
        'defensive': '399396.SZ',  # 国证食品 CNI Food & Beverage
        'cyclical_name': '消费',
        'defensive_name': '食品饮料'
    },
    'Tech vs Utilities': {
        'cyclical': '399932.SZ',  # 中证信息 CSI Info Tech
        'defensive': '000991.SH',  # 全指公用 CSI Utilities
        'cyclical_name': '信息技术',
        'defensive_name': '公用事业'
    },
    'Financial vs Industrial': {
        'cyclical': '399975.SZ',  # 证券公司 CSI Securities
        'defensive': '000993.SH',  # 全指工业 CSI Industrials
        'cyclical_name': '证券',
        'defensive_name': '工业'
    },
    'Healthcare vs Energy': {
        'cyclical': '399989.SZ',  # 中证医疗 CSI Healthcare
        'defensive': '000992.SH',  # 全指能源 CSI Energy
        'cyclical_name': '医疗',
        'defensive_name': '能源'
    }
}

# ============================================================================
# ADJUSTABLE ROLLING WINDOW SELECTOR
# ============================================================================
col_select1, col_select2 = st.columns([1, 3])

with col_select1:
    rolling_window = st.selectbox(
        "Rolling Window (Days):",
        options=[5, 10, 15, 30],
        index=1,  # Default to 10 days
        key="rotation_rolling_window"
    )

with col_select2:
    st.caption(f"Using {rolling_window}-day rolling correlation to measure sector rotation dynamics")

def calculate_rotation_metrics_daily(ts_code1, ts_code2, rolling_days=10, lookback_days=400):
    """
    Calculate rotation metrics between two indices using DAILY data
    Returns: correlation, ratio_change, status, correlation_history
    """
    import tushare as ts
    from datetime import datetime, timedelta

    try:
        pro = ts.pro_api()

        # Get daily data for both indices (1+ year)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y%m%d')

        df1 = pro.index_daily(ts_code=ts_code1, start_date=start_date, end_date=end_date)
        df2 = pro.index_daily(ts_code=ts_code2, start_date=start_date, end_date=end_date)

        if df1 is None or df2 is None or df1.empty or df2.empty:
            return None, None, "数据不足", None

        # Sort by date and calculate returns
        df1 = df1.sort_values('trade_date')
        df2 = df2.sort_values('trade_date')

        df1['returns'] = df1['close'].pct_change()
        df2['returns'] = df2['close'].pct_change()

        # Merge data
        merged = pd.merge(df1[['trade_date', 'returns', 'close']], 
                         df2[['trade_date', 'returns', 'close']], 
                         on='trade_date', 
                         suffixes=('_1', '_2'))

        if len(merged) < rolling_days:
            return None, None, "数据不足", None

        # Calculate rolling correlation
        merged['correlation'] = merged['returns_1'].rolling(window=rolling_days).corr(merged['returns_2'])

        # Get latest correlation
        correlation = merged['returns_1'].tail(rolling_days).corr(merged['returns_2'].tail(rolling_days))

        # Get correlation history for chart (last 252 days ~ 1 year of trading days)
        correlation_history = merged[['trade_date', 'correlation']].tail(252).copy()

        # Calculate relative strength ratio change (last 60 days)
        ratio_start = merged['close_1'].iloc[-60] / merged['close_2'].iloc[-60] if len(merged) >= 60 else merged['close_1'].iloc[0] / merged['close_2'].iloc[0]
        ratio_end = merged['close_1'].iloc[-1] / merged['close_2'].iloc[-1]
        ratio_change = ((ratio_end - ratio_start) / ratio_start) * 100

        # Determine rotation status
        if correlation < 0.3:
            status = "🔴 高度轮动"
        elif correlation < 0.5:
            status = "🟡 中度轮动"
        elif correlation < 0.7:
            status = "🟢 低度轮动"
        else:
            status = "⚪ 同步移动"

        return correlation, ratio_change, status, correlation_history

    except Exception as e:
        return None, None, f"错误: {str(e)}", None


# Calculate metrics for all pairs
rotation_results = []
correlation_histories = {}

with st.spinner(f"计算板块轮动指标（{rolling_window}日滚动相关系数）..."):
    for pair_name, pair_info in ROTATION_PAIRS.items():
        correlation, ratio_change, status, corr_history = calculate_rotation_metrics_daily(
            pair_info['cyclical'],
            pair_info['defensive'],
            rolling_days=rolling_window,
            lookback_days=400  # Get ~1+ year of data
        )

        # Store correlation history for chart
        if corr_history is not None:
            correlation_histories[pair_name] = corr_history

        # Determine which is leading
        if ratio_change is not None:
            if ratio_change > 5:
                leader = f"➡️ {pair_info['cyclical_name']} 强"
            elif ratio_change < -5:
                leader = f"⬅️ {pair_info['defensive_name']} 强"
            else:
                leader = "⚖️ 均衡"
        else:
            leader = "N/A"

        rotation_results.append({
            '板块对': pair_name,
            '周期/防御': f"{pair_info['cyclical_name']} vs {pair_info['defensive_name']}",
            '相关系数': f"{correlation:.2f}" if correlation is not None else "N/A",
            '轮动状态': status,
            '相对强度': f"{ratio_change:+.1f}%" if ratio_change is not None else "N/A",
            '领先板块': leader
        })

# Display results table
rotation_df = pd.DataFrame(rotation_results)

# Style the table
def style_rotation_status(val):
    if "高度轮动" in val:
        return "color: #b91c1c; background-color: #fee2e2; font-weight: 600"
    elif "中度轮动" in val:
        return "color: #d97706; background-color: #fef3c7; font-weight: 600"
    elif "低度轮动" in val:
        return "color: #15803d; background-color: #dcfce7; font-weight: 600"
    elif "同步移动" in val:
        return "color: #6b7280; background-color: #f3f4f6; font-weight: 600"
    return ""

def style_correlation(val):
    if val == "N/A":
        return ""
    try:
        corr = float(val)
        if corr < 0.3:
            return "color: #b91c1c; font-weight: 600"
        elif corr < 0.5:
            return "color: #d97706; font-weight: 600"
        elif corr < 0.7:
            return "color: #15803d; font-weight: 600"
        else:
            return "color: #6b7280; font-weight: 600"
    except:
        return ""

styled_rotation = rotation_df.style.map(style_rotation_status, subset=['轮动状态']).map(style_correlation, subset=['相关系数'])

st.dataframe(styled_rotation, hide_index=True, use_container_width=True, height=220)

# ============================================================================
# CORRELATION CHART VISUALIZATION (4 pairs, 1 year of data)
# ============================================================================

if correlation_histories:
    st.markdown("---")
    st.subheader(f"📈 Correlation Trends ({rolling_window}-Day Rolling - Last Year)")

    # Create plotly figure with 4 subplots (2x2 grid)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(correlation_histories.keys()),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Color mapping for correlation levels
    def get_color(corr_val):
        if pd.isna(corr_val):
            return '#9ca3af'
        if corr_val < 0.3:
            return '#ef4444'  # Red - High rotation
        elif corr_val < 0.5:
            return '#f59e0b'  # Orange - Medium rotation
        elif corr_val < 0.7:
            return '#10b981'  # Green - Low rotation
        else:
            return '#6b7280'  # Gray - Moving together

    # Plot each pair
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, (pair_name, corr_history) in enumerate(correlation_histories.items()):
        row, col = positions[idx]

        # Prepare data - show data every ~5 days to avoid overcrowding
        total_points = len(corr_history)
        step = max(1, total_points // 50)  # Show ~50 bars max

        sampled_history = corr_history.iloc[::step].copy()

        dates = pd.to_datetime(sampled_history['trade_date']).dt.strftime('%Y-%m-%d').tolist()
        correlations = sampled_history['correlation'].tolist()

        # Get colors for each bar
        colors = [get_color(c) for c in correlations]

        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=dates,
                y=correlations,
                marker_color=colors,
                name=pair_name,
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Correlation: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )

        # Add horizontal reference lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", line_width=1, 
                     opacity=0.3, row=row, col=col)
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", line_width=1, 
                     opacity=0.3, row=row, col=col)
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", line_width=1, 
                     opacity=0.3, row=row, col=col)

        # Update y-axis range
        fig.update_yaxes(range=[-0.2, 1.0], row=row, col=col)

    # Update layout
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # Rotate x-axis labels and reduce font size
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=8))

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown(f"""
    **📊 Correlation Levels ({rolling_window}-day rolling):**
    🔴 < 0.3 = High Rotation | 🟡 0.3-0.5 = Medium Rotation | 🟢 0.5-0.7 = Low Rotation | ⚪ > 0.7 = Moving Together
    """)

# ============================================================================
# MARKET CONCLUSION
# ============================================================================

# Calculate overall rotation intensity
correlations = [float(r['相关系数']) for r in rotation_results if r['相关系数'] != "N/A"]
if correlations:
    avg_correlation = sum(correlations) / len(correlations)
    high_rotation_count = sum(1 for c in correlations if c < 0.3)

    st.markdown("---")

    if avg_correlation < 0.4:
        market_status = "🔴 **市场处于高轮动期**"
        interpretation = "各板块走势分化明显，建议精选优势板块，避免弱势板块。"
    elif avg_correlation < 0.6:
        market_status = "🟡 **市场处于中度轮动期**"
        interpretation = "板块有一定分化，存在轮动机会，可考虑配置多个板块。"
    else:
        market_status = "🟢 **市场同步移动**"
        interpretation = "各板块走势趋同，市场趋势明确，建议跟随大盘方向。"

    st.markdown(f"### {market_status}")
    st.info(f"📊 **平均相关系数**: {avg_correlation:.2f} | **高轮动对数**: {high_rotation_count}/4\n\n{interpretation}")

    # Additional insights
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💡 轮动状态说明**")
        st.markdown("""
        - 🔴 **高度轮动** (相关系数 < 0.3): 板块严重分化
        - 🟡 **中度轮动** (0.3-0.5): 板块有所分化
        - 🟢 **低度轮动** (0.5-0.7): 板块轻微分化
        - ⚪ **同步移动** (> 0.7): 板块走势一致
        """)

    with col2:
        st.markdown("**🎯 投资建议**")
        if avg_correlation < 0.4:
            st.markdown("""
            - ✅ 精选领先板块，集中投资
            - ✅ 避免落后板块
            - ✅ 灵活调仓，跟随轮动节奏
            """)
        elif avg_correlation < 0.6:
            st.markdown("""
            - ✅ 均衡配置多个板块
            - ✅ 关注轮动机会
            - ✅ 适度分散风险
            """)
        else:
            st.markdown("""
            - ✅ 跟随市场整体方向
            - ✅ 配置指数型基金
            - ✅ 减少频繁调仓
            """)

st.caption(f"📅 数据基于过去一年日度数据 | 相关系数基于{rolling_window}日滚动计算 | 相对强度基于近60日变化")