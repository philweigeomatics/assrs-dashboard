"""
Sector Analysis - Dashboard
Shows overview of all sectors with key metrics and CSI 300 chart
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo
from datetime import datetime
import data_manager as dm

from sector_utils import (
    load_v2_data, 
    load_csi300_with_regime,
    create_sector_chart
)

import auth_manager
auth_manager.require_login()


# Load data
v2latest, v2hist, v2date, v2error = load_v2_data()
if v2latest is None:
    st.error(f"Error loading data: {v2error}")
    st.stop()
# ===================================================================
# SECTION 1: CSI 300 INDEX WITH VOLATILITY INDICATORS
# ===================================================================
st.subheader("🏦 CSI 300 指数")

# Frequency selector
freq = st.radio(
    "时间周期",
    ["日线", "周线"],
    key="csi300_freq",
    horizontal=True,
    label_visibility="collapsed"
)

with st.spinner("加载 CSI 300 数据..."):
    raw_df = load_csi300_with_regime(freq)

    if raw_df is not None and not raw_df.empty:
        if freq == "周线":
            chart_df = raw_df.tail(52).copy()  # 1 year weekly
            title = "CSI 300 指数 - 周K线 (52周)"
        else:
            chart_df = raw_df.tail(180).copy()  # 6 months daily
            title = "CSI 300 指数 - 日K线 (180天)"
    else:
        chart_df = None

if chart_df is not None and not chart_df.empty:
    # Show current regime
    if 'Market_Regime' in chart_df.columns and chart_df['Market_Regime'].notna().any():
        latest_regime = chart_df['Market_Regime'].dropna().iloc[-1]

        # Color-coded regime display
        if "Low" in latest_regime:
            st.success(f"✅ 当前波动状态: {latest_regime} (低波动)")
        elif "Normal" in latest_regime:
            st.info(f"ℹ️ 当前波动状态: {latest_regime} (正常波动)")
        elif "High" in latest_regime:
            st.warning(f"⚠️ 当前波动状态: {latest_regime} (高波动)")
        else:
            st.error(f"🔴 当前波动状态: {latest_regime} (极端波动)")

    # Prepare dates
    dates = chart_df.index.strftime('%Y-%m-%d').tolist()

    # Calculate tick spacing
    total_dates = len(dates)
    tick_interval = max(1, total_dates // 5)
    tick_vals = dates[::tick_interval][:5]
    tick_text = tick_vals

    # Create chart
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

        # Y ranges
        ymin_price = df_clean['Low'].min() * 0.98
        ymax_price = df_clean['High'].max() * 1.02
        ymax_vol = df_clean['Volume'].max() * 1.05

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
                type="rect",
                x0=start_date, x1=end_date,
                y0=ymin_price, y1=ymax_price,
                fillcolor=regime_colors[regime],
                line=dict(width=0),
                layer="below",
                row=1, col=1
            )

            # Volume panel shading
            fig.add_shape(
                type="rect",
                x0=start_date, x1=end_date,
                y0=0, y1=ymax_vol,
                fillcolor=regime_colors[regime],
                line=dict(width=0),
                layer="below",
                row=2, col=1
            )

    # Candlestick (Chinese style: red=up, green=down)
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='CSI 300',
            increasing_line_color='#ef4444',
            decreasing_line_color='#22c55e'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['#ef4444' if chart_df['Close'].iloc[i] >= chart_df['Open'].iloc[i] else '#22c55e' 
              for i in range(len(chart_df))]

    fig.add_trace(
        go.Bar(
            x=dates,
            y=chart_df['Volume'],
            name='成交量',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title=title,
        height=500,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=40)
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)

    # X-axis
    fig.update_xaxes(
        type='category',
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        row=2, col=1
    )
    fig.update_xaxes(type='category', showticklabels=False, row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("无法加载 CSI 300 数据")

# Market Breadth History
st.markdown("---")
st.subheader("📊 市场宽度历史 (Market Breadth History)")
st.caption("各板块中股价高于MA20的股票占比 - 数据来自数据库")

# Load breadth data from database (single query!)
breadth_df = dm.load_market_breadth_from_db()

if breadth_df is None or breadth_df.empty:
    st.warning("数据库中没有市场宽度数据")
else:
    # Get last 60 dates
    unique_dates = breadth_df.index.sort_values(ascending=False)[:60].tolist()

    if len(unique_dates) == 0:
        st.warning("没有历史宽度数据")
    else:
        # Pagination setup
        DAYS_PER_PAGE = 10
        total_pages = (len(unique_dates) + DAYS_PER_PAGE - 1) // DAYS_PER_PAGE

        if 'breadth_page' not in st.session_state:
            st.session_state.breadth_page = 0

        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.button("⬅ 前10天", disabled=(st.session_state.breadth_page >= total_pages - 1)):
                st.session_state.breadth_page += 1
                st.rerun()

        with col2:
            start_idx = st.session_state.breadth_page * DAYS_PER_PAGE
            page_end = min(start_idx + DAYS_PER_PAGE, len(unique_dates))
            date_range = f"{unique_dates[page_end-1].strftime('%m/%d')} 至 {unique_dates[start_idx].strftime('%m/%d')}"
            st.markdown(
                f"<center><b>第 {st.session_state.breadth_page + 1}/{total_pages} 页</b><br>{date_range}</center>",
                unsafe_allow_html=True
            )

        with col3:
            if st.button("后10天 ➡", disabled=(st.session_state.breadth_page == 0)):
                st.session_state.breadth_page -= 1
                st.rerun()

        # Get dates for current page
        end_idx = start_idx + DAYS_PER_PAGE
        page_dates = unique_dates[start_idx:end_idx]
        page_dates = page_dates[::-1]  # Reverse: Latest dates on RIGHT

        # Filter breadth_df for page dates
        page_df = breadth_df.loc[page_dates].copy()

        # Transpose so dates are columns, sectors are rows
        page_df = page_df.T
        page_df.columns = [d.strftime('%m/%d') for d in page_df.columns]
        page_df = page_df.reset_index()
        page_df = page_df.rename(columns={'index': '板块'})

        # Styling
        def style_breadth_cell(val):
            if pd.isna(val):
                return ''
            if val >= 0.5:
                return 'color: #b91c1c; background-color: #fee2e2; font-weight: 600'
            else:
                return 'color: #15803d; background-color: #dcfce7; font-weight: 600'

        date_cols = [col for col in page_df.columns if col != '板块']

        def format_breadth(val):
            if pd.isna(val):
                return ''
            return f'{val*100:.0f}%'

        styled = page_df.style            .map(style_breadth_cell, subset=date_cols)            .format(format_breadth, subset=date_cols)

        st.dataframe(styled, hide_index=True, use_container_width=True, height=600)
        st.caption("🟢 绿色 <50%: 多数股票低于MA20 (机会). 🔴 红色 >=50%: 多数股票高于MA20 (过热).")

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

        # ✅ FIX: Use Beijing time for end_date
        CHINA_TZ = ZoneInfo("Asia/Shanghai")
        beijing_now = datetime.now(CHINA_TZ)
        end_date = datetime.now(CHINA_TZ).strftime('%Y%m%d')
        start_date = (datetime.now(CHINA_TZ) - timedelta(days=lookback_days)).strftime('%Y%m%d')

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


        # Prepare data - show data every 5 days to avoid overcrowding
        # total_points = len(corr_history)
        # step = max(1, total_points // 50)  # Show 50 bars max
        # sampled_history = corr_history.iloc[::step].copy()

        # # NEW (CORRECT):
        # dates = pd.to_datetime(sampled_history['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d').tolist()

        dates = pd.to_datetime(corr_history['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d').tolist()
        correlations = corr_history['correlation'].tolist()


        # Get colors for each bar
        colors = [get_color(c) for c in correlations]

        # # Add bar chart
        # fig.add_trace(
        #     go.Bar(
        #         x=dates,
        #         y=correlations,
        #         marker_color=colors,
        #         name=pair_name,
        #         showlegend=False,
        #         hovertemplate='<b>%{x}</b><br>Correlation: %{y:.2f}<extra></extra>'
        #     ),
        #     row=row, col=col
        # )

        # Create line chart with gradient colors
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=correlations,
                mode='lines',
                fill='tozeroy',
                line=dict(color='rgb(59, 130, 246)', width=2),
                name=pair_name,
                showlegend=False,
                hovertemplate='<b>Date: %{x}</b><br>' + f'{rolling_window}-day Rolling Correlation: ' + '%{y:.4f}<br><extra></extra>'
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


# ============================================================================
# STATISTICAL WYCKOFF PHASE DETECTION (CSI 300)
# Append this to the bottom of sector_dashboard.py
# ============================================================================

st.markdown("---")
st.subheader("📊 Statistical Wyckoff Phase Detection (CSI 300)")
st.caption("A pure quantitative approach using 120-day rolling channels, return volatility, and volume Z-scores to mathematically define market regimes without moving average lag.")
import numpy as np

with st.spinner("Calculating Statistical Wyckoff Phases..."):
    # Fetch 500 days to ensure enough history for 120-day rolling windows and means
    wyckoff_stat_df = dm.get_index_data_live('000300.SH', lookback_days=500, freq='daily')

    if wyckoff_stat_df is not None and not wyckoff_stat_df.empty:
        df = wyckoff_stat_df.copy()
        
        lookback = 120 # Approx 6 months of trading days
        
        # 1. Market Structure: Donchian Channels & Range Positioning
        df['Max_120'] = df['High'].rolling(window=lookback).max()
        df['Min_120'] = df['Low'].rolling(window=lookback).min()
        df['Range_120'] = df['Max_120'] - df['Min_120']
        # Where is the price currently sitting within its 6-month range? (0.0 to 1.0)
        df['Position_120'] = (df['Close'] - df['Min_120']) / df['Range_120']
        
        # 2. Volatility: Rolling 20-day standard deviation vs historical baseline
        df['Returns'] = df['Close'].pct_change()
        df['Volat_20'] = df['Returns'].rolling(window=20).std()
        df['Volat_Baseline'] = df['Volat_20'].rolling(window=lookback).mean()
        
        # 3. Clean Volume Metric (Isolated for this module)
        df['Vol_Mean_60'] = df['Volume'].rolling(window=60).mean()
        df['Vol_Std_60'] = df['Volume'].rolling(window=60).std()
        df['Vol_Z'] = (df['Volume'] - df['Vol_Mean_60']) / df['Vol_Std_60']
        
        # 4. Statistical Phase Conditions
        conditions = [
            # MARKUP: Upper quartile of 6-month range, trending up (Close > 20-day mean)
            (df['Position_120'] > 0.75) & (df['Close'] > df['Close'].rolling(20).mean()),
            
            # MARKDOWN: Lower quartile of 6-month range, trending down
            (df['Position_120'] < 0.25) & (df['Close'] < df['Close'].rolling(20).mean()),
            
            # DISTRIBUTION: Upper half of range, but high volatility (churn) and failing to push new highs
            (df['Position_120'] >= 0.5) & (df['Volat_20'] > df['Volat_Baseline']),
            
            # ACCUMULATION: Lower half of range, low volatility (quiet absorption)
            (df['Position_120'] < 0.5) & (df['Volat_20'] <= df['Volat_Baseline'])
        ]
        
        choices = ['Markup', 'Markdown', 'Distribution', 'Accumulation']
        df['Wyckoff_Phase'] = np.select(conditions, choices, default='Transition')
        
        # Drop rows where our longest baseline hasn't calculated yet
        plot_df = df.dropna(subset=['Volat_Baseline']).copy()
        
        # Focus chart on the last 180 trading days
        plot_df = plot_df.tail(180)

        if not plot_df.empty:
            current_phase = plot_df['Wyckoff_Phase'].iloc[-1]
            current_pos = plot_df['Position_120'].iloc[-1]
            current_vol_z = plot_df['Vol_Z'].iloc[-1]
            
            # Display current metrics
            col_s1, col_s2, col_s3 = st.columns([1.5, 1, 1])
            with col_s1:
                st.metric("Current Phase (Statistical)", current_phase)
            with col_s2:
                st.metric("Range Position (120d)", f"{current_pos * 100:.1f}%")
            with col_s3:
                st.metric("Volume Z-Score", f"{current_vol_z:.2f}")

            # Contextual explainer
            if current_phase == 'Accumulation':
                st.info("🟦 **Accumulation:** Price is compressed in the lower half of its 6-month range. Volatility is below average, indicating quiet institutional absorption.")
            elif current_phase == 'Markup':
                st.success("🟩 **Markup:** Price has broken into the top 25% of its 6-month range. Upward momentum is statistically confirmed.")
            elif current_phase == 'Distribution':
                st.warning("🟧 **Distribution:** Price is high, but volatility is spiking above baseline. High churn indicates potential institutional selling.")
            elif current_phase == 'Markdown':
                st.error("🟥 **Markdown:** Price has collapsed into the bottom 25% of its range. Downward momentum is dominating.")
            else:
                st.info("⬜ **Transition:** The market is caught between defined statistical states.")

            # Create Plotly Chart
            fig_stat = go.Figure()

            # Phase colors
            phase_colors = {
                'Accumulation': 'rgba(59, 130, 246, 0.1)',
                'Markup': 'rgba(34, 197, 94, 0.1)',
                'Distribution': 'rgba(245, 158, 11, 0.1)',
                'Markdown': 'rgba(239, 68, 68, 0.1)',
                'Transition': 'rgba(156, 163, 175, 0.1)'
            }

            # Add background shading
            changes = plot_df['Wyckoff_Phase'].ne(plot_df['Wyckoff_Phase'].shift())
            change_indices = plot_df.index[changes].tolist()
            
            if len(change_indices) == 0 or change_indices[0] != plot_df.index[0]:
                change_indices.insert(0, plot_df.index[0])

            ymin = plot_df['Low'].min() * 0.95
            ymax = plot_df['High'].max() * 1.05

            dates_str = plot_df.index.strftime('%Y-%m-%d').tolist()

            for i in range(len(change_indices)):
                start_idx = change_indices[i]
                end_idx = change_indices[i + 1] if i + 1 < len(change_indices) else plot_df.index[-1]
                phase = plot_df.loc[start_idx, 'Wyckoff_Phase']

                if phase in phase_colors:
                    fig_stat.add_shape(
                        type="rect",
                        x0=start_idx.strftime('%Y-%m-%d'), 
                        x1=end_idx.strftime('%Y-%m-%d'),
                        y0=ymin, y1=ymax,
                        fillcolor=phase_colors[phase],
                        line=dict(width=0),
                        layer="below"
                    )

            # Candlestick Trace
            fig_stat.add_trace(go.Candlestick(
                x=dates_str,
                open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'],
                name='CSI 300',
                increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
            ))

            # 120-Day Donchian Channels (Max/Min)
            fig_stat.add_trace(go.Scatter(
                x=dates_str, y=plot_df['Max_120'],
                mode='lines', name='120d High',
                line=dict(color='rgba(156, 163, 175, 0.6)', width=1, dash='dash')
            ))
            
            fig_stat.add_trace(go.Scatter(
                x=dates_str, y=plot_df['Min_120'],
                mode='lines', name='120d Low',
                line=dict(color='rgba(156, 163, 175, 0.6)', width=1, dash='dash')
            ))

            # Layout updates
            fig_stat.update_layout(
                title='CSI 300 - Statistical Regimes & 120-Day Channels',
                height=500,
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Fix x-axis categorical spacing for weekends
            tick_interval = max(1, len(dates_str) // 6)
            fig_stat.update_xaxes(
                type='category',
                tickmode='array',
                tickvals=dates_str[::tick_interval],
                ticktext=dates_str[::tick_interval]
            )

            st.plotly_chart(fig_stat, use_container_width=True)

    else:
        st.error("Failed to load sufficient data for Statistical Wyckoff calculation.")