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
    st.subheader("📈 CSI 300")
    
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

# # Sector Deep Dive
# st.markdown("---")
# st.subheader("🔍 Sector Deep Dive")

# sector = st.selectbox("Select Sector", sorted(v2hist['Sector'].unique()))

# if sector:
#     data = v2hist[v2hist['Sector'] == sector]
#     fig = create_sector_chart(data)
#     st.plotly_chart(fig, use_container_width=True)
