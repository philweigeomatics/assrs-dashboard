import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import ta

import data_manager  # your data access module

# --- 1. CONFIGURATION ---
V1_RULES_FILE = 'assrs_backtest_results_SECTORS_V1_Rules.csv'
V2_REGIME_FILE = 'assrs_backtest_results_SECTORS_V2_Regime.csv'

# Set page config
st.set_page_config(
    page_title="ASSRS Sector Scoreboard",
    layout="wide"
)

# --- 2. HELPER FUNCTIONS (SECTOR DATA) ---

@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_data(filepath, model_name):
    """Loads and prepares all data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert all potential numeric columns
        cols_to_numeric = ['TOTAL_SCORE', 'Open', 'High', 'Low', 'Close', 'Volume_Metric']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.empty:
            return None, None, f"No data in {model_name} file."

        # 1. Get Latest Scores
        latest_date = df['Date'].max()
        latest_scores_df = df[df['Date'] == latest_date].copy()
        
        # 2. Get Full History
        full_history_df = df.copy()
        
        return latest_scores_df, full_history_df, latest_date.strftime('%Y-%m-%d'), None
        
    except FileNotFoundError:
        return None, None, None, f"ERROR: File not found: {filepath}. The data-update task may not have run yet."
    except Exception as e:
        return None, None, None, f"An error occurred: {str(e)}"

def style_action(action):
    """Applies color to the 'ACTION' column for the dataframe."""
    if 'GREEN' in action:
        return 'color: #15803d; background-color: #dcfce7; font-weight: 600;'
    if 'YELLOW' in action:
        return 'color: #a16207; background-color: #fef9c3; font-weight: 600;'
    if 'RED' in action:
        return 'color: #b91c1c; background-color: #fee2e2; font-weight: 600;'
    # Handle CONSOLIDATION or other states
    if 'CONSOLIDATION' in action:
        return 'color: #4b5563; background-color: #f3f4f6; font-weight: 500;'
    return ''

def create_drilldown_chart(chart_data, model_type):
    """
    Creates a 3-plot interactive chart:
    1. Price (Candlestick)
    2. Volume (Bar)
    3. Score (Line)
    """
    is_v1 = model_type == 'v1'
    y_title_score = 'V1 Score (-3 to 8)' if is_v1 else 'V2 Bull Probability (0 to 1)'
    y_range_score = [-3.1, 8.1] if is_v1 else [-0.1, 1.1]

    date_strings = chart_data['Date'].dt.strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price (PPI)', 'Volume (Z-Score)', 'Signal (Score)'),
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
        increasing=dict(line=dict(color='#b91c1c')),
        decreasing=dict(line=dict(color='#15803d'))
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=date_strings,
        y=chart_data['Volume_Metric'],
        name='Volume Metric',
        marker_color='rgba(107, 114, 128, 0.3)'
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
    if is_v1:
        fig.add_hline(y=2.5, line_dash="dash", line_color="#a16207",
                      annotation_text="Buy Threshold (2.5)", row=3, col=1)
        fig.add_hline(y=5.0, line_dash="dash", line_color="#15803d",
                      annotation_text="Green Threshold (5.0)", row=3, col=1)
    else:
        fig.add_hline(y=0.8, line_dash="dash", line_color="#15803d",
                      annotation_text="Green Threshold (0.8)", row=3, col=1)
        fig.add_hline(y=0.2, line_dash="dash", line_color="#b91c1c",
                      annotation_text="Red Threshold (0.2)", row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_type='category',
        xaxis2_type='category',
        xaxis3_type='category',
        xaxis_rangeslider_visible=False,
        xaxis_showticklabels=False,
        xaxis2_showticklabels=False,
        xaxis3_title='Date',
        yaxis1_title="PPI (Base 100)",
        yaxis2_title="Volume (Z-Score)",
        yaxis3_title=y_title_score,
        yaxis3_range=y_range_score
    )
    return fig

# --- 3. SINGLE STOCK ANALYSIS LOGIC (ported from webapp.py) ---

def run_single_stock_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds indicators and signals for a single stock.
    df is expected to come from data_manager.get_single_stock_data(...)
    with Date index and columns: Open, High, Low, Close, Volume.
    """
    df_analysis = df.copy()

    if not isinstance(df_analysis.index, pd.DatetimeIndex):
        df_analysis.index = pd.to_datetime(df_analysis.index)
    df_analysis = df_analysis.sort_index()

    # Moving averages
    df_analysis['MA20'] = ta.trend.sma_indicator(df_analysis['Close'], window=20)
    df_analysis['MA50'] = ta.trend.sma_indicator(df_analysis['Close'], window=50)
    df_analysis['MA200'] = ta.trend.sma_indicator(df_analysis['Close'], window=200)

    # MA(50) today vs 5 days ago
    df_analysis['MA50_Slope'] = df_analysis['MA50'] - df_analysis['MA50'].shift(5)

    # Uptrend filter: Close > MA20 > MA50 > MA200 and MA50 rising over last 5 days
    df_analysis['Uptrend_Filter'] = (
        (df_analysis['Close'] > df_analysis['MA20']) &
        (df_analysis['MA20'] > df_analysis['MA50']) &
        (df_analysis['MA50'] > df_analysis['MA200']) &
        (df_analysis['MA50_Slope'] > 0)
    )

    # Breakout logic
    df_analysis['20d_High'] = df_analysis['High'].rolling(window=20).max().shift(1)
    df_analysis['20d_Avg_Vol'] = df_analysis['Volume'].rolling(window=20).mean()

    df_analysis['Breakout_Signal'] = (
        df_analysis['Uptrend_Filter'] &
        (df_analysis['Close'] > df_analysis['20d_High']) &
        (df_analysis['Volume'] > (df_analysis['20d_Avg_Vol'] * 1.5))
    )

    # Pullback logic
    df_analysis['RSI_14'] = ta.momentum.rsi(df_analysis['Close'], window=14)
    df_analysis['RSI_Rising'] = df_analysis['RSI_14'].diff() > 0
    df_analysis['Near_MA20'] = (
        (df_analysis['Close'] > df_analysis['MA20']) &
        (df_analysis['Close'] < df_analysis['MA20'] * 1.03)
    )
    df_analysis['Near_MA50'] = (
        (df_analysis['Close'] > df_analysis['MA50']) &
        (df_analysis['Close'] < df_analysis['MA50'] * 1.03)
    )

    df_analysis['Pullback_Signal'] = (
        df_analysis['Uptrend_Filter'] &
        df_analysis['RSI_14'].between(40, 50) &
        df_analysis['RSI_Rising'] &
        (df_analysis['Near_MA20'] | df_analysis['Near_MA50'])
    )

    return df_analysis

def create_single_stock_chart(analysis_df: pd.DataFrame, window: int = 250) -> go.Figure:
    """
    Build a 3-panel Plotly chart for single stock:
    1) Price + MAs + signals
    2) Volume
    3) RSI
    """
    df = analysis_df.tail(window).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    date_strings = df.index.strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price / Trend / Signals', 'Volume', 'RSI (14)'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Price candlestick
    fig.add_trace(go.Candlestick(
        x=date_strings,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing=dict(line=dict(color='#b91c1c')),  # red up
        decreasing=dict(line=dict(color='#15803d'))   # green down
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=date_strings, y=df['MA20'], name='MA20',
        mode='lines', line=dict(width=1, dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=date_strings, y=df['MA50'], name='MA50',
        mode='lines', line=dict(width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=date_strings, y=df['MA200'], name='MA200',
        mode='lines', line=dict(width=2)
    ), row=1, col=1)

    # Breakout & Pullback markers
    breakout_days = df[df['Breakout_Signal']]
    pullback_days = df[df['Pullback_Signal']]

    fig.add_trace(go.Scatter(
        x=breakout_days.index.strftime('%Y-%m-%d'),
        y=breakout_days['High'],
        mode='markers',
        name='Breakout (A)',
        marker=dict(color='green', symbol='triangle-up', size=10)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pullback_days.index.strftime('%Y-%m-%d'),
        y=pullback_days['Low'] * 0.98,
        mode='markers',
        name='Pullback (B)',
        marker=dict(color='blue', symbol='circle', size=8)
    ), row=1, col=1)

    # Uptrend shading (top panel)
    shapes = []
    if 'Uptrend_Filter' in df.columns:
        uptrend_idx = df[df['Uptrend_Filter']].index
        for d in uptrend_idx:
            ds = d.strftime('%Y-%m-%d')
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=ds, x1=ds,
                y0=0.66, y1=1.0,
                fillcolor="#dcfce7",
                opacity=0.25,
                line_width=0
            ))
    # Volume panel
    fig.add_trace(go.Bar(
        x=date_strings,
        y=df['Volume'],
        name='Volume'
    ), row=2, col=1)

    if '20d_Avg_Vol' in df.columns:
        fig.add_trace(go.Scatter(
            x=date_strings,
            y=df['20d_Avg_Vol'],
            mode='lines',
            name='20d Avg Vol',
            line=dict(width=1)
        ), row=2, col=1)

    # RSI panel
    fig.add_trace(go.Scatter(
        x=date_strings,
        y=df['RSI_14'],
        mode='lines',
        name='RSI (14)',
        line=dict(width=2)
    ), row=3, col=1)

    fig.add_hline(y=40, line_dash="dash", line_color="grey", row=3, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="grey", row=3, col=1)

    fig.update_layout(
        height=900,
        showlegend=True,
        margin=dict(l=60, r=20, t=40, b=80),
        shapes=shapes,
        xaxis=dict(type='category', showticklabels=False, rangeslider=dict(visible=False)),
        xaxis2=dict(type='category', showticklabels=False),
        xaxis3=dict(
            type='category',
            title='Date',
            tickmode='auto',
            nticks=10,
            tickangle=-45
        ),
        yaxis=dict(domain=[0.66, 1.0], title="Price"),
        yaxis2=dict(domain=[0.33, 0.64], title="Volume"),
        yaxis3=dict(domain=[0.0, 0.31], title="RSI (14)", range=[0, 100])
    )

    return fig

@st.cache_data(ttl=600)
def load_single_stock(ticker: str):
    """Cached wrapper around data_manager.get_single_stock_data."""
    return data_manager.get_single_stock_data(ticker)

# --- 4. MAIN APP LAYOUT ---

st.title("ASSRS Sector Rotation & Single Stock Analysis")

# Tabs = separate “pages”
tab_dashboard, tab_single = st.tabs(["Sector Dashboard", "Single Stock Analysis"])

# ========= TAB 1: Sector Dashboard =========
with tab_dashboard:
    # --- Load Sector Data ---
    v1_latest, v1_hist, v1_date, v1_error = load_data(V1_RULES_FILE, "V1")
    v2_latest, v2_hist, v2_date, v2_error = load_data(V2_REGIME_FILE, "V2")

    st.markdown("### Sector Rotation Scoreboard")

    # --- Create 2-Column Layout ---
    col1, col2 = st.columns(2)

    # --- V1 (Rule-Based) Scorecard ---
    with col1:
        st.header("V1: Rule-Based Scorecard (8-Point)")
        if v1_latest is not None:
            st.caption(f"Last Updated: {v1_date}")
            
            v1_display_df = v1_latest[['Sector', 'TOTAL_SCORE', 'ACTION']].sort_values(by='TOTAL_SCORE', ascending=False)
            styled_v1_df = v1_display_df.style.map(style_action, subset=['ACTION'])
            
            st.dataframe(
                styled_v1_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "TOTAL_SCORE": st.column_config.NumberColumn(format="%.2f"),
                    "ACTION": st.column_config.TextColumn(width="medium")
                }
            )
            
            # --- V1 Charting ---
            v1_sector_to_chart = st.selectbox(
                "Select V1 Sector to Chart:",
                v1_hist['Sector'].unique(),
                key="v1_selector"
            )
            
            if v1_sector_to_chart:
                chart_data = v1_hist[v1_hist['Sector'] == v1_sector_to_chart]
                fig = create_drilldown_chart(chart_data, model_type='v1')
                st.plotly_chart(fig, use_container_width=True, key="v1_chart")

        else:
            st.error(v1_error)

    # --- V2 (Regime-Switching) Scorecard ---
    with col2:
        st.header("V2: Regime-Switching Model")
        if v2_latest is not None:
            st.caption(f"Last Updated: {v2_date}")
            
            v2_display_df = v2_latest[['Sector', 'TOTAL_SCORE', 'ACTION']].copy()
            v2_display_df = v2_display_df.sort_values(by='TOTAL_SCORE', ascending=False)
            v2_display_df['TOTAL_SCORE'] = (v2_display_df['TOTAL_SCORE'] * 100).map('{:.0f}%'.format)
            styled_v2_df = v2_display_df.style.map(style_action, subset=['ACTION'])
            
            st.dataframe(
                styled_v2_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "TOTAL_SCORE": "Bull Probability",
                    "ACTION": st.column_config.TextColumn(width="medium")
                }
            )
            
            # --- V2 Charting ---
            v2_sector_to_chart = st.selectbox(
                "Select V2 Sector to Chart:",
                v2_hist['Sector'].unique(),
                key="v2_selector"
            )
            
            if v2_sector_to_chart:
                chart_data = v2_hist[v2_hist['Sector'] == v2_sector_to_chart]
                fig = create_drilldown_chart(chart_data, model_type='v2')
                st.plotly_chart(fig, use_container_width=True, key="v2_chart")
                
        else:
            st.error(v2_error)


# ========= TAB 2: Single Stock Analysis =========
with tab_single:
    st.markdown("### Single Stock Analysis")

    c1, c2 = st.columns([3, 1])
    with c1:
        ticker = st.text_input("Enter Stock Code (e.g., 600760)", key="single_ticker")
    with c2:
        analyze_clicked = st.button("Analyze", key="single_analyze")

    if analyze_clicked and ticker:
        stock_df = load_single_stock(ticker.strip())
        if stock_df is None or stock_df.empty:
            st.error(
                "No data found for this ticker in the local database, "
                "and fetching from Tushare also failed. "
                "Check that the ticker is valid and that Tushare is configured correctly."
            )
        else:
            analysis_df = run_single_stock_analysis(stock_df)

            # latest valid RSI row
            valid_rows = analysis_df.dropna(subset=['RSI_14'])
            if valid_rows.empty:
                st.error("Not enough data to compute RSI / MA signals for this stock.")
            else:
                latest_row = valid_rows.iloc[-1]

                # Chart
                fig_stock = create_single_stock_chart(analysis_df)
                st.plotly_chart(fig_stock, use_container_width=True)

                # Latest signals summary
                st.subheader("Latest Day's Signals")
                col_a, col_b, col_c, col_d = st.columns(4)

                uptrend_pass = bool(latest_row.get('Uptrend_Filter', False))
                breakout = bool(latest_row.get('Breakout_Signal', False))
                pullback = bool(latest_row.get('Pullback_Signal', False))
                rsi_val = float(latest_row.get('RSI_14', float('nan')))

                with col_a:
                    st.caption("Uptrend Filter")
                    st.markdown("**:green[PASS]**" if uptrend_pass else "**:red[FAIL]**")
                with col_b:
                    st.caption("Breakout Signal")
                    st.markdown("**:green[ACTIVE]**" if breakout else ":grey[---]")
                with col_c:
                    st.caption("Pullback Signal")
                    st.markdown("**:blue[ACTIVE]**" if pullback else ":grey[---]")
                with col_d:
                    st.caption("RSI (14)")
                    st.markdown(f"**{rsi_val:.1f}**")

                # Debug table (last 50 rows)
                st.subheader("Recent Data (last 50 days)")
                table_cols = [
                    'Close', 'MA20', 'MA50', 'MA200',
                    'RSI_14', 'Uptrend_Filter', 'Breakout_Signal', 'Pullback_Signal'
                ]
                df_for_table = analysis_df[table_cols].copy()
                df_for_table = df_for_table.reset_index()
                df_for_table['Date'] = df_for_table['Date'].dt.strftime('%Y-%m-%d')
                df_for_table = df_for_table.tail(50)
                st.dataframe(df_for_table, use_container_width=True, hide_index=True)
