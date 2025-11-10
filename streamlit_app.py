import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. CONFIGURATION ---
V1_RULES_FILE = 'assrs_backtest_results_SECTORS_V1_Rules.csv'
V2_REGIME_FILE = 'assrs_backtest_results_SECTORS_V2_Regime.csv'

# Set page config
st.set_page_config(
    page_title="ASSRS Sector Scoreboard",
    layout="wide"
)

# --- 2. HELPER FUNCTIONS ---

@st.cache_data(ttl=600) # Cache data for 10 minutes
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

# ---!!!--- NEW: UNIFIED PLOTLY CHART FUNCTION ---!!!---

def create_drilldown_chart(chart_data, model_type):
    """
    Creates a single, unified, 3-plot interactive chart:
    1. Price (Candlestick)
    2. Volume (Bar)
    3. Score (Line)
    """
    
    is_v1 = model_type == 'v1'
    y_title_score = 'V1 Score (-3 to 8)' if is_v1 else 'V2 Bull Probability (0 to 1)'
    y_range_score = [-3.1, 8.1] if is_v1 else [-0.1, 1.1]

    # Create figure with 3 rows, sharing the X-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=('Price (PPI)', 'Volume (Z-Score)', 'Signal (Score)'), 
                       row_heights=[0.5, 0.2, 0.3]) # Give Price the most space

    # Plot 1: Candlestick
    fig.add_trace(go.Candlestick(x=chart_data['Date'], # <-- Use datetime object
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name='Price'), row=1, col=1)

    # Plot 2: Volume
    fig.add_trace(go.Bar(x=chart_data['Date'], y=chart_data['Volume_Metric'], # <-- Use datetime object
                         name='Volume Metric', marker_color='rgba(107, 114, 128, 0.3)'), row=2, col=1)

    # Plot 3: Score
    fig.add_trace(go.Scatter(
        x=chart_data['Date'], # <-- Use datetime object
        y=chart_data['TOTAL_SCORE'],
        name='Score',
        line=dict(color='#10b981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ), row=3, col=1)

    # Add Threshold lines for V1
    if is_v1:
        fig.add_hline(y=2.5, line_dash="dash", line_color="#a16207", 
                      annotation_text="Buy Threshold (2.5)", row=3, col=1)
        fig.add_hline(y=5.0, line_dash="dash", line_color="#15803d", 
                      annotation_text="Green Threshold (5.0)", row=3, col=1)
    else: # Add Threshold lines for V2
        fig.add_hline(y=0.8, line_dash="dash", line_color="#15803d", 
                      annotation_text="Green Threshold (0.8)", row=3, col=1)
        fig.add_hline(y=0.2, line_dash="dash", line_color="#b91c1c", 
                      annotation_text="Red Threshold (0.2)", row=3, col=1)

    # Update layout
    fig.update_layout(
        height=700, # Taller chart to fit all 3 plots
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        
        # ---!!!--- FIX: Remove weekend/holiday gaps ---!!!---
        # This tells Plotly to use a 'date' axis but to *only* plot
        # the dates for which we have data, removing gaps.
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='date'),
        xaxis2=dict(type='date'),
        xaxis3=dict(type='date', title='Date'),
        
        # Remove redundant x-axis labels
        xaxis_showticklabels=False,
        xaxis2_showticklabels=False,

        # Set Y-Axis titles
        yaxis1_title="PPI (Base 100)",
        yaxis2_title="Volume (Z-Score)",
        # ---!!!--- FIX: Corrected variable name ---!!!---
        yaxis3_title=y_title_score,
        yaxis3_range=y_range_score
    )
    
    return fig
# ---!!!--- END OF NEW FUNCTION ---!!!---


# --- 3. MAIN APP LAYOUT ---

st.title("ASSRS Sector Rotation Scoreboard")

# --- Load Data ---
v1_latest, v1_hist, v1_date, v1_error = load_data(V1_RULES_FILE, "V1")
v2_latest, v2_hist, v2_date, v2_error = load_data(V2_REGIME_FILE, "V2")

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
        v1_sector_to_chart = st.selectbox("Select V1 Sector to Chart:", v1_hist['Sector'].unique(), key="v1_selector")
        
        if v1_sector_to_chart:
            chart_data = v1_hist[v1_hist['Sector'] == v1_sector_to_chart]
            
            # ---!!!--- FIX: Call single chart function ---!!!---
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
        v2_sector_to_chart = st.selectbox("Select V2 Sector to Chart:", v2_hist['Sector'].unique(), key="v2_selector")
        
        if v2_sector_to_chart:
            chart_data = v2_hist[v2_hist['Sector'] == v2_sector_to_chart]

            # ---!!!--- FIX: Call single chart function ---!!!---
            fig = create_drilldown_chart(chart_data, model_type='v2')
            st.plotly_chart(fig, use_container_width=True, key="v2_chart")
            
    else:
        st.error(v2_error)
