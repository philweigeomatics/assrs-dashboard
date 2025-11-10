import streamlit as st
import pandas as pd
import altair as alt
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
        
        # 2. Get Full History (prepared for Altair charts)
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
        
        # ---!!!--- FIX: Create and style the DF *before* passing it ---!!!---
        v1_display_df = v1_latest[['Sector', 'TOTAL_SCORE', 'ACTION']].sort_values(by='TOTAL_SCORE', ascending=False)
        
        # Use .map() for modern pandas styling
        styled_v1_df = v1_display_df.style.map(style_action, subset=['ACTION'])
        
        st.dataframe(
            styled_v1_df, # Pass the styled object directly
            hide_index=True,
            use_container_width=True,
            column_config={
                "TOTAL_SCORE": st.column_config.NumberColumn(format="%.2f"),
                "ACTION": st.column_config.TextColumn(
                    width="medium"
                )
            }
        )
        # ---!!!--- END OF FIX ---!!!---
        
        # Sector selection
        v1_sector_to_chart = st.selectbox("Select V1 Sector to Chart:", v1_hist['Sector'].unique())
        
        if v1_sector_to_chart:
            chart_data = v1_hist[v1_hist['Sector'] == v1_sector_to_chart]
            
            # Chart 1: Price + Volume
            price_chart = alt.Chart(chart_data).mark_line(color="#3b82f6").encode(
                x=alt.X('Date', axis=alt.Axis(title='Date')),
                y=alt.Y('Close', axis=alt.Axis(title='PPI (Base 100)', titleColor="#3b82f6")),
            ).properties(title=f"{v1_sector_to_chart} - PPI Price History")

            volume_chart = alt.Chart(chart_data).mark_bar(color="#9ca3af", opacity=0.3).encode(
                x=alt.X('Date'),
                y=alt.Y('Volume_Metric', axis=alt.Axis(title='Volume Metric (Z-Score)', titleColor="#9ca3af")),
            )
            
            # Chart 2: Score
            score_chart = alt.Chart(chart_data).mark_line(color="#10b981").encode(
                x=alt.X('Date', axis=alt.Axis(title='Date')),
                y=alt.Y('TOTAL_SCORE', axis=alt.Axis(title='V1 Score (-3 to 8)', titleColor="#10b981")),
            ).properties(title=f"{v1_sector_to_chart} - V1 Score History")
            
            st.altair_chart((price_chart + volume_chart).resolve_scale(y='independent').interactive(), use_container_width=True)
            st.altair_chart(score_chart.interactive(), use_container_width=True)

    else:
        st.error(v1_error)

# --- V2 (Regime-Switching) Scorecard ---
with col2:
    st.header("V2: Regime-Switching Model")
    if v2_latest is not None:
        st.caption(f"Last Updated: {v2_date}")
        
        # ---!!!--- FIX: Create and style the DF *before* passing it ---!!!---
        v2_display_df = v2_latest[['Sector', 'TOTAL_SCORE', 'ACTION']].copy()
        v2_display_df = v2_display_df.sort_values(by='TOTAL_SCORE', ascending=False)
        
        # Format the probability
        v2_display_df['TOTAL_SCORE'] = (v2_display_df['TOTAL_SCORE'] * 100).map('{:.0f}%'.format)
        
        # Style the dataframe
        styled_v2_df = v2_display_df.style.map(style_action, subset=['ACTION'])
        
        st.dataframe(
            styled_v2_df, # Pass the styled object directly
            hide_index=True,
            use_container_width=True,
            column_config={
                "TOTAL_SCORE": "Bull Probability",
                "ACTION": st.column_config.TextColumn(
                    width="medium"
                )
            }
        )
        # ---!!!--- END OF FIX ---!!!---
        
        # Sector selection
        v2_sector_to_chart = st.selectbox("Select V2 Sector to Chart:", v2_hist['Sector'].unique())
        
        if v2_sector_to_chart:
            chart_data = v2_hist[v2_hist['Sector'] == v2_sector_to_chart]
            
            # Chart 1: Price + Volume
            price_chart = alt.Chart(chart_data).mark_line(color="#3b82f6").encode(
                x=alt.X('Date', axis=alt.Axis(title='Date')),
                y=alt.Y('Close', axis=alt.Axis(title='PPI (Base 100)', titleColor="#3b82f6")),
            ).properties(title=f"{v2_sector_to_chart} - PPI Price History")

            volume_chart = alt.Chart(chart_data).mark_bar(color="#9ca3af", opacity=0.3).encode(
                x=alt.X('Date'),
                y=alt.Y('Volume_Metric', axis=alt.Axis(title='Volume Metric (Z-Score)', titleColor="#9ca3af")),
            )
            
            # Chart 2: Score (Probability)
            score_chart = alt.Chart(chart_data).mark_line(color="#10b981").encode(
                x=alt.X('Date', axis=alt.Axis(title='Date')),
                y=alt.Y('TOTAL_SCORE', axis=alt.Axis(title='V2 Bull Probability (0 to 1)', titleColor="#10b981"), scale=alt.Scale(domain=[0, 1])),
            ).properties(title=f"{v2_sector_to_chart} - V2 Probability History")
            
            st.altair_chart((price_chart + volume_chart).resolve_scale(y='independent').interactive(), use_container_width=True)
            st.altair_chart(score_chart.interactive(), use_container_width=True)
            
    else:
        st.error(v2_error)
