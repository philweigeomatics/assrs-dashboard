"""
Sector Analysis - Performance & Rotation
Track sector performance and rotation patterns
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go



from sector_utils import (
    load_v2_data,
    build_sector_panels,
    create_performance_comparison_chart,
    create_sector_rotation_map,
    create_rolling_correlation_chart
)

st.title("📈 Performance & Rotation")

# Load data
v2latest, v2hist, v2date, v2error = load_v2_data()
if v2latest is None:
    st.error(f"Error loading data: {v2error}")
    st.stop()

# Build panels
close_panel, ret_panel, vol_panel, exret_panel = build_sector_panels(v2hist)

# Sector selector
all_sectors = sorted([s for s in v2hist['Sector'].unique() if s != "MARKET_PROXY"])
selected_sectors = st.multiselect(
    "Select sectors to compare",
    all_sectors,
    default=all_sectors[:5]
)

if not selected_sectors:
    st.warning("Please select at least one sector")
    st.stop()

# Performance Comparison
st.subheader("📊 Sector Performance Comparison")

lookback_days = st.slider("Lookback days", 30, 180, 60, 10)

fig = create_performance_comparison_chart(v2hist, lookback_days, selected_sectors)

if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Unable to create performance chart")

st.markdown("---")


st.subheader("📊 Rolling Sector Correlations")

col1, col2 = st.columns([1, 2])
with col1:
    reference_sector = st.selectbox("Reference Sector", all_sectors)
    window = st.slider("Rolling Window (days)", 5, 60, 5, 5)
with col2:
    compare_sectors = st.multiselect(
        "Compare Against",
        [s for s in all_sectors if s != reference_sector],
        default=[s for s in all_sectors if s != reference_sector][:4]
    )

if compare_sectors:
    fig = create_rolling_correlation_chart(ret_panel, reference_sector, compare_sectors, window)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Sector Rotation Map
st.subheader("🔄 Sector Rotation Map")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    lookback_short = st.selectbox("Short period (days)", [3, 5, 7, 10], index=1)
with col2:
    lookback_long = st.selectbox("Long period (days)", [15, 20, 30, 40], index=1)

fig, rotation_df = create_sector_rotation_map(v2hist, lookback_short, lookback_long)

if fig:
    st.plotly_chart(fig, use_container_width=True)
    
    # Quadrant Breakdown
    st.markdown("---")
    st.subheader("📊 Quadrant Breakdown")
    
    quad_cols = st.columns(4)
    
    for idx, (quadrant, color) in enumerate([
        ("Leading", "#dcfce7"),
        ("Improving", "#dbeafe"),
        ("Weakening", "#fef9c3"),
        ("Lagging", "#fee2e2")
    ]):
        with quad_cols[idx]:
            sectors_in_quad = rotation_df[rotation_df['Quadrant'] == quadrant]['Sector'].tolist()
            st.markdown(f"**{quadrant}**")
            if sectors_in_quad:
                for sector in sectors_in_quad:
                    st.markdown(f"• {sector}")
            else:
                st.caption("None")
else:
    st.error("Unable to create rotation map")

# Correlation Matrix
st.markdown("---")
st.subheader("🔗 Sector Correlation Matrix")

returns = ret_panel[selected_sectors]
returns = returns.dropna()

if len(returns) > 0:
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
        colorbar=dict(title='Correlation'),
        zmin=-1,
        zmax=1
    ))
    
    fig_corr.update_layout(
        title=f"Daily Return Correlation",
        height=max(400, len(selected_sectors) * 60),
        template='plotly_white'
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
