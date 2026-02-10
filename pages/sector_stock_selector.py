"""
Sector Stock Selector - 板块个股选择器
Analyze individual stocks within a sector: correlations and risk metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_manager

st.title("📊 Sector Stock Selector 板块个股选择器")

# Load sector map
SECTOR_STOCK_MAP = data_manager.SECTOR_STOCK_MAP

# Sector selector
st.sidebar.header("🎯 Select Sector")
available_sectors = [s for s in SECTOR_STOCK_MAP.keys() if s != "MARKET_PROXY"]
selected_sector = st.sidebar.selectbox(
    "Choose a sector to analyze",
    available_sectors,
    index=0
)

# Get stocks in selected sector
stock_list = SECTOR_STOCK_MAP[selected_sector]
st.sidebar.info(f"**{len(stock_list)}** stocks in {selected_sector}")

# Load data button
if st.sidebar.button("🔄 Load Data (3 Years)", type="primary"):
    with st.spinner(f"Loading data for {len(stock_list)} stocks in {selected_sector}..."):
        # Store in session state
        st.session_state.sector_data = {}
        st.session_state.sector_returns = {}

        progress_bar = st.progress(0)
        for idx, ticker in enumerate(stock_list):
            # Fetch 3 years of live data
            df = data_manager.get_single_stock_data_live(ticker, lookback_years=3)

            if df is not None and not df.empty:
                st.session_state.sector_data[ticker] = df
                # Calculate daily returns
                st.session_state.sector_returns[ticker] = df['Close'].pct_change()

            progress_bar.progress((idx + 1) / len(stock_list))

        st.success(f"✅ Loaded {len(st.session_state.sector_data)} stocks successfully!")
        st.rerun()

# Check if data is loaded
if 'sector_data' not in st.session_state or not st.session_state.sector_data:
    st.info("👆 Click **Load Data** in the sidebar to begin analysis")
    st.stop()

# Data is loaded - show analysis
sector_data = st.session_state.sector_data
sector_returns = st.session_state.sector_returns

st.markdown("---")

# ============================================================================
# SECTION 1: ROLLING CORRELATION MATRIX
# ============================================================================
st.header("1️⃣ Rolling Correlation Matrix")
st.markdown("See how stocks move together within the sector")

col1, col2 = st.columns([2, 1])
with col1:
    correlation_window = st.slider("Correlation Window (days)", 5, 60, 10, 5)
with col2:
    date_lookback = st.slider("Date Range (days)", 30, 252, 90, 30)

# Build returns dataframe
returns_df = pd.DataFrame(sector_returns)
returns_df = returns_df.tail(date_lookback).dropna()

# Calculate rolling correlations for each stock vs all others
if len(returns_df) > correlation_window:

    # Calculate correlation matrix for latest period
    latest_corr = returns_df.tail(correlation_window).corr()

    # Create heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=latest_corr.values,
        x=[data_manager.get_stock_name_from_db(t) or t for t in latest_corr.columns],
        y=[data_manager.get_stock_name_from_db(t) or t for t in latest_corr.index],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=latest_corr.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorbar=dict(title='Correlation')
    ))

    fig_corr.update_layout(
        title=f"Stock Correlation Matrix ({selected_sector}) - Last {correlation_window} Days",
        height=600,
        template='plotly_white',
        xaxis={'tickangle': -45},
        yaxis={'tickangle': 0}
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Show average correlation for each stock
    st.subheader("📊 Average Correlation with Sector")
    avg_correlations = latest_corr.mean(axis=1).sort_values(ascending=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔴 Lowest Correlation (Most Independent)**")
        for ticker in avg_correlations.tail(3).index:
            name = data_manager.get_stock_name_from_db(ticker) or ticker
            st.metric(name, f"{avg_correlations[ticker]:.3f}")

    with col2:
        st.markdown("**🟡 Moderate Correlation**")
        median_idx = len(avg_correlations) // 2
        for ticker in avg_correlations.iloc[median_idx-1:median_idx+2].index:
            name = data_manager.get_stock_name_from_db(ticker) or ticker
            st.metric(name, f"{avg_correlations[ticker]:.3f}")

    with col3:
        st.markdown("**🟢 Highest Correlation (Most Sector-Like)**")
        for ticker in avg_correlations.head(3).index:
            name = data_manager.get_stock_name_from_db(ticker) or ticker
            st.metric(name, f"{avg_correlations[ticker]:.3f}")

st.markdown("---")

# ============================================================================
# SECTION 2: INDIVIDUAL RISK METRICS
# ============================================================================
st.header("2️⃣ Individual Stock Risk Metrics")
st.markdown("Compare risk profiles of all stocks in the sector")

# Date range selector for risk metrics
st.subheader("📅 Risk Calculation Period")
col1, col2 = st.columns(2)

# Get earliest and latest dates from all stocks
all_dates = []
for df in sector_data.values():
    all_dates.extend(df.index.tolist())
min_date = min(all_dates).date()
max_date = max(all_dates).date()

with col1:
    risk_start_date = st.date_input(
        "Start Date",
        value=max_date - pd.Timedelta(days=365),  # Default: 1 year ago
        min_value=min_date,
        max_value=max_date,
        key='risk_start'
    )

with col2:
    risk_end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key='risk_end'
    )

# Convert to pandas Timestamp for filtering
risk_start_ts = pd.Timestamp(risk_start_date)
risk_end_ts = pd.Timestamp(risk_end_date)

if risk_start_ts >= risk_end_ts:
    st.error("⚠️ Start date must be before end date!")
    st.stop()

period_days = (risk_end_ts - risk_start_ts).days
st.info(f"📊 Calculating risk metrics using data from **{risk_start_date}** to **{risk_end_date}** ({period_days} days)")

# Calculate risk metrics for all stocks
risk_metrics = []

for ticker, df in sector_data.items():
    # Filter data by selected date range
    df_filtered = df[(df.index >= risk_start_ts) & (df.index <= risk_end_ts)]

    if len(df_filtered) < 30:
        continue  # Skip if insufficient data in this range

    returns = df_filtered['Close'].pct_change().dropna()

    if len(returns) < 20:
        continue

    # Calculate metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_drawdown = ((df_filtered['Close'] / df_filtered['Close'].cummax()) - 1).min()
    avg_return = returns.mean() * 252  # Annualized

    # Calculate beta vs sector average (using same date range)
    sector_avg_returns = returns_df.loc[
        (returns_df.index >= risk_start_ts) & (returns_df.index <= risk_end_ts)
    ].mean(axis=1)

    if len(sector_avg_returns) > 0 and len(returns) > 0:
        # Align the two series
        common_idx = returns.index.intersection(sector_avg_returns.index)
        if len(common_idx) > 20:
            aligned_stock = returns.loc[common_idx]
            aligned_sector = sector_avg_returns.loc[common_idx]

            covariance = np.cov(aligned_stock, aligned_sector)[0][1]
            sector_variance = aligned_sector.var()
            beta = covariance / sector_variance if sector_variance > 0 else 1.0
        else:
            beta = 1.0
    else:
        beta = 1.0

    risk_metrics.append({
        'Ticker': ticker,
        'Name': data_manager.get_stock_name_from_db(ticker) or ticker,
        'Volatility': volatility * 100,  # As percentage
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown * 100,  # As percentage
        'Avg Return': avg_return * 100,  # As percentage
        'Beta': beta,
        'Avg Correlation': avg_correlations.get(ticker, np.nan)
    })

risk_df = pd.DataFrame(risk_metrics)

if len(risk_df) == 0:
    st.error("❌ No stocks have sufficient data in the selected date range. Try a different period.")
    st.stop()

# Display metrics table
st.subheader("📋 Risk Metrics Table")
st.dataframe(
    risk_df.style.format({
        'Volatility': '{:.2f}%',
        'Sharpe Ratio': '{:.3f}',
        'Max Drawdown': '{:.2f}%',
        'Avg Return': '{:.2f}%',
        'Beta': '{:.2f}',
        'Avg Correlation': '{:.3f}'
    })        # ✅ More negative = Greener
    .background_gradient(subset=['Sharpe Ratio', 'Avg Return'], cmap='RdYlGn_r'), 
    use_container_width=True,
    height=400
)

# ============================================================================
# SECTION 3: RISK-RETURN SCATTER PLOT
# ============================================================================
st.subheader("📈 Risk-Return Profile")

fig_scatter = go.Figure()

# Add scatter points
fig_scatter.add_trace(go.Scatter(
    x=risk_df['Volatility'],
    y=risk_df['Avg Return'],
    mode='markers+text',
    marker=dict(
        size=risk_df['Beta'] * 10,  # Size based on beta
        color=risk_df['Avg Correlation'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Avg Correlation'),
        line=dict(width=1, color='white')
    ),
    text=risk_df['Name'],
    textposition='top center',
    textfont=dict(size=9),
    hovertemplate='<b>%{text}</b><br>' +
                  'Volatility: %{x:.2f}%<br>' +
                  'Return: %{y:.2f}%<br>' +
                  '<extra></extra>'
))

# Add quadrant lines
median_vol = risk_df['Volatility'].median()
median_ret = risk_df['Avg Return'].median()

fig_scatter.add_vline(x=median_vol, line_dash='dash', line_color='gray', opacity=0.5)
fig_scatter.add_hline(y=median_ret, line_dash='dash', line_color='gray', opacity=0.5)

fig_scatter.update_layout(
    title=f"Risk vs Return: {selected_sector} Stocks (Bubble Size = Beta) - {risk_start_date} to {risk_end_date}",
    xaxis_title="Volatility (Annualized %)",
    yaxis_title="Average Return (Annualized %)",
    height=600,
    template='plotly_white',
    hovermode='closest'
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ============================================================================
# SECTION 4: STOCK SELECTOR FILTERS
# ============================================================================
st.header("3️⃣ Stock Selector - Filter by Your Trading Criteria")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Volatility Range**")
    vol_min, vol_max = st.slider(
        "Annualized Volatility %",
        float(risk_df['Volatility'].min()),
        float(risk_df['Volatility'].max()),
        (float(risk_df['Volatility'].quantile(0.25)), 
         float(risk_df['Volatility'].quantile(0.75)))
    )

with col2:
    st.markdown("**Beta Range**")
    beta_min, beta_max = st.slider(
        "Beta (vs Sector)",
        float(risk_df['Beta'].min()),
        float(risk_df['Beta'].max()),
        (0.8, 1.2)
    )

with col3:
    st.markdown("**Correlation Range**")
    corr_min, corr_max = st.slider(
        "Avg Correlation",
        float(risk_df['Avg Correlation'].min()),
        float(risk_df['Avg Correlation'].max()),
        (0.3, 0.8)
    )

# Filter stocks
filtered_df = risk_df[
    (risk_df['Volatility'] >= vol_min) & (risk_df['Volatility'] <= vol_max) &
    (risk_df['Beta'] >= beta_min) & (risk_df['Beta'] <= beta_max) &
    (risk_df['Avg Correlation'] >= corr_min) & (risk_df['Avg Correlation'] <= corr_max)
]

st.subheader(f"🎯 Filtered Results: {len(filtered_df)} stocks match your criteria")

if len(filtered_df) > 0:
    # Display filtered results
    st.dataframe(
        filtered_df.style.format({
            'Volatility': '{:.2f}%',
            'Sharpe Ratio': '{:.3f}',
            'Max Drawdown': '{:.2f}%',
            'Avg Return': '{:.2f}%',
            'Beta': '{:.2f}',
            'Avg Correlation': '{:.3f}'
        }),
        use_container_width=True
    )

    # Show recommendations
    st.markdown("### 💡 Trading Recommendations")

    if vol_min < risk_df['Volatility'].median():
        st.success("**🟢 Conservative Play**: Low volatility stocks are good for stable returns with lower risk")
    else:
        st.warning("**🟠 Aggressive Play**: High volatility stocks for potential higher returns (and risk)")

    if corr_min < 0.5:
        st.info("**🔵 Diversification**: Low correlation stocks can reduce portfolio risk")
    else:
        st.info("**🔵 Sector Momentum**: High correlation stocks track sector trends closely")

    if beta_min < 0.9:
        st.success("**🟢 Defensive**: Low beta stocks are less sensitive to sector moves")
    elif beta_max > 1.1:
        st.warning("**🟠 Leveraged**: High beta stocks amplify sector movements")
    else:
        st.info("**🔵 Neutral**: Beta around 1.0 tracks sector performance")
else:
    st.warning("⚠️ No stocks match your criteria. Try adjusting the filters.")

st.markdown("---")
st.caption(f"💡 Tip: Adjust the date range above to focus on recent performance or analyze specific time periods (e.g., during market crashes)")