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
import time


st.title("📊 Sector Stock Selector 板块个股选择器")

import auth_manager
auth_manager.require_login()


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
        st.session_state.stock_market_caps = {}  # ✅ Add this


        # ✅ Clear derived calculations from previous sector
        if 'cap_weighted_sector_returns' in st.session_state:
            del st.session_state.cap_weighted_sector_returns
        if 'sector_weights' in st.session_state:
            del st.session_state.sector_weights
        if 'returns_df' in st.session_state:
            del st.session_state.returns_df
        if 'avg_correlations' in st.session_state:
            del st.session_state.avg_correlations

        # ✅ Store which sector was loaded
        st.session_state.loaded_sector = selected_sector

        progress_bar = st.progress(0)
        for idx, ticker in enumerate(stock_list):
            # Fetch 3 years of live data
            df = data_manager.get_single_stock_data_live(ticker, lookback_years=3)

            if df is not None and not df.empty:
                st.session_state.sector_data[ticker] = df
                # Calculate daily returns
                st.session_state.sector_returns[ticker] = df['Close'].pct_change()

                # ✅ Fetch latest market cap
                latest_date = df.index.max().strftime('%Y%m%d')
                fundamentals = data_manager.get_stock_fundamentals_live(
                    ticker,
                    start_date=latest_date,
                    end_date=latest_date
                )

                if fundamentals is not None and not fundamentals.empty:
                    # Store market cap in 万元 (as returned by Tushare)
                    st.session_state.stock_market_caps[ticker] = fundamentals['Total_MV'].iloc[-1]
                else:
                    st.session_state.stock_market_caps[ticker] = None
                
                # time.sleep(0.31)  # Rate limit

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
stock_market_caps = st.session_state.get('stock_market_caps', {})

# ✅ Calculate cap-weighted sector returns (only once, not in fragment)
if 'cap_weighted_sector_returns' not in st.session_state:
    # Build returns DataFrame
    returns_df_full = pd.DataFrame(sector_returns)
    
    # Filter stocks with valid market cap
    valid_tickers = [t for t in returns_df_full.columns if stock_market_caps.get(t) is not None]
    
    if len(valid_tickers) > 0:
        # Calculate weights
        total_cap = sum([stock_market_caps[t] for t in valid_tickers])
        weights = pd.Series({t: stock_market_caps[t] / total_cap for t in valid_tickers})
        
        # Calculate weighted returns for each date
        cap_weighted_returns = returns_df_full[valid_tickers].mul(weights, axis=1).sum(axis=1)
        
        st.session_state.cap_weighted_sector_returns = cap_weighted_returns
        st.session_state.sector_weights = weights  # Store weights for display
        
        st.success(f"✅ Calculated cap-weighted sector benchmark using {len(valid_tickers)} stocks")
    else:
        # Fallback to equal-weighted
        cap_weighted_returns = returns_df_full.mean(axis=1)
        st.session_state.cap_weighted_sector_returns = cap_weighted_returns
        st.warning("⚠️ No market cap data available. Using equal-weighted sector average.")

st.markdown("---")




# ============================================================================
# SECTION 1: ROLLING CORRELATION MATRIX
# ============================================================================
@st.fragment  # ✨ Only this section reruns when sliders change
def correlation_section():
    st.header("1️⃣ Rolling Correlation Matrix")
    st.markdown("See how stocks move together within the sector")


    # Initialize ONLY if not exists (first run only)
    if 'correlation_window' not in st.session_state:
        st.session_state.correlation_window = 10
    if 'date_lookback' not in st.session_state:
        st.session_state.date_lookback = 30

    col1, col2 = st.columns([2, 1])

    with col1:
        correlation_window = st.slider(
            "Correlation Window (days)", 
            5, 
            60, 
            10,
            5,
            key = 'corr_window'
        )

    with col2:
        date_lookback = st.slider(
            "Date Range (days)", 
            30, 
            252, 
            30,
            30,
            key='date_lookback_key'
        )

    # Validate: Date range must be at least 2x correlation window
    if date_lookback < correlation_window * 2:
        st.error(f"❌ Invalid configuration: Date range ({date_lookback} days) must be at least twice the correlation window ({correlation_window} days). Minimum required: {correlation_window * 2} days.")
        st.warning("⚠️ Please increase Date Range or decrease Correlation Window.")
        
        # Use last valid values from session state
        if 'last_valid_corr_window' in st.session_state and 'last_valid_date_lookback' in st.session_state:
            correlation_window = st.session_state.last_valid_corr_window
            date_lookback = st.session_state.last_valid_date_lookback
            st.info(f"💡 Using previous valid values: Window={correlation_window}, Range={date_lookback}")
        else:
            # First time invalid, just stop
            return
    else:
        # Valid - save for future invalid states
        st.session_state.last_valid_corr_window = correlation_window
        st.session_state.last_valid_date_lookback = date_lookback

    # Build returns dataframe
    returns_df = pd.DataFrame(sector_returns)
    returns_df = returns_df.tail(date_lookback).dropna()
    st.session_state.returns_df = returns_df

    # Calculate AVERAGE rolling correlation over the entire date range
    if len(returns_df) >= correlation_window:
        # Calculate rolling correlation for each pair of stocks
        n_stocks = len(returns_df.columns)
        avg_corr_matrix = pd.DataFrame(np.zeros((n_stocks, n_stocks)), 
                                        index=returns_df.columns, 
                                        columns=returns_df.columns)
        
        # For each pair of stocks, calculate average rolling correlation
        for i, stock1 in enumerate(returns_df.columns):
            for j, stock2 in enumerate(returns_df.columns):
                if i == j:
                    avg_corr_matrix.iloc[i, j] = 1.0  # Diagonal = 1
                elif i < j:  # Only calculate upper triangle, then mirror
                    # Calculate rolling correlation
                    rolling_corr = returns_df[stock1].rolling(window=correlation_window).corr(returns_df[stock2])
                    # Take average of all rolling correlations
                    avg_corr = rolling_corr.dropna().mean()
                    avg_corr_matrix.iloc[i, j] = avg_corr
                    avg_corr_matrix.iloc[j, i] = avg_corr  # Mirror to lower triangle
        
        # Calculate how many rolling windows were averaged
        num_windows = len(returns_df) - correlation_window + 1
        
        # Create heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=avg_corr_matrix.values,
            x=[data_manager.get_stock_name_from_db(t) or t for t in avg_corr_matrix.columns],
            y=[data_manager.get_stock_name_from_db(t) or t for t in avg_corr_matrix.index],
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=avg_corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorbar=dict(title='Avg Correlation')
        ))
        
        fig_corr.update_layout(
            title=f"Average Rolling Correlation Matrix ({selected_sector})<br>" +
                f"<sub>{correlation_window}-day window averaged over {num_windows} periods (last {date_lookback} days)</sub>",
            height=600,
            template='plotly_white',
            xaxis={'tickangle': -45},
            yaxis={'tickangle': 0}
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Show average correlation for each stock
        st.subheader("📊 Average Correlation with Sector")
        avg_correlations = avg_corr_matrix.mean(axis=1).sort_values(ascending=False)

        # ✅ Store avg_correlations in session state for other sections
        st.session_state.avg_correlations = avg_correlations
        
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

    else:
        st.error(f"❌ Insufficient data in the selected date range. Please try a longer period.")

# Store selected_sector in session state before calling fragment
st.session_state.selected_sector = selected_sector
# Call the fragment
correlation_section()
st.markdown("---")


# ============================================================================
# SECTION 2: INDIVIDUAL RISK METRICS
# ============================================================================
st.header("2️⃣ Individual Stock Risk Metrics")
st.markdown("Compare risk profiles of all stocks in the sector")

# ✅ Use returns_df and avg_correlations from session state
if 'returns_df' not in st.session_state or 'avg_correlations' not in st.session_state:
    st.error("❌ Please calculate correlation matrix first.")
    st.stop()

returns_df = st.session_state.returns_df
avg_correlations = st.session_state.avg_correlations

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

progress_placeholder = st.empty()
with progress_placeholder.container():
    # Add progress indicator for financial data fetching
    st.info("📊 Fetching financial metrics from ...")
    progress_bar_fin = st.progress(0)

for idx, (ticker, df) in enumerate(sector_data.items()):
    # Filter data by selected date range
    df_filtered = df[(df.index >= risk_start_ts) & (df.index <= risk_end_ts)]
    
    if len(df_filtered) < 30:
        continue
    
    returns = df_filtered['Close'].pct_change().dropna()
    
    if len(returns) < 20:
        continue
    
    # Calculate existing metrics
    volatility_daily = returns.std() * 100  # Daily volatility as percentage
    volatility_annual = returns.std() * np.sqrt(252) * 100  # Annualized volatility as percentage

    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_drawdown = ((df_filtered['Close'] / df_filtered['Close'].cummax()) - 1).min()
    avg_return = returns.mean() * 252


    # Calculate beta using cap-weighted sector returns
    cap_weighted_returns = st.session_state.cap_weighted_sector_returns
    sector_avg_returns = cap_weighted_returns.loc[
        (cap_weighted_returns.index >= risk_start_ts) & (cap_weighted_returns.index <= risk_end_ts)
    ]

    if len(sector_avg_returns) > 0 and len(returns) > 0:
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

    
    # 🆕 Fetch financial indicators from Tushare
    financial_data = data_manager.get_financial_indicators(ticker)
    time.sleep(0.31)  # Respect API rate limit
    
    # Build metrics dictionary
    metrics_dict = {
        'Ticker': ticker,
        'Name': data_manager.get_stock_name_from_db(ticker) or ticker,
        'Daily Vol (%)': volatility_daily,      # Daily volatility
        'Annual Vol (%)': volatility_annual,    # Annualized volatility
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown * 100,
        'Avg Return': avg_return * 100,
        'Beta': beta,
        'Avg Correlation': avg_correlations.get(ticker, np.nan)
    }
    
    # 🆕 Add financial metrics if available
    if financial_data:
        metrics_dict.update({
            'ROE': financial_data.get('ROE'),
            'ROA': financial_data.get('ROA'),
            'Op. Margin': financial_data.get('Operating_Margin'),
            'EPS Growth YoY': financial_data.get('EPS_Growth_YoY'),
            'FCFF Growth YoY': financial_data.get('FCFF_Growth_YoY'),
            'FCFE Growth YoY': financial_data.get('FCFE_Growth_YoY')
        })
    else:
        # Add None values if data unavailable
        metrics_dict.update({
            'ROE': None,
            'ROA': None,
            'Op. Margin': None,
            'EPS Growth YoY': None,
            'FCFF Growth YoY': None,
            'FCFE Growth YoY': None
        })
    
    risk_metrics.append(metrics_dict)
    progress_bar_fin.progress((idx + 1) / len(sector_data))

# ✅ Clear the progress placeholder completely after loop
progress_placeholder.empty()

risk_df = pd.DataFrame(risk_metrics)

if len(risk_df) == 0:
    st.error("❌ No stocks have sufficient data in the selected date range. Try a different period.")
    st.stop()

# Display metrics table with enhanced formatting
st.subheader("📋 Risk & Financial Metrics Table")
st.dataframe(
    risk_df.style.format({
        'Daily Vol (%)': '{:.2f}%',       # Daily volatility (2-3% typical)
        'Annual Vol (%)': '{:.1f}%',      # Annual volatility (30-50% typical)
        'Sharpe Ratio': '{:.3f}',
        'Max Drawdown': '{:.2f}%',
        'Avg Return': '{:.2f}%',
        'Beta': '{:.2f}',
        'Avg Correlation': '{:.3f}',
        'ROE': '{:.2f}%',
        'ROA': '{:.2f}%',
        'Op. Margin': '{:.2f}%',
        'EPS Growth YoY': '{:.2f}%',
        'FCFF Growth YoY': '{:.2f}%',
        'FCFE Growth YoY': '{:.2f}%'
    }, na_rep='N/A')
    .background_gradient(subset=['Sharpe Ratio', 'Avg Return'], cmap='RdYlGn_r'),
    # .background_gradient(subset=['ROE (%)', 'ROA (%)', 'EPS Growth YoY (%)'], cmap='RdYlGn'),
    use_container_width=True,
    height=400,
)

# ============================================================================
# SECTION 2.5: ROLLING BETA DECOUPLING ANALYSIS
# ============================================================================
st.markdown("---")

@st.fragment  # ✅ Isolate this section
def rolling_beta_section():
    st.subheader("📉 Rolling Beta Analysis - Detect Sector Decoupling")
    st.caption("💡 **Decoupling Signal**: When rolling beta drops sharply, the stock is breaking away from sector moves (idiosyncratic risk > systematic risk)")
    
    # Stock selector for beta analysis
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stock_for_beta = st.selectbox(
            "Select stock to analyze rolling beta",
            options=st.session_state.risk_df['Ticker'].tolist(),  # ✅ Access from session state
            format_func=lambda x: f"{data_manager.get_stock_name_from_db(x)} ({x})",
            key='beta_stock_selector'
        )
    
    with col2:
        beta_window = st.slider(
            "Rolling Window (days)",
            min_value=20,
            max_value=120,
            value=60,
            step=10,
            key='beta_rolling_window'
        )
    
    # Calculate rolling beta for selected stock
    if selected_stock_for_beta:
        ticker = selected_stock_for_beta
        
        # Get stock returns from session state
        if ticker in st.session_state.sector_data:
            df_stock = st.session_state.sector_data[ticker]
            stock_returns = df_stock['Close'].pct_change().dropna()
            
            # Get cap-weighted sector returns
            sector_returns_series = st.session_state.cap_weighted_sector_returns
            
            # Align dates
            common_dates = stock_returns.index.intersection(sector_returns_series.index)
            stock_returns_aligned = stock_returns.loc[common_dates]
            sector_returns_aligned = sector_returns_series.loc[common_dates]
            
            # Calculate rolling beta
            def calculate_rolling_beta(stock_ret, sector_ret, window):
                """Calculate rolling beta using covariance / variance"""
                rolling_cov = stock_ret.rolling(window).cov(sector_ret)
                rolling_var = sector_ret.rolling(window).var()
                rolling_beta = rolling_cov / rolling_var
                return rolling_beta
            
            rolling_beta = calculate_rolling_beta(stock_returns_aligned, sector_returns_aligned, beta_window)
            
            # Calculate beta change (derivative) to detect sharp drops
            beta_change = rolling_beta.diff()
            
            # Identify decoupling events (beta drop > 0.3 in one period)
            decoupling_threshold = -0.3
            decoupling_events = beta_change[beta_change < decoupling_threshold]
            
            # Create multi-panel chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f"{data_manager.get_stock_name_from_db(ticker)} Price",
                    f"Rolling Beta ({beta_window}-day) vs Sector",
                    "Beta Change (Decoupling Indicator)"
                ),
                row_heights=[0.4, 0.35, 0.25]
            )
            
            # Panel 1: Stock Price
            fig.add_trace(
                go.Scatter(
                    x=df_stock.index,
                    y=df_stock['Close'],
                    name='Price',
                    line=dict(color='#3b82f6', width=2)
                ),
                row=1, col=1
            )
            
            # Panel 2: Rolling Beta
            fig.add_trace(
                go.Scatter(
                    x=rolling_beta.index,
                    y=rolling_beta.values,
                    name=f'{beta_window}d Beta',
                    line=dict(color='#10b981', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ),
                row=2, col=1
            )
            
            # Add beta = 1.0 reference line
            fig.add_hline(
                y=1.0,
                line_dash='dash',
                line_color='gray',
                opacity=0.5,
                row=2, col=1,
                annotation_text="Beta = 1.0",
                annotation_position="right"
            )
            
            # Add current static beta from risk metrics
            current_static_beta = st.session_state.risk_df[st.session_state.risk_df['Ticker'] == ticker]['Beta'].iloc[0]
            fig.add_hline(
                y=current_static_beta,
                line_dash='dot',
                line_color='orange',
                opacity=0.7,
                row=2, col=1,
                annotation_text=f"Period Avg: {current_static_beta:.2f}",
                annotation_position="left"
            )
            
            # Panel 3: Beta Change (Derivative)
            fig.add_trace(
                go.Scatter(
                    x=beta_change.index,
                    y=beta_change.values,
                    name='Beta Change',
                    line=dict(color='#ef4444', width=1.5),
                    fill='tozeroy'
                ),
                row=3, col=1
            )
            
            # Highlight decoupling events
            if len(decoupling_events) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=decoupling_events.index,
                        y=decoupling_events.values,
                        mode='markers',
                        name='Decoupling Events',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='x',
                            line=dict(width=2)
                        )
                    ),
                    row=3, col=1
                )
            
            # Add threshold line
            fig.add_hline(
                y=decoupling_threshold,
                line_dash='dash',
                line_color='red',
                opacity=0.5,
                row=3, col=1,
                annotation_text=f"Decoupling Threshold",
                annotation_position="right"
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price (¥)", row=1, col=1)
            fig.update_yaxes(title_text="Beta", row=2, col=1)
            fig.update_yaxes(title_text="Δ Beta", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation and insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_beta = rolling_beta.iloc[-1] if not pd.isna(rolling_beta.iloc[-1]) else current_static_beta
                beta_trend = "📈 Increasing" if rolling_beta.iloc[-20:].mean() > rolling_beta.iloc[-40:-20].mean() else "📉 Decreasing"
                st.metric(
                    "Current Rolling Beta",
                    f"{current_beta:.2f}",
                    delta=f"{beta_trend}",
                    delta_color="off"
                )
            
            with col2:
                decoupling_count = len(decoupling_events)
                st.metric(
                    f"Decoupling Events ({len(rolling_beta)} days)",
                    f"{decoupling_count}",
                    delta="🔴 High idiosyncratic risk" if decoupling_count > 3 else "🟢 Sector-driven"
                )
            
            with col3:
                beta_volatility = rolling_beta.std()
                stability = "Stable" if beta_volatility < 0.3 else "Volatile"
                st.metric(
                    "Beta Stability",
                    f"{stability}",
                    delta=f"σ = {beta_volatility:.2f}",
                    delta_color="off"
                )
            
            # Interpretation guide
            with st.expander("📖 How to Interpret Rolling Beta Decoupling"):
                st.markdown("""
                **What is Beta Decoupling?**
                - **Normal Beta (~1.0)**: Stock moves in sync with sector (systematic risk dominates)
                - **Rising Beta (>1.2)**: Stock amplifies sector movements (high systematic risk)
                - **Falling Beta (<0.7)**: Stock decouples from sector (idiosyncratic risk dominates)
                
                **Decoupling Signals (Sharp Beta Drops)**:
                - 🔴 **Sudden drop in beta** (>0.3 decrease): Company-specific news is driving the stock
                - 📰 **Potential causes**: Earnings surprise, regulatory action, management change, M&A rumors
                - 💡 **Trading insight**: Stock is responding to its own story, not sector trends
                
                **Trading Implications**:
                - **High beta (>1.2)**: Use sector ETF hedges, trade sector momentum
                - **Low/falling beta**: Analyze company fundamentals, ignore sector noise
                - **Multiple decoupling events**: Stock has strong idiosyncratic drivers
                
                **Example**:
                - If 贵州茅台's beta drops from 1.1 to 0.6 → Check for Moutai-specific news (not sector-wide issues)
                - If beta stays near 1.0 → Stock moves with sector, use sector analysis for trading decisions
                """)
        else:
            st.error(f"❌ No data available for {ticker}")

# Store risk_df in session state before calling fragment
st.session_state.risk_df = risk_df

# Call the fragment
rolling_beta_section()

st.markdown("---")
st.markdown("### 🎯 Click to Analyze Individual Stock")


# Create buttons in 4 columns
num_cols = 4
cols = st.columns(num_cols)

for idx, row in risk_df.iterrows():
    col_idx = idx % num_cols
    with cols[col_idx]:
        button_label = f"{row['Name'][:12]}...\n{row['Ticker']}"
        if st.button(button_label, key=f"btn_{row['Ticker']}", use_container_width=True):
            st.session_state.active_ticker = row['Ticker']
            st.switch_page("pages/2_Single_Stock_Analysis_个股分析.py")

# ============================================================================
# SECTION 3: RISK-RETURN SCATTER PLOT
# ============================================================================
st.subheader("📈 Risk-Return Profile")

fig_scatter = go.Figure()

# Add scatter points
fig_scatter.add_trace(go.Scatter(
    x=risk_df['Annual Vol (%)'],
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
                  'Annual Vol (%): %{x:.2f}%<br>' +
                  'Return: %{y:.2f}%<br>' +
                  '<extra></extra>'
))

# Add quadrant lines
median_vol = risk_df['Annual Vol (%)'].median()
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
        float(risk_df['Annual Vol (%)'].min()),
        float(risk_df['Annual Vol (%)'].max()),
        (float(risk_df['Annual Vol (%)'].quantile(0.25)),
         float(risk_df['Annual Vol (%)'].quantile(0.75)))
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
    (risk_df['Annual Vol (%)'] >= vol_min) & (risk_df['Annual Vol (%)'] <= vol_max) &
    (risk_df['Beta'] >= beta_min) & (risk_df['Beta'] <= beta_max) &
    (risk_df['Avg Correlation'] >= corr_min) & (risk_df['Avg Correlation'] <= corr_max)
]

st.subheader(f"🎯 Filtered Results: {len(filtered_df)} stocks match your criteria")

if len(filtered_df) > 0:
    # Display filtered results
    st.dataframe(
        filtered_df.style.format({
            'Annual Vol (%)': '{:.2f}%',
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

    if vol_min < risk_df['Annual Vol (%)'].median():
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