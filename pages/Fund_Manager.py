import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import data_manager
import auth_manager
from db_manager import db
from zoneinfo import ZoneInfo

# Secure the page
auth_manager.require_login()
user_id = auth_manager.get_current_user_id()

st.set_page_config(page_title="Portfolio Manager | 机构级管理", page_icon="🏦", layout="wide")

st.title("🏦 Institutional Portfolio Manager")
st.markdown("Monitor AUM, track risk trends, and execute mandate rebalances.")

# ==========================================
# 1. FETCH USER'S FUNDS
# ==========================================
user_funds_df = db.read_table('funds', filters={'user_id': user_id})

if user_funds_df.empty:
    st.info("👆 You haven't created any funds yet. Go to the **Portfolio Optimization** page to launch your first mandate.")
    st.stop()

# ==========================================
# 2. FUND SELECTION HEADER
# ==========================================
col_sel, col_bench, col_admin = st.columns([2, 1, 1])
with col_sel:
    fund_names = user_funds_df['fund_name'].tolist()
    selected_fund_name = st.selectbox("Select Active Fund:", fund_names)
    
    fund_details = user_funds_df[user_funds_df['fund_name'] == selected_fund_name].iloc[0]
    fund_id = int(fund_details['id'])
    benchmark = fund_details['benchmark']


with col_bench:
    st.metric("Benchmark", benchmark)

with col_admin:
    st.markdown("<br>", unsafe_allow_html=True)
    # Allows you to manually trigger the math if the 1:00 AM job hasn't run yet
    if st.button("🔄 Force NAV Calculation", type="secondary", use_container_width=True):
        with st.spinner("Crunching portfolio math..."):
            data_manager.execute_daily_portfolio_rollup()
            st.toast("✅ NAV calculation complete!")
            st.rerun()

st.markdown("---")

# ==========================================
# 3. FETCH CORE DATA (METRICS & POSITIONS)
# ==========================================
CHINA_TZ = ZoneInfo("Asia/Shanghai")
today_str = datetime.now(CHINA_TZ).strftime('%Y-%m-%d')

# Get performance history
metrics_df = db.read_table('fund_daily_metrics', filters={'fund_id': fund_id}, order_by='trade_date')

# Get currently active positions (time-travel safe)
positions_df = db.read_table('fund_positions', filters={'fund_id': fund_id})
active_positions = pd.DataFrame()
if not positions_df.empty:
    active_positions = positions_df[positions_df['end_date'].isna()].copy()
else:
    active_positions = pd.DataFrame()


# ==========================================
# 4. TOP LEVEL KPIs
# ==========================================
# ==========================================
# 4. TOP LEVEL KPIs & BENCHMARK FETCHING
# ==========================================
m1, m2, m3, m4 = st.columns(4)

bench_df = pd.DataFrame()
portfolio_daily_ret = 0.0
portfolio_total_ret = 0.0
bench_daily_ret = 0.0
bench_total_ret = 0.0

if not metrics_df.empty:
    latest_metric = metrics_df.iloc[-1]
    aum = latest_metric['total_aum']
    
    # Base Capital Logic
    inception_aum = 10000000.0 
    
    # Calculate Portfolio Returns
    portfolio_total_ret = (aum - inception_aum) / inception_aum
    if len(metrics_df) > 1:
        prev_aum = metrics_df.iloc[-2]['total_aum']
        portfolio_daily_ret = (aum - prev_aum - latest_metric.get('net_flow', 0)) / prev_aum if prev_aum > 0 else 0
    else:
        portfolio_daily_ret = portfolio_total_ret

    # Fetch Benchmark Data to match our portfolio history
    start_date_ts = metrics_df['trade_date'].min().replace('-', '')
    end_date_ts = today_str.replace('-', '')
    
    with st.spinner(f"Fetching {benchmark} benchmark data..."):
        bench_df = data_manager.get_index_data_live(benchmark, start_date=start_date_ts, end_date=end_date_ts)
    
    # Calculate Benchmark Returns
    if bench_df is not None and not bench_df.empty:
        latest_bench = bench_df.iloc[-1]['Close']
        first_bench = bench_df.iloc[0]['Close']
        
        bench_total_ret = (latest_bench - first_bench) / first_bench
        if len(bench_df) > 1:
            prev_bench = bench_df.iloc[-2]['Close']
            bench_daily_ret = (latest_bench - prev_bench) / prev_bench
            
    # Display the Metrics
    m1.metric("Total AUM (¥)", f"{aum:,.2f}", f"Daily PnL: ¥{latest_metric['daily_pnl']:,.2f}")
    
    # Color-coded alpha comparison
    alpha = portfolio_total_ret - bench_total_ret
    alpha_str = f"Alpha: {'+' if alpha > 0 else ''}{alpha*100:.2f}%"
    m2.metric("Portfolio Return", f"{portfolio_total_ret*100:.2f}%", f"Daily: {portfolio_daily_ret*100:.2f}%")
    m3.metric(f"Benchmark ({benchmark.split('.')[0]})", f"{bench_total_ret*100:.2f}%", f"Daily: {bench_daily_ret*100:.2f}%")
else:
    m1.metric("Total AUM (¥)", "10,000,000.00", "Pending initial calculation")
    m2.metric("Portfolio Return", "-", "")
    m3.metric("Benchmark", "-", "")

m4.metric("Active Holdings", f"{len(active_positions)}" if not active_positions.empty else "0")

# ==========================================
# 4.5 INSTITUTIONAL RISK & ANALYTICS
# ==========================================
st.markdown("---")
st.subheader("🛡️ Risk & Performance Analytics")

if not metrics_df.empty and len(metrics_df) > 1:
    # 1. Fetch DB-stored rolling metrics
    latest_risk = metrics_df.iloc[-1]
    vol = latest_risk.get('volatility_annualized')
    var = latest_risk.get('var_95')
    beta = latest_risk.get('beta_30d')
    
    # 2. Calculate Max Drawdown dynamically
    # Formula: (Trough Value / Peak Value) - 1
    rolling_max = metrics_df['total_aum'].cummax()
    drawdowns = metrics_df['total_aum'] / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # 3. Calculate Sharpe Ratio dynamically
    # Formula: (Annualized Return - Risk Free Rate) / Annualized Volatility
    sharpe_ratio = 0.0
    days_active = len(metrics_df)
    risk_free_rate = 0.03 # 3% assumption
    
    if pd.notna(vol) and vol > 0 and days_active > 0:
        # Convert total return into an annualized return
        ann_return = (1 + portfolio_total_ret) ** (252 / days_active) - 1
        sharpe_ratio = (ann_return - risk_free_rate) / vol

    # 4. Format strings safely
    vol_str = f"{vol*100:.2f}%" if pd.notna(vol) else "Computing..."
    var_str = f"{var*100:.2f}%" if pd.notna(var) else "Computing..."
    beta_str = f"{beta:.2f}" if pd.notna(beta) else "Computing..."
    
    # 5. Render the UI Strip
    r1, r2, r3, r4, r5 = st.columns(5)
    
    r1.metric("Annualized Volatility", vol_str, 
              help="Standard deviation of daily returns, annualized. Measures overall price fluctuation.")
    
    r2.metric(f"Beta (30d) vs {benchmark.split('.')[0]}", beta_str, 
              help=">1.0 indicates higher volatility than the benchmark. <1.0 indicates lower volatility.")
    
    r3.metric("Max Drawdown", f"{max_drawdown*100:.2f}%", 
              help="The largest single peak-to-trough drop in the portfolio's history.")
    
    r4.metric("Value at Risk (95%)", var_str, 
              help="The expected worst daily loss on 95% of normal trading days.")
    
    r5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", 
              help="Risk-adjusted return. >1.0 is generally considered good, >2.0 is excellent.")
    
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info("⏳ Advanced risk metrics will populate after the first week of trading data is collected.")


# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab_perf, tab_holdings, tab_rebalance = st.tabs(["📈 Performance & Equity Curve", "📋 Current Allocation", "⚖️ Execute Rebalance"])

# --- TAB 1: EQUITY CURVE ---
with tab_perf:
    if not metrics_df.empty and len(metrics_df) > 1:
        st.subheader("Cumulative Return Comparison (%)")
        
        fig_curve = go.Figure()
        
        # 1. Plot Portfolio Cumulative Return
        # Rebase to 0% at inception
        metrics_df['cum_return'] = ((metrics_df['total_aum'] / inception_aum) - 1) * 100
        
        fig_curve.add_trace(go.Scatter(
            x=metrics_df['trade_date'], 
            y=metrics_df['cum_return'],
            mode='lines',
            name='Portfolio',
            fill='tozeroy',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # 2. Plot Benchmark Cumulative Return
        if bench_df is not None and not bench_df.empty:
            bench_df = bench_df.copy()
            first_close = bench_df['Close'].iloc[0]
            bench_df['bench_cum_return'] = ((bench_df['Close'] / first_close) - 1) * 100
            
            # Align dates
            bench_df['date_str'] = bench_df.index.strftime('%Y-%m-%d')
            
            fig_curve.add_trace(go.Scatter(
                x=bench_df['date_str'], 
                y=bench_df['bench_cum_return'],
                mode='lines',
                name=f'Benchmark ({benchmark.split(".")[0]})',
                line=dict(color='#9ca3af', width=2, dash='dot')
            ))

        fig_curve.update_layout(
            height=450,
            hovermode="x unified",
            xaxis_title="",
            yaxis_title="Cumulative Return (%)",
            template='plotly_white',
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Format the Y-axis to show percentage signs
        fig_curve.update_yaxes(ticksuffix="%")
        
        st.plotly_chart(fig_curve, use_container_width=True)

        # ==========================================
        # 3. Plot Rolling 30-Day Correlation
        # ==========================================
        st.markdown("---")
        st.subheader("🔄 Rolling 30-Day Correlation vs Benchmark")
        
        if bench_df is not None and not bench_df.empty and len(metrics_df) > 30:
            # 1. Calculate clean daily returns for the fund (excluding cash flows)
            metrics_df['prev_aum'] = metrics_df['total_aum'].shift(1)
            metrics_df['fund_ret'] = (metrics_df['total_aum'] - metrics_df['prev_aum'] - metrics_df['net_flow']) / metrics_df['prev_aum']
            
            # 2. Extract benchmark returns
            bench_returns = bench_df['Pct_Change'] / 100.0
            bench_returns.index = bench_returns.index.strftime('%Y-%m-%d')
            
            # 3. Merge and calculate rolling correlation
            corr_df = metrics_df[['trade_date', 'fund_ret']].set_index('trade_date')
            corr_df = corr_df.join(bench_returns.rename('bench_ret'), how='inner').dropna()
            
            # The Pandas magic: 30-day rolling Pearson correlation
            corr_df['rolling_corr'] = corr_df['fund_ret'].rolling(window=30).corr(corr_df['bench_ret'])
            
            # 4. Plot the chart
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=corr_df.index,
                y=corr_df['rolling_corr'],
                mode='lines',
                name='30-Day Correlation',
                line=dict(color='#8b5cf6', width=2),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ))
            
            # Add horizontal reference lines
            fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, annotation_text="Zero Correlation")
            fig_corr.add_hline(y=1, line_color="rgba(239, 68, 68, 0.3)", opacity=0.5)
            fig_corr.add_hline(y=-1, line_color="rgba(239, 68, 68, 0.3)", opacity=0.5)
            
            fig_corr.update_layout(
                height=300,
                hovermode="x unified",
                yaxis=dict(
                    range=[-1.1, 1.1], 
                    title="Pearson Correlation (r)",
                    tickmode='array',
                    tickvals=[-1, -0.5, 0, 0.5, 1]
                ),
                template='plotly_white',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("⏳ The Rolling Correlation chart will appear once the fund has at least 30 days of trading history.")
    else:
        st.info("⏳ Performance history will populate here after the first nightly NAV calculation.")


# --- TAB 2: CURRENT ALLOCATION & VALUATION ---
with tab_holdings:
    st.subheader("Current Holdings & Valuation")
    
    # We already have active_positions globally! Just check if it's empty.
    if active_positions.empty:
        st.info("No active positions found in this fund. Use the Rebalance tab to allocate capital.")
    else:
        # 1. Merge with Local stock_basic for Name and Industry
        stock_basic_df = db.read_table('stock_basic')
        if not stock_basic_df.empty:
            active_positions = active_positions.merge(
                stock_basic_df[['ts_code', 'name', 'industry']], 
                on='ts_code', 
                how='left'
            )
        else:
            active_positions['name'] = active_positions['ts_code']
            active_positions['industry'] = 'Unknown'
            
        active_positions['industry'] = active_positions['industry'].fillna('Other')

        # 2. Fetch Valuation from Local 'daily_basic' table
        with st.spinner("Fetching local valuation metrics..."):
            ts_codes = active_positions['ts_code'].tolist()
            metrics_list = []
            
            # Fetch the absolute latest daily_basic row for each active ticker
            for code in ts_codes:
                # Uses db_manager's native DESC order and limit features to grab the freshest row
                df = db.read_table('daily_basic', filters={'ts_code': code}, order_by='-trade_date', limit=1)
                if not df.empty:
                    metrics_list.append(df.iloc[0])
                    
            if metrics_list:
                daily_basic_df = pd.DataFrame(metrics_list)
                
                # Check which columns actually exist in your local table to prevent KeyErrors
                target_cols = ['close', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'trade_date']
                available_cols = ['ts_code'] + [c for c in target_cols if c in daily_basic_df.columns]
                
                active_positions = active_positions.merge(
                    daily_basic_df[available_cols], 
                    on='ts_code', 
                    how='left'
                )
                
        # Ensure all required UI columns exist (fills with None if data was missing)
        expected_cols = ['close', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'trade_date']
        for col in expected_cols:
            if col not in active_positions.columns:
                active_positions[col] = None

        # ==========================================
        # VISUALIZATION: SIDE-BY-SIDE PIE CHARTS
        # ==========================================
        col_pie1, col_pie2 = st.columns(2)
        
        with col_pie1:
            fig_stock = px.pie(
                active_positions, 
                values='weight', 
                names='name', 
                title="Allocation by Stock", 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_stock.update_traces(textposition='inside', textinfo='percent+label')
            fig_stock.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_stock, use_container_width=True)
            
        with col_pie2:
            industry_weights = active_positions.groupby('industry')['weight'].sum().reset_index()
            fig_ind = px.pie(
                industry_weights, 
                values='weight', 
                names='industry', 
                title="Allocation by Industry", 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_ind.update_traces(textposition='inside', textinfo='percent+label')
            fig_ind.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_ind, use_container_width=True)

        # ==========================================
        # DATA TABLE: ENRICHED HOLDINGS
        # ==========================================
        st.markdown("---")
        
        display_df = active_positions[[
            'ts_code', 'name', 'industry', 'weight', 
            'close', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'trade_date'
        ]].copy()
        
        st.dataframe(
            display_df,
            column_config={
                "ts_code": "Ticker",
                "name": "Company",
                "industry": "Industry",
                "weight": st.column_config.NumberColumn("Target Wgt", format="%.2f%%", help="Target allocation percentage"),
                "close": st.column_config.NumberColumn("Close (¥)", format="%.2f"),
                "pe": st.column_config.NumberColumn("PE", format="%.2f", help="Price to Earnings"),
                "pe_ttm": st.column_config.NumberColumn("PE (TTM)", format="%.2f", help="Trailing Twelve Months PE"),
                "pb": st.column_config.NumberColumn("PB", format="%.2f", help="Price to Book"),
                "ps": st.column_config.NumberColumn("PS", format="%.2f", help="Price to Sales"),
                "ps_ttm": st.column_config.NumberColumn("PS (TTM)", format="%.2f"),
                "trade_date": "Trade Date"
            },
            hide_index=True,
            use_container_width=True
        )

# --- TAB 3: THE REBALANCING ENGINE ---
with tab_rebalance:
    @st.fragment
    def rebalance_ui(fund_id, current_positions_df):
        st.write("Adjust the target weights below. Saving will cleanly terminate the current positions as of today, and launch the new weights effective tomorrow market open.")
        
        if current_positions_df.empty:
            st.warning("No positions to rebalance.")
            return

        # Prepare editor data
        edit_df = current_positions_df[['ts_code', 'weight']].copy()
        edit_df = edit_df.rename(columns={'ts_code': 'Ticker', 'weight': 'Target_Weight'})
        edit_df['Target_Weight'] = edit_df['Target_Weight'].round(4)
        
        col_ed, col_act = st.columns([2, 1])
        
        with col_ed:
            edited_reb = st.data_editor(
                edit_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker (e.g. 600519 or 600519.SH)", required=True),
                    "Target_Weight": st.column_config.NumberColumn("New Target Weight", required=True, min_value=0.0, max_value=1.0, format="%.4f", step=0.01)
                },
                num_rows="dynamic", # Add new stocks or delete old ones
                use_container_width=True,
                key=f"reb_editor_{fund_id}"
            )
            
            # --- 1. Ticker Validation Engine ---
            invalid_tickers = []
            valid_positions = {}
            
            for _, row in edited_reb.iterrows():
                raw_ticker = str(row['Ticker']).strip()
                weight = float(row.get('Target_Weight', 0.0))
                
                if raw_ticker and weight > 0:
                    # Automatically append .SH/.SZ/.BJ if the user forgot it
                    ts_code = data_manager.get_tushare_ticker(raw_ticker)
                    
                    # Verify it exists in our local stock_basic database
                    name = data_manager.get_stock_name_from_db(ts_code)
                    
                    if name is None:
                        invalid_tickers.append(raw_ticker)
                    else:
                        valid_positions[ts_code] = weight

            # --- 2. Weight Validation Engine ---
            total_w = edited_reb['Target_Weight'].sum()
            is_valid_weight = np.isclose(total_w, 1.0, atol=0.001)
            
            # --- 3. UI Feedback ---
            if not is_valid_weight:
                st.warning(f"⚖️ Total Weight: **{total_w*100:.2f}%**. Must equal exactly 100%.")
                
            if invalid_tickers:
                st.error(f"❌ Invalid Ticker(s) detected: **{', '.join(invalid_tickers)}**. Not found in A-share database.")

            # Lock the save button unless everything is perfect
            can_save = is_valid_weight and len(invalid_tickers) == 0
                
        with col_act:
            st.markdown("### Execution")
            if st.button("🚀 Execute Rebalance", type="primary", use_container_width=True, disabled=not can_save):
                with st.spinner("Locking in new mandate..."):
                    
                    # Send the strictly validated dictionary to the DB
                    success, msg = data_manager.execute_fund_rebalance(fund_id, valid_positions)
                    if success:
                        st.balloons()
                        st.success(msg)
                        st.rerun() # Refresh the page to show the new reality
                    else:
                        st.error(msg)
                        
    # Render the fragment
    rebalance_ui(fund_id, active_positions)