"""
Portfolio Optimization Page
Mean-Variance Optimization for A-share stocks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_manager
import tushare as ts
import json
import os
import time
from scipy.optimize import minimize
import api_config

import auth_manager
auth_manager.require_login()


st.set_page_config(
    page_title="💼 Portfolio | 投资组合优化",
    page_icon="💼",
    layout="wide"
)

# ==========================================
# CONSTANTS
# ==========================================

STOCK_NAME_CACHE_FILE = 'stock_names_cache.json'
TUSHARE_TOKEN = api_config.TUSHARE_TOKEN

# ==========================================
# TUSHARE API
# ==========================================

@st.cache_resource
def get_tushare_api():
    """Initialize Tushare API (cached)."""
    return ts.pro_api(TUSHARE_TOKEN)


# ==========================================
# STOCK NAME FUNCTIONS (with 北交所 support)
# ==========================================

def ticker_to_ts_code(ticker: str) -> str:
    """
    Convert 6-digit ticker to full ts_code format.
    Supports Shanghai (SH), Shenzhen (SZ), and Beijing (BJ) exchanges.
    """
    # Remove any existing suffix
    ticker = ticker.split('.')[0]
    
    # Beijing Stock Exchange (北交所)
    if ticker.startswith(('43', '83', '87')):
        return f"{ticker}.BJ"
    
    # Shenzhen
    elif ticker.startswith(('0', '2', '3')):
        return f"{ticker}.SZ"
    
    # Shanghai
    else:
        return f"{ticker}.SH"


def get_stock_name_from_tushare(ts_code: str, pro) -> str:
    """
    Get company name from Tushare (FREE API).
    Falls back to ts_code if failed.
    """
    try:
        df = pro.stock_company(ts_code=ts_code, fields='ts_code,com_name')
        if df is not None and not df.empty:
            return df.iloc[0]['com_name']
    except Exception as e:
        # Silent fail - will return ts_code
        pass
    
    return ts_code.split('.')[0]  # Return ticker without suffix


def load_or_fetch_stock_name(ticker: str, pro) -> str:
    """
    Get stock name with local caching.
    Automatically handles SH/SZ/BJ exchanges.
    """
    ts_code = ticker_to_ts_code(ticker)
    
    # Load cache
    cache = {}
    if os.path.exists(STOCK_NAME_CACHE_FILE):
        try:
            with open(STOCK_NAME_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except:
            cache = {}
    
    # Check cache first
    if ts_code in cache:
        return cache[ts_code]
    
    # Fetch from Tushare (FREE but rate-limited)
    time.sleep(0.31)  # Rate limit: 200 calls/min
    
    name = get_stock_name_from_tushare(ts_code, pro)
    
    # Save to cache
    cache[ts_code] = name
    try:
        with open(STOCK_NAME_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except:
        pass
    
    return name


def get_stock_display_name(ticker: str) -> str:
    """Get display name from stock_basic table in database."""
    name = data_manager.get_stock_name_from_db(ticker)
    if name:
        return f"{name} ({ticker})"
    else:
        return ticker




# ==========================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# ==========================================

@st.cache_data(ttl=600)
def calculate_returns_covariance(tickers: list, lookback: int = 252):
    """Calculate returns and covariance matrix using LIVE qfq-adjusted data."""
    price_data = {}
    for ticker in tickers:
        # Use get_single_stock_data_live for qfq-adjusted prices
        df = data_manager.get_single_stock_data_live(ticker, lookback_years=2)
        if df is not None and not df.empty:
            price_data[ticker] = df['Close']
    
    if not price_data:
        return None, None, None
    
    prices = pd.DataFrame(price_data).dropna()
    returns = prices.pct_change().dropna().tail(lookback)
    mean_returns = returns.mean() * 252  # Annualize
    cov_matrix = returns.cov() * 252
    
    return returns, mean_returns, cov_matrix



def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility."""
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    """Negative Sharpe ratio for minimization."""
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std


def optimize_portfolio(mean_returns, cov_matrix, target_return=None, 
                      max_weight=0.3, min_weight=0.0, risk_free_rate=0.03):
    """
    Optimize portfolio using mean-variance optimization.
    
    If target_return is None: Maximize Sharpe ratio
    Otherwise: Minimize variance for target return
    """
    n_assets = len(mean_returns)
    
    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if target_return is not None:
        # Add return target constraint
        constraints.append({
            'type': 'eq', 
            'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return
        })
    
    # Bounds for each weight
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    
    # Initial guess (equal weight)
    init_guess = np.array([1.0 / n_assets] * n_assets)
    
    # Optimize
    if target_return is None:
        # Maximize Sharpe ratio
        result = minimize(
            negative_sharpe,
            init_guess,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    else:
        # Minimize variance
        result = minimize(
            lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1],
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    return result


def calculate_efficient_frontier(mean_returns, cov_matrix, n_points=50, 
                                 max_weight=0.3, min_weight=0.0):
    """Calculate efficient frontier points."""
    min_return = mean_returns.min()
    max_return = mean_returns.max()
    
    target_returns = np.linspace(min_return, max_return, n_points)
    
    frontier_volatility = []
    frontier_returns = []
    
    for target in target_returns:
        try:
            result = optimize_portfolio(
                mean_returns, cov_matrix, 
                target_return=target,
                max_weight=max_weight,
                min_weight=min_weight
            )
            
            if result.success:
                p_return, p_std = portfolio_performance(result.x, mean_returns, cov_matrix)
                frontier_returns.append(p_return)
                frontier_volatility.append(p_std)
        except:
            continue
    
    return frontier_volatility, frontier_returns


def create_efficient_frontier_chart(frontier_vol, frontier_ret, 
                                   optimal_vol, optimal_ret,
                                   individual_vol, individual_ret, stock_names):
    """Create efficient frontier visualization."""
    fig = go.Figure()
    
    # Efficient frontier curve
    fig.add_trace(go.Scatter(
        x=frontier_vol, y=frontier_ret,
        mode='lines', name='Efficient Frontier',
        line=dict(color='#3b82f6', width=3)
    ))
    
    # Optimal portfolio point
    fig.add_trace(go.Scatter(
        x=[optimal_vol], y=[optimal_ret],
        mode='markers', name='Optimal Portfolio',
        marker=dict(color='#ef4444', size=15, symbol='star')
    ))
    
    # Individual stocks
    fig.add_trace(go.Scatter(
        x=individual_vol, y=individual_ret,
        mode='markers', name='Individual Stocks',
        marker=dict(color='#6b7280', size=10, symbol='circle'),
        text=stock_names,
        hovertemplate='<b>%{text}</b><br>Return: %{y:.1%}<br>Risk: %{x:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Efficient Frontier & Optimal Portfolio',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        height=600,
        template='plotly_white',
        hovermode='closest'
    )
    
    fig.update_xaxes(tickformat='.0%')
    fig.update_yaxes(tickformat='.0%')
    
    return fig

# ==========================================
# ADVANCED RISK METRICS (ADD AFTER EXISTING FUNCTIONS)
# ==========================================

def effective_number_of_bets(weights):
    """
    Calculate Effective Number of Bets (ENB)
    Measures true diversification (not just number of assets)
    ENB = 1 / sum(weights^2)
    Higher is better - closer to total number of stocks = better diversification
    """
    enb = 1 / np.sum(weights ** 2)
    return enb


def diversification_ratio(weights, cov_matrix):
    """
    Diversification Ratio = Weighted avg volatility / Portfolio volatility
    Higher is better (>1 means diversification benefit exists)
    Measures how much risk reduction you get from combining assets
    """
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vol = np.sum(weights * individual_vols)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return weighted_vol / portfolio_vol


def calculate_var(returns_df, weights, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR)
    Returns the maximum expected loss at given confidence level
    E.g., VaR 95% = worst loss you can expect on 95% of days
    """
    portfolio_returns = (returns_df @ weights)
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    return var


def calculate_cvar(returns_df, weights, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
    Average loss beyond VaR threshold
    More comprehensive than VaR as it captures tail risk severity
    """
    portfolio_returns = (returns_df @ weights)
    var = calculate_var(returns_df, weights, confidence_level)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return cvar


def calculate_tail_ratio(returns_df, weights, confidence_level=0.95):
    """
    Tail Ratio = abs(95th percentile) / abs(5th percentile)
    Measures asymmetry between gains and losses
    Higher is better (bigger upside relative to downside)
    >1 = upside exceeds downside, <1 = downside exceeds upside
    """
    portfolio_returns = (returns_df @ weights)
    upper_tail = np.percentile(portfolio_returns, confidence_level * 100)
    lower_tail = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    tail_ratio = abs(upper_tail) / abs(lower_tail) if lower_tail != 0 else np.nan
    return tail_ratio


def worst_drawdowns(returns_df, weights, n=5):
    """
    Calculate worst single loss and average of worst n losses
    Shows extreme downside scenarios actually experienced in data
    """
    portfolio_returns = (returns_df @ weights)
    sorted_returns = np.sort(portfolio_returns)
    worst_loss = sorted_returns[0]
    avg_worst_n = sorted_returns[:n].mean()
    return worst_loss, avg_worst_n


def generate_portfolio_assessment(optimal_return, optimal_vol, optimal_sharpe, 
                                  weights, returns_df, cov_matrix, tickers):
    """
    Generate intelligent assessment of portfolio quality.
    Returns verdict, score, and detailed analysis.
    """
    assessment = {
        'verdict': '',
        'color': '',
        'score': 0,
        'reasons': [],
        'warnings': [],
        'strengths': []
    }
    
    score = 0
    max_score = 100
    
    # Calculate all metrics
    enb = effective_number_of_bets(weights)
    div_ratio = diversification_ratio(weights, cov_matrix)
    
    var_95 = calculate_var(returns_df, weights, 0.95)
    var_99 = calculate_var(returns_df, weights, 0.99)
    
    cvar_95 = calculate_cvar(returns_df, weights, 0.95)
    cvar_99 = calculate_cvar(returns_df, weights, 0.99)
    
    tail_95 = calculate_tail_ratio(returns_df, weights, 0.95)
    tail_99 = calculate_tail_ratio(returns_df, weights, 0.99)
    
    worst_loss, avg_worst_5 = worst_drawdowns(returns_df, weights, 5)
    
    # Store metrics for display
    metrics = {
        'ENB': enb,
        'Div Ratio': div_ratio,
        'VaR 95%': var_95,
        'VaR 99%': var_99,
        'CVaR 95%': cvar_95,
        'CVaR 99%': cvar_99,
        'Tail 95%': tail_95,
        'Tail 99%': tail_99,
        'Worst Loss': worst_loss,
        'Avg Worst 5': avg_worst_5
    }
    
    # ==========================================
    # SCORING SYSTEM
    # ==========================================
    
    # 1. Sharpe Ratio Assessment (20 points)
    if optimal_sharpe > 1.5:
        score += 20
        assessment['strengths'].append(f"🎯 Excellent risk-adjusted returns (Sharpe: {optimal_sharpe:.2f})")
    elif optimal_sharpe > 1.0:
        score += 15
        assessment['strengths'].append(f"✅ Strong risk-adjusted returns (Sharpe: {optimal_sharpe:.2f})")
    elif optimal_sharpe > 0.5:
        score += 10
        assessment['reasons'].append(f"📊 Moderate risk-adjusted returns (Sharpe: {optimal_sharpe:.2f})")
    else:
        score += 5
        assessment['warnings'].append(f"⚠️ Below-average risk-adjusted returns (Sharpe: {optimal_sharpe:.2f})")
    
    # 2. Diversification Quality (20 points)
    n_assets = len(tickers)
    diversification_pct = (enb / n_assets) * 100
    
    if diversification_pct > 70:
        score += 20
        assessment['strengths'].append(f"🌟 Excellent diversification ({enb:.1f} effective bets from {n_assets} assets)")
    elif diversification_pct > 50:
        score += 15
        assessment['strengths'].append(f"✅ Good diversification ({enb:.1f} effective bets)")
    elif diversification_pct > 30:
        score += 10
        assessment['reasons'].append(f"📊 Moderate diversification ({enb:.1f} effective bets)")
    else:
        score += 5
        assessment['warnings'].append(f"⚠️ Concentrated portfolio ({enb:.1f} effective bets) - high single-asset risk")
    
    # 3. Diversification Benefit (15 points)
    if div_ratio > 1.3:
        score += 15
        assessment['strengths'].append(f"💪 Strong diversification benefit (Ratio: {div_ratio:.2f})")
    elif div_ratio > 1.15:
        score += 10
        assessment['strengths'].append(f"✅ Meaningful diversification benefit (Ratio: {div_ratio:.2f})")
    elif div_ratio > 1.0:
        score += 5
        assessment['reasons'].append(f"📊 Minimal diversification benefit (Ratio: {div_ratio:.2f})")
    else:
        assessment['warnings'].append(f"⚠️ No diversification benefit (Ratio: {div_ratio:.2f}) - assets highly correlated")
    
    # 4. Tail Risk Assessment (25 points)
    if tail_95 > 1.2 and tail_99 > 1.2:
        score += 25
        assessment['strengths'].append(f"🚀 Favorable risk asymmetry (upside > downside in extreme scenarios)")
    elif tail_95 > 1.0 and tail_99 > 1.0:
        score += 15
        assessment['reasons'].append(f"📊 Balanced upside/downside profile")
    elif tail_95 < 0.8 or tail_99 < 0.8:
        score += 5
        assessment['warnings'].append(f"⚠️ Negative skew: downside risk exceeds upside potential")
    else:
        score += 10
    
    # 5. Extreme Loss Risk (20 points)
    worst_loss_pct = worst_loss * 100
    avg_worst_5_pct = avg_worst_5 * 100
    
    if worst_loss_pct > -3 and avg_worst_5_pct > -2:
        score += 20
        assessment['strengths'].append(f"🛡️ Low extreme loss risk (worst day: {worst_loss_pct:.2f}%)")
    elif worst_loss_pct > -5 and avg_worst_5_pct > -3:
        score += 15
        assessment['reasons'].append(f"📊 Manageable extreme loss risk (worst day: {worst_loss_pct:.2f}%)")
    elif worst_loss_pct > -8:
        score += 8
        assessment['warnings'].append(f"⚠️ Moderate extreme loss risk (worst day: {worst_loss_pct:.2f}%)")
    else:
        assessment['warnings'].append(f"🚨 High extreme loss risk (worst day: {worst_loss_pct:.2f}%) - potential for severe drawdowns")
    
    # Final Verdict
    assessment['score'] = score
    
    if score >= 80:
        assessment['verdict'] = "✅ STRONG PORTFOLIO - RECOMMENDED"
        assessment['color'] = "success"
        assessment['summary'] = "This portfolio demonstrates strong risk-adjusted returns, excellent diversification, and manageable tail risks. **It is suitable for allocation.**"
    elif score >= 60:
        assessment['verdict'] = "⚠️ ACCEPTABLE PORTFOLIO - PROCEED WITH CAUTION"
        assessment['color'] = "warning"
        assessment['summary'] = "This portfolio shows acceptable characteristics but has some weaknesses. **Review the warnings carefully before committing capital.**"
    elif score >= 40:
        assessment['verdict'] = "🔶 MARGINAL PORTFOLIO - SIGNIFICANT CONCERNS"
        assessment['color'] = "warning"
        assessment['summary'] = "This portfolio has notable weaknesses in risk metrics or diversification. **Consider alternative allocations or different stock combinations.**"
    else:
        assessment['verdict'] = "❌ WEAK PORTFOLIO - NOT RECOMMENDED"
        assessment['color'] = "error"
        assessment['summary'] = "This portfolio exhibits poor risk characteristics across multiple dimensions. **Strongly reconsider this combination or optimization approach.**"
    
    assessment['metrics'] = metrics
    return assessment


# ==========================================
# MAIN APP
# ==========================================

st.header("💼 Portfolio Optimization")
st.markdown("**Mean-Variance Optimization** for A-share stocks.")

# Initialize Tushare API
pro = get_tushare_api()

# ==========================================
# STOCK INPUT
# ==========================================

# ==========================================
# STOCK INPUT
# ==========================================

# ==========================================
# STOCK INPUT
# ==========================================

st.subheader("1️⃣ Select Stocks")
st.markdown("Load an existing mandate to re-optimize, or enter stock tickers manually.")

import auth_manager
from db_manager import db
import pandas as pd

# 1. Initialize session state for the text area
if 'ticker_input_val' not in st.session_state:
    st.session_state.ticker_input_val = ""

# 2. Callback function (with explicit type casting and error tracking)
def load_fund_callback():
    try:
        selected_fund = st.session_state.fund_dropdown
        if selected_fund != "-- Select an existing mandate --":
            
            # Fetch user ID cleanly inside the callback
            uid = auth_manager.get_current_user_id()
            user_funds = db.read_table('funds', filters={'user_id': uid})
            
            if not user_funds.empty:
                match = user_funds[user_funds['fund_name'] == selected_fund]
                if not match.empty:
                    # CRITICAL FIX: Explicitly cast to Python int to prevent SQLite numpy errors
                    fund_id = int(match.iloc[0]['id']) 
                    pos_df = db.read_table('fund_positions', filters={'fund_id': fund_id})
                    
                    if not pos_df.empty:
                        # Extract the 6-digit tickers
                        tickers = pos_df['ts_code'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
                        # Push the tickers directly into the text area state
                        st.session_state.ticker_input_val = ", ".join(tickers)
                        # Visual confirmation
                        st.toast(f"✅ Successfully loaded {len(tickers)} tickers from '{selected_fund}'")
                    else:
                        st.toast(f"⚠️ '{selected_fund}' has no saved positions.")
                        st.session_state.ticker_input_val = ""
        else:
            # Clear the box if they select the default option
            st.session_state.ticker_input_val = ""
            st.toast("🧹 Cleared stock list.")
            
    except Exception as e:
        # If it fails, show exactly why in the UI
        st.toast(f"❌ Error loading fund: {str(e)}")

# 3. Fetch funds for the dropdown options
user_id = auth_manager.get_current_user_id()
user_funds_df = db.read_table('funds', filters={'user_id': user_id})

fund_names = ["-- Select an existing mandate --"]
if not user_funds_df.empty:
    fund_names += user_funds_df['fund_name'].tolist()

# 4. The Vertical Dropdown (Triggering the callback)
st.selectbox(
    "Load from Mandate:", 
    options=fund_names, 
    key="fund_dropdown",
    on_change=load_fund_callback
)

# 5. The Vertical Text Area
ticker_input = st.text_area(
    "Enter / Edit Tickers (comma-separated or one per line):", 
    key="ticker_input_val",
    height=100,
    placeholder="e.g., 600519, 000858, 000001\nOr paste a list here..."
)

# 6. Process the inputs into a clean list
selected_tickers = []
if ticker_input:
    # Handle both commas and newlines gracefully
    tickers_raw = ticker_input.replace('\n', ',').split(',')
    selected_tickers = [t.strip() for t in tickers_raw if t.strip() and t.strip().isdigit()]

# Remove duplicates
selected_tickers = list(set(selected_tickers))

# 7. Validation
if not selected_tickers:
    st.info("👆 Enter at least 3 stock tickers or load a mandate to begin")
    st.stop()

if len(selected_tickers) < 3:
    st.warning("⚠️ Please enter at least 3 stocks for portfolio optimization")
    st.stop()
# ==========================================
# FETCH STOCK NAMES
# ==========================================

st.subheader("2️⃣ Stock Information")


with st.spinner(f"Loading company names for {len(selected_tickers)} stocks..."):
    stock_names = {}
    for ticker in selected_tickers:
        stock_names[ticker] = data_manager.get_stock_name_from_db(ticker) or ticker


# Display stock table
stock_display = []
for ticker in selected_tickers:
    ts_code = ticker_to_ts_code(ticker)
    exchange = ts_code.split('.')[1]
    
    exchange_name = {
        'SH': '上海',
        'SZ': '深圳',
        'BJ': '北京'
    }.get(exchange, exchange)
    
    stock_display.append({
        'Ticker': ticker,
        'Company Name': stock_names[ticker],
        'Exchange': exchange_name,
        'TS Code': ts_code
    })

stock_df = pd.DataFrame(stock_display)
st.dataframe(stock_df, use_container_width=True, hide_index=True)

# Allow user to deselect stocks
tickers_to_use = st.multiselect(
    "Confirm stocks to include (or remove unwanted):",
    selected_tickers,
    default=selected_tickers,
    format_func=lambda x: f"{stock_names[x]} ({x})"
)

if not tickers_to_use or len(tickers_to_use) < 3:
    st.warning("⚠️ Please select at least 3 stocks")
    st.stop()


# ==========================================
# OPTIMIZATION PARAMETERS
# ==========================================

st.subheader("3️⃣ Optimization Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    lookback_days = st.selectbox(
        "Lookback Period (days):",
        [60, 90, 120, 180, 252],
        index=4,
        help="Historical period for calculating returns and covariance"
    )

with col2:
    max_allocation_pct = st.slider(
        "Max allocation per stock (%):",
        min_value=5,
        max_value=50,
        value=30,
        step=5,
        format="%d%%",
        help="Maximum weight any single stock can have in portfolio"
    )
    max_allocation = max_allocation_pct / 100  # Convert to decimal

with col3:
    risk_free_rate_pct = st.slider(
        "Risk-free rate (%):",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        format="%.1f%%",
        help="Annual risk-free rate for Sharpe ratio calculation"
    )
    risk_free_rate = risk_free_rate_pct / 100  # Convert to decimal


@st.fragment
def create_mandate_ui(tickers_to_use, optimal_weights):
    """
    Segmented UI for creating a mandate. 
    Interacting with the grid here will ONLY rerun this fragment, not the whole page!
    """
    st.markdown("---")
    st.subheader("🏦 Create Institutional Fund Mandate")
    st.info("Convert this optimized portfolio into a tracked fund. You can manually override the final weights or add/remove tickers before saving.")

    # Match tickers to their optimized weights
    weights_dict = dict(zip(tickers_to_use, optimal_weights))
    
    # Filter out near-zero weights (less than 0.1%)
    clean_weights = {k: v for k, v in weights_dict.items() if v > 0.001}
    
    # Format for the interactive dataframe
    df_weights = pd.DataFrame(list(clean_weights.items()), columns=['ticker', 'Suggested_Weight'])
    df_weights['Final_Weight'] = df_weights['Suggested_Weight'].round(4)
    
    col_edit, col_save = st.columns([2, 1])
    
    with col_edit:
        st.write("**Adjust Target Weights:**")
        edited_portfolio = st.data_editor(
            df_weights,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker (e.g. 600519)", required=True),
                "Suggested_Weight": st.column_config.NumberColumn("Model Output", disabled=True, format="%.4f"),
                "Final_Weight": st.column_config.NumberColumn("Target Weight", required=True, min_value=0.0, max_value=1.0, format="%.4f")
            },
            num_rows="dynamic", # Enables the '+' icon to add new rows
            use_container_width=True,
            key="portfolio_editor" 
        )
        
        # Real-time Mathematical Validation (Only reruns inside this fragment!)
        total_weight = edited_portfolio['Final_Weight'].sum()
        if not np.isclose(total_weight, 1.0, atol=0.001):
            st.warning(f"⚖️ Current Total Weight: **{total_weight*100:.2f}%**. Please adjust to exactly 100%.")
            is_valid = False
        else:
            st.success("✅ Allocation valid (100.00%).")
            is_valid = True

    with col_save:
        st.write("**Fund Details:**")
        fund_name = st.text_input("Fund Name", placeholder="e.g., Alpha Quant Q1")
        benchmark = st.selectbox("Benchmark", ["000300.SH (CSI 300)", "000905.SH (CSI 500)", "000852.SH (CSI 1000)"])
        benchmark_ticker = benchmark.split(" ")[0] 
        
        if st.button("💾 Save as Mandate", type="primary", use_container_width=True, disabled=not is_valid):
            if not fund_name:
                st.error("Please enter a Fund Name.")
            else:
                with st.spinner("Writing mandate to database..."):
                    # Convert the edited grid back to a dictionary of valid ts_codes
                    final_positions = {}
                    for _, row in edited_portfolio.iterrows():
                        ticker = str(row['ticker']).strip()
                        if ticker:
                            ts_code = ticker_to_ts_code(ticker) 
                            final_positions[ts_code] = float(row['Final_Weight'])
                    
                    # Save to DB tied to the logged-in user
                    success, msg = data_manager.save_fund_mandate(fund_name, benchmark_ticker, final_positions)
                    
                    if success:
                        # st.balloons()
                        st.success(msg)
                    else:
                        st.error(msg)

# ==========================================
# RUN OPTIMIZATION
# ==========================================

if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):
    
    with st.spinner("📊 Calculating returns and covariance..."):
        returns_df, mean_returns, cov_matrix = calculate_returns_covariance(
            tickers_to_use, 
            lookback=lookback_days
        )
    
    if returns_df is None or mean_returns is None:
        st.error("❌ Failed to load price data. Please check if tickers are valid.")
        st.stop()
    
    # Check for stocks with no data
    missing_stocks = set(tickers_to_use) - set(returns_df.columns)
    if missing_stocks:
        st.warning(f"⚠️ Could not load data for: {', '.join(missing_stocks)}")
    
    # Use only stocks with data
    tickers_to_use = list(returns_df.columns)
    
    if len(tickers_to_use) < 3:
        st.error("❌ Not enough stocks with valid data")
        st.stop()
    
    with st.spinner("🔄 Running optimization..."):
        result = optimize_portfolio(
            mean_returns, 
            cov_matrix,
            max_weight=max_allocation,
            risk_free_rate=risk_free_rate
        )
    
    if not result.success:
        st.error(f"❌ Optimization failed: {result.message}")
        st.stop()
    
    # Get optimal weights
    optimal_weights = result.x
    optimal_return, optimal_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_vol
    
    # ==========================================
    # RESULTS
    # ==========================================
    
    st.subheader("📊 Optimization Results")

        # ==========================================
    # GENERATE PORTFOLIO ASSESSMENT
    # ==========================================
    
    assessment = generate_portfolio_assessment(
        optimal_return, optimal_vol, optimal_sharpe,
        optimal_weights, returns_df, cov_matrix, tickers_to_use
    )
    
    # ==========================================
    # RESULTS - ASSESSMENT FIRST
    # ==========================================
    
    st.subheader("🎯 Portfolio Assessment | 投资组合评估")
    
    # Display verdict with appropriate styling
    if assessment['color'] == 'success':
        st.success(f"### {assessment['verdict']}")
    elif assessment['color'] == 'error':
        st.error(f"### {assessment['verdict']}")
    else:
        st.warning(f"### {assessment['verdict']}")
    
    # Score and summary
    col_score, col_summary = st.columns([1, 3])
    
    with col_score:
        st.metric("Overall Score", f"{assessment['score']}/100", 
                 help="Composite score based on 5 risk dimensions")
    
    with col_summary:
        st.markdown(assessment['summary'])
    
    # Detailed findings
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if assessment['strengths']:
            st.markdown("**✅ Strengths:**")
            for strength in assessment['strengths']:
                st.markdown(f"- {strength}")
    
    with col2:
        if assessment['reasons']:
            st.markdown("**ℹ️ Considerations:**")
            for reason in assessment['reasons']:
                st.markdown(f"- {reason}")
    
    with col3:
        if assessment['warnings']:
            st.markdown("**⚠️ Warnings:**")
            for warning in assessment['warnings']:
                st.markdown(f"- {warning}")
    
    # ==========================================
    # ADVANCED RISK METRICS TABLE
    # ==========================================
    
    st.markdown("---")
    st.subheader("⚠️ Advanced Risk Metrics | 高级风险指标")
    
    metrics = assessment['metrics']
    
    risk_metrics_df = pd.DataFrame({
        'Metric | 指标': [
            'Effective Number of Bets | 有效头寸数',
            'Diversification Ratio | 分散化比率',
            'VaR 95% (Daily) | 风险价值95%',
            'VaR 99% (Daily) | 风险价值99%',
            'CVaR 95% (Daily) | 条件风险价值95%',
            'CVaR 99% (Daily) | 条件风险价值99%',
            'Tail Ratio 95% | 尾部比率95%',
            'Tail Ratio 99% | 尾部比率99%',
            'Worst Single Loss | 最大单日损失',
            'Avg of Worst 5 Days | 最差5天平均'
        ],
        'Value | 数值': [
            f"{metrics['ENB']:.2f}",
            f"{metrics['Div Ratio']:.3f}",
            f"{metrics['VaR 95%']*100:.2f}%",
            f"{metrics['VaR 99%']*100:.2f}%",
            f"{metrics['CVaR 95%']*100:.2f}%",
            f"{metrics['CVaR 99%']*100:.2f}%",
            f"{metrics['Tail 95%']:.3f}",
            f"{metrics['Tail 99%']:.3f}",
            f"{metrics['Worst Loss']*100:.2f}%",
            f"{metrics['Avg Worst 5']*100:.2f}%"
        ],
        'Interpretation | 解释': [
            f"True diversification: {metrics['ENB']:.1f} effective bets from {len(tickers_to_use)} stocks",
            "Higher is better (>1 means diversification works)" if metrics['Div Ratio'] > 1 else "⚠️ Low diversification benefit",
            "Expected max daily loss on 95% of days",
            "Expected max daily loss on 99% of days (extreme scenarios)",
            "Average loss when exceeding VaR 95%",
            "Average loss in worst 1% of days",
            "Upside/downside asymmetry (>1 favors upside)" if metrics['Tail 95%'] > 1 else "⚠️ Downside exceeds upside",
            "Extreme tail asymmetry",
            "Single worst daily return in historical data",
            "Average of 5 worst daily returns"
        ]
    })
    
    st.dataframe(risk_metrics_df, use_container_width=True, hide_index=True)
    
    # Explanation expander
    with st.expander("📖 Understanding Risk Metrics"):
        st.markdown("""
        **Effective Number of Bets (ENB)**: Measures true diversification. If you have 10 stocks but ENB=3, 
        you're really only diversified as if you held 3 independent assets (others are correlated).
        
        **Diversification Ratio**: Compares weighted individual volatilities to portfolio volatility. 
        >1 means combining assets reduces risk (diversification works).
        
        **VaR (Value at Risk)**: Maximum expected loss at confidence level. VaR 95% = -2% means on 95% of days, 
        you won't lose more than 2%.
        
        **CVaR (Conditional VaR)**: Average loss when you exceed VaR. More comprehensive than VaR alone.
        
        **Tail Ratio**: Compares upside potential to downside risk. >1 means bigger gains than losses in extremes.
        
        **Worst Loss Metrics**: Actual historical worst-case scenarios from your data period.
        """)
    
    # ==========================================
    # ORIGINAL RESULTS (KEEP YOUR EXISTING CODE)
    # ==========================================
    
    st.markdown("---")
    st.subheader("📊 Portfolio Details | 投资组合详情")
    
    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Return | 预期收益", f"{optimal_return:.2%}", help="Annualized expected return")
    m2.metric("Volatility | 波动率", f"{optimal_vol:.2%}", help="Annualized standard deviation")
    m3.metric("Sharpe Ratio | 夏普比率", f"{optimal_sharpe:.2f}", help="Risk-adjusted return")
    m4.metric("Holdings | 持仓数", f"{(optimal_weights > 0.01).sum()}", help="Stocks with >1% allocation")
    

    # Portfolio allocation
    st.subheader("🎯 Optimal Allocation")
    
    allocation_df = pd.DataFrame({
        'Ticker': tickers_to_use,
        'Company Name': [stock_names[t] for t in tickers_to_use],
        'Weight': optimal_weights
    })
    
    allocation_df = allocation_df[allocation_df['Weight'] > 0.001]  # Filter tiny weights
    allocation_df = allocation_df.sort_values('Weight', ascending=False)
    allocation_df['Weight %'] = (allocation_df['Weight'] * 100).map('{:.2f}%'.format)
    allocation_df['Amount (¥10K)'] = (allocation_df['Weight'] * 10000).map('{:.0f}'.format)
    
    st.dataframe(
        allocation_df[['Company Name', 'Ticker', 'Weight %', 'Amount (¥10K)']],
        use_container_width=True,
        hide_index=True
    )
    
    # Allocation pie chart
    col_pie, col_bar = st.columns(2)
    
    with col_pie:
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"{stock_names[t]} ({t})" for t in allocation_df['Ticker']],
            values=allocation_df['Weight'],
            hole=0.3
        )])
        fig_pie.update_layout(title='Portfolio Weights', height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_bar:
        fig_bar = go.Figure(data=[go.Bar(
            x=allocation_df['Weight'],
            y=[f"{stock_names[t]}" for t in allocation_df['Ticker']],
            orientation='h',
            marker=dict(color='#3b82f6')
        )])
        fig_bar.update_layout(
            title='Allocation by Stock',
            xaxis_title='Weight',
            yaxis_title='',
            height=400,
            xaxis_tickformat='.0%'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    

    # ==================== INDUSTRY EXPOSURE ANALYSIS ====================

    st.markdown("---")
    st.subheader("🏭 Industry Exposure")

    def get_industry_exposure(tickers, weights):
        """Calculate industry exposure from portfolio weights using db manager."""
        from db_manager import db
        
        industry_data = []
        
        for ticker, weight in zip(tickers, weights):
            ts_code = data_manager.get_tushare_ticker(ticker)
            try:
                # Use db manager's read_table method (works for both SQLite and Supabase)
                df = db.read_table('stock_basic', 
                                filters={'ts_code': ts_code}, 
                                columns='name,industry')
                if not df.empty:
                    industry = df.iloc[0]['industry']
                    name = df.iloc[0]['name']
                    industry_data.append({
                        'Ticker': ticker,
                        'Name': name,
                        'Industry': industry if pd.notna(industry) else '未分类',
                        'Weight': weight
                    })
            except Exception as e:
                # Fallback if stock_basic lookup fails
                industry_data.append({
                    'Ticker': ticker,
                    'Name': ticker,
                    'Industry': '未分类',
                    'Weight': weight
                })
        
        return pd.DataFrame(industry_data)

    # Get industry breakdown
    industry_df = get_industry_exposure(tickers_to_use, optimal_weights)

    # Aggregate by industry
    industry_summary = industry_df.groupby('Industry')['Weight'].sum().sort_values(ascending=False)
    industry_summary_pct = industry_summary * 100

    col1, col2 = st.columns([2, 3])

    with col1:
        # Industry exposure pie chart
        import plotly.express as px
        
        fig_industry = go.Figure(data=[go.Pie(
            labels=industry_summary.index,
            values=industry_summary_pct.values,
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(
                colors=px.colors.qualitative.Set3[:len(industry_summary)]
            )
        )])
        
        fig_industry.update_layout(
            title="Industry Allocation",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig_industry, use_container_width=True)

    with col2:
        # Detailed industry breakdown table
        st.markdown("**Industry Breakdown**")
        
        industry_detail = []
        for industry in industry_summary.index:
            stocks_in_industry = industry_df[industry_df['Industry'] == industry]
            stock_count = len(stocks_in_industry)
            total_weight = stocks_in_industry['Weight'].sum()
            stock_list = ', '.join(stocks_in_industry['Name'].head(3).tolist())
            if stock_count > 3:
                stock_list += f" (+{stock_count - 3} more)"
            
            industry_detail.append({
                'Industry': industry,
                'Stocks': stock_count,
                'Weight': f"{total_weight * 100:.1f}%",
                'Companies': stock_list
            })
        
        industry_table = pd.DataFrame(industry_detail)
        st.dataframe(industry_table, use_container_width=True, hide_index=True)
        
        # Risk concentration warning
        max_industry_weight = industry_summary.max()
        if max_industry_weight > 0.5:
            st.error(f"⚠️ High concentration: {industry_summary.idxmax()} represents {max_industry_weight*100:.1f}% of portfolio")
        elif max_industry_weight > 0.35:
            st.warning(f"⚠️ Moderate concentration: {industry_summary.idxmax()} represents {max_industry_weight*100:.1f}% of portfolio")
        else:
            st.success(f"✅ Well diversified across {len(industry_summary)} industries")

    # ==================== END INDUSTRY EXPOSURE ====================

    # Efficient Frontier
    st.subheader("📈 Efficient Frontier")
    
    with st.spinner("Calculating efficient frontier..."):
        frontier_vol, frontier_ret = calculate_efficient_frontier(
            mean_returns, cov_matrix, n_points=30, max_weight=max_allocation
        )
        
        individual_vol = np.sqrt(np.diag(cov_matrix))
        individual_ret = mean_returns.values
        
        fig_ef = create_efficient_frontier_chart(
            frontier_vol, frontier_ret, optimal_vol, optimal_return,
            individual_vol, individual_ret,
            [f"{stock_names[t]} ({t})" for t in tickers_to_use]
        )
        
        st.plotly_chart(fig_ef, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Matrix")
    
    corr_matrix = returns_df.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[stock_names[t] for t in corr_matrix.columns],
        y=[stock_names[t] for t in corr_matrix.index],
        colorscale='RdBu', zmid=0,
        text=corr_matrix.values, texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig_corr.update_layout(
        title='Stock Return Correlations',
        height=max(600, len(tickers_to_use) * 30),
        template='plotly_white'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # ==========================================
    # INTERACTIVE PRE-TRADE PANEL (FRAGMENT)
    # ==========================================
    @st.fragment
    def interactive_pre_trade_panel(returns_data, base_weights):
        
        # --- 1. EDITABLE TABLE ---
        st.markdown("---")
        st.subheader("⚖️ Adjust Target Allocations")
        st.write("Fine-tune the optimizer's suggested weights. Set a stock's weight to `0.0` to drop it. **Weights must sum to exactly 100% to save the mandate.**")
        
        weights_df = pd.DataFrame({
            'Ticker': returns_data.columns,
            'Weight': base_weights
        })
        
        edited_df = st.data_editor(
            weights_df,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True), 
                "Weight": st.column_config.NumberColumn("Target Weight", min_value=0.0, max_value=1.0, format="%.4f", step=0.01)
            },
            hide_index=True,
            num_rows="fixed", # Prevents adding/deleting rows to keep the matrix intact
            use_container_width=True,
            key="weight_editor"
        )
        
        # Calculate live sum
        edited_weights = edited_df['Weight'].values
        total_weight = edited_weights.sum()
        is_valid_weights = np.isclose(total_weight, 1.0, atol=0.001)
        
        if not is_valid_weights:
            st.error(f"⚠️ Total Weight: **{total_weight*100:.2f}%**. Please adjust the values above so they equal exactly 100.00%.")
        else:
            st.success("✅ Total Weight: 100.00%. Allocation is mathematically valid.")

        # --- 2. LIVE SIMULATION (Only renders if weights = 100%) ---
        if is_valid_weights:
            st.markdown("---")
            st.subheader("📈 Historical Simulation vs Benchmark")
            
            col_bench, col_window = st.columns(2)
            with col_bench:
                bench_choice = st.selectbox("Benchmark for Simulation:", ["000300.SH", "000905.SH", "000852.SH"], index=0, key="sim_bench")
            with col_window:
                rolling_window = st.select_slider("Rolling Window (Days):", options=[5, 10, 20, 30, 60], value=30, key="sim_window")
            
            start_dt = returns_data.index.min().strftime('%Y%m%d')
            end_dt = returns_data.index.max().strftime('%Y%m%d')
            
            with st.spinner(f"Fetching {bench_choice} data..."):
                bench_df = data_manager.get_index_data_live(bench_choice, start_date=start_dt, end_date=end_dt)
                
            if bench_df is not None and not bench_df.empty:
                # Apply the EDITED weights to the historical returns
                port_returns = returns_data.dot(edited_weights)
                port_returns.name = 'Portfolio'
                
                bench_returns = bench_df['Pct_Change'] / 100.0
                bench_returns.index = pd.to_datetime(bench_returns.index)
                
                sim_df = pd.concat([port_returns, bench_returns.rename('Benchmark')], axis=1).dropna()
                
                sim_df['Port_Cum'] = (1 + sim_df['Portfolio']).cumprod() - 1
                sim_df['Bench_Cum'] = (1 + sim_df['Benchmark']).cumprod() - 1
                sim_df['Active_Spread'] = sim_df['Portfolio'] - sim_df['Benchmark']
                
                sim_df['Rolling_Corr'] = sim_df['Portfolio'].rolling(window=rolling_window).corr(sim_df['Benchmark'])
                roll_cov = sim_df['Portfolio'].rolling(window=rolling_window).cov(sim_df['Benchmark'])
                roll_var = sim_df['Benchmark'].rolling(window=rolling_window).var()
                sim_df['Rolling_Beta'] = roll_cov / roll_var

                fig_sim = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig_sim.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Port_Cum'], mode='lines', name='Portfolio', line=dict(color='#3b82f6')), row=1, col=1)
                fig_sim.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Bench_Cum'], mode='lines', name='Benchmark', line=dict(color='#9ca3af', dash='dot')), row=1, col=1)
                colors = np.where(sim_df['Active_Spread'] > 0, '#10b981', '#ef4444')
                fig_sim.add_trace(go.Bar(x=sim_df.index, y=sim_df['Active_Spread'], name='Daily Active Spread', marker_color=colors), row=2, col=1)
                
                fig_sim.update_layout(height=500, title="Cumulative Return & Daily Active Spread", template='plotly_white', hovermode='x unified')
                fig_sim.update_yaxes(tickformat=".1%", row=1, col=1)
                fig_sim.update_yaxes(tickformat=".2%", title_text="Spread", row=2, col=1)
                st.plotly_chart(fig_sim, use_container_width=True)
                
                col_corr, col_beta = st.columns(2)
                with col_corr:
                    fig_corr_roll = go.Figure()
                    fig_corr_roll.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Rolling_Corr'], mode='lines', fill='tozeroy', name='Correlation', line=dict(color='#8b5cf6')))
                    fig_corr_roll.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_corr_roll.update_layout(height=280, title=f"{rolling_window}-Day Rolling Correlation", template='plotly_white', hovermode='x unified', margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_corr_roll, use_container_width=True)
                with col_beta:
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Scatter(x=sim_df.index, y=sim_df['Rolling_Beta'], mode='lines', name='Beta', line=dict(color='#f59e0b')))
                    fig_beta.add_hline(y=1, line_dash="dash", line_color="gray")
                    fig_beta.update_layout(height=280, title=f"{rolling_window}-Day Rolling Beta", template='plotly_white', hovermode='x unified', margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_beta, use_container_width=True)

        # --- 3. SAVE MANDATE UI (Tied to the Live Editor) ---
        st.markdown("---")
        st.subheader("💾 Create Institutional Fund")
        
        col_name, col_bench, col_btn = st.columns([2, 1, 1])
        with col_name:
            fund_name = st.text_input("Fund Mandate Name", placeholder="e.g., Alpha Quant Q1", key="mandate_name")
        with col_bench:
            save_bench_choice = st.selectbox("Assign Benchmark", ["000300.SH", "000905.SH", "000852.SH"], key="mandate_bench")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Button is disabled if weights don't equal 100% or if there is no name
            if st.button("💾 Save Mandate", type="primary", disabled=not is_valid_weights or not fund_name, use_container_width=True):
                with st.spinner("Locking in mandate..."):
                    
                    # Build the clean database entry from EDITED weights
                    final_positions = {}
                    for _, row in edited_df.iterrows():
                        w = float(row['Weight'])
                        if w > 0: # This elegantly drops any stock you set to 0.0
                            try:
                                formatted_ticker = ticker_to_ts_code(str(row['Ticker']))
                            except:
                                formatted_ticker = str(row['Ticker'])
                            final_positions[formatted_ticker] = w
                    
                    success, msg = data_manager.save_fund_mandate(fund_name, save_bench_choice, final_positions)
                    if success:
                        st.balloons()
                        st.success(msg)
                    else:
                        st.error(msg)

    # Call the panel to render it on the page
    interactive_pre_trade_panel(returns_df, optimal_weights)
else:
    st.info("👆 Click 'Optimize Portfolio' to run mean-variance optimization")


