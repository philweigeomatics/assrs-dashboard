import pandas as pd
import sys 
import time
from datetime import datetime, timedelta 
import math 

# --- 1. CONFIGURATION ---
INPUT_FILENAME = 'assrs_backtest_results_STOCKS.csv'
TRANSACTION_COST = 0.001 # 0.1% per trade (buy or sell)
STARTING_CAPITAL = 10000.0 

# --- RISK MANAGEMENT PARAMETERS ---
USE_STOP_LOSS = True   
STOP_LOSS_PCT = 0.07   # 7% stop-loss (e.g., 0.07 = 7%)

# --- 2. CORE SIMULATION FUNCTIONS (T+1 LOGIC) ---

def simulate_trades_for_ticker(stock_df, ticker, 
                             STARTING_CAPITAL, TRANSACTION_COST, 
                             USE_STOP_LOSS, STOP_LOSS_PCT):
    """
    Runs the full trade simulation for a single stock's state history.
    """
    cash = STARTING_CAPITAL
    shares = 0.0
    entry_price = 0.0 
    entry_date = None
    trades_log = [] 
    
    stock_df = stock_df.sort_values(by='Date')
    
    stock_df['Next_Open'] = stock_df['Open'].shift(-1)
    stock_df.dropna(subset=['Next_Open'], inplace=True)

    for _, row in stock_df.iterrows():
        current_state = row['State']
        trade_price = row['Next_Open'] 
        trade_date = row['Date'] 
        
        # --- 1. CHECK FOR EXIT SIGNALS ---
        if shares > 0:
            
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            sell_signal = None
            
            if USE_STOP_LOSS and (trade_price < stop_loss_price):
                sell_signal = "Stop-Loss"
            elif current_state == "WEAKENING" or current_state == "STRUCTURAL_AVOID":
                sell_signal = current_state

            if sell_signal:
                sell_value = shares * trade_price
                trade_cost = sell_value * TRANSACTION_COST
                cash_from_sale = sell_value - trade_cost
                cash = cash + cash_from_sale
                
                pnl = (trade_price - entry_price) * shares - trade_cost
                trades_log.append({
                    'Date': trade_date,
                    'Action': 'SELL',
                    'Price': trade_price,
                    'Shares': shares,
                    'Cost': trade_cost,
                    'P&L': pnl,
                    'Signal': sell_signal
                })
                
                shares = 0.0
                entry_price = 0.0
                entry_date = None

        # --- 2. CHECK FOR ENTRY SIGNALS ---
        elif shares == 0: 
            if current_state == "ACCELERATION" or current_state == "CONTINUATION":
                shares_to_buy_float = (cash / (trade_price * (1 + TRANSACTION_COST)))
                shares_to_buy = math.floor(shares_to_buy_float)
                
                if shares_to_buy <= 0: 
                    continue

                buy_cost = shares_to_buy * trade_price
                trade_cost = buy_cost * TRANSACTION_COST
                
                if cash >= buy_cost + trade_cost:
                    cash = cash - buy_cost - trade_cost
                    shares = shares_to_buy
                    entry_price = trade_price 
                    entry_date = trade_date
                    
                    trades_log.append({
                        'Date': trade_date,
                        'Action': 'BUY',
                        'Price': trade_price,
                        'Shares': shares_to_buy,
                        'Cost': trade_cost,
                        'P&L': 0.0, 
                        'Signal': current_state
                    })
                    
    # --- 3. CALCULATE FINAL P&L ---
    if not stock_df.empty:
        last_known_trade_price = stock_df.iloc[-1]['Next_Open'] 
        final_equity = cash + (shares * last_known_trade_price)
    else:
        final_equity = cash
        
    total_return_pct = ((final_equity / STARTING_CAPITAL) - 1) * 100
    
    summary = {
        'Ticker': ticker,
        'Strategy_Equity': final_equity,
        'Strategy_Return_Pct': total_return_pct,
        'Total_Trades': len(trades_log), 
        'Start_Capital': STARTING_CAPITAL,
    }
    
    return summary, trades_log

# ---!!!--- NEW: BUY AND HOLD SIMULATOR ---!!!---
def calculate_buy_and_hold_pnl(stock_df, STARTING_CAPITAL, TRANSACTION_COST):
    """
    Calculates the P&L for a simple Buy and Hold strategy on the same data.
    """
    stock_df = stock_df.sort_values(by='Date')
    stock_df['Next_Open'] = stock_df['Open'].shift(-1)
    
    # We need at least 2 days to have an entry and exit
    if len(stock_df) < 2:
        return STARTING_CAPITAL, 0.0

    # Get the first available T+1 Open price
    buy_price = stock_df.iloc[0]['Next_Open']
    
    # Get the last available T+1 Open price (which is the 'Next_Open' of the second-to-last row)
    last_row = stock_df.dropna(subset=['Next_Open']).iloc[-1]
    sell_price = last_row['Next_Open']

    if pd.isna(buy_price) or pd.isna(sell_price) or buy_price == 0:
        return STARTING_CAPITAL, 0.0

    # --- Simulate Buy ---
    shares_to_buy_float = (STARTING_CAPITAL / (buy_price * (1 + TRANSACTION_COST)))
    shares = math.floor(shares_to_buy_float)
    buy_cost = shares * buy_price
    trade_cost_buy = buy_cost * TRANSACTION_COST
    cash = STARTING_CAPITAL - buy_cost - trade_cost_buy

    # --- Simulate Sell at the end ---
    sell_value = shares * sell_price
    trade_cost_sell = sell_value * TRANSACTION_COST
    final_cash = cash + sell_value - trade_cost_sell
    
    final_equity = final_cash
    total_return_pct = ((final_equity / STARTING_CAPITAL) - 1) * 100
    
    return final_equity, total_return_pct
# ---!!!--- END OF NEW FUNCTION ---!!!---


# --- 3. EXECUTION FUNCTIONS ---

def run_full_backtest(df):
    """
    Runs the simulation for all tickers found in the CSV.
    Calculates and prints the aggregate portfolio P&L.
    """
    print(f"--- Running Full Backtest on all {len(df['Ticker'].unique())} stocks ---")
    print(f"Parameters: Stop-Loss Enabled = {USE_STOP_LOSS}, Stop-Loss % = {STOP_LOSS_PCT*100}%")
    
    all_tickers = df['Ticker'].unique()
    all_results = []
    
    for ticker in all_tickers:
        stock_df = df[df['Ticker'] == ticker].copy()
        if stock_df.empty:
            print(f"  - Skipping {ticker}: No data found in CSV.")
            continue
        
        # ---!!!--- MODIFIED: Run both Strategy and B&H ---!!!---
        strategy_result, _ = simulate_trades_for_ticker(stock_df, ticker,
                                             STARTING_CAPITAL, TRANSACTION_COST,
                                             USE_STOP_LOSS, STOP_LOSS_PCT)
        
        bh_equity, bh_return_pct = calculate_buy_and_hold_pnl(stock_df,
                                                            STARTING_CAPITAL,
                                                            TRANSACTION_COST)
        
        strategy_result['B&H_Equity'] = bh_equity
        strategy_result['B&H_Return_Pct'] = bh_return_pct
        all_results.append(strategy_result)
        # ---!!!--- END OF MODIFICATION ---!!!---
        
    results_df = pd.DataFrame(all_results)
    
    # --- 1. Aggregate Portfolio Stats ---
    total_start_capital = results_df['Start_Capital'].sum()
    # Strategy
    total_strategy_equity = results_df['Strategy_Equity'].sum()
    total_strategy_net_profit = total_strategy_equity - total_start_capital
    total_strategy_return_pct = (total_strategy_net_profit / total_start_capital) * 100
    # Buy and Hold
    total_bh_equity = results_df['B&H_Equity'].sum()
    total_bh_net_profit = total_bh_equity - total_start_capital
    total_bh_return_pct = (total_bh_net_profit / total_start_capital) * 100
    
    total_trades = results_df['Total_Trades'].sum() 

    # --- 2. Create formatted DataFrame for printing ---
    print_df = results_df.copy()
    print_df['Strategy_vs_B&H'] = print_df['Strategy_Return_Pct'] - print_df['B&H_Return_Pct']
    
    print_df['Final_Equity'] = results_df['Strategy_Equity'].apply(lambda x: f"${x:,.2f}")
    print_df['Start_Capital'] = results_df['Start_Capital'].apply(lambda x: f"${x:,.2f}")
    print_df['Strategy_Return'] = results_df['Strategy_Return_Pct'].apply(lambda x: f"{x:.2f}%")
    print_df['B&H_Return'] = results_df['B&H_Return_Pct'].apply(lambda x: f"{x:.2f}%")
    print_df['Strategy_vs_B&H'] = print_df['Strategy_vs_B&H'].apply(lambda x: f"{x:+.2f}%")
    
    print_df = print_df[['Ticker', 'Final_Equity', 'Strategy_Return', 'B&H_Return', 'Strategy_vs_B&H', 'Total_Trades', 'Start_Capital']]

    print("\n--- Individual Stock P&L Summary ---")
    print(print_df.to_markdown(index=False, numalign="left", stralign="left"))
    
    # --- 3. Print Aggregate Portfolio Summary ---
    print("\n--- Aggregate Portfolio Summary ---")
    print(f"Total Starting Capital: ${total_start_capital:,.2f}")
    print(f"Total Transactions:     {total_trades}\n")
    
    print("--- Your Strategy (4-State Model) ---")
    print(f"Total Final Equity:     ${total_strategy_equity:,.2f}")
    print(f"Total Net Profit:       ${total_strategy_net_profit:,.2f}")
    print(f"Total % Gain:           {total_strategy_return_pct:.2f}%")
    
    print("\n--- Benchmark (Buy & Hold) ---")
    print(f"Total Final Equity:     ${total_bh_equity:,.2f}")
    print(f"Total Net Profit:       ${total_bh_net_profit:,.2f}")
    print(f"Total % Gain:           {total_bh_return_pct:.2f}%")
    
    print("\n---===================================---")
    alpha = total_strategy_return_pct - total_bh_return_pct
    print(f"Strategy vs. B&H (Alpha): {alpha:+.2f}%")
    print("---===================================---")


def run_single_stock_backtest(df, target_ticker):
    """
    Runs the simulation AND PRINTS THE TRANSACTION LOG for the user-specified ticker.
    """
    print(f"--- Running Single Stock Backtest for: {target_ticker} ---")
    print(f"Parameters: Stop-Loss Enabled = {USE_STOP_LOSS}, Stop-Loss % = {STOP_LOSS_PCT*100}%")
    
    stock_df = df[df['Ticker'].astype(str) == str(target_ticker)].copy() 
    
    if stock_df.empty:
        print(f"!! ERROR: No data found for ticker '{target_ticker}' in the provided date range.")
        return
        
    # --- Debug Print ---
    print(f"\n--- Debug: Input Data for {target_ticker} (First 15 Days) ---")
    print_columns = ['Date', 'Ticker', 'State', 'Open', 'Close', 'EMA_20', 'EMA_60', 'ADX_14', 'RSI_14']
    valid_print_columns = [col for col in print_columns if col in stock_df.columns]
    print(stock_df.head(15)[valid_print_columns].to_markdown(index=False))
    
    # ---!!!--- Run both Strategy and B&H ---!!!---
    result, trades_log = simulate_trades_for_ticker(stock_df, target_ticker,
                                                  STARTING_CAPITAL, TRANSACTION_COST,
                                                  USE_STOP_LOSS, STOP_LOSS_PCT)
    
    bh_equity, bh_return_pct = calculate_buy_and_hold_pnl(stock_df,
                                                        STARTING_CAPITAL,
                                                        TRANSACTION_COST)
    
    result['B&H_Equity'] = bh_equity
    result['B&H_Return_Pct'] = bh_return_pct
    # ---!!!--- End of modification ---!!!---

    result_df = pd.DataFrame([result])
    result_df['Strategy_vs_B&H'] = result_df['Strategy_Return_Pct'] - result_df['B&H_Return_Pct']
    
    result_df['Final_Equity'] = result_df['Strategy_Equity'].apply(lambda x: f"${x:,.2f}")
    result_df['Start_Capital'] = result_df['Start_Capital'].apply(lambda x: f"${x:,.2f}")
    result_df['Strategy_Return'] = result_df['Strategy_Return_Pct'].apply(lambda x: f"{x:.2f}%")
    result_df['B&H_Return'] = result_df['B&H_Return_Pct'].apply(lambda x: f"{x:.2f}%")
    result_df['Strategy_vs_B&H'] = result_df['Strategy_vs_B&H'].apply(lambda x: f"{x:+.2f}%")

    print_cols = ['Ticker', 'Final_Equity', 'Strategy_Return', 'B&H_Return', 'Strategy_vs_B&H', 'Total_Trades', 'Start_Capital']
    
    print("\n--- Single Stock P&L Summary ---")
    print(result_df[print_cols].to_markdown(index=False, numalign="left", stralign="left"))
    
    if not trades_log:
        print("\n--- Individual Transaction Log ---")
        print("No transactions were executed for this stock in this period.")
    else:
        log_df = pd.DataFrame(trades_log)
        log_df['Price'] = log_df['Price'].apply(lambda x: f"{x:.2f}")
        log_df['Shares'] = log_df['Shares'].apply(lambda x: f"{x:,.0f}")
        log_df['Cost'] = log_df['Cost'].apply(lambda x: f"{x:.2f}")
        log_df['P&L'] = log_df['P&L'].apply(lambda x: f"{x:,.2f}")
        
        print("\n--- Individual Transaction Log ---")
        print(log_df.to_markdown(index=False, numalign="left", stralign="left"))

# --- 4. MAIN ORCHESTRATOR ---

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        full_results_df = pd.read_csv(INPUT_FILENAME, encoding='utf-8-sig', dtype={'Ticker': str})
    except FileNotFoundError:
        print(f"!! ERROR: Input file not found: '{INPUT_FILENAME}'")
        print("!! Please run main.py first to generate the backtest results file.")
        sys.exit()
    except Exception as e:
        print(f"!! ERROR: Could not read CSV file. Error: {e}")
        sys.exit()

    if 'Open' not in full_results_df.columns:
        print(f"!! ERROR: 'Open' column not found in {INPUT_FILENAME}.")
        print("!! Please DELETE your old database (assrs_tushare_local.db) and re-run main.py to fetch Open prices.")
        sys.exit()
        
    full_results_df['Close'] = pd.to_numeric(full_results_df['Close'], errors='coerce')
    full_results_df['Open'] = pd.to_numeric(full_results_df['Open'], errors='coerce')
    full_results_df['Date'] = pd.to_datetime(full_results_df['Date']) 
    full_results_df.dropna(subset=['Close', 'Open', 'Date'], inplace=True) 

    target_ticker = 'all'
    start_date_filter = None
    end_date_filter = None

    if len(sys.argv) > 1:
        target_ticker = sys.argv[1] 
    if len(sys.argv) > 3:
        start_date_filter = sys.argv[2]
        end_date_filter = sys.argv[3]
        
    simulation_df = full_results_df.copy()

    if start_date_filter and end_date_filter:
        print(f"--- Filtering data for date range: {start_date_filter} to {end_date_filter} ---")
        try:
            start_dt = pd.to_datetime(start_date_filter)
            end_dt = pd.to_datetime(end_date_filter)
            simulation_df = simulation_df[
                (simulation_df['Date'] >= start_dt) & 
                (simulation_df['Date'] <= end_dt)
            ]
        except Exception as e:
            print(f"!! ERROR: Invalid date format. Please use YYYY-MM-DD. Error: {e}")
            sys.exit()
            
    if simulation_df.empty:
        print("!! ERROR: No data found for the specified date range.")
        sys.exit()

    if target_ticker.lower() == 'all':
        run_full_backtest(simulation_df)
    else:
        run_single_stock_backtest(simulation_df, target_ticker)
        
    end_time = time.time()
    print(f"\nSimulation runtime: {end_time - start_time:.2f} seconds.")

