import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo 
import data_manager
import assrs_logic      # <-- V1 (Rule-Based) Logic
import assrs_logic_2    # <-- V2 (Regime-Switching) Logic
import time
import sys
import os 
import subprocess # <-- NEW: To run Git commands

# --- 1. CONFIGURATION ---
TUSHARE_API_TOKEN = '36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312'

# ---!!!--- DYNAMIC PRODUCTION PATHS ---!!!---
# Use the GitHub workspace directory
PROJECT_PATH = os.environ.get('GITHUB_WORKSPACE', os.path.dirname(os.path.abspath(__file__)))
data_manager.DB_NAME = os.path.join(PROJECT_PATH, 'assrs_tushare_local.db')

# --- 2. BACKTEST PARAMETERS ---
DATA_START_DATE = '20240101'

CHINA_TZ = ZoneInfo("Asia/Shanghai")
TODAY_DATE = datetime.now(CHINA_TZ)
YESTERDAY_DATE = TODAY_DATE - timedelta(days=1) 
DATA_END_DATE = YESTERDAY_DATE.strftime('%Ym%d') 

BACKTEST_START_DATE = '2025-07-01'
BACKTEST_END_DATE = YESTERDAY_DATE.strftime('%Y-%m-%d')

CALENDAR_PROXY_TICKER = '601398' 

# (Stock Logic Parameters are not needed for this script)

# --- 4. MAIN EXECUTION ---
def run_all_sector_backtests():
    """
    Orchestrates the entire historical backtest for BOTH
    V1 (Rule-Based) and V2 (Regime-Switching) sector models.
    """
    if not data_manager.init_tushare(TUSHARE_API_TOKEN):
        print("!! ERROR: Tushare token appears to be invalid. Exiting.")
        sys.exit()

    print(f"--- Ensuring local stock database is up-to-date ({DATA_START_DATE} to {DATA_END_DATE})... ---")
    data_manager.ensure_data_in_db(DATA_START_DATE, DATA_END_DATE)
    print("--- Stock database sync complete. ---")

    all_stock_data = data_manager.get_all_stock_data_from_db()
    if not all_stock_data:
        print("!! ERROR: No stock data found in database. Exiting.")
        return

    all_ppi_data = data_manager.aggregate_ppi_data(all_stock_data)
    if not all_ppi_data:
        print("!! ERROR: Failed to aggregate PPIs. Exiting.")
        return
    else:
        data_manager.save_ppi_data_to_db(all_ppi_data)

    all_ppi_data_loaded = data_manager.load_ppi_data_from_db()
    if not all_ppi_data_loaded:
        print("!! ERROR: Failed to load PPIs from database. Exiting.")
        return

    if CALENDAR_PROXY_TICKER not in all_stock_data:
        print(f"!! ERROR: Calendar proxy stock '{CALENDAR_PROXY_TICKER}' not found.")
        return
            
    market_calendar = all_stock_data[CALENDAR_PROXY_TICKER].index
    
    backtest_date_range = market_calendar[
        (market_calendar >= pd.to_datetime(BACKTEST_START_DATE)) &
        (market_calendar <= pd.to_datetime(BACKTEST_END_DATE))
    ]
    
    if backtest_date_range.empty:
        print(f"!! ERROR: No valid *trading days* found in the date range.")
        return
    print(f"--- Found {len(backtest_date_range)} actual trading days for backtest ---")

    # --- 7. RUN BACKTEST 1: SECTOR-LEVEL (V1 - Rule-Based) ---
    print(f"\n--- Running Daily Backtest 1: SECTOR-LEVEL (V1) ---")
    
    all_sector_scores_v1 = []
    for date in backtest_date_range:
        daily_scorecard = assrs_logic.calculate_scorecard(all_ppi_data_loaded, date)
        if not daily_scorecard.empty:
            daily_scorecard['Date'] = date.strftime('%Y-%m-%d')
            all_sector_scores_v1.append(daily_scorecard.reset_index()) 

    if not all_sector_scores_v1:
        print("V1 Sector backtest finished with no results.")
    else:
        full_results_v1 = pd.concat(all_sector_scores_v1)
        output_filename_v1 = os.path.join(PROJECT_PATH, 'assrs_backtest_results_SECTORS_V1_Rules.csv')
        full_results_v1.to_csv(output_filename_v1, index=False, encoding='utf-8-sig')
        print(f"\n--- SECTOR V1 Backtest Complete! Results saved to '{output_filename_v1}' ---")

    # --- 8. RUN BACKTEST 2: SECTOR-LEVEL (V2 - Regime Switching) ---
    print(f"\n--- Running Daily Backtest 2: SECTOR-LEVEL (V2) ---")
    
    all_sector_scores_v2 = []
    
    for date in backtest_date_range:
        print(f"  -> V2: Fitting Regime Models for: {date.strftime('%Y-%m-%d')}")
        
        daily_scorecard_v2 = assrs_logic_2.calculate_regime_scores(all_ppi_data_loaded, date)
        
        if not daily_scorecard_v2.empty:
            all_sector_scores_v2.append(daily_scorecard_v2.reset_index(drop=True)) 

    if not all_sector_scores_v2:
        print("V2 Sector backtest finished with no results.")
    else:
        full_results_v2 = pd.concat(all_sector_scores_v2)
        
        output_filename_v2 = os.path.join(PROJECT_PATH, 'assrs_backtest_results_SECTORS_V2_Regime.csv')
        full_results_v2.to_csv(output_filename_v2, index=False, encoding='utf-8-sig')
        
        print(f"\n--- SECTOR V2 Backtest Complete! Results saved to '{output_filename_v2}' ---")
        
# --- 5. NEW: COMMIT RESULTS TO GITHUB ---
def commit_results():
    """Configures Git and commits the new CSV files to the repo."""
    print("\n--- Committing new data files to GitHub ---")
    try:
        # Configure Git
        subprocess.run(['git', 'config', '--global', 'user.email', 'action@github.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'GitHub Action'], check=True)
        
        # Add files
        subprocess.run(['git', 'add', 'assrs_backtest_results_SECTORS_V1_Rules.csv'], check=True)
        subprocess.run(['git', 'add', 'assrs_backtest_results_SECTORS_V2_Regime.csv'], check=True)
        
        # Check if there's anything to commit
        status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status_result.stdout:
            print("No changes to data files. Commit skipped.")
            return

        # Commit and Push
        commit_message = f"Data: Automated daily update for {TODAY_DATE.strftime('%Y-%m-%d')}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("--- Data commit successful! ---")
        
    except Exception as e:
        print(f"!! ERROR: Failed to commit new data to GitHub. Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_time = time.time()
    print(f"--- ASSRS Daily Task Started: {datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S')} (China Time) ---")
    
    # 1. Run the data/logic job
    run_all_sector_backtests()
    
    # 2. Commit the resulting CSVs
    commit_results()
    
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds.")
    print(f"--- ASSRS Daily Task Finished: {datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S')} (China Time) ---")