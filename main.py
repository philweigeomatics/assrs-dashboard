import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo 
import data_manager
from assrs_logic_V2_enhanced import calculate_regime_scores  # NEW: Only V2.5
import time
import sys
import os 


# --- 1. CONFIGURATION ---
TUSHARE_API_TOKEN = '36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312'

# Dynamic production paths for GitHub Actions
PROJECT_PATH = os.environ.get('GITHUB_WORKSPACE', os.path.dirname(os.path.abspath(__file__)))

# NOW set the DB path AFTER import
DB_PATH = os.path.join(PROJECT_PATH, 'assrs_tushare_local.db')

# Check if running in GitHub Actions
IS_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS') == 'true'

# Use different DB names
if IS_GITHUB_ACTIONS:
    DB_PATH = os.path.join(PROJECT_PATH, 'assrs_tushare_local.db')  # Production
else:
    DB_PATH = os.path.join(PROJECT_PATH, 'assrs_tushare_local_dev.db')  # Local dev
    
data_manager.DB_NAME = DB_PATH

print(f"[CONFIG] Using database: {DB_PATH}")
print(f"[CONFIG] Project path: {PROJECT_PATH}")


# --- 2. BACKTEST PARAMETERS ---
DATA_START_DATE = '20240101'

CHINA_TZ = ZoneInfo("Asia/Shanghai")
TODAY_DATE = datetime.now(CHINA_TZ)
YESTERDAY_DATE = TODAY_DATE - timedelta(days=1) 
DATA_END_DATE = YESTERDAY_DATE.strftime('%Y%m%d')  # Fixed typo: was '%Ym%d'

BACKTEST_START_DATE = '2025-07-01'
BACKTEST_END_DATE = YESTERDAY_DATE.strftime('%Y-%m-%d')

CALENDAR_PROXY_TICKER = '601398'  # ICBC for trading calendar


def calculate_sector_breadth(all_stock_data, sector_stocks, ma_period=20):
    """
    Calculate market breadth: % of stocks in sector trading above MA20
    
    Args:
        all_stock_data: Dict of DataFrames (from data_manager.get_all_stock_data_from_db())
        sector_stocks: List of ticker strings for the sector
        ma_period: Moving average period (default 20)
    
    Returns:
        float: Percentage (0-1) of stocks above their MA20
    """
    above_ma = 0
    total = 0
    
    for ticker in sector_stocks:
        if ticker not in all_stock_data:
            continue
            
        df = all_stock_data[ticker]
        if df is None or len(df) < ma_period:
            continue
        
        # Calculate MA20
        ma = df['Close'].rolling(window=ma_period).mean()
        
        # Check latest data point
        if len(df) > 0 and len(ma) > 0:
            latest_close = df['Close'].iloc[-1]
            latest_ma = ma.iloc[-1]
            
            if pd.notna(latest_close) and pd.notna(latest_ma):
                total += 1
                if latest_close > latest_ma:
                    above_ma += 1
    
    return above_ma / total if total > 0 else 0.0



# --- 3. MAIN EXECUTION ---

def run_sector_backtest_v2():
    """
    Orchestrates the entire backtest for V2 Enhanced (Regime-Switching)
    sector model with market context awareness.
    
    Drops V1 to simplify dashboard and reduce confusion.
    """
    
    # === STEP 1: Initialize Tushare ===
    if not data_manager.init_tushare(TUSHARE_API_TOKEN):
        print("!! ERROR: Tushare token appears to be invalid. Exiting.")
        sys.exit(1)

    # === STEP 2: Sync Stock Data ===
    print(f"\n{'='*60}")
    print(f"STEP 1: Syncing stock database")
    print(f"Date range: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"{'='*60}")
    
    data_manager.ensure_data_in_db(DATA_START_DATE, DATA_END_DATE)
    print("✅ Stock database sync complete.\n")

    # === STEP 3: Load Stock Data ===
    print(f"{'='*60}")
    print(f"STEP 2: Loading stock data from database")
    print(f"{'='*60}")
    
    all_stock_data = data_manager.get_all_stock_data_from_db()
    if not all_stock_data:
        print("!! ERROR: No stock data found in database. Exiting.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(all_stock_data)} stocks from database.\n")

    # === STEP 4: Aggregate PPIs ===
    print(f"{'='*60}")
    print(f"STEP 3: Aggregating sector PPIs")
    print(f"{'='*60}")
    
    all_ppi_data = data_manager.aggregate_ppi_data(all_stock_data)
    if not all_ppi_data:
        print("!! ERROR: Failed to aggregate PPIs. Exiting.")
        sys.exit(1)
    
    data_manager.save_ppi_data_to_db(all_ppi_data)
    print(f"✅ Aggregated and saved {len(all_ppi_data)} sector PPIs.\n")

    # === STEP 5: Load PPIs from DB ===
    print(f"{'='*60}")
    print(f"STEP 4: Loading PPIs from database")
    print(f"{'='*60}")
    
    all_ppi_data_loaded = data_manager.load_ppi_data_from_db()
    if not all_ppi_data_loaded:
        print("!! ERROR: Failed to load PPIs from database. Exiting.")
        sys.exit(1)
    
    # Verify MARKET_PROXY exists
    if 'MARKET_PROXY' not in all_ppi_data_loaded:
        print("!! WARNING: MARKET_PROXY not found in PPIs. Enhanced V2 needs it as benchmark!")
        print("!! Please ensure MARKET_PROXY is defined in data_manager.SECTOR_STOCK_MAP")
    
    print(f"✅ Loaded {len(all_ppi_data_loaded)} PPIs from database.")
    print(f"   Sectors: {', '.join(list(all_ppi_data_loaded.keys())[:5])}...")
    print()

    # === STEP 6: Build Trading Calendar ===
    print(f"{'='*60}")
    print(f"STEP 5: Building trading calendar")
    print(f"{'='*60}")
    
    if CALENDAR_PROXY_TICKER not in all_stock_data:
        print(f"!! ERROR: Calendar proxy stock '{CALENDAR_PROXY_TICKER}' not found. Exiting.")
        sys.exit(1)
    
    market_calendar = all_stock_data[CALENDAR_PROXY_TICKER].index
    
    backtest_date_range = market_calendar[
        (market_calendar >= pd.to_datetime(BACKTEST_START_DATE)) &
        (market_calendar <= pd.to_datetime(BACKTEST_END_DATE))
    ]
    
    if backtest_date_range.empty:
        print(f"!! ERROR: No valid trading days found between {BACKTEST_START_DATE} and {BACKTEST_END_DATE}.")
        sys.exit(1)
    
    print(f"✅ Found {len(backtest_date_range)} trading days for backtest.")
    print(f"   Range: {backtest_date_range[0].strftime('%Y-%m-%d')} to {backtest_date_range[-1].strftime('%Y-%m-%d')}")
    print()

    # === STEP 7: Run V2 Enhanced Backtest ===
    print(f"{'='*60}")
    print(f"STEP 6: Running V2 Enhanced (Regime-Switching) Backtest")
    print(f"{'='*60}")
    print(f"Features: Market context, volume confirmation, momentum, adaptive thresholds")
    print()
    
    all_sector_scores_v2 = []
    historical_scores = None  # For adaptive thresholds (optional)
    
    total_days = len(backtest_date_range)
    
    for idx, date in enumerate(backtest_date_range, 1):
        # Progress indicator
        if idx % 10 == 0 or idx == total_days:
            print(f"  Progress: {idx}/{total_days} ({idx/total_days*100:.1f}%) - {date.strftime('%Y-%m-%d')}")
        
        try:
            # Call enhanced V2.5 with optional historical scores
            daily_scorecard_v2 = calculate_regime_scores(
                all_ppi_data_loaded,
                date,
                historical_scores=historical_scores  # Pass None on first run
            )
            
            if not daily_scorecard_v2.empty:
                # Calculate breadth for each sector
                breadth_values = []
                for _, row in daily_scorecard_v2.iterrows():
                    sector = row['Sector']
                    if sector in data_manager.SECTOR_STOCK_MAP:
                        sector_stocks = data_manager.SECTOR_STOCK_MAP[sector]
                        breadth = calculate_sector_breadth(all_stock_data, sector_stocks)
                        breadth_values.append(breadth)
                    else:
                        breadth_values.append(0.0)
                
                daily_scorecard_v2['Market_Breadth'] = breadth_values
                all_sector_scores_v2.append(daily_scorecard_v2.reset_index(drop=True))
                
                # Optional: Build historical scores for adaptive thresholds
                # (Only keep last 120 days to save memory)
                if historical_scores is None:
                    historical_scores = daily_scorecard_v2[['Date', 'Sector', 'TOTAL_SCORE']].copy()
                else:
                    historical_scores = pd.concat([
                        historical_scores.tail(120 * len(all_ppi_data_loaded)),  # Keep ~120 days
                        daily_scorecard_v2[['Date', 'Sector', 'TOTAL_SCORE']]
                    ], ignore_index=True)
        
        except Exception as e:
            print(f"  !! ERROR processing {date.strftime('%Y-%m-%d')}: {str(e)}")
            continue
    
    print()

    # === STEP 8: Save Results ===
    print(f"{'='*60}")
    print(f"STEP 7: Saving backtest results")
    print(f"{'='*60}")
    
    if not all_sector_scores_v2:
        print("!! ERROR: V2 backtest finished with no results. Check logs above.")
        sys.exit(1)
    
    full_results_v2 = pd.concat(all_sector_scores_v2, ignore_index=True)
    
    # Output path
    output_filename_v2 = os.path.join(PROJECT_PATH, 'assrs_backtest_results_SECTORS_V2_Regime.csv')
    
    # Save CSV
    full_results_v2.to_csv(output_filename_v2, index=False, encoding='utf-8-sig')
    
    print(f"✅ V2 Enhanced backtest complete!")
    print(f"   Total rows: {len(full_results_v2):,}")
    print(f"   Date range: {full_results_v2['Date'].min()} to {full_results_v2['Date'].max()}")
    print(f"   Sectors: {full_results_v2['Sector'].nunique()}")
    print(f"   Output file: {output_filename_v2}")
    print()
    
    # === STEP 9: Validation Checks ===
    print(f"{'='*60}")
    print(f"STEP 8: Validation checks")
    print(f"{'='*60}")
    
    # Check for MARKET_PROXY
    market_rows = full_results_v2[full_results_v2['Sector'] == 'MARKET_PROXY']
    if market_rows.empty:
        print("⚠️  WARNING: MARKET_PROXY not found in results!")
    else:
        print(f"✅ MARKET_PROXY: {len(market_rows)} rows")
    
    # Check for MARKET_PROXY
    market_rows = full_results_v2[full_results_v2['Sector'] == 'MARKET_PROXY']
    if market_rows.empty:
        print("⚠️  WARNING: MARKET_PROXY not found in results!")
    else:
        print(f"✅ MARKET_PROXY: {len(market_rows)} rows")
    
    # Check new columns exist
    expected_cols = ['Market_Score', 'Market_Regime', 'Excess_Prob', 'Position_Size', 'Dispersion']
    missing_cols = [col for col in expected_cols if col not in full_results_v2.columns]
    
    if missing_cols:
        print(f"⚠️  WARNING: Missing expected columns: {missing_cols}")
    else:
        print(f"✅ All enhanced columns present: {', '.join(expected_cols)}")
    
    # Show sample stats
    print(f"\n📊 Sample statistics:")
    print(f"   Avg TOTAL_SCORE: {full_results_v2['TOTAL_SCORE'].mean():.2f}")
    print(f"   Score range: {full_results_v2['TOTAL_SCORE'].min():.2f} to {full_results_v2['TOTAL_SCORE'].max():.2f}")
    
    if 'Market_Score' in full_results_v2.columns:
        latest_market = full_results_v2[full_results_v2['Sector'] == 'MARKET_PROXY'].iloc[-1]
        print(f"   Latest market regime: {latest_market.get('Market_Regime', 'N/A')}")
        print(f"   Latest market score: {latest_market.get('Market_Score', 0):.2f}")
    
    print()
    print(f"{'='*60}")
    print(f"✅ ALL STEPS COMPLETE - V2 Enhanced backtest successful!")
    print(f"{'='*60}\n")


# --- 4. ENTRY POINT ---

if __name__ == "__main__":
    start_time = time.time()
    
    print(f"\n{'#'*60}")
    print(f"# ASSRS Daily Task - V2 Enhanced (V1 Removed)")
    print(f"# Started: {datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S')} (China Time)")
    print(f"{'#'*60}\n")
    
    try:
        run_sector_backtest_v2()
    except Exception as e:
        print(f"\n!! FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    end_time = time.time()
    
    print(f"\n{'#'*60}")
    print(f"# Total runtime: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.1f} minutes)")
    print(f"# Finished: {datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S')} (China Time)")
    print(f"{'#'*60}\n")
