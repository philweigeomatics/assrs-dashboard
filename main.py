import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import data_manager as dm
from assrs_logic_V2_enhanced import calculate_regime_scores
import time
import sys
import os
import db_config

# --- 1. CONFIGURATION ---
TUSHARE_API_TOKEN = '36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312'

# Dynamic production paths
PROJECT_PATH = os.environ.get('GITHUB_WORKSPACE', os.path.dirname(os.path.abspath(__file__)))
print(f"[CONFIG] Using database: {'SQLite' if db_config.USE_SQLITE else 'Supabase'}")
print(f"[CONFIG] Project path: {PROJECT_PATH}")

# --- 2. BACKTEST PARAMETERS ---
DATA_START_DATE = '20240101'
CHINA_TZ = ZoneInfo("Asia/Shanghai")
TODAY_DATE = datetime.now(CHINA_TZ)
YESTERDAY_DATE = TODAY_DATE - timedelta(days=1)
DATA_END_DATE = YESTERDAY_DATE.strftime('%Y%m%d')
BACKTEST_START_DATE = '2025-07-01'
BACKTEST_END_DATE = YESTERDAY_DATE.strftime('%Y-%m-%d')


def calculate_ppi_breadth_proxy(ppi_df, current_date, lookback=20):
    """
    Calculate breadth proxy based on PPI relative to its MA20.

    This is a smart alternative to counting individual stocks above MA20.
    Since PPI is market-cap weighted aggregate of sector stocks, its position
    relative to MA20 reflects overall sector strength (breadth).

    Args:
        ppi_df: DataFrame with PPI data (from all_ppi_data_loaded[sector])
        current_date: Date to calculate breadth for
        lookback: MA period (default 20)

    Returns:
        float: Breadth proxy in range [0.0, 1.0]
               1.0 = Strong breadth (PPI well above MA20)
               0.5 = Neutral breadth (PPI near MA20)
               0.0 = Weak breadth (PPI well below MA20)
    """
    try:
        # Get data up to current date
        df_up_to_date = ppi_df[ppi_df.index <= current_date].copy()

        if len(df_up_to_date) < lookback:
            return 0.5  # Neutral if insufficient data

        # Calculate MA20
        df_up_to_date['MA20'] = df_up_to_date['Close'].rolling(window=lookback).mean()

        if current_date not in df_up_to_date.index:
            return 0.5  # Neutral if date not found

        current_close = df_up_to_date.loc[current_date, 'Close']
        current_ma20 = df_up_to_date.loc[current_date, 'MA20']

        if pd.isna(current_close) or pd.isna(current_ma20) or current_ma20 == 0:
            return 0.5  # Neutral on invalid data

        # Calculate % distance from MA20
        pct_above_ma = (current_close - current_ma20) / current_ma20

        # Map percentage to breadth score [0.0, 1.0]
        # Logic:
        #   PPI > MA20 by 5%+ → Breadth = 1.0 (very strong)
        #   PPI = MA20        → Breadth = 0.5 (neutral)
        #   PPI < MA20 by 5%- → Breadth = 0.0 (very weak)

        # Linear mapping: [-0.05, +0.05] → [0.0, 1.0]
        breadth_proxy = (pct_above_ma + 0.05) / 0.10

        # Clamp to [0.0, 1.0]
        breadth_proxy = max(0.0, min(1.0, breadth_proxy))

        return breadth_proxy

    except Exception as e:
        # Fallback to neutral on any error
        return 0.5


# --- 3. MAIN EXECUTION ---
def run_sector_backtest_v2():
    """
    Orchestrates the entire backtest for V2 Enhanced (Regime-Switching)
    sector model with market context awareness.

    ✅ NEW: No local database for stock data
    ✅ NEW: Uses live qfq data from Tushare API
    ✅ NEW: Only stores PPIs in database
    ✅ NEW: PPI-based breadth proxy (fast, no extra API calls)
    """

    # === STEP 1: Initialize Tushare ===
    print("=" * 60)
    print("STEP 1: Initializing Tushare API")
    print("=" * 60)

    if not dm.init_tushare(TUSHARE_API_TOKEN):
        print("!! ERROR: Tushare token appears to be invalid. Exiting.")
        sys.exit(1)

    print("✅ Tushare API initialized.\n")

    # === STEP 1.5: Update Stock Basic ===
    print("=" * 60)
    print("STEP 1.5: Updating stock_basic table (company names)")
    print("=" * 60)

    try:
        updated = dm.update_stock_basic_table()
        if updated:
            print("✅ stock_basic table updated with latest company names")
        else:
            print("ℹ️  stock_basic already up-to-date (updated today)")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to update stock_basic: {e}")
        print("   Continuing with other updates...")

    print()

    # === STEP 2: Aggregate PPIs (Incremental) ===
    print(f"{'='*60}")
    print(f"STEP 2: Aggregating sector PPIs (incremental update)")
    print(f"{'='*60}")
    print(f"ℹ️  Using LIVE qfq data from Tushare API (not database)")
    print()

    # Build a dict of start_date per sector
    sector_start_dates = {}

    for sector in dm.SECTOR_STOCK_MAP.keys():
        tablename = f'PPI_{sector}'

        # Get latest date in PPI table
        ppi_latest_date = None
        try:
            sector_df = dm.db.read_table(tablename, columns='Date', orderby='-Date', limit=1)
            if not sector_df.empty:
                ppi_latest_date = pd.to_datetime(sector_df['Date'].iloc[0])
                print(f"   ℹ️  {sector}: PPI latest date = {ppi_latest_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"   ℹ️  {sector}: PPI table doesn't exist or is empty")
            ppi_latest_date = None

        # Simplified: Just check if PPI is up-to-date with yesterday
        # No need to check stock dates - aggregate_ppi_data fetches live
        if ppi_latest_date is None:
            # PPI table is empty, do full aggregation
            sector_start_dates[sector] = None
            print(f"   ✅ {sector}: PPI table empty, will do FULL aggregation")
        elif ppi_latest_date < pd.to_datetime(YESTERDAY_DATE):
            # PPI is behind, do incremental update from day after PPI latest
            next_date = ppi_latest_date + pd.Timedelta(days=1)
            sector_start_dates[sector] = next_date
            print(f"   ✅ {sector}: PPI behind, will aggregate from {next_date.strftime('%Y-%m-%d')}")
        else:
            # PPI is up-to-date, skip this sector
            print(f"   ⏭️  {sector}: PPI is up-to-date, skipping")

    # Only aggregate sectors that need updates
    if not sector_start_dates:
        print("   ℹ️  All PPIs are up-to-date, nothing to aggregate")
        all_ppi_data = {}
    else:
        # Aggregate with flexible date handling (data_manager handles format conversion)
        all_ppi_data = dm.aggregate_ppi_data(sector_start_dates=sector_start_dates)

        if all_ppi_data:
            # Save only new dates
            dm.save_ppi_data_to_db(all_ppi_data)
            print(f"✅ Aggregated and saved {len(all_ppi_data)} sector PPIs (incremental).\n")
        else:
            print(f"✅ No PPI updates needed.\n")

    # === STEP 3: Load PPIs from DB ===
    print(f"{'='*60}")
    print(f"STEP 3: Loading PPIs from database")
    print(f"{'='*60}")

    all_ppi_data_loaded = dm.load_ppi_data_from_db()

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

    # === STEP 4: Build Trading Calendar ===
    print(f"{'='*60}")
    print(f"STEP 4: Building trading calendar from MARKET_PROXY PPI")
    print(f"{'='*60}")

    if 'MARKET_PROXY' not in all_ppi_data_loaded:
        print(f"!! ERROR: MARKET_PROXY not found in PPIs. Exiting.")
        sys.exit(1)

    # Use MARKET_PROXY PPI dates as trading calendar
    market_calendar = all_ppi_data_loaded['MARKET_PROXY'].index

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

    # === STEP 5: Run V2 Enhanced Backtest ===
    print(f"{'='*60}")
    print(f"STEP 5: Running V2 Enhanced (Regime-Switching) Backtest")
    print(f"{'='*60}")
    print(f"Features: Market context, volume confirmation, momentum, adaptive thresholds")
    print(f"ℹ️  Using PPI-based breadth proxy (fast, no extra API calls)")
    print()

    all_sector_scores_v2 = []
    historical_scores = None
    total_days = len(backtest_date_range)

    for idx, date in enumerate(backtest_date_range, 1):
        # Progress indicator
        if idx % 10 == 0 or idx == total_days:
            print(f"   Progress: {idx}/{total_days} ({idx/total_days*100:.1f}%) - {date.strftime('%Y-%m-%d')}")

        try:
            # Call enhanced V2 with optional historical scores
            daily_scorecard_v2 = calculate_regime_scores(
                all_ppi_data_loaded,
                date,
                historical_scores=historical_scores
            )

            if not daily_scorecard_v2.empty:
                # Calculate PPI-based breadth proxy
                breadth_values = []
                for _, row in daily_scorecard_v2.iterrows():
                    sector = row['Sector']
                    if sector in all_ppi_data_loaded:
                        ppi_df = all_ppi_data_loaded[sector]
                        breadth = calculate_ppi_breadth_proxy(ppi_df, date)
                        breadth_values.append(breadth)
                    else:
                        breadth_values.append(0.5)  # Neutral for missing sectors

                daily_scorecard_v2['Market_Breadth'] = breadth_values

                all_sector_scores_v2.append(daily_scorecard_v2.reset_index(drop=True))

                # Build historical scores for adaptive thresholds
                if historical_scores is None:
                    historical_scores = daily_scorecard_v2[['Date', 'Sector', 'TOTAL_SCORE']].copy()
                else:
                    historical_scores = pd.concat([
                        historical_scores.tail(120 * len(all_ppi_data_loaded)),
                        daily_scorecard_v2[['Date', 'Sector', 'TOTAL_SCORE']]
                    ], ignore_index=True)

        except Exception as e:
            print(f"   !! ERROR processing {date.strftime('%Y-%m-%d')}: {str(e)}")
            continue

    print()

    # === STEP 6: Save Results ===
    print(f"{'='*60}")
    print(f"STEP 6: Saving backtest results")
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

    # === STEP 7: Validation Checks ===
    print(f"{'='*60}")
    print(f"STEP 7: Validation checks")
    print(f"{'='*60}")

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

    # Show breadth stats
    if 'Market_Breadth' in full_results_v2.columns:
        print(f"   Avg Market_Breadth: {full_results_v2['Market_Breadth'].mean():.2f}")
        print(f"   Breadth range: {full_results_v2['Market_Breadth'].min():.2f} to {full_results_v2['Market_Breadth'].max():.2f}")

    print()
    print(f"{'='*60}")
    print(f"✅ ALL STEPS COMPLETE - V2 Enhanced backtest successful!")
    print(f"{'='*60}\n")


# --- 4. ENTRY POINT ---
if __name__ == "__main__":
    start_time = time.time()

    print(f"\n{'#'*60}")
    print(f"# ASSRS Daily Task - V2 Enhanced (Live QFQ Data)")
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
