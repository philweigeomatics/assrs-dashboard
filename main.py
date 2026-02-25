import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import data_manager as dm
from assrs_logic_V2_enhanced import calculate_regime_scores
import time
import sys
import os
import db_config
import api_config

# --- 1. CONFIGURATION ---
TUSHARE_API_TOKEN = api_config.TUSHARE_TOKEN

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
    

def run_daily_ppi_and_market_breadth():
    """
    Orchestrates sector PPI aggregation and market breadth calculation.

    Uses PPI-based breadth proxy (PPI vs MA20) instead of counting stocks.
    This is faster and works without storing individual stock data.
    """
    print("=" * 60)
    print("STEP 1: Initializing Tushare API")
    print("=" * 60)

    if not dm.init_tushare(TUSHARE_API_TOKEN):
        print("!! ERROR: Tushare token appears to be invalid. Exiting.")
        sys.exit(1)
    print("✓ Tushare API initialized.")

    # STEP 1.5: Update Stock Basic
    print("=" * 60)
    print("STEP 1.5: Updating stock_basic table (company names)")
    print("=" * 60)
    try:
        updated = dm.update_stock_basic_table()
        if updated:
            print("✓ stock_basic table updated with latest company names")
        else:
            print("✓ stock_basic already up-to-date (updated today)")
    except Exception as e:
        print(f"⚠ WARNING: Failed to update stock_basic: {e}")
        print("Continuing with other updates...")
    print()


    # STEP 1.7: Update daily_basic
    print("=" * 60)
    print("STEP 1.7: Updating daily_basic (latest market metrics)")
    print("=" * 60)
    try:
        result = dm.update_daily_basic()
        if result is True:
            print("✓ daily_basic updated successfully")
        elif result is False:
            print("✓ daily_basic skipped (up-to-date or no market data today)")
        else:
            print("⚠ daily_basic update failed, continuing...")
    except Exception as e:
        print(f"⚠ WARNING: daily_basic update failed: {e}")
        print("Continuing with other updates...")
    print()

    # STEP 2: Aggregate PPIs (Incremental)
    print("=" * 60)
    print("STEP 2: Aggregating sector PPIs (incremental update)")
    print("=" * 60)
    print("✓ Using LIVE qfq data from Tushare API")
    print("✓ Return-based PPI (original logic preserved)")
    print("✓ Volume = dollar value traded (volume × mid_price)")
    print()

    sector_start_dates = {}
    for sector in dm.SECTOR_STOCK_MAP.keys():
        table_name = f"PPI_{sector}"

        ppi_latest_date = None
        try:
            sector_df = dm.db.read_table(table_name, columns='Date', order_by='-Date', limit=1)
            if not sector_df.empty:
                ppi_latest_date = pd.to_datetime(sector_df['Date'].iloc[0])
                print(f"{sector}: PPI latest date = {ppi_latest_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"{sector}: PPI table doesn't exist or is empty")
            ppi_latest_date = None

        if ppi_latest_date is None:
            sector_start_dates[sector] = None
            print(f"{sector}: PPI table empty, will do FULL aggregation")
        elif ppi_latest_date.date() >= YESTERDAY_DATE.date():
            print(f"{sector}: PPI is up-to-date, skipping")
        else:
            next_date = ppi_latest_date + pd.Timedelta(days=1)
            sector_start_dates[sector] = next_date
            print(f"{sector}: PPI behind, will aggregate from {next_date.strftime('%Y-%m-%d')}")

    if not sector_start_dates:
        print("✓ All PPIs are up-to-date, nothing to aggregate")
        all_ppi_data = {}
    else:
        all_ppi_data = dm.aggregate_ppi_data(sector_start_dates=sector_start_dates)

        if all_ppi_data:
            dm.save_ppi_data_to_db(all_ppi_data)
            print(f"✓ Aggregated and saved {len(all_ppi_data)} sector PPIs (incremental).")
        else:
            print("✓ No PPI updates needed.")

    # STEP 3: Load PPIs from DB
    print("=" * 60)
    print("STEP 3: Loading PPIs from database")
    print("=" * 60)

    all_ppi_data_loaded = dm.load_ppi_data_from_db()

    if not all_ppi_data_loaded:
        print("!! ERROR: Failed to load PPIs from database. Exiting.")
        sys.exit(1)

    print(f"✓ Loaded {len(all_ppi_data_loaded)} PPIs from database.")
    print(f"Sectors: {', '.join(list(all_ppi_data_loaded.keys())[:5])}...")
    print()

    # STEP 4: Build Trading Calendar
    print("=" * 60)
    print("STEP 4: Building trading calendar from PPI data")
    print("=" * 60)

    all_dates = set()
    for sector_ppi in all_ppi_data_loaded.values():
        all_dates.update(sector_ppi.index)

    market_calendar = pd.DatetimeIndex(sorted(all_dates))

    backtest_date_range = market_calendar[
        (market_calendar >= pd.to_datetime(BACKTEST_START_DATE)) &
        (market_calendar <= pd.to_datetime(BACKTEST_END_DATE))
    ]

    if backtest_date_range.empty:
        print(f"!! ERROR: No valid trading days found between {BACKTEST_START_DATE} and {BACKTEST_END_DATE}.")
        sys.exit(1)

    print(f"✓ Found {len(backtest_date_range)} trading days.")
    print(f"Range: {backtest_date_range[0].strftime('%Y-%m-%d')} to {backtest_date_range[-1].strftime('%Y-%m-%d')}")
    print()

    # STEP 5: Calculate Market Breadth Using PPI-based Proxy
    print("=" * 60)
    print("STEP 5: Calculating PPI-based market breadth")
    print("=" * 60)
    print("✓ Using PPI position relative to MA20 (fast, no stock data needed)")
    print("✓ Storing in single wide table: market_breadth")
    print()

    # Check if breadth data already exists
    existing_breadth = dm.load_market_breadth_from_db()

    if existing_breadth is not None and not existing_breadth.empty:
        latest_breadth_date = existing_breadth.index.max()
        print(f"✓ Found existing breadth data up to: {latest_breadth_date.strftime('%Y-%m-%d')}")

        # Only calculate for dates after the latest breadth date
        dates_to_calculate = backtest_date_range[backtest_date_range > latest_breadth_date]

        if len(dates_to_calculate) == 0:
            print("✓ Market breadth is already up-to-date!")
        else:
            print(f"✓ Need to calculate breadth for {len(dates_to_calculate)} new dates")
    else:
        print("✓ No existing breadth data, will calculate for all dates")
        dates_to_calculate = backtest_date_range

    if len(dates_to_calculate) > 0:
        # Calculate PPI-based breadth for all sectors on all new dates
        print("Calculating PPI-based breadth...")
        breadth_data_by_date = {}

        for idx, date in enumerate(dates_to_calculate, 1):
            if idx % 10 == 0 or idx == len(dates_to_calculate):
                print(f"  Progress: {idx}/{len(dates_to_calculate)} dates...")

            breadth_data_by_date[date] = {}

            for sector in dm.SECTOR_STOCK_MAP.keys():
                if sector in all_ppi_data_loaded:
                    ppi_df = all_ppi_data_loaded[sector]
                    # Use PPI-based breadth proxy
                    breadth = calculate_ppi_breadth_proxy(ppi_df, date)
                    breadth_data_by_date[date][sector] = breadth
                else:
                    breadth_data_by_date[date][sector] = 0.5  # Neutral if PPI not found

        # Save to database
        print(f"Saving breadth data for {len(breadth_data_by_date)} dates...")
        dm.save_market_breadth_to_db(breadth_data_by_date)
        print(f"✓ Saved PPI-based breadth to database")

    print()
    print("=" * 60)
    print("ALL STEPS COMPLETE!")
    print("✓ Return-based PPI with dollar volume")
    print("✓ PPI-based market breadth (no stock data needed)")
    print("✓ Breadth stored in single database table")
    print("=" * 60)



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


    # === STEP 2: Fetch CSI300 Index ===
    print(f"{'='*60}")
    print(f"STEP 2: Fetching CSI300 index for market regime")
    print(f"{'='*60}")

    # Calculate lookback to cover backtest period + model fitting
    backtest_start = pd.to_datetime(BACKTEST_START_DATE)
    yesterday_naive = YESTERDAY_DATE.replace(tzinfo=None)
    days_needed = (yesterday_naive  - backtest_start).days + 200  # +200 for model fitting
    print(f"ℹ️  Beijing time: {TODAY_DATE.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    print(f"ℹ️  Fetching {days_needed} days of CSI300 data...")
    csi300_df = dm.get_index_data_live('000300.SH', lookback_days=days_needed, freq='daily')

    if csi300_df is None or csi300_df.empty:
        print("⚠️  WARNING: Could not fetch CSI300 data.")
        print("   Will use sector median as fallback for market regime.")
        csi300_df = None
    else:
        print(f"✅ Fetched CSI300 data: {len(csi300_df)} days")
        print(f"   Date range: {csi300_df.index[0].strftime('%Y-%m-%d')} to {csi300_df.index[-1].strftime('%Y-%m-%d')}")

        # Verify CSI300 has required columns
        required_cols = ['Close']
        missing_cols = [col for col in required_cols if col not in csi300_df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: CSI300 missing columns: {missing_cols}")
            print("   Will use sector median as fallback.")
            csi300_df = None

    print()


    # === STEP 3: Load PPIs from DB ===
    print(f"{'='*60}")
    print(f"STEP 3: Loading PPIs from database")
    print(f"{'='*60}")

    all_ppi_data_loaded = dm.load_ppi_data_from_db()

    if not all_ppi_data_loaded:
        print("!! ERROR: Failed to load PPIs from database. Exiting.")
        sys.exit(1)

    print(f"✅ Loaded {len(all_ppi_data_loaded)} PPIs from database.")
    print(f"  Sectors: {', '.join(list(all_ppi_data_loaded.keys())[:5])}...")
    print()

    # === STEP 4: Build Trading Calendar ===
    print(f"{'='*60}")
    print(f"STEP 3: Building trading calendar from PPI data")
    print(f"{'='*60}")

    all_dates = set()
    for sector_ppi in all_ppi_data_loaded.values():
        all_dates.update(sector_ppi.index)

    market_calendar = pd.DatetimeIndex(sorted(all_dates))
    backtest_date_range = market_calendar[
        (market_calendar >= pd.to_datetime(BACKTEST_START_DATE)) &
        (market_calendar <= pd.to_datetime(BACKTEST_END_DATE))
    ]

    if backtest_date_range.empty:
        print(f"!! ERROR: No valid trading days found between {BACKTEST_START_DATE} and {BACKTEST_END_DATE}.")
        sys.exit(1)

    print(f"✅ Found {len(backtest_date_range)} trading days for backtest.")
    print(f"  Range: {backtest_date_range[0].strftime('%Y-%m-%d')} to {backtest_date_range[-1].strftime('%Y-%m-%d')}")
    print()

    # === STEP 5: Run V2 Enhanced Backtest ===
    print(f"{'='*60}")
    print(f"STEP 5: Running V2 Enhanced Backtest (CSV Generation)")
    print(f"{'='*60}")
    print(f"Features:")
    print(f"  • CSI300 index for market regime (real benchmark)")
    print(f"  • Market context, volume confirmation, momentum")
    print(f"  • Adaptive thresholds")
    print(f"ℹ️  Note: Market_Breadth stored separately in DB, NOT in CSV")
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
                historical_scores=historical_scores,
                market_index_df=csi300_df  # ← PASS CSI300 HERE
            )

            if not daily_scorecard_v2.empty:

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

    # Check expected columns (excluding Market_Breadth)
    expected_cols = ['Date', 'Sector', 'TOTAL_SCORE', 'ACTION', 
                     'Market_Score', 'Market_Regime', 'Market_Source',
                     'Excess_Prob', 'Position_Size', 'Dispersion']
    missing_cols = [col for col in expected_cols if col not in full_results_v2.columns]

    if missing_cols:
        print(f"⚠️  WARNING: Missing expected columns: {missing_cols}")
    else:
        print(f"✅ All expected columns present (excluding Market_Breadth)")

    # Verify Market_Breadth is NOT in CSV
    if 'Market_Breadth' in full_results_v2.columns:
        print(f"⚠️  WARNING: Market_Breadth found in CSV! Should be in database only.")
    else:
        print(f"✅ Confirmed: Market_Breadth NOT in CSV (correct)")

    # Check Market_Source distribution
    if 'Market_Source' in full_results_v2.columns:
        source_counts = full_results_v2['Market_Source'].value_counts()
        print(f"\n📊 Market regime data sources:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} days ({count/len(full_results_v2)*100:.1f}%)")

    # Show sample stats
    print(f"\n📊 Sample statistics:")
    print(f"  Avg TOTAL_SCORE: {full_results_v2['TOTAL_SCORE'].mean():.3f}")
    print(f"  Score range: {full_results_v2['TOTAL_SCORE'].min():.3f} to {full_results_v2['TOTAL_SCORE'].max():.3f}")

    if 'Market_Score' in full_results_v2.columns:
        latest_date = full_results_v2['Date'].max()
        latest_rows = full_results_v2[full_results_v2['Date'] == latest_date]
        if not latest_rows.empty:
            latest_market = latest_rows.iloc[0]
            print(f"  Latest market regime: {latest_market.get('Market_Regime', 'N/A')}")
            print(f"  Latest market score: {latest_market.get('Market_Score', 0):.3f}")
            print(f"  Market data source: {latest_market.get('Market_Source', 'N/A')}")

    print()
    print(f"{'='*60}")
    print(f"✅ CSV GENERATION COMPLETE!")
    print(f"{'='*60}\n")


# --- 4. ENTRY POINT ---
if __name__ == "__main__":
    start_time = time.time()

    print(f"\n{'#'*60}")
    print(f"# ASSRS Daily Task - V2 Enhanced (Live QFQ Data)")
    print(f"# Started: {datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S')} (China Time)")
    print(f"{'#'*60}\n")

    try:
        # sector = '专业设备制造'
        # stock_list = dm.SECTOR_STOCK_MAP[sector]
        # updated = dm.add_new_sector_breadth(sector, stock_list)
        # print(f"✅ Backfilled {updated} rows")

        #Task 1: Calculate and store market breadth in database
        print("\n" + "="*60)
        print("TASK 1: Market Breadth Calculation (Store in Database)")
        print("="*60 + "\n")
        run_daily_ppi_and_market_breadth()


        # Task 2: Generate CSV with regime scores (WITHOUT Market_Breadth)
        print("\n" + "="*60)
        print("TASK 2: CSV Generation (Regime Scores with CSI300)")
        print("="*60 + "\n")
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
