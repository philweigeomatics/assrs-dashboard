import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import data_manager as dm
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

    # STEP 1.8: Update Margin Data
    print("=" * 60)
    print("STEP 1.8: Updating Margin Data (融资融券)")
    print("=" * 60)
    try:
        dm.update_daily_margin_data()
    except Exception as e:
        print(f"⚠ WARNING: margin data update failed: {e}")
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

    # Load dynamic sector map from DB (falls back to hardcoded if empty)
    sector_map = dm.get_sector_stock_map()
    print(f"✓ Loaded sector map: {len(sector_map)} sectors")

    sector_start_dates = {}
    for sector in sector_map.keys():
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
        all_ppi_data = dm.aggregate_ppi_data(
            sector_start_dates=sector_start_dates,
            sector_stock_map=sector_map,
        )

        if all_ppi_data:
            dm.save_ppi_data_to_db(all_ppi_data)
            print(f"✓ Aggregated and saved {len(all_ppi_data)} sector PPIs (incremental).")
        else:
            print("✓ No PPI updates needed.")

    # STEP 2.5: rolll portfolio.
    print("\n" + "="*50)
    print("🕒 STEP 4: Institutional Portfolio NAV Roll-Forward")
    print("="*50)
    try:
        dm.execute_daily_portfolio_rollup()
    except Exception as e:
        print(f"❌ Portfolio Roll-Forward failed: {e}")
        import traceback
        traceback.print_exc()

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

            for sector in sector_map.keys():
                if sector in all_ppi_data_loaded:
                    ppi_df = all_ppi_data_loaded[sector]
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


# --- 3. ENTRY POINT ---
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

        print("\n" + "="*60)
        print("TASK: Market Breadth Calculation (Store in Database)")
        print("="*60 + "\n")
        run_daily_ppi_and_market_breadth()

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
