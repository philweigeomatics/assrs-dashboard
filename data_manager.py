import os
import pandas as pd
import tushare as ts
import time
from datetime import datetime, timedelta # <-- 1. FIX: Added timedelta
import sqlite3
import sys
import numpy as np # Added for NaN handling

# --- 1. CONFIGURATION ---
DB_NAME = 'assrs_tushare_local.db'
TUSHARE_API = None

# If you prefer, move this from main.py so it's defined only once
TUSHARE_API_TOKEN = '36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312'
DATA_START_DATE = '20240101'  # earliest date you care about for single-stock fetches

# ---!!!--- 您的新14扇区地图（包含市场代理） ---!!!---
SECTOR_STOCK_MAP = {
    'MARKET_PROXY':['601318','600519','300750','300308','600036','601899','300502','000333','300274','601166','600900'], # 中国平安, 贵州茅台, 宁德时代, 中际旭创, 招商银行, 紫金矿业, 新易盛, 美的集团, 阳光电源, 兴业银行，长江电力
    '银行': ['601398','601939','601288'], # 工商银行, 建设银行, 农业银行     
    '非银金融':['600030','601318','601628'], # 中信证券, 中国平安, 中国人寿
    '半导体': ['688981','688041','688256','002371','688347'], # 中芯国际, 海光信息, 寒武纪，北方华创，华虹公司
    '软件': ['688111','002230','600588','300033','601360','300339','600570'],   # 金山办公, 科大讯飞, 用友网络, 同花顺，三六零，润和软件，恒生电子
    '光模块中游': ['300308','300394','002281','603083','300620','300548'], # 中际旭创, 天孚通讯, 光迅科技，剑桥科技，光库科技，长兴博创
    '军工电子': ['600760','002414','600562'],  # 中航沈飞, 高德红外, 国睿科技
    '家用电器': ['000333','600690','000921'], # 美的集团, 海尔智家, 海信家电
    '电力': ['600900','601985','600886'], # 长江电力, 中国核电, 国投电力
    '白酒': ['000568','000596', '600809','600519'],    # 泸州老窖, 古井贡酒, 山西汾酒，贵州茅台
    '电网设备':['600406','002028','600089','601877','300274','600312','601179'], # 国电南瑞, 思源电气, 特变电工，生态电器，阳光电源，平高电气，中国西电
    '电池': ['300014','002074','300750','688778','300450'],   # 亿纬锂能, 国轩高科, 宁德时代，厦钨新能，先导智能
    '整车':['600104','601633','000625','601238','002594'], # 上汽集团, 长城汽车, 长安汽车，广汽汽车，比亚迪
    '能源':['601800','601857','601225','600028','600938','002353','600188'] # 中国交建, 中国石油, 陕西煤业, 中国石化, 中国海油, 杰瑞股份，兖创能源
}

ALL_STOCK_TICKERS = sorted(list(set(ticker for stocks in SECTOR_STOCK_MAP.values() for ticker in stocks)))
REQUIRED_COLUMNS = ['Open', 'Close', 'High', 'Low', 'Volume'] 
MIN_HISTORY_DAYS = 100 
# MAX_RETRIES = 5 # <-- 2. FIX: Removed
VOL_ZSCORE_LOOKBACK = 100 # Lookback period for normalizing volume

# --- 2. TUSHARE & DATABASE FUNCTIONS ---

def init_tushare(token=None):
    """Initializes the Tushare API."""
    global TUSHARE_API

    if token is None:
        token = TUSHARE_API_TOKEN  # fall back to the global constant

    # your existing weird check stays harmless:
    if token == '36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312':
        pass

    try:
        ts.set_token(token)
        TUSHARE_API = ts.pro_api()
        print("Tushare token set. (Test query skipped)")
        return True
    except Exception as e:
        print(f"Tushare initialization failed: {e}")
        return False


def get_tushare_ticker(ticker):
    """Converts 6-digit ticker to Tushare's format (e.g., 601398 -> 601398.SH)"""
    if ticker.startswith('6') or ticker.startswith('688'): return f"{ticker}.SH"
    elif ticker.startswith('0') or ticker.startswith('3'): return f"{ticker}.SZ"
    return ticker

def create_table(conn, ticker):
    """Creates a table for a specific stock."""
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS "{ticker}" (
        Date TEXT PRIMARY KEY,
        Open REAL, 
        High REAL,
        Low REAL,
        Close REAL,
        Volume REAL
    );
    """)

def create_ppi_table(conn, sector_name):
    """Creates a table for a specific aggregated PPI."""
    table_name = f"PPI_{sector_name}"
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        Date TEXT PRIMARY KEY,
        Open REAL, 
        High REAL,
        Low REAL,
        Close REAL,
        Norm_Vol_Metric REAL
    );
    """)

def get_last_date_in_db(conn, ticker):
    """Finds the last date of data stored for a specific ticker."""
    try:
        cursor = conn.execute(f'SELECT MAX(Date) FROM "{ticker}"')
        last_date = cursor.fetchone()[0]
        return last_date
    except sqlite3.OperationalError:
        return None

def insert_data(conn, ticker, df):
    """Inserts a DataFrame into the stock's table."""
    df_to_insert = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_to_insert.index.name = 'Date'
    df_to_insert.reset_index(inplace=True)
    df_to_insert.to_sql(ticker, conn, if_exists='append', index=False)

def fetch_stock_data_robust(ticker, start_date, end_date):
    """
    Fetches and reconstructs hfq (forward-adjusted) data using Tushare Pro.
    --- V2: Simplified - removed retry loop ---
    """
    global TUSHARE_API
    ts_code = get_tushare_ticker(ticker)
    
    # --- 2. FIX: Removed retry loop, simplified to one try/except ---
    try:
        df = TUSHARE_API.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,open,close,high,low,vol'
        )
        
        if df.empty:
            # This is a valid response (e.g., weekend/holiday)
            return pd.DataFrame() 
        
        adj_factor_df = TUSHARE_API.adj_factor(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='trade_date,adj_factor'
        )
        if adj_factor_df.empty:
            raise ValueError(f"Tushare 'adj_factor' returned empty for {ts_code}.")

        df = pd.merge(df, adj_factor_df, on='trade_date', how='left')
        df['adj_factor'] = df['adj_factor'].fillna(method='ffill') 
        df.dropna(inplace=True) 

        last_factor = df['adj_factor'].iloc[0] 
        df['adj_factor_hfq'] = df['adj_factor'] / last_factor
        
        df['Open'] = df['open'] * df['adj_factor_hfq']
        df['Close'] = df['close'] * df['adj_factor_hfq']
        df['High'] = df['high'] * df['adj_factor_hfq']
        df['Low'] = df['low'] * df['adj_factor_hfq']
        df['Volume'] = df['vol'] 
        
        df['Date'] = pd.to_datetime(df['trade_date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) 
        
        df = df[REQUIRED_COLUMNS] 
        
        return df # Success

    except Exception as e:
        # If the single attempt fails, log it and return None
        error_message = f"'{str(e)}'" 
        print(f"    - FATAL ERROR: Fetch failed for {ticker}. Error: {error_message}")
        return None
    # --- 2. END OF FIX ---

# --- 3. PUBLIC FUNCTIONS ---

def ensure_data_in_db(start_date, end_date):
    """
    Main data function. Ensures all data from start_date to end_date is in the DB.
    Only fetches new or missing data.
    """
    with sqlite3.connect(DB_NAME) as conn:
        for ticker in ALL_STOCK_TICKERS:
            create_table(conn, ticker) 
            last_db_date_str = get_last_date_in_db(conn, ticker)
            
            fetch_start_date = start_date
            if last_db_date_str:
                try:
                    last_db_date_dt = pd.to_datetime(last_db_date_str).to_pydatetime()
                except Exception:
                    print(f"Warning: Could not parse date {last_db_date_str} for {ticker}. Refetching all.")
                    last_db_date_str = None
                
                if last_db_date_str:
                    last_db_date_fmt = last_db_date_dt.strftime('%Y%m%d')
                    if last_db_date_fmt >= end_date:
                        continue
                    fetch_start_date = (last_db_date_dt + timedelta(days=1)).strftime('%Y%m%d')

            print(f"  -> Fetching new data for {ticker} from {fetch_start_date} to {end_date}...")
            new_data_df = fetch_stock_data_robust(ticker, fetch_start_date, end_date)
            
            # This logic now handles None (fatal error) and empty (no data)
            if new_data_df is not None and not new_data_df.empty:
                if last_db_date_str:
                     new_data_df = new_data_df[new_data_df.index > last_db_date_str]
                
                if not new_data_df.empty:
                    insert_data(conn, ticker, new_data_df)
                    print(f"  - Successfully updated {ticker} with {len(new_data_df)} new rows.")
                else:
                    print(f"  - No new data to add for {ticker}.")
            elif new_data_df is not None and new_data_df.empty:
                print(f"  - No new data found for {ticker} in this period (weekend/holiday).")
            else:
                # This will now correctly catch the "None" returned after all retries failed
                print(f"  - FAILED to fetch new data for {ticker}.")
            
            time.sleep(1.0) 

def get_all_stock_data_from_db():
    """
    Loads all stock data from the DB into a dictionary of DataFrames.
    --- NEW: Also calculates the Volume Z-Score for each stock ---
    """
    all_stock_data = {}
    with sqlite3.connect(DB_NAME) as conn:
        for ticker in ALL_STOCK_TICKERS:
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{ticker}"', conn, index_col='Date', parse_dates=['Date'])
                if not df.empty:
                    missing_cols_db = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                    if missing_cols_db:
                        print(f"!! WARNING: Data for {ticker} in DB is missing columns: {missing_cols_db}.")
                        print(f"!! Please delete '{DB_NAME}' and re-run main.py to fix this.")
                    else:
                        # ---!!!--- NEW: Calculate normalized volume metric ---!!!---
                        df['Vol_Mean'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
                        df['Vol_Std'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
                        df['Norm_Vol_Metric'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']
                        # ---!!!--- END OF NEW ---!!!---
                        all_stock_data[ticker] = df.sort_index()
            except Exception as e:
                print(f"Failed to load data for {ticker} from DB: {e}")
    return all_stock_data

# ---!!!--- CRITICAL FIX: Rewritten PPI Aggregation Logic (V3.6) ---!!!---
def aggregate_ppi_data(all_stock_data):
    """
    Aggregates individual stock data into sector-level Proxy Portfolio Indexes (PPIs).
    V3.6: Uses "normalize-first, average-second" for Volume/Turnover.
          Calculates a fully valid OHLC Index (all values base 100).
    """
    print("\n--- Aggregating Stock Data into Sector PPIs (V3.6 Logic) ---")
    all_sector_ppi_data = {}
    
    for sector, stock_list in SECTOR_STOCK_MAP.items():
        
        # 1. Align all constituent stock data using an 'outer' join
        named_dfs = []
        for ticker in stock_list:
             if ticker in all_stock_data:
                temp_df = all_stock_data[ticker].copy()
                
                # ---!!!--- FIX 1: Handle Trading Halts ---!!!---
                # A halt is defined as Volume == 0.
                is_halted = (temp_df['Volume'] == 0)
                temp_df[is_halted] = np.nan
                # ---!!!--- END OF FIX 1 ---!!!---
                
                temp_df.name = ticker 
                named_dfs.append(temp_df)
                
        if not named_dfs: continue
        
        aligned_df = pd.concat(named_dfs, axis=1, keys=[df.name for df in named_dfs], join='outer')
        
        valid_tickers = [t for t in stock_list if (t, 'Close') in aligned_df.columns]
        if not valid_tickers: continue

        # 2. Get constituent data slices
        open_prices = aligned_df.xs('Open', level=1, axis=1)[valid_tickers]
        high_prices = aligned_df.xs('High', level=1, axis=1)[valid_tickers]
        low_prices = aligned_df.xs('Low', level=1, axis=1)[valid_tickers]
        close_prices = aligned_df.xs('Close', level=1, axis=1)[valid_tickers]
        
        # ---!!!--- NEW: Get the pre-calculated normalized volume metric ---!!!---
        norm_vol_metric = aligned_df.xs('Norm_Vol_Metric', level=1, axis=1)[valid_tickers]
        
        prev_close_prices = close_prices.shift(1)
        
        # --- 3. Calculate PPI Index (OHLC Fix) ---
        # .mean(axis=1) automatically skips NaNs (our halt-fix)
        ret_open = (open_prices / prev_close_prices - 1).mean(axis=1)
        ret_high = (high_prices / prev_close_prices - 1).mean(axis=1)
        ret_low = (low_prices / prev_close_prices - 1).mean(axis=1)
        ret_close = (close_prices / prev_close_prices - 1).mean(axis=1)

        # 4. Chain the returns to build the index
        ppi_df = pd.DataFrame(index=aligned_df.index)
        
        ppi_df['Close'] = 100 * (1 + ret_close.fillna(0)).cumprod()
        ppi_df['Open'] = ppi_df['Close'].shift(1) * (1 + ret_open)
        ppi_df['High'] = ppi_df['Close'].shift(1) * (1 + ret_high)
        ppi_df['Low'] = ppi_df['Close'].shift(1) * (1 + ret_low)
        
        # 5. Calculate Normalized Volume Metric (Average of Z-Scores)
        # .mean(axis=1) correctly calculates the avg of *active* stocks
        ppi_df['Norm_Vol_Metric'] = norm_vol_metric.mean(axis=1)
        
        ppi_df.dropna(inplace=True) # Drop initial rows with NaNs from lookbacks
        
        if len(ppi_df) >= MIN_HISTORY_DAYS:
            all_sector_ppi_data[sector] = ppi_df
        
    print("--- PPI Aggregation Complete ---")
    return all_sector_ppi_data

def save_ppi_data_to_db(all_ppi_data):
    """
    Saves the aggregated PPI DataFrames into new tables in the DB.
    """
    print(f"\n--- Saving {len(all_ppi_data)} PPIs to database '{DB_NAME}' ---")
    with sqlite3.connect(DB_NAME) as conn:
        for sector_name, ppi_df in all_ppi_data.items():
            table_name = f"PPI_{sector_name}"
            create_ppi_table(conn, sector_name) # Create the 'PPI_' table
            
            # Prep for insert
            df_to_insert = ppi_df.copy()
            df_to_insert = df_to_insert[['Open', 'High', 'Low', 'Close', 'Norm_Vol_Metric']]
            df_to_insert.index.name = 'Date'
            df_to_insert.reset_index(inplace=True)
            
            df_to_insert.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"  - Successfully saved '{table_name}' with {len(df_to_insert)} rows.")
    print("--- PPI Database Save Complete ---")

def load_ppi_data_from_db():
    """
    Loads all 'PPI_' tables from the DB into a dictionary of DataFrames.
    """
    print("\n--- Loading Pre-Calculated PPIs from Database ---")
    all_ppi_data = {}
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PPI_%';")
        ppi_tables = cursor.fetchall()
        
        if not ppi_tables:
            print(f"!! ERROR: No PPI tables found in '{DB_NAME}'.")
            print("!! Please run 'main.py' first to build the PPI tables.")
            return None
            
        for (table_name,) in ppi_tables:
            try:
                sector_name = table_name.replace('PPI_', '')
                df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn, index_col='Date', parse_dates=['Date'])
                if not df.empty:
                    # Rename 'Norm_Vol_Metric' back to a generic name for the logic files
                    df.rename(columns={'Norm_Vol_Metric': 'Volume_Metric'}, inplace=True)
                    all_ppi_data[sector_name] = df.sort_index()
            except Exception as e:
                print(f"Failed to load data for {table_name} from DB: {e}")
                
    print(f"--- Successfully loaded {len(all_ppi_data)} PPIs ---")
    return all_ppi_data

# ---!!!--- NEW FUNCTION YOU ASKED FOR ---!!!---
def get_single_stock_data_from_db(ticker):
    """
    Loads a single stock's data from the DB.
    This is for the webapp's on-demand analysis.
    """
    # Ensure ticker is in the master list for security
    if ticker not in ALL_STOCK_TICKERS:
        print(f"Warning: Ticker {ticker} not in the defined SECTOR_STOCK_MAP.")
        # We can still try to load it if it's in the DB
        pass

    with sqlite3.connect(DB_NAME) as conn:
        try:
            # Check if table exists
            cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ticker}';")
            if cursor.fetchone() is None:
                return None # Stock not in DB

            df = pd.read_sql_query(f'SELECT * FROM "{ticker}"', conn, index_col='Date', parse_dates=['Date'])
            
            if not df.empty:
                missing_cols_db = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_cols_db:
                    print(f"!! WARNING: Data for {ticker} in DB is missing columns: {missing_cols_db}.")
                    return None
                else:
                     # Calculate the Z-Score on the fly
                    df['Vol_Mean_100d'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
                    df['Vol_Std_100d'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
                    df['Volume_ZScore'] = (df['Volume'] - df['Vol_Mean_100d']) / df['Vol_Std_100d']
                    return df.sort_index()
            else:
                return None # Stock table is empty
        except Exception as e:
            print(f"Failed to load data for {ticker} from DB: {e}")
            return None
        

def get_single_stock_data(ticker, use_data_start_date: bool = True, lookback_years: int = 3):
    """
    Unified accessor for the webapp's single-stock analysis.

    Logic:
    1) Try to load from local SQLite via get_single_stock_data_from_db.
    2) If not found or empty, fetch from Tushare using fetch_stock_data_robust.
       - Uses DATA_START_DATE if use_data_start_date=True,
         otherwise uses last N years via lookback_years.
    3) Save the fetched data into SQLite for future use.
    4) Return a DataFrame indexed by Date with at least:
       Open, High, Low, Close, Volume (+ Volume_ZScore when coming from DB).
    """

    # --- Step 1: Try DB first ---
    df_db = get_single_stock_data_from_db(ticker)
    if df_db is not None and not df_db.empty:
        print(f"[data_manager] Loaded {ticker} from DB")
        return df_db

    # --- Step 2: Need to fetch from Tushare ---
    global TUSHARE_API
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print("[data_manager] Tushare initialization failed inside get_single_stock_data.")
            return None

    # Determine fetch range
    if use_data_start_date:
        start_str = DATA_START_DATE
    else:
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=365 * lookback_years)
        start_str = start_dt.strftime("%Y%m%d")
    end_str = datetime.today().strftime("%Y%m%d")

    print(f"[data_manager] {ticker} not found in DB, fetching from Tushare ({start_str} → {end_str})...")
    df_ts = fetch_stock_data_robust(ticker, start_str, end_str)

    if df_ts is None or df_ts.empty:
        print(f"[data_manager] Tushare fetch failed or returned empty for {ticker}.")
        return None

    # --- Step 3: Save fetched data into DB for next time ---
    try:
        with sqlite3.connect(DB_NAME) as conn:
            create_table(conn, ticker)

            df_to_insert = df_ts[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df_to_insert.index.name = 'Date'
            df_to_insert.reset_index(inplace=True)

            # Replace table with full history from Tushare
            df_to_insert.to_sql(ticker, conn, if_exists='replace', index=False)
            print(f"[data_manager] Saved {ticker} from Tushare into DB ({len(df_to_insert)} rows).")
    except Exception as e:
        print(f"[data_manager] Failed to save {ticker} Tushare data to DB: {e}")
        # Even if save fails, we can still return df_ts (without Volume_ZScore)
        return df_ts

    # --- Step 4: Reload via DB helper so Volume_ZScore is calculated ---
    df_final = get_single_stock_data_from_db(ticker)
    if df_final is None or df_final.empty:
        return df_ts  # fallback if something went weird

    return df_final


# ---!!!--- END OF NEW FUNCTION ---!!!---
