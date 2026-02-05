import os
import pandas as pd
import tushare as ts
import time
from datetime import datetime, timedelta
import sys
import numpy as np
from db_manager import db
import db_config

TUSHARE_API = None
TUSHARE_API_TOKEN = '36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312'
DATA_START_DATE = '20240101'

SECTOR_STOCK_MAP = {
    'MARKET_PROXY':['601318','600519','300750','300308','600036','601899','300502','000333','300274','601166','600900'], # 中国平安, 贵州茅台, 宁德时代, 中际旭创, 招商银行, 紫金矿业, 新易盛, 美的集团, 阳光电源, 兴业银行，长江电力
    '银行': ['601398','601939','601288'], # 工商银行, 建设银行, 农业银行     
    '非银金融':['600030','601318','601628','000750','000776','002736','002670'], # 中信证券, 中国平安, 中国人寿, 国海证券, 广发证券, 国信证券
    '半导体': ['688981','688041','688256','002371','688347','001309','002049','603986'], # 中芯国际, 海光信息, 寒武纪，北方华创，华虹公司，德明利，紫光国微，兆易创新
    '软件': ['688111','002230','600588','300033','601360','300339','600570'],   # 金山办公, 科大讯飞, 用友网络, 同花顺，三六零，润和软件，恒生电子
    '光模块中游': ['300308','300394','002281','603083','300620','300548'], # 中际旭创, 天孚通讯, 光迅科技，剑桥科技，光库科技，长兴博创
    '液冷':['002837','300499','301018','603019','000977'],# 英维克，高澜股份，申菱环境，中科曙光，浪潮信息
    '军工电子': ['600760','002414','600562','002179','688002','600990'],  # 中航沈飞, 高德红外, 国睿科技, 中航光电, 睿创微纳, 四创电子
    '风电设备':['002202','002531','002487','300443'], # 金风科技, 天顺风能, 大金重工,金雷股份
    '家用电器': ['000333','600690','000921','000651','002050','603486'], # 美的集团, 海尔智家, 海信家电, 格力电器, 三花智控, 科沃斯
    '电力': ['600900','601985','600886','600905','600795','600157'], # 长江电力, 中国核电, 国投电力, 三峡能源, 国电电力, 永泰能源
    '白酒': ['000568','000596', '600809','600519'],    # 泸州老窖, 古井贡酒, 山西汾酒，贵州茅台
    '电网设备':['600406','002028','600089','601877','300274','600312','601179'], # 国电南瑞, 思源电气, 特变电工，生态电器，阳光电源，平高电气，中国西电
    '电池': ['300014','002074','300750','688778','300450'],   # 亿纬锂能, 国轩高科, 宁德时代，厦钨新能，先导智能
    '整车':['600104','601633','000625','601238','002594','600418'], # 上汽集团, 长城汽车, 长安汽车，广汽汽车，比亚迪，江淮汽车
    '有色金属':['000630','000878','601899','600362','601600','000426'], # 铜陵有色, 云铝股份, 紫金矿业, 江西铜业, 中国铝业，兴业银锡
    '能源':['601800','601857','601225','600028','600938','002353','600188'] # 中国交建, 中国石油, 陕西煤业, 中国石化, 中国海油, 杰瑞股份，兖创能源
}


ALL_STOCK_TICKERS = sorted(list(set(ticker for stocks in SECTOR_STOCK_MAP.values() for ticker in stocks)))
REQUIRED_COLUMNS = ['Open', 'Close', 'High', 'Low', 'Volume']
MIN_HISTORY_DAYS = 100
VOL_ZSCORE_LOOKBACK = 100

def init_tushare(token=None):
    """Initializes the Tushare API."""
    global TUSHARE_API
    if token is None:
        token = TUSHARE_API_TOKEN
    
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
    """Converts 6-digit ticker to Tushare's format."""
    ticker = ticker.split('.')[0]
    
    if ticker.startswith(('43', '83', '87','92')):
        return f"{ticker}.BJ"
    elif ticker.startswith('6') or ticker.startswith('688'):
        return f"{ticker}.SH"
    elif ticker.startswith(('0', '2', '3')):
        return f"{ticker}.SZ"
    return ticker

def create_stock_basic_table():
    """Creates stock_basic table (only for SQLite)."""
    if db_config.USE_SQLITE:
        schema = """CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            area TEXT,
            industry TEXT,
            market TEXT,
            list_date TEXT,
            last_updated TEXT
        )"""
        db.create_table_sqlite(schema)


def update_stock_basic_table():
    """Fetch all stock basic info from Tushare and update database."""
    global TUSHARE_API
    if TUSHARE_API is None:
        init_tushare()
    if TUSHARE_API is None:
        print("[data_manager] ❌ Cannot update stock_basic - Tushare not initialized")
        return False

    # Use Beijing time
    from zoneinfo import ZoneInfo
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
    today_beijing = datetime.now(CHINA_TZ).date()
    today_str = today_beijing.strftime('%Y-%m-%d')

    create_stock_basic_table()
    
    try:
        # Check last update (compare Beijing dates)
        existing = db.read_table('stock_basic', columns='last_updated', order_by='-last_updated', limit=1)
        if not existing.empty:
            last_update = existing['last_updated'].iloc[0]
            last_update_date = datetime.strptime(last_update, '%Y-%m-%d').date()
            
            if last_update_date == today_beijing:
                print(f"[data_manager] ℹ️ stock_basic already updated today (Beijing: {today_str})")
                return False
    except:
        pass
    
    print(f"[data_manager] 🔄 Fetching stock_basic from Tushare (Beijing date: {today_str})...")
    
    try:
        df = TUSHARE_API.stock_basic(fields='ts_code,symbol,name,area,industry,market,list_date')
        if df.empty:
            print("[data_manager] ⚠️ No data returned from stock_basic API")
            return False
        
        # Use Beijing date for last_updated
        df['last_updated'] = today_str
        
        # Clear old data
        db.delete_all_records('stock_basic')
        
        # Insert new data
        records = df.to_dict('records')
        db.insert_records('stock_basic', records, upsert=True)
        
        print(f"[data_manager] ✅ Updated stock_basic table with {len(df)} stocks (Beijing: {today_str})")
        return True
    except Exception as e:
        print(f"[data_manager] ❌ Failed to fetch stock_basic: {e}")
        return False


def get_company_name_from_api(ticker):
    """Fallback: Fetch company name from Tushare stock_company API."""
    global TUSHARE_API
    if TUSHARE_API is None:
        init_tushare()
    if TUSHARE_API is None:
        print(f"[data_manager] ❌ Cannot fetch from API - Tushare not initialized")
        return None
    
    try:
        ts_code = get_tushare_ticker(ticker)
        time.sleep(0.31)
        df = TUSHARE_API.stock_company(ts_code=ts_code, fields='ts_code,com_name')
        if not df.empty:
            name = df.iloc[0]['com_name']
            print(f"[data_manager] ✅ Fetched from API: {ticker} -> {name}")
            return name
        else:
            print(f"[data_manager] ⚠️ No company info found for {ticker}")
            return None
    except Exception as e:
        print(f"[data_manager] ❌ Failed to fetch from API for {ticker}: {e}")
        return None

def get_stock_name_from_db(ticker):
    """Get stock name from database stock_basic table."""
    ts_code = get_tushare_ticker(ticker)
    
    try:
        df = db.read_table('stock_basic', filters={'ts_code': ts_code}, columns='name')
        if not df.empty:
            return df.iloc[0]['name']
        else:
            print(f"[data_manager] ⚠️ Stock {ticker} not found in stock_basic table")
            return None
    except Exception as e:
        print(f"[data_manager] ❌ Error fetching stock name: {e}")
        return None

def ensure_stock_basic_updated():
    """Ensures stock_basic table is updated (max once per day). Call this at app startup."""
    try:
        update_stock_basic_table()
    except Exception as e:
        print(f"[data_manager] ⚠️ Could not update stock_basic: {e}")

def create_history_table():
    """Creates search history table (only for SQLite)."""
    if db_config.USE_SQLITE:
        schema = """CREATE TABLE IF NOT EXISTS search_history (
            ticker TEXT PRIMARY KEY,
            timestamp TEXT,
            company_name TEXT
        )"""
        db.create_table_sqlite(schema)

def update_search_history(ticker):
    """Updates search history table with company name, keeps only last 10."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[data_manager] 🔍 Updating search history for {ticker}...")
    
    create_history_table()
    
    company_name = get_stock_name_from_db(ticker)
    if company_name is not None:
        print(f"[data_manager] 📦 Found in stock_basic DB: {company_name}")
    else:
        print(f"[data_manager] ⚠️ Not in stock_basic DB, calling stock_company API...")
        company_name = get_company_name_from_api(ticker)
        if not company_name:
            company_name = ticker
            print(f"[data_manager] ⚠️ API also failed, using ticker: {ticker}")
    
    try:
        # Upsert record
        db.insert_records('search_history', [{
            'ticker': ticker,
            'timestamp': timestamp,
            'company_name': company_name
        }], upsert=True)
        
        # Keep only last 10
        all_history = db.read_table('search_history', columns='ticker,timestamp', order_by='-timestamp')
        if len(all_history) > 10:
            to_delete = all_history.iloc[10:]['ticker'].tolist()
            for t in to_delete:
                db.delete_records('search_history', {'ticker': t})
        
        print(f"[data_manager] ✅ History saved: {ticker} - {company_name}")
        return company_name
    except Exception as e:
        print(f"[data_manager] ❌ Error updating search history: {e}")
        return ticker

def get_search_history(limit=10):
    """Returns list of last 10 searched stocks with display names."""
    try:
        create_history_table()
        df = db.read_table('search_history', columns='ticker,company_name,timestamp', order_by='-timestamp', limit=limit)
        
        if df.empty:
            return []
        
        results = []
        for _, row in df.iterrows():
            ticker = row['ticker']
            company_name = row.get('company_name', ticker)
            results.append({
                'ticker': ticker,
                'display': f"{company_name} ({ticker})"
            })
        return results
    except Exception as e:
        print(f"[data_manager] ❌ Error fetching search history: {e}")
        return []

def create_table(ticker):
    """Creates a table for a specific stock (only for SQLite)."""
    if db_config.USE_SQLITE:
        schema = f"""CREATE TABLE IF NOT EXISTS "{ticker}" (
            Date TEXT PRIMARY KEY,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL
        )"""
        db.create_table_sqlite(schema)

def create_ppi_table(sector_name):
    """Creates a table for a specific aggregated PPI (only for SQLite)."""
    if db_config.USE_SQLITE:
        table_name = f"PPI_{sector_name}"
        schema = f"""CREATE TABLE IF NOT EXISTS "{table_name}" (
            Date TEXT PRIMARY KEY,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Norm_Vol_Metric REAL
        )"""
        db.create_table_sqlite(schema)

def get_last_date_in_db(ticker):
    """Finds the last date of data stored for a specific ticker."""
    try:
        if not db.table_exists(ticker):
            return None
        
        df = db.read_table(ticker, columns='Date', order_by='-Date', limit=1)
        if not df.empty:
            return df.iloc[0]['Date']
        return None
    except:
        return None

def insert_data(ticker, df):
    """Inserts a DataFrame into the stock's table."""
    df_to_insert = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_to_insert.index.name = 'Date'
    df_to_insert.reset_index(inplace=True)
    
    # Normalize Date to string format YYYY-MM-DD
    df_to_insert['Date'] = pd.to_datetime(df_to_insert['Date']).dt.strftime('%Y-%m-%d')

    records = df_to_insert.to_dict('records')
    db.insert_records(ticker, records, upsert=True)

def fetch_stock_data_robust(ticker, start_date, end_date):
    """Fetches and reconstructs hfq (forward-adjusted) data using Tushare Pro."""
    global TUSHARE_API
    ts_code = get_tushare_ticker(ticker)

    print(f" - Fetching data for {ticker} ({ts_code}) from {start_date} to {end_date}...")  
    
    try:
        df = TUSHARE_API.daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                               fields='trade_date,open,close,high,low,vol')
        if df.empty:
            return pd.DataFrame()
        
        adj_factor_df = TUSHARE_API.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date,
                                               fields='trade_date,adj_factor')
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
        
        return df
    except Exception as e:
        error_message = f"'{str(e)}'"
        print(f" - FATAL ERROR: Fetch failed for {ticker}. Error: {error_message}")
        return None

def ensure_data_in_db(start_date, end_date):
    """Main data function. Ensures all data from start_date to end_date is in the DB."""
    for ticker in ALL_STOCK_TICKERS:
        create_table(ticker)
        last_db_date_str = get_last_date_in_db(ticker)
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
        
        print(f" -> Fetching new data for {ticker} from {fetch_start_date} to {end_date}...")
        new_data_df = fetch_stock_data_robust(ticker, fetch_start_date, end_date)
        
        if new_data_df is not None and not new_data_df.empty:
            if last_db_date_str:
                new_data_df = new_data_df[new_data_df.index > last_db_date_str]
            
            if not new_data_df.empty:
                insert_data(ticker, new_data_df)
                print(f" - Successfully updated {ticker} with {len(new_data_df)} new rows.")
            else:
                print(f" - No new data to add for {ticker}.")
        elif new_data_df is not None and new_data_df.empty:
            print(f" - No new data found for {ticker} in this period (weekend/holiday).")
        else:
            print(f" - FAILED to fetch new data for {ticker}.")
        
        time.sleep(1.0)


def get_all_stock_data_live(progress_callback=None):
    """
    Fetch ALL stocks data LIVE from Tushare API (no database).
    Uses qfq (forward adjusted) prices.
    
    Args:
        progress_callback: Optional function(current, total, ticker) to report progress
    
    Returns:
        Dictionary of {ticker: DataFrame}
    """
    global TUSHARE_API
    
    print(f"[data_manager] 📡 Fetching ALL {len(ALL_STOCK_TICKERS)} stocks live from Tushare (qfq)...")
    
    # Initialize Tushare if needed
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print(f"[data_manager] ❌ Tushare initialization failed")
            return {}
    
    # Calculate date range (3 years of data for technical analysis)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 3)
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    all_stock_data = {}
    total = len(ALL_STOCK_TICKERS)
    
    for idx, ticker in enumerate(ALL_STOCK_TICKERS, 1):
        if progress_callback:
            progress_callback(idx, total, ticker)
        
        try:
            ts_code = get_tushare_ticker(ticker)
            
            # Fetch with qfq
            df = ts.pro_bar(
                ts_code=ts_code,
                adj='qfq',
                start_date=start_str,
                end_date=end_str,
                asset='E'
            )
            
            if df is None or df.empty:
                print(f"  ⚠️ {ticker}: No data")
                continue
            
            # Rename columns
            df = df.rename(columns={
                'trade_date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume'
            })
            
            # Convert date and set index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Keep only required columns
            df = df[REQUIRED_COLUMNS]
            
            # Calculate volume metrics
            df['Vol_Mean'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
            df['Vol_Std'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
            df['Norm_Vol_Metric'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']
            
            all_stock_data[ticker] = df
            
            # Rate limiting - sleep briefly to avoid hitting API limits
            time.sleep(0.3)  # 200 calls/min at 2000 points = 1 call per 0.3s
            
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")
            continue
    
    print(f"[data_manager] ✅ Loaded {len(all_stock_data)}/{total} stocks successfully (qfq adjusted)")
    
    return all_stock_data

def get_all_stock_data_from_db():
    """Loads all stock data from the DB into a dictionary of DataFrames."""
    all_stock_data = {}
    
    for ticker in ALL_STOCK_TICKERS:
        try:
            if not db.table_exists(ticker):
                continue
            
            df = db.read_table(ticker)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
                df = df.set_index('Date').sort_index()
                
                missing_cols_db = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_cols_db:
                    print(f"!! WARNING: Data for {ticker} in DB is missing columns: {missing_cols_db}.")
                else:
                    df['Vol_Mean'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
                    df['Vol_Std'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
                    df['Norm_Vol_Metric'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']
                    all_stock_data[ticker] = df
        except Exception as e:
            print(f"Failed to load data for {ticker} from DB: {e}")
    
    return all_stock_data

def get_single_stock_data_from_db(ticker):
    """Loads a single stock's data from the DB."""
    try:
        if not db.table_exists(ticker):
            return None
        
        df = db.read_table(ticker)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df = df.set_index('Date').sort_index()
            
            missing_cols_db = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols_db:
                print(f"!! WARNING: Data for {ticker} in DB is missing columns: {missing_cols_db}.")
                return None
            
            df['Vol_Mean_100d'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
            df['Vol_Std_100d'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
            df['Volume_ZScore'] = (df['Volume'] - df['Vol_Mean_100d']) / df['Vol_Std_100d']
            return df
        else:
            return None
    except Exception as e:
        print(f"Failed to load data for {ticker} from DB: {e}")
        return None

def get_single_stock_data(ticker, use_data_start_date: bool = True, lookback_years: int = 3):
    """
    Unified accessor for single-stock analysis.
    ✅ FIXED: Now checks if data is stale and updates accordingly
    """
    global TUSHARE_API

    if ticker not in ALL_STOCK_TICKERS:
        print(f"[data_manager] Warning: Ticker {ticker} not in the defined SECTOR_STOCK_MAP.")
    
    # === Step 1: Check if data exists in DB ===
    df_db = get_single_stock_data_from_db(ticker)
    
    if df_db is not None and not df_db.empty:
        # ✅ NEW: Check if data is stale
        last_date_in_db = df_db.index.max()
        
        # Get current date (Beijing time)
        from zoneinfo import ZoneInfo
        CHINA_TZ = ZoneInfo("Asia/Shanghai")
        today_beijing = datetime.now(CHINA_TZ).date()
        yesterday_beijing = today_beijing - timedelta(days=1)
        
        # Convert last_date_in_db to date for comparison
        if isinstance(last_date_in_db, pd.Timestamp):
            last_date_in_db = last_date_in_db.date()
        
        # ✅ If data is up-to-date (has yesterday's data), return it
        if last_date_in_db >= yesterday_beijing:
            print(f"[data_manager] Loaded {ticker} from DB (up-to-date: {last_date_in_db})")
            return df_db
        else:
            # ✅ Data is stale, need to update
            print(f"[data_manager] {ticker} data is stale (last date: {last_date_in_db}), updating...")
            
            # Fetch only missing dates
            fetch_start_date = (last_date_in_db + timedelta(days=1)).strftime('%Y%m%d')
            fetch_end_date = datetime.today().strftime('%Y%m%d')
            
            print(f"[data_manager] Fetching {ticker} from {fetch_start_date} to {fetch_end_date}...")
            
            if TUSHARE_API is None:
                ok = init_tushare()
                if not ok:
                    print("[data_manager] Tushare initialization failed, returning stale data.")
                    return df_db
            
            # Fetch incremental data
            df_new = fetch_stock_data_robust(ticker, fetch_start_date, fetch_end_date)
            
            if df_new is not None and not df_new.empty:
                try:
                    # Insert new data
                    insert_data(ticker, df_new)
                    print(f"[data_manager] Updated {ticker} with {len(df_new)} new rows.")
                    
                    # Reload from DB to get full dataset with calculated metrics
                    df_final = get_single_stock_data_from_db(ticker)
                    if df_final is not None:
                        return df_final
                except Exception as e:
                    print(f"[data_manager] Failed to update {ticker}: {e}")
            else:
                print(f"[data_manager] No new data available for {ticker} (weekend/holiday)")
            
            # Return existing data if update failed
            return df_db
    
    # === Step 2: No data in DB, fetch full history from Tushare ===
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print("[data_manager] Tushare initialization failed inside get_single_stock_data.")
            return None
    
    # Determine date range
    if use_data_start_date:
        start_str = DATA_START_DATE
    else:
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=365 * lookback_years)
        start_str = start_dt.strftime('%Y%m%d')
    
    end_str = datetime.today().strftime('%Y%m%d')
    
    print(f"[data_manager] {ticker} not found in DB, fetching from Tushare ({start_str} - {end_str})...")
    df_ts = fetch_stock_data_robust(ticker, start_str, end_str)
    
    if df_ts is None or df_ts.empty:
        print(f"[data_manager] Tushare fetch failed or returned empty for {ticker}.")
        return None
    
    # === Step 3: Save to DB ===
    try:
        create_table(ticker)
        insert_data(ticker, df_ts)
        print(f"[data_manager] Saved {ticker} from Tushare into DB ({len(df_ts)} rows).")
    except Exception as e:
        print(f"[data_manager] Failed to save {ticker} Tushare data to DB: {e}")
    
    # === Step 4: Reload from DB to get calculated metrics ===
    df_final = get_single_stock_data_from_db(ticker)
    if df_final is None or df_final.empty:
        return df_ts
    return df_final


def get_single_stock_data_live(ticker, lookback_years=3):
    """
    Fetch single stock data DIRECTLY from Tushare API (no database).
    Uses qfq (forward adjusted) prices for dividend/split adjustment.
    
    Args:
        ticker: 6-digit stock code (e.g., '600809')
        lookback_years: How many years of history to fetch (default: 3)
    
    Returns:
        DataFrame with OHLCV data and calculated technical indicators
    """
    global TUSHARE_API
    
    print(f"[data_manager] Fetching {ticker} live from Tushare API (qfq)...")
    
    # Initialize Tushare if needed
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print(f"[data_manager] ❌ Tushare initialization failed")
            return None
    
    # Calculate date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * lookback_years)
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    ts_code = get_tushare_ticker(ticker)
    
    try:
        # ✅ Fetch with qfq (forward adjusted) - current price = market price
        df = ts.pro_bar(
            ts_code=ts_code,
            adj='qfq',
            start_date=start_str,
            end_date=end_str,
            asset='E'
        )
        
        if df is None or df.empty:
            print(f"[data_manager] ❌ No data returned for {ticker}")
            return None
        
        # Rename columns to match your expected format
        df = df.rename(columns={
            'trade_date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        })
        
        # Convert date and set index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Keep only required columns
        df = df[REQUIRED_COLUMNS]
        
        # ✅ Calculate volume metrics (for technical indicators)
        df['Vol_Mean_100d'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
        df['Vol_Std_100d'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
        df['Volume_ZScore'] = (df['Volume'] - df['Vol_Mean_100d']) / df['Vol_Std_100d']
        
        print(f"[data_manager] ✅ Fetched {len(df)} rows for {ticker} (qfq adjusted)")
        print(f"[data_manager]    Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"[data_manager]    Latest Close: ¥{df['Close'].iloc[-1]:.2f} (market price)")
        
        return df
        
    except Exception as e:
        print(f"[data_manager] ❌ Failed to fetch {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_ppi_data(all_stock_data, sector_start_dates=None):
    """
    Aggregates individual stock data into sector-level Proxy Portfolio Indexes (PPIs).
    
    Args:
        all_stock_data: Dict of ticker -> DataFrame
        sector_start_dates: Dict of sector -> start_date (None = full aggregation for that sector)
                           If None (not provided), do full aggregation for all sectors
    """
    print("--- Aggregating Stock Data into Sector PPIs (V3.6 Logic) ---")
    
    all_sector_ppi_data = {}
    
    for sector, stock_list in SECTOR_STOCK_MAP.items():
        # ✅ Determine start_date for THIS sector
        if sector_start_dates and sector in sector_start_dates:
            start_date = sector_start_dates[sector]
            if start_date:
                print(f"  ℹ️ {sector}: Incremental mode from {start_date}")
            else:
                print(f"  ℹ️ {sector}: Full aggregation mode")
        else:
            start_date = None
        
        # Collect DataFrames for this sector
        named_dfs = []
        for ticker in stock_list:
            if ticker in all_stock_data:
                temp_df = all_stock_data[ticker].copy()
                
                # ✅ FILTER: Only process data from start_date onwards (if specified)
                if start_date:
                    temp_df = temp_df[temp_df.index >= pd.to_datetime(start_date)]
                    if temp_df.empty:
                        continue
                
                # Mark halted days
                is_halted = (temp_df['Volume'] == 0)
                temp_df['is_halted'] = np.nan
                temp_df = temp_df[~is_halted]
                temp_df.name = ticker
                named_dfs.append(temp_df)
        
        if not named_dfs:
            continue
        
        # Concatenate all stock data for this sector
        aligned_df = pd.concat(
            named_dfs, 
            axis=1, 
            keys=[df.name for df in named_dfs], 
            join='outer'
        )
        
        valid_tickers = [t for t in stock_list if (t, 'Close') in aligned_df.columns]
        if not valid_tickers:
            continue
        
        # Calculate returns
        open_prices = aligned_df.xs('Open', level=1, axis=1)[valid_tickers]
        high_prices = aligned_df.xs('High', level=1, axis=1)[valid_tickers]
        low_prices = aligned_df.xs('Low', level=1, axis=1)[valid_tickers]
        close_prices = aligned_df.xs('Close', level=1, axis=1)[valid_tickers]
        prev_close_prices = close_prices.shift(1)
        
        ret_open = ((open_prices / prev_close_prices) - 1).mean(axis=1)
        ret_high = ((high_prices / prev_close_prices) - 1).mean(axis=1)
        ret_low = ((low_prices / prev_close_prices) - 1).mean(axis=1)
        ret_close = ((close_prices / prev_close_prices) - 1).mean(axis=1)
        norm_vol_metric = aligned_df.xs('Norm_Vol_Metric', level=1, axis=1)[valid_tickers]
        
        # Build PPI DataFrame
        ppi_df = pd.DataFrame(index=aligned_df.index)
        ppi_df['Close'] = (100 * (1 + ret_close.fillna(0)).cumprod())
        ppi_df['Open'] = ppi_df['Close'].shift(1) * (1 + ret_open)
        ppi_df['High'] = ppi_df['Close'].shift(1) * (1 + ret_high)
        ppi_df['Low'] = ppi_df['Close'].shift(1) * (1 + ret_low)
        ppi_df['Norm_Vol_Metric'] = norm_vol_metric.mean(axis=1)
        ppi_df.dropna(inplace=True)
        
        # ✅ De-duplicate dates
        ppi_df = ppi_df[~ppi_df.index.duplicated(keep='last')]
        
        # ✅ Check minimum history requirement
        # For incremental updates, we only need at least 1 row
        # For full aggregation, we need MIN_HISTORY_DAYS
        min_required = 1 if start_date else MIN_HISTORY_DAYS
        
        if len(ppi_df) >= min_required:
            all_sector_ppi_data[sector] = ppi_df
            print(f"  ✅ {sector}: Aggregated {len(ppi_df)} dates")
        else:
            print(f"  ⚠️ {sector}: Insufficient data ({len(ppi_df)} < {min_required}), skipping")
    
    print("--- PPI Aggregation Complete ---")
    return all_sector_ppi_data

def save_ppi_data_to_db(all_ppi_data):
    """
    Saves the aggregated PPI DataFrames into tables in the DB.
    Only inserts NEW dates that don't already exist in the database.
    """
    print(f"--- Saving {len(all_ppi_data)} PPIs to database (incremental) ---")
    
    for sectorname, ppi_df in all_ppi_data.items():
        tablename = f"PPI_{sectorname}"
        create_ppi_table(sectorname)
        
        if ppi_df is None or ppi_df.empty:
            print(f"  ⚠️ Skipping {sectorname} - no data")
            continue
        
        # ✅ STEP 1: De-duplicate dates in new data (CRITICAL FIX)
        df_to_insert = ppi_df.copy()
        df_to_insert.index = pd.to_datetime(df_to_insert.index)
        df_to_insert = df_to_insert[~df_to_insert.index.duplicated(keep='last')]
        
        # ✅ STEP 2: Get existing dates from database
        try:
            existing_df = db.read_table(tablename, columns='Date')
            if not existing_df.empty:
                # Normalize existing dates to datetime for comparison
                existing_dates = pd.to_datetime(existing_df['Date']).dt.strftime('%Y-%m-%d').tolist()
                existing_dates_set = set(existing_dates)
            else:
                existing_dates_set = set()
        except Exception as e:
            print(f"  ℹ️ {tablename} doesn't exist yet or is empty, will create with all data")
            existing_dates_set = set()
        
        # ✅ STEP 3: Filter out dates that already exist
        new_dates_mask = ~df_to_insert.index.strftime('%Y-%m-%d').isin(existing_dates_set)
        df_new_only = df_to_insert[new_dates_mask].copy()
        
        if df_new_only.empty:
            print(f"  ✅ {sectorname}: Already up-to-date (no new dates to add)")
            continue
        
        # ✅ STEP 4: Prepare data for insertion
        df_new_only = df_new_only[['Open', 'High', 'Low', 'Close', 'Norm_Vol_Metric']]
        df_new_only.index.name = 'Date'
        df_new_only.reset_index(inplace=True)
        
        # Normalize dates to YYYY-MM-DD format (no timestamp)
        df_new_only['Date'] = pd.to_datetime(df_new_only['Date']).dt.strftime('%Y-%m-%d')
        
        # ✅ STEP 5: Final de-duplication (CRITICAL - prevents the duplicate key error)
        df_new_only = df_new_only.drop_duplicates(subset=['Date'], keep='last')
        
        records = df_new_only.to_dict('records')
        
        # ✅ STEP 6: Insert only new records (use upsert=False since we filtered out existing)
        try:
            db.insert_records(tablename, records, upsert=False)
            print(f"  ✅ {sectorname}: Added {len(df_new_only)} new dates "
                  f"({df_new_only['Date'].min()} to {df_new_only['Date'].max()})")
        except Exception as e:
            # If insert fails (rare case), try upsert as fallback
            print(f"  ⚠️ {sectorname}: Regular insert failed, trying upsert: {e}")
            try:
                # Extra de-duplication before upsert
                df_new_only = df_new_only.drop_duplicates(subset=['Date'], keep='last')
                records = df_new_only.to_dict('records')
                db.insert_records(tablename, records, upsert=True)
                print(f"  ✅ {sectorname}: Upserted {len(df_new_only)} records")
            except Exception as e2:
                print(f"  ❌ {sectorname}: Failed to save PPI data: {e2}")
                continue
    
    print("--- PPI Database Save Complete ---")



def load_ppi_data_from_db():
    """Loads all 'PPI_' tables from the DB into a dictionary of DataFrames."""
    print("\n--- Loading Pre-Calculated PPIs from Database ---")
    
    all_ppi_data = {}
    
    for sector_name in SECTOR_STOCK_MAP.keys():
        table_name = f"PPI_{sector_name}"
        try:
            if not db.table_exists(table_name):
                continue
            
            df = db.read_table(table_name)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
                df = df.set_index('Date').sort_index()
                df.rename(columns={'Norm_Vol_Metric': 'Volume_Metric'}, inplace=True)
                all_ppi_data[sector_name] = df
        except Exception as e:
            print(f"Failed to load data for {table_name} from DB: {e}")
    
    if not all_ppi_data:
        print("!! ERROR: No PPI tables found in database.")
        print("!! Please run 'main.py' first to build the PPI tables.")
        return None
    
    print(f"--- Successfully loaded {len(all_ppi_data)} PPIs ---")
    return all_ppi_data

def create_signals_tables():
    """Creates signals cache tables (for Today's Alerts page)."""
    if db_config.USE_SQLITE:
        schema1 = """CREATE TABLE IF NOT EXISTS daily_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT,
            type TEXT,
            ticker TEXT,
            name TEXT,
            signals TEXT,
            signal_count INTEGER,
            price REAL,
            rsi REAL,
            adx REAL,
            macd REAL,
            volume REAL,
            created_at TEXT
        )"""
        db.create_table_sqlite(schema1)
        
        try:
            db.execute_raw_sql("""CREATE INDEX IF NOT EXISTS idx_signals_scan_date 
                                 ON daily_signals(scan_date)""")
        except:
            pass
        
        schema2 = """CREATE TABLE IF NOT EXISTS signals_scan_metadata (
            scan_date TEXT PRIMARY KEY,
            total_stocks_scanned INTEGER,
            opportunities_found INTEGER,
            alerts_found INTEGER,
            scan_duration_seconds REAL,
            created_at TEXT
        )"""
        db.create_table_sqlite(schema2)
    # else: Supabase tables already exist from migration script

def get_cached_signals(scan_date):
    """Get cached signals for a specific date. Returns DataFrame if found, None otherwise."""
    try:
        # Check if data exists for this date
        check_df = db.read_table('daily_signals', filters={'scan_date': scan_date}, columns='*', limit=1)
        
        if check_df.empty:
            return None
        
        # Load cached data
        df = db.read_table('daily_signals', filters={'scan_date': scan_date}, columns='*', order_by='type, -signal_count')
        
        # Drop id and created_at columns
        df = df.drop(columns=['id', 'scan_date', 'created_at'], errors='ignore')
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'type': 'Type',
            'ticker': 'Ticker',
            'name': 'Name',
            'signals': 'Signals',
            'signal_count': 'Signal_Count',
            'price': 'Price',
            'rsi': 'RSI',
            'adx': 'ADX',
            'macd': 'MACD',
            'volume': 'Volume'
        })
        
        return df
    except Exception as e:
        print(f"Error loading cached signals: {e}")
        return None

def save_signals_to_cache(df, scan_date, scan_duration):
    """Save scanned signals to database cache."""
    if df is None or df.empty:
        return False
    
    try:
        # Delete old data for this date (in case of rescan)
        db.delete_records('daily_signals', {'scan_date': scan_date})
        db.delete_records('signals_scan_metadata', {'scan_date': scan_date})
        
        # Prepare data for insertion
        df_to_save = df.copy()
        df_to_save['scan_date'] = scan_date
        df_to_save['created_at'] = datetime.now().isoformat()
        
        # Rename columns to match database schema
        df_to_save = df_to_save.rename(columns={
            'Type': 'type',
            'Ticker': 'ticker',
            'Name': 'name',
            'Signals': 'signals',
            'Signal_Count': 'signal_count',
            'Price': 'price',
            'RSI': 'rsi',
            'ADX': 'adx',
            'MACD': 'macd',
            'Volume': 'volume'
        })
        
        # Save signals
        records = df_to_save.to_dict('records')
        db.insert_records('daily_signals', records, upsert=True)
        
        # Save metadata
        opportunities = len(df[df['Type'] == '🚀 Opportunity'])
        alerts = len(df[df['Type'] == '⚠️ Alert'])
        total_stocks = len(ALL_STOCK_TICKERS)
        
        metadata = {
            'scan_date': scan_date,
            'total_stocks_scanned': total_stocks,
            'opportunities_found': opportunities,
            'alerts_found': alerts,
            'scan_duration_seconds': scan_duration,
            'created_at': datetime.now().isoformat()
        }
        
        db.insert_records('signals_scan_metadata', [metadata], upsert=True)
        
        return True
    except Exception as e:
        print(f"Error saving signals to cache: {e}")
        return False

def get_scan_metadata(scan_date):
    """Get metadata about the last scan."""
    try:
        df = db.read_table('signals_scan_metadata', filters={'scan_date': scan_date})
        
        if df.empty:
            return None
        
        row = df.iloc[0]
        return {
            'scan_date': row['scan_date'],
            'total_stocks_scanned': row['total_stocks_scanned'],
            'opportunities_found': row['opportunities_found'],
            'alerts_found': row['alerts_found'],
            'scan_duration_seconds': row['scan_duration_seconds'],
            'created_at': row['created_at']
        }
    except:
        return None
