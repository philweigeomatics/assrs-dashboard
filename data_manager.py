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
    '银行': ['601398','601939','601288','600036','601998','000001'], # 工商银行, 建设银行, 农业银行, 招商银行, 中信银行, 平安银行     
    '非银金融':['600030','601318','601628','000750','000776','002736','002670'], # 中信证券, 中国平安, 中国人寿, 国海证券, 广发证券, 国信证券
    '半导体': ['688981','688041','688256','002371','688347','001309','002049','603986'], # 中芯国际, 海光信息, 寒武纪，北方华创，华虹公司，德明利，紫光国微，兆易创新
    '软件': ['688111','002230','600588','300033','601360','300339','600570'],   # 金山办公, 科大讯飞, 用友网络, 同花顺，三六零，润和软件，恒生电子
    '光模块中游': ['300308','300394','002281','603083','300620','300548'], # 中际旭创, 天孚通讯, 光迅科技，剑桥科技，光库科技，长兴博创
    '液冷':['002837','300499','301018','603019','000977','000938'],# 英维克，高澜股份，申菱环境，中科曙光，浪潮信息，紫光股份
    '军工电子': ['600760','002414','600562','002179','688002','600990'],  # 中航沈飞, 高德红外, 国睿科技, 中航光电, 睿创微纳, 四创电子
    '风电设备':['002202','002531','002487','300443'], # 金风科技, 天顺风能, 大金重工,金雷股份
    '家用电器': ['000333','600690','000921','000651','002050','603486'], # 美的集团, 海尔智家, 海信家电, 格力电器, 三花智控, 科沃斯
    '电力': ['600900','601985','600886','600905','600795','600157'], # 长江电力, 中国核电, 国投电力, 三峡能源, 国电电力, 永泰能源
    '白酒': ['000568','000596', '600809','600519','000858','002304'],    # 泸州老窖, 古井贡酒, 山西汾酒，贵州茅台, 五粮液，洋河股份
    '电网设备':['600406','002028','600089','601877','300274','600312','601179'], # 国电南瑞, 思源电气, 特变电工，生态电器，阳光电源，平高电气，中国西电
    '电池': ['300014','002074','300750','688778','300450'],   # 亿纬锂能, 国轩高科, 宁德时代，厦钨新能，先导智能
    '整车':['600104','601633','000625','601238','002594','600418'], # 上汽集团, 长城汽车, 长安汽车，广汽汽车，比亚迪，江淮汽车
    '有色金属':['000630','000878','601899','600362','601600','000426'], # 铜陵有色, 云铝股份, 紫金矿业, 江西铜业, 中国铝业，兴业银锡
    '能源':['601800','601857','601225','600028','600938','002353','600188'], # 中国交建, 中国石油, 陕西煤业, 中国石化, 中国海油, 杰瑞股份，兖创能源
    '机器人': ['300124','601689','688777','002008','002472','688017'] # 汇川技术, 拓普集团, 中控技术, 大族激光，双环传动，绿的谐波
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
        new_data_df = get_single_stock_data_live(ticker, start_date=fetch_start_date, end_date=end_date)
        
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

def migrate_to_qfq_data():
    """ONE-TIME migration using get_single_stock_data_live()."""
    # ... (same print statements) ...
    total = len(ALL_STOCK_TICKERS)
    success_count = 0
    for idx, ticker in enumerate(ALL_STOCK_TICKERS, 1):
        print(f"[{idx}/{total}] Processing {ticker}...", end=' ')
        
        try:
            # Use the multifunctional get_single_stock_data_live
            df = get_single_stock_data_live(
                ticker,
                start_date=DATA_START_DATE,
                end_date=datetime.today().strftime('%Y%m%d')
            )
            
            if df is None or df.empty:
                print(f"❌ No data")
                continue
            
            # DELETE old data and INSERT new qfq data
            if db.table_exists(ticker):
                db.delete_all_records(ticker)
                print(f"🗑️ Deleted old", end=' ')
            
            create_table(ticker)
            insert_data(ticker, df)
            
            print(f"✅ {len(df)} rows (qfq)")
            success_count += 1
            time.sleep(0.31)
            
        except Exception as e:
            print(f"❌ {e}")
            continue

    print("=" * 60)
    print(f"✅ Migration Complete: {success_count}/{total} stocks migrated to qfq")
    print("=" * 60)
    return True


def get_index_data_live(index_code='000300.SH', lookback_days=180, freq='daily'):
    """
    Fetch index data from Tushare (e.g., CSI 300).
    
    Args:
        index_code: Index code in Tushare format (e.g., '000300.SH' for CSI 300)
        lookback_days: Number of days of historical data
        freq: 'daily' or 'weekly' - determines which API to call
    
    Returns:
        DataFrame with index OHLC data
    """
    global TUSHARE_API
    
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print("[data_manager] ❌ Tushare initialization failed")
            return None
    
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime('%Y%m%d')
        
        # Choose API based on frequency
        if freq == 'weekly':
            # Use index_weekly API
            df = TUSHARE_API.index_weekly(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
            )
        else:  # daily
            # Use index_daily API
            df = TUSHARE_API.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
            )
        
        if df is None or df.empty:
            print(f"[data_manager] ❌ No {freq} data for index {index_code}")
            return None
        
        # Rename and format
        df = df.rename(columns={
            'trade_date': 'Date',
            'close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'pre_close': 'Pre_Close',
            'change': 'Change',
            'pct_chg': 'Pct_Change',
            'vol': 'Volume',
            'amount': 'Amount'
        })
        
        # Convert date and set index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"[data_manager] ✅ Fetched {len(df)} {freq} periods of index data for {index_code}")
        
        return df
        
    except Exception as e:
        print(f"[data_manager] ❌ Failed to fetch {freq} index {index_code}: {e}")
        return None


def get_stock_fundamentals_live(ticker, start_date, end_date):
    """
    Fetch daily fundamental metrics (PE, PB, Market Cap) from Tushare.
    
    Args:
        ticker: 6-digit stock code
        start_date: Start date (YYYYMMDD format string)
        end_date: End date (YYYYMMDD format string)
    
    Returns:
        DataFrame with PE, PB, Market Cap data indexed by date
    """
    global TUSHARE_API
    
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print("[data_manager] ❌ Tushare initialization failed")
            return None
    
    ts_code = get_tushare_ticker(ticker)
    
    try:
        df = TUSHARE_API.daily_basic(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,trade_date,close,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate,dv_ratio'
        )
        
        if df is None or df.empty:
            print(f"[data_manager] ❌ No fundamental data for {ticker}")
            return None
        
        # Rename and format columns
        df = df.rename(columns={
            'trade_date': 'Date',
            'pe': 'PE',
            'pe_ttm': 'PE_TTM',
            'pb': 'PB',
            'ps': 'PS',
            'ps_ttm': 'PS_TTM',
            'total_mv': 'Total_MV',
            'circ_mv': 'Circ_MV',
            'turnover_rate': 'Turnover_Rate',
            'dv_ratio': 'Dividend_Yield'
        })
        
        # Convert date and set index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert market cap from 万元 to 亿元 for readability
        df['Total_MV_Yi'] = df['Total_MV'] / 10000
        df['Circ_MV_Yi'] = df['Circ_MV'] / 10000
        
        print(f"[data_manager] ✅ Fetched {len(df)} days of fundamental data for {ticker}")
        
        return df
        
    except Exception as e:
        print(f"[data_manager] ❌ Failed to fetch fundamentals for {ticker}: {e}")
        return None


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


def get_single_stock_data_live(ticker, lookback_years=3, start_date=None, end_date =None):
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
        
    # Determine date range
    if start_date is None:
        # Use lookback_years
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=365 * lookback_years)
        start_str = start_dt.strftime('%Y%m%d')
    else:
        # Use provided start_date
        start_str = start_date
    
    if end_date is None:
        end_str = datetime.today().strftime('%Y%m%d')
    else:
        end_str = end_date

    
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


def aggregate_ppi_data(sector_start_dates=None):
    """
    Aggregates individual stock data into sector-level PPIs using MARKET CAP WEIGHTING.
    
    Uses RETURN-BASED calculation to handle sector composition changes correctly.
    """
    print("=" * 60)
    print("📊 Aggregating Stock Data into Sector PPIs (Market Cap Weighted)")
    print("=" * 60)
    
    all_sector_ppi_data = {}
    
    end_date = datetime.today()
    end_str = end_date.strftime('%Y%m%d')
    
    if sector_start_dates is None:
        sector_start_dates = {sector: DATA_START_DATE for sector in SECTOR_STOCK_MAP.keys()}
    
    for sector, stock_list in SECTOR_STOCK_MAP.items():
        if sector not in sector_start_dates:
            continue
        
        # ✅ FLEXIBLE DATE HANDLING - Accept any format
        start_date_input = sector_start_dates[sector]
        if start_date_input is None:
            # Full aggregation
            start_date_str = DATA_START_DATE
        elif isinstance(start_date_input, str):
            # Convert to YYYYMMDD format regardless of input format
            if '-' in start_date_input:
                # YYYY-MM-DD format
                start_date_str = start_date_input.replace('-', '')
            else:
                # Already YYYYMMDD format
                start_date_str = start_date_input
        else:
            # datetime or Timestamp object
            try:
                start_date_str = pd.to_datetime(start_date_input).strftime('%Y%m%d')
            except:
                print(f"   ⚠️ {sector}: Invalid start_date format, using DATA_START_DATE")
                start_date_str = DATA_START_DATE

        print(f"📊 {sector}: Aggregation from {start_date_str}")
        
        # ✅ STEP 1: Fetch OHLC data and market cap for all stocks in sector
        stock_data = {}
        market_caps = {}
        
        print(f"   → Fetching {len(stock_list)} stocks...")
        
        for ticker in stock_list:
            df_price = get_single_stock_data_live(
                ticker, 
                start_date=start_date_str,
                end_date=end_str
            )
            
            if df_price is None or df_price.empty:
                print(f"   ⚠️ {ticker}: No price data")
                continue
            
            df_fundamentals = get_stock_fundamentals_live(
                ticker,
                start_date=start_date_str,
                end_date=end_str
            )
            
            if df_fundamentals is None or df_fundamentals.empty:
                print(f"   ⚠️ {ticker}: No fundamental data")
                continue
            
            # ✅ Extract market cap (snake_case)
            if 'Total_MV' in df_fundamentals.columns:
                market_cap_col = 'Total_MV'
            elif 'Total_MV_Yi' in df_fundamentals.columns:
                market_cap_col = 'Total_MV_Yi'
            else:
                print(f"   ⚠️ {ticker}: No market cap column found. Available: {df_fundamentals.columns.tolist()}")
                continue
            
            stock_data[ticker] = df_price[['Open', 'High', 'Low', 'Close', 'Volume']]
            market_caps[ticker] = df_fundamentals[market_cap_col]
            
            time.sleep(0.35)
        
        if len(stock_data) < 2:
            print(f"   ❌ {sector}: Insufficient data ({len(stock_data)} stocks)")
            continue
        
        print(f"   ✅ Loaded {len(stock_data)} stocks with market cap data")
        
        # ✅ STEP 2: Get union of all dates
        all_dates = set()
        for ticker in stock_data.keys():
            ticker_dates = stock_data[ticker].index.intersection(market_caps[ticker].index)
            all_dates.update(ticker_dates)
        
        all_dates = pd.DatetimeIndex(sorted(all_dates))
        print(f"   → Total unique dates: {len(all_dates)}")
        
        # ✅ STEP 3: Calculate market-cap-weighted RETURNS for each date
        daily_returns = []
        valid_dates = []
        daily_volumes = []
        
        for i, date in enumerate(all_dates):
            if i == 0:
                # First day - no return to calculate yet
                continue
            
            prev_date = all_dates[i - 1]
            
            total_cap_today = 0
            weighted_return = 0
            weighted_volume = 0
            valid_stocks = 0
            
            # Calculate weighted return for this date
            for ticker in stock_data.keys():
                try:
                    # Check if stock has data for both dates
                    if date not in stock_data[ticker].index or prev_date not in stock_data[ticker].index:
                        continue
                    if date not in market_caps[ticker].index or prev_date not in market_caps[ticker].index:
                        continue
                    
                    # Use today's market cap for weighting
                    cap_today = market_caps[ticker].loc[date]
                    
                    if pd.isna(cap_today) or cap_today <= 0:
                        continue
                    
                    # Calculate return
                    close_prev = stock_data[ticker].loc[prev_date, 'Close']
                    close_today = stock_data[ticker].loc[date, 'Close']
                    
                    if pd.isna(close_prev) or pd.isna(close_today) or close_prev <= 0:
                        continue
                    
                    stock_return = (close_today - close_prev) / close_prev
                    volume_today = stock_data[ticker].loc[date, 'Volume']
                    
                    # Accumulate weighted values
                    total_cap_today += cap_today
                    weighted_return += stock_return * cap_today
                    
                    if pd.notna(volume_today):
                        weighted_volume += volume_today * cap_today
                    
                    valid_stocks += 1
                    
                except (KeyError, IndexError, ZeroDivisionError):
                    continue
            
            # Only add date if we have at least 2 valid stocks
            if valid_stocks >= 2 and total_cap_today > 0:
                daily_returns.append(weighted_return / total_cap_today)
                daily_volumes.append(weighted_volume / total_cap_today)
                valid_dates.append(date)
        
        if len(daily_returns) < 20:
            print(f"   ❌ {sector}: Insufficient valid dates ({len(daily_returns)})")
            continue
        
        # ✅ STEP 4: Build PPI by chaining returns (starting at 100)
        ppi_values = [100.0]  # Base value
        
        for ret in daily_returns:
            ppi_values.append(ppi_values[-1] * (1 + ret))
        
        # Create DataFrame
        ppi_df = pd.DataFrame({
            'Close': ppi_values[1:],  # Skip first 100 base value
            'Volume': daily_volumes
        }, index=valid_dates)
        
        # ✅ Generate OHLC from Close
        ppi_df['Open'] = ppi_df['Close'].shift(1)  # Yesterday's close = today's open
        ppi_df['High'] = ppi_df['Close']  # Conservative
        ppi_df['Low'] = ppi_df['Close']   # Conservative
        
        # Drop first row (no previous close for Open)
        ppi_df = ppi_df.dropna(subset=['Open'])

        # Drop the remaining with NA values (if any)
        ppi_df = ppi_df.dropna()
        
        # ✅ Calculate volume z-score (snake_case column name)
        ppi_df['Vol_Mean'] = ppi_df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
        ppi_df['Vol_Std'] = ppi_df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
        ppi_df['Norm_Vol_Metric'] = (ppi_df['Volume'] - ppi_df['Vol_Mean']) / ppi_df['Vol_Std']
        
        # Clean up - use snake_case column name
        ppi_df = ppi_df[['Open', 'High', 'Low', 'Close', 'Norm_Vol_Metric']]
        ppi_df = ppi_df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(ppi_df) < MIN_HISTORY_DAYS:
            print(f"   ❌ {sector}: Insufficient history ({len(ppi_df)} days)")
            continue
        
        all_sector_ppi_data[sector] = ppi_df
        print(f"   ✅ {sector}: Aggregated {len(ppi_df)} dates (return-based PPI)")
        print(f"      Date range: {ppi_df.index.min().strftime('%Y-%m-%d')} to {ppi_df.index.max().strftime('%Y-%m-%d')}")
        print(f"      PPI range: {ppi_df['Close'].min():.2f} to {ppi_df['Close'].max():.2f}")
    
    print("=" * 60)
    print(f"✅ PPI Aggregation Complete: {len(all_sector_ppi_data)} sectors")
    print("=" * 60)
    
    return all_sector_ppi_data

def save_ppi_data_to_db(all_ppi_data):
    """
    Saves the aggregated PPI DataFrames into tables in the DB.
    Only inserts NEW dates that don't already exist in the database.

    ✅ IMPROVED: Checks if table exists in Supabase before attempting insert
    """
    print(f"--- Saving {len(all_ppi_data)} PPIs to database (incremental) ---")

    # ✅ NEW: Track missing tables for Supabase
    missing_tables = []

    for sector_name, ppi_df in all_ppi_data.items():
        tablename = f'PPI_{sector_name}'

        # ✅ NEW: Check if table exists (especially important for Supabase)
        if not db.table_exists(tablename):
            if db_config.USE_SQLITE:
                # SQLite: Create table automatically
                print(f"   📝 {sector_name}: Creating new table {tablename}")
                create_ppi_table(sector_name)
            else:
                # Supabase: Cannot create tables on the fly
                print(f"   ❌ {sector_name}: Table {tablename} does not exist in Supabase!")
                missing_tables.append((sector_name, tablename))
                continue

        if ppi_df is None or ppi_df.empty:
            print(f"   ⏭️ Skipping {sector_name} - no data")
            continue

        try:
            # STEP 1: De-duplicate dates in new data
            ppi_df = ppi_df[~ppi_df.index.duplicated(keep='last')]

            # STEP 2: Get existing dates from database
            try:
                existing_df = db.read_table(tablename, columns='Date')
                if not existing_df.empty:
                    existing_dates = pd.to_datetime(existing_df['Date']).dt.strftime('%Y-%m-%d').tolist()
                    existing_dates_set = set(existing_dates)
                else:
                    existing_dates_set = set()
            except Exception as e:
                print(f"   ℹ️ {tablename} doesn't exist yet or is empty, will create with all data")
                existing_dates_set = set()

            # STEP 3: Filter out dates that already exist
            df_to_insert = ppi_df.copy()
            df_to_insert.index = pd.to_datetime(df_to_insert.index)
            df_to_insert = df_to_insert[~df_to_insert.index.duplicated(keep='last')]

            # STEP 4: Prepare data for insertion
            df_new_only = df_to_insert[['Open', 'High', 'Low', 'Close', 'Norm_Vol_Metric']]
            df_new_only.index.name = 'Date'
            df_new_only.reset_index(inplace=True)

            new_dates_mask = ~df_new_only['Date'].dt.strftime('%Y-%m-%d').isin(existing_dates_set)
            df_new_only = df_new_only[new_dates_mask].copy()
            df_new_only = df_new_only.replace([np.inf, -np.inf], np.nan)
            df_new_only = df_new_only.dropna()  # Drop rows with NaN

            if df_new_only.empty:
                print(f"   ⏭️ {sector_name}: Already up-to-date (no new dates to add)")
                continue

            # STEP 5: Final de-duplication
            df_new_only = df_new_only.drop_duplicates(subset='Date', keep='last')

            # STEP 6: Insert only new records
            records = df_new_only.to_dict('records')
            db.insert_records(tablename, records, upsert=True)

            print(f"   ✅ {sector_name}: Upserted {len(df_new_only)} records")

        except Exception as e2:
            print(f"   ❌ {sector_name}: Failed to save PPI data: {e2}")
            continue

    # ✅ NEW: Show helpful message for missing Supabase tables
    if missing_tables and not db_config.USE_SQLITE:
        print()
        print("=" * 70)
        print("⚠️ WARNING: Missing Supabase Tables")
        print("=" * 70)
        print("The following sectors cannot be saved because their tables")
        print("do not exist in Supabase. You need to create them manually.")
        print()
        print("Run this SQL in your Supabase SQL Editor:")
        print("-" * 70)

        for sector, tablename in missing_tables:
            sql = f"""
            -- Table for sector: {sector}
            CREATE TABLE IF NOT EXISTS "{tablename}" (
                "Date" TEXT PRIMARY KEY,
                "Open" REAL,
                "High" REAL,
                "Low" REAL,
                "Close" REAL,
                "Norm_Vol_Metric" REAL
            );
            """
            print(sql)

        print("-" * 70)
        print(f"After creating these {len(missing_tables)} table(s), run main.py again.")
        print("=" * 70)

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
                # ✅ Fix statsmodels warning: Set frequency
                try:
                    df.index.freq = pd.infer_freq(df.index)
                except:
                    pass

                if df.index.freq is None:
                    df = df.asfreq('B')  # Business day frequency
                    df = df.dropna(subset=['Close'])  # Remove any NaN rows

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
    

# ==================== WATCHLIST MANAGEMENT ====================

def create_watchlist_table():
    """Creates watchlist table."""
    if db_config.USE_SQLITE:
        # SQLite - create table directly
        schema = """CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            stock_name TEXT,
            added_date TEXT NOT NULL
        )"""
        return db.create_table_sqlite(schema)
    else:
        # Supabase - table must be created in dashboard first
        # Check if table exists
        if not db.table_exists('watchlist'):
            print("⚠️ Watchlist table must be created in Supabase dashboard")
            print("Run this SQL in Supabase SQL Editor:")
            print("""
            CREATE TABLE watchlist (
                id BIGSERIAL PRIMARY KEY,
                ticker TEXT NOT NULL UNIQUE,
                stock_name TEXT,
                added_date TEXT NOT NULL
            );
            """)
            return False
        return True


def add_to_watchlist(ticker, stock_name=None):
    """Add a stock to the watchlist. Validates ticker exists in Tushare first."""
    try:
        # Check if already exists in watchlist
        existing = db.read_table('watchlist', filters={'ticker': ticker})
        if not existing.empty:
            existing_name = existing.iloc[0]['stock_name']
            return False, f"⚠️ {ticker} ({existing_name}) already in watchlist"
        
        # Get stock name if not provided - this also validates ticker exists
        if not stock_name:
            # First try from stock_basic table
            stock_name = get_stock_name_from_db(ticker)
            
            # If not in stock_basic, try fetching from API (validates existence)
            if not stock_name:
                stock_name = get_company_name_from_api(ticker)
            
            # If still no name, ticker doesn't exist
            if not stock_name:
                return False, f"❌ Stock {ticker} not found. Please verify the ticker code."
        
        # Validate ticker actually has data by trying to fetch 1 day
        try:
            test_df = get_single_stock_data_live(ticker, lookback_years=0.01)  # ~3 days
            if test_df is None or test_df.empty:
                return False, f"❌ Stock {ticker} has no trading data. Please verify the ticker code."
        except Exception as e:
            return False, f"❌ Failed to validate {ticker}: Ticker may not exist or is delisted"
        
        # Get current date in Beijing time
        from zoneinfo import ZoneInfo
        CHINA_TZ = ZoneInfo("Asia/Shanghai")
        added_date = datetime.now(CHINA_TZ).strftime('%Y-%m-%d')
        
        # Prepare record
        record = {
            'ticker': ticker,
            'stock_name': stock_name,
            'added_date': added_date
        }
        
        # Insert new record
        db.insert_records('watchlist', [record], upsert=False)
        return True, f"✅ Added {ticker} ({stock_name}) to watchlist"
    
    except Exception as e:
        error_msg = str(e)
        if 'UNIQUE constraint failed' in error_msg or 'duplicate key' in error_msg.lower():
            return False, f"⚠️ {ticker} already in watchlist"
        return False, f"❌ Error: {error_msg}"




def remove_from_watchlist(ticker):
    """Remove a stock from the watchlist."""
    try:
        # Check if exists first
        existing = db.read_table('watchlist', filters={'ticker': ticker})
        if existing.empty:
            return False, f"⚠️ {ticker} not found in watchlist"
        
        # Delete
        db.delete_records('watchlist', {'ticker': ticker})
        return True, f"✅ Removed {ticker} from watchlist"
    
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def get_watchlist():
    """Get all stocks in the watchlist."""
    try:
        df = db.read_table('watchlist', 
                          columns='ticker,stock_name,added_date',
                          order_by='-added_date')
        
        if df.empty:
            return []
        
        return df.to_dict('records')
    
    except Exception as e:
        print(f"Error fetching watchlist: {e}")
        return []


def get_watchlist_tickers():
    """Get just the ticker symbols from watchlist (for scanning)."""
    try:
        df = db.read_table('watchlist', 
                          columns='ticker',
                          order_by='ticker')
        
        if df.empty:
            return []
        
        return df['ticker'].tolist()
    
    except Exception as e:
        print(f"Error fetching watchlist tickers: {e}")
        return []


def update_watchlist_notes(ticker, notes):
    """Update notes for a stock in the watchlist."""
    try:
        # Check if exists
        existing = db.read_table('watchlist', filters={'ticker': ticker})
        if existing.empty:
            return False, f"⚠️ {ticker} not found in watchlist"
        
        # For SQLite, we need to use raw SQL for UPDATE
        if db_config.USE_SQLITE:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE watchlist SET notes = ? WHERE ticker = ?', (notes, ticker))
            conn.commit()
            conn.close()
        else:
            # For Supabase, use update
            from db_manager import supabase_client
            supabase_client.table('watchlist').update({'notes': notes}).eq('ticker', ticker).execute()
        
        return True, f"✅ Updated notes for {ticker}"
    
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def is_in_watchlist(ticker):
    """Check if a ticker is in the watchlist."""
    try:
        df = db.read_table('watchlist', filters={'ticker': ticker}, limit=1)
        return not df.empty
    except:
        return False


def bulk_add_to_watchlist(tickers_list):
    """
    Add multiple stocks to watchlist at once.
    Retrieves stock names from stock_basic table.
    
    Args:
        tickers_list: List of ticker strings (6-digit codes)
        notes_prefix: Prefix for notes field
        
    Returns:
        (success_count, failed_count, messages)
    """
    success_count = 0
    failed_count = 0
    messages = []
    
    for ticker in tickers_list:
        # Get stock name from stock_basic table
        stock_name = get_stock_name_from_db(ticker)
        
        if not stock_name:
            # If not found in stock_basic, try to fetch from API
            stock_name = get_company_name_from_api(ticker)
            if not stock_name:
                stock_name = ticker  # Fallback to ticker
        
        # Add to watchlist with retrieved name
        success, msg = add_to_watchlist(
            ticker, 
            stock_name=stock_name
        )
        
        if success:
            success_count += 1
            print(f"✅ {ticker} ({stock_name})")
        else:
            failed_count += 1
            print(f"❌ {ticker}: {msg}")
            messages.append(msg)
    
    return success_count, failed_count, messages


