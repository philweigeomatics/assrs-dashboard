import os
import pandas as pd
import tushare as ts
import time
from datetime import datetime, timedelta
import sys
import numpy as np
from db_manager import db
import db_config
import api_config
import auth_manager

TUSHARE_API = None
TUSHARE_API_TOKEN = api_config.TUSHARE_TOKEN
DATA_START_DATE = '20240101'
WATCHLIST_MAX_STOCKS = 25

SECTOR_STOCK_MAP = {
    '银行': ['601398','601939','601288','600036','601998','000001'], # 工商银行, 建设银行, 农业银行, 招商银行, 中信银行, 平安银行     
    '非银金融':['600030','601318','601628','000750','000776','002736','002670'], # 中信证券, 中国平安, 中国人寿, 国海证券, 广发证券, 国信证券
    '半导体设计':['301269','688206','301095','688521','688256','688041','688047','603986','300223','688123','600460','300661','688052','300782','688153','300327','300077'],
    '半导体制造':['688981','688347','688469','688249','600584','002156','002185','603005','688362'],
    '半导体设备':['688012','002371','688072','688082','688120','688361','688037','600641','688200','300604','300480'],
    '半导体材料':['688126','605358','003026','688401','688138','300666','600206','688268','688106','300346','688019','300054','300655','603650','603931','300395','603688','688234'],
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
    '机器人': ['300124','601689','688777','002008','002472','688017','002747','300607'], # 汇川技术, 拓普集团, 中控技术, 大族激光，双环传动，绿的谐波，埃斯顿，拓斯达
    '专业设备制造':['600528','000157','600031','600980','601608','000425','601717'] # 中铁工业，中联重科，三一重工，北矿科技，中信重工，徐工机械，中创智领
}

# '半导体': ['688981','688041','688256','002371','688347','001309','002049','603986'], # 中芯国际, 海光信息, 寒武纪，北方华创，华虹公司，德明利，紫光国微，兆易创新


ALL_STOCK_TICKERS = sorted(list(set(ticker for stocks in SECTOR_STOCK_MAP.values() for ticker in stocks)))
REQUIRED_COLUMNS = ['Open', 'Close', 'High', 'Low', 'Volume']
MIN_HISTORY_DAYS = 100
VOL_ZSCORE_LOOKBACK = 100

def init_tushare(token=None):
    """Initializes the Tushare API."""
    global TUSHARE_API
    if token is None:
        token = TUSHARE_API_TOKEN

    if TUSHARE_API_TOKEN is None:
        token = api_config.TUSHARE_TOKEN
    
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
    user_id   = auth_manager.get_current_user_id()
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
            'user_id':      user_id,
            'timestamp': timestamp,
            'company_name': company_name
        }], upsert=True)
        
        # Trim to last 10 for this user only
        all_history = db.read_table(
            'search_history',
            filters={'user_id': user_id},
            columns='ticker, timestamp',
            order_by='-timestamp'
        )
        
        if len(all_history) > 10:
            to_delete = all_history.iloc[10:]['ticker'].tolist()
            for t in to_delete:
                db.delete_records('search_history', {'ticker': t, 'user_id': user_id})
        
        print(f"[data_manager] ✅ History saved: {ticker} - {company_name}")
        return company_name
    except Exception as e:
        print(f"[data_manager] ❌ Error updating search history: {e}")
        return ticker

def get_search_history(limit=10):
    """Returns last 10 searches for current user, latest first."""
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return []
    try:
        create_history_table()
        df = db.read_table(
            'search_history',
            filters={'user_id': user_id},
            columns='ticker, company_name, timestamp',
            order_by='-timestamp',
            limit=limit
        )
        if df.empty:
            return []
        results = []
        for _, row in df.iterrows():
            ticker       = row['ticker']
            company_name = row.get('company_name', ticker)
            results.append({
                'ticker':  ticker,
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


def get_index_data_live(index_code='000300.SH', lookback_days=180, freq='daily', start_date=None, end_date=None):
    """
    Fetch index data from Tushare (e.g., CSI 300).
    
    Args:
        index_code: Index code in Tushare format (e.g., '000300.SH' for CSI 300)
        lookback_days: Number of days of historical data (used if start_date is None)
        freq: 'daily' or 'weekly' - determines which API to call
        start_date: Optional start date in YYYY-MM-DD or YYYYMMDD format
        end_date: Optional end date in YYYY-MM-DD or YYYYMMDD format
    
    Returns:
        DataFrame with index OHLC data
    """
    global TUSHARE_API
    
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print("[data_manager] ❌ Tushare initialization failed")
            return None
    
    import time
    from datetime import datetime, timedelta
    import pandas as pd
    
    # 1. Date formatting logic (Merges explicit dates with legacy lookback_days)
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    else:
        end_date = end_date.replace('-', '')
        
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime('%Y%m%d')
    else:
        start_date = start_date.replace('-', '')
        
    # 2. Fetch with Retry Wrapper
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.35)  # Rate limit safety
            
            # Choose API based on frequency
            if freq == 'weekly':
                df = TUSHARE_API.index_weekly(
                    ts_code=index_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
                )
            else:  # daily
                df = TUSHARE_API.index_daily(
                    ts_code=index_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
                )
            
            if df is None or df.empty:
                print(f"[data_manager] ⚠️ Attempt {attempt+1}: No {freq} data for index {index_code}")
                continue
            
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
            
        except OSError:
            print(f"🔄 Tushare OSError on {index_code} (Attempt {attempt+1}). Retrying...")
            time.sleep(1.5)
        except Exception as e:
            print(f"[data_manager] ❌ Failed to fetch {freq} index {index_code}: {e}")
            break
            
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

# ─────────────────────────────────────────────────────────────────────────────
# WAVE TRADER — helper functions
# Uses existing get_single_stock_data_live() (pro_bar qfq) — 0 extra Tushare points
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# WAVE TRADER — helper functions
# Uses existing get_single_stock_data_live() (pro_bar qfq) — 0 extra Tushare points
# ─────────────────────────────────────────────────────────────────────────────

def get_stock_name_wave(ticker: str) -> str:
    """
    Return Chinese company name for the Wave Trader UI.
    Falls back to the ticker code if lookup fails.
    """
    try:
        name = get_stock_name_from_db(ticker)
        return name if name else ticker
    except Exception:
        return ticker


def get_ohlcv_for_wave(ticker: str, granularity: str = "Weekly", start_date: str = None) -> "pd.DataFrame | None":
    """
    Unified OHLCV fetcher for the Wave Trader page.
    Uses pro_bar directly with freq='D'/'W'/'M' and adj='qfq' — so Weekly and
    Monthly bars are native Tushare-computed (not resampled from daily), and all
    three granularities are forward-adjusted for dividends and splits.

    pro_bar is the same endpoint used elsewhere in data_manager — 0 extra
    Tushare points consumed beyond what the app already uses.

    Args:
        ticker      : 6-digit A-share code, e.g. '002080'
        granularity : "Daily" | "Weekly" | "Monthly"
        start_date  : "YYYYMMDD" string. If None, defaults to 2 years back.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume],
        indexed by period-end dates, sorted oldest-first.
        Returns None on failure.
    """
    import pandas as pd
    from datetime import datetime, timedelta

    global TUSHARE_API
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            return None

    if start_date is None:
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y%m%d")

    from zoneinfo import ZoneInfo
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
    now_beijing = datetime.now(CHINA_TZ)
    today_beijing = now_beijing.date()

    end_date = today_beijing.strftime("%Y%m%d")

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    freq = freq_map.get(granularity, "W")
    ts_code = get_tushare_ticker(ticker)

    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            freq=freq,
            start_date=start_date,
            end_date=end_date,
            asset="E",
        )
    except Exception as e:
        print(f"[data_manager] ❌ get_ohlcv_for_wave({ticker}, {granularity}): {e}")
        return None

    if df is None or df.empty:
        return None

    df = df.rename(columns={
        "trade_date": "Date",
        "open": "Open", "high": "High",
        "low": "Low",  "close": "Close",
        "vol": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Drop current incomplete bar (Tushare may include a partial bar for today)
    today = pd.Timestamp.today().normalize()
    if granularity == "Daily":
        df = df[df.index < today]
    elif granularity == "Weekly":
        # Keep only weeks whose bar date (Friday) is before today's week-start
        week_start = today - pd.Timedelta(days=today.dayofweek)
        df = df[df.index < week_start]
    elif granularity == "Monthly":
        month_start = today.replace(day=1)
        df = df[df.index < month_start]

    return df if not df.empty else None


def aggregate_ppi_data(sector_start_dates=None, sector_stock_map=None):
    """
    Builds sector-level PPIs using market-cap weighted returns.

    FULL REBUILD  (sector_start_dates=None):
        Fetches from DATA_START_DATE. PPI starts at 100.
        Saves all dates.

    INCREMENTAL   (sector_start_dates={sector: Timestamp}):
        Fetches 150 calendar days of buffer before first_new_date.
        150 cal days ≈ 100 trading days = full rolling window for Norm_Vol_Metric.
        NOTE: Tushare rate-limits by number of calls, not data size,
              so a larger date range costs the same as a smaller one.
        Anchors PPI from the DB's last known Close so the series is
        perfectly continuous. Saves only dates >= first_new_date.
    """

    BUFFER_CALENDAR_DAYS = 150  # ≈100 trading days — fills rolling(100) window fully

    print("=" * 60)
    print("📊 Aggregating Sector PPIs (Market Cap Weighted, Return-Based)")
    print("=" * 60)

    all_sector_ppi_data = {}
    end_str = datetime.today().strftime('%Y%m%d')

    if sector_stock_map is None:
        sector_stock_map = get_sector_stock_map()

    if sector_start_dates is None:
        sector_start_dates = {sector: None for sector in sector_stock_map.keys()}

    for sector, stock_list in sector_stock_map.items():
        if sector not in sector_start_dates:
            continue

        # ── 1. Determine fetch range ──────────────────────────────────────────
        raw_start      = sector_start_dates[sector]
        is_incremental = raw_start is not None

        if is_incremental:
            first_new_date   = pd.to_datetime(raw_start)
            fetch_start_str  = (first_new_date - pd.Timedelta(days=BUFFER_CALENDAR_DAYS)).strftime('%Y%m%d')
        else:
            first_new_date   = pd.to_datetime(DATA_START_DATE)
            fetch_start_str  = DATA_START_DATE

        print(f"\n📊 {sector}: {'Incremental from ' + first_new_date.strftime('%Y-%m-%d') if is_incremental else 'Full rebuild from ' + DATA_START_DATE}")
        print(f"   Fetching stock data from {fetch_start_str} → {end_str}")

        # ── 2. Fetch price + market cap for every stock in sector ─────────────
        stock_data  = {}
        market_caps = {}

        for ticker in stock_list:
            df_price = get_single_stock_data_live(ticker, start_date=fetch_start_str, end_date=end_str)
            if df_price is None or df_price.empty:
                print(f"   ⚠️  {ticker}: no price data, skipping")
                continue

            df_fund = get_stock_fundamentals_live(ticker, start_date=fetch_start_str, end_date=end_str)
            if df_fund is None or df_fund.empty:
                print(f"   ⚠️  {ticker}: no fundamental data, skipping")
                continue

            cap_col = next((c for c in ['Total_MV', 'Total_MV_Yi'] if c in df_fund.columns), None)
            if cap_col is None:
                print(f"   ⚠️  {ticker}: no market cap column, skipping")
                continue

            stock_data[ticker]  = df_price[['Open', 'High', 'Low', 'Close', 'Volume']]
            market_caps[ticker] = df_fund[cap_col]
            time.sleep(0.35)

        if len(stock_data) < 2:
            print(f"   ❌ {sector}: only {len(stock_data)} stocks loaded, need ≥2 — skipping")
            continue

        print(f"   ✅ {len(stock_data)}/{len(stock_list)} stocks loaded")

        # ── 3. Union of all dates across stocks ───────────────────────────────
        all_dates = pd.DatetimeIndex(sorted(
            set().union(*[
                stock_data[t].index.intersection(market_caps[t].index)
                for t in stock_data
            ])
        ))

        if len(all_dates) < 2:
            print(f"   ❌ {sector}: not enough shared dates — skipping")
            continue

        # ── 4. Market-cap weighted returns + dollar volumes ───────────────────
        daily_returns = []
        daily_volumes = []
        valid_dates   = []

        for i in range(1, len(all_dates)):
            date, prev_date = all_dates[i], all_dates[i - 1]
            total_cap = weighted_return = combined_vol = 0
            valid_stocks = 0

            for ticker in stock_data:
                try:
                    sd, mc = stock_data[ticker], market_caps[ticker]
                    if date not in sd.index or prev_date not in sd.index or date not in mc.index:
                        continue

                    cap = mc.loc[date]
                    if pd.isna(cap) or cap <= 0:
                        continue

                    c_prev, c_today = sd.loc[prev_date, 'Close'], sd.loc[date, 'Close']
                    if pd.isna(c_prev) or pd.isna(c_today) or c_prev <= 0:
                        continue

                    high = sd.loc[date, 'High']
                    low  = sd.loc[date, 'Low']
                    vol  = sd.loc[date, 'Volume']
                    mid  = (high + low) / 2 if pd.notna(high) and pd.notna(low) else c_today

                    total_cap       += cap
                    weighted_return += ((c_today - c_prev) / c_prev) * cap
                    combined_vol    += mid * vol if pd.notna(vol) else 0
                    valid_stocks    += 1

                except (KeyError, IndexError, ZeroDivisionError):
                    continue

            if valid_stocks >= 2 and total_cap > 0:
                daily_returns.append(weighted_return / total_cap)
                daily_volumes.append(combined_vol)
                valid_dates.append(date)

        if not daily_returns:
            print(f"   ❌ {sector}: no valid return dates calculated — skipping")
            continue

        valid_dates = pd.DatetimeIndex(valid_dates)

        # ── 5. Anchor: find the correct PPI starting value ───────────────────
        #
        # Full rebuild  → anchor = 100  (no prior history)
        #
        # Incremental   → the DB already has correct PPI values inside our
        #                 buffer range. Find the latest buffer date that is
        #                 in both DB and valid_dates, then back-calculate
        #                 what starting anchor produces that exact DB Close
        #                 at that date when we chain all our returns forward.
        #                 This guarantees perfect series continuity.
        #
        anchor_value = 100.0

        if is_incremental:
            table_name = f'PPI_{sector}'
            try:
                db_df = db.read_table(table_name, columns='Date,Close', order_by='Date')
                if not db_df.empty:
                    db_df['Date'] = pd.to_datetime(db_df['Date'])

                    # Latest DB date that is also in our calculated valid_dates
                    common = db_df[db_df['Date'].isin(valid_dates)]

                    if not common.empty:
                        ref_date  = common.iloc[-1]['Date']
                        ref_close = common.iloc[-1]['Close']

                        # Back-calculate: anchor × ∏(1+r) for all returns up
                        # to and including ref_date must equal ref_close
                        ref_idx          = valid_dates.get_loc(ref_date)
                        forward_product  = float(np.prod([1.0 + r for r in daily_returns[:ref_idx + 1]]))
                        anchor_value     = ref_close / forward_product

                    else:
                        # Buffer reaches before all DB history — rare edge case.
                        # Use the last DB close before first_new_date as anchor.
                        pre_new = db_df[db_df['Date'] < first_new_date]
                        if not pre_new.empty:
                            anchor_value = pre_new.iloc[-1]['Close']

            except Exception as e:
                print(f"   ⚠️  Could not load DB anchor for {sector}: {e} — using 100.0")

        print(f"   📌 Anchor: {anchor_value:.4f}")

        # ── 6. Chain returns → build full PPI series ──────────────────────────
        ppi_values = [anchor_value]
        for ret in daily_returns:
            ppi_values.append(ppi_values[-1] * (1.0 + ret))

        # ppi_values[0]  = anchor (the virtual "day before" valid_dates[0])
        # ppi_values[1:] = Close for each date in valid_dates

        ppi_df = pd.DataFrame({
            'Close':  ppi_values[1:],
            'Volume': daily_volumes
        }, index=valid_dates)

        # Open = previous Close — anchor fills the NaN on the very first row
        ppi_df['Open'] = [anchor_value] + list(ppi_values[1:-1])
        ppi_df['High'] = ppi_df['Close']
        ppi_df['Low']  = ppi_df['Close']

        # ── 7. Norm_Vol_Metric over the FULL buffer window ────────────────────
        #    150 cal days of buffer guarantees the rolling window has
        #    ~100 trading days of data → fully accurate z-score on new rows
        ppi_df['Vol_Mean']       = ppi_df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
        ppi_df['Vol_Std']        = ppi_df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
        ppi_df['Norm_Vol_Metric'] = (ppi_df['Volume'] - ppi_df['Vol_Mean']) / ppi_df['Vol_Std']

        # ── 8. Trim to only new dates before saving ───────────────────────────
        ppi_df = ppi_df[ppi_df.index >= first_new_date].copy()

        # ── 9. Final column selection and cleanup ─────────────────────────────
        ppi_df = ppi_df[['Open', 'High', 'Low', 'Close', 'Norm_Vol_Metric']]
        ppi_df = ppi_df.replace([np.inf, -np.inf], np.nan)
        ppi_df = ppi_df.dropna()  # all 5 columns must be valid

        if ppi_df.empty:
            print(f"   ❌ {sector}: 0 valid rows after final cleanup — skipping")
            continue

        all_sector_ppi_data[sector] = ppi_df
        print(f"   ✅ {sector}: {len(ppi_df)} date(s) ready to save")
        print(f"      Range:     {ppi_df.index.min().strftime('%Y-%m-%d')} → {ppi_df.index.max().strftime('%Y-%m-%d')}")
        print(f"      PPI Close: {ppi_df['Close'].min():.4f} – {ppi_df['Close'].max():.4f}")

    print("\n" + "=" * 60)
    print(f"✅ PPI Aggregation complete: {len(all_sector_ppi_data)}/{len(sector_start_dates)} sectors")
    print("=" * 60)
    return all_sector_ppi_data


# def aggregate_ppi_data(sector_start_dates=None):
#     """
#     Aggregates individual stock data into sector-level PPIs using MARKET CAP WEIGHTING.
    
#     Uses RETURN-BASED calculation to handle sector composition changes correctly.
#     """
#     print("=" * 60)
#     print("📊 Aggregating Stock Data into Sector PPIs (Market Cap Weighted)")
#     print("=" * 60)
    
#     all_sector_ppi_data = {}
    
#     end_date = datetime.today()
#     end_str = end_date.strftime('%Y%m%d')
    
#     if sector_start_dates is None:
#         sector_start_dates = {sector: DATA_START_DATE for sector in SECTOR_STOCK_MAP.keys()}
    
#     for sector, stock_list in SECTOR_STOCK_MAP.items():
#         if sector not in sector_start_dates:
#             continue
        
#         # ✅ FLEXIBLE DATE HANDLING - Accept any format
#         start_date_input = sector_start_dates[sector]
#         if start_date_input is None:
#             # Full aggregation
#             start_date_str = DATA_START_DATE
#         elif isinstance(start_date_input, str):
#             # Convert to YYYYMMDD format regardless of input format
#             if '-' in start_date_input:
#                 # YYYY-MM-DD format
#                 start_date_str = start_date_input.replace('-', '')
#             else:
#                 # Already YYYYMMDD format
#                 start_date_str = start_date_input
#         else:
#             # datetime or Timestamp object
#             try:
#                 start_date_str = pd.to_datetime(start_date_input).strftime('%Y%m%d')
#             except:
#                 print(f"   ⚠️ {sector}: Invalid start_date format, using DATA_START_DATE")
#                 start_date_str = DATA_START_DATE

#         print(f"📊 {sector}: Aggregation from {start_date_str}")
        
#         # ✅ STEP 1: Fetch OHLC data and market cap for all stocks in sector
#         stock_data = {}
#         market_caps = {}
        
#         print(f"   → Fetching {len(stock_list)} stocks...")
        
#         for ticker in stock_list:

#             LOOKBACK_BUFFER_DAYS = 15
#             original_start = pd.to_datetime(start_date_str)
#             fetch_start_str = (original_start - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)).strftime('%Y%m%d')

#             df_price = get_single_stock_data_live(
#                 ticker, 
#                 start_date=fetch_start_str,
#                 end_date=end_str
#             )
            
#             if df_price is None or df_price.empty:
#                 print(f"   ⚠️ {ticker}: No price data")
#                 continue
            
#             df_fundamentals = get_stock_fundamentals_live(
#                 ticker,
#                 start_date=fetch_start_str,
#                 end_date=end_str
#             )
            
#             if df_fundamentals is None or df_fundamentals.empty:
#                 print(f"   ⚠️ {ticker}: No fundamental data")
#                 continue
            
#             # ✅ Extract market cap (snake_case)
#             if 'Total_MV' in df_fundamentals.columns:
#                 market_cap_col = 'Total_MV'
#             elif 'Total_MV_Yi' in df_fundamentals.columns:
#                 market_cap_col = 'Total_MV_Yi'
#             else:
#                 print(f"   ⚠️ {ticker}: No market cap column found. Available: {df_fundamentals.columns.tolist()}")
#                 continue
            
#             stock_data[ticker] = df_price[['Open', 'High', 'Low', 'Close', 'Volume']]
#             market_caps[ticker] = df_fundamentals[market_cap_col]
            
#             time.sleep(0.35)
        
#         if len(stock_data) < 2:
#             print(f"   ❌ {sector}: Insufficient data ({len(stock_data)} stocks)")
#             continue
        
#         print(f"   ✅ Loaded {len(stock_data)} stocks with market cap data")
        
#         # ✅ STEP 2: Get union of all dates
#         all_dates = set()
#         for ticker in stock_data.keys():
#             ticker_dates = stock_data[ticker].index.intersection(market_caps[ticker].index)
#             all_dates.update(ticker_dates)
        
#         all_dates = pd.DatetimeIndex(sorted(all_dates))
#         print(f"   → Total unique dates: {len(all_dates)}")
        
#         # ✅ STEP 3: Calculate market-cap-weighted RETURNS for each date
#         daily_returns = []
#         valid_dates = []
#         daily_volumes = []
        
#         for i, date in enumerate(all_dates):
#             if i == 0:
#                 # First day - no return to calculate yet
#                 continue
            
#             prev_date = all_dates[i - 1]

#             # Skip buffer dates — don't write them, but use as prev_date anchor
#             if date < original_start:
#                 continue
            
#             total_cap_today = 0
#             weighted_return = 0
#             combined_dollar_volume = 0
#             valid_stocks = 0
            
#             # Calculate weighted return for this date
#             for ticker in stock_data.keys():
#                 try:
#                     # Check if stock has data for both dates
#                     if date not in stock_data[ticker].index or prev_date not in stock_data[ticker].index:
#                         continue
#                     if date not in market_caps[ticker].index or prev_date not in market_caps[ticker].index:
#                         continue
                    
#                     # Use today's market cap for weighting
#                     cap_today = market_caps[ticker].loc[date]
                    
#                     if pd.isna(cap_today) or cap_today <= 0:
#                         continue
                    
#                     # Calculate return
#                     close_prev = stock_data[ticker].loc[prev_date, 'Close']
#                     close_today = stock_data[ticker].loc[date, 'Close']
                    
#                     if pd.isna(close_prev) or pd.isna(close_today) or close_prev <= 0:
#                         continue
                    
#                     stock_return = (close_today - close_prev) / close_prev

#                     # Calculate dollar volume (NEW LOGIC)
#                     high_today = stock_data[ticker].loc[date, 'High']
#                     low_today = stock_data[ticker].loc[date, 'Low']
#                     volume_today = stock_data[ticker].loc[date, 'Volume']

#                     mid_price = (high_today + low_today) / 2 if pd.notna(high_today) and pd.notna(low_today) else close_today   
#                     dollar_volume = mid_price * volume_today if pd.notna(mid_price) and pd.notna(volume_today) else 0
                    
#                     # Accumulate weighted values
#                     total_cap_today += cap_today
#                     weighted_return += stock_return * cap_today
#                     combined_dollar_volume += dollar_volume
                    
#                     valid_stocks += 1
                    
#                 except (KeyError, IndexError, ZeroDivisionError):
#                     continue
            
#             # Only add date if we have at least 2 valid stocks
#             if valid_stocks >= 2 and total_cap_today > 0:
#                 daily_returns.append(weighted_return / total_cap_today)
#                 daily_volumes.append(combined_dollar_volume)
#                 valid_dates.append(date)
        
#         # Replace with:
#         is_incremental = sector_start_dates.get(sector) is not None  # None = full rebuild
#         min_required = 20 if not is_incremental else 1

#         if len(daily_returns) < min_required:
#             print(f"   ❌ {sector}: Insufficient valid dates ({len(daily_returns)})")
#             continue
        
#         # ✅ STEP 4: Build PPI by chaining returns (starting at 100)
#         ppi_values = [100.0]  # Base value
        
#         for ret in daily_returns:
#             ppi_values.append(ppi_values[-1] * (1 + ret))
        
#         # Create DataFrame
#         ppi_df = pd.DataFrame({
#             'Close': ppi_values[1:],  # Skip first 100 base value
#             'Volume': daily_volumes
#         }, index=valid_dates)
        
#         # ✅ Generate OHLC from Close
#         ppi_df['Open'] = ppi_df['Close'].shift(1)  # Yesterday's close = today's open
#         ppi_df['High'] = ppi_df['Close']  # Conservative
#         ppi_df['Low'] = ppi_df['Close']   # Conservative
        
    
#         if is_incremental and ppi_df['Open'].isna().any():
#             # Get the last Close already stored in the DB for this sector
#             try:
#                 table_name = f'PPI_{sector}'
#                 existing = db.read_table(table_name, columns='Date,Close', order_by='-Date', limit=1)
#                 if not existing.empty:
#                     last_close_in_db = existing['Close'].iloc[0]
#                     ppi_df['Open'] = ppi_df['Open'].fillna(last_close_in_db)
#                 else:
#                     ppi_df['Open'] = ppi_df['Open'].fillna(ppi_df['Close'])  # fallback
#             except:
#                 ppi_df['Open'] = ppi_df['Open'].fillna(ppi_df['Close'])  # fallback
#         else:
#             ppi_df = ppi_df.dropna(subset=['Open'])


#         # Drop the remaining with NA values (if any)
#         ppi_df = ppi_df.dropna()
        
#         # ✅ Calculate volume z-score (snake_case column name)
#         ppi_df['Vol_Mean'] = ppi_df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
#         ppi_df['Vol_Std'] = ppi_df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
#         ppi_df['Norm_Vol_Metric'] = (ppi_df['Volume'] - ppi_df['Vol_Mean']) / ppi_df['Vol_Std']
        
#         # Clean up - use snake_case column name
#         ppi_df = ppi_df[['Open', 'High', 'Low', 'Close', 'Norm_Vol_Metric']]
#         ppi_df = ppi_df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
#         min_history = MIN_HISTORY_DAYS if not is_incremental else 1
#         if len(ppi_df) < min_history:
#             print(f" ❌ {sector}: Insufficient history ({len(ppi_df)} days)")
#             continue
        
#         all_sector_ppi_data[sector] = ppi_df
#         print(f"   ✅ {sector}: Aggregated {len(ppi_df)} dates (return-based PPI)")
#         print(f"      Date range: {ppi_df.index.min().strftime('%Y-%m-%d')} to {ppi_df.index.max().strftime('%Y-%m-%d')}")
#         print(f"      PPI range: {ppi_df['Close'].min():.2f} to {ppi_df['Close'].max():.2f}")
    
#     print("=" * 60)
#     print(f"✅ PPI Aggregation Complete: {len(all_sector_ppi_data)} sectors")
#     print("=" * 60)
    
#     return all_sector_ppi_data

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



def _get_all_ppi_sector_names():
    """Return list of sector names that have PPI_ tables in the DB."""
    if db.use_supabase:
        # Supabase: fall back to the dynamic sector map
        return list(get_sector_stock_map().keys())
    else:
        import sqlite3
        with sqlite3.connect(db.dbname) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PPI_%'"
            ).fetchall()
        return [r[0][len('PPI_'):] for r in rows]


def load_ppi_data_from_db():
    """Loads all 'PPI_' tables from the DB into a dictionary of DataFrames."""
    print("\n--- Loading Pre-Calculated PPIs from Database ---")

    all_ppi_data = {}

    for sector_name in _get_all_ppi_sector_names():
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
    
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return None
    
    try:
        check_df = db.read_table(
            'daily_signals',
            filters={'scan_date': scan_date, 'user_id': user_id},
            columns='*', limit=1
        )
        if check_df.empty:
            return None

        df = db.read_table(
            'daily_signals',
            filters={'scan_date': scan_date, 'user_id': user_id},
            columns='*',
            order_by='type, -signal_count'
        )

        # Drop id and created_at columns
        df = df.drop(columns=['id', 'scan_date', 'created_at', 'user_id'], errors='ignore')
        
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

    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return False
    
    if df is None or df.empty:
        return False
    
    try:
        # Delete only THIS user's snapshot for THIS date — other dates untouched
        db.delete_records('daily_signals',         {'scan_date': scan_date, 'user_id': user_id})
        db.delete_records('signals_scan_metadata', {'scan_date': scan_date, 'user_id': user_id})
        
        # Prepare data for insertion
        df_to_save = df.copy()
        df_to_save['scan_date'] = scan_date
        df_to_save['user_id']    = user_id
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
            'user_id': user_id,
            'total_stocks_scanned': total_stocks,
            'opportunities_found': opportunities,
            'alerts_found': alerts,
            'scan_duration_seconds': scan_duration,
            'created_at': datetime.now().isoformat()
        }
        
        db.insert_records('signals_scan_metadata', [metadata], upsert=True)
        print(f"[data_manager] ✅ Saved {len(df)} signals for user {user_id} on {scan_date}")
        return True
    except Exception as e:
        print(f"Error saving signals to cache: {e}")
        return False

def get_scan_metadata(scan_date):
    """Get metadata about the last scan."""
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return None
    
    try:
        df = db.read_table('signals_scan_metadata', filters={'scan_date': scan_date, 'user_id': user_id})
        
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
    """Add a stock to the current user's watchlist. Validates ticker exists."""
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return False, "❌ Not logged in"

    try:
        # Check if already exists for this user
        existing = db.read_table('watchlist', filters={'ticker': ticker, 'user_id': user_id})
        if not existing.empty:
            existing_name = existing.iloc[0]['stock_name']
            return False, f"⚠️ {ticker} ({existing_name}) already in watchlist"
        
        # ── Enforce limit for non-admin users ───────────────────────
        user = auth_manager.get_current_user()
        if user.get('role') != 'admin':
            current_count_df = db.read_table('watchlist', filters={'user_id': user_id}, columns='ticker')
            current_count = len(current_count_df)
            if current_count >= WATCHLIST_MAX_STOCKS:
                return False, f"⚠️ Watchlist limit reached ({WATCHLIST_MAX_STOCKS} stocks max). Remove some stocks first."

        # Get stock name — also validates ticker exists
        if not stock_name:
            stock_name = get_stock_name_from_db(ticker)
            if not stock_name:
                stock_name = get_company_name_from_api(ticker)
            if not stock_name:
                return False, f"❌ Stock {ticker} not found. Please verify the ticker code."

        # # Validate ticker has actual trading data
        # do not need this, as for long breaks, this will not work.
        # try:
        #     test_df = get_single_stock_data_live(ticker, lookback_years=0.01)  # ~3 days
        #     if test_df is None or test_df.empty:
        #         return False, f"❌ Stock {ticker} has no trading data. Please verify the ticker code."
        # except Exception:
        #     return False, f"❌ Failed to validate {ticker}: Ticker may not exist or is delisted"

        # Use Beijing time for added_date
        from zoneinfo import ZoneInfo
        CHINA_TZ = ZoneInfo("Asia/Shanghai")
        added_date = datetime.now(CHINA_TZ).strftime('%Y-%m-%d')

        db.insert_records('watchlist', [{
            'ticker':     ticker,
            'stock_name': stock_name,
            'user_id':    user_id,
            'added_date': added_date
        }], upsert=False)
        return True, f"✅ Added {ticker} ({stock_name}) to watchlist"

    except Exception as e:
        error_msg = str(e)
        if 'UNIQUE constraint failed' in error_msg or 'duplicate key' in error_msg.lower():
            return False, f"⚠️ {ticker} already in watchlist"
        return False, f"❌ Error: {error_msg}"



def remove_from_watchlist(ticker):
    """Remove a stock from the current user's watchlist."""
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return False, "❌ Not logged in"

    try:
        # Check exists for this user first
        existing = db.read_table('watchlist', filters={'ticker': ticker, 'user_id': user_id})
        if existing.empty:
            return False, f"⚠️ {ticker} not found in your watchlist"

        db.delete_records('watchlist', {'ticker': ticker, 'user_id': user_id})
        return True, f"✅ Removed {ticker} from watchlist"

    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# T-Trading Scan Persistence  (page: pages/t_trading_scanner.py)
# ──────────────────────────────────────────────────────────────────────────────
# A user's scan results are persisted so the page loads instantly from DB on
# next visit instead of forcing a slow Tushare re-fetch. Re-scan replaces all
# rows for that user; the page also surfaces the age of the saved scan so
# users know when the values are stale.

def ensure_t_trading_scan_table() -> None:
    """
    Idempotent migration: creates t_trading_scans in SQLite if missing.
    Supabase users must run the equivalent SQL once via the editor:

        CREATE TABLE IF NOT EXISTS t_trading_scans (
            user_id      BIGINT      NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
            ticker       TEXT        NOT NULL,
            name         TEXT,
            t_score      DOUBLE PRECISION,
            verdict      TEXT,
            range_pct    DOUBLE PRECISION,
            turnover_pct DOUBLE PRECISION,
            meanrev_bias DOUBLE PRECISION,
            adx          DOUBLE PRECISION,
            range_pos    DOUBLE PRECISION,
            limit_event  INTEGER     DEFAULT 0,
            fail_reason  TEXT        DEFAULT '',
            scanned_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, ticker)
        );
        CREATE INDEX IF NOT EXISTS idx_tts_user ON t_trading_scans(user_id);
    """
    from db_config import USE_SQLITE
    if not USE_SQLITE:
        return
    from db_config import DBNAME
    import sqlite3
    with sqlite3.connect(DBNAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS t_trading_scans (
                user_id      INTEGER NOT NULL,
                ticker       TEXT    NOT NULL,
                name         TEXT,
                t_score      REAL,
                verdict      TEXT,
                range_pct    REAL,
                turnover_pct REAL,
                meanrev_bias REAL,
                adx          REAL,
                range_pos    REAL,
                limit_event  INTEGER DEFAULT 0,
                fail_reason  TEXT    DEFAULT '',
                scanned_at   TEXT    NOT NULL,
                PRIMARY KEY (user_id, ticker)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tts_user ON t_trading_scans(user_id)")
        conn.commit()


def save_t_trading_scan(user_id: int, rows: list[dict]) -> None:
    """
    Replace all saved scan rows for `user_id` with `rows`. Each row dict is
    expected to use the same keys as the page's results DataFrame
    (Ticker / Name / T-Score / Verdict / Range % / Turnover % / MeanRev bias
    / ADX / Range pos / Limit event? / Why).
    """
    if not user_id:
        return
    clear_t_trading_scan(user_id)
    if not rows:
        return
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = []
    for r in rows:
        payload.append({
            "user_id":      int(user_id),
            "ticker":       r.get("Ticker"),
            "name":         r.get("Name"),
            "t_score":      r.get("T-Score"),
            "verdict":      r.get("Verdict"),
            "range_pct":    r.get("Range %"),
            "turnover_pct": r.get("Turnover %"),
            "meanrev_bias": r.get("MeanRev bias"),
            "adx":          r.get("ADX"),
            "range_pos":    r.get("Range pos"),
            "limit_event":  1 if "Yes" in str(r.get("Limit event?", "")) else 0,
            "fail_reason":  r.get("Why", "") or "",
            "scanned_at":   now_iso,
        })
    try:
        db.insert_records("t_trading_scans", payload)
    except Exception as e:
        print(f"[t_trading] save failed: {e}")


def load_t_trading_scan(user_id: int):
    """
    Returns (df, scanned_at_iso) or (None, None) if no saved scan exists.
    df uses the original page column names so it can render without remapping.
    """
    if not user_id:
        return None, None
    try:
        df = db.read_table("t_trading_scans", filters={"user_id": int(user_id)})
        if df is None or df.empty:
            return None, None
        latest = str(df["scanned_at"].max())
        # Rebuild the display column names so the page can render this directly.
        out = df.rename(columns={
            "ticker":       "Ticker",
            "name":         "Name",
            "t_score":      "T-Score",
            "verdict":      "Verdict",
            "range_pct":    "Range %",
            "turnover_pct": "Turnover %",
            "meanrev_bias": "MeanRev bias",
            "adx":          "ADX",
            "range_pos":    "Range pos",
            "fail_reason":  "Why",
        })
        out["Limit event?"] = out["limit_event"].map(lambda v: "⚠️ Yes" if v else "—")
        out = out.drop(columns=[c for c in ("user_id", "limit_event", "scanned_at") if c in out.columns])
        return out, latest
    except Exception as e:
        print(f"[t_trading] load failed: {e}")
        return None, None


def clear_t_trading_scan(user_id: int) -> None:
    """Wipe every saved scan row for `user_id`."""
    if not user_id:
        return
    try:
        db.delete_records("t_trading_scans", {"user_id": int(user_id)})
    except Exception as e:
        print(f"[t_trading] clear failed: {e}")


def get_watchlist():
    """Get all stocks in the current user's watchlist."""
    import auth_manager
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return []
    try:
        df = db.read_table('watchlist',
                            filters={'user_id': user_id},
                            columns='ticker,stock_name,added_date',
                            order_by='-added_date')
        return df.to_dict('records') if not df.empty else []
    except Exception as e:
        print(f"Error fetching watchlist: {e}")
        return []


def get_watchlist_tickers():
    """Get just the ticker symbols from current user's watchlist (for scanning)."""
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return []
    try:
        df = db.read_table('watchlist',
                            filters={'user_id': user_id},
                            columns='ticker',
                            order_by='ticker')
        return df['ticker'].tolist() if not df.empty else []
    except Exception as e:
        print(f"Error fetching watchlist tickers: {e}")
        return []


def update_watchlist_notes(ticker, notes):
    """Update notes for a stock in the current user's watchlist."""
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return False, "❌ Not logged in"

    try:
        existing = db.read_table('watchlist', filters={'ticker': ticker, 'user_id': user_id})
        if existing.empty:
            return False, f"⚠️ {ticker} not found in your watchlist"

        if db_config.USE_SQLITE:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE watchlist SET notes = ? WHERE ticker = ? AND user_id = ?',
                (notes, ticker, user_id)
            )
            conn.commit()
            conn.close()
        else:
            from db_manager import supabase_client
            supabase_client.table('watchlist').update({'notes': notes})\
                .eq('ticker', ticker).eq('user_id', user_id).execute()

        return True, f"✅ Updated notes for {ticker}"

    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def is_in_watchlist(ticker):
    """Check if a ticker is in the current user's watchlist."""
    user_id = auth_manager.get_current_user_id()
    if user_id is None:
        return False
    try:
        df = db.read_table('watchlist',
                            filters={'ticker': ticker, 'user_id': user_id},
                            limit=1)
        return not df.empty
    except Exception:
        return False


def bulk_add_to_watchlist(tickers_list):
    """
    Add multiple stocks to current user's watchlist at once.
    Returns (success_count, failed_count, messages)
    """
    user_id = auth_manager.get_current_user_id()
    user    = auth_manager.get_current_user()

    success_count = 0
    failed_count  = 0
    messages      = []

    # Pre-check remaining slots for non-admin
    if user and user.get('role') != 'admin':
        current_count_df = db.read_table('watchlist', filters={'user_id': user_id}, columns='ticker')
        current_count    = len(current_count_df)
        remaining_slots  = WATCHLIST_MAX_STOCKS - current_count
        if remaining_slots <= 0:
            return 0, len(tickers_list), [f"⚠️ Watchlist full ({WATCHLIST_MAX_STOCKS} stocks max)"]
        if len(tickers_list) > remaining_slots:
            messages.append(f"⚠️ Only {remaining_slots} slot(s) left — importing first {remaining_slots} tickers")
            tickers_list = tickers_list[:remaining_slots]

    for ticker in tickers_list:
        stock_name = get_stock_name_from_db(ticker)
        if not stock_name:
            stock_name = get_company_name_from_api(ticker)
        if not stock_name:
            stock_name = ticker

        success, msg = add_to_watchlist(ticker, stock_name=stock_name)
        if success:
            success_count += 1
            print(f"✅ {ticker} ({stock_name})")
        else:
            failed_count += 1
            print(f"❌ {ticker}: {msg}")
            messages.append(msg)

    return success_count, failed_count, messages

# ==================== FINANCIAL INDICATORS ====================
def get_financial_indicators(ticker, quarters=8):
    """
    Fetch financial indicators for the specified number of quarters and calculate YoY growth.
    
    Args:
        ticker: 6-digit stock code
        quarters: Number of quarters to fetch (default 8 = 2 years, minimum 5 for YoY calculation)
    
    Returns:
        Dictionary with latest financial metrics and YoY growth rates, or None if failed
    """
    global TUSHARE_API
    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print(f"[data_manager] ❌ Tushare initialization failed")
            return None
    
    ts_code = get_tushare_ticker(ticker)
    
    try:
        from datetime import datetime, timedelta
        
        # Convert quarters to days (90 days per quarter + 1 extra quarter buffer)
        # Buffer ensures we get enough data even if fetching right before a new quarterly report
        days_back = (quarters + 1) * 90
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        
        df = TUSHARE_API.fina_indicator(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields='ts_code,end_date,roe,roa,q_netprofit_margin,basic_eps_yoy,fcff,fcfe'
        )
        
        if df is None or df.empty:
            print(f"[data_manager] ⚠️ No financial data for {ticker}")
            return None
        
        # Sort by date descending (most recent first)
        df = df.sort_values('end_date', ascending=False)
        
        if len(df) < 1:
            print(f"[data_manager] ⚠️ Insufficient data for {ticker}")
            return None
        
        # Get latest quarter (most recent)
        latest = df.iloc[0]
        
        # Calculate FCFF/FCFE YoY growth
        fcff_growth = None
        fcfe_growth = None
        
        if len(df) >= 5:
            # Use 5th quarter (4 quarters back = 1 year ago, same quarter)
            yoy_quarter = df.iloc[4]
            
            current_fcff = latest.get('fcff')
            current_fcfe = latest.get('fcfe')
            yoy_fcff = yoy_quarter.get('fcff')
            yoy_fcfe = yoy_quarter.get('fcfe')
            
            # Calculate growth (handle negative values and zeros)
            if current_fcff is not None and yoy_fcff is not None and yoy_fcff != 0:
                fcff_growth = ((current_fcff - yoy_fcff) / abs(yoy_fcff)) * 100
            
            if current_fcfe is not None and yoy_fcfe is not None and yoy_fcfe != 0:
                fcfe_growth = ((current_fcfe - yoy_fcfe) / abs(yoy_fcfe)) * 100
        else:
            print(f"[data_manager] ⚠️ Not enough quarters ({len(df)}) for YoY growth calculation for {ticker}")
        
        return {
            'ROE': latest.get('roe'),
            'ROA': latest.get('roa'),
            'Operating_Margin': latest.get('q_netprofit_margin'),
            'EPS_Growth_YoY': latest.get('basic_eps_yoy'),  # Pre-calculated by Tushare
            'FCFF_PS': latest.get('fcff_ps'),
            'FCFE_PS': latest.get('fcfe_ps'),
            'FCFF_Growth_YoY': fcff_growth,  # Manually calculated
            'FCFE_Growth_YoY': fcfe_growth,  # Manually calculated
            'Report_Date': latest.get('end_date'),
            'Quarters_Available': len(df)  # For debugging
        }
        
    except Exception as e:
        print(f"[data_manager] ❌ Failed to fetch financial indicators for {ticker}: {e}")
        return None




# ═══════════════════════════════════════════════════════════════
# SINGLE WIDE TABLE IMPLEMENTATION - market_breadth
# ═══════════════════════════════════════════════════════════════

## Add these functions to data_manager.py (REPLACE the separate table versions)

def _get_market_breadth_columns():
    """Return the set of sector columns currently in the market_breadth table."""
    if db.use_supabase:
        try:
            row = db.read_table('market_breadth', limit=1)
            return set(row.columns) - {'Date'} if not row.empty else set()
        except Exception:
            return set()
    else:
        import sqlite3
        with sqlite3.connect(db.dbname) as conn:
            cursor = conn.execute('PRAGMA table_info(market_breadth)')
            return {r[1] for r in cursor.fetchall()} - {'Date'}


def get_missing_breadth_columns(new_sectors):
    """
    Return list of sectors that don't yet have a column in market_breadth.
    Used by the admin page to warn before a rebuild.
    """
    if not db.table_exists('market_breadth'):
        return list(new_sectors)
    existing = _get_market_breadth_columns()
    return [s for s in new_sectors if s not in existing]


def get_missing_ppi_tables(sectors):
    """Return list of sectors whose PPI_{sector} table doesn't exist yet."""
    return [s for s in sectors if not db.table_exists(f'PPI_{s}')]


_PPI_CREATE_SQL = (
    'CREATE TABLE IF NOT EXISTS "PPI_{sector}" ('
    '"Date" TEXT PRIMARY KEY, "Open" REAL, "High" REAL, '
    '"Low" REAL, "Close" REAL, "Norm_Vol_Metric" REAL);'
)


def save_market_breadth_to_db(breadth_data_by_date):
    """
    Save market breadth data for all sectors in a single wide table.

    For SQLite: automatically creates or alters the table to add missing
    sector columns.
    For Supabase: raises a clear RuntimeError with the SQL to run manually
    if any columns are missing — Supabase columns must be added in the
    dashboard before a rebuild is triggered.
    """
    table_name = "market_breadth"

    if not breadth_data_by_date:
        return

    incoming_sectors = list(next(iter(breadth_data_by_date.values())).keys())

    try:
        if not db.table_exists(table_name):
            if db_config.USE_SUPABASE:
                sql_lines = ['CREATE TABLE market_breadth (', '  "Date" TEXT PRIMARY KEY,']
                sql_lines += [f'  "{s}" DOUBLE PRECISION,' for s in incoming_sectors]
                sql_lines.append(');')
                sql = '\n'.join(sql_lines)
                raise RuntimeError(
                    f"market_breadth table does not exist in Supabase.\n"
                    f"Run this in the SQL editor first:\n\n{sql}"
                )
            else:
                columns = ['Date TEXT PRIMARY KEY'] + [f'"{s}" REAL' for s in incoming_sectors]
                db.create_table_sqlite(
                    f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
                )
                print(f"✓ Created table: {table_name}")
        else:
            # Table exists — check for missing sector columns
            existing_cols = _get_market_breadth_columns()
            missing = [s for s in incoming_sectors if s not in existing_cols]

            if missing:
                if db_config.USE_SUPABASE:
                    alter_stmts = '\n'.join(
                        f'ALTER TABLE market_breadth ADD COLUMN "{s}" DOUBLE PRECISION;'
                        for s in missing
                    )
                    raise RuntimeError(
                        f"market_breadth is missing columns for: {missing}\n"
                        f"Run this in the Supabase SQL editor before rebuilding:\n\n{alter_stmts}"
                    )
                else:
                    import sqlite3
                    with sqlite3.connect(db.dbname) as conn:
                        for s in missing:
                            conn.execute(f'ALTER TABLE market_breadth ADD COLUMN "{s}" REAL')
                        conn.commit()
                    print(f"✓ Added {len(missing)} new column(s) to market_breadth: {missing}")

        records = []
        for date, breadth_dict in breadth_data_by_date.items():
            records.append({
                'Date': date.strftime('%Y-%m-%d') if isinstance(date, (pd.Timestamp, datetime)) else date,
                **breadth_dict,
            })

        if records:
            if db_config.USE_SUPABASE:
                # PostgREST merge-upsert updates only the columns provided, leaving others intact
                db.insert_records(table_name, records, upsert=True)
            else:
                # SQLite: INSERT OR REPLACE wipes the whole row — use column-level ON CONFLICT upsert
                import sqlite3
                with sqlite3.connect(db.dbname) as conn:
                    for record in records:
                        sector_cols = [k for k in record if k != 'Date']
                        if not sector_cols:
                            continue
                        col_names = ', '.join(['"Date"'] + [f'"{c}"' for c in sector_cols])
                        placeholders = ', '.join(['?'] * (len(sector_cols) + 1))
                        set_clause = ', '.join([f'"{c}" = excluded."{c}"' for c in sector_cols])
                        sql = (
                            f'INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) '
                            f'ON CONFLICT ("Date") DO UPDATE SET {set_clause}'
                        )
                        conn.execute(sql, [record['Date']] + [record[c] for c in sector_cols])
                    conn.commit()
            print(f"✓ Saved {len(records)} breadth records to database")

    except RuntimeError:
        raise   # re-raise so callers (rebuild runner, admin page) can surface it
    except Exception as e:
        print(f"Failed to save market breadth: {e}")
        import traceback
        traceback.print_exc()


def load_market_breadth_from_db(start_date=None, end_date=None):
    """
    Load market breadth data from the single wide table.
    Works with both SQLite and Supabase via DatabaseManager.

    Args:
        start_date: Optional start date filter (YYYY-MM-DD string)
        end_date: Optional end date filter (YYYY-MM-DD string)

    Returns:
        DataFrame with Date index and sector columns, or None
    """
    table_name = "market_breadth"

    try:
        if not db.table_exists(table_name):
            return None

        # Use db_manager's read_table method
        # Note: filters are for exact matches, so we need to get all and filter in pandas
        df = db.read_table(table_name, order_by='Date')

        if df.empty:
            return None

        # Convert Date column and set as index
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df = df.set_index('Date').sort_index()

        # Apply date filters if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    except Exception as e:
        print(f"Failed to load market breadth: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def add_new_sector_breadth(new_sector_name, stock_list):
    """
    Add market breadth data for a new sector to the existing table.
    Only calculates for dates that already exist in the table.
    
    IMPORTANT: For Supabase, you must manually add the column first:
    ALTER TABLE market_breadth ADD COLUMN "{new_sector_name}" REAL;
    
    Args:
        new_sector_name: Name of the new sector
        stock_list: List of stock tickers in the sector
    
    Returns:
        Number of rows updated, or None if failed
    """
    table_name = "market_breadth"
    
    try:
        print(f"Adding breadth for new sector: {new_sector_name}")
        
        # Step 1: Check if table exists
        if not db.table_exists(table_name):
            print(f"❌ Table {table_name} doesn't exist. Please create it first.")
            return None
        
        # Step 2: Add column (SQLite only)
        if db_config.USE_SUPABASE:
            print(f"⚠️  For Supabase, column must be added manually first:")
            print(f'   ALTER TABLE {table_name} ADD COLUMN "{new_sector_name}" REAL;')
            print()
        else:
            # SQLite - add column automatically
            try:
                conn = db.get_connection()
                cursor = conn.cursor()
                cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN "{new_sector_name}" REAL')
                conn.commit()
                conn.close()
                print(f"✓ Added column: {new_sector_name}")
            except Exception as e:
                if "duplicate column" in str(e).lower():
                    print(f"✓ Column {new_sector_name} already exists")
                else:
                    raise
        
        # Step 3: Get all existing dates from the table
        df_dates = db.read_table(table_name, columns='Date', order_by='Date')
        if df_dates.empty:
            print(f"⚠ No existing dates in {table_name}, nothing to update")
            return 0
        
        existing_dates = pd.to_datetime(df_dates['Date']).sort_values().tolist()
        print(f"Found {len(existing_dates)} existing dates")
        
        # Step 4: Fetch LIVE stock data for all tickers
        print(f"📡 Fetching LIVE data for {len(stock_list)} stocks...")
        all_stock_data = {}
        
        # Determine date range needed
        start_date = existing_dates[0]
        end_date = existing_dates[-1]
        
        for i, ticker in enumerate(stock_list, 1):
            print(f"  [{i}/{len(stock_list)}] Fetching {ticker}...", end='\r')
            
            # Fetch live data with enough lookback for MA20 calculation
            df = get_single_stock_data_live(
                ticker,
                start_date=(start_date - timedelta(days=30)).strftime('%Y%m%d'),  # Extra 30 days for MA20
                end_date=end_date.strftime('%Y%m%d')
            )
            
            if df is not None and not df.empty:
                all_stock_data[ticker] = df
            
            time.sleep(0.35)  # Rate limit for Tushare API
        
        print()  # New line after progress
        
        if not all_stock_data:
            print(f"❌ No stock data available for {new_sector_name}")
            return 0
        
        print(f"✓ Loaded {len(all_stock_data)}/{len(stock_list)} stocks")
        
        # Step 5: Calculate breadth for all dates
        print(f"🔄 Calculating breadth for {len(existing_dates)} dates...")
        updates = []
        success_count = 0
        fail_count = 0
        
        for i, date in enumerate(existing_dates):
            breadth = calculate_sector_market_breadth(
                new_sector_name, date, all_stock_data
            )
            if breadth is not None:
                updates.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    new_sector_name: breadth
                })
                success_count += 1
            else:
                fail_count += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0 or (i + 1) == len(existing_dates):
                print(f"  Progress: {i+1}/{len(existing_dates)} dates (✓ {success_count}, ✗ {fail_count})")
        
        if not updates:
            print(f"❌ No valid breadth values calculated for {new_sector_name}")
            return 0
        
        # Step 6: Batch update
        print(f"💾 Updating {len(updates)} rows in database...")
        db.insert_records(table_name, updates, upsert=True)
        
        print(f"✅ Successfully added {new_sector_name} breadth for {len(updates)} dates")
        print(f"   Success rate: {success_count}/{len(existing_dates)} ({100*success_count/len(existing_dates):.1f}%)")
        return len(updates)
    
    except Exception as e:
        print(f"❌ Failed to add new sector breadth: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_sector_market_breadth(sector, current_date, all_stock_data=None):
    """
    Calculate market breadth for a sector: % of stocks above their MA20.

    Args:
        sector: Sector name
        current_date: Date to calculate breadth for (datetime or Timestamp)
        all_stock_data: Dict of {ticker: DataFrame} with stock data
                       If None, will fetch from database

    Returns:
        float: Market breadth (0.0 to 1.0), or None if insufficient data
    """
    if sector not in SECTOR_STOCK_MAP:
        return None

    if sector == 'MARKET_PROXY':
        return 0.5  # Neutral for market proxy

    stock_list = SECTOR_STOCK_MAP[sector]

    # Load stock data if not provided
    if all_stock_data is None:
        all_stock_data = {}
        for ticker in stock_list:
            df = get_single_stock_data_from_db(ticker)
            if df is not None and not df.empty:
                all_stock_data[ticker] = df

    stocks_above_ma20 = 0
    total_valid_stocks = 0

    for ticker in stock_list:
        if ticker not in all_stock_data:
            continue

        df = all_stock_data[ticker]

        # Filter up to current date
        df_filtered = df[df.index <= current_date]
        if len(df_filtered) < 20:
            continue

        # Calculate MA20
        ma20 = df_filtered['Close'].rolling(window=20).mean()

        if current_date not in df_filtered.index:
            continue

        close_price = df_filtered.loc[current_date, 'Close']
        ma20_price = ma20.loc[current_date]

        if pd.notna(close_price) and pd.notna(ma20_price):
            total_valid_stocks += 1
            if close_price > ma20_price:
                stocks_above_ma20 += 1

    if total_valid_stocks == 0:
        return None

    return stocks_above_ma20 / total_valid_stocks



# --------------- Stock Daily Baiscs ---------------- 
def create_daily_basic_table():
    """Creates daily_basic table (SQLite only — Supabase needs manual creation)."""
    if db_config.USE_SQLITE:
        schema = """CREATE TABLE IF NOT EXISTS daily_basic (
            ts_code         TEXT,
            trade_date      TEXT,
            close           REAL,
            turnover_rate   REAL,
            turnover_rate_f REAL,
            volume_ratio    REAL,
            pe              REAL,
            pe_ttm          REAL,
            pb              REAL,
            ps              REAL,
            ps_ttm          REAL,
            dv_ratio        REAL,
            dv_ttm          REAL,
            total_share     REAL,
            float_share     REAL,
            free_share      REAL,
            total_mv        REAL,
            circ_mv         REAL,
            PRIMARY KEY (ts_code, trade_date)
        )"""
        db.create_table_sqlite(schema)


def should_update_daily_basic():
    """
    Returns True if daily_basic needs updating.
      - After 18:00 Beijing → update if latest date < today
      - Before 18:00 Beijing → update if latest date < yesterday
    """
    from zoneinfo import ZoneInfo
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
    now_beijing = datetime.now(CHINA_TZ)
    today_beijing = now_beijing.date()
    yesterday_beijing = today_beijing - timedelta(days=1)

    threshold = today_beijing if now_beijing.hour >= 18 else yesterday_beijing

    try:
        if not db.table_exists('daily_basic'):
            print("[data_manager] daily_basic table doesn't exist → needs load")
            return True

        df = db.read_table('daily_basic', columns='trade_date', order_by='-trade_date', limit=1)
        if df.empty:
            print("[data_manager] daily_basic is empty → needs load")
            return True

        latest_date = pd.to_datetime(df['trade_date'].iloc[0]).date()
        needs_update = latest_date < threshold
        print(f"[data_manager] daily_basic latest: {latest_date} | threshold: {threshold} | update needed: {needs_update}")
        return needs_update

    except Exception as e:
        print(f"[data_manager] Error checking daily_basic: {e}")
        return True


def update_daily_basic():
    """
    Fetches latest daily_basic for ALL stocks from Tushare and upserts to DB.
    Returns True if updated, False if skipped/no data, None if error.
    """
    global TUSHARE_API

    if not should_update_daily_basic():
        print("[data_manager] ✅ daily_basic already up-to-date, skipping.")
        return False

    if TUSHARE_API is None:
        ok = init_tushare()
        if not ok:
            print("[data_manager] ❌ Tushare init failed")
            return None
        
    from zoneinfo import ZoneInfo
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
    now_beijing = datetime.now(CHINA_TZ)
    today_beijing = now_beijing.date()

    # ── Use the same threshold logic as should_update_daily_basic ──
    if now_beijing.hour >= 18:
        fetch_date = today_beijing
    else:
        fetch_date = today_beijing - timedelta(days=1)  # yesterday

    fetch_date_str = fetch_date.strftime('%Y%m%d')

    print(f"[data_manager] 📡 Fetching daily_basic (date: {fetch_date_str})...")

    try:
        create_daily_basic_table()

        df = TUSHARE_API.daily_basic(
            trade_date=fetch_date_str,
            fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,'
                   'volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,'
                   'total_share,float_share,free_share,total_mv,circ_mv'
        )

        if df is None or df.empty:
            print(f"[data_manager] ℹ️ No data for {fetch_date_str} (weekend/holiday), skipping.")
            return False

        # Before converting to records and inserting:
        df = df.replace([np.inf, -np.inf], np.nan)  # convert inf → NaN
        records = df.to_dict('records')

        # Replace any remaining float NaN with None (JSON null, not NaN)
        records = [
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
            for row in records
        ]

        db.insert_records('daily_basic', records, upsert=True)
        print(f"[data_manager] ✅ Saved {len(df)} daily_basic records (trade_date: {df['trade_date'].iloc[0]})")
        return True

    except Exception as e:
        print(f"[data_manager] ❌ Failed to update daily_basic: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_daily_basic_for_tickers(tickers: list):
    """
    Returns latest daily_basic rows for the given list of 6-digit tickers.
    """
    try:
        if not db.table_exists('daily_basic'):
            return pd.DataFrame()

        df_date = db.read_table('daily_basic', columns='trade_date', order_by='-trade_date', limit=1)
        if df_date.empty:
            return pd.DataFrame()

        latest_date = df_date['trade_date'].iloc[0]
        df = db.read_table('daily_basic', filters={'trade_date': latest_date})
        if df.empty:
            return pd.DataFrame()

        df['ticker'] = df['ts_code'].str.split('.').str[0]
        df = df[df['ticker'].isin(tickers)].copy()
        df['total_mv_yi'] = df['total_mv'] / 10000
        df['circ_mv_yi']  = df['circ_mv']  / 10000
        return df

    except Exception as e:
        print(f"[data_manager] ❌ get_daily_basic_for_tickers failed: {e}")
        return pd.DataFrame()
    
def get_daily_basic_latest(tickers: list) -> pd.DataFrame:
    """
    Fetch the latest daily_basic row for a list of tickers from the DB.
    Returns a DataFrame indexed by ticker with trading metrics.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        # Read all rows for these tickers, then keep only the latest per ticker
        ts_codes = [get_tushare_ticker(t) for t in tickers]
        placeholders = ','.join(['?' for _ in ts_codes])  # SQLite
        df = db.read_table(
            'daily_basic',
            columns='ts_code, trade_date, close, pe_ttm, pb, turnover_rate, circ_mv',
            order_by='-trade_date'
        )
        if df.empty:
            return pd.DataFrame()
        df = df[df['ts_code'].isin(ts_codes)]
        # Keep latest row per ticker
        df = df.sort_values('trade_date', ascending=False).drop_duplicates('ts_code')
        # Normalize ticker back to 6-digit for merging
        df['ticker'] = df['ts_code'].str[:6]
        df['circ_mv_yi'] = (pd.to_numeric(df['circ_mv'], errors='coerce') / 10000).round(2)
        df['pe_ttm'] = pd.to_numeric(df['pe_ttm'], errors='coerce').round(1)
        df['pb'] = pd.to_numeric(df['pb'], errors='coerce').round(2)
        df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce').round(2)
        return df[['ticker', 'close', 'pe_ttm', 'pb', 'turnover_rate', 'circ_mv_yi', 'trade_date']]
    except Exception as e:
        print(f"[data_manager] ❌ get_daily_basic_latest failed: {e}")
        return pd.DataFrame()


# --------------- New FUnd Management Feature -------------#
def execute_daily_portfolio_rollup(target_date: str = None):
    """
    NIGHTLY CRON WRAPPER: Handles API constraints, halts, and pings Tushare.
    Passes data to process_fund_nav() and update_fund_risk_metrics().
    """
    from datetime import timezone, timedelta, datetime
    import time
    import pandas as pd
    
    BJT = timezone(timedelta(hours=8))
    now_bjt = datetime.now(BJT)
    
    if target_date is None:
        if now_bjt.hour >= 18:
            target_date = now_bjt.strftime('%Y%m%d')
        else:
            target_date = (now_bjt - timedelta(days=1)).strftime('%Y%m%d')
            
    target_date_sql = pd.to_datetime(target_date).strftime('%Y-%m-%d')
    start_buffer_date = (datetime.strptime(target_date, '%Y%m%d') - timedelta(days=15)).strftime('%Y%m%d')

    print(f"\n📊 --- Running Fund NAV Roll-Forward for {target_date_sql} ---")

    # SAFEGUARD 1: THE MARKET READY PING
    ping_df = get_single_stock_data_live('600519.SH', start_date=start_buffer_date, end_date=target_date)
    is_market_open = False
    
    if ping_df is not None and not ping_df.empty:
        latest_date_in_db = ping_df.index[-1].strftime('%Y%m%d')
        if latest_date_in_db == target_date:
            is_market_open = True
            
    if not is_market_open:
        print(f"   😴 ABORTING: Tushare data for {target_date} is not yet available, or it was a weekend/holiday.")
        return

    # FETCH ALL ACTIVE TICKERS
    funds_df = db.read_table('funds')
    if funds_df.empty: return
    
    positions_df = db.read_table('fund_positions')
    if positions_df.empty: return
    
    active_mask = (positions_df['effective_date'] <= target_date_sql) & \
                  (positions_df['end_date'].isna() | (positions_df['end_date'] > target_date_sql))
    active_positions = positions_df[active_mask]
    
    if active_positions.empty: return
    unique_tickers = active_positions['ts_code'].unique().tolist()

    # BUILD DAILY RETURNS DICTIONARY
    print(f"   📡 Fetching live return data for {len(unique_tickers)} portfolio tickers...")
    stock_returns = {}
    for ticker in unique_tickers:
        df = get_single_stock_data_live(ticker, start_date=start_buffer_date, end_date=target_date)
        
        if df is not None and not df.empty and len(df) >= 2:
            latest_date_in_df = df.index[-1].strftime('%Y%m%d')
            if latest_date_in_df == target_date:
                recent_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                stock_returns[ticker] = (recent_close - prev_close) / prev_close
            else:
                stock_returns[ticker] = 0.0 # Halted
                print(f"   ⏸️ {ticker} halted on {target_date}. Assuming 0.0% return.")
        else:
            stock_returns[ticker] = 0.0
            
        time.sleep(0.35) 

    # RUN THE MODULAR ENGINES
    for _, fund in funds_df.iterrows():
        fund_id = fund['id']
        fund_name = fund['fund_name']
        
        # 1. Math & Accounting
        success, new_aum, ret, flow = process_fund_nav(fund_id, target_date_sql, stock_returns)
        
        if success:
            # 2. Advanced Risk
            update_fund_risk_metrics(fund_id, target_date_sql)
            print(f"   ✅ {fund_name}: AUM ¥{new_aum:,.2f} | Return: {ret*100:.2f}% | Flow: ¥{flow}")

    print("📊 --- Fund NAV Roll-Forward Complete ---\n")

# 
# Notice that it sets the effective_date of the weights to tomorrow. 
# This is standard institutional practice: if you lock in a mandate today, 
# the backtest/tracking officially begins at tomorrow's market open using today's closing prices as the capital baseline.
#
def save_fund_mandate(fund_name, benchmark, positions_dict):
    """
    Saves a newly optimized fund and its initial target weights into the database,
    tied securely to the active user.
    """
    import auth_manager
    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta
    
    try:
        # Get the currently logged-in user
        user_id = auth_manager.get_current_user_id()
        if not user_id:
            return False, "❌ Authentication error: Please log in to save a mandate."

        CHINA_TZ = ZoneInfo("Asia/Shanghai")
        now_bjt = datetime.now(CHINA_TZ)
        
        today_str = now_bjt.strftime('%Y-%m-%d')
        # Target weights go live tomorrow to prevent look-ahead bias
        tomorrow_str = (now_bjt + timedelta(days=1)).strftime('%Y-%m-%d')

        # 1. Check for duplicates PER USER
        existing_fund = db.read_table('funds', filters={'fund_name': fund_name, 'user_id': user_id})
        if not existing_fund.empty:
            return False, f"⚠️ You already have a fund named '{fund_name}'."

        # 2. Insert the Mandate Header WITH user_id
        fund_record = [{
            'user_id': user_id,
            'fund_name': fund_name,
            'benchmark': benchmark,
            'inception_date': today_str
        }]
        db.insert_records('funds', fund_record)

        # 3. Retrieve the newly generated fund_id
        new_fund_df = db.read_table('funds', filters={'fund_name': fund_name, 'user_id': user_id})
        if new_fund_df.empty:
            return False, "❌ Failed to retrieve the new fund ID from the database."
        fund_id = int(new_fund_df.iloc[0]['id'])

        # 4. Insert the Positions
        position_records = []
        for ts_code, weight in positions_dict.items():
            if weight > 0: 
                position_records.append({
                    'fund_id': fund_id,
                    'ts_code': ts_code,
                    'weight': float(weight),
                    'effective_date': tomorrow_str,
                    'end_date': None 
                })
        
        db.insert_records('fund_positions', position_records)
        return True, f"✅ Successfully launched '{fund_name}' with {len(position_records)} active positions."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"❌ Database error: {str(e)}"



def process_fund_nav(fund_id: int, target_date_sql: str, daily_returns: dict):
    """
    CORE ENGINE 1: The Accounting Module.
    Takes a dictionary of {ticker: pct_return} for a specific date, applies active weights,
    calculates the new AUM, and preserves idempotency.

    Also computes and stores actual market-value weights (fund_daily_weights) so that
    weight drift — the divergence of actual holdings from the target mandate — is tracked
    daily without assuming continuous rebalancing.
    """

    # Fetch active holdings for this fund
    positions_df = db.read_table('fund_positions', filters={'fund_id': fund_id})
    if positions_df.empty:
        return False, 0, 0, 0

    active_mask = (positions_df['effective_date'] <= target_date_sql) & \
                  (positions_df['end_date'].isna() | (positions_df['end_date'] > target_date_sql))
    fund_holdings = positions_df[active_mask]

    if fund_holdings.empty:
        return False, 0, 0, 0

    all_metrics = db.read_table('fund_daily_metrics', filters={'fund_id': fund_id})

    # SAFEGUARD 2: Preserve user-inputted net_flow if overwriting a failed/duplicate run
    current_net_flow = 0.0
    if not all_metrics.empty:
        today_record = all_metrics[all_metrics['trade_date'] == target_date_sql]
        if not today_record.empty:
            current_net_flow = today_record.iloc[0].get('net_flow', 0.0)

    # SAFEGUARD 3: Strictly grab yesterday's AUM to prevent double-compounding
    starting_aum = 10000000.0
    prev_date_sql = None
    if not all_metrics.empty:
        past_metrics = all_metrics[all_metrics['trade_date'] < target_date_sql].sort_values(by='trade_date', ascending=False)
        if not past_metrics.empty:
            starting_aum = past_metrics.iloc[0]['total_aum']
            prev_date_sql = past_metrics.iloc[0]['trade_date']

    starting_capital = starting_aum + current_net_flow

    # Apply target weights to compute portfolio return
    portfolio_daily_return = 0.0
    for _, pos in fund_holdings.iterrows():
        ticker = pos['ts_code']
        weight = pos['weight']
        stock_ret = daily_returns.get(ticker, 0.0)
        portfolio_daily_return += (weight * stock_ret)

    new_aum = starting_capital * (1 + portfolio_daily_return)
    daily_pnl = new_aum - starting_capital

    # Save Accounting Data (Risk metrics default to None, updated by Risk Module next)
    metric_record = {
        'fund_id': fund_id,
        'trade_date': target_date_sql,
        'total_aum': new_aum,
        'daily_pnl': daily_pnl,
        'net_flow': current_net_flow,
        'beta_30d': None,
        'var_95': None,
        'volatility_annualized': None
    }
    db.insert_records('fund_daily_metrics', [metric_record], upsert=True)

    # ── Weight Drift Tracking ─────────────────────────────────────────────────
    # Fetch the previous day's REAL (non-simulated) actual weights.
    # Simulated backfill records are intentionally excluded so the nightly
    # rollup always starts fresh from target weights on its very first run,
    # rather than chaining onto the retroactive simulation.
    prev_actual_weights = {}
    if prev_date_sql:
        prev_w_df = db.read_table(
            'fund_daily_weights',
            filters={'fund_id': fund_id, 'trade_date': prev_date_sql}
        )
        if not prev_w_df.empty:
            # Keep only real rows — ignore any simulated backfill rows
            if 'is_simulated' in prev_w_df.columns:
                prev_w_df = prev_w_df[prev_w_df['is_simulated'].astype(int) == 0]
            if not prev_w_df.empty:
                prev_actual_weights = dict(zip(prev_w_df['ts_code'], prev_w_df['actual_weight']))

    # Drift formula: aw_i(t) = aw_i(t-1) * (1 + r_i(t)) / (1 + r_port(t))
    # On the first day (no prior record) aw_i(t-1) = target_weight_i
    actual_weights = {}
    divisor = 1.0 + portfolio_daily_return if portfolio_daily_return != -1.0 else 1.0
    for _, pos in fund_holdings.iterrows():
        ticker = pos['ts_code']
        target_w = float(pos['weight'])
        prev_aw = prev_actual_weights.get(ticker, target_w)
        stock_ret = daily_returns.get(ticker, 0.0)
        actual_weights[ticker] = prev_aw * (1.0 + stock_ret) / divisor

    # Normalise to guard against floating-point drift (theoretically already sums to 1)
    total_aw = sum(actual_weights.values())
    if total_aw > 0:
        actual_weights = {k: v / total_aw for k, v in actual_weights.items()}

    weight_records = []
    for _, pos in fund_holdings.iterrows():
        ticker = pos['ts_code']
        target_w = float(pos['weight'])
        actual_w = actual_weights.get(ticker, target_w)
        weight_records.append({
            'fund_id':        fund_id,
            'trade_date':     target_date_sql,
            'ts_code':        ticker,
            'target_weight':  target_w,
            'actual_weight':  actual_w,
            'drift_pct':      round(actual_w - target_w, 6),
            'is_simulated':   0,  # Real nightly rollup — not a retroactive simulation
        })
    if weight_records:
        db.insert_records('fund_daily_weights', weight_records, upsert=True)

    return True, new_aum, portfolio_daily_return, current_net_flow


def update_fund_risk_metrics(fund_id: int, target_date_sql: str, pre_fetched_bench_returns: dict = None):
    """
    CORE ENGINE 2: The Risk Module.
    Calculates institutional risk (VaR, Volatility, and Beta) based on the last 30 days.
    Accepts pre_fetched_bench_returns to prevent N+1 API calls during historical backfills.
    """
    import numpy as np
    import pandas as pd
    import db_config
    from db_manager import db
    
    # 1. Look up the fund's specific benchmark
    fund_df = db.read_table('funds', filters={'id': fund_id})
    if fund_df.empty: return
    benchmark = fund_df.iloc[0]['benchmark']
    if benchmark and " " in benchmark:
        benchmark = benchmark.split(" ")[0]

    # 2. Get past 30 days of AUM metrics
    metrics = db.read_table('fund_daily_metrics', filters={'fund_id': fund_id})
    if metrics.empty: return
        
    past_30 = metrics[metrics['trade_date'] <= target_date_sql].sort_values('trade_date').tail(30)
    if len(past_30) < 5: return 
        
    aum_series = past_30['total_aum'].values
    flows_series = past_30['net_flow'].values
    trade_dates = past_30['trade_date'].values
    
    fund_rets = []
    valid_dates = []
    for i in range(1, len(aum_series)):
        prev_aum = aum_series[i-1]
        curr_aum = aum_series[i]
        flow = flows_series[i]
        ret = (curr_aum - prev_aum - flow) / prev_aum if prev_aum > 0 else 0
        fund_rets.append(ret)
        valid_dates.append(trade_dates[i])
        
    if not fund_rets: return
    fund_rets = np.array(fund_rets)
    
    # --- RISK MATH 1: Volatility & VaR ---
    # A-shares trade ~242 days/year (not the US/global 252)
    volatility_ann = np.std(fund_rets) * np.sqrt(242)
    var_95 = np.percentile(fund_rets, 5) 
    
    # --- RISK MATH 2: Beta ---
    beta_30d = None
    if benchmark and len(valid_dates) > 2:
        start_dt = valid_dates[0].replace('-', '')
        end_dt = valid_dates[-1].replace('-', '')
        
        # USE INJECTED DATA IF AVAILABLE, OTHERWISE FETCH LIVE
        if pre_fetched_bench_returns is not None:
            bench_ret_dict = pre_fetched_bench_returns
        else:
            bench_df = get_index_data_live(benchmark, start_date=start_dt, end_date=end_dt)
            if bench_df is not None and not bench_df.empty:
                bench_df['date_str'] = bench_df.index.strftime('%Y-%m-%d')
                bench_ret_dict = dict(zip(bench_df['date_str'], bench_df['Pct_Change'] / 100.0))
            else:
                bench_ret_dict = {}
            
        aligned_fund = []
        aligned_bench = []
        
        for i, d in enumerate(valid_dates):
            if d in bench_ret_dict:
                aligned_fund.append(fund_rets[i])
                aligned_bench.append(bench_ret_dict[d])
                
        if len(aligned_bench) > 2:
            cov_matrix = np.cov(aligned_fund, aligned_bench)
            if cov_matrix[1, 1] != 0:
                beta_30d = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # 3. Update the database
    if db_config.USE_SUPABASE:
        from db_manager import supabase_client
        update_data = {'volatility_annualized': float(volatility_ann), 'var_95': float(var_95)}
        if beta_30d is not None: update_data['beta_30d'] = float(beta_30d)
        
        supabase_client.table('fund_daily_metrics')\
            .update(update_data)\
            .eq('fund_id', fund_id).eq('trade_date', target_date_sql).execute()
    else:
        conn = db.get_connection()
        cursor = conn.cursor()
        if beta_30d is not None:
            cursor.execute(
                "UPDATE fund_daily_metrics SET volatility_annualized = ?, var_95 = ?, beta_30d = ? WHERE fund_id = ? AND trade_date = ?", 
                (float(volatility_ann), float(var_95), float(beta_30d), fund_id, target_date_sql)
            )
        else:
            cursor.execute(
                "UPDATE fund_daily_metrics SET volatility_annualized = ?, var_95 = ? WHERE fund_id = ? AND trade_date = ?", 
                (float(volatility_ann), float(var_95), fund_id, target_date_sql)
            )
        conn.commit()
        conn.close()


def execute_fund_rebalance(fund_id: int, new_positions: dict):
    """
    Executes a portfolio rebalance for an existing mandate.
    Closes out the current active positions (sets end_date to China today) 
    and inserts the new target weights (effective_date to China tomorrow).
    """
    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta
    from db_manager import db
    
    try:
        # 1. Strictly define China Time
        CHINA_TZ = ZoneInfo("Asia/Shanghai")
        china_now = datetime.now(CHINA_TZ)
        
        # 2. Calculate Dates
        china_today = china_now.strftime('%Y-%m-%d')
        china_tomorrow = (china_now + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 3. Close out old positions (End date = Today)
        db.update_records(
            table_name='fund_positions',
            update_values={'end_date': china_tomorrow},
            filters={'fund_id': fund_id, 'end_date': None}
        )
        
        # 4. Insert the new target weights (Effective date = Tomorrow)
        insert_data = []
        for ts_code, weight in new_positions.items():
            insert_data.append({
                'fund_id': fund_id,
                'ts_code': ts_code,
                'weight': float(weight),
                'effective_date': china_tomorrow,
                'end_date': None
            })
        
        if insert_data:
            db.insert_records('fund_positions', insert_data, upsert=False)
            
        return True, "Rebalance executed successfully. New mandate goes live at tomorrow's open."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Database error during rebalance: {str(e)}"


def backfill_fund_history(fund_id, positions, initial_aum=10000000.0, lookback_days=180):
    """
    No-compromise backfill. Calculates base NAV instantly, then brute-forces
    the institutional risk metrics (Beta, VaR, Vol) for EVERY SINGLE historical day.

    Price source: Tushare pro_bar with adj='qfq' (forward-adjusted) via
    get_single_stock_data_live() — same basis as the Portfolio Optimizer, so
    backtest returns are directly comparable to optimisation outputs.

    Also populates fund_daily_weights with the actual market-value weight for
    every stock on every day so weight drift is visible from day one.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    import pandas as pd
    import numpy as np

    CHINA_TZ = ZoneInfo("Asia/Shanghai")
    end_date = datetime.now(CHINA_TZ).date()

    # Add 30 extra days of lookback buffer so the first day of the 180-day window
    # has enough history to calculate the 30-day rolling risk metrics.
    start_date = end_date - timedelta(days=lookback_days + 30)

    print(f"[data_manager] ⚙️ Starting COMPLETE historical backfill for Fund ID {fund_id}...")

    # 1. Fetch historical data for all stocks in the mandate via live qfq API
    ts_codes = list(positions.keys())
    historical_data = {}

    for ticker in ts_codes:
        clean_ticker = ticker.split('.')[0]
        # get_single_stock_data_live uses pro_bar adj='qfq' — same basis as the optimizer
        df = get_single_stock_data_live(
            clean_ticker,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
        )

        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df_window = df.loc[
                (df.index >= pd.to_datetime(start_date)) &
                (df.index <= pd.to_datetime(end_date))
            ].copy()

            if not df_window.empty:
                # pct_change from qfq Close gives the correct dividend/split-adjusted return
                df_window['pct_chg'] = df_window['Close'].pct_change().fillna(0)
                historical_data[ticker] = df_window['pct_chg']

    if not historical_data:
        print("[data_manager] ⚠️ No historical data found for backfill.")
        return False

    # 2. Combine all stock returns into a single DataFrame
    pivot_df = pd.DataFrame(historical_data).fillna(0)
    for code in ts_codes:
        if code not in pivot_df.columns:
            pivot_df[code] = 0.0

    # 3. Calculate Portfolio Daily Return & AUM (using target weights — no rebalancing assumed)
    weights = pd.Series(positions)
    weights = weights.reindex(pivot_df.columns).fillna(0)
    pivot_df['portfolio_return'] = pivot_df[list(weights.index)].dot(weights)
    pivot_df['cumulative_return'] = (1 + pivot_df['portfolio_return']).cumprod()
    pivot_df['total_aum'] = initial_aum * pivot_df['cumulative_return']

    # 4. Insert Base AUM Records into Database (Risk metrics temporarily None)
    records = []
    for date, row in pivot_df.iterrows():
        records.append({
            'fund_id':    fund_id,
            'trade_date': date.strftime('%Y-%m-%d'),
            'total_aum':  float(row['total_aum']),
            'daily_pnl':  float(
                row['total_aum'] - (row['total_aum'] / (1 + row['portfolio_return']))
                if row['portfolio_return'] != 0 else 0
            ),
            'net_flow':   0.0,
            'beta_30d':   None,
            'var_95':     None,
            'volatility_annualized': None,
        })

    if not records:
        return False

    db.insert_records('fund_daily_metrics', records, upsert=True)
    print(f"[data_manager] ✅ Base AUM inserted for {len(records)} days. Starting Weight Drift Engine...")

    # 5. Compute and store daily weight drift for every backfill date ──────────
    # Weight drift model (buy-and-hold between rebalances):
    #   actual_weight_i(t) = target_i × ∏_{s=1..t} [(1+r_i(s)) / (1+r_port(s))]
    #
    # Using vectorised cumprod:
    #   daily_drift_factor_i(t) = (1+r_i(t)) / (1+r_port(t))
    #   cum_drift_i(t)           = ∏ daily_drift_factor_i(1..t)
    #   actual_weight_i(t)       = target_i × cum_drift_i(t)
    #
    # On t=0 (inception day, after first day's return) the formula correctly shows the
    # weight after the first market move.  Row 0's 'pct_chg' is 0 (fillna), so
    # cum_drift = 1 there — effectively actual = target on the first row.

    stock_cols = [c for c in ts_codes if c in pivot_df.columns]

    daily_drift_factor = (1 + pivot_df[stock_cols]).div(
        (1 + pivot_df['portfolio_return']), axis=0
    )
    cum_drift = daily_drift_factor.cumprod()

    weight_records = []
    for date, drift_row in cum_drift.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        for code in stock_cols:
            target_w = float(weights.get(code, 0.0))
            actual_w = target_w * float(drift_row[code])
            weight_records.append({
                'fund_id':        fund_id,
                'trade_date':     date_str,
                'ts_code':        code,
                'target_weight':  target_w,
                'actual_weight':  actual_w,
                'drift_pct':      round(actual_w - target_w, 6),
                'is_simulated':   1,  # Retroactive simulation — NOT real forward drift
            })

    if weight_records:
        db.insert_records('fund_daily_weights', weight_records, upsert=True)
        print(f"[data_manager] ✅ Simulated weight drift stored for {len(weight_records)} position-days.")

    # 6. Pre-fetch Benchmark for the Risk Engine
    # (Fetch once and pass it down so Tushare doesn't block for duplicate calls)
    fund_df = db.read_table('funds', filters={'id': fund_id})
    benchmark = fund_df.iloc[0]['benchmark'] if not fund_df.empty else '000300.SH'
    if benchmark and " " in benchmark:
        benchmark = benchmark.split(" ")[0]

    bench_start = start_date.strftime('%Y%m%d')
    bench_end   = end_date.strftime('%Y%m%d')
    bench_df = get_index_data_live(benchmark, start_date=bench_start, end_date=bench_end)

    pre_fetched_bench = {}
    if bench_df is not None and not bench_df.empty:
        bench_df['date_str'] = bench_df.index.strftime('%Y-%m-%d')
        pre_fetched_bench = dict(zip(bench_df['date_str'], bench_df['Pct_Change'] / 100.0))

    # 7. BRUTE FORCE: Calculate Risk Metrics for EVERY SINGLE DAY
    # Only loop through the actual lookback window (ignoring the 30-day buffer)
    actual_start_date = end_date - timedelta(days=lookback_days)
    target_dates = [
        r['trade_date'] for r in records
        if pd.to_datetime(r['trade_date']).date() >= actual_start_date
    ]

    print(f"[data_manager] 🧮 Calculating historical risk metrics for {len(target_dates)} days...")
    for date_sql in target_dates:
        update_fund_risk_metrics(fund_id, date_sql, pre_fetched_bench_returns=pre_fetched_bench)

    print(f"[data_manager] ✅ 100% COMPLETE. All historical data, weight drift, and risk metrics filled.")
    return True


def init_portfolio_tables():
    """
    Initialize tables for Institutional Portfolio Management.

    ====================================================================
    DOCUMENTATION: TOTAL RETURN (TR) & DRIP ASSUMPTION
    ====================================================================
    This portfolio tracker operates as a Total Return Index using daily
    percentage changes (`pct_chg`), rather than absolute accounting of
    shares and cash.

    Because Tushare adjusts the `pre_close` on ex-dividend dates, the
    `pct_chg` intrinsically captures the value of the distribution. By
    applying this `pct_chg` directly to the active AUM, the system
    mathematically assumes Automatic Dividend Reinvestment (DRIP) with
    zero cash drag.

    Forward-adjusted prices (qfq) are EXPLICITLY NOT STORED to prevent
    historical database corruption during stock splits.
    ====================================================================

    ====================================================================
    SUPABASE MIGRATION — fund_daily_weights (run once in SQL Editor)
    ====================================================================
    CREATE TABLE IF NOT EXISTS fund_daily_weights (
        id            BIGSERIAL PRIMARY KEY,
        fund_id       INTEGER NOT NULL REFERENCES funds(id),
        trade_date    DATE NOT NULL,
        ts_code       TEXT NOT NULL,
        target_weight REAL NOT NULL,
        actual_weight REAL NOT NULL,
        drift_pct     REAL NOT NULL,
        is_simulated  BOOLEAN NOT NULL DEFAULT FALSE,
        UNIQUE(fund_id, trade_date, ts_code)
    );

    -- If the table already exists, add the column with:
    ALTER TABLE fund_daily_weights
        ADD COLUMN IF NOT EXISTS is_simulated BOOLEAN NOT NULL DEFAULT FALSE;
    ====================================================================
    """
    
    # 1. Funds Table (The Mandates) - UPDATED FOR MULTI-TENANT
    funds_schema = """
    CREATE TABLE IF NOT EXISTS funds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        fund_name TEXT NOT NULL,
        benchmark TEXT,
        inception_date DATE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES app_user(id) ON DELETE CASCADE,
        UNIQUE(user_id, fund_name)
    );
    """
    
    # 2. Fund Positions Table (Temporal tracking of weights)
    positions_schema = """
    CREATE TABLE IF NOT EXISTS fund_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fund_id INTEGER NOT NULL,
        ts_code TEXT NOT NULL,
        weight REAL NOT NULL,
        effective_date DATE NOT NULL,
        end_date DATE, 
        FOREIGN KEY (fund_id) REFERENCES funds(id)
    );
    """
    
    # 3. Fund Daily Risk Metrics 
    metrics_schema = """
    CREATE TABLE IF NOT EXISTS fund_daily_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fund_id INTEGER NOT NULL,
        trade_date DATE NOT NULL,
        total_aum REAL,
        daily_pnl REAL,
        net_flow REAL DEFAULT 0,
        beta_30d REAL,
        var_95 REAL,
        volatility_annualized REAL,
        FOREIGN KEY (fund_id) REFERENCES funds(id),
        UNIQUE(fund_id, trade_date)
    );
    """
    
    # 4. Daily Weight Drift — actual market-value weights vs. target mandate weights
    # is_simulated=1 → retroactive backfill (hypothetical demo history)
    # is_simulated=0 → real forward drift recorded by the nightly rollup
    daily_weights_schema = """
    CREATE TABLE IF NOT EXISTS fund_daily_weights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fund_id INTEGER NOT NULL,
        trade_date DATE NOT NULL,
        ts_code TEXT NOT NULL,
        target_weight REAL NOT NULL,
        actual_weight REAL NOT NULL,
        drift_pct REAL NOT NULL,
        is_simulated INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (fund_id) REFERENCES funds(id),
        UNIQUE(fund_id, trade_date, ts_code)
    );
    """

    # 5. Themes Mapping (Many-to-Many)
    themes_schema = """
    CREATE TABLE IF NOT EXISTS stock_themes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL,
        theme_name TEXT NOT NULL,
        UNIQUE(ts_code, theme_name)
    );
    """

    # 5. Local cache for Total Return architecture (Saves Tushare points)
    cache_schema = """
    CREATE TABLE IF NOT EXISTS stock_daily_cache (
        ts_code TEXT NOT NULL,
        trade_date DATE NOT NULL,
        close REAL NOT NULL,
        pct_chg REAL NOT NULL,
        UNIQUE(ts_code, trade_date)
    );
    """

    # Execute for local SQLite
    if db_config.USE_SQLITE:
        db.create_table_sqlite(funds_schema)
        db.create_table_sqlite(positions_schema)
        db.create_table_sqlite(metrics_schema)
        db.create_table_sqlite(daily_weights_schema)
        db.create_table_sqlite(themes_schema)
        db.create_table_sqlite(cache_schema)

        # ── Column migration: add is_simulated if this is an existing DB ──────
        # ALTER TABLE ADD COLUMN is idempotent here — the except swallows
        # "duplicate column name" from SQLite so re-runs are safe.
        try:
            db.execute_raw_sql(
                "ALTER TABLE fund_daily_weights "
                "ADD COLUMN is_simulated INTEGER NOT NULL DEFAULT 0"
            )
        except Exception:
            pass  # Column already present — nothing to do

        print("✅ Portfolio & TR Cache tables initialized in SQLite.")



def create_margin_tables():
    """Creates margin tables (SQLite only — Supabase needs manual creation)."""
    if db_config.USE_SQLITE:
        # Detail table has rqchl, Market table does NOT.
        schema_detail = """CREATE TABLE IF NOT EXISTS margin_detail (
            ts_code TEXT, trade_date TEXT, rzye REAL, rqye REAL, rzmre REAL, 
            rqyl REAL, rzche REAL, rqchl REAL, rqmcl REAL, rzrqye REAL,
            PRIMARY KEY (ts_code, trade_date)
        )"""
        
        # EXACT match to Tushare 'margin' API
        schema_market = """CREATE TABLE IF NOT EXISTS margin_market (
            exchange_id TEXT, trade_date TEXT, rzye REAL, rzmre REAL, 
            rzche REAL, rqye REAL, rqmcl REAL, rzrqye REAL, rqyl REAL,
            PRIMARY KEY (exchange_id, trade_date)
        )"""
        db.create_table_sqlite(schema_detail)
        db.create_table_sqlite(schema_market)


def update_daily_margin_data():
    """
    Fetches latest margin_detail and margin (market) data from Tushare.
    Includes a "Catch-Up" loop to patch any missing days caused by exchange delays.
    """
    global TUSHARE_API
    if TUSHARE_API is None:
        if not init_tushare():
            return False

    from zoneinfo import ZoneInfo
    from datetime import datetime, timedelta
    import numpy as np
    
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
    now_beijing = datetime.now(CHINA_TZ)
    
    # Determine the absolute latest date we expect to have
    if now_beijing.hour < 12:
        target_date = now_beijing.date() - timedelta(days=1)
    else:
        target_date = now_beijing.date()
        
    create_margin_tables()
    
    # We look back up to 4 days. 
    # If a day is missing (due to 1 AM delays or weekends), we try to fetch it.
    success_any = False
    
    for i in range(4, -1, -1):
        fetch_date = target_date - timedelta(days=i)
        fetch_date_str = fetch_date.strftime('%Y%m%d')
        
        # 1. Check if we already have this date
        try:
            if db.table_exists('margin_market'):
                existing_df = db.read_table('margin_market', filters={'trade_date': fetch_date_str}, limit=1)
                if not existing_df.empty:
                    # We already have this day, silently skip
                    continue
        except Exception as e:
            print(f"[data_manager] ⚠️ Error checking existing margin data: {e}")

        # 2. If we reach here, the date is missing in the DB. Try to fetch it.
        print(f"[data_manager] 📡 Attempting to fetch missing Margin Data for {fetch_date_str}...")

        try:
            # Fetch Market-wide Margin
            df_market = TUSHARE_API.margin(trade_date=fetch_date_str)
            
            # If empty, it's either a weekend, a holiday, or still delayed. 
            # We just move on to the next date in the loop.
            if df_market is None or df_market.empty:
                print(f"[data_manager] ℹ️ No market margin data available for {fetch_date_str}. (Likely weekend/holiday)")
                continue
                
            # Clean and Insert Market Data
            df_market = df_market.replace([np.inf, -np.inf], np.nan)
            records_market = [
                {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()} 
                for row in df_market.to_dict('records')
            ]
            db.insert_records('margin_market', records_market, upsert=True)
            print(f"[data_manager] ✅ Saved Market Margin for {fetch_date_str}: {len(df_market)} records.")

            # Fetch Individual Stock Margin Details
            df_detail = TUSHARE_API.margin_detail(trade_date=fetch_date_str)
            if df_detail is not None and not df_detail.empty:
                df_detail = df_detail.replace([np.inf, -np.inf], np.nan)
                records_detail = [
                    {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()} 
                    for row in df_detail.to_dict('records')
                ]
                db.insert_records('margin_detail', records_detail, upsert=True)
                print(f"[data_manager] ✅ Saved Stock Margin Detail for {fetch_date_str}: {len(df_detail)} records.")
                success_any = True
            
        except Exception as e:
            print(f"[data_manager] ❌ Failed to update margin data for {fetch_date_str}: {e}")
            
        # Quick sleep to respect Tushare API limits inside the loop
        import time
        time.sleep(1.0)

    return success_any
      
def get_stock_margin_history(ticker, limit=250):
    """Retrieves margin history for a single stock."""
    ts_code = get_tushare_ticker(ticker)
    if not db.table_exists('margin_detail'):
        return pd.DataFrame()
        
    df = db.read_table('margin_detail', filters={'ts_code': ts_code}, order_by='-trade_date', limit=limit)
    if not df.empty:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
    return df

def get_market_margin_history(limit=250):
    """Retrieves aggregated market margin history."""
    if not db.table_exists('margin_market'):
        return pd.DataFrame()

    df = db.read_table('margin_market', order_by='-trade_date', limit=limit * 3) # *3 for SSE, SZSE, BSE
    if not df.empty:
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # ADDED 'rqyl' to the list of columns to sum below!
        df_agg = df.groupby('trade_date')[['rzye', 'rzmre', 'rzche', 'rqye', 'rqmcl', 'rzrqye', 'rqyl']].sum()

        return df_agg.sort_index().tail(limit)
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLY CHAIN GRAPH FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# SQLite — table is auto-created on first upsert.
#
# Supabase — run this DDL once in the Supabase SQL editor:
#
#   CREATE TABLE IF NOT EXISTS supply_chain_graphs (
#       ticker       TEXT PRIMARY KEY,
#       company_name TEXT,
#       graph_json   JSONB,
#       last_updated TIMESTAMPTZ DEFAULT NOW()
#   );
#
# ══════════════════════════════════════════════════════════════════════════════

def _create_supply_chain_table():
    """Create supply_chain_graphs table in SQLite (no-op for Supabase)."""
    db.create_table_sqlite("""
        CREATE TABLE IF NOT EXISTS supply_chain_graphs (
            ticker       TEXT PRIMARY KEY,
            company_name TEXT,
            graph_json   TEXT,
            last_updated TEXT
        )
    """)


def supply_chain_graph_exists(ticker: str) -> bool:
    """Return True if a cached graph exists for this ticker."""
    if not db.table_exists('supply_chain_graphs'):
        return False
    result = db.read_table(
        'supply_chain_graphs',
        filters={'ticker': ticker},
        columns='ticker',
        limit=1,
    )
    return not result.empty


def get_supply_chain_graph(ticker: str):
    """Return the cached graph as a dict, or None if not found."""
    import json as _json
    if not db.table_exists('supply_chain_graphs'):
        return None
    df = db.read_table('supply_chain_graphs', filters={'ticker': ticker}, limit=1)
    if df.empty:
        return None
    raw = df.iloc[0].get('graph_json')
    if not raw:
        return None
    try:
        return _json.loads(raw)
    except Exception:
        return None


def upsert_supply_chain_graph(ticker: str, company_name: str, graph_data: dict) -> bool:
    """Insert or update the supply chain graph for a ticker. Returns True on success."""
    import json as _json
    from datetime import datetime as _dt
    if not db.table_exists('supply_chain_graphs'):
        _create_supply_chain_table()
    try:
        db.insert_records(
            'supply_chain_graphs',
            {
                'ticker':       ticker,
                'company_name': company_name,
                'graph_json':   _json.dumps(graph_data, ensure_ascii=False),
                'last_updated': _dt.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            upsert=True,
        )
        return True
    except Exception as e:
        print(f"[data_manager] ❌ upsert_supply_chain_graph({ticker}): {e}")
        return False


def get_all_supply_chain_tickers() -> set:
    """Return the set of tickers that already have a cached supply chain graph."""
    if not db.table_exists('supply_chain_graphs'):
        return set()
    df = db.read_table('supply_chain_graphs', columns='ticker')
    return set(df['ticker'].tolist()) if not df.empty else set()


# ══════════════════════════════════════════════════════════════════════════════
# MACRO SECTOR THEMES
# ══════════════════════════════════════════════════════════════════════════════
#
# Stores chronological supply-chain maps generated from rough industry themes
# ("ev", "data center", etc.) by DeepSeek.
#
# SQLite — auto-created on first insert.
#
# Supabase — run this DDL once in the Supabase SQL editor:
#
#   CREATE TABLE IF NOT EXISTS sector_themes (
#       id          BIGSERIAL PRIMARY KEY,
#       raw_input   TEXT NOT NULL UNIQUE,
#       formal_name TEXT NOT NULL,
#       layers_json JSONB NOT NULL,
#       created_by  TEXT,
#       created_at  TIMESTAMPTZ DEFAULT NOW()
#   );
#
# raw_input is UNIQUE so we can dedupe API calls and provide an audit trail.
# ══════════════════════════════════════════════════════════════════════════════

def _create_sector_themes_table():
    """Create sector_themes in SQLite (no-op for Supabase — run DDL above)."""
    db.create_table_sqlite("""
        CREATE TABLE IF NOT EXISTS sector_themes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_input   TEXT NOT NULL UNIQUE,
            formal_name TEXT NOT NULL,
            layers_json TEXT NOT NULL,
            created_by  TEXT,
            created_at  TEXT
        )
    """)


def get_all_sector_themes():
    """Return [{'id', 'raw_input', 'formal_name'}, ...] sorted by formal_name."""
    if not db.table_exists('sector_themes'):
        return []
    df = db.read_table(
        'sector_themes',
        columns='id,raw_input,formal_name',
        order_by='formal_name',
    )
    return df.to_dict('records') if not df.empty else []


def get_sector_theme_by_id(theme_id):
    """Return the full theme record incl. parsed 'layers' list, or None."""
    import json as _json
    if not db.table_exists('sector_themes'):
        return None
    df = db.read_table('sector_themes', filters={'id': theme_id}, limit=1)
    if df.empty:
        return None
    rec = df.iloc[0].to_dict()
    raw = rec.get('layers_json', '')
    try:
        full = _json.loads(raw) if isinstance(raw, str) else (raw or {})
        rec['data']   = full
        rec['layers'] = full.get('layers', []) if isinstance(full, dict) else []
    except Exception:
        rec['data']   = {}
        rec['layers'] = []
    return rec


def get_sector_theme_by_raw_input(raw_input):
    """Existence check by raw_input (case-insensitive on the stored value)."""
    if not db.table_exists('sector_themes'):
        return None
    key = (raw_input or '').strip().lower()
    df = db.read_table('sector_themes', filters={'raw_input': key}, limit=1)
    return df.iloc[0].to_dict() if not df.empty else None


def add_sector_theme(raw_input, formal_name, layers_data, created_by=None):
    """Insert a new theme. raw_input is normalised to lowercase. Returns True on success."""
    import json as _json
    from datetime import datetime as _dt
    if not db.table_exists('sector_themes'):
        _create_sector_themes_table()
    try:
        db.insert_records('sector_themes', {
            'raw_input':   (raw_input or '').strip().lower(),
            'formal_name': formal_name,
            'layers_json': _json.dumps(layers_data, ensure_ascii=False),
            'created_by':  created_by,
            'created_at':  _dt.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, upsert=False)
        return True
    except Exception as e:
        print(f"[data_manager] ❌ add_sector_theme: {e}")
        return False


def delete_sector_theme(theme_id):
    """Delete a theme by id. Returns True on success."""
    if not db.table_exists('sector_themes'):
        return False
    try:
        db.delete_records('sector_themes', filters={'id': theme_id})
        return True
    except Exception as e:
        print(f"[data_manager] ❌ delete_sector_theme({theme_id}): {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# CHAIN POSITIONS — maps (ticker, theme_id) → layer the company sits in
# ══════════════════════════════════════════════════════════════════════════════

def ensure_chain_positions_table() -> None:
    """Idempotent migration for SQLite; Supabase users add the table manually."""
    from db_config import USE_SQLITE
    if not USE_SQLITE:
        return
    from db_config import DBNAME
    import sqlite3
    with sqlite3.connect(DBNAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chain_positions (
                ticker         TEXT    NOT NULL,
                theme_id       INTEGER NOT NULL,
                layer_index    INTEGER,
                matched_items  TEXT    NOT NULL DEFAULT '[]',
                generated_at   TEXT    NOT NULL,
                PRIMARY KEY (ticker, theme_id)
            )
        """)
        conn.commit()


def get_chain_position(ticker: str, theme_id: int) -> dict | None:
    """Return the cached chain position for (ticker, theme_id), or None."""
    import json as _json
    if not db.table_exists("chain_positions"):
        return None
    df = db.read_table(
        "chain_positions",
        filters={"ticker": ticker, "theme_id": int(theme_id)},
        limit=1,
    )
    if df is None or df.empty:
        return None
    row = df.iloc[0].to_dict()
    try:
        row["matched_items"] = _json.loads(row.get("matched_items") or "[]")
    except Exception:
        row["matched_items"] = []
    return row


def upsert_chain_position(
    ticker: str,
    theme_id: int,
    layer_index: int | None,
    matched_items: list,
) -> None:
    """Insert or replace the chain position for (ticker, theme_id)."""
    import json as _json
    from datetime import datetime as _dt
    if not db.table_exists("chain_positions"):
        ensure_chain_positions_table()
    db.delete_records("chain_positions",
                      filters={"ticker": ticker, "theme_id": int(theme_id)})
    db.insert_records("chain_positions", [{
        "ticker":        ticker,
        "theme_id":      int(theme_id),
        "layer_index":   layer_index,
        "matched_items": _json.dumps(matched_items, ensure_ascii=False),
        "generated_at":  _dt.utcnow().isoformat(),
    }])


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT PEERS (Phase 1 of lead-lag analysis)
# ══════════════════════════════════════════════════════════════════════════════
#
# Cache of {product_name → list of A-share peer tickers that produce it}, sourced
# from DeepSeek. Reused across Stages 1 & 3 of the lead-lag pipeline so we only
# call DeepSeek once per unique product string.
#
# Supabase DDL (run once):
#   CREATE TABLE IF NOT EXISTS product_peers (
#       product_key   TEXT PRIMARY KEY,
#       display_name  TEXT NOT NULL,
#       peers_json    JSONB NOT NULL,
#       source_method TEXT,
#       last_updated  TIMESTAMPTZ DEFAULT NOW()
#   );
# ══════════════════════════════════════════════════════════════════════════════

def _create_product_peers_table():
    db.create_table_sqlite("""
        CREATE TABLE IF NOT EXISTS product_peers (
            product_key   TEXT PRIMARY KEY,
            display_name  TEXT NOT NULL,
            peers_json    TEXT NOT NULL,
            source_method TEXT,
            last_updated  TEXT
        )
    """)


def _normalize_product_key(display_name):
    return (display_name or "").strip().lower()


def get_product_peers(display_name):
    """Return cached {display_name, peers, source_method, last_updated} or None."""
    import json as _json
    if not db.table_exists('product_peers'):
        return None
    key = _normalize_product_key(display_name)
    if not key:
        return None
    df = db.read_table('product_peers', filters={'product_key': key}, limit=1)
    if df.empty:
        return None
    rec = df.iloc[0].to_dict()
    raw = rec.get('peers_json', '[]')
    try:
        rec['peers'] = _json.loads(raw) if isinstance(raw, str) else (raw or [])
    except Exception:
        rec['peers'] = []
    return rec


def upsert_product_peers(display_name, peers, source_method='deepseek'):
    """Insert or update peers for a product. peers = list of {ticker, name} dicts."""
    import json as _json
    from datetime import datetime as _dt
    if not db.table_exists('product_peers'):
        _create_product_peers_table()
    try:
        db.insert_records('product_peers', {
            'product_key':   _normalize_product_key(display_name),
            'display_name':  display_name,
            'peers_json':    _json.dumps(peers, ensure_ascii=False),
            'source_method': source_method,
            'last_updated':  _dt.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, upsert=True)
        return True
    except Exception as e:
        print(f"[data_manager] ❌ upsert_product_peers: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Stock validation & adjusted price helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_all_stock_basic() -> list:
    """
    Return every stock in stock_basic as [{ticker, name}] sorted by ticker.
    Used to populate the full-list selectbox (client-side filtering, no reruns).
    Returns [] if the table doesn't exist or is empty.
    """
    if not db.table_exists("stock_basic"):
        return []
    try:
        df = db.read_table("stock_basic", columns="symbol,name", order_by="symbol")
        if df.empty:
            return []
        return [
            {"ticker": str(r["symbol"]).strip(), "name": str(r["name"]).strip()}
            for _, r in df.iterrows()
            if r.get("symbol") and r.get("name")
        ]
    except Exception as exc:
        print(f"[data_manager] get_all_stock_basic error: {exc}")
        return []


def search_stock_basic(query: str, limit: int = 20) -> list:
    """
    Search stock_basic by ticker prefix OR company name contains (case-insensitive).

    Returns [{ticker: '6-digit', name: 'company name'}] sorted by relevance:
      0 — exact ticker match
      1 — ticker starts with query
      2 — name contains query

    Works for both SQLite (LIKE query) and Supabase (Python-side filter).
    """
    query = (query or "").strip()
    if not query or not db.table_exists("stock_basic"):
        return []

    try:
        if db.use_supabase:
            # Supabase: load symbol+name for all ~5 k stocks and filter in Python
            df = db.read_table("stock_basic", columns="symbol,name")
            if df.empty:
                return []
            q_low = query.lower()
            scored = []
            for _, row in df.iterrows():
                sym  = str(row.get("symbol") or "").strip()
                name = str(row.get("name")   or "").strip()
                if sym == query:
                    scored.append((0, sym, name))
                elif query.isdigit() and sym.startswith(query):
                    scored.append((1, sym, name))
                elif q_low in name.lower():
                    scored.append((2, sym, name))
            scored.sort(key=lambda x: (x[0], x[1]))
            return [{"ticker": s, "name": n} for _, s, n in scored[:limit]]
        else:
            # SQLite: parameterized LIKE with inline relevance score
            import sqlite3 as _sqlite3
            sql = """
                SELECT symbol, name,
                       CASE WHEN symbol = ?           THEN 0
                            WHEN symbol LIKE ?        THEN 1
                            ELSE                           2
                       END AS _score
                FROM   stock_basic
                WHERE  symbol LIKE ?
                   OR  LOWER(name) LIKE LOWER(?)
                ORDER  BY _score, symbol
                LIMIT  ?
            """
            params = (query, f"{query}%", f"{query}%", f"%{query}%", limit)
            with _sqlite3.connect(db.dbname) as conn:
                rows = conn.execute(sql, params).fetchall()
            return [{"ticker": row[0], "name": row[1]} for row in rows]
    except Exception as exc:
        print(f"[data_manager] search_stock_basic error: {exc}")
        return []


def validate_tickers_against_stock_basic(tickers: list) -> dict:
    """
    Check each 6-digit ticker against the local stock_basic table.

    Returns {ticker: {"valid": bool, "official_name": str|None, "ts_code": str|None}}.
    Useful for verifying AI-generated ticker lists before displaying or trading.
    """
    result = {}
    table_ok = db.table_exists("stock_basic")

    for t in tickers:
        if not table_ok:
            result[t] = {"valid": False, "official_name": None, "ts_code": None}
            continue
        ts_code = get_tushare_ticker(t)
        try:
            df = db.read_table(
                "stock_basic",
                filters={"ts_code": ts_code},
                columns="ts_code,name",
                limit=1,
            )
            if not df.empty:
                result[t] = {
                    "valid":         True,
                    "official_name": df.iloc[0]["name"],
                    "ts_code":       df.iloc[0]["ts_code"],
                }
            else:
                result[t] = {"valid": False, "official_name": None, "ts_code": ts_code}
        except Exception:
            result[t] = {"valid": False, "official_name": None, "ts_code": None}

    return result


def fetch_adjusted_daily(ticker: str, start_date: str, end_date: str,
                         adj: str = "qfq") -> "pd.DataFrame":
    """
    Fetch forward-adjusted (qfq) daily close prices for a single ticker.

    Returns DataFrame with columns [trade_date, close] sorted ascending,
    or an empty DataFrame on failure.

    Tries ts.pro_bar (adjusted) first; falls back to pro.daily (raw) if
    pro_bar is unavailable or fails.
    """
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()

    ts_code = get_tushare_ticker(ticker)

    # ── Primary: qfq-adjusted via ts.pro_bar ─────────────────────────────────
    try:
        import tushare as _ts
        df = _ts.pro_bar(
            ts_code=ts_code,
            api=TUSHARE_API,
            adj=adj,
            start_date=start_date,
            end_date=end_date,
            freq="D",
        )
        if df is not None and not df.empty:
            return (
                df[["trade_date", "close"]]
                .sort_values("trade_date")
                .reset_index(drop=True)
            )
    except Exception:
        pass

    # ── Fallback: raw daily (no split/dividend adjustment) ────────────────────
    try:
        df = TUSHARE_API.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="trade_date,close",
        )
        if df is not None and not df.empty:
            return (
                df[["trade_date", "close"]]
                .sort_values("trade_date")
                .reset_index(drop=True)
            )
    except Exception as e:
        print(f"[data_manager] fetch_adjusted_daily({ticker}): {e}")

    import pandas as _pd
    return _pd.DataFrame()


# ============================================================
# ADMIN: SECTOR-STOCK MAP (dynamic, DB-backed)
# ============================================================

def ensure_admin_tables_exist():
    """
    Create admin tables if they don't exist.
    SQLite: created automatically.
    Supabase: print SQL for manual creation in dashboard.
    """
    schemas = {
        'sector_stock_map': """
            CREATE TABLE IF NOT EXISTS sector_stock_map (
                sector TEXT NOT NULL,
                ticker TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                added_at TEXT,
                removed_at TEXT,
                PRIMARY KEY (sector, ticker)
            )""",
        'rebuild_jobs': """
            CREATE TABLE IF NOT EXISTS rebuild_jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT,
                sectors TEXT,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                progress_message TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT
            )""",
    }

    if db.use_supabase:
        missing = [name for name in schemas if not db.table_exists(name)]
        if missing:
            print("⚠️  Supabase: create these tables in the dashboard:")
            for name in missing:
                print(f"   - {name}")
        return

    for name, schema in schemas.items():
        if not db.table_exists(name):
            db.create_table_sqlite(schema)
            print(f"✅ Created table: {name}")


def seed_sector_stock_map():
    """Seed sector_stock_map from hardcoded SECTOR_STOCK_MAP if the table is empty."""
    ensure_admin_tables_exist()
    try:
        existing = db.read_table('sector_stock_map', limit=1)
        if not existing.empty:
            return
    except Exception:
        pass

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    records = []
    for sector, tickers in SECTOR_STOCK_MAP.items():
        for ticker in tickers:
            records.append({
                'sector': sector,
                'ticker': ticker,
                'is_active': 1,
                'added_at': now,
                'removed_at': None,
            })

    db.insert_records('sector_stock_map', records)
    print(f"✅ Seeded sector_stock_map: {len(records)} entries across {len(SECTOR_STOCK_MAP)} sectors")


def get_sector_stock_map():
    """
    Load active sector-to-stock mapping from DB.
    Falls back to hardcoded SECTOR_STOCK_MAP if DB is unavailable or empty.
    """
    try:
        ensure_admin_tables_exist()
        seed_sector_stock_map()
        df = db.read_table('sector_stock_map', filters={'is_active': 1})
        if df.empty:
            return SECTOR_STOCK_MAP.copy()
        result = {}
        for _, row in df.iterrows():
            result.setdefault(row['sector'], []).append(row['ticker'])
        return result
    except Exception as e:
        print(f"⚠️  get_sector_stock_map failed ({e}), using hardcoded map")
        return SECTOR_STOCK_MAP.copy()


def add_stock_to_sector(sector, ticker):
    """Add (or reactivate) a stock in a sector."""
    ensure_admin_tables_exist()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    existing = db.read_table('sector_stock_map', filters={'sector': sector, 'ticker': ticker})
    if not existing.empty:
        db.update_records('sector_stock_map',
                          {'is_active': 1, 'added_at': now, 'removed_at': None},
                          {'sector': sector, 'ticker': ticker})
        print(f"✅ Reactivated {ticker} in sector '{sector}'")
    else:
        db.insert_records('sector_stock_map', [{
            'sector': sector, 'ticker': ticker,
            'is_active': 1, 'added_at': now, 'removed_at': None,
        }])
        print(f"✅ Added {ticker} to sector '{sector}'")


def remove_stock_from_sector(sector, ticker):
    """Soft-delete a stock from a sector (marks is_active=0)."""
    ensure_admin_tables_exist()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db.update_records('sector_stock_map',
                      {'is_active': 0, 'removed_at': now},
                      {'sector': sector, 'ticker': ticker})
    print(f"✅ Removed {ticker} from sector '{sector}'")


def add_new_sector(sector_name, tickers):
    """Create a new sector and seed it with an initial stock list."""
    ensure_admin_tables_exist()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    records = [{'sector': sector_name, 'ticker': t,
                'is_active': 1, 'added_at': now, 'removed_at': None}
               for t in tickers]
    db.insert_records('sector_stock_map', records)
    print(f"✅ Created sector '{sector_name}' with {len(tickers)} stocks")


def get_all_sector_stock_map_raw():
    """Return full sector_stock_map table (active + inactive) as DataFrame."""
    ensure_admin_tables_exist()
    return db.read_table('sector_stock_map', order_by='sector')


# ============================================================
# ADMIN: REGIME SCORES (DB-backed, replaces CSV)
# ============================================================

# ============================================================
# ADMIN: REBUILD JOBS
# ============================================================

def create_rebuild_job(job_type, sectors):
    """Insert a new rebuild_jobs row and return the job_id."""
    import uuid, json
    ensure_admin_tables_exist()
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db.insert_records('rebuild_jobs', [{
        'job_id': job_id,
        'job_type': job_type,
        'sectors': '__all__' if sectors == '__all__' else json.dumps(sectors, ensure_ascii=False),
        'status': 'pending',
        'progress': 0,
        'progress_message': 'Job created',
        'created_at': now,
        'started_at': None,
        'completed_at': None,
        'error_message': None,
    }])
    return job_id


def update_rebuild_job(job_id, **kwargs):
    """Patch a rebuild_jobs row with arbitrary field updates."""
    if not kwargs:
        return
    try:
        db.update_records('rebuild_jobs', kwargs, {'job_id': job_id})
    except Exception as e:
        print(f"⚠️  update_rebuild_job({job_id}): {e}")


def get_rebuild_job(job_id):
    """Return a rebuild_jobs row as a dict, or None."""
    df = db.read_table('rebuild_jobs', filters={'job_id': job_id})
    return df.iloc[0].to_dict() if not df.empty else None


def get_recent_rebuild_jobs(limit=15):
    """Return the most recent rebuild_jobs rows as a DataFrame."""
    ensure_admin_tables_exist()
    return db.read_table('rebuild_jobs', order_by='-created_at', limit=limit)


# ── Equity Brief — Financial statement fetchers ───────────────────────────────

def _start_for_periods(periods):
    """Return (start_date, end_date) covering ~periods quarters of history."""
    from datetime import datetime as _dt, timedelta as _td
    end = _dt.now()
    start = end - _td(days=(periods + 1) * 95)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def fetch_income_statement(ticker: str, periods: int = 8) -> "pd.DataFrame":
    """
    Fetch consolidated income statement from Tushare income().

    Returns DataFrame sorted by end_date DESC (most recent first), trimmed to
    periods rows. Empty DataFrame on any failure.
    """
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()
    ts_code = get_tushare_ticker(ticker)
    start, end = _start_for_periods(periods)
    try:
        df = TUSHARE_API.income(
            ts_code=ts_code, start_date=start, end_date=end,
            fields=("ts_code,end_date,report_type,total_revenue,revenue,oper_cost,"
                    "operate_profit,total_profit,n_income,n_income_attr_p,ebit,ebitda,"
                    "basic_eps,diluted_eps,sell_exp,admin_exp,fin_exp,interest_exp,rd_exp"),
        )
        if df is None or df.empty:
            return _pd.DataFrame()
        # Keep consolidated reports only (report_type==1) when present
        if "report_type" in df.columns:
            df = df[df["report_type"].astype(str) == "1"]
        df = df.drop_duplicates(subset=["end_date"]).sort_values(
            "end_date", ascending=False
        ).head(periods).reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"[data_manager] fetch_income_statement({ticker}): {exc}")
        return _pd.DataFrame()


def fetch_balance_sheet(ticker: str, periods: int = 8) -> "pd.DataFrame":
    """Fetch consolidated balance sheet from Tushare balancesheet()."""
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()
    ts_code = get_tushare_ticker(ticker)
    start, end = _start_for_periods(periods)
    try:
        df = TUSHARE_API.balancesheet(
            ts_code=ts_code, start_date=start, end_date=end,
            fields=("ts_code,end_date,report_type,total_assets,total_liab,total_hldr_eqy_inc_min_int,"
                    "total_cur_assets,total_cur_liab,money_cap,inventories,accounts_receiv,"
                    "st_borr,lt_borr,bond_payable,non_cur_liab_due_1y,oth_pay_total,total_share"),
        )
        if df is None or df.empty:
            return _pd.DataFrame()
        if "report_type" in df.columns:
            df = df[df["report_type"].astype(str) == "1"]
        df = df.drop_duplicates(subset=["end_date"]).sort_values(
            "end_date", ascending=False
        ).head(periods).reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"[data_manager] fetch_balance_sheet({ticker}): {exc}")
        return _pd.DataFrame()


def fetch_cashflow(ticker: str, periods: int = 8) -> "pd.DataFrame":
    """Fetch consolidated cash-flow statement from Tushare cashflow()."""
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()
    ts_code = get_tushare_ticker(ticker)
    start, end = _start_for_periods(periods)
    try:
        df = TUSHARE_API.cashflow(
            ts_code=ts_code, start_date=start, end_date=end,
            fields=("ts_code,end_date,report_type,n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,"
                    "c_paid_for_assets,free_cashflow,depr_fa_coga_dpba,amort_intang_assets"),
        )
        if df is None or df.empty:
            return _pd.DataFrame()
        if "report_type" in df.columns:
            df = df[df["report_type"].astype(str) == "1"]
        df = df.drop_duplicates(subset=["end_date"]).sort_values(
            "end_date", ascending=False
        ).head(periods).reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"[data_manager] fetch_cashflow({ticker}): {exc}")
        return _pd.DataFrame()


def fetch_full_fina_indicator(ticker: str, periods: int = 8) -> "pd.DataFrame":
    """
    Wider field-set version of fina_indicator for the Equity Brief.
    Returns DESC by end_date, capped at `periods` rows.
    """
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()
    ts_code = get_tushare_ticker(ticker)
    start, end = _start_for_periods(periods)
    try:
        df = TUSHARE_API.fina_indicator(
            ts_code=ts_code, start_date=start, end_date=end,
            fields=("ts_code,end_date,roe,roa,grossprofit_margin,netprofit_margin,"
                    "q_netprofit_margin,debt_to_assets,current_ratio,quick_ratio,"
                    "assets_turn,or_yoy,netprofit_yoy,basic_eps_yoy,equity_yoy,"
                    "fcff,fcfe,ocf_to_or"),
        )
        if df is None or df.empty:
            return _pd.DataFrame()
        df = df.drop_duplicates(subset=["end_date"]).sort_values(
            "end_date", ascending=False
        ).head(periods).reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"[data_manager] fetch_full_fina_indicator({ticker}): {exc}")
        return _pd.DataFrame()


def get_latest_daily_basic(ticker):
    """
    Return the most recent daily_basic row as a dict (close, pe, pb, ps, total_mv,
    circ_mv, dv_ratio, etc.). None on any failure.
    """
    import pandas as _pd
    if not init_tushare():
        return None
    ts_code = get_tushare_ticker(ticker)
    from datetime import datetime as _dt, timedelta as _td
    end_date   = _dt.now().strftime("%Y%m%d")
    start_date = (_dt.now() - _td(days=15)).strftime("%Y%m%d")
    try:
        df = TUSHARE_API.daily_basic(
            ts_code=ts_code, start_date=start_date, end_date=end_date,
            fields="ts_code,trade_date,close,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,total_mv,circ_mv,total_share,turnover_rate",
        )
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date", ascending=False)
        row = df.iloc[0].to_dict()
        # Tushare market caps come in 万元 (10,000 RMB) — convert to 亿元 (100M)
        for k in ("total_mv", "circ_mv"):
            if row.get(k) is not None:
                row[k] = float(row[k]) / 10_000  # 万 → 亿
        return row
    except Exception as exc:
        print(f"[data_manager] get_latest_daily_basic({ticker}): {exc}")
        return None


def compute_derived_metrics(income_df, balance_df, cashflow_df, daily_basic):
    """
    Compute Equity Brief metrics that Tushare doesn't provide directly.

    Returns a dict with: ev_yi (亿元), ev_ebitda, p_fcf, fcf_yield_pct,
    debt_to_equity, net_debt_yi, net_debt_ebitda, interest_coverage,
    current_ratio, gross_margin_ttm_pct, net_margin_ttm_pct.
    Any metric that can't be computed is None.

    ── TTM methodology ──────────────────────────────────────────────────────────
    Tushare income/cashflow data is CUMULATIVE YTD within each fiscal year, not
    discrete quarters. Naively summing the last 4 rows would double-count:
      e.g. Q4 = full year, Q3 = Jan-Sep, Q2 = Jan-Jun, Q1 = Jan-Mar
      → sum = 4× Jan-Mar + 3× Apr-Jun + 2× Jul-Sep + 1× Oct-Dec

    Correct TTM formula for cumulative data:
      TTM = Latest_YTD + Prev_Year_End − Prev_Year_Same_Period
      e.g. if latest = Q1 2026:
        TTM = Q1_2026 + FY_2025 − Q1_2025  (= Apr-2025 to Mar-2026) ✓

    If the latest report IS the full-year annual, TTM = annual value (no adj).
    Falls back to latest YTD alone if the anchor rows aren't in the window.
    ─────────────────────────────────────────────────────────────────────────────
    """
    import pandas as _pd

    out = {
        "ev_yi": None, "ev_ebitda": None, "p_fcf": None, "fcf_yield_pct": None,
        "debt_to_equity": None, "net_debt_yi": None, "net_debt_ebitda": None,
        "interest_coverage": None, "current_ratio": None,
        "gross_margin_ttm_pct": None, "net_margin_ttm_pct": None,
    }

    if not daily_basic or income_df is None or income_df.empty:
        return out

    mv_yi = daily_basic.get("total_mv")   # already in 亿元
    if mv_yi is None:
        return out

    # ── TTM helper ────────────────────────────────────────────────────────────
    def _ttm(df, col):
        """
        Trailing-twelve-month value for one column of a cumulative YTD DataFrame.
        Formula: Latest_YTD + Prev_Year_End - Prev_Year_Same_Period
        Returns float (0.0 if data is absent).
        """
        if df is None or df.empty or col not in df.columns:
            return 0.0

        # Work on a clean, date-sorted copy (DESC)
        d = df[["end_date", col]].copy()
        d["_end"] = d["end_date"].astype(str)
        d = d.sort_values("_end", ascending=False).reset_index(drop=True)

        latest_date = d.iloc[0]["_end"]
        latest_val  = float(d.iloc[0][col] if _pd.notna(d.iloc[0][col]) else 0)

        # Annual report: no adjustment required
        if latest_date.endswith("1231"):
            return latest_val

        # Look up the two anchor rows
        latest_year  = int(latest_date[:4])
        prev_annual_key = f"{latest_year - 1}1231"
        prev_same_key   = f"{latest_year - 1}{latest_date[4:]}"

        def _lookup(key):
            rows = d[d["_end"] == key]
            if rows.empty:
                return None
            v = rows.iloc[0][col]
            return float(v) if _pd.notna(v) else 0.0

        prev_annual = _lookup(prev_annual_key)
        prev_same   = _lookup(prev_same_key)

        if prev_annual is None or prev_same is None:
            # Anchor rows missing — fall back to latest YTD as-is
            return latest_val

        return latest_val + prev_annual - prev_same

    # ── Balance sheet — point-in-time, no TTM ────────────────────────────────
    bal0 = balance_df.iloc[0] if (balance_df is not None and not balance_df.empty) else None

    if bal0 is not None:
        # Comprehensive financial debt:
        #   st_borr           = 短期借款
        #   lt_borr           = 长期借款
        #   bond_payable      = 应付债券
        #   non_cur_liab_due_1y = 一年内到期的非流动负债 (LT debt maturing soon)
        debt_fields = ("st_borr", "lt_borr", "bond_payable", "non_cur_liab_due_1y")
        debt = sum(float(bal0.get(c) or 0) for c in debt_fields)
        cash = float(bal0.get("money_cap") or 0)
        net_debt_yuan = debt - cash
        out["net_debt_yi"] = round(net_debt_yuan / 1e8, 2)
        out["ev_yi"]       = round(mv_yi + (net_debt_yuan / 1e8), 2)

        equity = float(bal0.get("total_hldr_eqy_inc_min_int") or 0)
        if equity > 0:
            out["debt_to_equity"] = round(debt / equity, 2)

        cur_assets = float(bal0.get("total_cur_assets") or 0)
        cur_liab   = float(bal0.get("total_cur_liab") or 0)
        if cur_liab > 0:
            out["current_ratio"] = round(cur_assets / cur_liab, 2)

    # ── TTM Income metrics ────────────────────────────────────────────────────
    rev_ttm  = _ttm(income_df, "total_revenue")
    cost_ttm = _ttm(income_df, "oper_cost")
    ni_ttm   = _ttm(income_df, "n_income_attr_p")
    ebit_ttm = _ttm(income_df, "ebit")

    if rev_ttm > 0:
        out["gross_margin_ttm_pct"] = round((rev_ttm - cost_ttm) / rev_ttm * 100, 2)
        out["net_margin_ttm_pct"]   = round(ni_ttm / rev_ttm * 100, 2)

    # ── TTM EBITDA ────────────────────────────────────────────────────────────
    ebitda_ttm = 0.0
    if "ebitda" in income_df.columns and income_df["ebitda"].notna().any():
        ebitda_ttm = _ttm(income_df, "ebitda")
    else:
        # EBITDA = EBIT + D&A (depreciation from cashflow statement)
        da_ttm = 0.0
        if cashflow_df is not None and not cashflow_df.empty:
            da_ttm = (_ttm(cashflow_df, "depr_fa_coga_dpba")
                      + _ttm(cashflow_df, "amort_intang_assets"))
        ebitda_ttm = ebit_ttm + da_ttm

    # EV/EBITDA and Net Debt/EBITDA are N/A when EBITDA ≤ 0
    # (negative multiples are not meaningful valuation metrics)
    if ebitda_ttm > 0 and out["ev_yi"] is not None:
        out["ev_ebitda"] = round((out["ev_yi"] * 1e8) / ebitda_ttm, 2)
    if ebitda_ttm > 0 and out["net_debt_yi"] is not None:
        out["net_debt_ebitda"] = round((out["net_debt_yi"] * 1e8) / ebitda_ttm, 2)

    # ── Interest coverage: EBIT / interest cost ───────────────────────────────
    # Prefer interest_exp (gross interest paid) over fin_exp (net financial cost
    # = interest expense − interest income). fin_exp can be negative for cash-
    # rich companies, which would produce a nonsensical coverage ratio.
    int_exp_ttm = None
    if "interest_exp" in income_df.columns:
        v = _ttm(income_df, "interest_exp")
        if v > 0:
            int_exp_ttm = v
    if int_exp_ttm is None:
        # Fallback: use fin_exp only when it is net-positive (company is a net
        # interest payer, so the sign is meaningful as a cost denominator).
        v = _ttm(income_df, "fin_exp")
        if v > 0:
            int_exp_ttm = v
    if int_exp_ttm and ebit_ttm:
        out["interest_coverage"] = round(ebit_ttm / int_exp_ttm, 2)

    # ── TTM Free Cash Flow ────────────────────────────────────────────────────
    if cashflow_df is not None and not cashflow_df.empty:
        if ("free_cashflow" in cashflow_df.columns
                and cashflow_df["free_cashflow"].notna().any()):
            fcf_ttm = _ttm(cashflow_df, "free_cashflow")
        else:
            ocf_ttm   = _ttm(cashflow_df, "n_cashflow_act")
            capex_ttm = _ttm(cashflow_df, "c_paid_for_assets")
            fcf_ttm   = ocf_ttm - capex_ttm

        if mv_yi > 0:
            # FCF yield is meaningful even when negative (signals cash burn)
            out["fcf_yield_pct"] = round((fcf_ttm / 1e8) / mv_yi * 100, 2)
            # P/FCF is only meaningful when FCF > 0
            if fcf_ttm > 0:
                out["p_fcf"] = round((mv_yi * 1e8) / fcf_ttm, 2)

    return out


def fetch_forecast(ticker, periods=8):
    """
    Tushare forecast (业绩预告): preliminary earnings warning released
    BEFORE the actual quarterly report. Contains the company's own
    narrative on why earnings moved.

    Returns DataFrame DESC by ann_date. Empty DataFrame on any failure.
    Key fields: ann_date, end_date, type, p_change_min, p_change_max,
    net_profit_min, net_profit_max, summary, change_reason.
    """
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()
    ts_code = get_tushare_ticker(ticker)
    start, end = _start_for_periods(periods + 4)  # widen — forecasts span 1-2 yrs ahead
    try:
        df = TUSHARE_API.forecast(
            ts_code=ts_code, start_date=start, end_date=end,
            fields=("ts_code,ann_date,end_date,type,p_change_min,p_change_max,"
                    "net_profit_min,net_profit_max,last_parent_net,"
                    "first_ann_date,summary,change_reason"),
        )
        if df is None or df.empty:
            return _pd.DataFrame()
        return (df.drop_duplicates(subset=["ann_date", "end_date"])
                  .sort_values("ann_date", ascending=False)
                  .head(periods)
                  .reset_index(drop=True))
    except Exception as exc:
        print(f"[data_manager] fetch_forecast({ticker}): {exc}")
        return _pd.DataFrame()


def fetch_express(ticker, periods=4):
    """
    Tushare express (业绩快报): full preliminary results released AFTER
    quarter-end but BEFORE the formal 10-Q-equivalent. Contains revenue,
    profits, balance-sheet headlines for the period.

    Returns DataFrame DESC by ann_date. Empty DataFrame on any failure.
    """
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()
    ts_code = get_tushare_ticker(ticker)
    start, end = _start_for_periods(periods + 2)
    try:
        df = TUSHARE_API.express(
            ts_code=ts_code, start_date=start, end_date=end,
            fields=("ts_code,ann_date,end_date,revenue,operate_profit,"
                    "total_profit,n_income,total_assets,"
                    "total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int,"
                    "yoy_sales,yoy_op,yoy_tp,yoy_dedu_np,yoy_eps,yoy_roe,"
                    "bps,eps,diluted_eps,diluted_roe"),
        )
        if df is None or df.empty:
            return _pd.DataFrame()
        return (df.drop_duplicates(subset=["ann_date", "end_date"])
                  .sort_values("ann_date", ascending=False)
                  .head(periods)
                  .reset_index(drop=True))
    except Exception as exc:
        print(f"[data_manager] fetch_express({ticker}): {exc}")
        return _pd.DataFrame()


def fetch_fina_mainbz(ticker: str, bz_type: str = "P") -> "pd.DataFrame":
    """
    Fetch main-business revenue breakdown from Tushare fina_mainbz.

    bz_type="P"  →  product-level breakdown (products as rows)
    bz_type="D"  →  department/segment breakdown

    Returns a DataFrame with columns:
        ts_code, end_date, bz_item, bz_sales, bz_profit, bz_cost, curr_type
    sorted by end_date descending (most recent period first).
    Returns an empty DataFrame on any failure.
    """
    import pandas as _pd
    if not init_tushare():
        return _pd.DataFrame()

    ts_code = get_tushare_ticker(ticker)
    try:
        df = TUSHARE_API.fina_mainbz(
            ts_code=ts_code,
            type=bz_type,
            fields="ts_code,end_date,bz_item,bz_sales,bz_profit,bz_cost,curr_type",
        )
        if df is None or df.empty:
            return _pd.DataFrame()

        for col in ("bz_sales", "bz_profit", "bz_cost"):
            df[col] = _pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df = df.sort_values("end_date", ascending=False).reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"[data_manager] fetch_fina_mainbz({ticker}): {exc}")
        return _pd.DataFrame()

