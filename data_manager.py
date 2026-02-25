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

    if sector_start_dates is None:
        sector_start_dates = {sector: None for sector in SECTOR_STOCK_MAP.keys()}

    for sector, stock_list in SECTOR_STOCK_MAP.items():
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

def save_market_breadth_to_db(breadth_data_by_date):
    """
    Save market breadth data for all sectors in a single wide table.
    Works with both SQLite and Supabase via DatabaseManager.

    Args:
        breadth_data_by_date: Dict of {date: {sector: breadth_value}}

    Example:
        breadth_data_by_date = {
            datetime(2025, 1, 15): {'消费': 0.65, '科技': 0.42, '医药': 0.58},
            datetime(2025, 1, 16): {'消费': 0.67, '科技': 0.44, '医药': 0.60}
        }
    """
    table_name = "market_breadth"

    try:
        # Supabase requires table to be created manually first
        if not db.table_exists(table_name):
            if db_config.USE_SUPABASE:
                print(f"⚠️  Table '{table_name}' doesn't exist in Supabase.")
                print(f"Please create it manually in Supabase dashboard with this schema:")
                print(f"")
                print(f"CREATE TABLE market_breadth (")
                print(f'  "Date" TEXT PRIMARY KEY,')

                # Get all sectors from SECTOR_STOCK_MAP
                sectors = [s for s in SECTOR_STOCK_MAP.keys() if s != 'MARKET_PROXY']
                for sector in sectors:
                    # Replace Chinese characters with safe column names if needed
                    print(f'  "{sector}" REAL,')
                print(f");")
                print(f"")
                return
            else:
                # SQLite - create table automatically
                columns = ['Date TEXT PRIMARY KEY']
                sectors = [s for s in SECTOR_STOCK_MAP.keys() if s != 'MARKET_PROXY']
                for sector in sectors:
                    columns.append(f'"{sector}" REAL')

                schema = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
                db.create_table_sqlite(schema)
                print(f"✓ Created table: {table_name}")

        # Prepare records for insertion
        records = []
        for date, breadth_dict in breadth_data_by_date.items():
            record = {
                'Date': date.strftime('%Y-%m-%d') if isinstance(date, (pd.Timestamp, datetime)) else date,
                **breadth_dict  # Unpack all sector breadths
            }
            records.append(record)

        # Batch insert with upsert (uses your db_manager)
        if records:
            db.insert_records(table_name, records, upsert=True)
            print(f"✓ Saved {len(records)} breadth records to database")

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
