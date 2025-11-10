import pandas as pd
import tushare as ts
import time
from datetime import datetime, timedelta
import random
import sqlite3
import sys
import numpy as np # Added for NaN handling

# --- 1. CONFIGURATION ---
DB_NAME = 'assrs_tushare_local.db'
TUSHARE_API = None

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

# set() 会自动处理重复的股票代码
ALL_STOCK_TICKERS = sorted(list(set(ticker for stocks in SECTOR_STOCK_MAP.values() for ticker in stocks)))
REQUIRED_COLUMNS = ['Open', 'Close', 'High', 'Low', 'Volume'] 
MIN_HISTORY_DAYS = 100 
MAX_RETRIES = 5
VOL_ZSCORE_LOOKBACK = 100 # 用于标准化成交量的回看期

# --- 2. TUSHARE & DATABASE FUNCTIONS ---

def init_tushare(token):
    """初始化 Tushare API。"""
    global TUSHARE_API
    if token == '36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312':
        pass # 令牌已硬编码
    try:
        ts.set_token(token)
        TUSHARE_API = ts.pro_api()
        print("Tushare 令牌设置成功。(测试查询已跳过)")
        return True
    except Exception as e:
        print(f"Tushare 初始化失败: {e}")
        return False

def get_tushare_ticker(ticker):
    """将6位股票代码转换为 Tushare 格式 (例如 601398 -> 601398.SH)"""
    if ticker.startswith('6') or ticker.startswith('688'): return f"{ticker}.SH"
    elif ticker.startswith('0') or ticker.startswith('3'): return f"{ticker}.SZ"
    return ticker

def create_table(conn, ticker):
    """为特定股票创建表格。"""
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
    """为特定的聚合PPI创建表格。"""
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
    """查找特定股票代码在数据库中存储的最后日期。"""
    try:
        cursor = conn.execute(f'SELECT MAX(Date) FROM "{ticker}"')
        last_date = cursor.fetchone()[0]
        return last_date
    except sqlite3.OperationalError:
        return None

def insert_data(conn, ticker, df):
    """将 DataFrame 插入到股票的表格中。"""
    df_to_insert = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_to_insert.index.name = 'Date'
    df_to_insert.reset_index(inplace=True)
    df_to_insert.to_sql(ticker, conn, if_exists='append', index=False)

def fetch_stock_data_robust(ticker, start_date, end_date):
    """
    使用 Tushare Pro 稳健地抓取并重建 hfq (前复权) 数据。
    """
    global TUSHARE_API
    ts_code = get_tushare_ticker(ticker)
    
    for attempt in range(MAX_RETRIES):
        try:
            df = TUSHARE_API.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,open,close,high,low,vol'
            )
            if df.empty:
                return pd.DataFrame() 
            
            adj_factor_df = TUSHARE_API.adj_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,adj_factor'
            )
            if adj_factor_df.empty:
                raise ValueError(f"Tushare 'adj_factor' 为 {ts_code} 返回了空值。")

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
            print(f"    - 第 {attempt + 1}/{MAX_RETRIES} 次尝试抓取 {ticker} 失败。错误: {error_message}")
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait_time)
            else:
                print(f"    - 致命错误: {ticker} 的所有 {MAX_RETRIES} 次尝试均失败。")
                return None

# --- 3. PUBLIC FUNCTIONS ---

def ensure_data_in_db(start_date, end_date):
    """
    主数据函数。确保从 start_date 到 end_date 的所有数据都在数据库中。
    仅抓取新的或缺失的数据。
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
                    print(f"警告: 无法解析日期 {last_db_date_str} (股票: {ticker})。将重新抓取所有数据。")
                    last_db_date_str = None
                
                if last_db_date_str:
                    last_db_date_fmt = last_db_date_dt.strftime('%Y%m%d')
                    if last_db_date_fmt >= end_date:
                        continue
                    fetch_start_date = (last_db_date_dt + timedelta(days=1)).strftime('%Y%m%d')

            print(f"  -> 正在为 {ticker} 抓取新数据，从 {fetch_start_date} 到 {end_date}...")
            new_data_df = fetch_stock_data_robust(ticker, fetch_start_date, end_date)
            
            if new_data_df is not None and not new_data_df.empty:
                if last_db_date_str:
                     new_data_df = new_data_df[new_data_df.index > last_db_date_str]
                
                if not new_data_df.empty:
                    insert_data(conn, ticker, new_data_df)
                    print(f"  - 成功为 {ticker} 更新了 {len(new_data_df)} 行新数据。")
                else:
                    print(f"  - {ticker} 没有新数据可添加。")
            elif new_data_df is not None and new_data_df.empty:
                print(f"  - 在此期间未找到 {ticker} 的新数据。")
            else:
                print(f"  - 抓取 {ticker} 的新数据失败。")
            
            time.sleep(1.0) # Tushare 速率限制

def get_all_stock_data_from_db():
    """
    从数据库加载所有股票数据到 DataFrame 字典中。
    --- 新增: 同时为每只股票计算成交量 Z-Score ---
    """
    all_stock_data = {}
    with sqlite3.connect(DB_NAME) as conn:
        for ticker in ALL_STOCK_TICKERS:
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{ticker}"', conn, index_col='Date', parse_dates=['Date'])
                if not df.empty:
                    missing_cols_db = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                    if missing_cols_db:
                        print(f"!! 警告: 数据库中 {ticker} 的数据缺少列: {missing_cols_db}。")
                        print(f"!! 请删除 '{DB_NAME}' 并重新运行 main.py 来修复。")
                    else:
                        # ---!!!--- 新增: 计算标准化的成交量指标 ---!!!---
                        df['Vol_Mean'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).mean()
                        df['Vol_Std'] = df['Volume'].rolling(window=VOL_ZSCORE_LOOKBACK, min_periods=20).std()
                        df['Norm_Vol_Metric'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']
                        # ---!!!--- 新增结束 ---!!!---
                        all_stock_data[ticker] = df.sort_index()
            except Exception as e:
                print(f"从数据库加载 {ticker} 的数据失败: {e}")
    return all_stock_data

def aggregate_ppi_data(all_stock_data):
    """
    将单个股票数据聚合成扇区级别的代理组合指数 (PPIs)。
    V3.6: 对成交量/换手率使用 "先标准化，后平均" 的方法。
          计算一个完全有效的 OHLC 指数 (所有值均以100为基准)。
    """
    print("\n--- 正在聚合股票数据为扇区 PPIs (V3.6 逻辑) ---")
    all_sector_ppi_data = {}
    
    for sector, stock_list in SECTOR_STOCK_MAP.items():
        
        # 1. 使用 'outer' 连接对齐所有成分股数据
        named_dfs = []
        for ticker in stock_list:
             if ticker in all_stock_data:
                temp_df = all_stock_data[ticker].copy()
                
                # ---!!!--- 修复: 处理交易暂停 (Volume == 0) ---!!!---
                is_halted = (temp_df['Volume'] == 0)
                temp_df[is_halted] = np.nan
                # ---!!!--- 修复结束 ---!!!---
                
                temp_df.name = ticker 
                named_dfs.append(temp_df)
                
        if not named_dfs: continue
        
        aligned_df = pd.concat(named_dfs, axis=1, keys=[df.name for df in named_dfs], join='outer')
        
        valid_tickers = [t for t in stock_list if (t, 'Close') in aligned_df.columns]
        if not valid_tickers: continue

        # 2. 获取成分股数据切片
        open_prices = aligned_df.xs('Open', level=1, axis=1)[valid_tickers]
        high_prices = aligned_df.xs('High', level=1, axis=1)[valid_tickers]
        low_prices = aligned_df.xs('Low', level=1, axis=1)[valid_tickers]
        close_prices = aligned_df.xs('Close', level=1, axis=1)[valid_tickers]
        
        # ---!!!--- 新增: 获取预先计算的标准化成交量指标 ---!!!---
        norm_vol_metric = aligned_df.xs('Norm_Vol_Metric', level=1, axis=1)[valid_tickers]
        
        prev_close_prices = close_prices.shift(1)
        
        # --- 3. 计算 PPI 指数 (OHLC 修复) ---
        # .mean(axis=1) 自动跳过 NaNs (我们的停牌修复)
        ret_open = (open_prices / prev_close_prices - 1).mean(axis=1)
        ret_high = (high_prices / prev_close_prices - 1).mean(axis=1)
        ret_low = (low_prices / prev_close_prices - 1).mean(axis=1)
        ret_close = (close_prices / prev_close_prices - 1).mean(axis=1)

        # 4. 链接回报率以构建指数
        ppi_df = pd.DataFrame(index=aligned_df.index)
        
        # 我们必须在计算回报率 *之后* 对其 fillna(0)
        # 这样 cumprod() 才不会在全停牌的日子中断
        ppi_df['Close'] = 100 * (1 + ret_close.fillna(0)).cumprod()
        ppi_df['Open'] = ppi_df['Close'].shift(1) * (1 + ret_open)
        ppi_df['High'] = ppi_df['Close'].shift(1) * (1 + ret_high)
        ppi_df['Low'] = ppi_df['Close'].shift(1) * (1 + ret_low)
        
        # 5. 计算标准化成交量指标 (Z-Scores 的平均值)
        # .mean(axis=1) 正确地计算 *活跃* 股票的平均值
        ppi_df['Norm_Vol_Metric'] = norm_vol_metric.mean(axis=1)
        
        ppi_df.dropna(inplace=True) # 删除因回看而产生的初始 NaN 行
        
        if len(ppi_df) >= MIN_HISTORY_DAYS:
            all_sector_ppi_data[sector] = ppi_df
        
    print("--- PPI 聚合完成 ---")
    return all_sector_ppi_data

def save_ppi_data_to_db(all_ppi_data):
    """
    将聚合的 PPI DataFrames 保存到数据库的新表中。
    """
    print(f"\n--- 正在保存 {len(all_ppi_data)} 个 PPIs 到数据库 '{DB_NAME}' ---")
    with sqlite3.connect(DB_NAME) as conn:
        for sector_name, ppi_df in all_ppi_data.items():
            table_name = f"PPI_{sector_name}"
            create_ppi_table(conn, sector_name) # 创建 'PPI_' 表
            
            # 准备插入
            df_to_insert = ppi_df.copy()
            df_to_insert = df_to_insert[['Open', 'High', 'Low', 'Close', 'Norm_Vol_Metric']]
            df_to_insert.index.name = 'Date'
            df_to_insert.reset_index(inplace=True)
            
            df_to_insert.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"  - 成功保存 '{table_name}' ({len(df_to_insert)} 行)。")
    print("--- PPI 数据库保存完毕 ---")

def load_ppi_data_from_db():
    """
    从数据库加载所有 'PPI_' 表到 DataFrame 字典中。
    """
    print("\n--- 正在从数据库加载预计算的 PPIs ---")
    all_ppi_data = {}
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'PPI_%';")
        ppi_tables = cursor.fetchall()
        
        if not ppi_tables:
            print(f"!! 错误: 在 '{DB_NAME}' 中未找到 PPI 表。")
            print("!! 请先运行 'main.py' 来构建 PPI 表。")
            return None
            
        for (table_name,) in ppi_tables:
            try:
                sector_name = table_name.replace('PPI_', '')
                df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn, index_col='Date', parse_dates=['Date'])
                if not df.empty:
                    # 将 'Norm_Vol_Metric' 重命名回一个通用名称
                    df.rename(columns={'Norm_Vol_Metric': 'Volume_Metric'}, inplace=True)
                    all_ppi_data[sector_name] = df.sort_index()
            except Exception as e:
                print(f"从数据库加载 {table_name} 失败: {e}")
                
    print(f"--- 成功加载 {len(all_ppi_data)} 个 PPIs ---")
    return all_ppi_data