"""
Stock Name Cache Manager
Fetches stock names from Tushare stock_company API and caches locally
"""

import json
import os
import time
import tushare as ts

CACHE_FILE = "stock_names_cache.json"
TUSHARE_TOKEN = "36838688c6455de2e3affca37060648de15b94b9707a43bb05a38312"

def ticker_to_ts_code(ticker: str) -> str:
    """Convert 6-digit ticker to full ts_code format."""
    ticker = ticker.split('.')[0]  # Remove any existing suffix
    
    if ticker.startswith(('43', '83', '87')):
        return f"{ticker}.BJ"  # Beijing
    elif ticker.startswith('6') or ticker.startswith('688'):
        return f"{ticker}.SH"  # Shanghai
    elif ticker.startswith(('0', '2', '3')):
        return f"{ticker}.SZ"  # Shenzhen
    return ticker

def load_cache():
    """Load stock name cache from JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    """Save stock name cache to JSON file."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except:
        pass

def fetch_stock_name_from_tushare(ts_code: str) -> str:
    """
    Fetch company name from Tushare stock_company API.
    Requires 120 points.
    """
    try:
        # Initialize Tushare API locally (avoid circular import)
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        
        # Rate limit: 200 calls/minute = 0.31s per call
        time.sleep(0.31)
        
        # Use stock_company API - field is 'com_name' not 'name'!
        df = pro.stock_company(
            ts_code=ts_code,
            fields='ts_code,com_name'
        )
        
        if not df.empty:
            name = df.iloc[0]['com_name']
            print(f"[stock_name_cache] ✅ Fetched: {ts_code} -> {name}")
            return name
        else:
            print(f"[stock_name_cache] ⚠️ No info found for {ts_code}")
            return None
            
    except Exception as e:
        print(f"[stock_name_cache] ❌ Failed to fetch {ts_code}: {e}")
        return None

def get_stock_name(ticker: str) -> str:
    """
    Get stock name with local caching.
    Returns: Company name or ticker if not found.
    """
    ts_code = ticker_to_ts_code(ticker)
    
    # Load cache
    cache = load_cache()
    
    # Check cache first
    if ts_code in cache:
        print(f"[stock_name_cache] 📦 Cache hit: {ts_code} -> {cache[ts_code]}")
        return cache[ts_code]
    
    # Fetch from Tushare
    print(f"[stock_name_cache] 🔍 Fetching from Tushare: {ts_code}")
    name = fetch_stock_name_from_tushare(ts_code)
    
    if name:
        # Save to cache
        cache[ts_code] = name
        save_cache(cache)
        return name
    else:
        # Fallback to ticker
        return ticker.split('.')[0]

def batch_fetch_names(tickers: list) -> dict:
    """
    Fetch names for multiple tickers.
    Returns dict: {ticker: name}
    """
    results = {}
    cache = load_cache()
    need_fetch = []
    
    # Separate cached vs need-to-fetch
    for ticker in tickers:
        ts_code = ticker_to_ts_code(ticker)
        if ts_code in cache:
            results[ticker] = cache[ts_code]
        else:
            need_fetch.append(ticker)
    
    # Fetch missing ones
    if need_fetch:
        print(f"[stock_name_cache] Fetching {len(need_fetch)} names from Tushare...")
        for ticker in need_fetch:
            ts_code = ticker_to_ts_code(ticker)
            name = fetch_stock_name_from_tushare(ts_code)
            
            if name:
                cache[ts_code] = name
                results[ticker] = name
            else:
                results[ticker] = ticker
        
        # Save updated cache
        save_cache(cache)
        print(f"[stock_name_cache] ✅ Cache updated with {len(need_fetch)} new names")
    
    return results
