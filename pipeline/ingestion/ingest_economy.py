"""
ingest_economy.py
=================
Fetches economy data (stock indices, commodities, forex, interest rates).

FIXES APPLIED:
- [CRITICAL-2] Uploads to 'economic_drivers' table (was 'market_data').
- [MAJOR-2] Creates 'commodity_gold_price_usd' column for feature registry compatibility.
- [MAJOR-3] Uses .where() for type-safe NaN handling (avoids object dtype).
- [FIX] Flattens MultiIndex columns from yfinance.
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs

SCHEMA = "car_cewp"
TARGET_TABLE = "economic_drivers" # [CRITICAL-2 FIX]

# --- Configuration Constants ---
GOLD_FUTURES_TICKER = "GC=F"
GOLD_DATA_MIN_DATE = "2000-08-30"
FALLBACK_YEARS = 5
MAX_RETRIES = 3
RETRY_DELAY = 2

TICKERS = {
    "indices": ["^GSPC", "^IXIC", "^DJI"],
    "commodities": [GOLD_FUTURES_TICKER, "CL=F"],
    "forex": ["EURUSD=X", "GBPUSD=X", "JPYUSD=X"],
    "rates": ["^TNX", "^FVX", "^TYX"]
}

def fetch_yahoo_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch price data from Yahoo Finance with column flattening fix."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Fetching {ticker} (attempt {attempt+1})")
            
            df = yf.download(
                ticker, start=start_date, end=end_date, progress=False, auto_adjust=False
            )
            
            if df.empty:
                return None

            # [FIX] Flatten MultiIndex Columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # Normalize column names
            cols_map = {c: str(c).lower().replace(' ', '_') for c in df.columns}
            df = df.rename(columns=cols_map)
            
            if 'date' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'date'})

            if 'close' not in df.columns:
                if 'adj_close' in df.columns:
                    df = df.rename(columns={'adj_close': 'close'})
                else:
                    logger.warning(f"  Missing 'close' column for {ticker}")
                    return None

            df['ticker'] = ticker
            return df
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(RETRY_DELAY)
    return None

def fetch_gold_prices(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    start_dt = pd.to_datetime(start_date)
    gold_min_dt = pd.to_datetime(GOLD_DATA_MIN_DATE)
    
    adjusted_start = GOLD_DATA_MIN_DATE if start_dt < gold_min_dt else start_date
    df = fetch_yahoo_prices(GOLD_FUTURES_TICKER, adjusted_start, end_date)
    
    if df is None or df.empty:
        end_dt_obj = pd.to_datetime(end_date)
        fallback_start = (end_dt_obj - timedelta(days=FALLBACK_YEARS * 365)).strftime('%Y-%m-%d')
        if pd.to_datetime(fallback_start) < gold_min_dt:
            fallback_start = GOLD_DATA_MIN_DATE
        df = fetch_yahoo_prices(GOLD_FUTURES_TICKER, fallback_start, end_date)
        
    return df

def fetch_market_category(category: str, start_date: str, end_date: str) -> pd.DataFrame:
    all_data = []
    for ticker in TICKERS.get(category, []):
        if ticker == GOLD_FUTURES_TICKER:
            df = fetch_gold_prices(start_date, end_date)
        else:
            df = fetch_yahoo_prices(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            df['category'] = category
            all_data.append(df)
            
    if not all_data:
        return pd.DataFrame()
        
    combined = pd.concat(all_data, ignore_index=True)
    return combined

def upload_to_database(df: pd.DataFrame, table_name: str, engine):
    if df.empty: return
    if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
    
    rename_map = {
        'open': 'open_price',
        'high': 'high_price',
        'low': 'low_price',
        'close': 'close_price'
    }
    df = df.rename(columns=rename_map)
    
    # [MAJOR-3 FIX] Use .where() for type-safe NaN assignment (avoids object dtype)
    if 'ticker' in df.columns:
        df['commodity_gold_price_usd'] = df['close_price'].where(
            df['ticker'] == GOLD_FUTURES_TICKER
        )

    target_cols = ['date', 'ticker', 'category', 'open_price', 'high_price', 
                   'low_price', 'close_price', 'volume', 'commodity_gold_price_usd']
    
    # Ensure all target cols exist
    for c in target_cols:
        if c not in df.columns:
            df[c] = None
    
    df[target_cols].to_sql(
        table_name, engine, schema=SCHEMA, if_exists='replace',
        index=False, method='multi', chunksize=1000
    )
    logger.info(f"Uploaded {len(df)} rows to {table_name}")

def run(configs, engine):
    data_cfg = configs['data']
    start_date = data_cfg.get('global_date_window', {}).get('start_date', '2017-01-01')
    end_date = data_cfg.get('global_date_window', {}).get('end_date', '2025-12-31')

    all_dfs = []
    for cat in TICKERS.keys():
        df = fetch_market_category(cat, start_date, end_date)
        if not df.empty: all_dfs.append(df)
        
    if not all_dfs:
        logger.warning("No economy data fetched.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    
    # [CRITICAL-2 FIX] Use correct table name 'economic_drivers'
    upload_to_database(combined, TARGET_TABLE, engine)

def main():
    try:
        cfgs = load_configs()
        if isinstance(cfgs, tuple):
            configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
        else:
            configs = cfgs
            
        engine = get_db_engine()
        run(configs, engine)
        engine.dispose()
    except Exception as e:
        logger.error(f"Economy ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()