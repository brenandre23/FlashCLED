"""
ingest_economy.py
=================
Fetches economy data in WIDE format for cleaner modeling.

RESTRUCTURED (Task 2):
- Fetches ONLY: GC=F (Gold), CL=F (Oil), ^GSPC (S&P500), EURUSD=X (EUR/USD)
- Outputs WIDE format: one row per date with columns:
    date, gold_price_usd, oil_price_usd, sp500_index, eur_usd_rate
- Uploads to car_cewp.economic_drivers (drop/recreate schema)
"""

import sys
import pandas as pd
import yfinance as yf
import time
from pathlib import Path
from typing import Optional
from sqlalchemy import text

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis

SCHEMA = "car_cewp"
TARGET_TABLE = "economic_drivers"

# --- Configuration Constants ---
MAX_RETRIES = 3
RETRY_DELAY = 2

# Tickers to fetch and their column mappings
TICKER_CONFIG = {
    "GC=F": "gold_price_usd",      # Gold Futures
    "CL=F": "oil_price_usd",       # Crude Oil Futures
    "^GSPC": "sp500_index",        # S&P 500 Index
    "EURUSD=X": "eur_usd_rate",    # EUR/USD Exchange Rate
}


def fetch_yahoo_prices(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch price data from Yahoo Finance with robust column handling."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Fetching {ticker} (attempt {attempt+1})")
            df = yf.download(
                ticker, start=start_date, end=end_date, progress=False, auto_adjust=False
            )
            if df.empty:
                logger.warning(f"  Empty data for {ticker}")
                return None

            # Flatten MultiIndex Columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # Normalize column names
            cols_map = {c: str(c).lower().replace(' ', '_') for c in df.columns}
            df = df.rename(columns=cols_map)

            if 'date' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'date'})

            # Use 'close' price as the value
            if 'close' not in df.columns:
                if 'adj_close' in df.columns:
                    df = df.rename(columns={'adj_close': 'close'})
                else:
                    logger.warning(f"  Missing 'close' column for {ticker}")
                    return None

            # Return only date and close
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df[['date', 'close']].copy()

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(RETRY_DELAY)
    
    return None


def fetch_all_tickers(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch all configured tickers and pivot to wide format."""
    logger.info(f"Fetching {len(TICKER_CONFIG)} economic indicators...")
    
    all_dfs = []
    
    for ticker, col_name in TICKER_CONFIG.items():
        df = fetch_yahoo_prices(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            df = df.rename(columns={'close': col_name})
            all_dfs.append(df)
            logger.info(f"  ✓ {ticker} -> {col_name}: {len(df)} rows")
        else:
            logger.warning(f"  ✗ {ticker}: No data fetched")
    
    if not all_dfs:
        logger.error("No economic data fetched from any ticker!")
        return pd.DataFrame()
    
    # Merge all DataFrames on date (wide format)
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = pd.merge(result, df, on='date', how='outer')
    
    # Sort by date and reset index
    result = result.sort_values('date').reset_index(drop=True)
    
    # Convert date to proper datetime for DB
    result['date'] = pd.to_datetime(result['date'])
    
    logger.info(f"Combined Wide DataFrame: {len(result)} rows, {len(result.columns)} columns")
    
    return result


def recreate_table(engine):
    """Drop and recreate the economic_drivers table with new wide schema."""
    logger.info(f"Recreating {SCHEMA}.{TARGET_TABLE} with wide schema...")
    
    with engine.begin() as conn:
        # Create schema if not exists
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        
        # Drop existing table
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{TARGET_TABLE};"))
        
        # Create new table with wide schema
        conn.execute(text(f"""
            CREATE TABLE {SCHEMA}.{TARGET_TABLE} (
                date DATE PRIMARY KEY,
                gold_price_usd FLOAT,
                oil_price_usd FLOAT,
                sp500_index FLOAT,
                eur_usd_rate FLOAT
            );
        """))
        
        # Add index for date queries
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_date 
            ON {SCHEMA}.{TARGET_TABLE} (date);
        """))
    
    logger.info(f"  ✓ Table {SCHEMA}.{TARGET_TABLE} recreated with wide schema")


def upload_wide_data(df: pd.DataFrame, engine):
    """Upload the wide-format DataFrame to PostgreSQL."""
    if df.empty:
        logger.warning("Empty DataFrame - nothing to upload")
        return
    
    # Ensure all columns exist
    expected_cols = ['date', 'gold_price_usd', 'oil_price_usd', 'sp500_index', 'eur_usd_rate']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[expected_cols].copy()
    
    # Upload using upsert on primary key 'date'
    upload_to_postgis(engine, df, TARGET_TABLE, SCHEMA, primary_keys=['date'])
    
    logger.info(f"  ✓ Uploaded {len(df)} rows to {SCHEMA}.{TARGET_TABLE}")


def run(configs, engine):
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("ECONOMY INGESTION (Wide Format)")
    logger.info("=" * 60)
    
    data_cfg = configs['data']
    start_date = data_cfg.get('global_date_window', {}).get('start_date', '2000-01-01')
    end_date = data_cfg.get('global_date_window', {}).get('end_date', '2025-12-31')
    
    logger.info(f"Date Range: {start_date} to {end_date}")
    
    # 1. Recreate table with new schema
    recreate_table(engine)
    
    # 2. Fetch all tickers in wide format
    df = fetch_all_tickers(start_date, end_date)
    
    if df.empty:
        logger.error("No economy data to upload.")
        return
    
    # 3. Upload
    upload_wide_data(df, engine)
    
    # 4. Summary stats
    logger.info("Summary Statistics:")
    for col in ['gold_price_usd', 'oil_price_usd', 'sp500_index', 'eur_usd_rate']:
        if col in df.columns:
            non_null = df[col].notna().sum()
            logger.info(f"  {col}: {non_null} non-null values")
    
    logger.info("=" * 60)
    logger.info("✓ ECONOMY INGESTION COMPLETE")
    logger.info("=" * 60)


def main():
    try:
        cfgs = load_configs()
        configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]} \
            if isinstance(cfgs, tuple) else cfgs

        engine = get_db_engine()
        run(configs, engine)
        engine.dispose()
    except Exception as e:
        logger.error(f"Economy ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
