"""
ingest_economy.py
=================
Fetches economy data in WIDE format for cleaner modeling.

RESTRUCTURED (Task 2):
- Fetches ONLY: GC=F (Gold), CL=F (Oil), ^GSPC (S&P500), EURUSD=X (EUR/USD)
- Outputs WIDE format: one row per date with columns:
    date, gold_price_usd, oil_price_usd, sp500_index, eur_usd_rate
- Uploads to car_cewp.economic_drivers (upsert on date)

INCREMENTAL LOADING (2026-01-24):
- Checks MAX(date) before fetch to avoid redundant API calls
- Only fetches new data from (MAX(date) + 1 day) onwards
- Supports --full flag for complete refresh
"""

import sys
import argparse
import pandas as pd
import yfinance as yf
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from sqlalchemy import text

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import (
    logger,
    get_db_engine,
    load_configs,
    upload_to_postgis,
    get_incremental_window,
)

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


def _get_max_existing_date(engine) -> Optional[date]:
    """
    Get the maximum date from existing economic data.
    Returns None if table doesn't exist or is empty.
    """
    try:
        with engine.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = :schema AND table_name = :table
                );
            """), {"schema": SCHEMA, "table": TARGET_TABLE}).scalar()
            
            if not exists:
                return None
            
            result = conn.execute(text(f"SELECT MAX(date) FROM {SCHEMA}.{TARGET_TABLE}")).scalar()
            return result
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}")
        return None


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

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
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

            df['date'] = pd.to_datetime(df['date']).dt.date
            return df[['date', 'close']].copy()

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(RETRY_DELAY)
    
    return None


def fetch_all_tickers(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch all configured tickers and pivot to wide format."""
    logger.info(f"Fetching {len(TICKER_CONFIG)} economic indicators...")
    logger.info(f"  Date range: {start_date} to {end_date}")
    
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
    
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = pd.merge(result, df, on='date', how='outer')
    
    result = result.sort_values('date').reset_index(drop=True)
    result['date'] = pd.to_datetime(result['date'])
    logger.info(f"Combined Wide DataFrame: {len(result)} rows, {len(result.columns)} columns")
    return result


def ensure_table_exists(engine):
    """Create the economic_drivers table if it doesn't exist."""
    logger.info(f"Ensuring {SCHEMA}.{TARGET_TABLE} exists...")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TARGET_TABLE} (
                date DATE PRIMARY KEY,
                gold_price_usd FLOAT,
                oil_price_usd FLOAT,
                sp500_index FLOAT,
                eur_usd_rate FLOAT
            );
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_date 
            ON {SCHEMA}.{TARGET_TABLE} (date);
        """))
    logger.info(f"  ✓ Table {SCHEMA}.{TARGET_TABLE} ready")


def upload_wide_data(df: pd.DataFrame, engine):
    """Upload the wide-format DataFrame to PostgreSQL."""
    if df.empty:
        logger.warning("Empty DataFrame - nothing to upload")
        return
    expected_cols = ['date', 'gold_price_usd', 'oil_price_usd', 'sp500_index', 'eur_usd_rate']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    df = df[expected_cols].copy()
    upload_to_postgis(engine, df, TARGET_TABLE, SCHEMA, primary_keys=['date'])
    logger.info(f"  ✓ Uploaded {len(df)} rows to {SCHEMA}.{TARGET_TABLE}")


def run(configs, engine, full_refresh: bool = False):
    """
    Main execution function.
    Args:
        configs: Configuration dict
        engine: SQLAlchemy engine
        full_refresh: If True, force full window fetch (no drop)
    """
    logger.info("=" * 60)
    logger.info("ECONOMY INGESTION (Wide Format)")
    logger.info("=" * 60)
    
    data_cfg = configs['data']
    start_date = data_cfg.get('global_date_window', {}).get('start_date', '2000-01-01')
    end_date = data_cfg.get('global_date_window', {}).get('end_date', None)
    min_start = pd.to_datetime("2003-12-01").date()
    start_date = max(pd.to_datetime(start_date).date(), min_start)
    
    ensure_table_exists(engine)

    start, end = get_incremental_window(
        engine=engine,
        table=TARGET_TABLE,
        date_col="date",
        requested_end_date=end_date or date.today().strftime("%Y-%m-%d"),
        default_start_date=str(start_date),
        force_full=full_refresh,
        schema=SCHEMA,
    )

    if start is None:
        logger.info("✓ Economic data already up to date. No new data to fetch.")
        return

    # Normalize dates for yfinance; avoid zero-length window
    clean_start = pd.to_datetime(start).strftime("%Y-%m-%d")
    clean_end = pd.to_datetime(end).strftime("%Y-%m-%d")
    if clean_start == clean_end:
        clean_end = (pd.to_datetime(clean_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Date Range determined: {clean_start} to {clean_end}")
    df = fetch_all_tickers(clean_start, clean_end)
    
    if df.empty:
        logger.info("No new economic data to upload.")
        return
    
    upload_wide_data(df, engine)
    
    logger.info("Summary Statistics:")
    for col in ['gold_price_usd', 'oil_price_usd', 'sp500_index', 'eur_usd_rate']:
        if col in df.columns:
            non_null = df[col].notna().sum()
            logger.info(f"  {col}: {non_null} non-null values")
    
    logger.info("=" * 60)
    logger.info("✓ ECONOMY INGESTION COMPLETE")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance economic indicators")
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Force full refresh (fetch entire window from default start)"
    )
    args = parser.parse_args()
    
    try:
        cfgs = load_configs()
        configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]} \
            if isinstance(cfgs, tuple) else cfgs

        engine = get_db_engine()
        run(configs, engine, full_refresh=args.full)
        engine.dispose()
    except Exception as e:
        logger.error(f"Economy ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
