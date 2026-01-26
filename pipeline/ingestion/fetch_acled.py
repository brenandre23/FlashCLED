"""
fetch_acled.py
==============
Comprehensive ACLED Ingestion for CAR:
1. Dynamic Windowing: Fetches only missing data (MAX(date) to end_date).
2. Hybrid Retrieval: API-First (bandre1@worldbank.org) with local CSV fallback.
3. Full Schema: Includes all 6 event-type counts (battles, vac, etc.).
4. Spatial Indexing: Generates H3 Resolution 6 indices for every event.
5. Upsert Safety: Handles conflicts on 'event_id_cnty'.

UPDATES (2026-01-24):
- ID NORMALIZATION: Robust event_id_cnty handling with fallback synthesis
- ARGS DEFAULTS: Makes args optional; derives dates from config if absent
- Uses centralized get_incremental_window helper
"""

import sys
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import h3.api.basic_int as h3
from sqlalchemy import text

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import (
    PATHS, 
    get_db_engine, 
    load_configs, 
    logger, 
    upload_to_postgis, 
    get_incremental_window
)

SCHEMA = "car_cewp"
TABLE_NAME = "acled_events"

# Master SQL for table creation to ensure details are preserved
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
    event_id_cnty VARCHAR(50) PRIMARY KEY,
    event_date DATE NOT NULL,
    year INTEGER,
    time_precision INTEGER,
    event_type VARCHAR(100),
    sub_event_type VARCHAR(100),
    actor1 VARCHAR(255),
    assoc_actor_1 VARCHAR(255),
    interaction INTEGER,
    actor2 VARCHAR(255),
    assoc_actor_2 VARCHAR(255),
    admin1 VARCHAR(100),
    admin2 VARCHAR(100),
    admin3 VARCHAR(100),
    location VARCHAR(255),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geo_precision INTEGER,
    source VARCHAR(255),
    source_scale VARCHAR(50),
    notes TEXT,
    fatalities INTEGER,
    timestamp BIGINT,
    h3_index BIGINT,
    
    -- 6 Event Type Counts (Stock/Flow indicators)
    acled_count_battles INTEGER DEFAULT 0,
    acled_count_vac INTEGER DEFAULT 0,
    acled_count_explosions INTEGER DEFAULT 0,
    acled_count_protests INTEGER DEFAULT 0,
    acled_count_riots INTEGER DEFAULT 0,
    acled_count_strategic_developments INTEGER DEFAULT 0
);
"""


@dataclass
class ACLEDArgs:
    """Default arguments when CLI args are not provided."""
    end_date: str = None
    no_incremental: bool = False
    
    def __post_init__(self):
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")


def ensure_table_structure(engine):
    """Ensure the full schema exists before ingestion."""
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))


def fetch_from_api(start, end, email, key):
    """Programmatic API fetch using ACLED /read endpoint."""
    base_url = "https://api.acleddata.com/acled/read"
    params = {
        "email": email,
        "key": key,
        "event_date": f"{start}|{end}",
        "event_date_where": "BETWEEN",
        "country": "Central African Republic",
        "limit": 0  # Get all records in window
    }
    
    try:
        logger.info(f"📡 API Request: {start} -> {end}")
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json().get("data", [])
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        logger.warning(f"⚠️ API failed ({e}). Reverting to CSV...")
        return None


def fetch_from_csv_fallback(start, end):
    """Fallback to local raw/acled.csv if API is unreachable."""
    csv_path = PATHS["data_raw"] / "acled.csv"
    if not csv_path.exists():
        logger.error(f"❌ Fallback failed: {csv_path} not found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df["event_date"] = pd.to_datetime(df["event_date"])
        mask = (df["event_date"] >= pd.to_datetime(start)) & (df["event_date"] <= pd.to_datetime(end))
        return df.loc[mask].copy()
    except Exception as e:
        logger.error(f"❌ Fallback Error: {e}")
        return pd.DataFrame()


def normalize_event_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize event_id_cnty to ensure a valid primary key exists.
    
    Priority:
    1. If event_id_cnty exists → strip/astype str
    2. If data_id exists → use as event_id_cnty
    3. Else → synthesize: event_date.strftime('%Y%m%d') + "_" + actor1 + "_" + location + "_" + row_index
    
    Returns:
        DataFrame with guaranteed non-null event_id_cnty column
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Case 1: event_id_cnty exists
    if 'event_id_cnty' in df.columns:
        df['event_id_cnty'] = df['event_id_cnty'].astype(str).str.strip()
        # Check for nulls/empty strings
        null_mask = df['event_id_cnty'].isna() | (df['event_id_cnty'] == '') | (df['event_id_cnty'] == 'nan')
    else:
        null_mask = pd.Series([True] * len(df), index=df.index)
        df['event_id_cnty'] = None
    
    # Case 2: data_id exists (ACLED API v3 uses this)
    if null_mask.any() and 'data_id' in df.columns:
        df.loc[null_mask, 'event_id_cnty'] = df.loc[null_mask, 'data_id'].astype(str).str.strip()
        null_mask = df['event_id_cnty'].isna() | (df['event_id_cnty'] == '') | (df['event_id_cnty'] == 'nan')
    
    # Case 3: Synthesize from event attributes
    if null_mask.any():
        logger.warning(f"Synthesizing event_id_cnty for {null_mask.sum()} rows with missing IDs")
        
        def synthesize_id(row):
            try:
                date_str = pd.to_datetime(row.get('event_date')).strftime('%Y%m%d')
            except:
                date_str = 'NODATE'
            
            actor = str(row.get('actor1', 'UNKNOWN'))[:20].replace(' ', '_')
            location = str(row.get('location', 'UNKNOWN'))[:20].replace(' ', '_')
            idx = str(row.name)  # Use dataframe index as tiebreaker
            
            return f"{date_str}_{actor}_{location}_{idx}"
        
        synthetic_ids = df.loc[null_mask].apply(synthesize_id, axis=1)
        df.loc[null_mask, 'event_id_cnty'] = synthetic_ids
    
    # Final cleanup
    df['event_id_cnty'] = df['event_id_cnty'].astype(str).str.strip()
    
    return df


def process_data(df, resolution=6):
    """Applies H3 indexing and 6-pillar feature engineering."""
    if df.empty:
        return df
        
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    
    # Pillar Mapping (Flow Variables)
    event_type_map = {
        'Battles': 'acled_count_battles',
        'Violence against civilians': 'acled_count_vac',
        'Explosions/Remote violence': 'acled_count_explosions',
        'Protests': 'acled_count_protests',
        'Riots': 'acled_count_riots',
        'Strategic developments': 'acled_count_strategic_developments'
    }
    
    for col in event_type_map.values():
        df[col] = 0
        
    for event_type, col_name in event_type_map.items():
        df.loc[df['event_type'] == event_type, col_name] = 1

    # Spatial H3 Conversion
    def to_h3(row):
        try:
            return h3.geo_to_h3(float(row['latitude']), float(row['longitude']), resolution)
        except:
            return None

    df['h3_index'] = df.apply(to_h3, axis=1)
    df = df.dropna(subset=['h3_index', 'event_date'])
    
    # Final cleanup of types
    numeric_cols = ['fatalities', 'year', 'geo_precision', 'time_precision']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
    return df


def run(configs=None, engine=None, args=None):
    """
    Primary orchestration logic.
    
    Args:
        configs: Configuration bundle (optional, will load if not provided)
        engine: SQLAlchemy engine (optional, will create if not provided)
        args: CLI arguments (optional, will use defaults if not provided)
    """
    # Handle optional configs
    if configs is None:
        configs = load_configs()
    
    # Handle optional engine
    if engine is None:
        engine = get_db_engine()
    
    # Handle optional args with defaults from config
    if args is None:
        args = ACLEDArgs()
    
    # Ensure args has required attributes with defaults
    if not hasattr(args, 'end_date') or args.end_date is None:
        # Derive from config
        data_cfg = configs.get("data", configs[0] if isinstance(configs, tuple) else {})
        args.end_date = data_cfg.get("global_date_window", {}).get(
            "end_date", 
            datetime.now().strftime("%Y-%m-%d")
        )
    
    if not hasattr(args, 'no_incremental'):
        args.no_incremental = False
    
    force_full = getattr(args, 'no_incremental', False)

    # 1. Calculate the dynamic gap using centralized helper
    start, end = get_incremental_window(
        engine=engine,
        table=TABLE_NAME,
        date_col="event_date",
        requested_end_date=args.end_date,
        default_start_date="2000-01-01",
        force_full=force_full,
        schema=SCHEMA
    )

    if start is None:
        logger.info("✅ ACLED is already fresh. No fetch needed.")
        return

    # 2. Schema Guard
    ensure_table_structure(engine)

    # 3. Hybrid Retrieval
    email = os.getenv("ACLED_EMAIL")
    password = os.getenv("ACLED_PASSWORD")
    
    df = fetch_from_api(start, end, email, password)
    if df is None or df.empty:
        df = fetch_from_csv_fallback(start, end)

    if df.empty:
        logger.warning(f"🛑 No data found for window {start} to {end}.")
        return

    # 4. Feature Engineering & Spatial Indexing
    df_clean = process_data(df)
    
    if df_clean.empty:
        logger.warning("🛑 No valid records after processing (all dropped due to missing coords/dates).")
        return

    # 5. Normalize event_id_cnty (CRITICAL - ensures valid PK)
    df_clean = normalize_event_id(df_clean)
    
    # 6. Validate: Assert no nulls in PK before upload
    null_count = df_clean["event_id_cnty"].isna().sum()
    assert null_count == 0, f"FATAL: {null_count} rows have null event_id_cnty after normalization"
    
    # Also check for duplicates
    dup_count = df_clean["event_id_cnty"].duplicated().sum()
    if dup_count > 0:
        logger.warning(f"⚠️ {dup_count} duplicate event_id_cnty values found. Keeping first occurrence.")
        df_clean = df_clean.drop_duplicates(subset=["event_id_cnty"], keep="first")
    
    # Ensure event_id_cnty is in the outgoing DataFrame
    assert "event_id_cnty" in df_clean.columns, "event_id_cnty column missing from DataFrame"

    # 7. Upsert (Merges new data into history)
    logger.info(f"Upserting {len(df_clean)} events into DB...")
    upload_to_postgis(
        engine=engine,
        df=df_clean,
        table_name=TABLE_NAME,
        schema=SCHEMA,
        primary_keys=["event_id_cnty"] 
    )
    logger.info("✅ ACLED Ingestion Complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD). Defaults to config or today.")
    parser.add_argument("--no-incremental", action="store_true", help="Force full refresh")
    args = parser.parse_args()
    run(args=args)