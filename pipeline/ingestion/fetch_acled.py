"""
fetch_acled.py
==============
ACLED Conflict Event Ingestion.

UPDATES:
1. TRANSACTION FIX: Separated table creation and inspection to prevent NoSuchTableError.
2. SCHEMA FIX: Added 'geo_precision' and 'time_precision' checks/defaults.
3. COLUMN RESTORATION: Added 'admin1' and 'admin2' back to the DataFrame.
"""
import sys
from pathlib import Path
import pandas as pd
import h3.api.basic_int as h3
import logging
from sqlalchemy import text, inspect
import numpy as np

# --- Setup Imports ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import load_configs, get_db_engine, PATHS, upload_to_postgis, logger

SCHEMA = "car_cewp"
TABLE_NAME = "acled_events"

def ensure_table_structure(engine):
    """
    Creates table if missing, then adds missing columns if needed.
    Split into two transactions to ensure table visibility.
    """
    logger.info(f"Verifying schema for {SCHEMA}.{TABLE_NAME}...")
    
    # --- STEP 1: Create Table (Commit immediately) ---
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                event_id_cnty VARCHAR(50) PRIMARY KEY,
                event_date DATE NOT NULL,
                year INTEGER,
                time_precision INTEGER,
                event_type VARCHAR(100),
                sub_event_type VARCHAR(100),
                geo_precision INTEGER,
                actor1 VARCHAR(255),
                actor2 VARCHAR(255),
                latitude FLOAT,
                longitude FLOAT,
                location VARCHAR(255),
                admin1 VARCHAR(255),
                admin2 VARCHAR(255),
                fatalities INTEGER,
                notes TEXT,
                h3_index BIGINT,
                conflict_binary INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        # Create indexes
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_acled_date ON {SCHEMA}.{TABLE_NAME}(event_date);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_acled_h3 ON {SCHEMA}.{TABLE_NAME}(h3_index);"))
    
    # --- STEP 2: Schema Evolution (Inspect separate from creation) ---
    # Now that Step 1 is committed, the inspector can see the table.
    inspector = inspect(engine)
    
    if not inspector.has_table(TABLE_NAME, schema=SCHEMA):
        # This should theoretically not happen if Step 1 worked, but good for safety
        logger.warning(f"Table {SCHEMA}.{TABLE_NAME} still not found after creation. Skipping column checks.")
        return

    existing_cols = {col['name'] for col in inspector.get_columns(TABLE_NAME, schema=SCHEMA)}
    
    required_cols = {
        'geo_precision': 'INTEGER', 
        'time_precision': 'INTEGER', 
        'admin1': 'VARCHAR(255)', 
        'admin2': 'VARCHAR(255)'
    }
    
    with engine.begin() as conn:
        for col, dtype in required_cols.items():
            if col not in existing_cols:
                logger.warning(f"Column '{col}' missing in DB. Adding it now...")
                try:
                    conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE_NAME} ADD COLUMN {col} {dtype}"))
                except Exception as e:
                    logger.warning(f"Could not add column {col}: {e}")

def process_data(df: pd.DataFrame, resolution: int = 5) -> pd.DataFrame:
    """
    Clean and map ACLED data to H3.
    """
    initial_count = len(df)
    logger.info(f"  Raw ACLED rows: {initial_count:,}")

    # 1. Standardize Date
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    
    # Drop rows with bad dates
    df = df.dropna(subset=['event_date'])
    date_count = len(df)
    logger.info(f"  Rows after date cleaning: {date_count:,} (Dropped {initial_count - date_count:,})")

    if df.empty:
        return df

    # 2. H3 Mapping
    def get_h3(lat, lon, res):
        try:
            return h3.latlng_to_cell(lat, lon, res)
        except:
            return 0

    df['h3_index'] = df.apply(lambda x: get_h3(x['latitude'], x['longitude'], resolution), axis=1)
    
    # Filter invalid H3
    df = df[df['h3_index'] != 0]
    h3_count = len(df)
    logger.info(f"  Rows after H3 mapping: {h3_count:,} (Dropped {date_count - h3_count:,})")
    
    # 3. Conflict Binary & Integers
    df['conflict_binary'] = (df['fatalities'] > 0).astype(int)
    df['h3_index'] = df['h3_index'].astype('int64')
    df['fatalities'] = df['fatalities'].fillna(0).astype(int)

    # 4. Column Selection & Defaults
    # Ensure all required columns exist in DataFrame, filling defaults if missing
    defaults = {
        'geo_precision': 1,
        'time_precision': 1,
        'admin1': None,
        'admin2': None
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns:
            logger.warning(f"  Column '{col}' missing in CSV. Filling with default.")
            df[col] = default_val
        elif col in ['geo_precision', 'time_precision']:
            # Ensure numeric types for precision columns
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val).astype(int)

    cols = [
        'event_id_cnty', 'event_date', 'year', 'event_type', 'sub_event_type', 
        'actor1', 'actor2', 'location', 'latitude', 'longitude', 
        'geo_precision', 'time_precision', 
        'admin1', 'admin2', 
        'fatalities', 'notes', 'h3_index', 'conflict_binary'
    ]
    
    # Only keep columns that actually exist in the DF (safety)
    final_cols = [c for c in cols if c in df.columns]
    
    return df[final_cols].copy()

def run(configs, engine):
    logger.info("=" * 60)
    logger.info("ACLED INGESTION (Transaction Fixed)")
    logger.info("=" * 60)
    
    # Configs
    features_cfg = configs['features']
    resolution = features_cfg['spatial']['h3_resolution']
    
    # Locate File
    acled_path = PATHS['data_raw'] / 'acled.csv'
    if not acled_path.exists():
        logger.error(f"ACLED file not found at {acled_path}")
        return

    # Load
    logger.info(f"Loading {acled_path}...")
    df = pd.read_csv(acled_path, low_memory=False)
    
    # Ensure Table Structure (Auto-Fix)
    ensure_table_structure(engine)

    # Process
    df_clean = process_data(df, resolution)
    
    if df_clean.empty:
        logger.warning("⚠️ No valid events found after processing! Check date formats or coordinate columns.")
        return

    # Upload (UPSERT)
    logger.info(f"Upserting {len(df_clean):,} events to {SCHEMA}.{TABLE_NAME}...")
    upload_to_postgis(
        engine, 
        df_clean, 
        TABLE_NAME, 
        SCHEMA, 
        primary_keys=['event_id_cnty']
    )
    logger.info("✓ ACLED Ingestion Complete.")

if __name__ == "__main__":
    from utils import load_configs
    cfg = load_configs()
    eng = get_db_engine()
    run(cfg, eng)