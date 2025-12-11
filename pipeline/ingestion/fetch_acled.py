"""
fetch_acled.py
==============
ACLED Conflict Event Ingestion.

AUDIT FIXES:
1. IDEMPOTENCY: Replaced raw to_sql('append') with utils.upload_to_postgis (Upsert).
2. TYPE SAFETY: Enforces H3 BIGINT and proper Date types.
"""
import sys
from pathlib import Path
import pandas as pd
import h3.api.basic_int as h3
import logging
from sqlalchemy import text
import numpy as np

# --- Setup Imports ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import load_configs, get_db_engine, PATHS, upload_to_postgis, logger

ACLED_COLUMNS = [
    'event_id_cnty', 'event_date', 'year', 'time_precision', 'event_type',
    'sub_event_type', 'disorder_type', 'actor1', 'assoc_actor_1', 'inter1',
    'actor2', 'assoc_actor_2', 'inter2', 'interaction', 'civilian_targeting',
    'iso', 'region', 'country', 'admin1', 'admin2', 'admin3', 'location',
    'latitude', 'longitude', 'geo_precision', 'source', 'source_scale', 'notes',
    'fatalities', 'tags', 'timestamp', 'geometry_wkt', 'h3_index_acled'
]

SCHEMA = "car_cewp"
TABLE_NAME = "acled_events"

def ensure_table_exists(engine):
    """
    Creates table with strict typing. 
    Note: We rely on upload_to_postgis to handle schema evolution, 
    but this ensures the target table exists for the first run.
    """
    logger.info(f"Verifying schema for {SCHEMA}.{TABLE_NAME}...")
    
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                event_id_cnty VARCHAR(50) PRIMARY KEY,
                event_date DATE NOT NULL,
                year INTEGER,
                event_type VARCHAR(100),
                sub_event_type VARCHAR(100),
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
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_acled_type ON {SCHEMA}.{TABLE_NAME}(event_type);"))

def safe_h3_to_int(h3_str):
    try:
        if pd.isna(h3_str): return None
        return int(h3_str, 16)
    except (ValueError, TypeError):
        return None

def process_data(df: pd.DataFrame, resolution: int = 5) -> pd.DataFrame:
    logger.info("Processing ACLED events...")
    
    # 1. Date Parsing
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce').dt.date
    df = df.dropna(subset=['event_date']).copy()
    
    # 2. H3 Calculation
    logger.info(f"Calculating H3 indices (Resolution {resolution})...")
    
    def get_h3(row):
        # Prefer pre-calculated
        if pd.notnull(row.get('h3_index_acled')) and len(str(row['h3_index_acled'])) == 15:
            return row['h3_index_acled']
        # Calculate
        try:
            return h3.latlng_to_cell(row['latitude'], row['longitude'], resolution)
        except:
            return None

    df['h3_hex'] = df.apply(get_h3, axis=1)
    df['h3_index'] = df['h3_hex'].apply(safe_h3_to_int)
    
    # Filter valid H3
    df = df.dropna(subset=['h3_index'])
    df['h3_index'] = df['h3_index'].astype('int64')

    # 3. Derived Columns
    df['conflict_binary'] = (df['fatalities'] > 0).astype(int)
    
    # 4. Filter Columns
    cols = [
        'event_id_cnty', 'event_date', 'year', 'event_type', 'sub_event_type',
        'actor1', 'actor2', 'latitude', 'longitude', 'location', 'admin1', 'admin2',
        'fatalities', 'notes', 'h3_index', 'conflict_binary'
    ]
    return df[cols].copy()

def run(configs, engine):
    logger.info("=" * 60)
    logger.info("ACLED INGESTION (Idempotent Fix)")
    logger.info("=" * 60)
    
    # Configs
    features_cfg = configs['features']
    resolution = features_cfg['spatial']['h3_resolution']
    
    # Locate File
    acled_path_1 = PATHS['data_raw'] / 'acled.csv'
    if not acled_path_1.exists():
        logger.error(f"ACLED file not found at {acled_path_1}")
        return

    # Load & Process
    logger.info(f"Loading {acled_path_1}...")
    df = pd.read_csv(acled_path_1, header=None, names=ACLED_COLUMNS, low_memory=False)
    
    ensure_table_exists(engine)
    df_clean = process_data(df, resolution)
    
    if df_clean.empty:
        logger.warning("No valid events found after processing.")
        return

    # Upload (UPSERT)
    logger.info(f"Upserting {len(df_clean):,} events to {SCHEMA}.{TABLE_NAME}...")
    upload_to_postgis(
        engine, 
        df_clean, 
        TABLE_NAME, 
        SCHEMA, 
        primary_keys=['event_id_cnty']  # Critical for Idempotency
    )
    logger.info("✓ ACLED Ingestion Complete.")

if __name__ == '__main__':
    data_config, features_config, models_config = load_configs()
    engine = get_db_engine()
    configs = {'data': data_config, 'features': features_config, 'models': models_config}
    run(configs, engine)