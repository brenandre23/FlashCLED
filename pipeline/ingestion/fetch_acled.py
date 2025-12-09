"""
fetch_acled.py
==============
ACLED Conflict Event Ingestion.
CRITICAL FIX: Enforces h3_index as BIGINT for pipeline consistency.
"""
import sys
from pathlib import Path
import pandas as pd
import h3
import logging
from sqlalchemy import text
import numpy as np

# --- Setup Imports ---
# Adjust path to find utils
ROOT_DIR = Path(__file__).resolve().parents[2] # pipeline/ingestion -> root
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import load_configs, get_db_engine, PATHS

logger = logging.getLogger(__name__)

ACLED_COLUMNS = [
    'event_id_cnty', 'event_date', 'year', 'time_precision', 'event_type',
    'sub_event_type', 'disorder_type', 'actor1', 'assoc_actor_1', 'inter1',
    'actor2', 'assoc_actor_2', 'inter2', 'interaction', 'civilian_targeting',
    'iso', 'region', 'country', 'admin1', 'admin2', 'admin3', 'location',
    'latitude', 'longitude', 'geo_precision', 'source', 'source_scale', 'notes',
    'fatalities', 'tags', 'timestamp', 'geometry_wkt', 'h3_index_acled'
]

def ensure_table_exists(engine):
    """Recreates table with strictly typed H3 column."""
    logger.info("Verifying schema for car_cewp.acled_events (Enforcing BIGINT H3)...")
    
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS car_cewp;"))
        
        # Check if we need to migrate or drop. 
        # For prototype safety, we drop/recreate to guarantee types.
        conn.execute(text("DROP TABLE IF EXISTS car_cewp.acled_events CASCADE;"))
        
        conn.execute(text("""
            CREATE TABLE car_cewp.acled_events (
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
                h3_index BIGINT, -- FIXED: Was VARCHAR
                conflict_binary INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_acled_date ON car_cewp.acled_events(event_date);
            CREATE INDEX IF NOT EXISTS idx_acled_h3 ON car_cewp.acled_events(h3_index);
            CREATE INDEX IF NOT EXISTS idx_acled_type ON car_cewp.acled_events(event_type);
        """))
    logger.info("✓ Table schema enforced.")

def safe_h3_to_int(h3_str):
    """Safely converts H3 Hex String to Int64."""
    try:
        if pd.isna(h3_str): return None
        return int(h3_str, 16)
    except (ValueError, TypeError):
        return None

def process_data(df: pd.DataFrame, resolution: int = 5) -> pd.DataFrame:
    logger.info("Processing ACLED events...")
    
    # 1. Date Parsing
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    df = df.dropna(subset=['event_date']).copy()
    
    # 2. H3 Calculation (Resolution 5)
    logger.info(f"Calculating H3 indices (Resolution {resolution})...")
    
    # Logic: Prefer existing column, else calculate
    # The library 'h3' returns Hex Strings. We must convert to Int.
    
    def get_h3(row):
        # Use pre-calculated if valid
        if pd.notnull(row.get('h3_index_acled')) and len(str(row['h3_index_acled'])) == 15:
            return row['h3_index_acled']
        # Calculate from Lat/Lon
        try:
            return h3.latlng_to_cell(row['latitude'], row['longitude'], resolution)
        except:
            return None

    # Apply generation
    df['h3_hex'] = df.apply(get_h3, axis=1)
    
    # Convert Hex String -> BigInt
    df['h3_index'] = df['h3_hex'].apply(safe_h3_to_int)
    
    # Filter invalid H3
    df = df.dropna(subset=['h3_index'])
    df['h3_index'] = df['h3_index'].astype('int64')

    # 3. Derived Columns
    df['conflict_binary'] = (df['fatalities'] > 0).astype(int)
    
    # 4. Filter Columns
    cols = ['event_id_cnty', 'event_date', 'year', 'event_type', 'sub_event_type',
            'actor1', 'actor2', 'latitude', 'longitude', 'location', 'admin1', 'admin2',
            'fatalities', 'notes', 'h3_index', 'conflict_binary']
    
    return df[cols].copy()

def run(configs_bundle: dict, engine):
    logger.info("=" * 60)
    logger.info("ACLED INGESTION (Type-Safe)")
    logger.info("=" * 60)
    
    data_cfg = configs_bundle['data']
    features_cfg = configs_bundle['features']
    resolution = features_cfg['spatial']['h3_resolution']
    
    # Find file
    acled_path_1 = PATHS['data_raw'] / 'acled.csv'
    acled_path_2 = Path(PATHS['root']) / 'data' / 'acled.csv'
    
    acled_path = acled_path_1 if acled_path_1.exists() else acled_path_2
    if not acled_path.exists():
        logger.error("ACLED file not found.")
        return

    # Load
    logger.info(f"Loading {acled_path}...")
    df = pd.read_csv(acled_path, header=None, names=ACLED_COLUMNS, low_memory=False)
    
    # Process
    ensure_table_exists(engine)
    df_clean = process_data(df, resolution)
    
    # Upload
    logger.info(f"Uploading {len(df_clean):,} events...")
    with engine.begin() as conn:
        df_clean.to_sql(
            'acled_events',
            con=conn,
            schema='car_cewp',
            if_exists='append',
            index=False,
            method='multi',
            chunksize=2000
        )
    logger.info("✓ ACLED Ingestion Complete.")

if __name__ == '__main__':
    data_config, features_config, models_config = load_configs()
    engine = get_db_engine()
    configs = {'data': data_config, 'features': features_config, 'models': models_config}
    run(configs, engine)