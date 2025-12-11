"""
fetch_epr_core.py
=================
Purpose: Ingest Ethnic Power Relations (EPR) Core data.
Output: car_cewp.epr_core

AUDIT FIX:
1. Removed destructive DROP TABLE.
2. Implemented ensure_table_exists with correct schema.
3. Switched to upload_to_postgis for IDEMPOTENT UPSERTS.
"""
import sys
from pathlib import Path
import logging
import pandas as pd
from sqlalchemy import text

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, load_configs, get_db_engine, PATHS, upload_to_postgis

logger = logging.getLogger(__name__)

SCHEMA = "car_cewp"
TABLE_NAME = "epr_core"

STATUS_MAPPING = {
    'MONOPOLY': 5, 'DOMINANT': 4, 'SENIOR PARTNER': 3,
    'JUNIOR PARTNER': 2, 'POWERLESS': 1, 'DISCRIMINATED': 0,
    'SELF-EXCLUSION': -1, 'IRRELEVANT': -2
}

def load_epr_data(filepath: Path, target_country: str = 'Central African Republic') -> pd.DataFrame:
    logger.info(f"Loading EPR data from {filepath}")
    
    # Check if headerless
    test_df = pd.read_csv(filepath, nrows=1)
    if 'gwid' in test_df.columns or 'statename' in test_df.columns:
        df = pd.read_csv(filepath, encoding='utf-8')
    else:
        df = pd.read_csv(
            filepath, header=None, encoding='utf-8',
            names=['gwid', 'statename', 'from', 'to', 'group', 
                   'groupid', 'gwgroupid', 'umbrella', 'size', 'status', 'reg_aut']
        )
    
    df_country = df[df['statename'] == target_country].copy()
    if df_country.empty:
        raise ValueError(f"No EPR data for {target_country}")
    
    return df_country

def process_epr_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing EPR data...")
    records = []
    for _, row in df.iterrows():
        for year in range(int(row['from']), int(row['to']) + 1):
            records.append({
                'gwid': row['gwid'],
                'statename': row['statename'],
                'year': year,
                'group_name': row['group'], # Renamed for DB safety
                'groupid': row['groupid'],
                'gwgroupid': row['gwgroupid'],
                'size': row['size'],
                'status': row['status'],
                'reg_aut': row.get('reg_aut', None)
            })
    
    df_expanded = pd.DataFrame(records)
    
    # Feature Engineering
    df_expanded['status_numeric'] = df_expanded['status'].map(STATUS_MAPPING)
    df_expanded['is_excluded'] = df_expanded['status'].isin(['POWERLESS', 'DISCRIMINATED', 'SELF-EXCLUSION']).astype(int)
    df_expanded['is_included'] = df_expanded['status'].isin(['MONOPOLY', 'DOMINANT', 'SENIOR PARTNER', 'JUNIOR PARTNER']).astype(int)
    df_expanded['is_discriminated'] = (df_expanded['status'] == 'DISCRIMINATED').astype(int)
    df_expanded['has_autonomy'] = (df_expanded['reg_aut'] == 'TRUE').astype(int)
    
    return df_expanded

def ensure_table_exists(engine):
    """Creates table ONLY if it does not exist."""
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                gwid INTEGER,
                statename TEXT,
                year INTEGER,
                group_name TEXT,
                groupid INTEGER,
                gwgroupid BIGINT,
                size DOUBLE PRECISION,
                status TEXT,
                status_numeric INTEGER,
                is_excluded INTEGER,
                is_included INTEGER,
                is_discriminated INTEGER,
                has_autonomy INTEGER,
                reg_aut TEXT,
                PRIMARY KEY (gwgroupid, year)
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_year ON {SCHEMA}.{TABLE_NAME}(year);"))

def run(configs, engine):
    logger.info("=" * 60)
    logger.info("EPR CORE INGESTION (Safe Upsert)")
    logger.info("=" * 60)
    
    epr_path = PATHS['data_raw'] / 'EPR-2021.csv'
    if not epr_path.exists():
        logger.error(f"EPR file not found: {epr_path}")
        return

    # 1. Load & Process
    df = load_epr_data(epr_path)
    df_processed = process_epr_data(df)
    
    # 2. Schema Check
    ensure_table_exists(engine)
    
    # 3. Upsert
    logger.info(f"Upserting {len(df_processed)} records to {SCHEMA}.{TABLE_NAME}...")
    upload_to_postgis(
        engine, 
        df_processed, 
        TABLE_NAME, 
        SCHEMA, 
        primary_keys=['gwgroupid', 'year']
    )
    logger.info("✓ EPR Core ingestion complete.")

def main():
    data_config, features_config, models_config = load_configs()
    engine = get_db_engine()
    configs = {'data': data_config}
    try:
        run(configs, engine)
    finally:
        engine.dispose()

if __name__ == '__main__':
    main()