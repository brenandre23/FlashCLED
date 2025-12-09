"""
fetch_epr_core.py - FIXED TO MATCH ACTUAL DATABASE SCHEMA
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sqlalchemy import text

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import load_configs, get_db_engine, PATHS

logger = logging.getLogger(__name__)

STATUS_MAPPING = {
    'MONOPOLY': 5, 'DOMINANT': 4, 'SENIOR PARTNER': 3,
    'JUNIOR PARTNER': 2, 'POWERLESS': 1, 'DISCRIMINATED': 0,
    'SELF-EXCLUSION': -1, 'IRRELEVANT': -2
}


def load_epr_data(filepath: Path, target_country: str = 'Central African Republic') -> pd.DataFrame:
    """Load EPR CSV - handles BOTH headerless AND with-headers formats."""
    logger.info(f"Loading EPR data from {filepath}")
    
    # Try reading first to detect format
    test_df = pd.read_csv(filepath, nrows=1)
    
    # Check if first column is 'gwid' (has headers) or numeric (no headers)
    if 'gwid' in test_df.columns or 'statename' in test_df.columns:
        # HAS HEADERS
        logger.info("EPR file has headers - reading with header=0")
        df = pd.read_csv(filepath, encoding='utf-8')
        # Keep original column names from CSV
    else:
        # NO HEADERS
        logger.info("EPR file is headerless - reading with header=None")
        df = pd.read_csv(
            filepath,
            header=None,
            encoding='utf-8',
            names=['gwid', 'statename', 'from', 'to', 'group', 
                   'groupid', 'gwgroupid', 'umbrella', 'size', 'status', 'reg_aut']
        )
    
    logger.info(f"Loaded {len(df):,} total EPR records")
    
    # Filter for target country
    df_country = df[df['statename'] == target_country].copy()
    
    if df_country.empty:
        logger.warning(f"No EPR data found for '{target_country}'")
        raise ValueError(f"No EPR data for {target_country}")
    
    logger.info(f"✓ Found {len(df_country):,} records for {target_country}")
    return df_country


def process_epr_data(df: pd.DataFrame) -> pd.DataFrame:
    """Expand year ranges and create features."""
    logger.info("Processing EPR data...")
    
    records = []
    for _, row in df.iterrows():
        # Use 'from' and 'to' column names from CSV
        for year in range(int(row['from']), int(row['to']) + 1):
            records.append({
                'gwid': row['gwid'],
                'statename': row['statename'],
                'year': year,
                'group': row['group'],
                'groupid': row['groupid'],
                'gwgroupid': row['gwgroupid'],
                'size': row['size'],
                'status': row['status'],
                'reg_aut': row.get('reg_aut', None)
            })
    
    df_expanded = pd.DataFrame(records)
    logger.info(f"Expanded to {len(df_expanded):,} group-year records")
    
    # Create features
    df_expanded['status_numeric'] = df_expanded['status'].map(STATUS_MAPPING)
    df_expanded['is_excluded'] = df_expanded['status'].isin(['POWERLESS', 'DISCRIMINATED', 'SELF-EXCLUSION']).astype(int)
    df_expanded['is_included'] = df_expanded['status'].isin(['MONOPOLY', 'DOMINANT', 'SENIOR PARTNER', 'JUNIOR PARTNER']).astype(int)
    df_expanded['is_discriminated'] = (df_expanded['status'] == 'DISCRIMINATED').astype(int)
    df_expanded['has_autonomy'] = (df_expanded['reg_aut'] == 'TRUE').astype(int)
    
    logger.info("✓ EPR data processed")
    return df_expanded


def create_epr_table(engine):
    """Create EPR table matching actual schema from image."""
    logger.info("Creating EPR table schema...")
    
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE SCHEMA IF NOT EXISTS car_cewp;
            
            DROP TABLE IF EXISTS car_cewp.epr_core;
            
            CREATE TABLE car_cewp.epr_core (
                gwid INTEGER,
                statename VARCHAR(255),
                year INTEGER,
                group_name VARCHAR(255),
                groupid INTEGER,
                gwgroupid BIGINT,
                size FLOAT,
                status VARCHAR(50),
                status_numeric INTEGER,
                is_excluded INTEGER,
                is_included INTEGER,
                is_discriminated INTEGER,
                has_autonomy INTEGER,
                reg_aut VARCHAR(10),
                PRIMARY KEY (gwgroupid, year)
            );
            
            CREATE INDEX IF NOT EXISTS idx_epr_year ON car_cewp.epr_core(year);
            CREATE INDEX IF NOT EXISTS idx_epr_group ON car_cewp.epr_core(group_name);
            CREATE INDEX IF NOT EXISTS idx_epr_status ON car_cewp.epr_core(status);
        """))
    
    logger.info("✓ EPR table schema ready")


def upload_to_database(df: pd.DataFrame, engine):
    """Upload EPR data."""
    logger.info(f"Uploading {len(df):,} records to car_cewp.epr_core...")
    
    # Rename 'group' to 'group_name' for database
    df_upload = df.rename(columns={'group': 'group_name'})
    
    with engine.begin() as conn:
        # Table was already dropped and recreated, so just insert
        df_upload.to_sql(
            'epr_core',
            con=conn,
            schema='car_cewp',
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
    
    logger.info("✓ Upload complete")


def run(configs_bundle: dict, engine):
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("EPR CORE CSV INGESTION")
    logger.info("=" * 60)
    
    # USE PATHS FROM UTILS
    epr_path = PATHS['data_raw'] / 'EPR-2021.csv'
    
    if not epr_path.exists():
        logger.error(f"EPR-2021.csv not found at: {epr_path}")
        raise FileNotFoundError(f"EPR-2021.csv not found")
    
    logger.info(f"Using EPR file: {epr_path}")
    
    df = load_epr_data(epr_path, target_country='Central African Republic')
    df_processed = process_epr_data(df)
    create_epr_table(engine)  # This drops and recreates the table
    upload_to_database(df_processed, engine)
    
    logger.info("=" * 60)
    logger.info("✓ EPR CORE INGESTION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    data_config, features_config, models_config = load_configs()
    engine = get_db_engine()
    configs_bundle = {'data': data_config, 'features': features_config, 'models': models_config}
    run(configs_bundle, engine)
