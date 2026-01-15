"""
sync_predictions.py
===================
Scans the processed data directory for 'predictions_*.parquet' files
and performs a bulk upload to PostGIS.

Usage: python pipeline/database/sync_predictions.py
"""
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import text

# Add root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, SCHEMA, PATHS


def run_bulk_sync():
    logger.info("💾 STARTING BULK PREDICTION SYNC")
    
    # 1. Find all prediction files
    pred_files = list(PATHS["data_proc"].glob("predictions_*.parquet"))
    # Also support CSVs if you switched types
    pred_files += list(PATHS["data_proc"].glob("predictions_*.csv"))
    
    if not pred_files:
        logger.warning("⚠️ No prediction files found in data/processed/")
        return

    logger.info(f"   Found {len(pred_files)} files to sync.")
    
    # 2. Combine into one Master DataFrame
    dfs = []
    for p in pred_files:
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            logger.error(f"❌ Failed to read {p.name}: {e}")

    if not dfs:
        return

    master_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"   Combined Shape: {master_df.shape}")

    # 3. Clean Types for PostGIS
    if "date" in master_df.columns:
        master_df["date"] = pd.to_datetime(master_df["date"])
    if "h3_index" in master_df.columns:
        master_df["h3_index"] = master_df["h3_index"].astype("int64")

    # 4. Atomic Replace (Transaction)
    engine = get_db_engine()
    table_name = "predictions_latest"
    
    try:
        with engine.begin() as conn:
            # Option A: TRUNCATE (Wipe and Replace everything)
            # This is safest if you want the DB to reflect exactly what is on disk
            logger.info(f"   Truncating {SCHEMA}.{table_name}...")
            conn.execute(text(f"TRUNCATE TABLE {SCHEMA}.{table_name}"))
            
            # Option B: DELETE specific horizons (if you want to be incremental)
            # horizons = master_df['horizon'].unique()
            # conn.execute(text(f\"DELETE FROM {SCHEMA}.{table_name} WHERE horizon = ANY(:h)\"), {\"h\": list(horizons)})

        # 5. Bulk Upload
        logger.info(f"   Uploading {len(master_df)} rows...")
        master_df.to_sql(
            table_name, 
            engine, 
            schema=SCHEMA, 
            if_exists="append", 
            index=False, 
            method="multi", 
            chunksize=10000 
        )
        logger.info("✅ BULK SYNC COMPLETE")
        
    except Exception as e:
        logger.error(f"❌ DB Sync Failed: {e}")


if __name__ == "__main__":
    run_bulk_sync()
