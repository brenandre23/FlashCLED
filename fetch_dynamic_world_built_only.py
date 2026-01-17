"""
fetch_dynamic_world_built_only.py
=================================
Purpose: Ingests ONLY 'landcover_built' fraction into the EXISTING landcover_features table.

- TARGET: car_cewp.landcover_features
- ACTION: Updates existing rows with 'dw_built_frac' data.
- SAFETY: Uses a temp-table UPDATE approach to avoid overwriting existing
          grass/crop/tree data with NULLs.
"""

import sys
import ee
import pandas as pd
import time
import json
import gc
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, ensure_h3_int64, init_gee
# Note: upload_to_postgis is replaced by a custom update function below for safety

# --- Constants ---
SCHEMA = "car_cewp"
GRID_TABLE = "features_static"
TARGET_TABLE = "landcover_features"

MAX_WORKERS = 8
H3_BATCH_SIZE = 200
MAX_RETRIES = 5

# Dynamic World collection
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

# ONLY fetch the Built class
DW_CLASSES = {
    'built': 6,      # Built-up area
}


# -------------------------------------------------------------------------
# 1. DATABASE & HELPERS (UPDATED FOR SAFETY)
# -------------------------------------------------------------------------

def ensure_landcover_table_exists(engine, schema="car_cewp"):
    """
    Ensure landcover_features table exists with expected schema and PK.
    Borrowed from the full ingestion to allow standalone updates.
    """
    create_sql = text(f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    CREATE TABLE IF NOT EXISTS {schema}.{TARGET_TABLE} (
        h3_index BIGINT NOT NULL,
        date DATE NOT NULL,
        dw_grass_frac FLOAT,
        dw_crops_frac FLOAT,
        dw_trees_frac FLOAT,
        dw_bare_frac FLOAT,
        dw_built_frac FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (h3_index, date)
    );
    CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_date ON {schema}.{TARGET_TABLE} (date);
    CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_h3 ON {schema}.{TARGET_TABLE} (h3_index);
    """)
    with engine.begin() as conn:
        conn.execute(create_sql)
    logger.info(f"Table {schema}.{TARGET_TABLE} is ready.")


def ensure_built_column_exists(engine, schema="car_cewp"):
    """
    Ensures the 'dw_built_frac' column exists in the table.
    Does NOT recreate the table, preserving existing data.
    """
    check_sql = text(f"""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_schema = '{schema}' 
      AND table_name = '{TARGET_TABLE}' 
      AND column_name = 'dw_built_frac';
    """)
    
    add_col_sql = text(f"""
    ALTER TABLE {schema}.{TARGET_TABLE}
    ADD COLUMN IF NOT EXISTS dw_built_frac FLOAT;
    """)
    
    with engine.connect() as conn:
        result = conn.execute(check_sql).fetchone()
        if not result:
            logger.info(f"Column 'dw_built_frac' missing. Adding to {schema}.{TARGET_TABLE}...")
            conn.execute(add_col_sql)
            conn.commit()
        else:
            logger.info("Column 'dw_built_frac' already exists.")


def batch_upsert_built_column(engine, df, schema="car_cewp"):
    """
    Safely upserts ONLY the dw_built_frac column.
    Uses a temporary table + INSERT...ON CONFLICT to avoid touching other cols.
    """
    if df.empty:
        return

    # 1. Upload to a temporary staging table
    temp_table = f"temp_built_update_{int(time.time())}"
    from sqlalchemy import BigInteger, Date, Float
    df.to_sql(
        temp_table,
        engine,
        schema=schema,
        if_exists='replace',
        index=False,
        dtype={"h3_index": BigInteger(), "date": Date(), "dw_built_frac": Float()},
    )
    
    # 2. Perform the upsert on the main table using the temp table
    update_sql = text(f"""
    INSERT INTO {schema}.{TARGET_TABLE} (h3_index, date, dw_built_frac)
    SELECT h3_index::bigint, date::date, dw_built_frac FROM {schema}.{temp_table}
    ON CONFLICT (h3_index, date) DO UPDATE
    SET dw_built_frac = EXCLUDED.dw_built_frac;

    DROP TABLE {schema}.{temp_table};
    """)
    
    with engine.begin() as conn:
        conn.execute(update_sql)


def get_h3_grid_data(engine):
    logger.info("Loading H3 Grid from PostGIS...")
    sql = f"SELECT to_hex(h3_index::bigint) AS h3_id, ST_AsGeoJSON(geometry) AS geom FROM {SCHEMA}.{GRID_TABLE}"
    df = pd.read_sql(sql, engine)
    grid_data = [{"h3_id": r["h3_id"], "geom": json.loads(r["geom"])} for _, r in df.iterrows()]
    logger.info(f"Loaded {len(grid_data)} H3 cells.")
    return grid_data


def make_ee_feature_collection(grid_subset):
    features = [ee.Feature(ee.Geometry(c['geom']), {"h3_index": c['h3_id']}) for c in grid_subset]
    return ee.FeatureCollection(features)


# -------------------------------------------------------------------------
# 2. DATE LOGIC
# -------------------------------------------------------------------------

def get_publication_lag(features_cfg):
    """
    Get publication lag in days from features.yaml config.
    """
    temporal_cfg = features_cfg.get('temporal', {})
    step_days = temporal_cfg.get('step_days', 14)
    pub_lags = temporal_cfg.get('publication_lags', {})
    
    lag_steps = pub_lags.get('DynamicWorld', pub_lags.get('GEE', 1))
    lag_days = lag_steps * step_days
    
    logger.info(f"Dynamic World publication lag: {lag_steps} steps × {step_days} days = {lag_days} days")
    return lag_days


def get_collection_last_date():
    try:
        # Keep EE requests from hanging indefinitely
        ee.data.setDeadline(60)
        last_img = ee.ImageCollection(DW_COLLECTION).limit(1, 'system:time_start', False).first()
        last_ts = last_img.get('system:time_start').getInfo()
        if last_ts:
            return datetime.fromtimestamp(last_ts / 1000.0, tz=timezone.utc).replace(tzinfo=None)
    except Exception as e:
        logger.warning(f"Could not verify end date for Dynamic World: {e}")
    return None


def get_collection_first_date():
    return datetime(2015, 6, 27)


def build_14day_spine(start_year: int, end_year: int):
    epoch = datetime(2000, 1, 1)
    req_start = datetime(start_year, 1, 1)
    req_end = datetime(end_year, 12, 31)
    spine = []
    curr = epoch
    while curr < req_start:
        curr += timedelta(days=14)
    while curr <= req_end:
        spine.append((curr, curr + timedelta(days=13)))
        curr += timedelta(days=14)
    return spine


def window_strs(start_dt: datetime, end_dt: datetime):
    return start_dt.strftime("%Y-%m-%d"), (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")


# -------------------------------------------------------------------------
# 3. GEE PROCESSING (BUILT ONLY)
# -------------------------------------------------------------------------

def process_single_batch(batch_cells, s_str, e_excl_str):
    """
    Process a batch of H3 cells for a single time window.
    Strictly selects ONLY the 'built' band.
    """
    batch_fc = make_ee_feature_collection(batch_cells)
    
    for attempt in range(MAX_RETRIES):
        try:
            dw = ee.ImageCollection(DW_COLLECTION).filterDate(s_str, e_excl_str)
            
            count = dw.size().getInfo()
            if count == 0:
                return None
            
            # UPDATED: Select ONLY 'built'
            dw_mean = dw.select(['built']).mean()
            
            results = dw_mean.reduceRegions(
                collection=batch_fc,
                reducer=ee.Reducer.mean(),
                scale=100, 
                tileScale=4
            )
            
            return results.getInfo()
            
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "too many requests" in err:
                time.sleep((attempt + 1) * 10)
            elif "timeout" in err or "remote end closed" in err:
                time.sleep((attempt + 1) * 5)
            else:
                logger.warning(f"Batch attempt {attempt+1} failed: {e}")
                time.sleep(2)
    
    raise RuntimeError("Worker failed after max retries")


# -------------------------------------------------------------------------
# 4. MAIN ORCHESTRATOR
# -------------------------------------------------------------------------

def process_year_batch(year: int, all_cells: list, windows, engine, lag_days: int):
    """Process all windows for a single year (Built Features Only)."""
    logger.info(f"--- Processing {len(windows)} windows for Year {year} (BUILT ONLY) ---")
    logger.info(f"    Publication lag applied: {lag_days} days")
    
    cell_batches = [all_cells[i:i + H3_BATCH_SIZE] for i in range(0, len(all_cells), H3_BATCH_SIZE)]
    
    for start_dt, end_dt in tqdm(windows, desc=f"Year {year}"):
        s_str, e_excl_str = window_strs(start_dt, end_dt)
        daily_rows = []
        
        work_items = [(batch, s_str, e_excl_str) for batch in cell_batches]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(process_single_batch, *args): i for i, args in enumerate(work_items)}
            
            for future in as_completed(future_to_batch):
                try:
                    result = future.result()
                    if result is None:
                        continue
                    
                    for feat in result.get("features", []):
                        props = feat.get("properties", {})
                        h3_idx = props.get("h3_index")
                        
                        lagged_date = start_dt + timedelta(days=lag_days)
                        
                        daily_rows.append({
                            "h3_index": h3_idx,
                            "date": lagged_date.strftime("%Y-%m-%d"),
                            # ONLY built fraction
                            "dw_built_frac": props.get("built"),
                        })
                        
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
        
        if daily_rows:
            df = pd.DataFrame(daily_rows)
            df["h3_index"] = df["h3_index"].apply(ensure_h3_int64)
            df = df.dropna(subset=["h3_index"])
            df["h3_index"] = df["h3_index"].astype("int64")
            
            # UPDATED: Use Safe Upsert instead of generic Upload
            batch_upsert_built_column(engine, df, SCHEMA)
        
        gc.collect()


def run(configs, engine):
    """Main entry point for Dynamic World ingestion."""
    logger.info("=" * 60)
    logger.info("DYNAMIC WORLD (BUILT ONLY) UPDATE")
    logger.info("=" * 60)
    
    data_cfg = configs["data"]
    features_cfg = configs["features"]
    project_id = data_cfg["gee"]["project_id"]
    g_start = int(data_cfg["global_date_window"]["start_date"][:4])
    g_end = int(data_cfg["global_date_window"]["end_date"][:4])
    
    # UPDATED: Ensure table/column exist without dropping data
    ensure_landcover_table_exists(engine)
    ensure_built_column_exists(engine)
    
    lag_days = get_publication_lag(features_cfg)
    init_gee(project_id)
    
    dw_start = get_collection_first_date()
    dw_end = get_collection_last_date()
    
    if dw_end:
        logger.info(f"✓ Dynamic World available: {dw_start.date()} to {dw_end.date()}")
    else:
        logger.warning("Could not determine Dynamic World end date. Proceeding with config dates.")
        dw_end = datetime(g_end, 12, 31)
    
    effective_start = max(g_start, 2015)
    
    all_cells = get_h3_grid_data(engine)
    
    # Check for BUILT column processing, not just any row presence
    # We check if dw_built_frac is NOT NULL to skip processed dates
    try:
        existing = pd.read_sql(
            f"SELECT DISTINCT date FROM {SCHEMA}.{TARGET_TABLE} WHERE dw_built_frac IS NOT NULL", 
            engine
        )
        processed_dates = set(pd.to_datetime(existing['date']).dt.strftime('%Y-%m-%d'))
        logger.info(f"Found {len(processed_dates)} dates with BUILT data already present.")
    except:
        processed_dates = set()
    
    for year in range(effective_start, g_end + 1):
        full_spine = build_14day_spine(year, year)
        valid_spine = [w for w in full_spine if w[0] >= dw_start and w[0] <= dw_end]
        
        if not valid_spine:
            logger.info(f"Skipping Year {year} (no valid windows)")
            continue
        
        windows_to_run = []
        for w in valid_spine:
            lagged_date = (w[0] + timedelta(days=lag_days)).strftime("%Y-%m-%d")
            if lagged_date not in processed_dates:
                windows_to_run.append(w)
        
        if windows_to_run:
            process_year_batch(year, all_cells, windows_to_run, engine, lag_days)
        else:
            logger.info(f"Year {year} complete (built features present).")
    
    logger.info("✓ Dynamic World BUILT update complete.")


def main():
    try:
        cfgs = load_configs()
        if isinstance(cfgs, tuple):
            configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
        else:
            configs = cfgs
        engine = get_db_engine()
        run(configs, engine)
    finally:
        if 'engine' in locals():
            engine.dispose()


if __name__ == "__main__":
    main()
