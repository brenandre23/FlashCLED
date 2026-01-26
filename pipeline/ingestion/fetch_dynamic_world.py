"""
fetch_dynamic_world.py
======================
Ingests Dynamic World land cover fractions from Google Earth Engine.
Full Ingestion (Grass, Crops, Trees, Bare, Built).

- START DATE: 2017-03-07 (Sentinel-2B Launch)
- OPTIMIZED: Uses monkey-patched timeouts and robust error handling.

UPDATES (2026-01-24):
- Uses centralized get_incremental_window helper
- Supports --full / --no-incremental flags for complete refresh
- Adds overlap buffer for late-arriving data
"""

import sys
import os
import socket
import ee
import requests
import pandas as pd
import geopandas as gpd
import time
import json
import gc
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text

# --- MONKEY PATCH: FORCE REQUESTS TIMEOUT ---
_original_session_request = requests.Session.request

def _forced_timeout_request(self, method, url, *args, **kwargs):
    current_timeout = kwargs.get('timeout')
    if current_timeout is None or (isinstance(current_timeout, (int, float)) and current_timeout < 30):
        kwargs['timeout'] = 600
    return _original_session_request(self, method, url, *args, **kwargs)

requests.Session.request = _forced_timeout_request
socket.setdefaulttimeout(600)
os.environ["EE_HTTP_TIMEOUT"] = "600"
# --------------------------------------------

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis, ensure_h3_int64, init_gee, get_incremental_window, SCHEMA

# --- Constants ---
GRID_TABLE = "features_static"
TARGET_TABLE = "landcover_features"

# --- UPDATED CUTOFF: Sentinel-2B Launch ---
CUTOFF_DATE = pd.Timestamp("2017-03-07")
DEFAULT_START_DATE = "2017-03-07"

# Overlap buffer for late-arriving/reprocessed data
OVERLAP_BUFFER_DAYS = 14

MAX_WORKERS = 6
H3_BATCH_SIZE = 200
MAX_RETRIES = 5
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

# -------------------------------------------------------------------------
# DATABASE & HELPERS
# -------------------------------------------------------------------------

def ensure_landcover_table_exists(engine, schema=SCHEMA):
    sql = text(f"""
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
        conn.execute(sql)

def get_h3_grid_data(engine):
    query = f"SELECT h3_index, geometry FROM {SCHEMA}.{GRID_TABLE}"
    df = gpd.read_postgis(query, engine, geom_col="geometry")
    df['h3_index'] = df['h3_index'].apply(ensure_h3_int64)
    return df.to_dict('records')

# -------------------------------------------------------------------------
# DATE UTILS
# -------------------------------------------------------------------------

def build_14day_spine(start_year, end_year):
    GLOBAL_START = datetime(2000, 1, 1)
    req_start = datetime(start_year, 1, 1)
    req_end = datetime(end_year, 12, 31)
    
    delta_days = (req_start - GLOBAL_START).days
    remainder = delta_days % 14
    current_date = req_start if remainder == 0 else req_start + timedelta(days=(14 - remainder))
    
    spine = []
    while current_date <= req_end:
        spine.append((current_date, current_date + timedelta(days=13)))
        current_date += timedelta(days=14)
    return spine

# -------------------------------------------------------------------------
# GEE PROCESSING
# -------------------------------------------------------------------------

def process_single_batch(batch_cells, s_str, e_excl_str):
    features = []
    for row in batch_cells:
        geom = row['geometry']
        if hasattr(geom, '__geo_interface__'):
             coords = geom.__geo_interface__['coordinates']
             features.append(ee.Feature(ee.Geometry.Polygon(coords), {'h3': str(row['h3_index'])}))
    
    if not features: return None
    batch_fc = ee.FeatureCollection(features)
    
    for attempt in range(MAX_RETRIES):
        try:
            dw = ee.ImageCollection(DW_COLLECTION).filterDate(s_str, e_excl_str)
            
            # Optimization: Try to reduce; catch "no bands" if empty
            dw_mean = dw.select(['grass', 'crops', 'trees', 'bare', 'built']).mean()
            
            results = dw_mean.reduceRegions(
                collection=batch_fc,
                reducer=ee.Reducer.mean(),
                scale=100,
                tileScale=4
            )
            return results.getInfo()
            
        except Exception as e:
            msg = str(e).lower()
            if "image has no bands" in msg or "element.select" in msg:
                return None  # Correctly handle empty data
            if "429" in msg or "timeout" in msg:
                time.sleep((attempt + 1) * 5)
            else:
                logger.warning(f"Batch attempt {attempt+1} failed: {e}")
                time.sleep(2)
    
    raise RuntimeError("Worker failed after max retries")

# -------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -------------------------------------------------------------------------

def process_year_batch(year, all_cells, windows, engine, lag_days):
    logger.info(f"--- Processing Year {year} ({len(windows)} windows) ---")
    cell_batches = [all_cells[i:i + H3_BATCH_SIZE] for i in range(0, len(all_cells), H3_BATCH_SIZE)]
    
    for start_dt, end_dt in tqdm(windows, desc=f"Year {year}"):
        s_str = start_dt.strftime("%Y-%m-%d")
        e_excl_str = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        daily_rows = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(process_single_batch, batch, s_str, e_excl_str): batch 
                for batch in cell_batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    result = future.result()
                    if result is None: continue
                    
                    for feat in result.get("features", []):
                        props = feat.get("properties", {})
                        h3_idx = props.get("h3")
                        if h3_idx:
                            daily_rows.append({
                                "h3_index": int(h3_idx),
                                "date": (start_dt + timedelta(days=lag_days)).strftime("%Y-%m-%d"),
                                "dw_grass_frac": props.get("grass"),
                                "dw_crops_frac": props.get("crops"),
                                "dw_trees_frac": props.get("trees"),
                                "dw_bare_frac": props.get("bare"),
                                "dw_built_frac": props.get("built"),
                            })
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
        
        if daily_rows:
            df = pd.DataFrame(daily_rows)
            upload_to_postgis(engine, df, TARGET_TABLE, SCHEMA, primary_keys=["h3_index", "date"])
        gc.collect()

def run(configs=None, engine=None, full_refresh: bool = False):
    """
    Main Dynamic World ingestion function.
    
    Args:
        configs: Configuration dict
        engine: SQLAlchemy engine
        full_refresh: If True, ignore existing data and fetch all records
    """
    logger.info("DYNAMIC WORLD FULL INGESTION (Optimized)")
    
    # Handle optional configs/engine
    if configs is None:
        cfgs = load_configs()
        configs = cfgs if isinstance(cfgs, dict) else {"data": cfgs[0], "features": cfgs[1]}
    if engine is None:
        engine = get_db_engine()
    
    data_cfg = configs.get("data", configs[0] if isinstance(configs, tuple) else {})
    features_cfg = configs.get("features", configs[1] if isinstance(configs, tuple) else {})
    
    project_id = data_cfg.get("gee", {}).get("project_id")
    if not project_id:
        logger.error("GEE project_id not found in data.yaml")
        return
    
    # Get config end date
    requested_end_date = data_cfg.get("global_date_window", {}).get("end_date")
    if not requested_end_date:
        requested_end_date = datetime.now().strftime("%Y-%m-%d")
    
    ensure_landcover_table_exists(engine)
    
    # Lag
    lag_steps = features_cfg.get('temporal', {}).get('publication_lags', {}).get('DynamicWorld', 1)
    step_days = features_cfg.get('temporal', {}).get('step_days', 14)
    lag_days = lag_steps * step_days
    
    init_gee(project_id)
    ee.data.setDeadline(600)
    
    # -------------------------------------------------------------------------
    # INCREMENTAL LOADING: Use centralized helper
    # -------------------------------------------------------------------------
    start, end = get_incremental_window(
        engine=engine,
        table=TARGET_TABLE,
        date_col="date",
        requested_end_date=requested_end_date,
        default_start_date=DEFAULT_START_DATE,
        force_full=full_refresh,
        schema=SCHEMA
    )
    
    if start is None:
        logger.info("✅ Dynamic World data already up to date. No fetch needed.")
        return
    
    # Apply overlap buffer for late-arriving/reprocessed data
    if not full_refresh:
        effective_start_date = pd.to_datetime(start) - timedelta(days=OVERLAP_BUFFER_DAYS)
    else:
        effective_start_date = pd.to_datetime(DEFAULT_START_DATE)
    
    # Ensure we don't go before Sentinel-2B launch
    effective_start_date = max(effective_start_date, pd.Timestamp(DEFAULT_START_DATE))
    
    logger.info(f"Start Date: {effective_start_date.date()} (Sentinel-2B)")
    logger.info(f"End Date: {end}")
    
    all_cells = get_h3_grid_data(engine)
    logger.info(f"Loaded {len(all_cells)} H3 cells.")
    
    # Check already processed dates (for within-year incremental)
    try:
        existing = pd.read_sql(f"SELECT DISTINCT date FROM {SCHEMA}.{TARGET_TABLE}", engine)
        processed_dates = set(pd.to_datetime(existing['date']).dt.strftime('%Y-%m-%d'))
    except:
        processed_dates = set()

    end_year = pd.to_datetime(end).year

    for year in range(effective_start_date.year, end_year + 1):
        full_spine = build_14day_spine(year, year)
        valid_spine = [w for w in full_spine if w[0] >= effective_start_date.to_pydatetime() and w[0] <= pd.to_datetime(end)]
        
        windows_to_run = []
        for w in valid_spine:
            lagged_date = (w[0] + timedelta(days=lag_days)).strftime("%Y-%m-%d")
            if lagged_date not in processed_dates:
                windows_to_run.append(w)
        
        if windows_to_run:
            process_year_batch(year, all_cells, windows_to_run, engine, lag_days)
        else:
            logger.info(f"Year {year} already processed.")

    # Prune rows before cutoff
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {SCHEMA}.{TARGET_TABLE} WHERE date < :cutoff"), {"cutoff": CUTOFF_DATE})
    
    logger.info("✓ Dynamic World full ingestion complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Dynamic World land cover data")
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Force full refresh (ignore existing data)"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Alias for --full"
    )
    args = parser.parse_args()
    
    force_full = args.full or args.no_incremental
    
    try:
        cfgs = load_configs()
        configs = cfgs if isinstance(cfgs, dict) else {"data": cfgs[0], "features": cfgs[1]}
        eng = get_db_engine()
        run(configs, eng, full_refresh=force_full)
    finally:
        if 'eng' in locals(): eng.dispose()
