"""
fetch_dynamic_world.py
======================
Ingests Dynamic World land cover fractions from Google Earth Engine.
Full Ingestion (Grass, Crops, Trees, Bare, Built).

- START DATE: 2017-03-07 (Sentinel-2B Launch)
- OPTIMIZED: Uses monkey-patched timeouts and robust error handling.
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

from utils import logger, get_db_engine, load_configs, upload_to_postgis, ensure_h3_int64, init_gee

# --- Constants ---
SCHEMA = "car_cewp"
GRID_TABLE = "features_static"
TARGET_TABLE = "landcover_features"

# --- UPDATED CUTOFF: Sentinel-2B Launch ---
CUTOFF_DATE = pd.Timestamp("2017-03-07")

MAX_WORKERS = 6
H3_BATCH_SIZE = 200
MAX_RETRIES = 5
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

# -------------------------------------------------------------------------
# DATABASE & HELPERS
# -------------------------------------------------------------------------

def ensure_landcover_table_exists(engine, schema="car_cewp"):
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

def run(configs, engine):
    logger.info("DYNAMIC WORLD FULL INGESTION (Optimized)")
    
    data_cfg = configs["data"]
    features_cfg = configs["features"]
    project_id = data_cfg["gee"]["project_id"]
    g_end = int(data_cfg["global_date_window"]["end_date"][:4])
    
    ensure_landcover_table_exists(engine)
    
    # Lag
    lag_steps = features_cfg['temporal']['publication_lags'].get('DynamicWorld', 1)
    lag_days = lag_steps * 14
    
    init_gee(project_id)
    ee.data.setDeadline(600)
    
    # START DATE: 2017-03-07
    effective_start_date = datetime(2017, 3, 7)
    
    all_cells = get_h3_grid_data(engine)
    logger.info(f"Loaded {len(all_cells)} H3 cells.")
    
    # Check processed
    try:
        existing = pd.read_sql(f"SELECT DISTINCT date FROM {SCHEMA}.{TARGET_TABLE}", engine)
        processed_dates = set(pd.to_datetime(existing['date']).dt.strftime('%Y-%m-%d'))
    except:
        processed_dates = set()
    
    logger.info(f"Start Date: {effective_start_date.date()} (Sentinel-2B)")

    for year in range(effective_start_date.year, g_end + 1):
        full_spine = build_14day_spine(year, year)
        valid_spine = [w for w in full_spine if w[0] >= effective_start_date]
        
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
    try:
        cfgs = load_configs()
        configs = cfgs if isinstance(cfgs, dict) else {"data": cfgs[0], "features": cfgs[1]}
        eng = get_db_engine()
        run(configs, eng)
    finally:
        if 'eng' in locals(): eng.dispose()