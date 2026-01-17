"""
fetch_dynamic_world.py
======================
Ingests Dynamic World land cover fractions from Google Earth Engine.

Dynamic World provides near real-time 10m land cover classification from Sentinel-2.
For H3 r5 cells (~252 km²), we compute mean probability for each class.

Features extracted:
- dw_grass_frac: Grassland fraction (pastoralist zones)
- dw_crops_frac: Cropland fraction (farmer-herder interface)
- dw_trees_frac: Forest fraction (remote/rebel areas)
- dw_bare_frac: Bare ground fraction (mining proxy)

Publication lag: 1 step (Sentinel-2 has ~2-5 day latency)
Temporal coverage: 2015-06-27 to present
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

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis, ensure_h3_int64, init_gee

# --- Constants ---
SCHEMA = "car_cewp"
GRID_TABLE = "features_static"
TARGET_TABLE = "landcover_features"

MAX_WORKERS = 8
H3_BATCH_SIZE = 200
MAX_RETRIES = 5

# Dynamic World collection
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

# Classes we care about (indices in DW probability bands)
# Full list: water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice
DW_CLASSES = {
    'grass': 2,      # Grassland/savanna - pastoralist zones
    'crops': 4,      # Cultivated land - farmer zones
    'trees': 1,      # Forest/woodland
    'bare': 7,       # Bare ground - mining proxy
    'built': 6,      # Built-up area
}


# -------------------------------------------------------------------------
# 1. DATABASE & HELPERS
# -------------------------------------------------------------------------

def ensure_landcover_table_exists(engine, schema="car_cewp"):
    from sqlalchemy import text
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
    logger.info(f"✓ Table {schema}.{TARGET_TABLE} is ready.")


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
    Dynamic World uses Sentinel-2 with ~2-5 day latency.
    """
    temporal_cfg = features_cfg.get('temporal', {})
    step_days = temporal_cfg.get('step_days', 14)
    pub_lags = temporal_cfg.get('publication_lags', {})
    
    # Use DynamicWorld key, fallback to GEE default
    lag_steps = pub_lags.get('DynamicWorld', pub_lags.get('GEE', 1))
    lag_days = lag_steps * step_days
    
    logger.info(f"Dynamic World publication lag: {lag_steps} steps × {step_days} days = {lag_days} days")
    return lag_days


def get_collection_last_date():
    """Queries GEE to find the last available date for Dynamic World."""
    try:
        last_img = ee.ImageCollection(DW_COLLECTION).limit(1, 'system:time_start', False).first()
        last_ts = last_img.get('system:time_start').getInfo()
        if last_ts:
            return datetime.fromtimestamp(last_ts / 1000.0, tz=timezone.utc).replace(tzinfo=None)
    except Exception as e:
        logger.warning(f"Could not verify end date for Dynamic World: {e}")
    return None


def get_collection_first_date():
    """Dynamic World starts 2015-06-27."""
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
# 3. GEE PROCESSING
# -------------------------------------------------------------------------

def process_single_batch(batch_cells, s_str, e_excl_str):
    """
    Process a batch of H3 cells for a single time window.
    
    Dynamic World provides probability bands for each class.
    We compute mean probability (fraction) per H3 cell.
    """
    batch_fc = make_ee_feature_collection(batch_cells)
    
    for attempt in range(MAX_RETRIES):
        try:
            # Filter Dynamic World to time window
            dw = ee.ImageCollection(DW_COLLECTION).filterDate(s_str, e_excl_str)
            
            # Check if any images exist in this window
            count = dw.size().getInfo()
            if count == 0:
                # No data for this window - return None values
                return None
            
            # Select probability bands for our classes
            # Dynamic World bands: water, trees, grass, flooded_vegetation, crops,
            #                      shrub_and_scrub, built, bare, snow_and_ice
            dw_mean = dw.select(['grass', 'crops', 'trees', 'bare', 'built']).mean()
            
            # Reduce to H3 cells
            results = dw_mean.reduceRegions(
                collection=batch_fc,
                reducer=ee.Reducer.mean(),
                scale=100,  # 100m for reasonable performance
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
    """Process all windows for a single year."""
    logger.info(f"--- Processing {len(windows)} windows for Year {year} ---")
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
                        
                        # PUBLICATION LAG: Shift date forward
                        lagged_date = start_dt + timedelta(days=lag_days)
                        
                        daily_rows.append({
                            "h3_index": h3_idx,
                            "date": lagged_date.strftime("%Y-%m-%d"),
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
            df["h3_index"] = df["h3_index"].apply(ensure_h3_int64)
            df = df.dropna(subset=["h3_index"])
            df["h3_index"] = df["h3_index"].astype("int64")
            
            # Upload to DB
            upload_to_postgis(engine, df, TARGET_TABLE, SCHEMA, primary_keys=["h3_index", "date"])
        
        gc.collect()


def run(configs, engine):
    """Main entry point for Dynamic World ingestion."""
    logger.info("=" * 60)
    logger.info("DYNAMIC WORLD LAND COVER INGESTION")
    logger.info("=" * 60)
    
    data_cfg = configs["data"]
    features_cfg = configs["features"]
    project_id = data_cfg["gee"]["project_id"]
    g_start = int(data_cfg["global_date_window"]["start_date"][:4])
    g_end = int(data_cfg["global_date_window"]["end_date"][:4])
    
    ensure_landcover_table_exists(engine)
    
    # Get publication lag from config
    lag_days = get_publication_lag(features_cfg)
    
    # Authenticate GEE
    init_gee(project_id)
    
    # Check data availability
    dw_start = get_collection_first_date()
    dw_end = get_collection_last_date()
    
    if dw_end:
        logger.info(f"✓ Dynamic World available: {dw_start.date()} to {dw_end.date()}")
    else:
        logger.warning("Could not determine Dynamic World end date. Proceeding with config dates.")
        dw_end = datetime(g_end, 12, 31)
    
    # Adjust start year based on DW availability (2015-06-27)
    effective_start = max(g_start, 2015)  # Use 2015 to include partial first year
    if g_start < 2015:
        logger.warning("Dynamic World starts 2015-06-27. Skipping years before 2015.")
    
    # Load H3 grid
    all_cells = get_h3_grid_data(engine)
    
    # Check what's already processed
    try:
        existing = pd.read_sql(
            f"SELECT DISTINCT date FROM {SCHEMA}.{TARGET_TABLE}", 
            engine
        )
        processed_dates = set(pd.to_datetime(existing['date']).dt.strftime('%Y-%m-%d'))
        logger.info(f"Found {len(processed_dates)} dates already processed.")
    except:
        processed_dates = set()
    
    # Process each year
    for year in range(effective_start, g_end + 1):
        full_spine = build_14day_spine(year, year)
        
        # Filter to valid windows (within DW availability)
        valid_spine = [w for w in full_spine if w[0] >= dw_start and w[0] <= dw_end]
        
        if not valid_spine:
            logger.info(f"Skipping Year {year} (no valid windows)")
            continue
        
        # Filter out already processed
        windows_to_run = []
        for w in valid_spine:
            lagged_date = (w[0] + timedelta(days=lag_days)).strftime("%Y-%m-%d")
            if lagged_date not in processed_dates:
                windows_to_run.append(w)
        
        if windows_to_run:
            process_year_batch(year, all_cells, windows_to_run, engine, lag_days)
        else:
            logger.info(f"Year {year} complete (all windows processed).")
    
    logger.info("✓ Dynamic World ingestion complete.")


def main():
    """CLI entry point."""
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
