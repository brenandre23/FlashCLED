"""
fetch_population.py
========================
Purpose: Download WorldPop and aggregate to H3 using FAST zonal sums.
Output: Populates 'car_cewp.population_h3' table (h3_index, year, pop_count).

FIXES APPLIED:
  1. Hybrid Filename Logic: 
     - 2000-2014: caf_ppp_{year}_UNadj.tif
     - 2015-2030: caf_pop_{year}_CN_100m_R2025A_v1.tif
  2. Priority Local Check: Checks data/raw/worldpop before attempting download.
  3. CRS Alignment: Explicitly reprojects H3 polygons to match Raster CRS.
  4. Fix upload_to_postgis signature mismatch (removed conflict_columns arg).
  5. DB-Only Cache: Skips years already present in car_cewp.population_h3 unless force_recompute=True.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import requests
from sqlalchemy import text

# Project imports
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import (
    logger,
    load_configs,
    get_db_engine,
    upload_to_postgis,
    PATHS,
)

SCHEMA = "car_cewp"
TABLE_NAME = "population_h3"

# Range of years to process (User has data 2000-2025, script can handle up to 2030 if avail)
YEARS_TO_PROCESS = list(range(2000, 2026))

# -------------------------------------------------------------------------
# 1. FILENAME LOGIC (The Fix)
# -------------------------------------------------------------------------
def get_expected_filename(year):
    """
    Returns the specific filename expected for the given year based on user structure.
    """
    if year < 2015:
        # Era 1: 2000 - 2014 (UN Adjusted)
        return f"caf_ppp_{year}_UNadj.tif"
    else:
        # Era 2: 2015+ (Constrained / R2025A)
        return f"caf_pop_{year}_CN_100m_R2025A_v1.tif"


def get_worldpop_url(year, filename):
    """
    Constructs a fallback download URL. 
    Note: Automatic download for older naming conventions is tricky, 
    but we attempt standard WorldPop paths if local is missing.
    """
    base_v1 = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020"
    base_v2 = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A"
    
    if year < 2015:
        # This is a guess for the UNadj URL structure, usually Country/Year...
        # If the file is missing locally, this might still 404, but it's a best effort.
        return f"{base_v1}/CAF/{filename}" 
    else:
        return f"{base_v2}/CAF/{filename}"


def get_raster_path(year):
    """
    Locates raster. Priority:
    1. Local Exact Match (data/raw/worldpop/{exact_name})
    2. Download attempt
    """
    filename = get_expected_filename(year)
    local_path = PATHS["data_raw"] / "worldpop" / filename
    
    # 1. Local Check
    if local_path.exists():
        logger.info(f"  Found local raster: {local_path.name}")
        return local_path
    
    # Debug: Check if maybe it's in the parent dir?
    if (PATHS["data_raw"] / filename).exists():
        return PATHS["data_raw"] / filename

    # 2. Download (Fallback)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = get_worldpop_url(year, filename)
    logger.info(f"  Local not found. Downloading {year} from {url}...")
    
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"  Download successful: {filename}")
            return local_path
        else:
            logger.warning(f"  Download failed for {year} (Status {r.status_code}). URL: {url}")
            return None
    except Exception as e:
        logger.warning(f"  Download error {year}: {e}")
        return None

# -------------------------------------------------------------------------
# 2. H3 & ZONAL STATS
# -------------------------------------------------------------------------
def load_h3_polygons(engine):
    """Load H3 grid as GeoDataFrame."""
    query = f"SELECT h3_index, geometry FROM {SCHEMA}.features_static"
    try:
        gdf = gpd.read_postgis(query, engine, geom_col="geometry")
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        return gdf
    except Exception as e:
        logger.error(f"Failed to load H3 grid: {e}")
        return gpd.GeoDataFrame()


def compute_zonal_sum(raster_path, h3_gdf):
    """
    Computes sum of population per H3 cell.
    """
    results = []
    
    with rasterio.open(raster_path) as src:
        # 1. CRS Check & Reprojection
        if h3_gdf.crs != src.crs:
            logger.info(f"  Reprojecting H3 polygons from {h3_gdf.crs} to {src.crs}...")
            h3_aligned = h3_gdf.to_crs(src.crs)
        else:
            h3_aligned = h3_gdf

        nodata = src.nodata if src.nodata is not None else -9999
        
        # 2. Iterate cells
        for _, row in tqdm(h3_aligned.iterrows(), total=len(h3_aligned), desc="Zonal Stats", leave=False):
            geom = [row['geometry']]
            
            try:
                out_image, _ = mask(src, geom, crop=True)
                data = out_image[0]
                
                # Mask out nodata
                if np.isnan(nodata):
                    valid_data = data[~np.isnan(data)]
                else:
                    valid_data = data[data != nodata]
                
                pop_sum = np.sum(valid_data) if valid_data.size > 0 else 0.0
                
                results.append({
                    'h3_index': row['h3_index'],
                    'pop_count': pop_sum
                })
            except Exception:
                results.append({
                    'h3_index': row['h3_index'],
                    'pop_count': 0.0
                })

    return pd.DataFrame(results)

# -------------------------------------------------------------------------
# 3. DB-ONLY CACHE HELPER
# -------------------------------------------------------------------------
def worldpop_year_in_db(engine, year: int) -> bool:
    """
    Returns True if there is at least one row for this year
    in the population_h3 table.
    """
    sql = f"""
        SELECT 1
        FROM {SCHEMA}.{TABLE_NAME}
        WHERE year = :year
        LIMIT 1
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"year": year}).scalar()
    return result is not None

# -------------------------------------------------------------------------
# 4. ORCHESTRATOR
# -------------------------------------------------------------------------
def run(configs, engine, force_recompute: bool = False):
    logger.info("="*60)
    logger.info("WORLDPOP INGESTION (Local Priority + DB Cache)")
    logger.info("="*60)
    
    # 1. Load H3 Grid
    h3_gdf = load_h3_polygons(engine)
    if h3_gdf.empty:
        logger.error("H3 Grid empty. Run create_h3_grid.py first.")
        return

    # Ensure table exists
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                h3_index BIGINT NOT NULL,
                year INTEGER NOT NULL,
                pop_count DOUBLE PRECISION,
                PRIMARY KEY (h3_index, year)
            );
        """))

    # 2. Process Years
    for year in YEARS_TO_PROCESS:
        # DB-only caching: skip years already present unless force_recompute=True
        if not force_recompute and worldpop_year_in_db(engine, year):
            logger.info(f"Year {year} already present in {SCHEMA}.{TABLE_NAME}; skipping recompute.")
            continue

        logger.info(f"Processing Year {year}...")
        
        # Locate Raster (Local First)
        raster_path = get_raster_path(year)
        
        if not raster_path: 
            logger.warning(f"Skipping {year} (Raster not found locally or remote).")
            continue

        # Compute
        df_pop = compute_zonal_sum(raster_path, h3_gdf)
        
        if df_pop.empty:
            logger.warning(f"Year {year} produced NO rows. Skipping upload.")
            continue
            
        df_pop['year'] = year
        
        # 3. Upload (Upsert)
        df_pop['h3_index'] = df_pop['h3_index'].astype('int64')
        
        upload_to_postgis(
            engine, 
            df_pop, 
            TABLE_NAME, 
            SCHEMA, 
            primary_keys=['h3_index', 'year']
        )
        logger.info(f"✓ Uploaded {len(df_pop)} rows for {year}.")

    logger.info("WorldPop Ingestion Complete.")


def main():
    try:
        cfg = load_configs()
        if isinstance(cfg, tuple):
            configs = {"data": cfg[0], "features": cfg[1], "models": cfg[2]}
        else:
            configs = cfg
            
        engine = get_db_engine()
        run(configs, engine)
    except Exception as e:
        logger.error(f"WorldPop failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.dispose()


if __name__ == "__main__":
    main()
