"""
fetch_population.py
========================
Purpose: Download WorldPop and aggregate to H3 using FAST zonal sums.
Output: Populates 'car_cewp.population_h3' table (h3_index, year, pop_count).

OPTIMIZATION:
- Memory Safe: Loads H3 grid in chunks (50k cells) to prevent OOM.
- Idempotent: Uses DB-based caching to skip completed years.
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import requests
from sqlalchemy import text
from pathlib import Path

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
GRID_CHUNK_SIZE = 50000  # Process 50k cells at a time

# Range of years to process
YEARS_TO_PROCESS = list(range(2000, 2026))

# -------------------------------------------------------------------------
# 1. FILENAME LOGIC
# -------------------------------------------------------------------------
def get_expected_filename(year):
    if year < 2015:
        return f"caf_ppp_{year}_UNadj.tif"
    else:
        return f"caf_pop_{year}_CN_100m_R2025A_v1.tif"

def get_worldpop_url(year, filename):
    base_v1 = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020"
    base_v2 = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A"
    
    if year < 2015:
        return f"{base_v1}/CAF/{filename}"
    else:
        return f"{base_v2}/CAF/{filename}"

def get_raster_path(year):
    filename = get_expected_filename(year)
    local_path = PATHS["data_raw"] / "worldpop" / filename
    
    if local_path.exists():
        logger.info(f"  Found local raster: {local_path.name}")
        return local_path
    
    if (PATHS["data_raw"] / filename).exists():
        return PATHS["data_raw"] / filename

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = get_worldpop_url(year, filename)
    logger.info(f"  Local not found. Downloading {year} from {url}...")
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"  Download successful: {filename}")
            return local_path
        else:
            logger.warning(
                f"  Download failed for {year} (Status {r.status_code}). "
                f"URL: {url}. Please manually download the file to data/raw/worldpop/"
            )
            return None
    except Exception as e:
        logger.warning(
            f"  Download error for {year}: {e}. "
            "Automated retrieval failed. Please manually download the file to data/raw/worldpop/"
        )
        return None

# -------------------------------------------------------------------------
# 2. MEMORY-SAFE GRID ITERATOR
# -------------------------------------------------------------------------
def iter_h3_grid_chunks(engine, chunk_size=GRID_CHUNK_SIZE):
    """
    Yields GeoDataFrames of the H3 grid in chunks to avoid OOM.
    """
    # Get total count
    with engine.connect() as conn:
        total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.features_static")).scalar()
    
    if not total_rows:
        return

    logger.info(f"  Processing {total_rows:,} H3 cells in chunks of {chunk_size:,}...")
    
    for offset in range(0, total_rows, chunk_size):
        query = f"""
            SELECT h3_index, geometry 
            FROM {SCHEMA}.features_static 
            ORDER BY h3_index 
            LIMIT {chunk_size} OFFSET {offset}
        """
        gdf = gpd.read_postgis(query, engine, geom_col="geometry")
        if not gdf.empty:
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            yield gdf

# -------------------------------------------------------------------------
# 3. ZONAL STATS
# -------------------------------------------------------------------------
def compute_zonal_sum_chunk(raster_path, h3_gdf):
    """
    Computes sum of population per H3 cell for a single chunk.
    """
    results = []
    
    with rasterio.open(raster_path) as src:
        # CRS Check & Reprojection
        if h3_gdf.crs != src.crs:
            h3_aligned = h3_gdf.to_crs(src.crs)
        else:
            h3_aligned = h3_gdf

        nodata = src.nodata if src.nodata is not None else -9999
        
        for _, row in h3_aligned.iterrows():
            geom = [row["geometry"]]
            try:
                out_image, _ = mask(src, geom, crop=True)
                data = out_image[0]
                
                if np.isnan(nodata):
                    valid_data = data[~np.isnan(data)]
                else:
                    valid_data = data[data != nodata]
                
                pop_sum = np.sum(valid_data) if valid_data.size > 0 else 0.0
                
                results.append({
                    "h3_index": row["h3_index"],
                    "pop_count": float(pop_sum),
                })
            except Exception:
                results.append({
                    "h3_index": row["h3_index"],
                    "pop_count": 0.0,
                })

    return pd.DataFrame(results)

def worldpop_year_in_db(engine, year: int) -> bool:
    """Returns True if this year is already populated."""
    sql = f"SELECT 1 FROM {SCHEMA}.{TABLE_NAME} WHERE year = :year LIMIT 1"
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"year": year}).scalar()
    return result is not None

# -------------------------------------------------------------------------
# 4. ORCHESTRATOR
# -------------------------------------------------------------------------
def run(configs, engine, force_recompute: bool = False):
    logger.info("=" * 60)
    logger.info("WORLDPOP INGESTION (Memory Safe)")
    logger.info("=" * 60)
    
    # Ensure table exists
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                h3_index BIGINT NOT NULL,
                year INTEGER NOT NULL,
                pop_count DOUBLE PRECISION,
                PRIMARY KEY (h3_index, year)
            );
        """))

    for year in YEARS_TO_PROCESS:
        if not force_recompute and worldpop_year_in_db(engine, year):
            logger.info(f"Year {year} cached; skipping.")
            continue

        logger.info(f"Processing Year {year}...")
        raster_path = get_raster_path(year)
        if not raster_path:
            continue

        # Process in Chunks
        year_total_rows = 0
        for gdf_chunk in iter_h3_grid_chunks(engine):
            df_pop = compute_zonal_sum_chunk(raster_path, gdf_chunk)
            
            if not df_pop.empty:
                df_pop["year"] = year
                df_pop["h3_index"] = df_pop["h3_index"].astype("int64")
                
                upload_to_postgis(
                    engine,
                    df_pop,
                    TABLE_NAME,
                    SCHEMA,
                    primary_keys=["h3_index", "year"],
                )
                year_total_rows += len(df_pop)
        
        logger.info(f"✓ Uploaded {year_total_rows:,} rows for {year}.")

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
        if "engine" in locals():
            engine.dispose()

if __name__ == "__main__":
    main()