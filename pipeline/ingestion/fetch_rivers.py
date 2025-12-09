"""
fetch_rivers.py
===========================
Fetches HydroSHEDS HydroRIVERS (Africa subset).
Fixes: 
1. Robust Zip Extraction (solves 'pyogrio' errors).
2. 'Replace' mode for DB upload (solves 'UndefinedColumn' errors).
3. CACHING: Skips download if table exists and is populated.
"""

import sys
import requests
import zipfile
import tempfile
import os
from pathlib import Path
import geopandas as gpd
from sqlalchemy import text, inspect

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, get_boundary

SCHEMA = "car_cewp"
TABLE = "rivers"

# ---------------------------------------------------------
# 1. DOWNLOAD + LOAD HYDROSHEDS (Robust Extraction)
# ---------------------------------------------------------

def download_hydrosheds(url: str) -> gpd.GeoDataFrame:
    logger.info("Downloading HydroSHEDS HydroRIVERS (Africa)...")

    # Use a temporary directory for the whole operation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        zip_path = temp_dir_path / "hydrorivers.zip"

        # 1. Download
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            logger.error(f"Failed to download HydroSHEDS: {e}")
            raise

        # 2. Extract
        logger.info("Extracting zip file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(temp_dir_path)
        except zipfile.BadZipFile:
            raise ValueError("Downloaded file is not a valid zip archive.")

        # 3. Find the .shp file
        shp_files = list(temp_dir_path.rglob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("No .shp file found in the downloaded zip.")
        
        target_shp = shp_files[0]
        logger.info(f"Found shapefile: {target_shp.name}")

        # 4. Read the Shapefile
        try:
            gdf = gpd.read_file(target_shp)
            logger.info(f"HydroSHEDS loaded: {len(gdf)} rows.")
            
            # Normalize columns immediately to lowercase
            gdf.columns = [c.lower() for c in gdf.columns]
            return gdf
        except Exception as e:
            logger.error(f"Failed to read shapefile {target_shp}: {e}")
            raise

# ---------------------------------------------------------
# 2. CLIP + FILTER STREAM ORDER
# ---------------------------------------------------------

def preprocess_rivers(gdf, boundary_gdf, min_order):
    """Clip to country boundary and filter small streams."""
    logger.info("Clipping HydroSHEDS to CAR boundary...")
    
    if gdf.crs != boundary_gdf.crs:
        gdf = gdf.to_crs(boundary_gdf.crs)

    clipped = gpd.clip(gdf, boundary_gdf)

    logger.info(f"Filtering HydroSHEDS by stream order >= {min_order}...")
    
    if "ord_stra" in clipped.columns:
        clipped = clipped[clipped["ord_stra"] >= min_order]
    else:
        logger.warning(f"ord_stra column missing. Columns: {list(clipped.columns)}")

    if clipped.empty:
        logger.warning("⚠ HydroSHEDS clipping produced an empty dataset!")
    else:
        logger.info(f"Remaining rivers: {len(clipped):,}")

    return clipped

# ---------------------------------------------------------
# 3. WRITE TO POSTGIS (REPLACE MODE)
# ---------------------------------------------------------

def upload_rivers(engine, gdf):
    """
    Uploads data in REPLACE mode.
    This deletes the old 'corrupted' table and writes a fresh one with the correct schema.
    """
    logger.info(f"Uploading to {SCHEMA}.{TABLE} (REPLACE mode)...")

    # Determine Primary Key
    if 'hyriv_id' in gdf.columns:
        pk_col = 'hyriv_id'
    else:
        gdf['river_id'] = gdf.index
        pk_col = 'river_id'
        logger.warning(f"'hyriv_id' not found. Using 'river_id' as PK.")

    # 1. Write Table (Destroys old table, creates new one with correct columns)
    gdf.to_postgis(TABLE, engine, schema=SCHEMA, if_exists='replace', index=False)
    
    # 2. Add Constraints & Indexes
    with engine.begin() as conn:
        try:
            # Add PK
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE} ADD PRIMARY KEY ({pk_col});"))
            # Add Spatial Index
            conn.execute(text(f"CREATE INDEX idx_{TABLE}_geom ON {SCHEMA}.{TABLE} USING GIST (geometry)"))
            # Cast H3 to BIGINT (Standardization)
            if 'h3_index' in gdf.columns:
                 conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE} ALTER COLUMN h3_index TYPE BIGINT USING CAST(h3_index AS BIGINT)"))
                 conn.execute(text(f"CREATE INDEX idx_{TABLE}_h3 ON {SCHEMA}.{TABLE} (h3_index)"))
                 
            logger.info("Table optimization complete (PK + Indexes applied).")
        except Exception as e:
            logger.warning(f"Non-critical index/constraint error: {e}")

    logger.info("HydroSHEDS ingestion complete.")

# ---------------------------------------------------------
# 4. MAIN PIPELINE FUNCTION
# ---------------------------------------------------------

def run(configs, engine):
    try:
        logger.info("=== HYDROSHEDS INGESTION START ===")
        
        # --- CACHING CHECK START ---
        inspector = inspect(engine)
        if inspector.has_table(TABLE, schema=SCHEMA):
            with engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{TABLE}")).scalar()
            
            # Heuristic: If we have > 1000 rivers, assume it's already done
            if count and count > 1000:
                logger.info(f"✓ Table {SCHEMA}.{TABLE} already exists ({count} rows). Skipping download.")
                logger.info("=== HYDROSHEDS INGESTION COMPLETE (CACHED) ===")
                return
        # --- CACHING CHECK END ---

        ds_cfg = configs["data"]["hydrorivers"]
        url = ds_cfg["url"]
        min_order = ds_cfg["min_stream_order"]

        boundary = get_boundary(configs["data"], configs["features"])

        gdf = download_hydrosheds(url)
        
        if gdf.empty:
            logger.warning('gdf is empty - skipping processing')
            return

        gdf = preprocess_rivers(gdf, boundary, min_order)
        
        if not gdf.empty:
            upload_rivers(engine, gdf)
        else:
            logger.warning("No rivers found to upload after preprocessing.")

        logger.info("=== HYDROSHEDS INGESTION COMPLETE ===")

    except Exception as e:
        logger.error(f"HydroSHEDS ingestion failed: {e}", exc_info=True)
        raise

def main():
    """Entry point wrapper."""
    from utils import load_configs, get_db_engine
    
    configs_tuple = load_configs()
    configs = {"data": configs_tuple[0], "features": configs_tuple[1], "models": configs_tuple[2]}
    engine = get_db_engine()
    
    try:
        run(configs, engine)
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    main()