"""
fetch_rivers.py
===========================
Fetches HydroSHEDS HydroRIVERS (Africa subset).
Fixes: 
1. Robust Zip Extraction.
2. Idempotent Upsert (upload_to_postgis).
3. CACHING: Skips download if table exists and is populated.
4. Retry logic with exponential backoff for network requests.
"""

import sys
import zipfile
import tempfile
import pandas as pd
from pathlib import Path
import geopandas as gpd
from sqlalchemy import text, inspect

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import (
    logger, get_db_engine, load_configs, get_boundary, 
    upload_to_postgis, download_file_with_retry
)

SCHEMA = "car_cewp"
TABLE = "rivers"


# ---------------------------------------------------------
# 1. DOWNLOAD + LOAD HYDROSHEDS (WITH RETRY)
# ---------------------------------------------------------

def download_hydrosheds(url: str) -> gpd.GeoDataFrame:
    """
    Download HydroSHEDS data using retry-enabled helper.
    """
    logger.info("Downloading HydroSHEDS HydroRIVERS (Africa)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        zip_path = temp_dir_path / "hydrorivers.zip"

        # Use retry-enabled download helper
        try:
            download_file_with_retry(url, zip_path)
        except Exception as e:
            logger.error(f"Failed to download HydroSHEDS after retries: {e}")
            raise

        logger.info("Extracting zip file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(temp_dir_path)
        except zipfile.BadZipFile:
            raise ValueError("Downloaded file is not a valid zip archive.")

        shp_files = list(temp_dir_path.rglob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("No .shp file found in the downloaded zip.")
        
        target_shp = shp_files[0]
        try:
            gdf = gpd.read_file(target_shp)
            gdf.columns = [c.lower() for c in gdf.columns]
            return gdf
        except Exception as e:
            logger.error(f"Failed to read shapefile {target_shp}: {e}")
            raise


# ---------------------------------------------------------
# 2. CLIP + FILTER
# ---------------------------------------------------------

def preprocess_rivers(gdf, boundary_gdf, min_order):
    logger.info("Clipping HydroSHEDS to CAR boundary...")
    if gdf.crs != boundary_gdf.crs:
        gdf = gdf.to_crs(boundary_gdf.crs)

    clipped = gpd.clip(gdf, boundary_gdf)
    logger.info(f"Filtering HydroSHEDS by stream order >= {min_order}...")
    
    if "ord_stra" in clipped.columns:
        clipped = clipped[clipped["ord_stra"] >= min_order]
    
    if clipped.empty:
        logger.warning("⚠ HydroSHEDS clipping produced an empty dataset!")

    return clipped


# ---------------------------------------------------------
# 3. UPSERT TO POSTGIS
# ---------------------------------------------------------

def ensure_table_exists(engine, pk_col):
    """Creates rivers table if not exists."""
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE} (
                {pk_col} BIGINT PRIMARY KEY,
                geometry GEOMETRY(Geometry, 4326),
                ord_stra INTEGER,
                h3_index BIGINT
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_geom ON {SCHEMA}.{TABLE} USING GIST (geometry);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_h3 ON {SCHEMA}.{TABLE} (h3_index);"))


def upload_rivers(engine, gdf):
    logger.info(f"Uploading to {SCHEMA}.{TABLE} (Upsert mode)...")

    pk_col = 'hyriv_id' if 'hyriv_id' in gdf.columns else 'river_id'
    if pk_col == 'river_id' and 'river_id' not in gdf.columns:
        gdf['river_id'] = gdf.index

    ensure_table_exists(engine, pk_col)

    df = pd.DataFrame(gdf)
    df['geometry'] = df['geometry'].apply(lambda x: x.wkt)

    cols = [pk_col, 'geometry', 'ord_stra', 'h3_index']
    available = [c for c in cols if c in df.columns]

    upload_to_postgis(engine, df[available], TABLE, SCHEMA, primary_keys=[pk_col])
    logger.info("HydroSHEDS ingestion complete.")


# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------

def run(configs, engine):
    try:
        logger.info("=== HYDROSHEDS INGESTION START ===")
        inspector = inspect(engine)
        if inspector.has_table(TABLE, schema=SCHEMA):
            with engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{TABLE}")).scalar()
            if count and count > 1000:
                logger.info(f"✓ Table {SCHEMA}.{TABLE} already exists ({count} rows). Skipping.")
                return

        ds_cfg = configs["data"]["hydrorivers"]
        url = ds_cfg["url"]
        min_order = ds_cfg["min_stream_order"]
        boundary = get_boundary(configs["data"], configs["features"])

        gdf = download_hydrosheds(url)
        if gdf.empty: return

        gdf = preprocess_rivers(gdf, boundary, min_order)
        if not gdf.empty:
            upload_rivers(engine, gdf)
        else:
            logger.warning("No rivers found after preprocessing.")

        logger.info("=== HYDROSHEDS INGESTION COMPLETE ===")

    except Exception as e:
        logger.error(f"HydroSHEDS ingestion failed: {e}", exc_info=True)
        raise


def main():
    from utils import load_configs, get_db_engine
    cfgs = load_configs()
    configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
    engine = get_db_engine()
    try:
        run(configs, engine)
    finally:
        if engine: engine.dispose()


if __name__ == "__main__":
    main()
