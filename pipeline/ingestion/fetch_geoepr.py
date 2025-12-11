"""
fetch_geoepr.py
===============================
Efficient ingestion of GeoEPR polygons (ETH Zürich).

FIXES APPLIED:
1. TEMPORAL FILTER: Excludes polygons that expired before 2000.
2. DEDUPLICATION: Keeps only the most recent polygon for each group ID to prevent PK violations.
3. UPSERT MODE: Uses upload_to_postgis for safe database insertion.
"""

import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests
from sqlalchemy import text

# --- Import central utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, load_configs, get_db_engine, upload_to_postgis, get_boundary

SCHEMA = "car_cewp"
TABLE_POLYGONS = "geoepr_polygons"


# ------------------------------------------------------------
# Helper: Download With Cache
# ------------------------------------------------------------
def download_if_needed(url: str, cache_path: Path) -> Path:
    if cache_path.exists():
        logger.info(f"Using cached GeoEPR file: {cache_path}")
        return cache_path
    
    logger.info(f"Downloading GeoEPR dataset from: {url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(cache_path, 'wb') as f:
            f.write(r.content)
        return cache_path
    except Exception as e:
        logger.error(f"Failed to download GeoEPR: {e}")
        raise

# ------------------------------------------------------------
# Schema Management
# ------------------------------------------------------------
def ensure_table_exists(engine):
    """Creates the geoepr_polygons table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_POLYGONS} (
                gwgroupid BIGINT PRIMARY KEY,
                group_name TEXT,
                geometry GEOMETRY(Geometry, 4326),
                "from" INTEGER,
                "to" INTEGER,
                type TEXT
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_POLYGONS}_geom ON {SCHEMA}.{TABLE_POLYGONS} USING GIST (geometry);"))

# ------------------------------------------------------------
# Main Polygon Ingestion
# ------------------------------------------------------------
def ingest_geoepr_polygons(geoepr_url: str, boundary_gdf, engine):

    cache_path = PATHS["cache"] / "GeoEPR-2021.geojson"
    local_file = download_if_needed(geoepr_url, cache_path)

    # 1. Load Polygons
    logger.info("Loading GeoEPR polygons...")
    try:
        # Load entire file first to ensure we can filter correctly
        gdf = gpd.read_file(local_file)
    except Exception as e:
        logger.error(f"Failed to read GeoEPR file: {e}")
        raise

    # 2. Filter & Deduplicate (CRITICAL FIX)
    logger.info(f"  Raw polygons: {len(gdf)}")

    # A. Temporal Filter: Keep only polygons valid during/after 2000
    if 'to' in gdf.columns:
        gdf = gdf[gdf['to'] >= 2000].copy()
    
    # B. Deduplication: Sort by 'from' year descending (newest first)
    if 'from' in gdf.columns:
        gdf = gdf.sort_values(by='from', ascending=False)
    
    # C. Keep only the newest record for each gwgroupid
    gdf = gdf.drop_duplicates(subset=['gwgroupid'], keep='first')
    logger.info(f"  Polygons after temporal filter & deduplication: {len(gdf)}")

    # 3. Spatial Processing
    if gdf.crs != boundary_gdf.crs:
        gdf = gdf.to_crs(boundary_gdf.crs)

    logger.info("Clipping polygons to CAR boundary...")
    gdf = gpd.clip(gdf, boundary_gdf)

    if gdf.empty:
        logger.warning("No GeoEPR polygons found inside CAR boundary after clipping.")
        return

    # 4. Validate Geometries
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    
    # 5. Prepare for Upload
    ensure_table_exists(engine)

    # Convert to DataFrame for upload helper
    df = pd.DataFrame(gdf)
    df['geometry'] = df['geometry'].apply(lambda x: x.wkt)
    
    # Handle column naming (Map 'group' -> 'group_name')
    if 'group' in df.columns: 
        df.rename(columns={'group': 'group_name'}, inplace=True)
    
    # Select strict columns matching schema
    target_cols = ['gwgroupid', 'group_name', 'geometry', 'from', 'to', 'type']
    available_cols = [c for c in target_cols if c in df.columns]
    
    # 6. Upload (Upsert Mode)
    logger.info(f"Uploading {len(df)} polygons to {SCHEMA}.{TABLE_POLYGONS} (Upsert Mode)...")
    
    upload_to_postgis(
        engine, 
        df[available_cols], 
        TABLE_POLYGONS, 
        SCHEMA, 
        primary_keys=['gwgroupid']
    )

    logger.info(f"Ingestion complete.")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def run():
    engine = None
    try:
        logger.info("=== Fetching GeoEPR Polygons ===")
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()

        geoepr_url = data_config["epr"]["geoepr_url"]
        boundary_gdf = get_boundary(data_config, features_config)

        ingest_geoepr_polygons(geoepr_url, boundary_gdf, engine)

        logger.info("=== GeoEPR polygon ingestion complete ===")

    except Exception as e:
        logger.error(f"GeoEPR polygon ingestion failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()

def main():
    run()

if __name__ == "__main__":
    main()