"""
fetch_geoepr.py
===============================
Efficient ingestion of GeoEPR polygons (ETH Zürich).

- Downloads GeoEPR GeoJSON (cached)
- Clips by CAR boundary
- Uploads to PostGIS (Replace Mode)

FIXES APPLIED:
- Fixed SyntaxError in main() (orphaned finally block).
- Added proper resource management (engine.dispose) in run().
"""

import sys
from pathlib import Path
import geopandas as gpd
import requests
from sqlalchemy import text, inspect

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
# Main Polygon Ingestion
# ------------------------------------------------------------
def ingest_geoepr_polygons(geoepr_url: str, boundary_gdf, engine):

    cache_path = PATHS["cache"] / "GeoEPR-2021.geojson"
    local_file = download_if_needed(geoepr_url, cache_path)

    # 1. Load & Filter
    # Using bbox for speed, though GeoJSON reading is often full-scan
    bbox = boundary_gdf.total_bounds
    logger.info("Loading GeoEPR polygons...")
    
    try:
        gdf = gpd.read_file(local_file, bbox=bbox)
    except Exception:
        gdf = gpd.read_file(local_file)

    if gdf.crs != boundary_gdf.crs:
        gdf = gdf.to_crs(boundary_gdf.crs)

    # 2. Clip to CAR
    logger.info("Clipping polygons to CAR boundary...")
    gdf = gpd.clip(gdf, boundary_gdf)

    if gdf.empty:
        raise RuntimeError("GeoEPR returned no polygons in CAR.")

    # 3. Validate Geometries
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    
    # 4. Upload (Replace Mode)
    logger.info(f"Uploading {len(gdf)} polygons to {SCHEMA}.{TABLE_POLYGONS} (Replace Mode)...")
    
    # Determine PK
    # GeoEPR usually has 'gwgroupid' and 'from', 'to', but we treat it as static polygons here
    # or use year if available.
    
    # Write Table (Handles Creation)
    gdf.to_postgis(TABLE_POLYGONS, engine, schema=SCHEMA, if_exists='replace', index=False)
    
    # Add Indexes & Constraints
    with engine.begin() as conn:
        try:
            # Note: GeoEPR polygons might overlap over time, so a simple PK on groupid might fail 
            # if multiple years are present. For now, we index geometry.
            conn.execute(text(f"CREATE INDEX idx_{TABLE_POLYGONS}_geom ON {SCHEMA}.{TABLE_POLYGONS} USING GIST (geometry);"))
            logger.info("Table optimized with Spatial Index.")
        except Exception as e:
            logger.warning(f"Constraint application warning (ignorable): {e}")

    logger.info(f"Ingestion complete.")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def run():
    engine = None
    try:
        logger.info("=== Fetching GeoEPR Polygons ===")
        # Load configs locally since main.py calls this without args
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
    """Entry point wrapper."""
    run()

if __name__ == "__main__":
    main()