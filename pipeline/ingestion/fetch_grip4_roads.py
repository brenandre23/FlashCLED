"""
fetch_grip4_roads.py
=================================================
Purpose: Download & ingest GRIP4 historical road network (Region 3) into PostGIS.
Output: car_cewp.grip4_roads_h3

OPTIMIZATION:
- Smart Caching: Checks DB first. If data exists, skips processing.
- Parallel Processing: Uses multiple cores for Geometry -> H3 conversion.
"""

import sys
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import h3.api.basic_int as h3
from h3 import LatLngPoly
from sqlalchemy import text
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing
import zipfile
import os

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, cast_h3_index

SCHEMA = "car_cewp"
TABLE_NAME = "grip4_roads_h3"
CORES = max(1, multiprocessing.cpu_count() - 1)


# ---------------------------------------------------------
# DB Check Helper
# ---------------------------------------------------------
def check_data_exists(engine):
    """Check if the table exists and has rows."""
    try:
        with engine.connect() as conn:
            # Check existence
            exists = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{SCHEMA}' 
                    AND table_name = '{TABLE_NAME}'
                );
            """)).scalar()
            
            if not exists:
                return False
            
            # Check row count (fast estimate)
            count = conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.{TABLE_NAME}")).scalar()
            return count > 0
    except Exception:
        return False


# ---------------------------------------------------------
# Worker: Convert polygons/lines → H3 cells
# ---------------------------------------------------------
def _poly_to_h3(geom, resolution):
    """Convert any GRIP4 geometry (LineString or Polygon) to H3 cells."""
    try:
        # Buffer lines slightly to turn them into polygons for H3 polyfill
        buffered = geom.buffer(0.0002) 
        if buffered.is_empty:
            return set()

        # Handle MultiPolygon vs Polygon
        if buffered.geom_type == 'MultiPolygon':
            polys = list(buffered.geoms)
        else:
            polys = [buffered]
            
        cells = set()
        for poly in polys:
            exterior = [(y, x) for x, y in poly.exterior.coords]
            holes = [[(y, x) for x, y in interior.coords] for interior in poly.interiors]
            h3_poly = LatLngPoly(exterior, *holes)
            cells.update(h3.polygon_to_cells(h3_poly, resolution))

        return {c for c in cells if c > 0}
    except Exception:
        return set()


def _convert_chunk(chunk, resolution):
    out = set()
    for geom in chunk:
        out |= _poly_to_h3(geom, resolution)
    return out


def _parallel_convert(geoms, resolution):
    chunk_size = int(np.ceil(len(geoms) / CORES))
    chunks = [geoms[i:i + chunk_size] for i in range(0, len(geoms), chunk_size)]

    road_cells = set()
    with ProcessPoolExecutor(max_workers=CORES) as ex:
        futures = [ex.submit(_convert_chunk, chunk, resolution) for chunk in chunks]
        for f in futures:
            road_cells.update(f.result())
    return road_cells


# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
def main():
    engine = None
    try:
        # FIXED: Load both data_config and features_config
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        # FIXED: Get resolution from features_config instead of data_config
        resolution = features_config["spatial"]["h3_resolution"]

        logger.info("=== GRIP4 Region 3 Road Ingestion ===")

        # 1. Smart Cache Check
        if check_data_exists(engine):
            logger.info(f"Table {SCHEMA}.{TABLE_NAME} already exists and is populated. Skipping.")
            return

        # 2. Setup Paths
        url = data_config["grip4"]["region3_url"]
        zip_cache = PATHS["cache"] / "GRIP4_Region3_vector_shp.zip"
        extract_dir = PATHS["cache"] / "grip4_extracted"
        extract_dir.mkdir(exist_ok=True)

        # 3. Download if missing
        if not zip_cache.exists():
            logger.info(f"Downloading GRIP4 data:\n {url}")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(zip_cache, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            logger.info("Download complete.")
        else:
            logger.info("Using cached GRIP4 ZIP.")

        # 4. Extract
        # Check if already extracted to save time
        shp_files = list(extract_dir.rglob("*.shp"))
        if not shp_files:
            logger.info("Extracting GRIP4 ZIP...")
            with zipfile.ZipFile(zip_cache, "r") as z:
                z.extractall(extract_dir)
            shp_files = list(extract_dir.rglob("*.shp"))

        if not shp_files:
            raise FileNotFoundError("No GRIP4 shapefile found after extraction.")

        shp_path = shp_files[0]
        logger.info(f"Reading shapefile: {shp_path}")

        # 5. Load & Convert
        gdf = gpd.read_file(shp_path)
        if gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        logger.info(f"{len(gdf):,} geometries loaded. Starting H3 conversion (Parallel)...")

        # 6. Convert to H3
        road_cells = _parallel_convert(list(gdf.geometry), resolution)
        logger.info(f"Generated {len(road_cells):,} unique road H3 cells.")

        # 7. Upload
        logger.info(f"Uploading to {SCHEMA}.{TABLE_NAME}...")
        df = pd.DataFrame({"h3_index": list(road_cells)})
        
        # Replace mode is fine here as it's a static dataset rebuild
        df.to_sql(TABLE_NAME, engine, schema=SCHEMA, if_exists="replace", index=False)

        # 8. Post-Upload Maintenance (Indexes & Types)
        with engine.begin() as conn:
            # Standardize to BIGINT first (consistent with rest of pipeline)
            conn.execute(text(f"""
                ALTER TABLE {SCHEMA}.{TABLE_NAME}
                ALTER COLUMN h3_index TYPE BIGINT
                USING (h3_index::bigint);
            """))
            # Add PK
            conn.execute(text(f"""
                ALTER TABLE {SCHEMA}.{TABLE_NAME} ADD PRIMARY KEY (h3_index);
            """))
            
            # Add GIST index for spatial ops if needed (requires cast back to h3index type for function support)
            # For now, basic B-Tree index on BIGINT is enough for joins
            conn.execute(text(f"CREATE INDEX idx_{TABLE_NAME}_h3 ON {SCHEMA}.{TABLE_NAME} (h3_index);"))

        logger.info("=== GRIP4 Ingestion Complete ===")

    except Exception as e:
        logger.error(f"GRIP4 ingestion failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()