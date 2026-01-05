"""
fetch_grip4_roads.py
=================================================
Purpose: Download & ingest GRIP4 historical road network (Region 3) into PostGIS.
Output: car_cewp.grip4_roads_h3

AUDIT FIX:
- Replaced to_sql(replace) with upload_to_postgis(upsert).
- Added ensure_table_exists.
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

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, upload_to_postgis

SCHEMA = "car_cewp"
TABLE_NAME = "grip4_roads_h3"
CORES = max(1, multiprocessing.cpu_count() - 1)

def check_data_exists(engine):
    try:
        with engine.connect() as conn:
            exists = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{SCHEMA}' AND table_name = '{TABLE_NAME}'
                );
            """)).scalar()
            if not exists: return False
            count = conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.{TABLE_NAME}")).scalar()
            return count > 0
    except Exception:
        return False

def _poly_to_h3(geom, resolution):
    try:
        buffered = geom.buffer(0.0002) 
        if buffered.is_empty: return set()
        polys = list(buffered.geoms) if buffered.geom_type == 'MultiPolygon' else [buffered]
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
    for geom in chunk: out |= _poly_to_h3(geom, resolution)
    return out

def _parallel_convert(geoms, resolution):
    chunk_size = int(np.ceil(len(geoms) / CORES))
    chunks = [geoms[i:i + chunk_size] for i in range(0, len(geoms), chunk_size)]
    road_cells = set()
    with ProcessPoolExecutor(max_workers=CORES) as ex:
        futures = [ex.submit(_convert_chunk, chunk, resolution) for chunk in chunks]
        for f in futures: road_cells.update(f.result())
    return road_cells

def ensure_table_exists(engine):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                h3_index BIGINT PRIMARY KEY
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_h3 ON {SCHEMA}.{TABLE_NAME} (h3_index);"))

def main():
    engine = None
    try:
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        resolution = features_config["spatial"]["h3_resolution"]
        logger.info("=== GRIP4 Region 3 Road Ingestion ===")

        if check_data_exists(engine):
            logger.info(f"Table {SCHEMA}.{TABLE_NAME} already exists. Skipping.")
            return

        url = data_config["grip4"]["region3_url"]
        zip_cache = PATHS["cache"] / "GRIP4_Region3_vector_shp.zip"
        extract_dir = PATHS["cache"] / "grip4_extracted"
        extract_dir.mkdir(exist_ok=True)

        if not zip_cache.exists():
            logger.info(f"Downloading GRIP4 data...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(zip_cache, "wb") as f:
                for chunk in r.iter_content(8192): f.write(chunk)
        
        shp_files = list(extract_dir.rglob("*.shp"))
        if not shp_files:
            with zipfile.ZipFile(zip_cache, "r") as z: z.extractall(extract_dir)
            shp_files = list(extract_dir.rglob("*.shp"))
        
        if not shp_files: raise FileNotFoundError("No GRIP4 shapefile found.")

        logger.info(f"Reading {shp_files[0]}...")
        gdf = gpd.read_file(shp_files[0])
        if gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")

        logger.info(f"Converting {len(gdf):,} roads to H3 (Parallel)...")
        road_cells = _parallel_convert(list(gdf.geometry), resolution)
        
        # Upload using Upsert
        ensure_table_exists(engine)
        logger.info(f"Uploading {len(road_cells):,} cells to {SCHEMA}.{TABLE_NAME}...")
        df = pd.DataFrame({"h3_index": list(road_cells)})
        df['h3_index'] = df['h3_index'].astype('int64')
        
        upload_to_postgis(engine, df, TABLE_NAME, SCHEMA, primary_keys=['h3_index'])
        logger.info("=== GRIP4 Ingestion Complete ===")

    except Exception as e:
        logger.error(f"GRIP4 ingestion failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine: engine.dispose()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()