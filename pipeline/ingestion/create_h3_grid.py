"""
create_h3_grid.py
============================
Purpose: Generate master H3 grid for CAR study area.
Output: car_cewp.features_static table with h3_index + geometry columns.

AUDIT FIX:
- Replaced to_postgis(replace) with upload_to_postgis(upsert).
- Added ensure_table_exists.
"""
import sys
import geopandas as gpd
import pandas as pd
import h3.api.basic_int as h3
from h3 import LatLngPoly
from sqlalchemy import text
from shapely.geometry import Polygon as ShapelyPolygon
from pathlib import Path

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, get_boundary, upload_to_postgis

SCHEMA = "car_cewp"
STATIC_TABLE = "features_static"

def check_grid_exists(engine):
    with engine.connect() as conn:
        exists = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{SCHEMA}' AND table_name = '{STATIC_TABLE}'
            );
        """)).scalar()
        if exists:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE}")).scalar()
            return True, count
    return False, 0

def generate_h3_grid(boundary_gdf, buffer_km, resolution, metric_crs, geodetic_crs):
    logger.info(f"Generating H3 grid (Res: {resolution}, Buffer: {buffer_km}km)...")
    boundary_metric = boundary_gdf.to_crs(metric_crs)
    buffered_metric = boundary_metric.buffer(buffer_km * 1000)
    buffered_geodetic = buffered_metric.to_crs(geodetic_crs)
    
    try: buffered_geom = buffered_geodetic.union_all()
    except AttributeError: buffered_geom = buffered_geodetic.unary_union

    polys = list(buffered_geom.geoms) if buffered_geom.geom_type == 'MultiPolygon' else [buffered_geom]
    all_h3_ints = set()
    for poly in polys:
        try:
            exterior = [(y, x) for x, y in poly.exterior.coords]
            holes = [[(y, x) for x, y in interior.coords] for interior in poly.interiors]
            h3_poly = LatLngPoly(exterior, *holes)
            all_h3_ints.update(h3.polygon_to_cells(h3_poly, resolution))
        except Exception: continue

    if not all_h3_ints: raise ValueError("No H3 cells generated.")
    return all_h3_ints

def create_geometries(h3_cells):
    geometries = []
    valid_cells = []
    for cell in h3_cells:
        if not h3.is_valid_cell(cell) or cell <= 0: continue
        try:
            boundary = h3.cell_to_boundary(cell)
            geom = ShapelyPolygon([(lng, lat) for lat, lng in boundary])
            geometries.append(geom)
            valid_cells.append(cell)
        except Exception: continue
    return valid_cells, geometries

def ensure_table_exists(engine):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{STATIC_TABLE} (
                h3_index BIGINT PRIMARY KEY,
                geometry GEOMETRY(Polygon, 4326)
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{STATIC_TABLE}_geom ON {SCHEMA}.{STATIC_TABLE} USING GIST (geometry);"))

def upload_grid_to_postgis(gdf, engine):
    logger.info(f"Uploading {len(gdf):,} cells to {SCHEMA}.{STATIC_TABLE} (Upsert)...")
    ensure_table_exists(engine)
    
    # Convert to DF and WKT
    df = pd.DataFrame(gdf)
    df['geometry'] = df['geometry'].apply(lambda x: x.wkt)
    df['h3_index'] = df['h3_index'].astype('int64')
    
    upload_to_postgis(engine, df, STATIC_TABLE, SCHEMA, primary_keys=['h3_index'])
    logger.info("Upload complete.")

def main():
    engine = None
    try:
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        
        buffer_km = features_config["spatial"]["buffer_km"]
        resolution = features_config["spatial"]["h3_resolution"]
        metric_crs = features_config["spatial"]["crs"]["metric"]
        geodetic_crs = features_config["spatial"]["crs"]["geodetic"]
        
        logger.info("="*60)
        logger.info("STEP 1: H3 GRID GENERATION")
        logger.info("="*60)
        
        exists, count = check_grid_exists(engine)
        if exists:
            logger.info(f"✓ Grid already exists ({count:,} cells). Skipping.")
            return
        
        boundary_gdf = get_boundary(data_config, features_config)
        h3_cells = generate_h3_grid(boundary_gdf, buffer_km, resolution, metric_crs, geodetic_crs)
        valid_cells, geometries = create_geometries(h3_cells)
        
        gdf_grid = gpd.GeoDataFrame(
            {"h3_index": valid_cells, "geometry": geometries}, 
            crs=geodetic_crs
        )
        
        upload_grid_to_postgis(gdf_grid, engine)
        
        logger.info("="*60)
        logger.info("✓ GRID GENERATION SUCCESSFUL")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"✗ Grid generation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine: engine.dispose()

if __name__ == "__main__":
    main()