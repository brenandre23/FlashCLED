"""
01_build_static_features.py
============================
Purpose: Generate master H3 grid for CAR study area.
Output: car_cewp.features_static table with h3_index + geometry columns.

Checkpointing: Skips generation if grid already exists in database.
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
# Add the parent directory (scripts/) to sys.path to locate utils.py
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
import utils
from utils import logger

SCHEMA = "car_cewp"
STATIC_TABLE = "features_static"

def check_grid_exists(engine):
    """
    Check if the features_static table exists and is populated.
    """
    with engine.connect() as conn:
        # Check table existence in information_schema
        exists = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{SCHEMA}' 
                AND table_name = '{STATIC_TABLE}'
            );
        """)).scalar()
        
        if exists:
            # Check if it actually has data
            count = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE}")).scalar()
            return True, count
    
    return False, 0

def generate_h3_grid(boundary_gdf, buffer_km, resolution, metric_crs, geodetic_crs):
    """
    Generate H3 cells covering the boundary with a specified buffer.
    """
    logger.info(f"Generating H3 grid (Res: {resolution}, Buffer: {buffer_km}km)...")
    
    # 1. Project to Metric CRS for accurate buffering
    boundary_metric = boundary_gdf.to_crs(metric_crs)
    
    # 2. Buffer
    buffered_metric = boundary_metric.buffer(buffer_km * 1000) # km to meters
    
    # 3. Project back to Geodetic (Lat/Lon) for H3
    buffered_geodetic = buffered_metric.to_crs(geodetic_crs)
    
    # 4. Handle Geometry merging (Compat for Geopandas 1.0+ and older)
    try:
        buffered_geom = buffered_geodetic.union_all()
    except AttributeError:
        buffered_geom = buffered_geodetic.unary_union

    # Ensure we iterate over a list of polygons (handle MultiPolygon vs Polygon)
    if buffered_geom.geom_type == 'Polygon':
        polys = [buffered_geom]
    else:
        polys = list(buffered_geom.geoms)

    all_h3_ints = set()
    
    # 5. Fill polygons with H3 cells
    for poly in polys:
        try:
            # Extract exterior and interiors (holes)
            exterior = [(y, x) for x, y in poly.exterior.coords] # (lat, lng)
            holes = [[(y, x) for x, y in interior.coords] for interior in poly.interiors]
            
            h3_poly = LatLngPoly(exterior, *holes)
            new_cells = h3.polygon_to_cells(h3_poly, resolution)
            all_h3_ints.update(new_cells)
            
        except Exception as e:
            logger.warning(f"Polygon conversion failed: {e}")
            continue

    logger.info(f"Generated {len(all_h3_ints):,} raw cells.")
    
    if not all_h3_ints:
        raise ValueError("No H3 cells generated - check boundary geometry and CRS.")
    
    return all_h3_ints

def create_geometries(h3_cells):
    """
    Convert H3 indices to Shapely Polygons for storage/visualization.
    """
    logger.info("Creating geometries from H3 indices...")
    geometries = []
    valid_cells = []
    
    for cell in h3_cells:
        if not h3.is_valid_cell(cell) or cell <= 0:
            continue
            
        try:
            boundary = h3.cell_to_boundary(cell)
            # H3 returns (lat, lng), Shapely expects (lng, lat)
            geom = ShapelyPolygon([(lng, lat) for lat, lng in boundary])
            geometries.append(geom)
            valid_cells.append(cell)
        except Exception:
            continue
            
    return valid_cells, geometries

def upload_grid_to_postgis(gdf, engine):
    """
    Upload the grid to PostgreSQL/PostGIS.
    """
    logger.info(f"Uploading {len(gdf):,} cells to {SCHEMA}.{STATIC_TABLE}...")
    
    with engine.begin() as conn:
        # Ensure schema exists
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        # Drop table if exists to ensure clean state
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{STATIC_TABLE} CASCADE;"))
    
    # Write to PostGIS
    gdf.to_postgis(STATIC_TABLE, engine, schema=SCHEMA, if_exists="replace", index=False)
    
    # Create Indices
    with engine.begin() as conn:
        logger.info("Creating spatial indices...")
        conn.execute(text(f"CREATE INDEX idx_{STATIC_TABLE}_geom ON {SCHEMA}.{STATIC_TABLE} USING GIST (geometry);"))
        conn.execute(text(f"CREATE INDEX idx_{STATIC_TABLE}_h3 ON {SCHEMA}.{STATIC_TABLE} (h3_index);"))
        
    logger.info("Upload complete.")

def main():
    try:
        # 1. Load Configs
        data_config, features_config, _ = utils.load_configs()
        engine = utils.get_db_engine()
        
        # Extract parameters
        buffer_km = data_config["h3"]["buffer_km"]
        resolution = data_config["h3"]["resolution"]
        metric_crs = features_config["metric_crs"]
        geodetic_crs = features_config["geodetic_crs"]
        
        logger.info("="*60)
        logger.info("STEP 1: H3 GRID GENERATION")
        logger.info("="*60)
        
        # 2. Checkpoint
        exists, count = check_grid_exists(engine)
        if exists:
            logger.info(f" CHECKPOINT: Grid already exists ({count:,} cells).")
            logger.info("Skipping generation. To regenerate: TRUNCATE TABLE car_cewp.features_static;")
            return
        
        # 3. Get Boundary
        # utils.get_boundary handles downloading and caching the GeoJSON
        boundary_gdf = utils.get_boundary(data_config, features_config)
        
        # 4. Generate Cells
        h3_cells = generate_h3_grid(
            boundary_gdf, buffer_km, resolution, metric_crs, geodetic_crs
        )
        
        # 5. Create Geometries
        valid_cells, geometries = create_geometries(h3_cells)
        
        # 6. Create GeoDataFrame
        gdf_grid = gpd.GeoDataFrame(
            {"h3_index": valid_cells, "geometry": geometries}, 
            crs=geodetic_crs
        )
        
        # 7. Upload
        upload_grid_to_postgis(gdf_grid, engine)
        
        logger.info("="*60)
        logger.info("GRID GENERATION SUCCESSFUL")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Grid generation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()