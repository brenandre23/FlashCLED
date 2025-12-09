"""
calculate_static_distances.py
============================
Purpose: Calculates static distance features from H3 grid to various sources.

FIXES APPLIED (2025-11-24):
- FIXED: Uses explicit IPIS columns ('minerals_diamant', 'minerals_or') found in logs.
"""
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import text
import h3.api.basic_int as h3
from scipy.spatial import cKDTree

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from utils import logger, PATHS, get_db_engine, get_boundary, load_configs, upload_to_postgis

SCHEMA = "car_cewp"
STATIC_TABLE = "features_static"

def add_distance_columns(engine):
    """Ensure all distance columns exist."""
    cols = [
        "dist_to_capital FLOAT", "dist_to_border FLOAT", "dist_to_city FLOAT",
        "dist_to_road FLOAT", "dist_to_diamond_mine FLOAT", "dist_to_gold_mine FLOAT", 
        "dist_to_river FLOAT", "dist_to_lake FLOAT"
    ]
    with engine.begin() as conn:
        for c in cols:
            try: conn.execute(text(f"ALTER TABLE {SCHEMA}.{STATIC_TABLE} ADD COLUMN IF NOT EXISTS {c}"))
            except Exception: pass

def calculate_capital_distance(engine):
    with engine.connect() as conn:
        if conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE} WHERE dist_to_capital IS NOT NULL")).scalar() > 0:
            logger.info("  dist_to_capital already exists. Skipping.")
            return
    
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE}
            SET dist_to_capital = ST_Distance(
                ST_Centroid(geometry)::geography, ST_Point(18.5550, 4.3612)::geography
            ) / 1000.0
            WHERE dist_to_capital IS NULL
        """))
    logger.info("  ✓ Capital distance calculated.")

def calculate_border_distance(engine, configs):
    with engine.connect() as conn:
        if conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE} WHERE dist_to_border IS NOT NULL")).scalar() > 0:
            logger.info("  dist_to_border already exists. Skipping.")
            return

    boundary_gdf = get_boundary(configs['data'], configs['features'])
    try: boundary_line = boundary_gdf.union_all().boundary
    except AttributeError: boundary_line = boundary_gdf.unary_union.boundary
    
    border_gdf = gpd.GeoDataFrame({'geometry': [boundary_line]}, crs=boundary_gdf.crs)
    border_gdf.to_postgis("temp_border", engine, schema=SCHEMA, if_exists='replace', index=False)
    
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE} f
            SET dist_to_border = ST_Distance(ST_Centroid(f.geometry)::geography, b.geometry::geography) / 1000.0
            FROM {SCHEMA}.temp_border b WHERE f.dist_to_border IS NULL
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.temp_border"))
    logger.info("  ✓ Border distance calculated.")

# ---------------------------------------------------------
# SMART DISTANCE CALCULATION
# ---------------------------------------------------------
def calculate_point_distances_smart(engine, target_col, source_table, filter_condition=None, metric_crs="EPSG:32634"):
    logger.info(f"Calculating {target_col} (Smart H3 Mode)...")
    
    with engine.connect() as conn:
        if conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE} WHERE {target_col} IS NOT NULL")).scalar() > 0:
            logger.info(f"  {target_col} already exists. Skipping.")
            return

    # 1. Load Grid
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT h3_index FROM {SCHEMA}.{STATIC_TABLE}"))
            grid_df = pd.DataFrame([{'h3_index': row[0]} for row in result])
    except Exception: return
    
    if grid_df.empty: return

    # 2. Check Source Table
    with engine.connect() as conn:
        if not conn.execute(text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema='{SCHEMA}' AND table_name='{source_table}')")).scalar():
            logger.warning(f"  Table {source_table} not found.")
            return

    # 3. Load Source
    query = f"SELECT h3_index FROM {SCHEMA}.{source_table}"
    if filter_condition: query += f" WHERE {filter_condition}"
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            source_df = pd.DataFrame([{'h3_index': row[0]} for row in result])
    except Exception as e:
        logger.warning(f"  Error reading {source_table}: {e}")
        return

    # 4. Process H3
    def safe_h3_to_int(val):
        try:
            if pd.isna(val): return None
            if isinstance(val, (int, float, np.integer)): return int(val)
            if isinstance(val, str): return int(val, 16)
        except: return None
        return None

    grid_df['h3_index'] = grid_df['h3_index'].apply(safe_h3_to_int)
    source_df['h3_index'] = source_df['h3_index'].apply(safe_h3_to_int)
    grid_df.dropna(subset=['h3_index'], inplace=True)
    source_df.dropna(subset=['h3_index'], inplace=True)
    
    if source_df.empty:
        logger.warning(f"  No valid H3 in {source_table} after filtering.")
        return

    # 5. KDTree Calc
    grid_coords = [h3.cell_to_latlng(int(i)) for i in grid_df['h3_index']]
    source_coords = [h3.cell_to_latlng(int(i)) for i in source_df['h3_index']]
    
    grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p[1] for p in grid_coords], [p[0] for p in grid_coords]), crs="EPSG:4326").to_crs(metric_crs)
    source_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p[1] for p in source_coords], [p[0] for p in source_coords]), crs="EPSG:4326").to_crs(metric_crs)
    
    tree = cKDTree(np.array([[p.x, p.y] for p in source_gdf.geometry]))
    dists, _ = tree.query(np.array([[p.x, p.y] for p in grid_gdf.geometry]))
    
    grid_df[target_col] = dists / 1000.0
    upload_to_postgis(engine, grid_df[['h3_index', target_col]], STATIC_TABLE, SCHEMA, ['h3_index'])
    logger.info(f"  ✓ {target_col} complete.")

def calculate_infrastructure_distances(engine, metric_crs):
    calculate_point_distances_smart(engine, "dist_to_city", "osm_cities_h3", metric_crs=metric_crs)
    calculate_point_distances_smart(engine, "dist_to_road", "grip4_roads_h3", metric_crs=metric_crs)

def calculate_mine_distances(engine, metric_crs):
    """
    Uses specific IPIS columns: minerals_diamant, minerals_or, minerals
    """
    logger.info("Using IPIS columns for filtering...")
    
    # Logic: Either the boolean flag is True (1) OR the text column contains the word
    # Note: We cast to text to handle boolean/integer variations
    
    diamond_filter = """
        (minerals_diamant::text = '1' OR minerals_diamant::text = 'true') 
        OR LOWER(minerals) LIKE '%diamant%' 
        OR LOWER(minerals) LIKE '%diamond%'
    """
    
    gold_filter = """
        (minerals_or::text = '1' OR minerals_or::text = 'true') 
        OR LOWER(minerals) LIKE '%gold%' 
        OR LOWER(minerals) LIKE '%or%'
    """

    calculate_point_distances_smart(engine, "dist_to_diamond_mine", "mines_h3", diamond_filter, metric_crs)
    calculate_point_distances_smart(engine, "dist_to_gold_mine", "mines_h3", gold_filter, metric_crs)

def calculate_water_distances(engine, metric_crs):
    logger.info("Calculating water distances...")
    with engine.connect() as conn:
        if conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE} WHERE dist_to_river IS NOT NULL")).scalar() > 0:
            logger.info("  dist_to_river already exists. Skipping.")
        else:
            try:
                with engine.begin() as trans_conn:
                    trans_conn.execute(text(f"""
                        UPDATE {SCHEMA}.{STATIC_TABLE} f
                        SET dist_to_river = (
                            SELECT MIN(ST_Distance(ST_Centroid(f.geometry)::geography, r.geometry::geography)) / 1000.0
                            FROM {SCHEMA}.rivers r WHERE r.ord_stra >= 3
                        ) WHERE f.dist_to_river IS NULL
                    """))
                logger.info("  ✓ River distances calculated.")
            except Exception as e: logger.warning(f"  River calc failed: {e}")

        if conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE} WHERE dist_to_lake IS NOT NULL")).scalar() > 0:
            logger.info("  dist_to_lake already exists. Skipping.")
        else:
            try:
                with engine.begin() as trans_conn:
                    trans_conn.execute(text(f"""
                        UPDATE {SCHEMA}.{STATIC_TABLE} f
                        SET dist_to_lake = (
                            SELECT MIN(ST_Distance(ST_Centroid(f.geometry)::geography, l.geometry::geography)) / 1000.0
                            FROM {SCHEMA}.lakes l
                        ) WHERE f.dist_to_lake IS NULL
                    """))
                logger.info("  ✓ Lake distances calculated.")
            except Exception: pass

def main():
    engine = None
    try:
        logger.info("=" * 60)
        logger.info("STATIC DISTANCE CALCULATIONS (FIXED)")
        logger.info("=" * 60)
        
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        metric_crs = features_config["spatial"]["crs"]["metric"]
        geodetic_crs = features_config["spatial"]["crs"]["geodetic"]
        
        add_distance_columns(engine)
        calculate_capital_distance(engine)
        calculate_border_distance(engine, {'data': data_config, 'features': features_config})
        calculate_infrastructure_distances(engine, metric_crs)
        calculate_mine_distances(engine, metric_crs)
        calculate_water_distances(engine, metric_crs)
        
        logger.info("✓ ALL DISTANCE CALCULATIONS COMPLETE")
        
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    main()