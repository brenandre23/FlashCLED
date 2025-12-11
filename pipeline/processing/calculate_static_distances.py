"""
calculate_static_distances.py
============================
Purpose: Calculates static distance features from H3 grid to various sources.

AUDIT FIXES:
1. MEMORY SAFETY: Replaced Pandas/cKDTree with PostGIS SQL (prevents OOM).
2. PERFORMANCE: Uses <-> (KNN) operator for efficient nearest-neighbor search.
3. ROBUSTNESS: Handles cases where source tables lack geometry by joining to features_static.
"""
import sys
from pathlib import Path
from sqlalchemy import text

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from utils import logger, get_db_engine, load_configs, get_boundary

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
    logger.info("Calculating Capital Distance (SQL)...")
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE}
            SET dist_to_capital = ST_Distance(
                ST_Centroid(geometry)::geography, ST_Point(18.5550, 4.3612)::geography
            ) / 1000.0
            WHERE dist_to_capital IS NULL
        """))

def calculate_border_distance(engine, configs):
    logger.info("Calculating Border Distance (SQL)...")
    boundary_gdf = get_boundary(configs['data'], configs['features'])
    try: boundary_line = boundary_gdf.union_all().boundary
    except AttributeError: boundary_line = boundary_gdf.unary_union.boundary
    
    # Upload temp border
    import geopandas as gpd
    border_gdf = gpd.GeoDataFrame({'geometry': [boundary_line]}, crs=boundary_gdf.crs)
    border_gdf.to_postgis("temp_border", engine, schema=SCHEMA, if_exists='replace', index=False)
    
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE} f
            SET dist_to_border = ST_Distance(ST_Centroid(f.geometry)::geography, b.geometry::geography) / 1000.0
            FROM {SCHEMA}.temp_border b WHERE f.dist_to_border IS NULL
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.temp_border"))

# ---------------------------------------------------------
# SQL-BASED SMART DISTANCE CALCULATION
# ---------------------------------------------------------
def calculate_distance_sql(engine, target_col, source_table, filter_sql=None):
    """
    Calculates distance to nearest feature in source_table using PostGIS.
    Handles tables with 'geometry' column or joins to features_static if only h3_index exists.
    """
    logger.info(f"Calculating {target_col} using PostGIS Nearest Neighbor...")
    
    with engine.connect() as conn:
        # Check if already calculated
        if conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE} WHERE {target_col} IS NOT NULL")).scalar() > 0:
            logger.info(f"  {target_col} already populated. Skipping.")
            return

        # Check source exists
        exists = conn.execute(text(f"SELECT to_regclass('{SCHEMA}.{source_table}')")).scalar()
        if not exists:
            logger.warning(f"  Source table {source_table} not found. Skipping.")
            return

    # Determine if source table has geometry
    has_geom = False
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SELECT geometry FROM {SCHEMA}.{source_table} LIMIT 1"))
            has_geom = True
    except Exception:
        has_geom = False

    # Construct the subquery for nearest neighbor
    if has_geom:
        # Source has geometry (e.g., mines_h3, rivers)
        # Use <-> operator for KNN index scan (Fast!)
        where_clause = f"WHERE {filter_sql}" if filter_sql else ""
        subquery = f"""
            SELECT ST_Distance(s.geometry::geography, t.geometry::geography) / 1000.0
            FROM {SCHEMA}.{source_table} t
            {where_clause}
            ORDER BY s.geometry <-> t.geometry
            LIMIT 1
        """
    else:
        # Source has NO geometry (e.g., osm_cities_h3, grip4_roads_h3)
        # We must join to features_static to get geometry of the source cells
        logger.info(f"  {source_table} lacks geometry. Joining to features_static...")
        where_clause = f"AND {filter_sql}" if filter_sql else ""
        subquery = f"""
            SELECT ST_Distance(s.geometry::geography, t_geom.geometry::geography) / 1000.0
            FROM {SCHEMA}.{source_table} t
            JOIN {SCHEMA}.{STATIC_TABLE} t_geom ON t.h3_index = t_geom.h3_index
            WHERE 1=1 {where_clause}
            ORDER BY s.geometry <-> t_geom.geometry
            LIMIT 1
        """

    # Execute Update
    # Note: Running this on the whole table at once can be slow but memory safe.
    # Postgres is good at optimizing KNN.
    sql = f"""
        UPDATE {SCHEMA}.{STATIC_TABLE} s
        SET {target_col} = ({subquery})
        WHERE {target_col} IS NULL;
    """
    
    with engine.begin() as conn:
        conn.execute(text(sql))
    
    logger.info(f"  ✓ {target_col} updated.")

def main():
    engine = None
    try:
        logger.info("=" * 60)
        logger.info("STATIC DISTANCE CALCULATIONS (PostGIS Optimized)")
        logger.info("=" * 60)
        
        configs = load_configs()
        engine = get_db_engine()
        
        add_distance_columns(engine)
        
        # 1. Geography
        calculate_capital_distance(engine)
        calculate_border_distance(engine, configs)
        
        # 2. Infrastructure (No Geom in source, uses Join logic)
        calculate_distance_sql(engine, "dist_to_city", "osm_cities_h3")
        calculate_distance_sql(engine, "dist_to_road", "grip4_roads_h3")
        
        # 3. Mines (Has Geom, uses Filter logic)
        calculate_distance_sql(engine, "dist_to_diamond_mine", "mines_h3", 
                             filter_sql="(minerals_diamant::text = '1' OR minerals_diamant::text = 'true')")
        calculate_distance_sql(engine, "dist_to_gold_mine", "mines_h3", 
                             filter_sql="(minerals_or::text = '1' OR minerals_or::text = 'true')")
        
        # 4. Water (Already robust in previous version, keeping logic or using SQL helper if standard)
        # Re-using the logic from previous audit since fetch_rivers uploads geometry
        calculate_distance_sql(engine, "dist_to_river", "rivers", filter_sql="ord_stra >= 3")
        
        logger.info("✓ ALL DISTANCE CALCULATIONS COMPLETE")
        
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine: engine.dispose()

if __name__ == "__main__":
    main()