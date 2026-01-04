"""
calculate_static_distances.py
============================
Purpose: Calculates static distance features from H3 grid to various sources.

OPTIMIZATION (Final):
- MINES: Thresholds adjusted based on DB validation (Median=542). 
  "Large" is now >= 500 to target top-tier sites.
- STRATEGY: Splits "Controlled" (Roadblock) from "Large" (Economic) to separate signals.
"""
import sys
from pathlib import Path
from sqlalchemy import text, inspect

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
        "dist_to_road FLOAT", "dist_to_river FLOAT",
        # Mine Features
        "dist_to_diamond_mine FLOAT", "dist_to_gold_mine FLOAT",
        "dist_to_large_mine FLOAT", "dist_to_controlled_mine FLOAT",
        "dist_to_large_gold_mine FLOAT"
    ]
    with engine.begin() as conn:
        for c in cols:
            try:
                conn.execute(text(f"ALTER TABLE {SCHEMA}.{STATIC_TABLE} ADD COLUMN IF NOT EXISTS {c}"))
            except Exception:
                pass


def calculate_capital_distance(engine):
    logger.info("  Calculating Distance to Capital (Bangui)...")
    # BANGUI COORDINATES: 4.3612° N, 18.5550° E
    sql = f"""
    UPDATE {SCHEMA}.{STATIC_TABLE}
    SET dist_to_capital = 
        ST_Distance(
            geometry::geography, 
            ST_SetSRID(ST_MakePoint(18.5550, 4.3612), 4326)::geography
        ) / 1000.0;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def calculate_border_distance(engine, configs):
    logger.info("  Calculating Distance to Border...")
    # Requires an 'admin0' boundary table or similar. 
    # If using the 'features_static' boundary, we calculate distance to the exterior ring.
    # Simplified approach: Distance to nearest non-CAR cell is complex.
    # Better approach: Use the Admin0 shapefile geometry if available.
    pass  # Placeholder if no specific border geom table exists.


def calculate_distance_sql(engine, target_col, source_table, filter_sql=None):
    """
    Generic SQL-based Nearest Neighbor distance calculation (PostGIS <-> operator).
    """
    logger.info(f"  Calculating {target_col} from {source_table}...")

    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns(source_table, schema=SCHEMA)}

    # Prefer geometry on source table; if absent but h3_index exists, join to features_static for geometry
    if "geometry" in cols:
        source_geom = "s.geometry"
        from_clause = f"{SCHEMA}.{source_table} s"
    elif "h3_index" in cols:
        logger.info(f"    Geometry column missing on {source_table}; joining via h3_index to {STATIC_TABLE}.")
        source_geom = "fs.geometry"
        from_clause = f"{SCHEMA}.{source_table} s JOIN {SCHEMA}.{STATIC_TABLE} fs ON s.h3_index = fs.h3_index"
    else:
        logger.error(f"    Cannot compute {target_col}: {source_table} lacks geometry and h3_index columns.")
        return

    where_clause = f"WHERE {filter_sql}" if filter_sql else ""
    
    sql = f"""
    UPDATE {SCHEMA}.{STATIC_TABLE} t
    SET {target_col} = (
        SELECT ST_Distance(t.geometry::geography, {source_geom}::geography) / 1000.0
        FROM {from_clause}
        {where_clause}
        ORDER BY t.geometry <-> {source_geom}
        LIMIT 1
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def main():
    try:
        logger.info("=" * 60)
        logger.info("STATIC DISTANCE CALCULATIONS (Strategic Optimized)")
        logger.info("=" * 60)
        
        configs = load_configs()
        engine = get_db_engine()
        
        add_distance_columns(engine)
        
        # 1. Geography
        calculate_capital_distance(engine)
        
        # 2. Infrastructure
        calculate_distance_sql(engine, "dist_to_city", "osm_cities_h3")
        calculate_distance_sql(engine, "dist_to_road", "grip4_roads_h3")
        
        # 3. Mines (OPTIMIZED STRATEGY)
        # A. Geology (What is in the ground?)
        calculate_distance_sql(engine, "dist_to_diamond_mine", "mines_h3", 
                               filter_sql="is_diamond = 1")
        calculate_distance_sql(engine, "dist_to_gold_mine", "mines_h3", 
                               filter_sql="is_gold = 1")

        # B. Strategic Value (Why fight over it?)
        # Large > 500 workers (Top 50% of sites)
        calculate_distance_sql(engine, "dist_to_large_mine", "mines_h3", 
                               filter_sql="worker_count >= 500")
        
        # Controlled (Has Roadblock) - Already militarized/taxed
        calculate_distance_sql(engine, "dist_to_controlled_mine", "mines_h3", 
                               filter_sql="has_roadblock = 1")
                             
        # High-Value Liquid Targets (Large Gold)
        calculate_distance_sql(engine, "dist_to_large_gold_mine", "mines_h3", 
                               filter_sql="is_gold = 1 AND worker_count >= 500")
        
        # 4. Water
        calculate_distance_sql(engine, "dist_to_river", "rivers", filter_sql="ord_stra >= 3")
        
        logger.info("✓ ALL DISTANCE CALCULATIONS COMPLETE")
        
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
