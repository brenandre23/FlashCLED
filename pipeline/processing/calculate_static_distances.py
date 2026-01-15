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
        "dist_to_road FLOAT", "dist_to_river FLOAT", "dist_to_market_km FLOAT",
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
    try:
        boundary = get_boundary(configs['data'], configs['features'])
    except Exception as e:
        logger.warning(f"    Failed to load boundary for border distance: {e}")
        return

    if boundary is None or boundary.empty or 'geometry' not in boundary:
        logger.warning("    Boundary geometry missing; skipping dist_to_border.")
        return

    # Use union of all admin0 polygons
    boundary = boundary.to_crs(epsg=4326)
    geom = boundary.unary_union
    if geom.is_empty:
        logger.warning("    Boundary geometry empty; skipping dist_to_border.")
        return

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}._tmp_border"))
        conn.execute(text(f"CREATE TABLE {SCHEMA}._tmp_border (geom geometry(Geometry, 4326))"))
        conn.execute(
            text(f"INSERT INTO {SCHEMA}._tmp_border (geom) VALUES (ST_GeomFromText(:wkt, 4326))"),
            {"wkt": geom.wkt}
        )
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE} t
            SET dist_to_border = (
                SELECT ST_Distance(
                    t.geometry::geography, 
                    ST_Boundary(b.geom)::geography
                ) / 1000.0
                FROM {SCHEMA}._tmp_border b
            )
            WHERE dist_to_border IS NULL;
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}._tmp_border"))


def calculate_distance_sql(engine, target_col, source_table, filter_sql=None):
    """
    Generic SQL-based Nearest Neighbor distance calculation (PostGIS <-> operator).
    """
    logger.info(f"  Calculating {target_col} from {source_table}...")

    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns(source_table, schema=SCHEMA)}

    source_geom = None
    if "geometry" in cols:
        source_geom = "s.geometry"
    elif "geom" in cols:
        source_geom = "s.geom"

    # Prefer geometry on source table; if absent but h3_index exists, join to features_static for geometry
    if source_geom:
        from_clause = f"{SCHEMA}.{source_table} s"
    elif "h3_index" in cols:
        logger.info(f"    Geometry column missing on {source_table}; joining via h3_index to {STATIC_TABLE}.")
        source_geom = "fs.geometry"
        from_clause = f"{SCHEMA}.{source_table} s JOIN {SCHEMA}.{STATIC_TABLE} fs ON s.h3_index = fs.h3_index"
    else:
        logger.error(f"    Cannot compute {target_col}: {source_table} lacks geometry/geom and h3_index columns.")
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


def calculate_market_distance(engine):
    """
    Compute distance from each H3 cell to nearest market (market_locations table).
    Assumes market_locations has latitude/longitude columns.
    """
    logger.info("  Calculating Distance to Nearest Market...")
    tmp_table = f"{SCHEMA}._tmp_markets"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))
        conn.execute(text(f"CREATE TABLE {tmp_table} (geom geometry(Point, 4326))"))
        conn.execute(text(f"""
            INSERT INTO {tmp_table} (geom)
            SELECT ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            FROM {SCHEMA}.market_locations
            WHERE longitude IS NOT NULL AND latitude IS NOT NULL
        """))
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE} t
            SET dist_to_market_km = (
                SELECT ST_Distance(t.geometry::geography, m.geom::geography) / 1000.0
                FROM {tmp_table} m
                ORDER BY t.geometry <-> m.geom
                LIMIT 1
            );
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))


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
        calculate_border_distance(engine, configs)
        
        # 2. Infrastructure
        calculate_distance_sql(engine, "dist_to_city", "osm_cities_h3")
        calculate_distance_sql(engine, "dist_to_road", "grip4_roads_h3")
        calculate_market_distance(engine)
        
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
