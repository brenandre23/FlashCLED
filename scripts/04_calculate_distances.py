"""
06_save_to_postgres.py (Recommended: 04_calculate_distances.py)
================================================================
Purpose: Calculate distance features using server-side H3 k-ring expansion.
Output: Updates 'car_cewp.features_static' with distance columns (roads/cities).

Prerequisites: 
1. 'features_static' table (from 01_build_static_features.py)
2. 'osm_roads_h3_{year}' and 'osm_cities_h3' tables (from 03_fetch_temporal_roads_HYBRID.py)
3. PostgreSQL with h3-pg extension installed.
"""
import sys
from sqlalchemy import text
from pathlib import Path

# --- Import Centralized Utilities ---
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
import utils
from utils import logger

SCHEMA = "car_cewp"
STATIC_TABLE = "features_static"

# Approx. edge length + center-to-center distance for Res 5
# Used to estimate km distance from k-ring radius (k)
KM_PER_RING = 14.7 

def check_column_has_data(engine, column_name):
    """Check if a column exists and is populated (non-null)."""
    try:
        with engine.connect() as conn:
            # 1. Check if column exists
            check_col = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_schema = '{SCHEMA}'
                    AND table_name = '{STATIC_TABLE}'
                    AND column_name = '{column_name}'
                );
            """)
            exists = conn.execute(check_col).scalar()
            
            if not exists:
                return False, 0
            
            # 2. Check for actual data
            count = conn.execute(text(
                f"SELECT COUNT({column_name}) FROM {SCHEMA}.{STATIC_TABLE} WHERE {column_name} IS NOT NULL"
            )).scalar()
            
            return count > 0, count
    except Exception as e:
        logger.error(f"Column check failed: {e}")
        return False, 0

def add_distance_columns(engine, years):
    """Ensure the target columns exist in the database."""
    logger.info("Ensuring distance columns exist in schema...")
    
    with engine.begin() as conn:
        # Temporal road distances
        for year in years:
            col_name = f"dist_road_{year}_km"
            conn.execute(text(f"""
                ALTER TABLE {SCHEMA}.{STATIC_TABLE} 
                ADD COLUMN IF NOT EXISTS {col_name} FLOAT;
            """))
        
        # City distance
        conn.execute(text(f"""
            ALTER TABLE {SCHEMA}.{STATIC_TABLE} 
            ADD COLUMN IF NOT EXISTS dist_city_km FLOAT;
        """))
    
    logger.info(" Schema columns ready.")

def calculate_temporal_road_distances(engine, years):
    """
    Calculate distance to nearest road for each year using efficient H3 k-ring expansion.
    """
    logger.info("\n--- Calculating Temporal Road Distances ---")
    
    for year in years:
        col_name = f"dist_road_{year}_km"
        road_table = f"osm_roads_h3_{year}"
        
        # 1. CHECKPOINT
        has_data, count = check_column_has_data(engine, col_name)
        if has_data:
            logger.info(f" CHECKPOINT: {year} already calculated ({count:,} cells). Skipping.")
            continue
        
        logger.info(f"Processing {year} (Target: {col_name})...")
        
        # 2. Check source table
        with engine.connect() as conn:
            # Verify road table exists and has data
            try:
                road_count = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{road_table}")).scalar()
            except Exception:
                road_count = 0
            
            if road_count == 0:
                logger.warning(f"  ⚠ No roads found in {road_table}. Setting distances to NULL.")
                # Explicitly set to NULL to avoid stale data
                with engine.begin() as upd_conn:
                    upd_conn.execute(text(f"UPDATE {SCHEMA}.{STATIC_TABLE} SET {col_name} = NULL"))
                continue
        
        logger.info(f"  Source: {road_count:,} road cells available.")
        
        # 3. Calculation Query (Server-side H3)
        # Note: We cast to ::h3index to ensure compatibility with h3-pg functions
        sql_update = f"""
        UPDATE {SCHEMA}.{STATIC_TABLE} f
        SET {col_name} = sub.min_k * {KM_PER_RING}
        FROM (
            SELECT f.h3_index, 
                   (
                       SELECT k 
                       FROM generate_series(0, 50) as k
                       WHERE EXISTS (
                           SELECT 1 
                           FROM h3_grid_disk(f.h3_index::h3index, k) as ring_cell
                           JOIN {SCHEMA}.{road_table} r ON r.h3_index::h3index = ring_cell
                       )
                       ORDER BY k ASC 
                       LIMIT 1
                   ) as min_k
            FROM {SCHEMA}.{STATIC_TABLE} f
        ) as sub
        WHERE f.h3_index = sub.h3_index;
        """
        
        with engine.begin() as conn:
            conn.execute(text(sql_update))
        
        logger.info(f"   {year} calculations complete.")

def calculate_city_distances(engine):
    """Calculate distance to nearest city."""
    logger.info("\n--- Calculating City Distances ---")
    
    col_name = "dist_city_km"
    
    # 1. CHECKPOINT
    has_data, count = check_column_has_data(engine, col_name)
    if has_data:
        logger.info(f" CHECKPOINT: Cities already calculated ({count:,} cells). Skipping.")
        return
    
    # 2. Check source
    with engine.connect() as conn:
        city_count = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.osm_cities_h3")).scalar()
        
        if city_count == 0:
            logger.warning("  ⚠ No cities found in osm_cities_h3. Setting distances to NULL.")
            with engine.begin() as upd_conn:
                upd_conn.execute(text(f"UPDATE {SCHEMA}.{STATIC_TABLE} SET {col_name} = NULL"))
            return
    
    logger.info(f"  Source: {city_count:,} city cells available.")
    
    # 3. Calculation Query
    sql_update = f"""
    UPDATE {SCHEMA}.{STATIC_TABLE} f
    SET dist_city_km = sub.min_k * {KM_PER_RING}
    FROM (
        SELECT f.h3_index, 
               (
                   SELECT k 
                   FROM generate_series(0, 50) as k
                   WHERE EXISTS (
                       SELECT 1 
                       FROM h3_grid_disk(f.h3_index::h3index, k) as ring_cell
                       JOIN {SCHEMA}.osm_cities_h3 c ON c.h3_index::h3index = ring_cell
                   )
                   ORDER BY k ASC 
                   LIMIT 1
               ) as min_k
        FROM {SCHEMA}.{STATIC_TABLE} f
    ) as sub
    WHERE f.h3_index = sub.h3_index;
    """
    
    with engine.begin() as conn:
        conn.execute(text(sql_update))
    
    logger.info("   City calculations complete.")

def verify_distances(engine, years):
    """Print summary stats to verify data quality."""
    logger.info("\n--- Verification Statistics ---")
    
    with engine.connect() as conn:
        # 1. Temporal Roads
        for year in years:
            col_name = f"dist_road_{year}_km"
            try:
                stats = conn.execute(text(f"""
                    SELECT 
                        COUNT({col_name}) as non_null,
                        ROUND(AVG({col_name})::numeric, 2) as avg_km,
                        ROUND(MAX({col_name})::numeric, 2) as max_km
                    FROM {SCHEMA}.{STATIC_TABLE}
                """)).fetchone()
                logger.info(f"  Roads {year}: {stats.non_null:,} cells | Avg: {stats.avg_km} km | Max: {stats.max_km} km")
            except Exception:
                logger.warning(f"  Could not verify {col_name}")

        # 2. Cities
        try:
            stats = conn.execute(text(f"""
                SELECT 
                    COUNT(dist_city_km) as non_null,
                    ROUND(AVG(dist_city_km)::numeric, 2) as avg_km,
                    ROUND(MAX(dist_city_km)::numeric, 2) as max_km
                FROM {SCHEMA}.{STATIC_TABLE}
            """)).fetchone()
            logger.info(f"  Cities:    {stats.non_null:,} cells | Avg: {stats.avg_km} km | Max: {stats.max_km} km")
        except Exception:
            logger.warning("  Could not verify dist_city_km")

def main():
    try:
        # 1. Setup
        # We don't need data/features config here, just the DB connection
        engine = utils.get_db_engine()
        years = [2010, 2015, 2020, 2025]
        
        logger.info("="*60)
        logger.info("STEP 3b: DISTANCE CALCULATIONS (H3 K-RING)")
        logger.info("="*60)
        
        # 2. Add Columns
        add_distance_columns(engine, years)
        
        # 3. Calculate
        calculate_temporal_road_distances(engine, years)
        calculate_city_distances(engine)
        
        # 4. Verify
        verify_distances(engine, years)
        
        logger.info("\n" + "="*60)
        logger.info("DISTANCE PROCESSING COMPLETE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()