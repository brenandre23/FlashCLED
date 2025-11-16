"""
08_build_feature_matrix.py
===========================
Purpose: Build ML-ready feature matrix with temporal distance selection.
"""
import sys
import pandas as pd
from sqlalchemy import text
from pathlib import Path

# --- Import Centralized Utilities ---
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
import utils
from utils import logger, PATHS

SCHEMA = "car_cewp"

def build_feature_matrix_sql():
    # Note: Removed the trailing semi-colon to allow LIMIT to be appended safely
    sql = f"""
    WITH time_spine AS (
        SELECT 
            generate_series(
                '2010-01-01'::date,
                '2025-01-01'::date,
                '14 days'::interval
            )::date AS period_start
    ),
    
    spatial_temporal_grid AS (
        SELECT 
            f.h3_index,
            f.geometry,
            ts.period_start,
            (ts.period_start + INTERVAL '14 days')::date AS period_end,
            
            -- 1. Population (Coalesce nulls to 0)
            COALESCE(p.pop_count, 0) as population, 

            -- 2. Temporal Road Selection
            CASE 
                WHEN ts.period_start >= '2022-07-01' THEN f.dist_road_2025_km
                WHEN ts.period_start >= '2017-07-01' THEN f.dist_road_2020_km
                WHEN ts.period_start >= '2012-07-01' THEN f.dist_road_2015_km
                ELSE f.dist_road_2010_km
            END AS dist_road_km,
            
            -- 3. Static features
            f.dist_city_km,
            f.dist_capital_km,
            f.dist_border_km,
            f.elevation_mean,
            f.slope_mean,
            f.terrain_ruggedness_mean
            
        FROM {SCHEMA}.features_static f
        CROSS JOIN time_spine ts
        
        -- JOIN FIX: Cast p.h3_index (TEXT) to BIGINT to match f.h3_index
        LEFT JOIN {SCHEMA}.population_h3 p 
            ON f.h3_index = p.h3_index::bigint
            AND p.year = GREATEST(2015, EXTRACT(YEAR FROM ts.period_start))
    )
    
    SELECT * FROM spatial_temporal_grid
    ORDER BY h3_index, period_start
    """
    return sql

def demonstrate_temporal_selection(engine):
    logger.info("Running temporal selection demonstration...")
    
    sql_demo = f"""
    WITH sample_dates AS (
        SELECT date::date AS observation_date
        FROM unnest(ARRAY['2011-06-15', '2016-03-20', '2021-09-10', '2024-11-01']::date[]) AS date
    ),
    sample_cell AS ( SELECT h3_index FROM {SCHEMA}.features_static LIMIT 1 )
    
    SELECT 
        observation_date,
        CASE 
            WHEN observation_date >= '2022-07-01' THEN 'dist_road_2025_km'
            WHEN observation_date >= '2017-07-01' THEN 'dist_road_2020_km'
            WHEN observation_date >= '2012-07-01' THEN 'dist_road_2015_km'
            ELSE 'dist_road_2010_km'
        END AS selected_column_logic
    FROM sample_dates
    """
    try:
        df = pd.read_sql(sql_demo, engine)
        logger.info(f"\nLogic Verification:\n{df.to_string()}\n")
    except Exception as e:
        logger.warning(f"Could not run demonstration: {e}")

def main():
    try:
        utils.load_configs()
        engine = utils.get_db_engine()
        
        logger.info("="*60)
        logger.info("STEP 8: BUILD FEATURE MATRIX (SAMPLE)")
        logger.info("="*60)
        
        demonstrate_temporal_selection(engine)
        
        sql = build_feature_matrix_sql()
        
        logger.info("Fetching sample matrix (Limit 10,000 rows)...")
        # This appends correctly now because the function doesn't return a trailing ;
        sample_sql = f"{sql} LIMIT 10000;" 
        
        df = pd.read_sql(sample_sql, engine)
        
        output_path = PATHS["data_proc"] / "sample_feature_matrix.parquet"
        df.to_parquet(output_path, index=False)
        
        logger.info(f"✓ Sample saved to {output_path}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Feature matrix build failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()