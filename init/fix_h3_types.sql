-- init/fix_h3_types.sql
-- Standardize H3 columns to BIGINT for consistent joins
-- Optimization: Re-creates indexes immediately to prevent slow sequential scans.

BEGIN;

-- 1. GRIP4 Roads
ALTER TABLE car_cewp.grip4_roads_h3
    ALTER COLUMN h3_index TYPE BIGINT 
    USING CAST(h3_index AS BIGINT);

-- Re-index immediately for join performance
CREATE INDEX IF NOT EXISTS idx_grip4_h3_bigint ON car_cewp.grip4_roads_h3 (h3_index);

-- 2. Rivers
ALTER TABLE car_cewp.rivers
    ALTER COLUMN h3_index TYPE BIGINT 
    USING CAST(h3_index AS BIGINT);

CREATE INDEX IF NOT EXISTS idx_rivers_h3_bigint ON car_cewp.rivers (h3_index);

-- 3. Mines (Robust check)
-- Checks if the table/column exists before attempting conversion
DO $$ 
BEGIN 
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='mines_h3' AND column_name='h3_index') THEN
        ALTER TABLE car_cewp.mines_h3
        ALTER COLUMN h3_index TYPE BIGINT 
        USING CAST(h3_index AS BIGINT);
        
        CREATE INDEX IF NOT EXISTS idx_mines_h3_bigint ON car_cewp.mines_h3 (h3_index);
    END IF;
END $$;

COMMIT;