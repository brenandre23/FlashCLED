-- =============================================================================
-- CEWP DATA SOURCE START DATES AUDIT
-- =============================================================================
-- Run in pgAdmin against your PostGIS database
-- Purpose: Find true MIN(date) for each source table to validate structural breaks.
-- Expected dates are aligned with configs/data.yaml: structural_breaks.
-- Schema: car_cewp
-- =============================================================================

-- Summary view: All tables with their date ranges
WITH table_dates AS (
    -- ACLED Events
    SELECT 
        'acled_events' as table_name,
        'ACLED' as source,
        MIN(event_date) as min_date,
        MAX(event_date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.acled_events
    
    UNION ALL
    
    -- ACLED Hybrid (NLP mechanisms)
    SELECT 
        'features_acled_hybrid' as table_name,
        'ACLED_Hybrid' as source,
        MIN(event_date) as min_date,
        MAX(event_date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.features_acled_hybrid
    
    UNION ALL
    
    -- Environmental Features (ERA5, CHIRPS, MODIS, VIIRS)
    SELECT 
        'environmental_features' as table_name,
        'GEE (ERA5/CHIRPS/MODIS/VIIRS)' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.environmental_features
    
    UNION ALL
    
    -- Landcover (Dynamic World)
    SELECT 
        'landcover_features' as table_name,
        'Dynamic World' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.landcover_features
    
    UNION ALL
    
    -- GDELT (features_dynamic_daily)
    SELECT 
        'features_dynamic_daily' as table_name,
        'GDELT' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.features_dynamic_daily
    
    UNION ALL
    
    -- IOM Displacement (raw)
    SELECT 
        'iom_dtm_raw' as table_name,
        'IOM DTM' as source,
        MIN(reporting_date) as min_date,
        MAX(reporting_date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT admin2_pcode) as unique_h3_cells  -- admin2, not h3
    FROM car_cewp.iom_dtm_raw
    
    UNION ALL
    
    -- IOM Displacement (H3 disaggregated)
    SELECT 
        'iom_displacement_h3' as table_name,
        'IOM DTM (H3)' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.iom_displacement_h3
    
    UNION ALL
    
    -- IODA (Internet Outages)
    SELECT 
        'internet_outages' as table_name,
        'IODA' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.internet_outages
    
    UNION ALL
    
    -- Economic Drivers
    SELECT 
        'economic_drivers' as table_name,
        'Economy (Yahoo Finance)' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        1 as unique_h3_cells  -- National level, no H3
    FROM car_cewp.economic_drivers
    
    UNION ALL
    
    -- Food Security
    SELECT 
        'food_security' as table_name,
        'FEWS NET' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT market) as unique_h3_cells  -- Markets, not H3
    FROM car_cewp.food_security
    
    UNION ALL
    
    -- CrisisWatch NLP
    SELECT 
        'features_crisiswatch' as table_name,
        'CrisisWatch NLP' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.features_crisiswatch
    
    UNION ALL
    
    -- Population
    SELECT 
        'population_h3' as table_name,
        'WorldPop' as source,
        MIN(MAKE_DATE(year, 1, 1)) as min_date,
        MAX(MAKE_DATE(year, 12, 31)) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.population_h3
    
    UNION ALL
    
    -- Temporal Features (final output table)
    SELECT 
        'temporal_features' as table_name,
        'SPINE (Output)' as source,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as row_count,
        COUNT(DISTINCT h3_index) as unique_h3_cells
    FROM car_cewp.temporal_features
)
SELECT 
    table_name,
    source,
    min_date::date as start_date,
    max_date::date as end_date,
    row_count,
    unique_h3_cells as spatial_coverage,
    -- Compare against expected structural breaks
    CASE 
        WHEN source = 'ACLED' THEN '2000-01-10'
        WHEN source = 'ACLED_Hybrid' THEN '2000-01-15'
        WHEN source LIKE '%VIIRS%' THEN '2012-01-28'
        WHEN source = 'Dynamic World' THEN '2017-04-01'
        WHEN source = 'GDELT' THEN '2015-02-18'
        WHEN source LIKE 'IOM%' THEN '2018-01-31'
        WHEN source = 'IODA' THEN '2022-01-01'
        WHEN source LIKE 'Economy%' THEN '2003-12-01'
        WHEN source = 'FEWS NET' THEN '2018-01-31'
        WHEN source = 'CrisisWatch NLP' THEN '2003-08-01'
        WHEN source = 'WorldPop' THEN '2000-01-01'
        ELSE NULL
    END as expected_start,
    CASE 
        WHEN min_date::date <= CASE 
            WHEN source = 'ACLED' THEN '2000-01-10'::date
            WHEN source = 'ACLED_Hybrid' THEN '2000-01-15'::date
            WHEN source LIKE '%VIIRS%' THEN '2012-01-28'::date
            WHEN source = 'Dynamic World' THEN '2017-04-01'::date
            WHEN source = 'GDELT' THEN '2015-02-18'::date
            WHEN source LIKE 'IOM%' THEN '2018-01-31'::date
            WHEN source = 'IODA' THEN '2022-01-01'::date
            WHEN source LIKE 'Economy%' THEN '2003-12-01'::date
            WHEN source = 'FEWS NET' THEN '2018-01-31'::date
            WHEN source = 'CrisisWatch NLP' THEN '2003-08-01'::date
            ELSE min_date::date
        END THEN 'OK'
        ELSE 'LATER_THAN_EXPECTED'
    END as status
FROM table_dates
ORDER BY min_date;


-- =============================================================================
-- DETAILED QUERIES FOR EACH SOURCE
-- =============================================================================

-- 1. VIIRS Nightlights - Check for pre-2012 data (should be none)
SELECT 
    'VIIRS Check' as check_name,
    MIN(date) as earliest_date,
    COUNT(*) FILTER (WHERE date < '2012-01-28') as records_before_viirs_launch,
    COUNT(*) FILTER (WHERE ntl_mean IS NOT NULL) as records_with_ntl_data
FROM car_cewp.environmental_features;


-- 2. Dynamic World - Check for pre-2017 data
SELECT 
    'Dynamic World Check' as check_name,
    MIN(date) as earliest_date,
    COUNT(*) FILTER (WHERE date < '2017-04-01') as records_before_dynamic_world_start,
    COUNT(*) as total_records
FROM car_cewp.landcover_features;


-- 3. IOM - Check raw vs disaggregated coverage
SELECT 
    'IOM Raw' as table_type,
    MIN(reporting_date) as min_date,
    MAX(reporting_date) as max_date,
    COUNT(DISTINCT admin2_name) as unique_admin2,
    COUNT(*) as total_records
FROM car_cewp.iom_dtm_raw
UNION ALL
SELECT 
    'IOM H3' as table_type,
    MIN(date) as min_date,
    MAX(date) as max_date,
    COUNT(DISTINCT h3_index) as unique_h3,
    COUNT(*) as total_records
FROM car_cewp.iom_displacement_h3;


-- 4. GDELT - Check v2 start date
SELECT 
    'GDELT Check' as check_name,
    MIN(date) as earliest_date,
    MAX(date) as latest_date,
    COUNT(*) FILTER (WHERE date < '2015-02-18') as records_before_v2,
    COUNT(DISTINCT variable) as unique_variables
FROM car_cewp.features_dynamic_daily;


-- 5. Economic Drivers - Check Yahoo Finance start
SELECT 
    'Economy Check' as check_name,
    MIN(date) as earliest_date,
    MAX(date) as latest_date,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE gold_price_usd IS NOT NULL) as gold_records,
    COUNT(*) FILTER (WHERE oil_price_usd IS NOT NULL) as oil_records
FROM car_cewp.economic_drivers;


-- 6. Food Security - Markets and commodities coverage
SELECT 
    market,
    MIN(date) as first_observation,
    MAX(date) as last_observation,
    COUNT(DISTINCT commodity) as commodities_tracked,
    COUNT(*) as total_records
FROM car_cewp.food_security
GROUP BY market
ORDER BY first_observation;


-- 7. IODA - Internet outages start
SELECT 
    'IODA Check' as check_name,
    MIN(date) as earliest_date,
    MAX(date) as latest_date,
    COUNT(*) as total_records,
    COUNT(DISTINCT h3_index) as h3_coverage
FROM car_cewp.internet_outages;


-- 8. CrisisWatch NLP - Topic coverage
SELECT 
    cw_topic_id,
    MIN(date) as first_observation,
    MAX(date) as last_observation,
    COUNT(*) as records,
    COUNT(DISTINCT h3_index) as h3_coverage
FROM car_cewp.features_crisiswatch
GROUP BY cw_topic_id
ORDER BY cw_topic_id;


-- =============================================================================
-- SPINE COVERAGE CHECK
-- =============================================================================
-- Check how much of the temporal_features spine has actual data vs NaN

SELECT 
    'Spine Coverage Summary' as report,
    COUNT(*) as total_spine_rows,
    COUNT(*) FILTER (WHERE fatalities_14d_sum IS NOT NULL) as rows_with_acled,
    COUNT(*) FILTER (WHERE ntl_mean IS NOT NULL) as rows_with_viirs,
    COUNT(*) FILTER (WHERE gdelt_event_count IS NOT NULL AND gdelt_event_count > 0) as rows_with_gdelt,
    COUNT(*) FILTER (WHERE iom_displacement_count_lag1 IS NOT NULL AND iom_displacement_count_lag1 > 0) as rows_with_iom,
    COUNT(*) FILTER (WHERE ioda_outage_score IS NOT NULL AND ioda_outage_score > 0) as rows_with_ioda,
    COUNT(*) FILTER (WHERE price_maize IS NOT NULL) as rows_with_food_prices,
    COUNT(*) FILTER (WHERE cw_score_local IS NOT NULL AND cw_score_local > 0) as rows_with_crisiswatch
FROM car_cewp.temporal_features;


-- =============================================================================
-- DATE ALIGNMENT CHECK
-- =============================================================================
-- Verify 14-day step alignment

SELECT 
    'Date Alignment Check' as check_name,
    MIN(date) as spine_start,
    MAX(date) as spine_end,
    COUNT(DISTINCT date) as unique_dates,
    (MAX(date) - MIN(date))::int / 14 + 1 as expected_steps,
    CASE 
        WHEN COUNT(DISTINCT date) = (MAX(date) - MIN(date))::int / 14 + 1 
        THEN 'ALIGNED'
        ELSE 'GAPS_DETECTED'
    END as alignment_status
FROM car_cewp.temporal_features;


-- =============================================================================
-- H3 SPATIAL COVERAGE
-- =============================================================================
-- Check how many H3 cells have data in each source

SELECT 
    'features_static' as table_name,
    COUNT(DISTINCT h3_index) as h3_cells
FROM car_cewp.features_static
UNION ALL
SELECT 
    'temporal_features' as table_name,
    COUNT(DISTINCT h3_index) as h3_cells
FROM car_cewp.temporal_features
UNION ALL
SELECT 
    'environmental_features' as table_name,
    COUNT(DISTINCT h3_index) as h3_cells
FROM car_cewp.environmental_features
UNION ALL
SELECT 
    'landcover_features' as table_name,
    COUNT(DISTINCT h3_index) as h3_cells
FROM car_cewp.landcover_features
UNION ALL
SELECT 
    'acled_events' as table_name,
    COUNT(DISTINCT h3_index) as h3_cells
FROM car_cewp.acled_events
UNION ALL
SELECT 
    'iom_displacement_h3' as table_name,
    COUNT(DISTINCT h3_index) as h3_cells
FROM car_cewp.iom_displacement_h3;



