"""
fetch_mines.py
=======================
Purpose: Import CAR artisanal mining sites from IPIS WFS.
Output: car_cewp.mines_h3

FIX APPLIED:
- FIXED: Smart deduplication aggregation using proper groupby().agg() syntax.
- Eliminates SettingWithCopyWarning by using safe copies and direct column assignment.
- Ensures deterministic output for Vid, Company, Mineral, and Geometry.
"""

import sys
import requests
import geopandas as gpd
import pandas as pd
import h3.api.basic_int as h3
from io import BytesIO
from sqlalchemy import text
from pathlib import Path

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis

# --- Configuration ---
SCHEMA = "car_cewp"
TABLE_NAME = "mines_h3"

def get_mining_config(data_config):
    """Robust config retrieval."""
    if "ipis_mines" in data_config:
        return data_config["ipis_mines"]
    elif "mines_h3" in data_config:
        return data_config["mines_h3"]
    else:
        raise KeyError("Could not find 'ipis_mines' in data.yaml")

def fetch_wfs_data(data_config):
    """Fetch mining data from IPIS WFS server."""
    cfg = get_mining_config(data_config)
    wfs_url = cfg["wfs_url"]
    type_name = cfg["type_name"]

    # WFS 2.0 Request
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": type_name,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326"
    }

    logger.info(f"Fetching IPIS mining sites from {wfs_url}...")
    try:
        r = requests.get(wfs_url, params=params, timeout=120)
        r.raise_for_status()
        
        gdf = gpd.read_file(BytesIO(r.content))
        
    except Exception as e:
        logger.error(f"Failed to fetch/parse WFS data: {e}")
        raise

    # Ensure CRS is WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    logger.info(f"Retrieved {len(gdf)} mining sites.")
    return gdf

def smart_deduplication(gdf, pk_col):
    """
    SMART DEDUPLICATION LOGIC (REWRITTEN):
    - Uses correct groupby().agg() syntax with a dictionary.
    - Aggregates:
        - vid: first (deterministic ID)
        - company: mode (most frequent)
        - mineral: joined string of unique values
        - geometry: unary_union (combines overlapping points/shapes)
    - Returns a new GeoDataFrame with one row per unique 'h3_index'.
    """
    logger.info("Applying smart deduplication by H3 Index...")
    
    # 1. Normalize Columns first
    gdf.columns = [c.lower() for c in gdf.columns]
    
    # Identify key columns based on partial matches or defaults
    # We need: vid (id), company (operator), minerals (commodity), geometry
    
    # Vid / ID
    vid_col = next((c for c in ['id', 'gml_id', 'mine_id', 'pcode'] if c in gdf.columns), None)
    if not vid_col:
        # Fallback if no ID column exists
        gdf['vid'] = range(1, len(gdf) + 1)
        vid_col = 'vid'
        
    # Company / Operator
    company_col = next((c for c in gdf.columns if 'company' in c or 'operator' in c), None)
    if not company_col:
        gdf['company'] = 'Unknown'
        company_col = 'company'
        
    # Minerals
    mineral_col = next((c for c in gdf.columns if 'mineral' in c or 'commodity' in c), None)
    if not mineral_col:
        gdf['mineral'] = 'Unknown'
        mineral_col = 'mineral'

    # 2. Define Aggregation Logic
    def agg_mode(x):
        # Return most frequent value (mode), or first if empty/tie
        m = x.mode()
        return str(m.iloc[0]) if not m.empty else None

    def agg_set_join(x):
        # Join unique non-null string values
        vals = {str(v).strip() for v in x if pd.notnull(v) and str(v).strip() != ''}
        return ", ".join(sorted(vals)) if vals else None

    # 3. GroupBy H3 Index
    # We group by H3 to ensure one record per cell
    if 'h3_index' not in gdf.columns:
        raise ValueError("h3_index missing before deduplication!")

    # Groupby object
    grouped = gdf.groupby('h3_index')

    try:
        # Perform Aggregation
        # We process geometry separately because it requires GeoPandas logic (unary_union)
        # Standard columns first
        agg_schema = {
            vid_col: 'first',
            company_col: agg_mode,
            mineral_col: agg_set_join
        }
        
        df_agg = grouped.agg(agg_schema).reset_index()
        
        # Geometry aggregation (dissolve)
        # Using the built-in dissolve would act like groupby, effectively doing what we need for geom
        # But we already aggregated attributes manually to control logic. 
        # So we just get the geometry union per group.
        # Faster way: simply take the first geometry if they are points (mines usually are),
        # OR use unary_union if we want to be safe about overlaps.
        # Let's use unary_union for robustness.
        
        # Note: applying unary_union on a Series of geometries can be slow. 
        # If mines are points, we can just take the first or centroid of union.
        # 'first' is sufficient for point-in-polygon logic usually, but let's stick to requirements.
        
        # Optimization: If all are points, unioning them creates a MultiPoint.
        # This preserves all locations within the hex.
        geom_series = grouped['geometry'].apply(lambda x: x.unary_union)
        
        # Merge Geometry back
        df_agg = df_agg.merge(geom_series.rename('geometry'), on='h3_index')
        
        # Rename columns to standard schema
        df_agg = df_agg.rename(columns={
            vid_col: 'vid',
            company_col: 'company',
            mineral_col: 'mineral'
        })
        
        # Ensure final schema
        final_cols = ['h3_index', 'vid', 'company', 'mineral', 'geometry']
        gdf_final = gpd.GeoDataFrame(df_agg[final_cols], geometry='geometry', crs=gdf.crs)
        
        logger.info(f"Smart aggregation reduced {len(gdf)} rows to {len(gdf_final)} unique H3 cells.")
        return gdf_final

    except Exception as e:
        logger.error(f"Smart aggregation failed: {e}")
        logger.warning("Falling back to simple deduplication (keep first per H3).")
        
        # Fallback: Sort by ID and keep first
        gdf_simple = gdf.sort_values(vid_col).drop_duplicates(subset=['h3_index'], keep='first').copy()
        
        # Standardize columns manually for fallback
        rename_map = {vid_col: 'vid', company_col: 'company', mineral_col: 'mineral'}
        gdf_simple = gdf_simple.rename(columns=rename_map)
        
        # Ensure missing columns exist
        for c in ['vid', 'company', 'mineral']:
            if c not in gdf_simple.columns:
                gdf_simple[c] = None
                
        return gdf_simple[['h3_index', 'vid', 'company', 'mineral', 'geometry']]

def process_mines(gdf, resolution):
    """
    Clean data and calculate H3 indices.
    """
    # 1. Normalize Columns
    gdf.columns = [c.lower() for c in gdf.columns]
    
    # 2. Determine Primary Key (for logging/sorting)
    # We map this to 'vid' in smart_deduplication, but good to have a handle here
    pk_col = next((c for c in ['id', 'gml_id', 'mine_id', 'pcode'] if c in gdf.columns), None)
    if not pk_col:
        gdf['mine_id'] = range(1, len(gdf) + 1)
        pk_col = 'mine_id'
    
    # 3. Generate H3 Index (Before Dedup)
    logger.info(f"Calculating H3 indices at Resolution {resolution}...")
    gdf['h3_index'] = gdf.geometry.apply(
        lambda geom: h3.latlng_to_cell(geom.y, geom.x, resolution)
    )
    
    # 4. Smart Deduplication
    gdf_final = smart_deduplication(gdf, pk_col)
    
    return gdf_final, 'h3_index'

def import_to_postgres(gdf, pk_col, engine):
    """
    Uploads to PostGIS using REPLACE mode.
    """
    logger.info(f"Uploading {len(gdf)} mines to {SCHEMA}.{TABLE_NAME}...")

    # Write to PostGIS
    gdf.to_postgis(TABLE_NAME, engine, schema=SCHEMA, if_exists='replace', index=False)
    
    # Post-processing: Indexes
    with engine.begin() as conn:
        try:
            # Primary Key (H3 Index is unique after dedup)
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE_NAME} ADD PRIMARY KEY ({pk_col});"))
            
            # Spatial Index
            conn.execute(text(f"CREATE INDEX idx_{TABLE_NAME}_geom ON {SCHEMA}.{TABLE_NAME} USING GIST (geometry);"))
            
            # Ensure BigInt type for H3
            conn.execute(text(f"""
                ALTER TABLE {SCHEMA}.{TABLE_NAME} 
                ALTER COLUMN h3_index TYPE BIGINT 
                USING h3_index::bigint
            """))
            
            logger.info("Table optimized (PK + Spatial/H3 Indexes).")
        except Exception as e:
            logger.warning(f"Non-critical indexing error: {e}")

def main():
    logger.info("=" * 60)
    logger.info("IPIS MINING SITES IMPORT (SMART DEDUPLICATION - FIXED)")
    logger.info("=" * 60)
    
    engine = None
    try:
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        resolution = features_config["spatial"]["h3_resolution"]

        # 1. Fetch
        gdf = fetch_wfs_data(data_config)
        
        if gdf.empty:
            logger.warning("No mining data found.")
            return

        # 2. Process (includes H3 calc + smart dedup)
        gdf_processed, pk_col = process_mines(gdf, resolution)

        # 3. Upload
        import_to_postgres(gdf_processed, pk_col, engine)

        logger.info("✓ IPIS MINES IMPORT COMPLETE")

    except Exception as e:
        logger.error(f"IPIS mines import failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    main()