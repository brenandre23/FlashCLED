"""
fetch_mines.py
=======================
Purpose: Import CAR artisanal mining sites from IPIS WFS.
Output: car_cewp.mines_h3

AUDIT FIX:
- Replaced to_postgis(replace) with upload_to_postgis(upsert).
- Added ensure_table_exists.
- Added retry logic with exponential backoff for network requests.
- FIXED: Preserve is_diamond, is_gold, has_roadblock, worker_count columns.
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

from utils import logger, get_db_engine, load_configs, upload_to_postgis, retry_request

SCHEMA = "car_cewp"
TABLE_NAME = "mines_h3"


def get_mining_config(data_config):
    if "ipis_mines" in data_config:
        return data_config["ipis_mines"]
    elif "mines_h3" in data_config:
        return data_config["mines_h3"]
    else:
        raise KeyError("Could not find 'ipis_mines' in data.yaml")


@retry_request
def _fetch_wfs_request(wfs_url: str, params: dict) -> bytes:
    """
    Execute WFS request with retry logic.
    Retries on Timeout, ConnectionError, and 5xx HTTPError.
    """
    logger.info(f"  Attempting WFS request to {wfs_url}...")
    r = requests.get(wfs_url, params=params, timeout=(10, 120))
    r.raise_for_status()
    return r.content


def fetch_wfs_data(data_config):
    cfg = get_mining_config(data_config)
    wfs_url = cfg["wfs_url"]
    type_name = cfg["type_name"]
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
        content = _fetch_wfs_request(wfs_url, params)
    except requests.exceptions.Timeout as e:
        logger.error(f"WFS request timed out after retries: {e}")
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"WFS connection failed after retries: {e}")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"WFS HTTP error after retries: {e}")
        raise
    
    gdf = gpd.read_file(BytesIO(content))
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    logger.info(f"Retrieved {len(gdf)} mining sites.")
    return gdf


def clean_mining_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean and standardize mining attribute columns.
    FIX: Uses numeric conversion to handle '1.0' (float) vs '1' (str).
    """
    logger.info("Cleaning mining attribute columns...")
    gdf = gdf.copy()
    gdf.columns = [c.lower() for c in gdf.columns]
    
    # Helper for boolean conversion
    def to_bool_int(series):
        # Coerce to numbers (handles '1', '1.0', 1, 1.0)
        # Anything that isn't a number becomes NaN
        numeric = pd.to_numeric(series, errors='coerce').fillna(0)
        # Return 1 if > 0, else 0
        return (numeric > 0).astype(int)

    # --- is_diamond ---
    if 'minerals_diamant' in gdf.columns:
        gdf['is_diamond'] = to_bool_int(gdf['minerals_diamant'])
    else:
        gdf['is_diamond'] = 0
    
    # --- is_gold ---
    if 'minerals_or' in gdf.columns:
        gdf['is_gold'] = to_bool_int(gdf['minerals_or'])
    else:
        gdf['is_gold'] = 0
    
    # --- worker_count ---
    if 'workers_numb' in gdf.columns:
        gdf['worker_count'] = pd.to_numeric(gdf['workers_numb'], errors='coerce').fillna(0).astype(int)
    else:
        gdf['worker_count'] = 0
    
    # --- has_roadblock ---
    if 'roadblocks' in gdf.columns:
        # This fixes the specific bug by treating 1.0 and 1 identically
        gdf['has_roadblock'] = to_bool_int(gdf['roadblocks'])
    else:
        logger.warning("Column 'roadblocks' not found. Setting has_roadblock to 0.")
        gdf['has_roadblock'] = 0
    
    logger.info(f"  is_diamond: {gdf['is_diamond'].sum()} sites")
    logger.info(f"  is_gold: {gdf['is_gold'].sum()} sites")
    logger.info(f"  has_roadblock: {gdf['has_roadblock'].sum()} sites")
    logger.info(f"  worker_count total: {gdf['worker_count'].sum()}")
    
    return gdf


def smart_aggregation(gdf: gpd.GeoDataFrame, pk_col: str) -> gpd.GeoDataFrame:
    """
    Aggregate mines by H3 index with smart aggregation rules.
    
    Aggregation dictionary:
      - is_diamond: max (if any mine in hex is diamond -> 1)
      - is_gold: max (if any mine in hex is gold -> 1)
      - has_roadblock: max (if any mine has roadblock -> 1)
      - worker_count: sum (total workers in hex)
      - geometry: first (keep representative centroid)
    """
    logger.info("Applying smart aggregation by H3 Index...")
    
    # Build aggregation dict
    agg_dict = {
        'is_diamond': 'max',
        'is_gold': 'max',
        'has_roadblock': 'max',
        'worker_count': 'sum',
        'geometry': 'first'
    }
    
    # Group and aggregate
    grouped = gdf.groupby(pk_col, as_index=False).agg(agg_dict)
    
    # Convert back to GeoDataFrame
    gdf_agg = gpd.GeoDataFrame(grouped, geometry='geometry', crs=gdf.crs)
    
    logger.info(f"Smart aggregation reduced {len(gdf)} rows to {len(gdf_agg)} unique H3 cells.")
    logger.info(f"  Aggregated is_diamond: {gdf_agg['is_diamond'].sum()} hexes")
    logger.info(f"  Aggregated is_gold: {gdf_agg['is_gold'].sum()} hexes")
    logger.info(f"  Aggregated has_roadblock: {gdf_agg['has_roadblock'].sum()} hexes")
    logger.info(f"  Aggregated worker_count: {gdf_agg['worker_count'].sum()} total")
    
    return gdf_agg


def process_mines(gdf: gpd.GeoDataFrame, resolution: int):
    """
    Process mining data:
      1. Clean columns (is_diamond, is_gold, worker_count, has_roadblock)
      2. Calculate H3 indices
      3. Aggregate by H3
    """
    pk_col = 'h3_index'
    
    # Clean columns first
    gdf = clean_mining_columns(gdf)
    
    # Calculate H3 indices
    logger.info(f"Calculating H3 indices at Resolution {resolution}...")
    gdf['h3_index'] = gdf.geometry.apply(
        lambda geom: h3.latlng_to_cell(geom.y, geom.x, resolution)
    )
    
    # Smart aggregation
    gdf_final = smart_aggregation(gdf, pk_col)
    
    return gdf_final, pk_col


def ensure_table_exists(engine):
    """Create the mines_h3 table with proper schema including new columns."""
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                h3_index BIGINT PRIMARY KEY,
                is_diamond INTEGER DEFAULT 0,
                is_gold INTEGER DEFAULT 0,
                has_roadblock INTEGER DEFAULT 0,
                worker_count INTEGER DEFAULT 0,
                geometry GEOMETRY(Geometry, 4326)
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_geom ON {SCHEMA}.{TABLE_NAME} USING GIST (geometry);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_diamond ON {SCHEMA}.{TABLE_NAME} (is_diamond);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_gold ON {SCHEMA}.{TABLE_NAME} (is_gold);"))


def import_to_postgres(gdf: gpd.GeoDataFrame, pk_col: str, engine):
    """Upload processed mines to PostGIS with upsert."""
    logger.info(f"Uploading {len(gdf)} mines to {SCHEMA}.{TABLE_NAME} (Upsert mode)...")
    
    ensure_table_exists(engine)
    
    df = pd.DataFrame(gdf)
    df['geometry'] = df['geometry'].apply(lambda x: x.wkt)
    df['h3_index'] = df['h3_index'].astype('int64')
    
    # Ensure integer types
    df['is_diamond'] = df['is_diamond'].astype(int)
    df['is_gold'] = df['is_gold'].astype(int)
    df['has_roadblock'] = df['has_roadblock'].astype(int)
    df['worker_count'] = df['worker_count'].astype(int)

    cols = ['h3_index', 'is_diamond', 'is_gold', 'has_roadblock', 'worker_count', 'geometry']
    upload_to_postgis(engine, df[cols], TABLE_NAME, SCHEMA, primary_keys=[pk_col])
    
    logger.info("IPIS Mines import complete.")


def main():
    logger.info("=" * 60)
    logger.info("IPIS MINING SITES IMPORT")
    logger.info("=" * 60)
    
    engine = None
    try:
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        resolution = features_config["spatial"]["h3_resolution"]

        gdf = fetch_wfs_data(data_config)
        if gdf.empty:
            logger.warning("No mining sites retrieved. Exiting.")
            return

        gdf_processed, pk_col = process_mines(gdf, resolution)
        import_to_postgres(gdf_processed, pk_col, engine)

    except Exception as e:
        logger.error(f"IPIS mines import failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()


if __name__ == "__main__":
    main()
