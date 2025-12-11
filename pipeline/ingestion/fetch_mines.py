"""
fetch_mines.py
=======================
Purpose: Import CAR artisanal mining sites from IPIS WFS.
Output: car_cewp.mines_h3

AUDIT FIX:
- Replaced to_postgis(replace) with upload_to_postgis(upsert).
- Added ensure_table_exists.
- Added retry logic with exponential backoff for network requests.
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
    if "ipis_mines" in data_config: return data_config["ipis_mines"]
    elif "mines_h3" in data_config: return data_config["mines_h3"]
    else: raise KeyError("Could not find 'ipis_mines' in data.yaml")


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


def smart_deduplication(gdf, pk_col):
    logger.info("Applying smart deduplication by H3 Index...")
    gdf.columns = [c.lower() for c in gdf.columns]
    
    vid_col = next((c for c in ['id', 'gml_id', 'mine_id', 'pcode'] if c in gdf.columns), None)
    if not vid_col:
        gdf['vid'] = range(1, len(gdf) + 1)
        vid_col = 'vid'
        
    company_col = next((c for c in gdf.columns if 'company' in c or 'operator' in c), None) or 'company'
    mineral_col = next((c for c in gdf.columns if 'mineral' in c or 'commodity' in c), None) or 'mineral'
    
    if company_col not in gdf.columns: gdf[company_col] = 'Unknown'
    if mineral_col not in gdf.columns: gdf[mineral_col] = 'Unknown'

    def agg_mode(x):
        m = x.mode()
        return str(m.iloc[0]) if not m.empty else None

    def agg_set_join(x):
        vals = {str(v).strip() for v in x if pd.notnull(v) and str(v).strip() != ''}
        return ", ".join(sorted(vals)) if vals else None

    grouped = gdf.groupby('h3_index')
    
    df_agg = grouped.agg({
        vid_col: 'first',
        company_col: agg_mode,
        mineral_col: agg_set_join
    }).reset_index()

    geom_series = grouped['geometry'].apply(lambda x: x.unary_union)
    df_agg = df_agg.merge(geom_series.rename('geometry'), on='h3_index')
    
    df_agg = df_agg.rename(columns={vid_col: 'vid', company_col: 'company', mineral_col: 'mineral'})
    gdf_final = gpd.GeoDataFrame(df_agg, geometry='geometry', crs=gdf.crs)
    
    logger.info(f"Smart aggregation reduced {len(gdf)} rows to {len(gdf_final)} unique H3 cells.")
    return gdf_final


def process_mines(gdf, resolution):
    gdf.columns = [c.lower() for c in gdf.columns]
    pk_col = 'h3_index'
    
    logger.info(f"Calculating H3 indices at Resolution {resolution}...")
    gdf['h3_index'] = gdf.geometry.apply(
        lambda geom: h3.latlng_to_cell(geom.y, geom.x, resolution)
    )
    
    gdf_final = smart_deduplication(gdf, pk_col)
    return gdf_final, pk_col


def ensure_table_exists(engine):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                h3_index BIGINT PRIMARY KEY,
                vid TEXT,
                company TEXT,
                mineral TEXT,
                geometry GEOMETRY(Geometry, 4326)
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_geom ON {SCHEMA}.{TABLE_NAME} USING GIST (geometry);"))


def import_to_postgres(gdf, pk_col, engine):
    logger.info(f"Uploading {len(gdf)} mines to {SCHEMA}.{TABLE_NAME} (Upsert mode)...")
    
    ensure_table_exists(engine)
    
    df = pd.DataFrame(gdf)
    df['geometry'] = df['geometry'].apply(lambda x: x.wkt)
    df['h3_index'] = df['h3_index'].astype('int64')

    cols = ['h3_index', 'vid', 'company', 'mineral', 'geometry']
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
        if gdf.empty: return

        gdf_processed, pk_col = process_mines(gdf, resolution)
        import_to_postgres(gdf_processed, pk_col, engine)

    except Exception as e:
        logger.error(f"IPIS mines import failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine: engine.dispose()


if __name__ == "__main__":
    main()
