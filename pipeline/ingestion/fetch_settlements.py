"""
fetch_settlements.py
=================================================
Purpose: Fetch HDX Settlements (Cities) and CLEANUP old OSM Road tables.
Output: Upserts table: osm_cities_h3

AUDIT FIX:
- Replaced to_sql(replace) with upload_to_postgis(upsert).
- Added ensure_table_exists.
"""
import sys
import zipfile
import tempfile
import geopandas as gpd
import pandas as pd
import h3.api.basic_int as h3
from sqlalchemy import text, inspect
from pathlib import Path

# Setup Root Directory
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, download_file_with_retry, upload_to_postgis

SCHEMA = "car_cewp"

def check_table_exists(engine, table_name):
    insp = inspect(engine)
    return insp.has_table(table_name, schema=SCHEMA)

def ensure_cities_table_exists(engine, table_name):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{table_name} (
                h3_index BIGINT PRIMARY KEY
            );
        """))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_h3 ON {SCHEMA}.{table_name} (h3_index);"))

def upload_h3_table(engine, table_name, h3_list):
    """Uploads ONLY H3 indices using Upsert."""
    if not h3_list: return

    logger.info(f"  Uploading {len(h3_list)} cells to {table_name} (Upsert)...")
    df = pd.DataFrame({'h3_index': list(h3_list)})
    df['h3_index'] = df['h3_index'].astype('int64')
    
    ensure_cities_table_exists(engine, table_name)
    upload_to_postgis(engine, df, table_name, SCHEMA, primary_keys=['h3_index'])

def nuke_osm_road_tables(engine):
    """Clean up old OSM road tables."""
    logger.info("\n--- Cleaning up Old OSM Road Tables ---")
    years_to_purge = [2010, 2015, 2020, 2025]
    with engine.begin() as conn:
        for year in years_to_purge:
            table_name = f"osm_roads_h3_{year}"
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{table_name} CASCADE;"))

def fetch_cities(data_config, resolution, engine):
    TABLE_NAME = "osm_cities_h3"
    
    if check_table_exists(engine, TABLE_NAME):
        logger.info(f"✓ Cities table ({TABLE_NAME}) exists. Skipping download.")
        return

    logger.info("\n--- Fetching HDX Settlements (Cities) ---")
    try:
        url = data_config["osm"]["settlements_hdx_url"]
    except KeyError:
        logger.error("Config key ['osm']['settlements_hdx_url'] not found.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / "settlements.zip"
        
        try: 
            logger.info(f"  Downloading from: {url}")
            download_file_with_retry(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(temp_path)
        except Exception as e: 
            logger.error(f"Download/Extraction failed: {e}")
            return

        shp_files = list(temp_path.rglob("*.shp"))
        if not shp_files: return
        
        try:
            cities_gdf = gpd.read_file(shp_files[0])
            if cities_gdf.crs and cities_gdf.crs.to_string() != "EPSG:4326":
                cities_gdf = cities_gdf.to_crs("EPSG:4326")
                
            city_cells = set()
            for geom in cities_gdf.geometry:
                if geom is None: continue
                try:
                    cell = h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, resolution)
                    if cell > 0: city_cells.add(cell)
                except Exception: continue
            
            if city_cells: 
                upload_h3_table(engine, TABLE_NAME, city_cells)
            else:
                logger.warning("No valid city cells found.")
        except Exception as e: 
            logger.error(f"Error processing geometry: {e}")

def main():
    engine = None
    try:
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        resolution = features_config["spatial"]["h3_resolution"]
        
        logger.info("="*60)
        logger.info("SETTLEMENTS FETCH & ROADS CLEANUP")
        logger.info("="*60)

        nuke_osm_road_tables(engine)
        fetch_cities(data_config, resolution, engine)
            
        logger.info("="*60)
        logger.info("✓ SETTLEMENTS COMPLETE & ROADS REMOVED")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine: engine.dispose()

if __name__ == "__main__":
    main()