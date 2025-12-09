"""
fetch_settlements.py
=================================================
Purpose: Fetch HDX Settlements (Cities) and CLEANUP old OSM Road tables.
Output: 
  1. Upserts table: osm_cities_h3
  2. DROPS tables: osm_roads_h3_{year}
Details: Stores ONLY H3 indices (BigInt).
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

from utils import logger, get_db_engine, load_configs, download_file_with_retry

SCHEMA = "car_cewp"

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------

def check_table_exists(engine, table_name):
    insp = inspect(engine)
    return insp.has_table(table_name, schema=SCHEMA)

def upload_h3_table(engine, table_name, h3_list):
    """Uploads ONLY H3 indices. No Geometry bloat."""
    if not h3_list:
        logger.warning(f"No data to upload for {table_name}.")
        return

    logger.info(f"  Uploading {len(h3_list)} cells to {table_name}...")
    df = pd.DataFrame({'h3_index': list(h3_list)})
    
    # Replace ensures a fresh table
    df.to_sql(table_name, engine, schema=SCHEMA, if_exists="replace", index=False)
    
    # Optimize column type and add index
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE {SCHEMA}.{table_name}
            ALTER COLUMN h3_index TYPE BIGINT
            USING (h3_index::bigint);
        """))
        conn.execute(text(f"ALTER TABLE {SCHEMA}.{table_name} ADD PRIMARY KEY (h3_index);"))
        conn.execute(text(f"CREATE INDEX idx_{table_name}_h3 ON {SCHEMA}.{table_name} (h3_index);"))

# ---------------------------------------------------------
# 2. CORE LOGIC
# ---------------------------------------------------------

def nuke_osm_road_tables(engine):
    """
    Destructive function to remove OSM road tables from the database.
    """
    logger.info("\n--- Cleaning up Old OSM Road Tables ---")
    # Years previously used in the script
    years_to_purge = [2010, 2015, 2020, 2025]
    
    with engine.begin() as conn:
        for year in years_to_purge:
            table_name = f"osm_roads_h3_{year}"
            sql = text(f"DROP TABLE IF EXISTS {SCHEMA}.{table_name} CASCADE;")
            conn.execute(sql)
            logger.info(f"  ✓ Dropped table (if existed): {SCHEMA}.{table_name}")

def fetch_cities(data_config, resolution, engine):
    """
    Fetches settlement data from HDX (defined in config) and stores H3 indices.
    """
    TABLE_NAME = "osm_cities_h3"
    
    # Optional: If you want to force refresh cities, remove the check below
    if check_table_exists(engine, TABLE_NAME):
        logger.info(f"✓ Cities table ({TABLE_NAME}) exists. Skipping download.")
        return

    logger.info("\n--- Fetching HDX Settlements (Cities) ---")
    
    # Ensure URL exists in config
    try:
        url = data_config["osm"]["settlements_hdx_url"]
    except KeyError:
        logger.error("Config key ['osm']['settlements_hdx_url'] not found.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / "settlements.zip"
        
        # 1. Download
        try: 
            logger.info(f"  Downloading from: {url}")
            download_file_with_retry(url, zip_path)
        except Exception as e: 
            logger.error(f"Failed to download settlements: {e}")
            return

        # 2. Extract
        try:
            with zipfile.ZipFile(zip_path, 'r') as z: 
                z.extractall(temp_path)
        except Exception as e: 
            logger.error(f"Failed to extract zip: {e}")
            return

        # 3. Find Shapefile
        shp_files = list(temp_path.rglob("*.shp"))
        if not shp_files: 
            logger.error("No .shp file found in downloaded zip.")
            return
        
        # 4. Process Geometry to H3
        try:
            cities_gdf = gpd.read_file(shp_files[0])
            if cities_gdf.crs and cities_gdf.crs.to_string() != "EPSG:4326":
                cities_gdf = cities_gdf.to_crs("EPSG:4326")
                
            city_cells = set()
            # Simple centroid logic is usually sufficient for point settlements
            # If polygons, this gets the cell of the center
            for geom in cities_gdf.geometry:
                if geom is None: continue
                try:
                    # Point (y=lat, x=lon)
                    cell = h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, resolution)
                    if cell > 0: 
                        city_cells.add(cell)
                except Exception: 
                    continue
            
            if city_cells: 
                upload_h3_table(engine, TABLE_NAME, city_cells)
            else:
                logger.warning("No valid city cells found to upload.")
                
        except Exception as e: 
            logger.error(f"Error processing cities geometry: {e}")

# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    engine = None
    try:
        data_config, features_config, _ = load_configs()
        engine = get_db_engine()
        resolution = features_config["spatial"]["h3_resolution"]
        
        logger.info("="*60)
        logger.info("SETTLEMENTS FETCH & ROADS CLEANUP")
        logger.info("="*60)

        # 1. Nuke the old roads
        nuke_osm_road_tables(engine)

        # 2. Process Cities
        fetch_cities(data_config, resolution, engine)
            
        logger.info("="*60)
        logger.info("✓ SETTLEMENTS COMPLETE & ROADS REMOVED")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"✗ Script failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    main()