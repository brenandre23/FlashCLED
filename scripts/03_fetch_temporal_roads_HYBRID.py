"""
03_fetch_temporal_roads_HYBRID.py
=================================================
Purpose: Fetch historical road networks (Ohsome API) and Settlement data (HDX).
Output: Temporal tables in PostgreSQL (osm_roads_h3_{year}, osm_cities_h3).

Optimization:
- Uses Multiprocessing to convert Road Geometries -> H3 Cells (CPU Bound).
- Keeps API requests sequential to avoid Rate Limiting (Network Bound).
- Centralized caching and configuration.
"""
import sys
import time
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import h3.api.basic_int as h3
from h3 import LatLngPoly
from sqlalchemy import text
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

# --- Import Centralized Utilities ---
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
import utils
from utils import logger, PATHS

SCHEMA = "car_cewp"
COUNTRY_BBOX_STR = "14.1,2.0,27.5,11.0"
GRID_SIZE = 4
# Leave 1 core free for the OS and API handling
CORES = max(1, multiprocessing.cpu_count() - 1)

# ---------------------------------------------------------
# 1. WORKER FUNCTIONS (Must be top-level)
# ---------------------------------------------------------

def _process_geom_chunk(geoms, resolution):
    """
    Worker function: Converts a chunk of LineStrings to H3 cells.
    This runs on separate CPU cores.
    """
    local_cells = set()
    buffer_deg = 0.0001
    
    for geom in geoms:
        try:
            if geom.geom_type == 'MultiLineString':
                lines = list(geom.geoms)
            elif geom.geom_type == 'LineString':
                lines = [geom]
            else:
                continue
                
            for line in lines:
                # Buffer line to polygon (required for H3 polyfill)
                buffered = line.buffer(buffer_deg)
                if buffered.geom_type != 'Polygon': continue
                
                exterior = [(y, x) for x, y in buffered.exterior.coords]
                h3_poly = LatLngPoly(exterior)
                
                # Geometric conversion (CPU Intensive part)
                raw_cells = h3.polygon_to_cells(h3_poly, resolution)
                
                # Filter valid
                for c in raw_cells:
                    if c > 0: local_cells.add(c)
                    
        except Exception:
            continue
            
    return local_cells

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def check_table_exists(engine, table_name):
    """Check if table exists and has data."""
    with engine.connect() as conn:
        exists = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{SCHEMA}' 
                AND table_name = '{table_name}'
            );
        """)).scalar()
        
        if exists:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{table_name}")).scalar()
            return True, count
    
    return False, 0

def create_bbox_grid(bbox_str, grid_size):
    """Create spatial grid for tiled API requests."""
    w, s, e, n = [float(c) for c in bbox_str.split(',')]
    lon_lines = np.linspace(w, e, grid_size + 1)
    lat_lines = np.linspace(s, n, grid_size + 1)
    
    bboxes = []
    for i in range(grid_size):
        for j in range(grid_size):
            bboxes.append(f"{lon_lines[i]},{lat_lines[j]},{lon_lines[i+1]},{lat_lines[j+1]}")
    
    return bboxes

def fetch_ohsome_tile(year, bbox, api_url, road_filter, retries=3):
    """Fetch single tile from Ohsome API with retry logic."""
    timestamp = f"{year}-01-01"
    payload = {"bboxes": bbox, "filter": road_filter, "time": timestamp}
    
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(api_url, data=payload, timeout=120)
            resp.raise_for_status()
            feats = resp.json().get("features", [])
            
            if not feats:
                return None
            
            return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
            
        except Exception as e:
            logger.debug(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    
    return None

def upload_road_table(engine, year, road_cells):
    """Upload temporal road table to PostgreSQL."""
    table_name = f"osm_roads_h3_{year}"
    logger.info(f"  Uploading {table_name}...")
    
    df = pd.DataFrame({
        'h3_index': list(road_cells),
        'year': year
    })
    
    df.to_sql(table_name, engine, schema=SCHEMA, if_exists="replace", index=False)
    
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_h3 
            ON {SCHEMA}.{table_name} (h3_index)
        """))
    
    logger.info(f"   Upload complete.")

def upload_cities_table(engine, city_cells):
    """Upload cities table to PostgreSQL."""
    table_name = "osm_cities_h3"
    logger.info(f"  Uploading {table_name}...")
    
    df = pd.DataFrame({'h3_index': list(city_cells)})
    df.to_sql(table_name, engine, schema=SCHEMA, if_exists="replace", index=False)
    
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_h3 ON {SCHEMA}.{table_name} (h3_index)"))

# ---------------------------------------------------------
# 3. CORE LOGIC
# ---------------------------------------------------------

def fetch_cities(data_config, resolution):
    """
    Fetch cities using the HDX/SIGCAF Settlements Shapefile.
    """
    logger.info("\n--- Fetching Cities (HDX SIGCAF Data) ---")
    
    # Use centralized cache paths
    zip_path = PATHS["cache"] / "caf_settlements_sigcaf.zip"
    cities_h3_cache = PATHS["cache"] / f"sigcaf_cities_h3_r{resolution}.parquet"
    
    # 1. Check Cache
    if cities_h3_cache.exists():
        logger.info(f" Loading cities from processed cache: {cities_h3_cache}")
        try:
            return set(pd.read_parquet(cities_h3_cache)['h3_index'])
        except Exception:
            logger.warning("Cache read failed. Re-fetching...")

    # 2. Download if missing
    url = data_config["osm"]["settlements_hdx_url"]
    if not zip_path.exists():
        logger.info(f"Downloading from HDX: {url}")
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
            logger.info(" Download complete.")
        except Exception as e:
            logger.error(f"Failed to download settlements: {e}")
            return set()

    # 3. Process Shapefile
    try:
        logger.info("Reading Shapefile from Zip...")
        cities_gdf = gpd.read_file(zip_path)
        
        if cities_gdf.crs and cities_gdf.crs.to_string() != "EPSG:4326":
            cities_gdf = cities_gdf.to_crs("EPSG:4326")
            
        city_cells = []
        for geom in cities_gdf.geometry:
            try:
                # Handle Point or Polygon centroid
                lat = geom.centroid.y
                lon = geom.centroid.x
                cell = h3.latlng_to_cell(lat, lon, resolution)
                if cell > 0 and h3.is_valid_cell(cell):
                    city_cells.append(cell)
            except: continue
        
        city_cells = set(city_cells)
        logger.info(f" {len(city_cells):,} unique settlement cells")
        
        # Save to Cache
        if city_cells:
            pd.DataFrame({'h3_index': list(city_cells)}).to_parquet(cities_h3_cache)
            
        return city_cells
        
    except Exception as e:
        logger.warning(f"City data processing failed: {e}")
        return set()

def fetch_temporal_roads_parallel(year, bboxes, resolution, api_url, road_tags):
    """
    Fetch roads sequentially, but process geometries in parallel.
    """
    logger.info(f"--- Fetching & Processing roads for {year} (Cores: {CORES}) ---")
    
    filter_parts = [f"highway={tag}" for tag in road_tags["highway"]]
    road_filter = " or ".join(filter_parts)

    all_geometries = []
    
    # 1. FETCH PHASE (Sequential Network I/O)
    logger.info("Phase 1: Downloading tiles...")
    for i, bbox in enumerate(bboxes, 1):
        # Log occasionally
        if i % 5 == 0 or i == 1:
            logger.info(f"  Tile {i}/{len(bboxes)}")
            
        gdf = fetch_ohsome_tile(year, bbox, api_url, road_filter)
        if gdf is not None and not gdf.empty:
            # Filter for lines only
            gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
            all_geometries.extend(gdf.geometry.tolist())
            
    total_geoms = len(all_geometries)
    logger.info(f"  Downloaded {total_geoms:,} road segments.")
    
    if total_geoms == 0:
        return set()

    # 2. PROCESS PHASE (Parallel CPU)
    logger.info("Phase 2: Converting Geometries to H3 (Parallel)...")
    
    # Split geometries into chunks for each core
    chunk_size = int(np.ceil(total_geoms / CORES))
    chunks = [all_geometries[i:i + chunk_size] for i in range(0, total_geoms, chunk_size)]
    
    road_cells = set()
    
    with ProcessPoolExecutor(max_workers=CORES) as executor:
        # Submit all chunks
        futures = [executor.submit(_process_geom_chunk, chunk, resolution) for chunk in chunks]
        
        # Gather results as they complete
        for f in tqdm(futures, desc="Processing Chunks"):
            result_set = f.result()
            road_cells.update(result_set)
            
    logger.info(f" {year}: {len(road_cells):,} unique road cells generated.")
    return road_cells

# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------

def main():
    try:
        # 1. Load Configs
        data_config, _, _ = utils.load_configs()
        engine = utils.get_db_engine()
        
        resolution = data_config["h3"]["resolution"]
        years = [2010, 2015, 2020, 2025]
        
        # Extract OSM settings
        osm_cfg = data_config["osm"]
        api_url = osm_cfg["ohsome_api_url"]
        road_tags = osm_cfg["roads_tags"]
        
        logger.info("="*60)
        logger.info("STEP 2: TEMPORAL ROAD & CITY INGESTION (PARALLELIZED)")
        logger.info("="*60)
        logger.info(f"Resolution: {resolution} | Cores: {CORES}")

        # A. Cities
        cities_table = "osm_cities_h3"
        exists, count = check_table_exists(engine, cities_table)
        if exists and count > 0:
            logger.info(f" Cities already exist ({count:,} cells). Skipping.")
        else:
            city_cells = fetch_cities(data_config, resolution)
            if city_cells:
                upload_cities_table(engine, city_cells)
            else:
                logger.warning("No cities found to upload.")

        # B. Roads
        bboxes = create_bbox_grid(COUNTRY_BBOX_STR, GRID_SIZE)
        
        for year in years:
            table_name = f"osm_roads_h3_{year}"
            exists, count = check_table_exists(engine, table_name)
            
            if exists and count > 0:
                logger.info(f"\n {year} roads already exist ({count:,} cells). Skipping.")
                continue
            
            # Parallel Fetch & Process
            road_cells = fetch_temporal_roads_parallel(year, bboxes, resolution, api_url, road_tags)
            
            if not road_cells:
                logger.warning(f"No roads found for {year}. Creating empty table.")
            
            upload_road_table(engine, year, road_cells)
            
        logger.info("\n" + "="*60)
        logger.info("INGESTION COMPLETE")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()