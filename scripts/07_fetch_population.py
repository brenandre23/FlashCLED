"""
07b_fetch_population.py
========================
Purpose: Download WorldPop (Constrained) data and aggregate to H3 grid.
Output: Populates 'car_cewp.population_h3' table (h3_index, year, pop_count).

Method:
- Downloads Raster (TIFF) for each year from WorldPop.
- Uses Zonal Statistics (Parallelized) to sum population per H3 cell.
- Uploads to PostGIS.
"""
import sys
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
from sqlalchemy import text
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

# --- Import Centralized Utilities ---
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
import utils
from utils import logger, PATHS

SCHEMA = "car_cewp"
TABLE_NAME = "population_h3"

# Auto-detect cores (leave 1 free)
CORES = max(1, multiprocessing.cpu_count() - 1)

def download_file(year, output_path, base_url, filename_template):
    """Download a single WorldPop TIFF file."""
    # Construct URL (WorldPop structure: Year / ISO3 / v1 / resolution / type / filename)
    # Note: The base_url in data.yaml should point to .../Global_2000_2020_Constrained
    
    filename = filename_template.format(year=year)
    
    # Construct specific URL path for CAR (ISO3: CAF)
    # URL structure matches WorldPop Constrained data hierarchy
    url = f"{base_url}/{year}/CAF/v1/100m/constrained/{filename}"
    
    if output_path.exists():
        return True

    logger.info(f"  ⬇ Downloading {year} from {url}...")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Remove partial file if failed
        if output_path.exists():
            output_path.unlink()
        return False

def check_year_exists(engine, year):
    """Check if data for this year already exists in the DB."""
    try:
        with engine.connect() as conn:
            # Check if table exists first
            exists = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{SCHEMA}' 
                    AND table_name = '{TABLE_NAME}'
                );
            """)).scalar()
            
            if not exists: return False
            
            # Check for data rows
            count = conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.{TABLE_NAME} WHERE year = {year}")).scalar()
            return count > 0
    except Exception:
        return False

def _zonal_worker(args):
    """Helper worker for parallel processing."""
    gdf_chunk, raster_path = args
    
    # Calculate Sum of population per cell
    # nodata=-9999 ensures we don't count null space as population
    stats = zonal_stats(
        gdf_chunk, 
        str(raster_path), # rasterstats requires string path
        stats=["sum"], 
        all_touched=True
    )
    
    # Extract sums (handle Nones -> 0)
    pop_values = [s['sum'] if s['sum'] is not None else 0 for s in stats]
    
    return pd.DataFrame({
        'h3_index': gdf_chunk['h3_index'].values,
        'pop_count': pop_values
    })

def process_population_year_parallel(engine, year, raster_path, grid_gdf):
    """Calculate population sum per H3 cell using Multiprocessing."""
    logger.info(f"  Processing {year} (Zonal Stats on {CORES} cores)...")
    
    # 1. Split Grid into Chunks
    with np.errstate(invalid='ignore'): # Suppress numpy warnings
        chunks = np.array_split(grid_gdf, CORES)
    
    worker_args = [(chunk, raster_path) for chunk in chunks]
    
    # 2. Run in Parallel
    results_list = []
    with ProcessPoolExecutor(max_workers=CORES) as executor:
        for res in executor.map(_zonal_worker, worker_args):
            results_list.append(res)
            
    # 3. Combine Results
    df_year = pd.concat(results_list)
    df_year['year'] = year
    
    # 4. Upload
    df_year.to_sql(TABLE_NAME, engine, schema=SCHEMA, if_exists='append', index=False)
    logger.info(f"   Uploaded {len(df_year):,} rows for {year}")

def main():
    try:
        # 1. Setup
        data_config, features_config, _ = utils.load_configs()
        engine = utils.get_db_engine()
        
        # Load Population Configs
        pop_conf = data_config["population"]
        base_url = pop_conf["base_url"]
        start_year = pop_conf["start_year"]
        end_year = pop_conf["end_year"]
        fname_template = pop_conf["filename_template"]
        
        # Define/Create Directory
        pop_dir = PATHS["data_raw"] / "worldpop"
        pop_dir.mkdir(parents=True, exist_ok=True)
            
        years = range(start_year, end_year + 1)
        
        logger.info("="*60)
        logger.info(f"STEP 7b: POPULATION INGESTION ({start_year}-{end_year})")
        logger.info("="*60)

        # 2. Initialize Table
        with engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                    h3_index TEXT,
                    year INTEGER,
                    pop_count FLOAT
                );
                CREATE INDEX IF NOT EXISTS idx_pop_h3 ON {SCHEMA}.{TABLE_NAME} (h3_index);
                CREATE INDEX IF NOT EXISTS idx_pop_year ON {SCHEMA}.{TABLE_NAME} (year);
            """))
            
        # 3. Load Grid Geometry (Once)
        logger.info("Loading H3 Grid...")
        grid_gdf = gpd.read_postgis(
            f"SELECT h3_index, geometry FROM {SCHEMA}.features_static", 
            engine, geom_col='geometry'
        )

        # 4. Process Loop
        for year in tqdm(years, desc="Processing Years"):
            
            # CHECKPOINT
            if check_year_exists(engine, year):
                logger.info(f"   Year {year} exists in DB. Skipping.")
                continue
            
            # Download
            filename = fname_template.format(year=year)
            file_path = pop_dir / filename
            
            if download_file(year, file_path, base_url, fname_template):
                # Process
                process_population_year_parallel(engine, year, file_path, grid_gdf)
                
                # Optional: Clean up to save space
                # file_path.unlink() 
            else:
                logger.warning(f"Skipping {year} due to download failure")

        logger.info("\n" + "="*60)
        logger.info("POPULATION INGESTION COMPLETE")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Population script failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()