import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
from rasterstats import zonal_stats
from sqlalchemy import text
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import requests 
import shutil # Added for file copying

# --- Import Utilities ---
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent.parent / "scripts"))
try:
    import utils
    from utils import logger, PATHS
except ImportError:
    print("FATAL: Could not import 'utils.py'.")
    print("Ensure 'utils.py' is in the 'scripts/' folder.")
    sys.exit(1)

# --- Script Constants ---
SCHEMA = "car_cewp"
WORLDPOP_TABLE = "population_h3" # WorldPop v2 (Already in DB)
GLOBPOP_TABLE = "population_globpop_h3"
WORLDPOP_V1_TABLE = "population_worldpop_v1_h3" 
H3_GRID_TABLE = "features_static"

# *** CONFIGURATION ***
# Set to True to re-run the ingest for specific datasets
FORCE_REINGEST_GLOBPOP = False 
FORCE_REINGEST_WP_V1 = True   # Set to True to load your new local files

# Path to your local WorldPop v1 downloads
LOCAL_V1_SOURCE_DIR = Path(r"C:\Users\Brenan\Downloads")

# Years for 1-to-1 comparison
YEARS_TO_TEST = [2015, 2016, 2017, 2018, 2019, 2020]
ZENODO_RECORD_ID = "11179644"
CORES = max(1, multiprocessing.cpu_count() - 1)
TEST_DIR = PATHS["root"] / "test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PATHS["data_raw"] / "worldpop_comparison"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------
# MODULE 0: DATA ACQUISITION
# -----------------------------------------------------------------

def download_globpop_data(data_dir, years):
    """Ensures GlobPop TIF files exist, downloading from Zenodo if missing."""
    logger.info(f"\n--- MODULE 0a: Verifying/Downloading GlobPop Data ---")
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    
    # Check if all files already exist to avoid API call if possible
    all_exist = all((data_dir / f"GlobPOP_{y}.tif").exists() for y in years)
    if all_exist:
        # logger.info("All GlobPop files found locally.")
        return

    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Failed to access Zenodo API: {e}")
        return
        
    file_map = {}
    for file_info in data.get('files', []):
        api_filename = file_info.get('key') or file_info.get('filename')
        if not api_filename: continue
        if "Count_30arc" in api_filename and "I32.tiff" in api_filename:
            file_map[api_filename] = (file_info['links']['self'], file_info['size'])

    for year in tqdm(years, desc="Verifying GlobPop Files"):
        api_filename = f"GlobPOP_Count_30arc_{year}_I32.tiff"
        target_filename = f"GlobPOP_{year}.tif"
        target_path = data_dir / target_filename
        
        if api_filename not in file_map: continue
        
        download_url, expected_size = file_map[api_filename]
        if target_path.exists() and target_path.stat().st_size == expected_size: continue
        
        logger.info(f"  Downloading {target_filename}...")
        try:
            with requests.get(download_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
        except Exception as e:
            logger.error(f"  ✗ FAILED to download {target_filename}: {e}")
            if target_path.exists(): target_path.unlink()

def import_local_worldpop_v1(data_dir, source_dir, years):
    """
    Copies WorldPop v1 files from your local Downloads folder.
    Expected pattern: caf_ppp_{YEAR}_UNadj.tif
    """
    logger.info(f"\n--- MODULE 0b: Importing WorldPop v1 from Local Drive ---")
    logger.info(f"Source: {source_dir}")

    if not source_dir.exists():
        logger.error(f"CRITICAL: Source directory not found: {source_dir}")
        return

    for year in tqdm(years, desc="Copying Local Files"):
        # Pattern from your description
        src_filename = f"caf_ppp_{year}_UNadj.tif"
        src_path = source_dir / src_filename
        
        # Renaming to match script convention
        dst_filename = f"WP_v1_{year}.tif"
        dst_path = data_dir / dst_filename
        
        if not src_path.exists():
            logger.warning(f"  Missing source file: {src_filename}")
            continue
            
        if dst_path.exists():
            # Optional: check size or just skip if it exists
            if dst_path.stat().st_size == src_path.stat().st_size:
                continue
        
        try:
            shutil.copy2(src_path, dst_path)
            # logger.info(f"  ✓ Copied {src_filename}")
        except Exception as e:
            logger.error(f"  ✗ Failed to copy {src_filename}: {e}")

# -----------------------------------------------------------------
# MODULE 1: DATA INGESTION (WITH CLIPPING)
# -----------------------------------------------------------------

def _zonal_worker(args):
    """Worker for zonal stats using pre-clipped array."""
    gdf_chunk, raster_data, affine, nodata_val = args
    try:
        stats = zonal_stats(
            gdf_chunk, raster_data, affine=affine, stats=["sum"], 
            all_touched=True, nodata=nodata_val
        )
        pop_values = [s['sum'] if s['sum'] is not None else 0 for s in stats]
    except Exception as e:
        logger.error(f"Zonal stats error: {e}")
        pop_values = [0] * len(gdf_chunk)
    return pd.DataFrame({'h3_index': gdf_chunk['h3_index'].values, 'pop_count': pop_values})

def ingest_dataset(engine, grid_gdf, table_name, file_prefix, years, force_reingest):
    """Ingest dataset with CRS alignment and Clipping."""
    logger.info(f"\n--- MODULE 1: Ingesting {table_name} ---")
    
    with engine.begin() as conn:
        if force_reingest:
            logger.warning(f"FORCE_REINGEST=True. Dropping {SCHEMA}.{table_name}...")
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{table_name};"))
            
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{table_name} (
                h3_index TEXT, year INTEGER, pop_count FLOAT );
            CREATE INDEX IF NOT EXISTS idx_{table_name}_h3 ON {SCHEMA}.{table_name} (h3_index);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_year ON {SCHEMA}.{table_name} (year);
        """))

    for year in tqdm(years, desc=f"Ingesting {table_name}"):
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT count(*) FROM {SCHEMA}.{table_name} WHERE year = {year}")).scalar()
        if count > 0: continue

        tif_path = DATA_DIR / f"{file_prefix}{year}.tif"
        if not tif_path.exists():
            logger.warning(f"  File missing: {tif_path}. Skipping {year}.")
            continue
            
        logger.info(f"  Processing {year} (Clipping & Zonal Stats)...")

        try:
            with rasterio.open(tif_path) as src:
                raster_crs = src.crs; nodata_val = src.nodata
                
                # 1. Reproject H3 grid if needed
                if grid_gdf.crs != raster_crs:
                    logger.info(f"  Reprojecting grid to {raster_crs}...")
                    grid_for_clip = grid_gdf.to_crs(raster_crs)
                else:
                    grid_for_clip = grid_gdf

                # 2. Clip raster to grid extent
                clipped_data, clipped_affine = rasterio.mask.mask(
                    src, grid_for_clip.geometry, crop=True, all_touched=True
                )
                clipped_data = clipped_data.squeeze(0)

            # 3. Parallel Processing
            with np.errstate(invalid='ignore'):
                chunks = np.array_split(grid_for_clip, CORES) 
            
            worker_args = [(chunk, clipped_data, clipped_affine, nodata_val) for chunk in chunks]
            results = []

            with ProcessPoolExecutor(max_workers=CORES) as executor:
                for res in executor.map(_zonal_worker, worker_args):
                    results.append(res)
                    
            df_year = pd.concat(results)
            df_year['year'] = year
            
            df_year.to_sql(table_name, engine, schema=SCHEMA, if_exists='append', index=False, chunksize=10000)
            logger.info(f"  ✓ Uploaded {len(df_year):,} rows")

        except Exception as e:
            logger.error(f"  ✗ FAILED ingest for {year}: {e}", exc_info=True)

# -----------------------------------------------------------------
# MODULE 2: ANALYSIS & VIZ (Standard)
# -----------------------------------------------------------------

def run_comparison_analysis(engine):
    logger.info(f"\n--- MODULE 2: Running 3-Way Comparison ---")
    sql = f"""
    WITH wp_v2 AS (SELECT h3_index, year, pop_count AS p2 FROM {SCHEMA}.{WORLDPOP_TABLE} WHERE year BETWEEN {YEARS_TO_TEST[0]} AND {YEARS_TO_TEST[-1]}),
         gp AS (SELECT h3_index, year, pop_count AS gp FROM {SCHEMA}.{GLOBPOP_TABLE} WHERE year BETWEEN {YEARS_TO_TEST[0]} AND {YEARS_TO_TEST[-1]}),
         wp_v1 AS (SELECT h3_index, year, pop_count AS p1 FROM {SCHEMA}.{WORLDPOP_V1_TABLE} WHERE year BETWEEN {YEARS_TO_TEST[0]} AND {YEARS_TO_TEST[-1]})
    SELECT wp_v2.h3_index, wp_v2.year, wp_v2.p2 AS pop_wp_v2, 
           COALESCE(gp.gp, 0) AS pop_globpop, COALESCE(wp_v1.p1, 0) AS pop_wp_v1
    FROM wp_v2 
    LEFT JOIN gp ON wp_v2.h3_index = gp.h3_index AND wp_v2.year = gp.year
    LEFT JOIN wp_v1 ON wp_v2.h3_index = wp_v1.h3_index AND wp_v2.year = wp_v1.year;
    """
    try:
        df = pd.read_sql(sql, engine)
    except Exception as e:
        logger.error(f"Fetch failed: {e}"); return None

    if df.empty: logger.error("No data found."); return None

    stats = []
    for year in YEARS_TO_TEST:
        df_y = df[df['year'] == year]
        if df_y.empty: continue
        
        # Stats
        df_filt = df_y[(df_y['pop_wp_v2']>0) | (df_y['pop_globpop']>0) | (df_y['pop_wp_v1']>0)]
        r2_v2_gp = df_filt['pop_wp_v2'].corr(df_filt['pop_globpop'])**2
        r2_v2_v1 = df_filt['pop_wp_v2'].corr(df_filt['pop_wp_v1'])**2
        r2_v1_gp = df_filt['pop_wp_v1'].corr(df_filt['pop_globpop'])**2
        
        stats.append({
            "Year": year,
            "Total_WP_v2": df_y['pop_wp_v2'].sum(),
            "Total_GlobPop": df_y['pop_globpop'].sum(),
            "Total_WP_v1": df_y['pop_wp_v1'].sum(),
            "R2 (v2-GP)": r2_v2_gp,
            "R2 (v2-v1)": r2_v2_v1,
            "R2 (v1-GP)": r2_v1_gp
        })

    pd.DataFrame(stats).to_csv(TEST_DIR / "population_comparison_summary_3way.csv", index=False)
    print(pd.DataFrame(stats).to_string(index=False, float_format="%.2f"))
    return df

def visualize_comparison(df_compare, grid_geom):
    if df_compare is None or df_compare.empty: return
    logger.info(f"\n--- MODULE 3: Visualization ---")
    
    try:
        grid_geom['h3_index'] = grid_geom['h3_index'].astype(np.int64)
        df_compare['h3_index_int'] = df_compare['h3_index'].astype(np.int64)
        gdf = grid_geom.merge(df_compare, left_on='h3_index', right_on='h3_index_int')
    except Exception: return

    pairs = [
        ('pop_wp_v2', 'pop_globpop', 'WP v2', 'GlobPop'),
        ('pop_wp_v2', 'pop_wp_v1', 'WP v2', 'WP v1'),
        ('pop_wp_v1', 'pop_globpop', 'WP v1', 'GlobPop')
    ]

    for col_x, col_y, name_x, name_y in pairs:
        # Scatter
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        fig.suptitle(f"{name_y} vs. {name_x}", fontsize=16)
        max_val = max(gdf[col_x].max(), gdf[col_y].max())
        
        for i, year in enumerate(YEARS_TO_TEST):
            ax = axes.flat[i]
            yd = gdf[gdf['year'] == year]
            if yd.empty: continue
            samp = yd.sample(n=min(20000, len(yd)), random_state=1)
            ax.scatter(samp[col_x], samp[col_y], alpha=0.1, s=2)
            ax.plot([0, max_val], [0, max_val], 'r--', lw=1)
            ax.set_xlim(0, max_val*0.6); ax.set_ylim(0, max_val*0.6)
            ax.set_title(f"{year}")
        
        plt.savefig(TEST_DIR / f"scatter_{name_x}_vs_{name_y}.png", dpi=100)
        plt.close(fig)

def main():
    try:
        engine = utils.get_db_engine()
        
        # 1. Get Data
        download_globpop_data(DATA_DIR, YEARS_TO_TEST)
        import_local_worldpop_v1(DATA_DIR, LOCAL_V1_SOURCE_DIR, YEARS_TO_TEST)
        
        # 2. Load Grid
        logger.info("Loading Grid...")
        grid_gdf = gpd.read_postgis(f"SELECT h3_index, geometry FROM {SCHEMA}.{H3_GRID_TABLE}", engine, geom_col='geometry')
        if grid_gdf.crs is None: grid_gdf.set_crs("EPSG:4326", inplace=True)
        
        # 3. Ingest
        ingest_dataset(engine, grid_gdf, GLOBPOP_TABLE, "GlobPOP_", YEARS_TO_TEST, FORCE_REINGEST_GLOBPOP)
        ingest_dataset(engine, grid_gdf, WORLDPOP_V1_TABLE, "WP_v1_", YEARS_TO_TEST, FORCE_REINGEST_WP_V1)
        
        # 4. Analyze & Viz
        df = run_comparison_analysis(engine)
        visualize_comparison(df, grid_gdf)
        
        logger.info("Done. Results in 'test/' folder.")
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    multiprocessing.freeze_support()
    main()