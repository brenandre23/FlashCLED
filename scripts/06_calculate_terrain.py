"""
07_calculate_terrain.py
=================================
Purpose: Calculate terrain features (Elevation, Slope, Ruggedness) & Geographic distances.
Output: Updates 'car_cewp.features_static' columns.

Methods:
- DEM: Copernicus 90m (fetched via Sentinel Hub)
- Slope/TRI: Vectorized Numpy calculations
- Zonal Stats: Parallel processing (Multiprocessing)
- Distances: PostGIS ST_Distance (Capital, Border)
"""
import sys
import shutil
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats
from sqlalchemy import text
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from dotenv import load_dotenv

# --- Import Centralized Utilities ---
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
import utils
from utils import logger, PATHS

SCHEMA = "car_cewp"
STATIC_TABLE = "features_static"

# Coordinates for Bangui (could also be moved to config)
BANGUI_LAT = 4.3612
BANGUI_LNG = 18.5550

# Auto-detect cores (leave 1 free for system)
CORES = max(1, multiprocessing.cpu_count() - 1)

# ---------------------------------------------------------
# 1. AUTH & DOWNLOAD
# ---------------------------------------------------------

def get_sentinel_credentials():
    """Load credentials securely from .env via utils paths."""
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)
    
    client_id = utils.os.getenv("SH_CLIENT_ID")
    client_secret = utils.os.getenv("SH_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise ValueError(
            f"Missing Sentinel Hub credentials in {env_path}.\n"
            "Please add SH_CLIENT_ID and SH_CLIENT_SECRET."
        )
    return client_id, client_secret

def get_sentinel_token(client_id, client_secret, token_url):
    try:
        payload = {
            "grant_type": "client_credentials", 
            "client_id": client_id, 
            "client_secret": client_secret
        }
        response = requests.post(token_url, data=payload, timeout=30)
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        logger.error(f"Sentinel Hub Authentication failed: {e}")
        sys.exit(1)

def fetch_dem_tile(bbox, token, output_path, process_url):
    """Fetch a single DEM tile from Sentinel Hub."""
    evalscript = """
    //VERSION=3
    function setup() { return { input: ["DEM"], output: { bands: 1, sampleType: "FLOAT32" } }; }
    function evaluatePixel(sample) { return [sample.DEM]; }
    """
    payload = {
        "input": {
            "bounds": {
                "bbox": list(bbox), 
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{"type": "DEM", "dataFilter": {"demInstance": "COPERNICUS_90"}}]
        },
        "output": {
            "width": 2000, 
            "height": 2000, 
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript
    }
    try:
        resp = requests.post(
            process_url, 
            json=payload, 
            headers={"Authorization": f"Bearer {token}"}, 
            timeout=60
        )
        resp.raise_for_status()
        with open(output_path, "wb") as f: 
            f.write(resp.content)
        return True
    except Exception:
        return False

def fetch_copernicus_dem(boundary_gdf, output_path, data_config):
    """Main logic to download and merge DEM tiles."""
    # 1. Auth
    client_id, client_secret = get_sentinel_credentials()
    
    cop_cfg = data_config["copernicus"]
    token_url = cop_cfg["token_url"]
    process_url = cop_cfg["process_api_url"]
    
    token = get_sentinel_token(client_id, client_secret, token_url)
    
    logger.info("Fetching Copernicus 90m DEM (Tiled)...")
    
    # 2. Define Grid
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    step = 1.5 # Degrees per tile
    x_ranges = np.arange(minx, maxx + step, step)
    y_ranges = np.arange(miny, maxy + step, step)
    
    # 3. Prepare Cache
    temp_dir = PATHS["cache"] / "dem_tiles"
    temp_dir.mkdir(exist_ok=True)
    
    tile_files = []
    total = (len(x_ranges)-1) * (len(y_ranges)-1)
    logger.info(f"Downloading ~{total} tiles...")
    
    # 4. Download Loop
    for i in range(len(x_ranges)-1):
        for j in range(len(y_ranges)-1):
            tile_path = temp_dir / f"tile_{i}_{j}.tif"
            
            if not tile_path.exists():
                bbox = (x_ranges[i], y_ranges[j], x_ranges[i+1], y_ranges[j+1])
                if fetch_dem_tile(bbox, token, tile_path, process_url):
                    tile_files.append(str(tile_path))
            else:
                tile_files.append(str(tile_path))

    if not tile_files: 
        return False

    # 5. Merge
    logger.info("Merging tiles...")
    src_files = [rasterio.open(fp) for fp in tile_files]
    mosaic, out_trans = merge(src_files)
    
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff", 
        "height": mosaic.shape[1], 
        "width": mosaic.shape[2], 
        "transform": out_trans, 
        "compress": "lzw"
    })
    
    with rasterio.open(output_path, "w", **out_meta) as dest: 
        dest.write(mosaic)
    
    # 6. Cleanup
    for src in src_files: src.close()
    shutil.rmtree(temp_dir)
    return True

# ---------------------------------------------------------
# 2. FAST VECTORIZED TERRAIN CALCULATIONS
# ---------------------------------------------------------

def calculate_terrain_derivatives_fast(dem_path, slope_path, tri_path):
    # CHECKPOINT
    if slope_path.exists() and tri_path.exists():
        logger.info(" Terrain derivatives cached (Slope & TRI). Skipping.")
        return True

    logger.info("Calculating terrain derivatives (Vectorized)...")
    
    # Use str(path) for rasterio
    with rasterio.open(str(dem_path)) as src:
        elev = src.read(1)
        profile = src.profile
        res_x_deg, res_y_deg = src.res
        
        # CONVERSION: 1 degree ≈ 111,120 meters (Simplified)
        scale_factor = 111120
        res_x_m = res_x_deg * scale_factor
        res_y_m = res_y_deg * scale_factor
        
        # A. SLOPE
        if not slope_path.exists():
            logger.info("  Vectorized Slope...")
            py, px = np.gradient(elev, res_y_m, res_x_m)
            slope = np.degrees(np.arctan(np.sqrt(px**2 + py**2)))
            
            if src.nodata is not None:
                slope[elev == src.nodata] = -9999
                profile.update(nodata=-9999)
                
            profile.update(dtype=rasterio.float32, compress='lzw')
            with rasterio.open(str(slope_path), 'w', **profile) as dst:
                dst.write(slope.astype(rasterio.float32), 1)
        
        # B. TRI (Terrain Ruggedness Index)
        if not tri_path.exists():
            logger.info("  Vectorized TRI...")
            padded = np.pad(elev, 1, mode='edge')
            diff_sum = np.zeros_like(elev, dtype=np.float32)
            shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            
            for rs, cs in shifts:
                neighbor = padded[1+rs : padded.shape[0]-1+rs, 1+cs : padded.shape[1]-1+cs]
                diff_sum += np.abs(elev - neighbor)
                
            tri = diff_sum / 8.0
            
            # Normalize TRI (0-1)
            mn, mx = np.nanmin(tri), np.nanmax(tri)
            tri_norm = (tri - mn) / (mx - mn) if mx > mn else np.zeros_like(tri)
            
            if src.nodata is not None:
                tri_norm[elev == src.nodata] = -9999
                
            profile.update(dtype=rasterio.float32, compress='lzw')
            with rasterio.open(str(tri_path), 'w', **profile) as dst:
                dst.write(tri_norm.astype(rasterio.float32), 1)
                
    return True

# ---------------------------------------------------------
# 3. MULTIPROCESSING ZONAL STATS
# ---------------------------------------------------------

def _zonal_worker(args):
    """Worker for parallel zonal stats."""
    gdf_chunk, raster_paths = args
    results = {}
    
    for col, r_path in raster_paths.items():
        # Use str(path) for rasterstats
        stats = zonal_stats(
            gdf_chunk, 
            str(r_path), 
            stats=["mean"], 
            all_touched=True, 
            nodata=-9999
        )
        results[col] = [s['mean'] for s in stats]
        
    return pd.DataFrame(results, index=gdf_chunk.index)

def calculate_zonal_stats_parallel(engine, dem_path, slope_path, tri_path):
    logger.info(f"\n--- Calculating Zonal Stats (Parallel: {CORES} cores) ---")
    
    # Checkpoint: Check if columns have data
    try:
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(elevation_mean) FROM {SCHEMA}.{STATIC_TABLE} WHERE elevation_mean IS NOT NULL")).scalar()
            if count > 0:
                logger.info(" Zonal stats already exist. Skipping.")
                return
    except Exception: pass

    # Load Geometry
    grid_gdf = gpd.read_postgis(
        f"SELECT h3_index, geometry FROM {SCHEMA}.{STATIC_TABLE}", 
        engine, geom_col='geometry'
    )
    
    # Add Columns
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE {SCHEMA}.{STATIC_TABLE}
            ADD COLUMN IF NOT EXISTS elevation_mean FLOAT,
            ADD COLUMN IF NOT EXISTS slope_mean FLOAT,
            ADD COLUMN IF NOT EXISTS terrain_ruggedness_mean FLOAT;
        """))
    
    raster_paths = {
        "elevation_mean": dem_path,
        "slope_mean": slope_path,
        "terrain_ruggedness_mean": tri_path
    }
    
    # Split for Multiprocessing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        chunks = np.array_split(grid_gdf, CORES)
    
    worker_args = [(chunk, raster_paths) for chunk in chunks]
    
    results_list = []
    with ProcessPoolExecutor(max_workers=CORES) as executor:
        for res in tqdm(executor.map(_zonal_worker, worker_args), total=CORES, desc="Processing Chunks"):
            results_list.append(res)
            
    # Merge & Upload
    logger.info("Combining results...")
    full_results = pd.concat(results_list)
    grid_gdf = grid_gdf.join(full_results)
    
    logger.info("Uploading stats to DB...")
    cols_to_upload = ["h3_index", "elevation_mean", "slope_mean", "terrain_ruggedness_mean"]
    temp_table = "temp_terrain_upd"
    
    grid_gdf[cols_to_upload].to_sql(temp_table, engine, schema=SCHEMA, if_exists='replace', index=False)
    
    # Fast Update via Join
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX ON {SCHEMA}.{temp_table} (h3_index)"))
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE} f
            SET elevation_mean = t.elevation_mean,
                slope_mean = t.slope_mean,
                terrain_ruggedness_mean = t.terrain_ruggedness_mean
            FROM {SCHEMA}.{temp_table} t
            WHERE f.h3_index = t.h3_index
        """))
        conn.execute(text(f"DROP TABLE {SCHEMA}.{temp_table}"))
        
    logger.info(" Stats updated.")

# ---------------------------------------------------------
# 4. GEOGRAPHIC DISTANCE
# ---------------------------------------------------------

def calculate_geo_distances(engine, boundary_gdf):
    logger.info("\n--- Calculating Geographic Distances ---")
    
    # 1. Distance to Capital
    try:
        with engine.connect() as conn:
            c = conn.execute(text(f"SELECT count(dist_capital_km) FROM {SCHEMA}.{STATIC_TABLE}")).scalar()
            if c > 0:
                logger.info(" Capital Distance exists. Skipping.")
            else:
                raise Exception("Recalculate")
    except:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{STATIC_TABLE} ADD COLUMN IF NOT EXISTS dist_capital_km FLOAT"))
            conn.execute(text(f"""
                UPDATE {SCHEMA}.{STATIC_TABLE}
                SET dist_capital_km = ST_Distance(
                    ST_Centroid(geometry)::geography, 
                    ST_Point({BANGUI_LNG}, {BANGUI_LAT})::geography
                ) / 1000.0
                WHERE dist_capital_km IS NULL;
            """))
            logger.info(" Calculated Capital Distance")

    # 2. Distance to Border
    has_border_data = False
    try:
        with engine.connect() as conn:
            r = conn.execute(text(f"SELECT count(dist_border_km) FROM {SCHEMA}.{STATIC_TABLE}")).scalar()
            if r > 0: has_border_data = True
    except: pass

    if not has_border_data:
        logger.info("Calculating Border Distance...")
        
        # Handle Union (GeoPandas 1.0+ vs older)
        try:
            boundary_line = boundary_gdf.union_all().boundary
        except AttributeError:
            boundary_line = boundary_gdf.unary_union.boundary
        
        gpd.GeoDataFrame({'geometry': [boundary_line]}, crs=boundary_gdf.crs)\
           .to_postgis("temp_border", engine, schema=SCHEMA, if_exists='replace')
        
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{STATIC_TABLE} ADD COLUMN IF NOT EXISTS dist_border_km FLOAT"))
            conn.execute(text(f"""
                UPDATE {SCHEMA}.{STATIC_TABLE} f
                SET dist_border_km = ST_Distance(
                    ST_Centroid(f.geometry)::geography, 
                    b.geometry::geography
                ) / 1000.0
                FROM {SCHEMA}.temp_border b
                WHERE f.dist_border_km IS NULL
            """))
            conn.execute(text(f"DROP TABLE {SCHEMA}.temp_border"))
        logger.info(" Calculated Border Distance")
    else:
        logger.info(" Border Distance exists. Skipping.")

def main():
    try:
        # 1. Setup
        data_config, features_config, _ = utils.load_configs()
        engine = utils.get_db_engine()
        
        logger.info("="*60)
        logger.info(f"STEP 4: TERRAIN & GEOGRAPHY (OPTIMIZED - {CORES} CORES)")
        logger.info("="*60)
        
        # 2. Paths
        dem_path = PATHS["data_proc"] / "copernicus_dem_90m.tif"
        slope_path = PATHS["data_proc"] / "slope_car.tif"
        tri_path = PATHS["data_proc"] / "tri_car.tif"
        
        # 3. Fetch DEM
        if not dem_path.exists():
            # Use utils.get_boundary for logic consistency
            boundary = utils.get_boundary(data_config, features_config)
            # Buffer slightly for raster context
            boundary_buffered = boundary.to_crs(features_config["metric_crs"]).buffer(5000).to_crs("EPSG:4326")
            
            if not fetch_copernicus_dem(boundary_buffered, str(dem_path), data_config):
                logger.error("Failed to fetch DEM.")
                sys.exit(1)
        else:
            logger.info(f" DEM cached: {dem_path}")
            
        # 4. Derivatives & Stats
        calculate_terrain_derivatives_fast(dem_path, slope_path, tri_path)
        calculate_zonal_stats_parallel(engine, dem_path, slope_path, tri_path)
        
        # 5. Geographic Distances
        # Refetch strict boundary (no buffer) for border distance
        boundary_strict = utils.get_boundary(data_config, features_config)
        calculate_geo_distances(engine, boundary_strict)
        
        logger.info("\nTERRAIN PROCESSING COMPLETE")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()