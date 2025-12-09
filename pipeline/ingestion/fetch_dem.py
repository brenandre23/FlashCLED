"""
fetch_dem.py
=================================================
Pipeline: Ingestion & Terrain Processing
Task: 
  1. Fetches Copernicus DEM via Sentinel Hub.
  2. AUTOMATICALLY generates Slope and TRI rasters using GDAL.
  3. Prepares all static terrain rasters for zonal statistics.

FIXES APPLIED (Fix #8):
- Added automatic generation of 'slope_car.tif' and 'tri_car.tif'.
- Uses 'gdaldem' via subprocess to ensure robust, standard derivatives.
- Idempotent: Skips generation if files already exist.
"""
import sys
import geopandas as gpd
import requests
import os
import zipfile
import shutil
import numpy as np
import rasterio
from rasterio.merge import merge
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import shutil

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, get_boundary

SCHEMA = "car_cewp"

# --- 1. Copernicus DEM ---
def get_sentinel_credentials():
    """Load credentials securely from .env via utils paths."""
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)
    
    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")
    
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

def fetch_copernicus_dem(configs):
    """Downloads tiles and merges them into a single DEM raster."""
    dem_path = PATHS["data_proc"] / "copernicus_dem_90m.tif"
    
    if dem_path.exists():
        logger.info(f"DEM cached: {dem_path}. Skipping download.")
        return dem_path
    
    logger.info("Fetching Copernicus 90m DEM...")
    data_config = configs['data']
    features_config = configs['features']
    cop_cfg = data_config["copernicus"]
    
    client_id, client_secret = get_sentinel_credentials()
    token = get_sentinel_token(client_id, client_secret, cop_cfg["token_url"])
    
    boundary = get_boundary(data_config, features_config)
    metric_crs = features_config["spatial"]["crs"]["metric"]
    # Buffer to ensure coverage for slope calcs at edges
    boundary_buffered = boundary.to_crs(metric_crs).buffer(5000).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = boundary_buffered.total_bounds
    
    step = 1.5
    x_ranges = np.arange(minx, maxx + step, step)
    y_ranges = np.arange(miny, maxy + step, step)
    
    temp_dir = PATHS["cache"] / "dem_tiles"
    temp_dir.mkdir(exist_ok=True)
    tile_files = []
    
    logger.info(f"Downloading tiles to {temp_dir}...")
    for i in range(len(x_ranges)-1):
        for j in range(len(y_ranges)-1):
            tile_path = temp_dir / f"tile_{i}_{j}.tif"
            if not tile_path.exists():
                bbox = (x_ranges[i], y_ranges[j], x_ranges[i+1], y_ranges[j+1])
                fetch_dem_tile(bbox, token, tile_path, cop_cfg["process_api_url"])
            tile_files.append(str(tile_path))

    logger.info("Merging DEM tiles...")
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
    
    with rasterio.open(dem_path, "w", **out_meta) as dest: 
        dest.write(mosaic)
        
    for src in src_files: src.close()
    shutil.rmtree(temp_dir)
    logger.info(f"DEM saved to {dem_path}")
    return dem_path

# --- 2. Terrain Derivatives (GDAL) ---
def generate_terrain_derivatives(dem_path):
    """
    Generates Slope and TRI rasters from the base DEM using GDAL.
    Requirement: 'gdal-bin' must be installed in the environment.
    """
    logger.info("Generating Terrain Derivatives (Slope & TRI)...")
    
    # Define output paths
    slope_path = PATHS["data_proc"] / "slope_car.tif"
    tri_path = PATHS["data_proc"] / "tri_car.tif"
    
    # Check if GDAL is available
    if not shutil.which("gdaldem"):
        logger.critical("❌ 'gdaldem' command not found! Cannot compute Slope/TRI.")
        logger.critical("   Please install GDAL (apt install gdal-bin or conda install gdal).")
        # Fail loudly so user knows why features are missing
        sys.exit(1)

    # 1. Slope Generation
    if slope_path.exists():
        logger.info(f"  Slope raster already exists: {slope_path}")
    else:
        logger.info("  Computing Slope...")
        # gdaldem slope input output -compute_edges
        cmd = [
            "gdaldem", "slope", 
            str(dem_path), 
            str(slope_path),
            "-compute_edges"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("  ✓ Slope generated.")
        except subprocess.CalledProcessError as e:
            logger.error(f"  Failed to generate slope: {e.stderr.decode()}")
            sys.exit(1)

    # 2. TRI Generation (Terrain Ruggedness Index)
    if tri_path.exists():
        logger.info(f"  TRI raster already exists: {tri_path}")
    else:
        logger.info("  Computing Terrain Ruggedness Index (TRI)...")
        # gdaldem TRI input output -compute_edges
        cmd = [
            "gdaldem", "TRI", 
            str(dem_path), 
            str(tri_path),
            "-compute_edges"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("  ✓ TRI generated.")
        except subprocess.CalledProcessError as e:
            logger.error(f"  Failed to generate TRI: {e.stderr.decode()}")
            sys.exit(1)

def run(configs, engine):
    try:
        logger.info("--- Running Static Geospatial Ingestion ---")
        
        # 1. Fetch/Cache Base DEM
        dem_path = fetch_copernicus_dem(configs)
        
        # 2. Generate Derivatives (Slope, TRI) locally
        generate_terrain_derivatives(dem_path)
        
        logger.info("--- Static Geospatial Ingestion Complete ---")
        
    except Exception as e:
        logger.error(f"Static Geospatial Ingestion failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Standalone execution
    from utils import load_configs, get_db_engine
    
    cfg = load_configs()
    # Handle tuple vs dict return from load_configs
    if isinstance(cfg, tuple):
        configs = {"data": cfg[0], "features": cfg[1], "models": cfg[2]}
    else:
        configs = cfg
        
    run(configs, get_db_engine())