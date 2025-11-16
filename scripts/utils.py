"""
utils.py
========
Centralized utilities for the H3-based conflict early warning pipeline.
Handles paths, configuration, database connections, and geospatial helpers.

KEY FEATURES:
- Pathlib Integration: Robust, cross-platform file paths.
- Config Loading: Single source of truth for YAML configs.
- Database Safety: Secure connection handling via .env.
- Geodata Caching: Intelligent caching for boundary files and OSM data.
"""
import os
import logging
import time
import yaml
import geopandas as gpd
import h3.api.basic_int as h3
import osmnx as ox
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# ================================================================
# 1. CENTRALIZED PATHS (Using pathlib)
# ================================================================
# Assumes this file is located in: Scratch/scripts/utils.py
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# Define the directory structure
PATHS = {
    "root": ROOT_DIR,
    "scripts": SCRIPT_DIR,
    "configs": ROOT_DIR / "configs",
    "data": ROOT_DIR / "data",
    "data_raw": ROOT_DIR / "data" / "raw",
    "data_proc": ROOT_DIR / "data" / "processed",
    "cache": ROOT_DIR / "cache",
    "logs": ROOT_DIR / "logs"
}

# Automatically ensure these directories exist
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# ================================================================
# 2. LOGGING CONFIGURATION
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Log to file as well for audit trails
        logging.FileHandler(PATHS["logs"] / "pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# 3. DATABASE CONNECTION
# ================================================================
def get_db_engine():
    """
    Create SQLAlchemy engine for PostgreSQL.
    Loads credentials securely from .env in the project root.
    """
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError(
            f"DATABASE_URL not found in {env_path}.\n"
            "Expected format: postgresql://user:pass@host:port/database"
        )
    
    try:
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise

# ================================================================
# 4. CONFIGURATION LOADING (DRY Principle)
# ================================================================
def load_configs():
    """
    Load all YAML configuration files from the configs/ directory.
    Returns: (data_config, features_config, models_config)
    """
    config_files = {
        "data": PATHS["configs"] / "data.yaml",
        "features": PATHS["configs"] / "features.yaml",
        "models": PATHS["configs"] / "models.yaml"
    }

    loaded = {}
    
    for name, path in config_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required config missing: {path}")
        
        try:
            with open(path, "r") as f:
                loaded[name] = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid YAML in {path}: {e}")

    return loaded["data"], loaded["features"], loaded["models"]

# ================================================================
# 5. OSMNX SETTINGS
# ================================================================
# Set OSMnx to use our centralized cache directory
ox.settings.use_cache = True
ox.settings.cache_folder = PATHS["cache"] / "osmnx"
ox.settings.log_console = False
ox.settings.requests_timeout = 240
ox.settings.overpass_rate_limit = True

# Backup endpoints if the main Overpass API is down
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api",
    "https://overpass-api.de/api",
    "https://z.overpass-api.de/api",
]

def set_overpass_endpoint(idx):
    """Switch Overpass API endpoint to handle timeouts/failures."""
    ox.settings.overpass_endpoint = OVERPASS_ENDPOINTS[idx % len(OVERPASS_ENDPOINTS)]

def try_overpass(fn, *args, retries=3, **kwargs):
    """Execute OSMnx function with failover endpoints."""
    last_err = None
    for attempt in range(retries):
        set_overpass_endpoint(attempt)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            logger.warning(f"Overpass attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(2 * (attempt + 1))
    raise last_err

# ================================================================
# 6. GEOSPATIAL HELPERS
# ================================================================
def get_h3_int_from_coords(lat, lon, resolution):
    """Convert Lat/Lon to H3 Index (Integer)."""
    return h3.latlng_to_cell(lat, lon, resolution)

def get_boundary(data_config, features_config):
    """
    Fetch or load the country boundary (Admin 0).
    
    1. Checks 'data/raw/wbgCAF.geojson'.
    2. If missing, downloads from the URL in data.yaml.
    3. Filters by ISO3 code.
    4. Projects to Geodetic CRS (EPSG:4326).
    """
    iso3 = data_config["world_bank_boundary"]["iso3"]
    filter_prop = data_config["world_bank_boundary"]["filter_property"]
    geodetic_crs = features_config["geodetic_crs"]
    
    # Use pathlib path from centralized dict
    local_path = PATHS["data_raw"] / "wbgCAF.geojson"
    
    # Download if missing
    if not local_path.exists():
        url = data_config["world_bank_boundary"]["url"]
        logger.info("Downloading boundary from World Bank API...")
        try:
            boundary_gdf = gpd.read_file(url)
            boundary_gdf.to_file(local_path, driver="GeoJSON")
            logger.info(f"Boundary cached to {local_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download boundary: {e}")
    
    # Load and Filter
    boundary_gdf = gpd.read_file(local_path)
    
    # Filter for specific country (e.g., "CAF")
    if filter_prop in boundary_gdf.columns:
        car_boundary = boundary_gdf[boundary_gdf[filter_prop] == iso3].copy()
    else:
        # Fallback if filter property is wrong (or single-country file)
        logger.warning(f"Filter property '{filter_prop}' not found. Returning full file.")
        car_boundary = boundary_gdf

    if car_boundary.empty:
        raise ValueError(f"ISO3 code '{iso3}' resulted in empty boundary.")
    
    return car_boundary.to_crs(geodetic_crs)