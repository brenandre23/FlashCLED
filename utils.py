"""
utils.py
========
Core utilities for the CEWP pipeline.
Handles configuration loading, database connections, and path management.
"""
import os
import sys
import logging
import yaml
import time
import io as sys_io
import requests
import pandas as pd
import geopandas as gpd
from io import StringIO
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -----------------------------------------------------------------
# 1. CENTRALIZED PATHS
# -----------------------------------------------------------------
# Robustly resolve the project root relative to this file
ROOT_DIR = Path(__file__).resolve().parent

PATHS = {
    "root": ROOT_DIR,
    "pipeline": ROOT_DIR / "pipeline",
    "ingestion": ROOT_DIR / "pipeline" / "ingestion",
    "processing": ROOT_DIR / "pipeline" / "processing",
    "modeling": ROOT_DIR / "pipeline" / "modeling",
    "feature_engineering": ROOT_DIR / "pipeline" / "feature_engineering",
    "configs": ROOT_DIR / "configs",
    "data": ROOT_DIR / "data",
    "data_raw": ROOT_DIR / "data" / "raw",
    "data_proc": ROOT_DIR / "data" / "processed",
    "cache": ROOT_DIR / "cache",
    "logs": ROOT_DIR / "logs",
    "models": ROOT_DIR / "models",
}

# Ensure critical directories exist
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')),
        logging.FileHandler(PATHS["logs"] / "pipeline.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("CEWP")

# -----------------------------------------------------------------
# 3. CONFIGURATION LOADER
# -----------------------------------------------------------------
class ConfigBundle:
    """
    Backwards/forwards-compatible config container.

    Supports:
      - tuple unpack:   data, feats, models = load_configs()
      - dict access:    cfg["data"], cfg.get("models")
      - numeric access: cfg[0], cfg[1], cfg[2]   (some scripts do this)
    """
    __slots__ = ("data", "features", "models")

    def __init__(self, data, features, models):
        self.data = data or {}
        self.features = features or {}
        self.models = models or {}

    def __iter__(self):
        yield self.data
        yield self.features
        yield self.models

    def __len__(self):
        return 3

    def __getitem__(self, k):
        # numeric indexing support
        if isinstance(k, int):
            if k == 0:
                return self.data
            if k == 1:
                return self.features
            if k == 2:
                return self.models
            raise IndexError(k)

        # dict-style keys support
        if k == "data":
            return self.data
        if k == "features":
            return self.features
        if k == "models":
            return self.models
        raise KeyError(k)

    def get(self, k, default=None):
        try:
            return self[k]
        except (KeyError, IndexError):
            return default


def load_configs():
    """
    Loads the three core YAML configuration files.

    Returns:
      ConfigBundle, which supports tuple unpacking + dict-style access + [0]/[1]/[2] indexing.
    """
    files = ["data.yaml", "features.yaml", "models.yaml"]
    loaded = []

    for filename in files:
        path = PATHS["configs"] / filename
        if not path.exists():
            logger.critical(f"Configuration file missing: {path}")
            raise FileNotFoundError(f"Missing {filename}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            loaded.append(cfg if cfg is not None else {})

    if len(loaded) != 3:
        raise RuntimeError(f"Expected 3 config files, got {len(loaded)}")

    return ConfigBundle(loaded[0], loaded[1], loaded[2])

def get_global_window(data_config):
    """
    Helper to extract start/end dates from data.yaml as datetime objects.
    """
    fmt = "%Y-%m-%d"
    try:
        win = data_config["global_date_window"]
        start = datetime.strptime(win["start_date"], fmt).date()
        end = datetime.strptime(win["end_date"], fmt).date()
        return start, end
    except KeyError as e:
        logger.error(f"Missing date key in data.yaml: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid date format in data.yaml (use YYYY-MM-DD): {e}")
        raise

# -----------------------------------------------------------------
# 4. SECRETS & DATABASE (With Pooling and Debugging)
# -----------------------------------------------------------------
def get_secrets():
    """Loads and returns all environment variables."""
    # Centralized loading ensures the same settings for every script
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)
    return os.environ

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine with connection pooling.
    Includes a critical check for authentication parameters.
    """
    env = get_secrets()
    
    try:
        # 1. Retrieve essential connection parameters
        db_user = env.get('DB_USER', 'postgres')
        db_pass = env.get('DB_PASS')
        db_host = env.get('DB_HOST', 'localhost')
        db_name = env.get('DB_NAME', 'car_cewp')
        # The default port is explicitly handled here
        db_port = env.get('DB_PORT', '5432') 
        
        if not db_pass:
             raise ValueError("DB_PASS not found in .env")

        # 2. Build URL and LOG for verification
        # Log the connection details *without* the password for security and debugging
        safe_db_url = f"postgresql://{db_user}:********@{db_host}:{db_port}/{db_name}"
        logger.info(f"DB Connection URL (Safe): {safe_db_url}")
        
        # 3. Full URL with password
        full_db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        
        # 4. Return Engine with Connection Pooling (retaining best practices)
        return create_engine(
            full_db_url,
            pool_size=10, 
            max_overflow=20, 
            pool_timeout=30, 
            pool_pre_ping=True
        )
        
    except KeyError as e:
        logger.critical(f"Missing essential DB environment variable: {e}. Check your .env file.")
        raise

# ================================================================
# 5. RESILIENCE (Retry Logic)
# ================================================================
# Decorator for API calls: 3 retries, exponential wait (1s, 2s, 4s...)
retry_request = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
)

@retry_request
def download_file_with_retry(url, local_path):
    """
    Robust downloader with retries and browser-like headers.
    Fixes 'hanging' downloads on servers that throttle Python scripts.
    """
    # Masquerade as a standard Chrome browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "*/*",
        "Connection": "keep-alive"
    }
    
    # Timeout tuple: (connect_timeout, read_timeout)
    # 10s to connect, 300s to read data (prevents hanging forever)
    with requests.get(url, stream=True, headers=headers, timeout=(10, 300)) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192 * 4): # Increased buffer size
                if chunk: 
                    f.write(chunk)
    return True

# ================================================================
# 6. GEOSPATIAL HELPERS
# ================================================================
def get_boundary(data_config, features_config):
    iso3 = data_config["world_bank_boundary"]["iso3"]
    filter_prop = data_config["world_bank_boundary"]["filter_property"]
    geodetic_crs = features_config["spatial"]["crs"]["geodetic"]
    
    local_path = PATHS["data_raw"] / "wbgCAF.geojson"
    
    if not local_path.exists():
        url = data_config["world_bank_boundary"]["url"]
        logger.info("Downloading boundary from World Bank API...")
        try:
            # Use new robust downloader
            download_file_with_retry(url, local_path)
            logger.info(f"Boundary cached to {local_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download boundary: {e}")
    
    boundary_gdf = gpd.read_file(local_path)
    
    if filter_prop in boundary_gdf.columns:
        car_boundary = boundary_gdf[boundary_gdf[filter_prop] == iso3].copy()
    else:
        car_boundary = boundary_gdf

    return car_boundary.to_crs(geodetic_crs)

# ================================================================
# 7. UPLOAD HELPER (Transaction Safe)
# ================================================================
def upload_to_postgis(engine, df, table_name, schema, primary_keys):
    if df.empty: return
        
    temp_table = f"temp_{table_name}_{int(time.time())}"
    
    # Use connection context to ensure cleanup
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        conn.execute(text(f"CREATE TEMPORARY TABLE {temp_table} (LIKE {schema}.{table_name} INCLUDING ALL) ON COMMIT DROP;"))
        
        raw_conn = conn.connection
        with raw_conn.cursor() as cur:
            s_buf = StringIO()
            df.to_csv(s_buf, index=False, header=False, sep='\t', na_rep='\\N')
            s_buf.seek(0)
            columns = ",".join([f'"{c}"' for c in df.columns])
            cur.copy_expert(f"COPY {temp_table} ({columns}) FROM STDIN", s_buf)

        cols = ', '.join([f'"{c}"' for c in df.columns])
        pk_cols = ', '.join([f'"{c}"' for c in primary_keys])
        update_set = ', '.join([f'"{c}" = EXCLUDED."{c}"' for c in df.columns if c not in primary_keys])
        
        if not update_set: 
            sql = f"INSERT INTO {schema}.{table_name} ({cols}) SELECT {cols} FROM {temp_table} ON CONFLICT ({pk_cols}) DO NOTHING"
        else:
            sql = f"INSERT INTO {schema}.{table_name} ({cols}) SELECT {cols} FROM {temp_table} ON CONFLICT ({pk_cols}) DO UPDATE SET {update_set}"
             
        conn.execute(text(sql))
        logger.info(f"Upserted {len(df)} rows to {schema}.{table_name}")

# ================================================================
# 8. H3 TYPE HELPER
# ================================================================
def cast_h3_index(engine, table_name: str, schema: str = "car_cewp", index: bool = True):
    sql_cast = f"""
        ALTER TABLE {schema}.{table_name}
        ALTER COLUMN h3_index TYPE h3index
        USING h3index(CAST(h3_index AS TEXT));
    """
    sql_index = f"CREATE INDEX IF NOT EXISTS {table_name}_h3idx ON {schema}.{table_name} USING GIST (h3_index);"
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text(sql_cast))
            if index: conn.execute(text(sql_index))
            trans.commit()
            logger.info(f"h3_index cast for {table_name}")
        except Exception as e:
            trans.rollback()
            # Log debug to avoid spamming warnings for already-cast tables
            logger.debug(f"Cast H3 skipped/failed: {e}")