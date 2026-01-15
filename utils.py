"""
utils.py
========
Core utilities for the CEWP pipeline.
Handles configuration loading, database connections, and path management.

UPDATES:
- Added `init_gee`: Robust, silent Google Earth Engine auth using Service Accounts.
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
import subprocess
from io import StringIO
import geoalchemy2
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from geoalchemy2 import Geometry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, List, Tuple, Optional

# --- NEW IMPORTS FOR GEE AUTH ---
import ee
from google.oauth2 import service_account

# -----------------------------------------------------------------
# 1. CENTRALIZED PATHS
# -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
SCHEMA = "car_cewp"

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
# 3. TYPE SAFETY
# -----------------------------------------------------------------
def ensure_h3_int64(h3_val):
    """
    Converts any H3 representation (Hex String, Unsigned Int) to Signed Int64.
    Compatible with PostgreSQL BIGINT.
    """
    try:
        if pd.isna(h3_val): return None
        
        if isinstance(h3_val, str):
            clean_hex = h3_val.lower().replace('0x', '').strip()
            val = int(clean_hex, 16)
        else:
            val = int(h3_val)
            
        if val > 0x7FFFFFFFFFFFFFFF:
            val = int(val - 0x10000000000000000)
            
        return val
    except (ValueError, TypeError):
        return None

# -----------------------------------------------------------------
# 3b. IMPUTATION & TYPE VALIDATION HELPERS
# -----------------------------------------------------------------

def apply_forward_fill(df, col, groupby_col="h3_index", limit=None, config=None, domain=None):
    """
    Applies forward fill per group. 
    Priority: Explicit limit > Domain-specific limit from config > Global default > Hardcoded 4.
    """
    if limit is None and config:
        # 1. Try to find domain-specific override in features.yaml
        if domain:
            limit = config.get("imputation", {}).get("domains", {}).get(domain, {}).get("limit")
        
        # 2. Fallback to global default in config
        if limit is None:
            limit = config.get("imputation", {}).get("defaults", {}).get("limit", 4)
            
    if limit is None:
        limit = 4
        
    return df.groupby(groupby_col)[col].ffill(limit=limit)


def validate_h3_types(engine):
    query = text(
        """
        SELECT table_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'car_cewp'
          AND column_name = 'h3_index'
          AND data_type NOT IN ('bigint', 'int8')
        """
    )
    with engine.connect() as conn:
        violations = pd.read_sql(query, conn)
    if not violations.empty:
        raise TypeError(
            "H3 type violations found:\n"
            f"{violations.to_string(index=False)}\n"
            "Run init_db.py or fix schemas before proceeding."
        )

# -----------------------------------------------------------------
# 4. CONFIGURATION LOADER
# -----------------------------------------------------------------
class ConfigBundle:
    __slots__ = ("data", "features", "models")
    def __init__(self, data, features, models):
        self.data, self.features, self.models = data or {}, features or {}, models or {}
    def __iter__(self):
        yield self.data; yield self.features; yield self.models
    def __len__(self): return 3
    def __getitem__(self, k):
        if isinstance(k, int): return [self.data, self.features, self.models][k]
        if k == "data": return self.data
        if k == "features": return self.features
        if k == "models": return self.models
        raise KeyError(k)
    def get(self, k, default=None):
        try: return self[k]
        except: return default

def load_configs() -> ConfigBundle:
    files = ["data.yaml", "features.yaml", "models.yaml"]
    loaded = []
    for filename in files:
        path = PATHS["configs"] / filename
        if not path.exists(): raise FileNotFoundError(f"Missing {filename}")
        with open(path, "r", encoding="utf-8") as f:
            loaded.append(yaml.safe_load(f) or {})
    return ConfigBundle(loaded[0], loaded[1], loaded[2])

# -----------------------------------------------------------------
# 5. DATABASE
# -----------------------------------------------------------------
def get_secrets() -> dict:
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)

    try:
        uname_release = os.uname().release.lower()
    except AttributeError:
        import platform
        uname_release = platform.uname().release.lower()

    if os.environ.get("DB_HOST") == "localhost" and "microsoft" in uname_release:
        try:
            cmd = "ip route show | awk '/default/ {print $3}'"
            gateway_ip = subprocess.check_output(cmd, shell=True).decode().strip()
            os.environ["DB_HOST"] = gateway_ip
            logger.info(f"WSL Detected: Auto-routed DB connection to Windows Host ({gateway_ip})")
        except Exception as e:
            logger.warning(f"Could not auto-resolve WSL host: {e}")

    return os.environ

def get_db_engine():
    env = get_secrets()
    try:
        url = f"postgresql://{env.get('DB_USER','postgres')}:{env['DB_PASS']}@{env.get('DB_HOST','localhost')}:{env.get('DB_PORT','5432')}/{env.get('DB_NAME','car_cewp')}"
        return create_engine(url, pool_size=10, max_overflow=20, pool_timeout=30, pool_pre_ping=True)
    except KeyError as e:
        logger.critical(f"Missing DB env var: {e}")
        raise

# -----------------------------------------------------------------
# 6. RESILIENCE
# -----------------------------------------------------------------
retry_request = retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
)

@retry_request
def download_file_with_retry(url: str, local_path: Path) -> bool:
    headers = {"User-Agent": "Mozilla/5.0", "Connection": "keep-alive"}
    with requests.get(url, stream=True, headers=headers, timeout=(10, 300)) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32768):
                if chunk: f.write(chunk)
    return True

# -----------------------------------------------------------------
# 7. GEOSPATIAL
# -----------------------------------------------------------------
def get_boundary(data_config: dict, features_config: dict):
    iso3 = data_config["world_bank_boundary"]["iso3"]
    filter_prop = data_config["world_bank_boundary"]["filter_property"]
    geodetic_crs = features_config["spatial"]["crs"]["geodetic"]
    
    local_path = PATHS["data_raw"] / "wbgCAF.geojson"
    
    if not local_path.exists():
        logger.info("Downloading boundary from World Bank API...")
        try:
            download_file_with_retry(data_config["world_bank_boundary"]["url"], local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download boundary: {e}")
    
    boundary_gdf = gpd.read_file(local_path)
    
    if filter_prop in boundary_gdf.columns:
        car_boundary = boundary_gdf[boundary_gdf[filter_prop] == iso3].copy()
    else:
        car_boundary = boundary_gdf

    return car_boundary.to_crs(geodetic_crs)

# -----------------------------------------------------------------
# 8. UPLOAD HELPER (UPSERT + CSV SAFETY)
# -----------------------------------------------------------------
def _infer_sql_type(dtype_str: str) -> str:
    """
    Lightweight dtype -> SQL type mapper for dynamic table creation.
    """
    dtype_str = str(dtype_str).lower()
    if 'int' in dtype_str:
        return "BIGINT"
    if 'float' in dtype_str or 'double' in dtype_str:
        return "DOUBLE PRECISION"
    if 'datetime' in dtype_str or 'timestamp' in dtype_str:
        return "TIMESTAMP"
    if 'date' in dtype_str:
        return "DATE"
    if 'bool' in dtype_str:
        return "BOOLEAN"
    return "TEXT"

def upload_to_postgis(engine, df: pd.DataFrame, table_name: str, schema: str, primary_keys: list):
    if df.empty: return
        
    temp_table = f"temp_{table_name}_{int(time.time())}"
    
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        
        # Ensure target table exists (needed for LIKE on temp table)
        insp = inspect(engine)
        if not insp.has_table(table_name, schema=schema):
            col_defs = []
            for col, dtype in df.dtypes.items():
                col_defs.append(f'"{col}" {_infer_sql_type(dtype)}')
            pk_clause = ""
            if primary_keys:
                pk_cols = ', '.join([f'"{c}"' for c in primary_keys])
                pk_clause = f", PRIMARY KEY ({pk_cols})"
            create_sql = f"CREATE TABLE IF NOT EXISTS {schema}.{table_name} ({', '.join(col_defs)}{pk_clause});"
            conn.execute(text(create_sql))
        
        conn.execute(text(f"CREATE TEMPORARY TABLE {temp_table} (LIKE {schema}.{table_name} INCLUDING ALL) ON COMMIT DROP;"))
        
        raw_conn = conn.connection
        with raw_conn.cursor() as cur:
            s_buf = StringIO()
            df.to_csv(s_buf, index=False, header=False, sep=',', na_rep='') 
            s_buf.seek(0)
            
            columns = ",".join([f'"{c}"' for c in df.columns])
            cur.copy_expert(f"COPY {temp_table} ({columns}) FROM STDIN WITH (FORMAT CSV, NULL '')", s_buf)

        cols = ', '.join([f'"{c}"' for c in df.columns])
        pk_cols = ', '.join([f'"{c}"' for c in primary_keys])
        
        update_set = ', '.join([f'"{c}" = EXCLUDED."{c}"' for c in df.columns if c not in primary_keys])
        
        if not update_set: 
            sql = f"INSERT INTO {schema}.{table_name} ({cols}) SELECT {cols} FROM {temp_table} ON CONFLICT ({pk_cols}) DO NOTHING"
        else:
            sql = f"INSERT INTO {schema}.{table_name} ({cols}) SELECT {cols} FROM {temp_table} ON CONFLICT ({pk_cols}) DO UPDATE SET {update_set}"
             
        conn.execute(text(sql))
        logger.info(f"Upserted {len(df)} rows to {schema}.{table_name}")

# -----------------------------------------------------------------
# 9. GOOGLE EARTH ENGINE AUTH (THE ELEGANT SOLUTION)
# -----------------------------------------------------------------
def init_gee(project_id: str):
    """
    Initializes Google Earth Engine using a Service Account JSON key if available.
    Falls back to standard auth if not.
    """
    # 1. Try Loading from Environment Variable (Best Practice)
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    # 2. Hardcoded fallback (For your specific WSL setup)
    if not key_path:
        # Check common location
        candidate = Path("/home/brenan/.config/google_key.json")
        if candidate.exists():
            key_path = str(candidate)

    if key_path and os.path.exists(key_path):
        try:
            credentials = service_account.Credentials.from_service_account_file(key_path)
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
            ee.Initialize(credentials=scoped_credentials, project=project_id)
            logger.info(f"âœ“ GEE Authenticated via Service Account: {key_path}")
            return
        except Exception as e:
            logger.warning(f"Service Account auth failed: {e}. Falling back to default.")

    # 3. Fallback to Standard Interactive Auth
    try:
        ee.Initialize(project=project_id)
        logger.info(f"âœ“ GEE Initialized (Standard Auth)")
    except Exception as e:
        logger.warning(f"GEE init failed ({e}); attempting Authenticate()...")
        ee.Authenticate()
        ee.Initialize(project=project_id)


# -----------------------------------------------------------------
# 10. PRE-FLIGHT VALIDATION (Unchanged)
# -----------------------------------------------------------------
PHASE_PREREQUISITES = {
    "static": {
        "extensions": ["postgis", "h3", "h3_postgis"],
        "tables": {}, 
        "description": "Static phase requires PostgreSQL extensions only."
    },
    "dynamic": {
        "extensions": ["postgis", "h3", "h3_postgis"],
        "tables": {
            "features_static": ["h3_index", "geometry"],
        },
        "description": "Dynamic phase requires static features table."
    },
    "feature_engineering": {
        "extensions": ["postgis", "h3", "h3_postgis"],
        "tables": {
            "features_static": ["h3_index", "geometry"],
            "acled_events": ["h3_index", "event_date", "fatalities"],
            "environmental_features": ["h3_index", "date"],
        },
        "description": "Feature engineering requires static and dynamic data."
    },
    "modeling": {
        "extensions": ["postgis", "h3", "h3_postgis"],
        "tables": {
            "temporal_features": ["h3_index", "date"],
            "features_static": ["h3_index"],
        },
        "description": "Modeling phase requires temporal_features table."
    }
}

class ValidationResult:
    def __init__(self, phase: str):
        self.phase = phase
        self.passed = True
        self.missing_extensions: List[str] = []
        self.missing_tables: List[str] = []
        self.missing_columns: Dict[str, List[str]] = {}
        self.warnings: List[str] = []
    
    def add_missing_extension(self, ext: str):
        self.passed = False
        self.missing_extensions.append(ext)
    
    def add_missing_table(self, table: str):
        self.passed = False
        self.missing_tables.append(table)
    
    def add_missing_columns(self, table: str, columns: List[str]):
        self.passed = False
        self.missing_columns[table] = columns
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def get_error_message(self) -> str:
        if self.passed: return f"âœ“ Phase '{self.phase}' validation passed."
        lines = [f"âŒ Phase '{self.phase}' validation FAILED:"]
        if self.missing_extensions:
            lines.append(f"  Missing PostgreSQL extensions: {', '.join(self.missing_extensions)}")
        if self.missing_tables:
            lines.append(f"  Missing tables: {', '.join(self.missing_tables)}")
        if self.missing_columns:
            for table, cols in self.missing_columns.items():
                lines.append(f"  Table '{table}' missing columns: {', '.join(cols)}")
        return "\n".join(lines)
    
    def __bool__(self): return self.passed

def _check_extensions(engine, required: List[str]) -> List[str]:
    missing = []
    with engine.connect() as conn:
        result = conn.execute(text("SELECT extname FROM pg_extension"))
        installed = {row[0] for row in result}
    for ext in required:
        if ext not in installed: missing.append(ext)
    return missing

def _check_table_exists(engine, table: str, schema: str = SCHEMA) -> bool:
    return inspect(engine).has_table(table, schema=schema)

def _check_columns_exist(engine, table: str, required_cols: List[str], schema: str = SCHEMA) -> List[str]:
    inspector = inspect(engine)
    if not inspector.has_table(table, schema=schema): return required_cols
    existing = {col['name'] for col in inspector.get_columns(table, schema=schema)}
    return [col for col in required_cols if col not in existing]

def validate_pipeline_prerequisites(engine, phase: str, schema: str = SCHEMA) -> ValidationResult:
    if phase not in PHASE_PREREQUISITES:
        raise ValueError(f"Unknown phase: {phase}")
    prereqs = PHASE_PREREQUISITES[phase]
    result = ValidationResult(phase)
    logger.info(f"Validating prerequisites for phase: {phase}")
    
    req_exts = prereqs.get("extensions", [])
    if req_exts:
        missing = _check_extensions(engine, req_exts)
        for ext in missing: result.add_missing_extension(ext)
    
    req_tables = prereqs.get("tables", {})
    for table, columns in req_tables.items():
        if not _check_table_exists(engine, table, schema):
            result.add_missing_table(table)
        else:
            missing = _check_columns_exist(engine, table, columns, schema)
            if missing: result.add_missing_columns(table, missing)
            
    if result.passed: logger.info(f"âœ“ Phase '{phase}' prerequisites validated successfully")
    else: logger.error(result.get_error_message())
    return result

def validate_all_phases(engine, schema: str = SCHEMA) -> Dict[str, ValidationResult]:
    results = {}
    logger.info("FULL PIPELINE VALIDATION")
    for phase in PHASE_PREREQUISITES.keys():
        results[phase] = validate_pipeline_prerequisites(engine, phase, schema)
    return results

