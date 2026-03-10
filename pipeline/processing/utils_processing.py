"""
utils_processing.py
====================
Shared utilities for CEWP processing pipeline.
Includes feature sanitization, imputation helpers, and NaN validation
with fully dynamic expected vs observed comparison based on database introspection.
"""

import sys
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import h3.api.basic_int as h3
from pathlib import Path
from sqlalchemy import text, inspect
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Set

# Add root to path for main utils import
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, upload_to_postgis, apply_forward_fill

# --- SHARED CONSTANTS ---
SCHEMA = "car_cewp"
OUTPUT_TABLE = "temporal_features"
PRIMARY_KEYS = ['h3_index', 'date']
CHUNK_SIZE = 50000

# Preferred source->table mapping (processed first, raw staging second)
SOURCE_TO_TABLE = {
    "ACLED": ["temporal_features", "acled_events"],
    "ACLED_Hybrid": ["features_acled_hybrid", "temporal_features"],
    "NLP_CrisisWatch": ["features_crisiswatch", "temporal_features"],
    "GDELT": ["temporal_features", "features_dynamic_daily"],
    "IODA": ["internet_outages", "temporal_features"],
    "IOM_DTM": ["iom_displacement_h3", "temporal_features"],
    "Food_Security": ["food_security", "temporal_features"],
    "Economy": ["economic_drivers", "temporal_features"],
    "YahooFinance": ["economic_drivers", "temporal_features"],
    "FEWS_NET": ["food_security", "temporal_features"],
    "WorldPop": ["population_h3", "temporal_features"],
    "DynamicWorld": ["landcover_features", "temporal_features"],
    "CHIRPS": ["environmental_features", "temporal_features"],
    "ERA5": ["environmental_features", "temporal_features"],
    "MODIS": ["environmental_features", "temporal_features"],
    "VIIRS": ["environmental_features", "temporal_features"],
    "JRC_Landsat": ["environmental_features", "temporal_features"],
    "Fusion": ["temporal_features"],
    "Temporal": ["temporal_features"],
}


# =============================================================================
# DYNAMIC COLUMN CLASSIFICATION
# =============================================================================
# These patterns are used to auto-classify columns when not found in features.yaml

# Patterns that indicate FLOW variables (zero-fill, no event = 0)
FLOW_PATTERNS = [
    r'^national_',
    r'^(fatalities|casualties|deaths)',
    r'_count$',
    r'^acled_count_',
    r'^gdelt_.*_count',
    r'_event_count$',
    r'^(protest|riot|battle|explosion)_',
    r'outage_score$',
    r'^mech_',  # Mechanism scores from ACLED hybrid
    r'^ioda_',
]

# Patterns that indicate SPARSE spatial coverage (market-based, survey-based)
SPARSE_PATTERNS = [
    r'^price_',
    r'_price$',
    r'food_price',
    r'displacement',
    r'^iom_',
    r'_recency_days$',
]

# Patterns that indicate computed/derived columns (expected 0% NaN)
# NOTE: Recency flags are *sparse* (undefined when the parent series is missing),
# so they are classified above in SPARSE_PATTERNS, not here.
COMPUTED_PATTERNS = [
    r'^target_',
    r'_available$',
    r'^is_',
    r'^has_',
    r'_flag$',
    r'^month_(sin|cos)$',
    r'^is_dry_season$',
    r'^epoch$',
    r'^cw_',
    r'^narrative_',
    r'^fusion_',
    r'_decay_',
    r'_spatial_lag',
    r'_lag\\d+$',
]


def _matches_any_pattern(col: str, patterns: List[str]) -> bool:
    """Check if column name matches any of the regex patterns."""
    for pattern in patterns:
        if re.search(pattern, col, re.IGNORECASE):
            return True
    return False


def classify_column(col: str, features_config: Optional[dict] = None) -> str:
    """
    Classify a column as 'flow', 'sparse', 'computed', or 'stock'.
    
    Priority:
    1. Check features.yaml registry for explicit domain classification
    2. Fall back to pattern matching on column name
    
    Args:
        col: Column name
        features_config: Optional features.yaml config dict
    
    Returns:
        One of: 'flow', 'sparse', 'computed', 'stock'
    """
    # Try to get classification from features.yaml
    if features_config:
        registry = features_config.get('registry', [])
        imputation_cfg = features_config.get('imputation', {}).get('domains', {})
        
        for item in registry:
            if item.get('output_col') == col or item.get('raw') == col:
                source = item.get('source', '')
                
                # Map source to domain
                source_to_domain = {
                    'ACLED': 'conflict',
                    'GDELT': 'news',
                    'IODA': 'outage',
                    'CHIRPS': 'environmental',
                    'ERA5': 'environmental',
                    'MODIS': 'environmental',
                    'VIIRS': 'environmental',
                    'DynamicWorld': 'environmental',
                    'JRC_Landsat': 'environmental',
                    'Economy': 'economic',
                    'YahooFinance': 'economic',
                    'Food_Security': 'economic',
                    'FEWS_NET': 'social',
                    'IOM_DTM': 'social',
                    'WorldPop': 'demographic',
                }
                
                domain = source_to_domain.get(source)
                if domain:
                    domain_cfg = imputation_cfg.get(domain, {})
                    domain_type = domain_cfg.get('type', 'stock')
                    if domain_type == 'flow':
                        return 'flow'
                break
    
    # Fall back to pattern matching
    if _matches_any_pattern(col, COMPUTED_PATTERNS):
        return 'computed'
    if _matches_any_pattern(col, FLOW_PATTERNS):
        return 'flow'
    if _matches_any_pattern(col, SPARSE_PATTERNS):
        return 'sparse'
    
    return 'stock'


# --- SHARED FUNCTIONS ---

def get_h3_centroids(h3_indices):
    """Vectorized conversion of H3 Index -> Lat/Lon Centroid."""
    return np.array([h3.cell_to_latlng(x) for x in h3_indices])


def parse_registry(features_config):
    """Parses features.yaml registry into categorized buckets."""
    registry = features_config.get('registry', [])
    specs = {
        'environmental': [],
        'conflict': [],
        'economic': [],
        'social': [],
        'demographic': [],
        'nlp': []
    }
    source_map = {
        'CHIRPS': 'environmental', 'ERA5': 'environmental', 
        'MODIS': 'environmental', 'VIIRS': 'environmental', 'JRC_Landsat': 'environmental',
        'DynamicWorld': 'environmental',
        'ACLED': 'conflict', 'GDELT': 'conflict',
        'NLP_ACLED': 'nlp', 'NLP_CrisisWatch': 'nlp',
        'YahooFinance': 'economic', 'Economy': 'economic',
        'FEWS_NET': 'social', 'Food_Security': 'social', 'IOM_DTM': 'social', 'EPR': 'social',
        'WorldPop': 'demographic'
    }
    for item in registry:
        source = item.get('source')
        category = source_map.get(source, 'other')
        if category in specs:
            specs[category].append(item)
    return specs


def _infer_sql_type(dtype_str):
    dtype_str = str(dtype_str).lower()
    if 'int64' in dtype_str or 'int32' in dtype_str:
        return 'BIGINT'
    elif 'float' in dtype_str:
        return 'DOUBLE PRECISION'
    elif 'datetime64' in dtype_str:
        return 'TIMESTAMP'
    elif 'date' in dtype_str:
        return 'DATE'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    else:
        return 'TEXT'


def ensure_output_table_schema(engine, spine_sample):
    """Creates the output table with correct schema, dropping if exists."""
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))


def create_table_if_not_exists(engine, spine_sample):
    """Safe table creation without dropping."""
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        col_defs = []
        for col in spine_sample.columns:
            sql_type = _infer_sql_type(spine_sample[col].dtype)
            col_defs.append(f'"{col}" {sql_type}')
        pk_clause = ', '.join([f'"{pk}"' for pk in PRIMARY_KEYS])
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{OUTPUT_TABLE} (
                {', '.join(col_defs)},
                PRIMARY KEY ({pk_clause})
            )
        """
        conn.execute(text(create_sql))
        
        # Add columns if they don't exist (Schema Evolution)
        for col in spine_sample.columns:
            sql_type = _infer_sql_type(spine_sample[col].dtype)
            conn.execute(text(f"""
                ALTER TABLE {SCHEMA}.{OUTPUT_TABLE} 
                ADD COLUMN IF NOT EXISTS "{col}" {sql_type}
            """))


def discover_schema_tables(engine, schema: str = SCHEMA) -> Dict[str, Dict]:
    """
    Discover all tables in the schema and their date columns + min dates.
    
    Returns:
        Dict mapping table_name -> {
            'date_column': str or None,
            'min_date': pd.Timestamp or None,
            'columns': Set[str]
        }
    """
    inspector = inspect(engine)
    tables_info = {}
    
    # Get all tables in schema
    try:
        table_names = inspector.get_table_names(schema=schema)
    except Exception as e:
        logger.warning(f"Could not list tables in schema {schema}: {e}")
        return {}
    
    with engine.connect() as conn:
        for table_name in table_names:
            try:
                # Get column info
                columns = inspector.get_columns(table_name, schema=schema)
                col_names = {col['name'] for col in columns}
                
                # Find date/temporal column
                date_col = None
                for candidate in ['date', 'event_date', 'timestamp', 'time', 'year']:
                    if candidate in col_names:
                        date_col = candidate
                        break
                
                # Query min date if we found a date column
                min_date = None
                if date_col:
                    try:
                        if date_col == 'year':
                            result = conn.execute(text(
                                f'SELECT MIN("{date_col}") FROM {schema}."{table_name}"'
                            ))
                            min_year = result.scalar()
                            if min_year is not None:
                                min_date = pd.Timestamp(f"{min_year}-01-01")
                        else:
                            result = conn.execute(text(
                                f'SELECT MIN("{date_col}") FROM {schema}."{table_name}"'
                            ))
                            val = result.scalar()
                            if val is not None:
                                min_date = pd.Timestamp(val)
                    except Exception as e:
                        logger.debug(f"Could not query MIN({date_col}) from {table_name}: {e}")
                
                tables_info[table_name] = {
                    'date_column': date_col,
                    'min_date': min_date,
                    'columns': col_names,
                }
                
            except Exception as e:
                logger.debug(f"Could not inspect table {table_name}: {e}")
                continue
    
    return tables_info


def find_column_source(
    col: str, 
    tables_info: Dict[str, Dict],
    features_config: Optional[dict] = None
) -> Tuple[Optional[str], Optional[pd.Timestamp], str]:
    """
    Find which source table a column likely comes from.
    
    IMPORTANT: Derived/computed columns should NOT be mapped to temporal_features
    (the sink table), as this makes NaN validation meaningless. Instead, they are:
    - Mapped to their parent source table if identifiable (e.g., cw_* -> features_crisiswatch)
    - Marked as "Computed" with no expected NaN% calculation
    
    Strategy:
    1. Check features.yaml registry for explicit source mapping
    2. Handle derived/computed columns specially (not mapped to sink)
    3. Search all tables for a column with matching or similar name
    4. Handle common transformations (_lag1, _shock, _anomaly, etc.)
    
    Args:
        col: Output column name
        tables_info: Dict from discover_schema_tables()
        features_config: Optional features.yaml config
    
    Returns:
        Tuple of (table_name, min_date, provenance_type) or (None, None, provenance_type)
        provenance_type: 'raw', 'derived', 'computed', or 'unknown'
    """
    # Helper to resolve registry source -> physical table using SOURCE_TO_TABLE preference
    def _resolve_source(src: str) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
        for table_name in SOURCE_TO_TABLE.get(src, []):
            # CRITICAL: Skip temporal_features as it's the sink, not a source
            if table_name == 'temporal_features':
                continue
            info = tables_info.get(table_name)
            if info and info.get("min_date") is not None:
                return table_name, info["min_date"]
        return (None, None)

    col_lower = col.lower()
    
    # ==========================================================================
    # EARLY EXIT: Identify computed/derived columns that should NOT be mapped
    # to a source table for NaN validation purposes
    # ==========================================================================
    
    # Temporal context features (always computed, always 0% NaN expected)
    if col in ('month_sin', 'month_cos', 'is_dry_season', 'epoch'):
        return (None, None, 'computed')
    
    # Availability flags (always computed)
    if col_lower.endswith('_available') or col_lower.endswith('_data_available'):
        return (None, None, 'computed')
    
    # Structural break flags (always computed)
    if col_lower.startswith('is_') and ('worldpop' in col_lower or 'viirs' in col_lower):
        return (None, None, 'computed')
    
    # Fusion features (CW × ACLED) - no single parent source
    if col_lower.startswith('fusion_'):
        return (None, None, 'computed')
    
    # NTL kinetic delta is a derived feature (computed from ntl_peak - ntl_mean)
    if col_lower == 'ntl_kinetic_delta':
        # Map to environmental_features since it's derived from VIIRS data
        if 'environmental_features' in tables_info:
            info = tables_info['environmental_features']
            if info.get('min_date') is not None:
                return ('environmental_features', info['min_date'], 'derived')
        return (None, None, 'derived')
    
    # Decay features - derived from parent, map to parent's source if possible
    if '_decay_' in col_lower:
        # Try to find the base column's source
        base_col = col_lower.split('_decay_')[0]
        for table_name, info in tables_info.items():
            if table_name == 'temporal_features':
                continue
            if base_col in [c.lower() for c in info.get('columns', set())]:
                return (table_name, info['min_date'], 'derived')
        return (None, None, 'derived')
    
    # Spatial lag features - derived from parent
    if '_spatial_lag' in col_lower:
        base_col = col_lower.replace('_spatial_lag', '').replace('_lag1', '')
        for table_name, info in tables_info.items():
            if table_name == 'temporal_features':
                continue
            if base_col in [c.lower() for c in info.get('columns', set())]:
                return (table_name, info['min_date'], 'derived')
        return (None, None, 'derived')
    
    # CrisisWatch-derived features - map to features_crisiswatch
    if col_lower.startswith('cw_') or col_lower.startswith('narrative_'):
        if 'features_crisiswatch' in tables_info:
            info = tables_info['features_crisiswatch']
            if info.get('min_date') is not None:
                provenance = 'derived' if ('_lag' in col_lower or '_delta' in col_lower or '_decay' in col_lower) else 'raw'
                return ('features_crisiswatch', info['min_date'], provenance)
        return (None, None, 'derived')
    
    # Pillar regime columns from CrisisWatch
    if col_lower.startswith('regime_'):
        if 'features_crisiswatch' in tables_info:
            info = tables_info['features_crisiswatch']
            if info.get('min_date') is not None:
                return ('features_crisiswatch', info['min_date'], 'raw')
        return (None, None, 'derived')
    
    # ==========================================================================
    # Strategy 1: Check features.yaml registry (authoritative mapping)
    # ==========================================================================
    registry_source = None
    if features_config:
        registry = features_config.get('registry', [])
        for item in registry:
            if item.get('output_col') == col or item.get('raw') == col:
                registry_source = item.get('source', '')
                break
    if registry_source:
        table_name, min_date = _resolve_source(registry_source)
        if table_name:
            return (table_name, min_date, 'raw')

    # ==========================================================================
    # Strategy 2: Strip common suffixes and search for base column
    # ==========================================================================
    base_col = col
    is_derived = False
    for suffix in ['_lag1', '_lag2', '_shock', '_anomaly', '_mean', '_max', '_sum', 
                   '_decay_30d', '_decay_90d', '_decay_14d', '_spatial_lag', '_recency_days']:
        if col.endswith(suffix):
            base_col = col[:-len(suffix)]
            is_derived = True
            break
    
    # Also try common prefixes
    for prefix in ['log_', 'ln_', 'log1p_']:
        if col.startswith(prefix):
            base_col = col[len(prefix):]
            is_derived = True
            break
    
    # Search tables for matching column (excluding sink table)
    for table_name, info in tables_info.items():
        if table_name == 'temporal_features':
            continue
        table_cols = info.get('columns', set())
        table_cols_lower = {c.lower() for c in table_cols}
        
        # Direct match
        if col.lower() in table_cols_lower or base_col.lower() in table_cols_lower:
            if info['min_date'] is not None:
                provenance = 'derived' if is_derived else 'raw'
                return (table_name, info['min_date'], provenance)

    # ==========================================================================
    # Strategy 3: Keyword-based table matching
    # ==========================================================================
    
    # GDELT columns
    if col_lower.startswith('gdelt_') or col_lower.startswith('national_'):
        if 'features_dynamic_daily' in tables_info:
            info = tables_info['features_dynamic_daily']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived or '_decay' in col_lower else 'raw'
                return ('features_dynamic_daily', info['min_date'], provenance)
        return (None, None, 'derived' if is_derived else 'unknown')
    
    # ACLED Hybrid mechanism columns
    if col_lower.startswith('mech_') or col_lower.startswith('acled_') and 'hybrid' in col_lower:
        if 'features_acled_hybrid' in tables_info:
            info = tables_info['features_acled_hybrid']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('features_acled_hybrid', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    # ACLED conflict columns
    if any(kw in col_lower for kw in ['fatalities', 'battle', 'protest', 'riot', 'acled_count']):
        if 'acled_events' in tables_info:
            info = tables_info['acled_events']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('acled_events', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    # IODA / outages
    if col_lower.startswith('ioda_') or 'outage' in col_lower:
        if 'internet_outages' in tables_info:
            info = tables_info['internet_outages']
            if info.get('min_date') is not None:
                return ('internet_outages', info['min_date'], 'raw')
        return (None, None, 'unknown')
    
    # Environmental features (including NTL/VIIRS)
    if any(kw in col_lower for kw in ['precip', 'temp', 'ndvi', 'ntl', 'nightlight', 'soil', 'water', 'chirps', 'era5', 'modis', 'viirs']):
        if 'environmental_features' in tables_info:
            info = tables_info['environmental_features']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('environmental_features', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    # Landcover features
    if any(kw in col_lower for kw in ['landcover', 'dw_']):
        if 'landcover_features' in tables_info:
            info = tables_info['landcover_features']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('landcover_features', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    # Food / price columns
    if any(kw in col_lower for kw in ['price_', 'food_']):
        if 'food_security' in tables_info:
            info = tables_info['food_security']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('food_security', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    # Economic columns
    if any(kw in col_lower for kw in ['gold_', 'oil_', 'sp500', 'eur_usd', 'econ']):
        if 'economic_drivers' in tables_info:
            info = tables_info['economic_drivers']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('economic_drivers', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    # IOM displacement
    if any(kw in col_lower for kw in ['iom_', 'displacement']):
        if 'iom_displacement_h3' in tables_info:
            info = tables_info['iom_displacement_h3']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('iom_displacement_h3', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    # Population
    if 'pop' in col_lower:
        if 'population_h3' in tables_info:
            info = tables_info['population_h3']
            if info.get('min_date') is not None:
                provenance = 'derived' if is_derived else 'raw'
                return ('population_h3', info['min_date'], provenance)
        return (None, None, 'unknown')
    
    return (None, None, 'unknown')


def get_available_from_date(col: str, features_config: Optional[dict]) -> Optional[str]:
    """
    Look up the available_from date for a column from features.yaml.

    Args:
        col: Column name (output_col)
        features_config: Parsed features.yaml config dict

    Returns:
        available_from date string (e.g., "2012-01-28") or None
    """
    if not features_config:
        return None

    # Check 'registry' (Standard CEWP structure)
    registry = features_config.get('registry', [])
    for item in registry:
        if item.get('output_col') == col:
            return item.get('available_from')

    # Fallback: check feature_groups section
    feature_groups = features_config.get('feature_groups', [])
    for group in feature_groups:
        features = group.get('features', [])
        for feature in features:
            if feature.get('output_col') == col:
                return feature.get('available_from')

    return None


def compute_expected_nan_percentage(
    col: str,
    spine_start: pd.Timestamp,
    spine_end: pd.Timestamp,
    step_days: int,
    source_min_date: Optional[pd.Timestamp],
    col_type: str,
    available_from: Optional[str] = None
) -> Tuple[Optional[float], str]:
    """
    Compute expected NaN percentage for a column.

    Args:
        col: Column name
        spine_start: Start date of the temporal spine
        spine_end: End date of the temporal spine
        step_days: Temporal step size in days
        source_min_date: MIN(date) from the source table, or None
        col_type: One of 'flow', 'sparse', 'computed', 'stock'
        available_from: Optional per-column availability date from features.yaml

    Returns:
        Tuple of (expected_nan_pct, reason)
    """
    # Computed columns should always be 0% NaN
    if col_type == 'computed':
        return (0.0, "Computed/derived column")

    # Prefer per-column available_from date over table-level MIN(date)
    if available_from:
        try:
            availability_date = pd.Timestamp(available_from)
        except:
            logger.warning(f"Invalid available_from date '{available_from}' for {col}, falling back to table MIN(date)")
            availability_date = source_min_date
    else:
        availability_date = source_min_date

    # If no availability date found, we can't compute expectation
    if availability_date is None:
        return (None, "Source table not found")

    # Total number of steps in spine
    total_days = (spine_end - spine_start).days
    total_steps = max(1, total_days // step_days)

    # Calculate expected NaN% based on availability date
    if availability_date <= spine_start:
        return (0.0, f"Available from {availability_date.date()}")

    if availability_date >= spine_end:
        return (100.0, f"Starts {availability_date.date()} (after spine)")

    # Partial availability
    unavailable_days = (availability_date - spine_start).days
    unavailable_steps = unavailable_days // step_days
    expected_nan_pct = (unavailable_steps / total_steps) * 100

    return (expected_nan_pct, f"Available from {availability_date.date()}")


def sanitize_numeric_columns(
    df: pd.DataFrame,
    spine_start: Optional[pd.Timestamp] = None,
    spine_end: Optional[pd.Timestamp] = None,
    step_days: int = 14,
    tolerance_pct: float = 5.0,
    engine=None,
    features_config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Fix corrupted list-string floats and validate NaN percentages against expectations.
    
    Fully dynamic: discovers source tables, maps columns to tables, and computes
    expected NaN% based on actual MIN(date) in each source table.
    
    Args:
        df: DataFrame to sanitize
        spine_start: Start date of temporal spine (auto-detected if None)
        spine_end: End date of temporal spine (auto-detected if None)
        step_days: Temporal step in days
        tolerance_pct: Acceptable deviation from expected NaN% before flagging
        engine: SQLAlchemy engine (auto-created if None)
        features_config: Optional features.yaml config for better column classification
    
    Returns:
        Sanitized DataFrame
    """
    logger.info("Sanitizing numeric columns...")
    
    # Auto-detect date range if not provided
    if spine_start is None and 'date' in df.columns:
        spine_start = pd.Timestamp(df['date'].min())
    if spine_end is None and 'date' in df.columns:
        spine_end = pd.Timestamp(df['date'].max())
    
    # Step 1: Fix corrupted list-string floats
    converted_count = 0
    for col in df.columns:
        if df[col].dtype != 'object':
            continue
        if col in ('h3_index', 'admin1', 'admin2', 'admin3', 'nearest_market', 'market'):
            continue
        try:
            sample = df[col].dropna().head(10)
            if sample.empty:
                continue
            cleaned = df[col].astype(str).str.replace(r'[\[\]\s]', '', regex=True)
            cleaned = cleaned.replace('', np.nan)
            numeric_col = pd.to_numeric(cleaned, errors='coerce')
            valid_count = numeric_col.notna().sum()
            original_non_null = df[col].notna().sum()
            if valid_count > 0 and valid_count >= original_non_null * 0.5:
                df[col] = numeric_col
                converted_count += 1
        except Exception:
            continue
    
    if converted_count > 0:
        logger.info(f"  Converted {converted_count} object columns to numeric")
    
    # Step 2: Discover schema and compute expected NaN%
    if spine_start is not None and spine_end is not None:
        if engine is None:
            engine = get_db_engine()
        
        logger.info("  Discovering source tables and min dates...")
        tables_info = discover_schema_tables(engine)
        
        # Log discovered tables
        tables_with_dates = {k: v for k, v in tables_info.items() if v['min_date'] is not None}
        if tables_with_dates:
            logger.info(f"  Found {len(tables_with_dates)} tables with temporal data:")
            for table, info in sorted(tables_with_dates.items()):
                logger.info(f"    {table}: min_date={info['min_date'].date()}, cols={len(info['columns'])}")
        
        # Step 3: Validate each numeric column
        logger.info("")
        logger.info("=" * 110)
        logger.info("NaN VALIDATION: Observed vs Expected (dynamically computed from source tables)")
        logger.info("=" * 110)
        logger.info(f"Spine: {spine_start.date()} to {spine_end.date()} ({step_days}-day steps)")
        logger.info("")
        
        # Get numeric columns
        exclude_cols = {'h3_index', 'date', 'year'}
        numeric_cols = sorted([
            c for c in df.columns 
            if c not in exclude_cols 
            and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8', 'uint8']
        ])
        
        # Track results
        anomalies = []
        unmapped = []
        
        header = f"{'Column':<40} {'Observed':>10} {'Expected':>10} {'Delta':>10} {'Type':<10} {'Status':<12} {'Source'}"
        logger.info(header)
        logger.info("-" * 110)
        
        for col in numeric_cols:
            observed_nan_pct = df[col].isna().mean() * 100
            
            # Classify column type
            col_type = classify_column(col, features_config)
            
            # Find source table and min date (now returns provenance type as 3rd element)
            source_table, source_min_date, provenance = find_column_source(col, tables_info, features_config)
            
            # Override col_type if provenance indicates computed/derived
            if provenance == 'computed':
                col_type = 'computed'

            # Look up per-column availability date from features.yaml
            available_from = get_available_from_date(col, features_config)

            # Compute expected NaN% (uses available_from if provided, else falls back to table MIN(date))
            expected_nan_pct, reason = compute_expected_nan_percentage(
                col, spine_start, spine_end, step_days, source_min_date, col_type, available_from
            )
            
            # Determine status
            if source_table is None and provenance in ('computed', 'derived'):
                # Computed/derived columns with no source: expected 0% NaN
                if col_type == 'computed' or provenance == 'computed':
                    if observed_nan_pct > tolerance_pct:
                        status = "⚠️ COMPUTED"
                        anomalies.append((col, observed_nan_pct, 0.0, "Computed column has NaN", 'computed'))
                    else:
                        status = "✓ computed"
                    delta_str = f"{observed_nan_pct:+9.3f}%"
                    expected_str = "     0.000%"
                else:  # derived
                    status = "✓ derived"
                    delta_str = "--"
                    expected_str = "derived"
                source_str = f"Computed ({provenance})"
            elif expected_nan_pct is None:
                status = "UNMAPPED"
                unmapped.append(col)
                delta_str = "--"
                expected_str = "?"
                source_str = reason
            else:
                delta = observed_nan_pct - expected_nan_pct
                delta_str = f"{delta:+9.3f}%"
                expected_str = f"{expected_nan_pct:9.3f}%"
                
                if col_type == 'computed':
                    if observed_nan_pct > tolerance_pct:
                        status = "⚠️ COMPUTED"
                        anomalies.append((col, observed_nan_pct, 0.0, "Computed column has NaN", col_type))
                    else:
                        status = "✓ computed"
                elif col_type == 'flow':
                    if observed_nan_pct > tolerance_pct:
                        status = "⚠️ FLOW"
                        anomalies.append((col, observed_nan_pct, 0.0, "Flow variable has NaN", col_type))
                    else:
                        status = "✓ flow"
                elif col_type == 'sparse':
                    # Sparse columns: higher NaN expected, don't flag
                    status = "✓ sparse"
                else:  # stock
                    if abs(delta) <= tolerance_pct:
                        status = "✓ OK"
                    elif delta > tolerance_pct:
                        status = "⚠️ HIGH"
                        anomalies.append((col, observed_nan_pct, expected_nan_pct, reason, col_type))
                    else:
                        status = "✓ better"
                
                # Format source info with provenance
                source_str = source_table if source_table else reason
                if provenance and provenance != 'raw':
                    source_str = f"{source_str} ({provenance})"
            
            if len(source_str) > 35:
                source_str = source_str[:32] + "..."
            
            logger.info(
                f"{col:<40} {observed_nan_pct:>9.3f}% {expected_str:>10} {delta_str:>10} "
                f"{col_type:<10} {status:<12} {source_str}"
            )
        
        logger.info("-" * 110)
        
        # Summary
        if anomalies:
            logger.warning("")
            logger.warning(f"⚠️  {len(anomalies)} COLUMNS WITH UNEXPECTED NaN LEVELS:")
            for col, obs, exp, reason, ctype in anomalies:
                logger.warning(f"   [{ctype}] {col}: {obs:.3f}% observed vs {exp:.3f}% expected ({reason})")
            logger.warning("")
            logger.warning("Possible causes:")
            logger.warning("  - ETL failures or data ingestion gaps")
            logger.warning("  - Spatial coverage issues (not all H3 cells have data)")
            logger.warning("  - Publication lag shifting data outside query window")
            logger.warning("  - Forward-fill limit exhausted")
        
        if unmapped:
            logger.info("")
            logger.info(f"ℹ️  {len(unmapped)} columns could not be mapped to source tables:")
            logger.info(f"   {', '.join(unmapped[:10])}" + ("..." if len(unmapped) > 10 else ""))
        
    if not anomalies:
        logger.info("")
        logger.info("✓ All mapped columns within expected NaN tolerances")
    
    logger.info("")
    
    return df


def validate_scalar_integrity(df: pd.DataFrame, numeric_cols: List[str], warn_only: bool = False) -> None:
    """
    Ensure the provided numeric columns contain only scalar values (no lists/ndarrays).
    
    Args:
        df: DataFrame to inspect
        numeric_cols: Columns that are expected to be scalar numeric
        warn_only: If True, log warnings instead of raising; otherwise raise ValueError
    """
    offenders = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        mask = df[col].apply(lambda x: isinstance(x, (list, np.ndarray)))
        if mask.any():
            offenders[col] = int(mask.sum())
    if offenders:
        msg = f"Non-scalar values detected in columns: {offenders}"
        if warn_only:
            logger.warning(msg)
        else:
            raise ValueError(msg)


def validate_numeric_integrity(df: pd.DataFrame, feature_cols: list = None) -> None:
    """Check for object columns that look numeric (data corruption indicator)."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ('h3_index', 'date', 'admin1', 'admin2', 'admin3')]
    problematic = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(5).astype(str)
            if sample.str.match(r'^[\[\]\-\d\.eE\s]+$').any():
                problematic.append(col)
    if problematic:
        logger.warning(f"POTENTIAL DATA CORRUPTION: {len(problematic)} columns have object dtype but numeric-looking values.")


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on, how='left'):
    """Merge with consistent sort order."""
    df = left.merge(right, on=on, how=how)
    if 'h3_index' in df.columns and 'date' in df.columns:
        return df.sort_values(['h3_index', 'date']).reset_index(drop=True)
    return df.reset_index(drop=True)


def add_spatial_diffusion_features(df: pd.DataFrame, target_col: str, k: int = 1) -> pd.DataFrame:
    """Add spatial lag features using H3 k-ring neighbors."""
    if target_col not in df.columns:
        return df
    out_col = f"{target_col}_spatial_lag"
    if df.empty:
        df[out_col] = 0
        return df
    df[target_col] = df[target_col].fillna(0)
    df[out_col] = 0.0
    unique_h3 = df['h3_index'].astype('int64').unique()
    neighbor_map = {int(h): [int(n) for n in h3.grid_disk(int(h), k) if int(n) != int(h)] for h in unique_h3}
    for dt, sub in df.groupby('date'):
        values = dict(zip(sub['h3_index'].astype('int64'), sub[target_col]))
        sums = []
        for _, row in sub.iterrows():
            neighbors = neighbor_map.get(int(row['h3_index']), [])
            sums.append(sum(values.get(n, 0) for n in neighbors))
        df.loc[sub.index, out_col] = sums
    df[out_col] = df[out_col].fillna(0)
    return df


def apply_halflife_decay(df: pd.DataFrame, target_col: str, config_obj) -> pd.DataFrame:
    """Apply exponential decay with configurable half-life."""
    if target_col not in df.columns:
        return df
    temporal_cfg = config_obj.get('temporal', {}) if isinstance(config_obj, dict) else {}
    steps = temporal_cfg.get('decays', {}).get('half_life_30d', {}).get('steps', 2.14)
    alpha = 1 - np.exp(-np.log(2) / float(steps))
    out_col = f"{target_col}_decay_30d"
    df[target_col] = df[target_col].fillna(0)
    df[out_col] = df.groupby('h3_index')[target_col].apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean()).reset_index(level=0, drop=True)
    df[out_col] = df[out_col].fillna(0)
    return df


def process_food_prices_spatial(engine, h3_gdf, start_date, end_date):
    """
    Broadcasts market prices to H3 cells based on NEAREST market.
    """
    logger.info("Processing Food Security: Broadcasting nearest market prices...")

    # 1. Load Market Locations
    locs = pd.read_sql(
        "SELECT market_name AS market, latitude, longitude FROM car_cewp.market_locations",
        engine
    )
    if locs.empty:
        logger.warning("No market locations found! Food prices will be null.")
        return pd.DataFrame()

    # 2. Load Prices (Wide Format)
    prices = pd.read_sql(f"""
        SELECT date, market, commodity, value 
        FROM car_cewp.food_security 
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """, engine)
    
    if prices.empty:
        return pd.DataFrame()

    # Pivot: date, market -> price_maize, price_rice, etc.
    prices_wide = prices.pivot_table(
        index=['date', 'market'], 
        columns='commodity', 
        values='value'
    ).reset_index()
    prices_wide['date'] = pd.to_datetime(prices_wide['date'])
    
    # Simple column cleanup and map to standardized price_* names
    cleaned_cols = []
    for c in prices_wide.columns:
        if c in ['date', 'market']:
            cleaned_cols.append(c)
        else:
            cleaned = c.lower().replace(' ', '_').replace('(', '').replace(')', '')
            cleaned_cols.append(cleaned)
    prices_wide.columns = cleaned_cols

    def _map_price_col(col: str) -> str:
        if 'maize' in col:
            return 'price_maize'
        if 'rice' in col:
            return 'price_rice'
        if 'oil' in col:
            return 'price_oil'
        if 'sorghum' in col:
            return 'price_sorghum'
        if 'cassava' in col:
            return 'price_cassava'
        if 'groundnuts' in col:
            return 'price_groundnuts'
        return None

    rename_map = {c: _map_price_col(c) for c in prices_wide.columns if c not in ['date', 'market']}
    rename_map = {k: v for k, v in rename_map.items() if v}
    prices_wide = prices_wide.rename(columns=rename_map)

    # Collapse duplicate commodity columns by mean and keep only approved commodities
    keep_price_cols = ['price_maize', 'price_rice', 'price_oil', 'price_sorghum', 'price_cassava', 'price_groundnuts']
    collapsed = prices_wide[['date', 'market']].copy()
    for col in keep_price_cols:
        matching = [c for c in prices_wide.columns if c == col]
        if matching:
            collapsed[col] = prices_wide[matching].mean(axis=1)
    prices_wide = collapsed

    # 3. Spatial Join: Find nearest market for each H3 cell
    market_coords = locs[['latitude', 'longitude']].values
    tree = cKDTree(market_coords)

    # Get H3 Centroids
    h3_points = h3_gdf.centroid
    h3_coords = np.array(list(zip(h3_points.y, h3_points.x)))  # Lat, Lon

    # Query nearest market
    dists, idxs = tree.query(h3_coords, k=1)
    
    # Map H3 Index -> Nearest Market Name
    h3_to_market = pd.DataFrame({
        'h3_index': h3_gdf['h3_index'].values,
        'nearest_market': locs.iloc[idxs]['market'].values,
        'dist_to_market_km': dists * 111  # Approx deg->km
    })

    # 4. Merge Prices onto Grid
    merged = pd.merge(h3_to_market, prices_wide, left_on='nearest_market', right_on='market', how='left')
    merged['date'] = pd.to_datetime(merged['date'])
    merged['h3_index'] = merged['h3_index'].astype('int64')

    # Keep only the fields we want to broadcast (price columns) plus keys
    price_cols = ['price_maize', 'price_rice', 'price_oil', 'price_sorghum', 'price_cassava', 'price_groundnuts']
    keep_cols = ['h3_index', 'date'] + [c for c in price_cols if c in merged.columns]
    return merged[keep_cols]
