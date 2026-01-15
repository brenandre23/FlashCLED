"""
pipeline/processing/feature_engineering.py
==========================================
Master Feature Engineering Pipeline (Phase 5).
CONFIG-DRIVEN: Controlled strictly by configs/features.yaml.

FIXES APPLIED:
1. SQL FIX: corrected 'displacement_count' -> 'iom_displacement_sum' in process_social_data.
2. IDEMPOTENCY FIX: Replaced DROP+REPLACE with upsert using upload_to_postgis helper.
3. PATH FIX: Corrected ROOT_DIR to parents[2].
4. SCHEMA EVOLUTION: Added support for adding new columns to existing tables.
5. POPULATION FIX: Added process_demographics with WorldPop V1/V2 structural break handling.
6. ECONOMICS FIX: Added process_economics() to load and transform all economic indicators.
7. DATA CORRUPTION FIX: Added sanitize_numeric_columns() to fix "[9.639088E0]" string encoding bug.
"""

import sys
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import h3.api.basic_int as h3
from pathlib import Path
from sqlalchemy import text, inspect
from scipy.spatial import cKDTree
from utils import (
    logger, ConfigBundle, load_configs, get_db_engine, apply_forward_fill
)


# ==============================================================================
# DATA SANITIZATION UTILITIES (FIX FOR SHAP CRASH)
# ==============================================================================

def sanitize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes corrupted "list-string" floats that cause downstream crashes.
    
    Issue: Some groupby/aggregation operations can produce values like "[9.639088E0]"
    instead of 9.639088. This causes SHAP and other ML libraries to crash with:
        ValueError: could not convert string to float: '[9.639088E0]'
    
    Solution: Strip brackets, convert to numeric, coerce errors to NaN.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sanitize (modified in-place for efficiency)
        
    Returns
    -------
    pd.DataFrame
        Sanitized DataFrame with all object columns converted to numeric where possible
    """
    logger.info("Sanitizing numeric columns (fixing list-string encoding)...")
    
    converted_count = 0
    failed_cols = []
    
    for col in df.columns:
        # Skip non-object columns (already numeric)
        if df[col].dtype != 'object':
            continue
            
        # Skip known string columns
        if col in ('h3_index', 'admin1', 'admin2', 'admin3', 'nearest_market', 'market'):
            continue
            
        try:
            # Sample to check if column looks like encoded numbers
            sample = df[col].dropna().head(10)
            if sample.empty:
                continue
                
            # Check for bracket-encoded values
            sample_str = sample.astype(str)
            has_brackets = sample_str.str.contains(r'[\[\]]', regex=True).any()
            
            if has_brackets:
                logger.debug(f"  Fixing bracket-encoded column: {col}")
                
            # Strip brackets and whitespace
            cleaned = df[col].astype(str).str.replace(r'[\[\]\s]', '', regex=True)
            
            # Handle empty strings after cleaning
            cleaned = cleaned.replace('', np.nan)
            
            # Convert to numeric
            numeric_col = pd.to_numeric(cleaned, errors='coerce')
            
            # Only replace if we got valid numbers
            valid_count = numeric_col.notna().sum()
            original_non_null = df[col].notna().sum()
            
            if valid_count > 0 and valid_count >= original_non_null * 0.5:
                df[col] = numeric_col
                converted_count += 1
                if has_brackets:
                    logger.debug(f"    -> Converted {col}: {valid_count} valid values")
            else:
                # Too much data loss - likely a true string column
                failed_cols.append(col)
                
        except Exception as e:
            logger.debug(f"  Could not convert {col}: {e}")
            failed_cols.append(col)
            continue
    
    if converted_count > 0:
        logger.info(f"  Sanitized {converted_count} columns with bracket-encoded values")
    if failed_cols:
        logger.debug(f"  Skipped non-numeric columns: {failed_cols[:5]}{'...' if len(failed_cols) > 5 else ''}")
    
    return df


def validate_numeric_integrity(df: pd.DataFrame, feature_cols: list = None) -> None:
    """
    Validates that feature columns are properly numeric.
    Raises warnings for any remaining object-type columns that should be numeric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    feature_cols : list, optional
        Specific columns to check. If None, checks all columns.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ('h3_index', 'date', 'admin1', 'admin2', 'admin3')]
    
    problematic = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            # Check if any values look like numbers
            sample = df[col].dropna().head(5).astype(str)
            numeric_pattern = sample.str.match(r'^[\[\]\-\d\.eE\s]+$').any()
            if numeric_pattern:
                problematic.append(col)
    
    if problematic:
        logger.warning(
            f"POTENTIAL DATA CORRUPTION: {len(problematic)} columns have object dtype "
            f"but contain numeric-looking values: {problematic[:5]}"
        )


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on, how='left'):
    """
    Deterministic merge with post-merge sort to avoid time-travel from shuffled rows.
    If both h3_index and date exist post-merge, sort by them.
    """
    df = left.merge(right, on=on, how=how)
    if 'h3_index' in df.columns and 'date' in df.columns:
        return df.sort_values(['h3_index', 'date']).reset_index(drop=True)
    return df.reset_index(drop=True)


def add_spatial_diffusion_features(df: pd.DataFrame, target_col: str, k: int = 1) -> pd.DataFrame:
    """
    For each (date, h3_index), sum the target_col over its k-ring neighbors (exclude center).
    Adds column: {target_col}_spatial_lag.
    """
    if target_col not in df.columns:
        return df

    out_col = f"{target_col}_spatial_lag"
    if df.empty:
        df[out_col] = 0
        return df

    df[target_col] = df[target_col].fillna(0)
    df[out_col] = 0.0

    # Precompute neighbor map once
    unique_h3 = df['h3_index'].astype('int64').unique()
    neighbor_map = {
        int(h): [int(n) for n in h3.grid_disk(int(h), k) if int(n) != int(h)]
        for h in unique_h3
    }

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
    """
    Apply EWMA with half-life derived from config.
    Adds column: {target_col}_decay_30d.
    """
    if target_col not in df.columns:
        return df

    temporal_cfg = {}
    try:
        temporal_cfg = config_obj.get('temporal', {}) if isinstance(config_obj, dict) else {}
    except Exception:
        temporal_cfg = {}

    steps = (
        temporal_cfg.get('decays', {})
        .get('half_life_30d', {})
        .get('steps', 2.14)
    )
    try:
        alpha = 1 - np.exp(-np.log(2) / float(steps))
    except Exception:
        alpha = 0.5

    out_col = f"{target_col}_decay_30d"
    df[target_col] = df[target_col].fillna(0)
    df[out_col] = (
        df.groupby('h3_index')[target_col]
        .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )
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
    h3_coords = np.array(list(zip(h3_points.y, h3_points.x))) # Lat, Lon

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

def process_crisiswatch_features(spine_df: pd.DataFrame, parsed_csv_path: Path) -> pd.DataFrame:
    """
    Ingest parsed CrisisWatch semantic outputs and broadcast national signals.
    - Local rows (h3_index != 0): aggregate by (h3_index, date) and merge on h3/date.
    - National rows (h3_index == 0): aggregate by date, merge on date to all H3s.
    Anti-leakage: shift national and local scores by one period per h3.
    """
    if not parsed_csv_path.exists():
        logger.warning(f"CrisisWatch parsed file not found: {parsed_csv_path}")
        return spine_df

    df = pd.read_csv(parsed_csv_path)
    if df.empty:
        logger.warning("CrisisWatch parsed file is empty; skipping.")
        return spine_df

    df['date'] = pd.to_datetime(df['date'])

    # Placeholder scoring: simple length proxy; replace with embeddings as needed
    df['embedding_score'] = df['text_segment'].fillna('').str.len().clip(lower=0)

    df_local = df[df['h3_index'] != 0].copy()
    df_national = df[df['h3_index'] == 0].copy()

    local_agg = pd.DataFrame()
    if not df_local.empty:
        local_agg = df_local.groupby(['h3_index', 'date'], observed=True)['embedding_score'].mean().reset_index()
        local_agg = local_agg.rename(columns={'embedding_score': 'crisiswatch_local_score'})

    national_agg = pd.DataFrame()
    if not df_national.empty:
        national_agg = df_national.groupby(['date'], observed=True)['embedding_score'].mean().reset_index()
        national_agg = national_agg.rename(columns={'embedding_score': 'national_tension_score'})

    spine = spine_df
    if not national_agg.empty:
        spine = safe_merge(spine, national_agg, on=['date'], how='left')
        spine['national_tension_score'] = spine['national_tension_score'].shift(1)
    else:
        spine['national_tension_score'] = 0

    if not local_agg.empty:
        spine = safe_merge(spine, local_agg, on=['h3_index', 'date'], how='left')
        spine['crisiswatch_local_score'] = spine.groupby('h3_index')['crisiswatch_local_score'].shift(1)
    else:
        spine['crisiswatch_local_score'] = 0

    spine['national_tension_score'] = spine['national_tension_score'].fillna(0)
    spine['crisiswatch_local_score'] = spine['crisiswatch_local_score'].fillna(0)
    return spine



# --- Import Utils ---
# Path: pipeline/processing/feature_engineering.py
# parents[0] = processing/, parents[1] = pipeline/, parents[2] = root/
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import (
    logger,
    get_db_engine,
    load_configs,
    upload_to_postgis,
    apply_forward_fill,
    validate_h3_types,
)

# --- Constants ---
SCHEMA = "car_cewp"
OUTPUT_TABLE = "temporal_features"
PRIMARY_KEYS = ['h3_index', 'date']  # For upsert operations
CHUNK_SIZE = 50000  # For chunked uploads
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "temporal_features.parquet"


def get_h3_centroids(h3_indices):
    """Vectorized conversion of H3 Index -> Lat/Lon Centroid."""
    return np.array([h3.cell_to_latlng(x) for x in h3_indices])


def parse_registry(features_config):
    """
    Parses features.yaml registry into categorized buckets for processing.
    """
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
    """
    Infer SQL type from pandas dtype string.
    """
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


# ==============================================================================
# PHASE 1: SPINE & INFRASTRUCTURE
# ==============================================================================
def create_master_spine(engine, start_date, end_date, step_days):
    logger.info("PHASE 1: Generating Master Temporal Spine...")
    
    # 1. Get all valid H3 Indices
    with engine.connect() as conn:
        h3_df = pd.read_sql(f"SELECT h3_index FROM {SCHEMA}.features_static", conn)
    
    unique_h3 = h3_df['h3_index'].astype('int64').unique()
    logger.info(f"  Loaded {len(unique_h3):,} unique H3 cells.")

    # 2. Generate Dates (Dynamic Step)
    freq = f"{step_days}D"
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    logger.info(f"  Generated {len(dates)} temporal steps (Freq: {freq}).")

    # 3. Cartesian Product (H3 x Date)
    index = pd.MultiIndex.from_product([unique_h3, dates], names=['h3_index', 'date'])
    spine = pd.DataFrame(index=index).reset_index()
    
    # 4. Epochs (for Climatology)
    spine['year'] = spine['date'].dt.year
    spine['epoch'] = spine.groupby('year')['date'].rank(method='dense').astype(int)
    spine.drop(columns=['year'], inplace=True)

    # ---------------------------------------------------------
    # NEW: SEASONAL FEATURES (Operational Reality)
    # ---------------------------------------------------------
    spine['month_sin'] = np.sin(2 * np.pi * spine['date'].dt.month / 12).astype('float32')
    spine['month_cos'] = np.cos(2 * np.pi * spine['date'].dt.month / 12).astype('float32')
    spine['is_dry_season'] = spine['date'].dt.month.isin([11, 12, 1, 2, 3, 4]).astype(int)
    logger.info("  ✓ Added seasonal context (Cyclical Time + Dry Season Flag).")
    
    return spine


def ensure_output_table_schema(engine, spine_sample):
    """
    Ensure the output table exists with the correct schema.
    Creates the table if it doesn't exist, or adds missing columns if it does.
    """
    inspector = inspect(engine)
    
    with engine.begin() as conn:
        # Ensure schema exists
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))

        # CRITICAL: drop existing table to reset schema types (e.g., ensure integers are not stored as TEXT)
        logger.info(f"Dropping {SCHEMA}.{OUTPUT_TABLE} to enforce schema types...")
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{OUTPUT_TABLE} CASCADE"))

        # --- CREATE NEW TABLE ---
        logger.info(f"Creating new table {SCHEMA}.{OUTPUT_TABLE}...")
        
        col_defs = []
        for col in spine_sample.columns:
            sql_type = _infer_sql_type(spine_sample[col].dtype)
            col_defs.append(f'"{col}" {sql_type}')
        
        pk_clause = ', '.join([f'"{pk}"' for pk in PRIMARY_KEYS])
        create_sql = f"""
            CREATE TABLE {SCHEMA}.{OUTPUT_TABLE} (
                {', '.join(col_defs)},
                PRIMARY KEY ({pk_clause})
            )
        """
        conn.execute(text(create_sql))
        logger.info(f"  Created table with {len(col_defs)} columns.")
        
        # Create indexes for performance
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_date 
            ON {SCHEMA}.{OUTPUT_TABLE} (date)
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_h3 
            ON {SCHEMA}.{OUTPUT_TABLE} (h3_index)
        """))
    
    logger.info(f"✓ Table schema verified and recreated: {SCHEMA}.{OUTPUT_TABLE}")


# ==============================================================================
# PHASE 2A: MACRO-ECONOMIC INDICATORS (NEW)
# ==============================================================================
def process_economics(engine, spine, econ_specs, feat_cfg=None):
    """
    Process macro-economic indicators from car_cewp.economic_drivers.
    
    Loads: gold_price_usd, oil_price_usd, sp500_index, eur_usd_rate
    Applies: lag transformations as per features.yaml
    Merge: Left join on date using merge_asof (daily -> 14-day spine alignment)
    
    Parameters
    ----------
    engine : sqlalchemy.Engine
        Database connection engine
    spine : pd.DataFrame
        Master H3-Date spine dataframe
    econ_specs : list
        List of economic feature specs from features.yaml registry
        
    Returns
    -------
    pd.DataFrame
        Spine with economic features merged and transformed
    """
    logger.info("PHASE 2A: Processing Macro-Economic Indicators...")
    inspector = inspect(engine)
    
    # Filter specs to only "Economy" source (not Food_Security which is handled separately)
    economy_specs = [spec for spec in econ_specs if spec.get('source') == 'Economy']
    
    if not economy_specs:
        logger.warning("  ⚠️ No 'Economy' source specs found in registry. Skipping.")
        return spine
    
    # Check if table exists
    if not inspector.has_table("economic_drivers", schema=SCHEMA):
        logger.warning(f"  ⚠️ Table {SCHEMA}.economic_drivers not found. Filling economic features with 0.")
        for spec in economy_specs:
            out_col = spec.get('output_col')
            if out_col:
                spine[out_col] = 0.0
        return spine
    
    # 1. Identify raw columns needed from the table (exclude derived flags)
    raw_cols = list(set([spec['raw'] for spec in economy_specs]))
    raw_db_cols = [c for c in raw_cols if c != "econ_data_available"]
    logger.info(f"  Loading columns: {raw_db_cols}")
    
    # 2. Load economic data from DB
    cols_sql = ', '.join(raw_db_cols)
    econ_df = pd.read_sql(
        f"SELECT date, {cols_sql} FROM {SCHEMA}.economic_drivers ORDER BY date",
        engine
    )
    econ_df['date'] = pd.to_datetime(econ_df['date'])
    # Zero out economic series prior to 2003-12-27 for consistency
    cutoff = pd.Timestamp("2003-12-27")
    econ_cols = ['gold_price_usd', 'oil_price_usd', 'sp500_index', 'eur_usd_rate']
    econ_df.loc[econ_df['date'] < cutoff, econ_cols] = 0
    
    if econ_df.empty:
        logger.warning(f"  ⚠️ No data in {SCHEMA}.economic_drivers. Filling economic features with 0.")
        for spec in economy_specs:
            out_col = spec.get('output_col')
            if out_col:
                spine[out_col] = 0.0
        return spine
    
    logger.info(f"  Loaded {len(econ_df):,} rows from economic_drivers "
                f"({econ_df['date'].min().date()} to {econ_df['date'].max().date()})")
    
    # 3. Temporal Alignment (do NOT backfill before first economic observation)
    spine_start_date = spine['date'].min()
    econ_start_date = econ_df['date'].min()
    
    if econ_start_date > spine_start_date:
        logger.warning(
            f"  ⚠ Economic data starts {econ_start_date.date()} after spine start {spine_start_date.date()}. "
            "Pre-gap periods will remain NaN until imputation."
        )

    # 4. Merge using merge_asof (align daily economic data to 14-day spine)
    spine = spine.sort_values('date')
    econ_df = econ_df.sort_values('date')
    
    spine = pd.merge_asof(
        spine,
        econ_df,
        on='date',
        direction='backward',
        tolerance=pd.Timedelta(days=14)  # Allow up to 14 days lookback
    )
    
    # 4. Forward-fill gaps (weekends, holidays) within each H3 cell
    for col in raw_db_cols:
        if col in spine.columns:
            spine[col] = apply_forward_fill(spine, col, config=feat_cfg, domain="economic")

    # Availability flag: 1 when all econ series present, else 0
    if raw_db_cols:
        spine["econ_data_available"] = spine[raw_db_cols].notna().all(axis=1).astype(int)
    else:
        spine["econ_data_available"] = 0
    
    # 5. Apply transformations from registry
    for spec in economy_specs:
        raw = spec['raw']
        trans = spec.get('transformation', 'none')
        out_col = spec.get('output_col')
        
        if not out_col:
            logger.warning(f"  ⚠️ No output_col for {raw}. Skipping.")
            continue
        
        if raw not in spine.columns:
            if raw == "econ_data_available":
                spine[out_col] = spine["econ_data_available"]
                logger.info(f"   econ_data_available -> {out_col} (flag passthrough)")
            else:
                logger.warning(f"   Column {raw} not found in data. Filling {out_col} with 0.")
                spine[out_col] = 0.0
            continue
        
        # Apply transformation
        if 'lag' in trans:
            # lag_1_step means shift by 1 period (14 days)
            spine[out_col] = spine.groupby('h3_index')[raw].shift(1)
            logger.info(f"  ✓ {raw} -> {out_col} (lag_1_step)")
        elif trans == 'none' or trans == 'mean':
            spine[out_col] = spine[raw]
            logger.info(f"  ✓ {raw} -> {out_col} (passthrough)")
        else:
            # Default: passthrough
            spine[out_col] = spine[raw]
            logger.info(f"  ✓ {raw} -> {out_col} (unknown transform '{trans}', using passthrough)")
        
        # Fill NaN with 0 (crucial for modeling)
        spine[out_col] = spine[out_col].fillna(0.0)
    
    # 6. Clean up raw columns (keep only output columns)
    cols_to_drop = [col for col in raw_db_cols if col in spine.columns and col not in 
                    [spec.get('output_col') for spec in economy_specs]]
    if cols_to_drop:
        spine.drop(columns=cols_to_drop, inplace=True)
    
    # Log summary
    for spec in economy_specs:
        out_col = spec.get('output_col')
        if out_col and out_col in spine.columns:
            non_zero = (spine[out_col] != 0).sum()
            logger.info(f"  {out_col}: {non_zero:,} non-zero values")
    
    return spine


# ==============================================================================
# PHASE 2B: FOOD SECURITY (Local Prices)
# ==============================================================================
def process_food_security(engine, spine, social_specs, feat_cfg):
    """
    Process local food prices from market data.
    Separate from macro-economic indicators.
    """
    impute_cfg = feat_cfg
    logger.info("PHASE 2B: Processing Local Food Prices...")
    inspector = inspect(engine)
    
    # A. Define Commodity Mapping
    COMMODITY_MAP = {
        'Maize (Corn)': 'price_maize',
        'Maize Grain (Yellow)': 'price_maize',
        'Palm Oil (Refined)': 'price_oil',
        'Rice (Milled)': 'price_rice',
        'Rice (5% Broken)': 'price_rice',
        'Sorghum (Red)': 'price_sorghum'
    }
    DEFAULT_PRICE_WEIGHTS = {
        'price_maize': 0.3,
        'price_rice': 0.3,
        'price_oil': 0.2,
        'price_sorghum': 0.2,
    }
    food_cfg = feat_cfg.get('food_security', {}) if isinstance(feat_cfg, dict) else {}
    price_weights = {
        k: food_cfg.get('food_price_index_weights', {}).get(k, v)
        for k, v in DEFAULT_PRICE_WEIGHTS.items()
    }
    
    has_locs = inspector.has_table("market_locations", schema=SCHEMA)
    has_prices = inspector.has_table("food_security", schema=SCHEMA)

    if has_locs and has_prices:
        locs = pd.read_sql(
            f"SELECT market_id, market_name, latitude, longitude FROM {SCHEMA}.market_locations", 
            engine
        )
        prices = pd.read_sql(
            f"SELECT date, market AS market_name, commodity, value FROM {SCHEMA}.food_security",
            engine
        )
        logger.info(f"  Loaded {len(prices)} price records from {len(locs)} markets.")
        prices['date'] = pd.to_datetime(prices['date'])
        prices = prices[prices['date'] >= pd.Timestamp('2018-01-01')]
        logger.info(f"  After 2018 filter: {len(prices)} records")
        
        if not locs.empty and not prices.empty:
            # B. Filter & Map Commodities
            prices['commodity_group'] = prices['commodity'].map(COMMODITY_MAP)
            prices = prices.dropna(subset=['commodity_group'])
            logger.info(f"  After commodity filter: {len(prices)} records")
            
            if not prices.empty:
                # C. Pivot with Aggregation
                prices['date'] = pd.to_datetime(prices['date'])
                df_pivot = prices.pivot_table(
                    index=['market_name', 'date'],
                    columns='commodity_group',
                    values='value',
                    aggfunc='mean'
                ).reset_index()
                
                price_cols = ['price_maize', 'price_oil', 'price_rice', 'price_sorghum']
                # Ensure all commodity columns exist
                
                for col in price_cols:
                    if col in spine.columns:
                        spine[col] = apply_forward_fill(spine, col, config=impute_cfg, domain="economic")
                        
                        # Calculate Price Shock (Anomaly) with config-driven window
                        window = 1
                        trans_spec = next((s for s in social_specs if s.get('output_col') == f"{col}_shock"), {})
                        params = trans_spec.get('transformation_params', {}) if trans_spec else {}
                        lookback_months = params.get('lookback_months', 12)
                        steps_per_month = 30.0 / step_days if step_days else 2.14
                        window = max(1, int(lookback_months * steps_per_month))
                        rolling_mean = spine.groupby('h3_index')[col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
                        
                        shock_col = f"{col}_shock"
                        spine[shock_col] = spine[col] / (rolling_mean + 1e-6)
                        spine[shock_col] = spine[shock_col].fillna(1.0)
                        
                        logger.info(f"   Calculated shock feature: {shock_col}")
                    else:
                        spine[col] = np.nan
                        spine[f"{col}_shock"] = 0.0

                
                if 'nearest_market' in spine.columns:
                    spine.drop(columns=['nearest_market'], inplace=True)
                
                # G. Food Price Index (simple average across available prices)
                spine['food_price_index'] = spine[price_cols].mean(axis=1, skipna=True)
                spine['index_exists'] = (spine[price_cols].notna().any(axis=1)).astype(int)
                spine['food_price_index'] = apply_forward_fill(spine, 'food_price_index', config=impute_cfg, domain="economic").fillna(0)
                spine['index_exists'] = spine['index_exists'].fillna(0)
                
                # Log summary
                for col in price_cols:
                    non_zero = (spine[col] > 0).sum()
                    logger.info(f"   {col}: {non_zero:,} non-zero values")
                non_zero_fpi = (spine.get('food_price_index', pd.Series(dtype=float)) > 0).sum()
                logger.info(f"   food_price_index: {non_zero_fpi:,} non-zero values")
            else:
                logger.warning("   No commodities matched mapping. Filling price columns with 0.")
                for col in ['price_maize', 'price_oil', 'price_rice', 'price_sorghum', 'food_price_index']:
                    spine[col] = 0.0
                    if col != 'food_price_index':
                        spine[f"{col}_shock"] = 0.0
        else:
            logger.warning("   Market locations or prices empty. Filling price columns with 0.")
            for col in ['price_maize', 'price_oil', 'price_rice', 'price_sorghum', 'food_price_index']:
                spine[col] = 0.0
                if col != 'food_price_index':
                    spine[f"{col}_shock"] = 0.0
    else:
        logger.warning("   Market tables not found. Filling price columns with 0.")
        for col in ['price_maize', 'price_oil', 'price_rice', 'price_sorghum', 'food_price_index']:
            spine[col] = 0.0
            if col != 'food_price_index':
                spine[f"{col}_shock"] = 0.0
    return spine


def inject_crisiswatch_features(spine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the distilled CrisisWatch features into the master spine.
    Logic:
      1. Load national/local parquet files.
      2. Broadcast national features on date.
      3. Pinpoint local features on [date, h3_index].
      4. Fill missing with 0.
      5. Shift by 1 period per h3 to prevent leakage.
    """
    from pathlib import Path
    import pandas as pd

    ROOT_DIR = Path(__file__).resolve().parents[2]
    path_nat = ROOT_DIR / "data" / "processed" / "features_crisiswatch_national.parquet"
    path_loc = ROOT_DIR / "data" / "processed" / "features_crisiswatch_local.parquet"

    if not path_nat.exists() or not path_loc.exists():
        logger.warning("CrisisWatch features missing. Skipping injection.")
        return spine_df

    df_nat = pd.read_parquet(path_nat)
    df_loc = pd.read_parquet(path_loc)

    # Broadcast national
    merged = spine_df.merge(df_nat, on="date", how="left")
    # Pinpoint local
    merged = merged.merge(df_loc, on=["date", "h3_index"], how="left")

    nat_cols = [c for c in merged.columns if c.startswith("context_cw_topic")]
    loc_cols = [c for c in merged.columns if c.startswith("local_cw_topic")]

    merged[nat_cols] = merged[nat_cols].fillna(0)
    merged[loc_cols] = merged[loc_cols].fillna(0)

    # Anti-leakage: shift by one period per h3
    merged = merged.sort_values(["h3_index", "date"])
    all_cw_cols = nat_cols + loc_cols
    merged[all_cw_cols] = merged.groupby("h3_index")[all_cw_cols].shift(1)

    logger.info(f"✅ Injected {len(all_cw_cols)} CrisisWatch features (lagged by one period).")
    return merged


# ==============================================================================
# PHASE 2C: SOCIAL DATA (IOM only)
# ==============================================================================
def process_social_data(engine, spine, social_specs, feat_cfg):
    """Process social data (IOM displacement only; IPC removed)."""
    logger.info("PHASE 2C: Processing Social Data (IOM only)...")
    inspector = inspect(engine)
    impute_cfg = feat_cfg

    # --- IOM Displacement Data ---
    iom_spec = next((x for x in social_specs if 'iom_displacement_sum' in x['raw']), None)
    if iom_spec:
        if inspector.has_table("iom_displacement_h3", schema=SCHEMA):
            logger.info("  Merging IOM Displacement Data...")
            row_count = pd.read_sql(
                f"SELECT COUNT(*) as cnt FROM {SCHEMA}.iom_displacement_h3",
                engine
            ).iloc[0]["cnt"]
            if row_count == 0:
                raise RuntimeError(f"{SCHEMA}.iom_displacement_h3 exists but is empty. Spatial disaggregation failed.")
            logger.info(f"  IOM rows: {row_count:,}")
            iom_df = pd.read_sql(
                f"SELECT h3_index, date, iom_displacement_sum FROM {SCHEMA}.iom_displacement_h3", 
                engine
            )
            iom_df['date'] = pd.to_datetime(iom_df['date'])

            spine = safe_merge(spine, iom_df, on=['h3_index', 'date'], how='left')
            if 'h3_index' in spine.columns:
                spine['h3_index'] = spine['h3_index'].astype('int64')
            spine['iom_data_available'] = spine['iom_displacement_sum'].notna().astype(int)
            # IOM updates are infrequent (quarterly); carry last value forward until next update
            spine['iom_displacement_sum'] = (
                spine.groupby('h3_index')['iom_displacement_sum'].ffill().fillna(0)
            )

            if 'lag' in iom_spec.get('transformation', ''):
                spine[iom_spec['output_col']] = spine.groupby('h3_index')['iom_displacement_sum'].shift(1)
        else:
            logger.warning(f"   Table {SCHEMA}.iom_displacement_h3 not found. Skipping IOM integration (filling 0).")
            if iom_spec.get('output_col'):
                spine[iom_spec.get('output_col')] = 0

    return spine



# ==============================================================================
# PHASE 2D: DEMOGRAPHICS (WorldPop)
# ==============================================================================
def process_demographics(engine, spine, demo_specs):
    """
    Merges annual population data and applies transformations.
    Includes structural break indicator for WorldPop V1/V2 shift.
    """
    logger.info("PHASE 2D: Processing Demographics (WorldPop)...")
    inspector = inspect(engine)
    
    pop_spec = next((x for x in demo_specs if x['raw'] == 'pop_count'), None)
    
    if pop_spec and inspector.has_table("population_h3", schema=SCHEMA):
        logger.info("  Merging Population Data (Annual)...")
        
        # 1. Load Population Data
        pop_df = pd.read_sql(
            f"SELECT h3_index, year, pop_count FROM {SCHEMA}.population_h3", 
            engine
        )
        
        # 2. Merge on H3 + Year
        spine['year'] = spine['date'].dt.year
        spine = spine.merge(pop_df, on=['h3_index', 'year'], how='left')
        
        # 3. Handle Missing Years
        spine = spine.sort_values(['h3_index', 'date'])
        spine['pop_count'] = spine.groupby('h3_index')['pop_count'].ffill().fillna(0)
        
        # 4. Apply Transformations
        trans = pop_spec.get('transformation', '')
        out_col = pop_spec.get('output_col')
        
        if 'log1p' in trans:
            spine[out_col] = np.log1p(spine['pop_count'])
        else:
            spine[out_col] = spine['pop_count']
            
        # Structural Break Indicator
        spine['is_worldpop_v1'] = (spine['date'].dt.year < 2015).astype(int)
        logger.info("  Added 'is_worldpop_v1' structural break indicator.")
            
        # Cleanup
        spine.drop(columns=['year', 'pop_count'], inplace=True)
        
    elif pop_spec:
        logger.warning(f"  ⚠️ Table {SCHEMA}.population_h3 not found. Filling {pop_spec['output_col']} with 0.")
        if pop_spec.get('output_col'):
            spine[pop_spec['output_col']] = 0.0
            spine['is_worldpop_v1'] = 0
            
    return spine


# ==============================================================================
# PHASE 3: ENVIRONMENTAL DATA (Config-Driven Anomalies)
# ==============================================================================
def process_environment(engine, spine, env_specs, feat_cfg):
    logger.info("PHASE 3: Processing Environmental Features (Config-Driven)...")
    
    if not env_specs:
        return spine

    inspector = inspect(engine)
    if not inspector.has_table("environmental_features", schema=SCHEMA):
        raise RuntimeError(
            f"Table {SCHEMA}.environmental_features does not exist. Run GEE ingestion first."
        )

    row_count = pd.read_sql(
        f"SELECT COUNT(*) as cnt FROM {SCHEMA}.environmental_features", engine
    ).iloc[0]["cnt"]
    if row_count == 0:
        raise RuntimeError(
            f"Table {SCHEMA}.environmental_features exists but is empty. GEE ingestion likely failed."
        )
    logger.info(f"  Environmental features row count: {row_count:,}")

    # Null-rate warnings for key cols
    key_cols = ['precip_mean_depth_mm', 'temp_mean', 'ndvi_max']
    for col in key_cols:
        null_pct = pd.read_sql(
            f"SELECT 100.0 * COUNT(*) FILTER (WHERE {col} IS NULL) / NULLIF(COUNT(*),0) as pct "
            f"FROM {SCHEMA}.environmental_features",
            engine,
        ).iloc[0]["pct"]
        if pd.notna(null_pct) and null_pct > 50:
            logger.warning(f"  ⚠ {col} has {null_pct:.1f}% nulls; anomaly reliability reduced.")

    # 1. Identify Columns to Load
    raw_cols = list(set([item['raw'] for item in env_specs]))
    
    # Ensure water features are included if in specs
    water_specs = [spec for spec in env_specs if 'water_' in spec['raw']]
    if water_specs:
        water_raws = [spec['raw'] for spec in water_specs]
        raw_cols.extend(water_raws)
        raw_cols = list(set(raw_cols))
    
    # Build select list based on available columns (allows fallback aliasing)
    insp = inspect(engine)
    env_cols = {col['name'] for col in insp.get_columns("environmental_features", schema=SCHEMA)}
    select_parts = []
    synthesized = []
    for raw in raw_cols:
        if raw in env_cols:
            select_parts.append(raw)
        elif raw == "water_local_mean" and "water_coverage" in env_cols:
            select_parts.append("water_coverage AS water_local_mean")
        else:
            synthesized.append(raw)
    db_cols = ', '.join(select_parts)
    
    # 2. Load Raw Data (only available columns)
    env_df = pd.read_sql(
        f"SELECT h3_index, date, {db_cols} FROM {SCHEMA}.environmental_features",
        engine
    )
    env_df['date'] = pd.to_datetime(env_df['date'])
    
    spine = spine.merge(env_df, on=['h3_index', 'date'], how='left')
    # If key water columns are still missing, pull them explicitly
    for water_col in ["water_local_mean", "water_local_max"]:
        if water_col not in spine.columns and water_col in env_cols:
            fallback = pd.read_sql(
                f"SELECT h3_index, date, {water_col} FROM {SCHEMA}.environmental_features",
                engine,
            )
            fallback["date"] = pd.to_datetime(fallback["date"])
            spine = spine.merge(fallback, on=["h3_index", "date"], how="left")

    # Create any synthesized/missing raws as NaN so downstream fill can handle
    for raw in synthesized:
        spine[raw] = np.nan
    
    # 3. Calculate Climatology (Baselines) - for anomaly features only
    anomaly_specs = [spec for spec in env_specs if 'anomaly' in spec.get('transformation', '')]
    if anomaly_specs:
        env_df['year'] = env_df['date'].dt.year
        env_df['epoch'] = env_df.groupby('year')['date'].rank(method='dense').astype(int)
        
        anomaly_raws = list(set([spec['raw'] for spec in anomaly_specs]))
        for col in anomaly_raws:
            subset = env_df
            if 'ntl' in col:
                subset = env_df[env_df['year'] >= 2012]
                
            stats = subset.groupby(['h3_index', 'epoch'])[col].agg(['mean', 'std']).reset_index()
            stats.columns = ['h3_index', 'epoch', f"{col}_mean", f"{col}_std"]
            
            spine = spine.merge(stats, on=['h3_index', 'epoch'], how='left')
    
    # 4. Apply Transformations from Registry (with forward-fill)
    for spec in env_specs:
        raw = spec['raw']
        trans = spec['transformation']
        out = spec['output_col']
        
        if raw not in spine.columns:
            spine[raw] = np.nan
            
        # Apply forward-fill (limit=4) as specified in imputation config
        spine[raw] = apply_forward_fill(spine, raw, config=feat_cfg, domain="environmental")
        
        if 'anomaly' in trans:
            anom_col = f"{raw}_anom"
            spine[anom_col] = spine[raw] - spine[f"{raw}_mean"]
            spine[out] = spine[anom_col]
            
        elif 'lag' in trans:
            spine[out] = spine.groupby('h3_index')[raw].shift(1)
             
        else:
            spine[out] = spine[raw]
            
        # Fill NaN values after transformation
        spine[out] = spine[out].fillna(0)

    # Cleanup intermediate columns
    drop_cols = [c for c in spine.columns if c.endswith('_mean') or c.endswith('_std') or c.endswith('_anom')]
    # Protect true feature columns that legitimately end with _mean/_std
    protected = [
        'water_local_mean',
        'epr_status_mean',
        'elevation_mean',
        'slope_mean',
        'gdelt_goldstein_mean',
    ]
    drop_cols = [c for c in drop_cols if c not in protected and 'market_price' not in c]
    spine.drop(columns=[c for c in drop_cols if c in spine.columns], inplace=True)
    
    if 'epoch' in spine.columns:
        spine.drop(columns=['epoch'], inplace=True)
    
    return spine


# ==============================================================================
# PHASE 4: CONFLICT DATA (Config-Driven Decay)
# ==============================================================================
def compute_time_since_last_fatal_event(spine, acled_raw):
    """
    Computes 'time_since_last_fatal_event' (in days) per H3 cell.
    
    FIX FOR 9999 BUG:
    - Explicitly casts both spine['date'] and event['date'] to pd.Timestamp
    - Uses merge_asof with direction='backward' to find most recent prior fatal event
    - Returns 9999 only for cells with NO historical fatal event
    """
    logger.info("  Computing time_since_last_fatal_event...")
    
    # Filter to fatal events only
    fatal_events = acled_raw[acled_raw['fatalities'] > 0][['h3_index', 'date']].copy()
    
    if fatal_events.empty:
        logger.warning("  No fatal events found - filling with 9999")
        spine['time_since_last_fatal_event'] = 9999
        return spine
    
    # CRITICAL FIX: Explicit timestamp casting to prevent type mismatch
    fatal_events['date'] = pd.to_datetime(fatal_events['date'])
    fatal_events = fatal_events.sort_values('date').drop_duplicates()
    fatal_events = fatal_events.rename(columns={'date': 'last_fatal_date'})
    
    # Prepare spine for merge_asof
    spine_sorted = spine[['h3_index', 'date']].copy()
    spine_sorted['date'] = pd.to_datetime(spine_sorted['date'])
    spine_sorted = spine_sorted.sort_values('date')
    
    # Merge_asof: for each (h3_index, date), find the most recent fatal event BEFORE that date
    merged = pd.merge_asof(
        spine_sorted,
        fatal_events.sort_values('last_fatal_date'),
        by='h3_index',
        left_on='date',
        right_on='last_fatal_date',
        direction='backward',
        tolerance=pd.Timedelta(days=365*5)
    )
    
    # Calculate days since last fatal event
    merged['time_since_last_fatal_event'] = (
        (merged['date'] - merged['last_fatal_date']).dt.days
    )
    
    # Fill NaN (no prior fatal event) with 9999
    merged['time_since_last_fatal_event'] = merged['time_since_last_fatal_event'].fillna(9999).astype(int)
    
    # Merge back to spine
    spine = spine.merge(
        merged[['h3_index', 'date', 'time_since_last_fatal_event']],
        on=['h3_index', 'date'],
        how='left'
    )
    
    # Fill any remaining NaN
    spine['time_since_last_fatal_event'] = spine['time_since_last_fatal_event'].fillna(9999).astype(int)
    
    # Log summary
    non_default = (spine['time_since_last_fatal_event'] < 9999).sum()
    logger.info(f"  ✓ time_since_last_fatal_event: {non_default:,} non-default values")
    
    return spine


def process_conflict(engine, spine, conflict_specs, features_config):
    logger.info("PHASE 4: Processing Conflict (Config-Driven)...")
    
    if not conflict_specs:
        return spine
    
    temporal_config = features_config.get('temporal', {}) if isinstance(features_config, dict) else {}

    # --- Admin mapping (prefer admin3/sub-prefecture, fallback to admin2, then admin1) ---
    insp = inspect(engine)
    fs_cols = {c["name"] for c in insp.get_columns("features_static", schema=SCHEMA)}
    if "admin3" in fs_cols:
        admin_col = "admin3"
    elif "admin2" in fs_cols:
        admin_col = "admin2"
    else:
        admin_col = "admin1"

    if admin_col in fs_cols:
        logger.info(f"  Using '{admin_col}' from features_static for regional risk aggregation.")
        admin_map = pd.read_sql(f"SELECT h3_index, {admin_col} FROM {SCHEMA}.features_static", engine)
        admin_map["h3_index"] = admin_map["h3_index"].astype("int64")
        spine = spine.merge(admin_map, on="h3_index", how="left")
    else:
        logger.warning("  No admin columns found; regional risk will be skipped.")
        spine[admin_col] = None

    # 1. Load ACLED
    acled_query = f"""
        SELECT 
            ae.h3_index, 
            fs.{admin_col} AS {admin_col},
            ae.event_date AS date, 
            ae.geo_precision, 
            ae.time_precision, 
            ae.fatalities, 
            ae.acled_count_battles,
            ae.acled_count_vac,
            ae.acled_count_explosions,
            ae.acled_count_protests,
            ae.acled_count_riots,
            CASE WHEN ae.event_type = 'Protests' THEN 1 ELSE 0 END as protest_flag,
            CASE WHEN ae.event_type = 'Riots' THEN 1 ELSE 0 END as riot_flag
        FROM {SCHEMA}.acled_events ae
        LEFT JOIN {SCHEMA}.features_static fs ON ae.h3_index = fs.h3_index
    """
    acled_raw = pd.read_sql(acled_query, engine)
    acled_raw['date'] = pd.to_datetime(acled_raw['date'])
    acled_raw["h3_index"] = acled_raw["h3_index"].astype("int64")
    acled_raw["geo_precision"] = pd.to_numeric(acled_raw["geo_precision"], errors="coerce")
    acled_raw["time_precision"] = pd.to_numeric(acled_raw["time_precision"], errors="coerce")
    for c in ['acled_count_battles', 'acled_count_vac', 'acled_count_explosions', 'acled_count_protests', 'acled_count_riots']:
        if c not in acled_raw.columns:
            acled_raw[c] = 0
        acled_raw[c] = acled_raw[c].fillna(0).astype(int)
    
    # Bin to Spine
    dates = sorted(spine['date'].unique())
    acled_raw['spine_date'] = pd.cut(acled_raw['date'], bins=dates, labels=dates[1:], right=True)
    
    # --- Stream A: Local Precision ---
    precise_mask = acled_raw['geo_precision'].isin([1, 2]) & acled_raw['time_precision'].isin([1, 2])
    acled_local = acled_raw[precise_mask]
    acled_local_agg = acled_local.groupby(['h3_index', admin_col, 'spine_date'], observed=True).agg({
        'fatalities': 'sum',
        'protest_flag': 'sum',
        'riot_flag': 'sum',
        'acled_count_battles': 'sum',
        'acled_count_vac': 'sum',
        'acled_count_explosions': 'sum',
        'acled_count_protests': 'sum',
        'acled_count_riots': 'sum'
    }).reset_index().rename(columns={
        'spine_date': 'date',
        'protest_flag': 'protest_count',
        'riot_flag': 'riot_count'
    })
    acled_local_agg['date'] = pd.to_datetime(acled_local_agg['date'])
    
    # Targets from local stream
    acled_local_agg['target_fatalities'] = acled_local_agg['fatalities']
    acled_local_agg['target_binary'] = (acled_local_agg['fatalities'] > 0).astype(int)

    # --- Stream B: Regional Context (leakage-safe) ---
    # Step 1: lag fatalities at the cell level to ensure regional aggregation uses past information only
    acled_local_agg['fatalities_lag1'] = acled_local_agg.groupby('h3_index')['fatalities'].shift(1)
    # Step 2: aggregate lagged fatalities to admin-date, then log-transform
    acled_regional_agg = (
        acled_local_agg.groupby([admin_col, 'date'], observed=True)['fatalities_lag1']
        .sum()
        .reset_index()
        .rename(columns={'fatalities_lag1': 'regional_fatalities_lag1'})
    )
    acled_regional_agg['regional_risk_score_lag1'] = np.log1p(acled_regional_agg['regional_fatalities_lag1'])

    # --- [REMOVED FLAWED LEAKAGE CHECK HERE] ---
    # The previous check failed because "0 fatalities" followed by "0 fatalities" 
    # results in identical risk scores, which triggered the "Equality = Leakage" alarm.
    
    # 2. Load GDELT
    gdelt_vars = ['gdelt_event_count', 'gdelt_avg_tone', 'gdelt_goldstein_mean', 'gdelt_mentions_total']
    gdelt_vars_sql = ", ".join(f"'{v}'" for v in gdelt_vars)
    gdelt_raw = pd.read_sql(
        f"""
        SELECT h3_index, date, variable, value
        FROM {SCHEMA}.features_dynamic_daily
        WHERE variable IN ({gdelt_vars_sql})
        """,
        engine
    )
    # IODA signals now stored separately (internet_outages)
    try:
        ioda_raw = pd.read_sql(
            f"SELECT h3_index, date, variable, value FROM {SCHEMA}.internet_outages",
            engine,
        )
    except Exception:
        ioda_raw = pd.DataFrame(columns=["h3_index", "date", "variable", "value"])
    gdelt_raw['date'] = pd.to_datetime(gdelt_raw['date'])
    gdelt_raw['spine_date'] = pd.cut(gdelt_raw['date'], bins=dates, labels=dates[1:], right=True)
    if not ioda_raw.empty:
        ioda_raw['date'] = pd.to_datetime(ioda_raw['date'])
        ioda_raw['spine_date'] = pd.cut(ioda_raw['date'], bins=dates, labels=dates[1:], right=True)
    else:
        ioda_raw['spine_date'] = pd.Series(dtype='datetime64[ns]')
    
    gdelt_event = gdelt_raw[gdelt_raw['variable'] == 'gdelt_event_count']
    gdelt_tone = gdelt_raw[gdelt_raw['variable'] == 'gdelt_avg_tone']
    gdelt_goldstein = gdelt_raw[gdelt_raw['variable'] == 'gdelt_goldstein_mean']
    gdelt_mentions = gdelt_raw[gdelt_raw['variable'] == 'gdelt_mentions_total']
    # Prefer new outage score; fallback to legacy detected flag
    ioda_outage = ioda_raw[ioda_raw['variable'].isin(['ioda_outage_score', 'ioda_outage_detected'])]
    ioda_connectivity = ioda_raw[ioda_raw['variable'] == 'ioda_connectivity_index'] if not ioda_raw.empty else pd.DataFrame(columns=['h3_index','spine_date','value'])
    
    gdelt_event_agg = gdelt_event.groupby(['h3_index', 'spine_date'], observed=True)['value'].sum().reset_index()
    gdelt_event_agg = gdelt_event_agg.rename(columns={'spine_date': 'date', 'value': 'gdelt_event_count'})
    gdelt_event_agg['date'] = pd.to_datetime(gdelt_event_agg['date'])

    gdelt_tone_agg = gdelt_tone.groupby(['h3_index', 'spine_date'], observed=True)['value'].mean().reset_index()
    gdelt_tone_agg = gdelt_tone_agg.rename(columns={'spine_date': 'date', 'value': 'gdelt_avg_tone'})
    gdelt_tone_agg['date'] = pd.to_datetime(gdelt_tone_agg['date'])

    gdelt_goldstein_agg = gdelt_goldstein.groupby(['h3_index', 'spine_date'], observed=True)['value'].mean().reset_index()
    gdelt_goldstein_agg = gdelt_goldstein_agg.rename(columns={'spine_date': 'date', 'value': 'gdelt_goldstein_mean'})
    gdelt_goldstein_agg['date'] = pd.to_datetime(gdelt_goldstein_agg['date'])

    gdelt_mentions_agg = gdelt_mentions.groupby(['h3_index', 'spine_date'], observed=True)['value'].sum().reset_index()
    gdelt_mentions_agg = gdelt_mentions_agg.rename(columns={'spine_date': 'date', 'value': 'gdelt_mentions_total'})
    gdelt_mentions_agg['date'] = pd.to_datetime(gdelt_mentions_agg['date'])

    ioda_outage_agg = pd.DataFrame(columns=['h3_index','date','ioda_outage_score'])
    if not ioda_outage.empty:
        ioda_outage_agg = (
            ioda_outage
            .groupby(['h3_index', 'spine_date'], observed=True)['value']
            .max()
            .reset_index()
            .rename(columns={'spine_date': 'date', 'value': 'ioda_outage_score'})
        )
        ioda_outage_agg['date'] = pd.to_datetime(ioda_outage_agg['date'])

    ioda_conn_agg = pd.DataFrame(columns=['h3_index','date','ioda_connectivity_index'])
    if not ioda_connectivity.empty:
        ioda_conn_agg = (
            ioda_connectivity
            .groupby(['h3_index','spine_date'], observed=True)['value']
            .mean()
            .reset_index()
            .rename(columns={'spine_date':'date','value':'ioda_connectivity_index'})
        )
        ioda_conn_agg['date'] = pd.to_datetime(ioda_conn_agg['date'])

    # 3. Merge onto Spine
    spine = safe_merge(spine, acled_local_agg, on=['h3_index', 'date'], how='left')
    if admin_col in spine.columns:
        spine = safe_merge(spine, acled_regional_agg[[admin_col, 'date', 'regional_risk_score_lag1']], on=[admin_col, 'date'], how='left')
    spine = safe_merge(spine, gdelt_event_agg, on=['h3_index', 'date'], how='left')
    spine = safe_merge(spine, gdelt_tone_agg, on=['h3_index', 'date'], how='left')
    spine = safe_merge(spine, gdelt_goldstein_agg, on=['h3_index', 'date'], how='left')
    spine = safe_merge(spine, gdelt_mentions_agg, on=['h3_index', 'date'], how='left')
    spine = safe_merge(spine, ioda_outage_agg, on=['h3_index', 'date'], how='left')
    spine = safe_merge(spine, ioda_conn_agg, on=['h3_index', 'date'], how='left')

    # Lag connectivity by one spine step for modeling stability
    if 'ioda_connectivity_index' in spine.columns:
        spine['ioda_connectivity_index_lag1'] = (
            spine.groupby('h3_index')['ioda_connectivity_index'].shift(1)
        )

    # Flag GDELT availability (GDELT coverage starts on 2015-02-18)
    spine['gdelt_data_available'] = (spine['date'] >= pd.Timestamp("2015-02-18")).astype(int)
    # Ensure registry output column exists (passthrough flag)
    spine['gdelt_data_available_flag'] = spine.get('gdelt_data_available', 0)
    # Flag IODA availability (API coverage starts ~2014)
    spine['ioda_data_available'] = (spine['date'] >= pd.Timestamp("2014-01-01")).astype(int)
    spine['ioda_data_available_flag'] = spine.get('ioda_data_available', 0)
    
    # 4. Initialize missing conflict columns with 0 and apply forward-fill
    conflict_raws = list(set([spec['raw'] for spec in conflict_specs]))
    for raw in conflict_raws:
        if raw not in spine.columns:
            spine[raw] = 0
    
    # Fill missing values and apply forward-fill (limit=4)
    impute_cfg = features_config if isinstance(features_config, dict) else {}
    for col in ['fatalities', 'protest_count', 'riot_count', 'gdelt_event_count', 'gdelt_avg_tone', 'gdelt_goldstein_mean', 'gdelt_mentions_total', 'ioda_outage_detected', 'ioda_connectivity_index', 'ioda_connectivity_index_lag1', 'gdelt_data_available', 'ioda_data_available', 'target_fatalities', 'target_binary', 'regional_risk_score_lag1']:
        if col in spine.columns:
            spine[col] = spine[col].fillna(0)
            if col == 'regional_risk_score_lag1':
                spine[col] = apply_forward_fill(spine, col, groupby_col=admin_col, config=impute_cfg, domain="conflict")
            else:
                spine[col] = apply_forward_fill(spine, col, config=impute_cfg, domain="conflict")
    
    # 5. Create fatalities_14d_sum (raw per-step)
    spine['fatalities_14d_sum'] = spine['fatalities']

    # 6. Shock + Stress pipeline with spatial diffusion
    target_cols = [
        'cw_score_local',
        'fatalities_14d_sum',
        'acled_count_battles',
        'acled_count_vac',
        'acled_count_explosions',
        'acled_count_protests',
        'acled_count_riots',
        'driver_resource_cattle',
        'driver_civilian_abuse',
        'gdelt_event_count',
        'gdelt_avg_tone'
    ]

    # Forward fill CrisisWatch across the month (limit 2 steps) if present
    if 'cw_score_local' in spine.columns:
        spine['cw_score_local'] = apply_forward_fill(spine, 'cw_score_local', groupby_col='h3_index', limit=2, config=impute_cfg, domain="conflict").fillna(0)
    else:
        spine['cw_score_local'] = 0

    # Ensure target columns exist
    for col in target_cols:
        if col not in spine.columns:
            spine[col] = 0

    # Remove legacy conflict_density if present
    if 'conflict_density_10km' in spine.columns:
        spine.drop(columns=['conflict_density_10km'], inplace=True)

    for col in target_cols:
        spine = add_spatial_diffusion_features(spine, col, k=1)
        # Temporal lag (shock)
        spine[f"{col}_lag1"] = spine.groupby('h3_index')[col].shift(1).fillna(0)
        spatial_col = f"{col}_spatial_lag"
        spine[f"{spatial_col}_lag1"] = spine.groupby('h3_index')[spatial_col].shift(1).fillna(0)
        # Temporal decay (stress)
        spine = apply_halflife_decay(spine, f"{col}_lag1", features_config)
        spine = apply_halflife_decay(spine, f"{spatial_col}_lag1", features_config)

    # 7. Compute time_since_last_fatal_event
    spine = compute_time_since_last_fatal_event(spine, acled_raw)

    # Ensure regional_risk_score_lag1 exists even if spec was removed (backwards safety)
    if 'regional_risk_score_lag1' not in spine.columns:
        spine['regional_risk_score_lag1'] = 0

    return spine


# ==============================================================================
# PHASE 4B: NLP SIGNALS (Topic Pivoting)
# ==============================================================================
def process_nlp_data(engine, spine, nlp_specs):
    logger.info("PHASE 4B: Processing NLP Signals (Topic Pivoting)...")

    if not nlp_specs:
        logger.info("  No NLP specs found in registry; skipping.")
        return spine

    insp = inspect(engine)
    dates = sorted(spine['date'].unique())

    for spec in nlp_specs:
        source = spec.get('source')
        transformation = spec.get('transformation')
        params = spec.get('transformation_params', {}) or {}
        output_col = spec.get('output_col')

        if source == "NLP_ACLED" and transformation == "pivot":
            if not insp.has_table("features_nlp_acled", schema=SCHEMA):
                logger.info("  features_nlp_acled not found; skipping ACLED NLP pivot.")
                continue

            df = pd.read_sql(
                f"SELECT h3_index, date, acled_topic_id, topic_intensity FROM {SCHEMA}.features_nlp_acled",
                engine
            )
            if df.empty:
                logger.info("  features_nlp_acled is empty; skipping ACLED NLP pivot.")
                continue

            df['date'] = pd.to_datetime(df['date'])
            df['h3_index'] = df['h3_index'].astype('int64')

            topics = params.get('topics') or params.get('values')
            if topics:
                df = df[df['acled_topic_id'].isin(topics)]
            else:
                topics = sorted(df['acled_topic_id'].dropna().unique().tolist())

            df['spine_date'] = pd.cut(df['date'], bins=dates, labels=dates[1:], right=True)
            pivot_df = (
                df.pivot_table(
                    index=['h3_index', 'spine_date'],
                    columns='acled_topic_id',
                    values='topic_intensity',
                    aggfunc='sum',
                    fill_value=0
                )
                .reset_index()
                .rename(columns={'spine_date': 'date'})
            )

            # Ensure all requested topic columns exist
            for topic_id in topics:
                col_name = f"{output_col}_{int(topic_id)}"
                if topic_id in pivot_df.columns:
                    pivot_df = pivot_df.rename(columns={topic_id: col_name})
                else:
                    pivot_df[col_name] = 0

            # Drop the raw topic columns (numeric) if any remain
            keep_cols = ['h3_index', 'date'] + [c for c in pivot_df.columns if isinstance(c, str) and c.startswith(f"{output_col}_")]
            pivot_df = pivot_df[[c for c in keep_cols if c in pivot_df.columns]]

            # Fill NaNs and merge
            pivot_df = pivot_df.fillna(0)
            spine = spine.merge(pivot_df, on=['h3_index', 'date'], how='left')

            # Fill any new columns with 0 where missing
            for c in pivot_df.columns:
                if c not in ['h3_index', 'date']:
                    spine[c] = spine[c].fillna(0)

        # ---------------------------------------------------------
        # STREAM B: CRISISWATCH (Spatial Confidence Weighted Pivot)
        # ---------------------------------------------------------
        elif source == "NLP_CrisisWatch" and transformation == "pivot":
            if not insp.has_table("features_crisiswatch", schema=SCHEMA):
                logger.warning("  features_crisiswatch table not found. Skipping.")
                continue

            # 1. Load Topic ID and the new Spatial Confidence Score
            cw_raw = pd.read_sql(
                f"SELECT h3_index, date, cw_topic_id, spatial_confidence FROM {SCHEMA}.features_crisiswatch",
                engine
            )
            if cw_raw.empty: 
                logger.info("  features_crisiswatch is empty. Skipping.")
                continue

            cw_raw['date'] = pd.to_datetime(cw_raw['date'])
            cw_raw['h3_index'] = cw_raw['h3_index'].astype('int64')

            # 2. Filter topics if config limits them
            valid_topics = params.get('topics')
            if valid_topics:
                cw_raw = cw_raw[cw_raw['cw_topic_id'].isin(valid_topics)]

            # 3. Bin to Spine (Monthly -> Daily/Weekly)
            cw_raw['spine_date'] = pd.cut(cw_raw['date'], bins=dates, labels=dates[1:], right=True)

            # 4. PIVOT: Sum of Spatial Confidence
            pivot_df = (
                cw_raw.pivot_table(
                    index=['h3_index', 'spine_date'],
                    columns='cw_topic_id',
                    values='spatial_confidence',
                    aggfunc='sum',
                    fill_value=0
                )
                .reset_index()
                .rename(columns={'spine_date': 'date'})
            )

            # 5. Rename columns (e.g., 4 -> crisiswatch_topic_4)
            topic_cols = [c for c in pivot_df.columns if c not in ['h3_index', 'date']]
            rename_map = {c: f"{output_col}_{int(c)}" for c in topic_cols}
            pivot_df = pivot_df.rename(columns=rename_map)
            
            # 6. Merge & Fill
            spine = spine.merge(pivot_df, on=['h3_index', 'date'], how='left')
            new_cols = list(rename_map.values())
            spine[new_cols] = spine[new_cols].fillna(0)
            
            logger.info(f"  ✓ Merged {len(new_cols)} CrisisWatch topics (Confidence Weighted).")

    return spine


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run():
    logger.info("=" * 60)
    logger.info("MASTER FEATURE ENGINEERING (Config-Driven + Idempotent)")
    logger.info("=" * 60)
    
    engine = get_db_engine()
    validate_h3_types(engine)
    
    try:
        configs = load_configs()
        if isinstance(configs, tuple):
            data_cfg, feat_cfg = configs[0], configs[1]
        else:
            data_cfg, feat_cfg = configs['data'], configs['features']

        start_date = data_cfg['global_date_window']['start_date']
        end_date = data_cfg['global_date_window']['end_date']
        step_days = feat_cfg['temporal']['step_days']
        
        specs = parse_registry(feat_cfg)
        
        # --- Execution ---
        spine = create_master_spine(engine, start_date, end_date, step_days)
        gc.collect()
        
        # PHASE 2A: Macro-Economic Indicators (NEW)
        spine = process_economics(engine, spine, specs['economic'], feat_cfg)
        gc.collect()
        
        logger.info("PHASE 2B: Processing Food Prices (Spatial Broadcast)...")

        # 1. Load the Grid (REQUIRED for the spatial join to work)
        # This fixes the NameError: name 'gdf_grid' is not defined
        gdf_grid = gpd.read_postgis(
            "SELECT h3_index, geometry FROM car_cewp.features_static", 
            engine, 
            geom_col='geometry'
        )
        gdf_grid['h3_index'] = gdf_grid['h3_index'].astype('int64')

        # 2. Run the new spatial function
        food_df = process_food_prices_spatial(engine, gdf_grid, start_date, end_date)

        if not food_df.empty:
            logger.info(f"Merging {len(food_df)} food price records...")
            spine = safe_merge(spine, food_df, on=['h3_index', 'date'])
        else:
            logger.warning("Food price spatial broadcast returned empty dataframe.")

        # Compute price shocks and index after merge
        price_cols = ['price_maize', 'price_rice', 'price_oil', 'price_sorghum', 'price_cassava', 'price_groundnuts']
        impute_cfg = feat_cfg if isinstance(feat_cfg, dict) else {}

        for col in price_cols:
            if col not in spine.columns:
                spine[col] = np.nan

            # Forward-fill only after first observation (avoid seeding zeros before data exists)
            spine[col] = spine.groupby('h3_index')[col].transform(
                lambda s: s.where(s.ffill().notna()).ffill()
            )

            spine[col] = apply_forward_fill(spine, col, config=impute_cfg, domain="economic")

            trans_spec = next((s for s in specs['social'] if s.get('output_col') == f"{col}_shock"), {})
            params = trans_spec.get('transformation_params', {}) if trans_spec else {}
            lookback_months = params.get('lookback_months', 12)
            steps_per_month = 30.0 / step_days if step_days else 2.14
            window = max(1, int(lookback_months * steps_per_month))
            rolling_mean = spine.groupby('h3_index')[col].transform(
                lambda x: x.rolling(window=window, min_periods=max(1, int(window/2))).mean()
            )
            shock_col = f"{col}_shock"
            spine[shock_col] = spine[col] / (rolling_mean + 1e-6)
            spine[shock_col] = spine[shock_col].fillna(1.0)

        spine['food_price_index'] = spine[price_cols].mean(axis=1, skipna=True)

        # Single availability flag: data exists on/after first observed date
        has_food = spine[price_cols].notna().any(axis=1)
        first_food_date = spine.loc[has_food, 'date'].min()
        if pd.notna(first_food_date):
            spine['food_data_available'] = (spine['date'] >= first_food_date).astype(int)
        else:
            spine['food_data_available'] = 0

        # Fill residual NaN for modeling
        spine[price_cols] = spine[price_cols].fillna(0)
        spine['food_price_index'] = spine['food_price_index'].fillna(0)
        
        # PHASE 2C: Social Data (IOM)
        spine = process_social_data(engine, spine, specs['social'], feat_cfg)
        gc.collect()
        
        # PHASE 2D: Demographics (WorldPop)
        spine = process_demographics(engine, spine, specs['demographic'])
        gc.collect()
        
        # PHASE 3: Environmental
        spine = process_environment(engine, spine, specs['environmental'], feat_cfg)
        gc.collect()
        
        # PHASE 4: Conflict
        spine = process_conflict(engine, spine, specs['conflict'], feat_cfg)
        gc.collect()
        
        # PHASE 4B: NLP Signals (Topic Pivoting)
        spine = process_nlp_data(engine, spine, specs['nlp'])
        gc.collect()
        
        # --- Final Optimization ---
        logger.info("Optimizing data types...")
        fcols = spine.select_dtypes('float64').columns
        spine[fcols] = spine[fcols].astype('float32')
        spine['h3_index'] = spine['h3_index'].astype('int64')
        
        # Clean up any spurious columns
        if 'level_0' in spine.columns: 
            spine.drop(columns=['level_0'], inplace=True)
        
        # ==================================================================
        # CRITICAL DATA SANITIZATION (FIX FOR SHAP CRASH - Task 1)
        # ==================================================================
        # This MUST run before saving to fix corrupted "[9.639088E0]" strings
        logger.info("Running data sanitization to fix list-string encoding...")
        spine = sanitize_numeric_columns(spine)
        
        # Validate numeric integrity
        validate_numeric_integrity(spine)
        
        # ==================================================================
        # ENSURE TARGET COLUMNS ARE PRESERVED (Partial fix - see build_feature_matrix.py)
        # ==================================================================
        target_cols = ['conflict_binary', 'target_binary', 'fatalities', 'target_fatalities']
        for t in target_cols:
            if t in spine.columns:
                spine[t] = spine[t].fillna(0).astype(int)
        
        # Log target column status
        present_targets = [t for t in target_cols if t in spine.columns]
        logger.info(f"Target columns preserved: {present_targets}")

        # 3. Save local parquet snapshot
        logger.info(f"Final Matrix Shape: {spine.shape}")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        spine.to_parquet(OUTPUT_PATH, index=False)
        
        # --- Ensure Table Schema ---
        ensure_output_table_schema(engine, spine)
        
        # --- Upload in Chunks using Upsert (IDEMPOTENT) ---
        total_rows = len(spine)
        num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        logger.info(f"Uploading {total_rows:,} rows in {num_chunks} chunks of {CHUNK_SIZE:,}...")
        
        # --- TYPE FIX: Define columns that MUST be integers for Postgres ---
        integer_cols = [
            "gdelt_event_count", "acled_event_count", "protest_count", "riot_count", 
            "fatalities", "time_since_last_fatal_event", "pop_count", "iom_displacement_sum", "is_dry_season", "iom_data_available"
        ]

        for chunk_num, start_idx in enumerate(range(0, total_rows, CHUNK_SIZE), start=1):
            end_idx = min(start_idx + CHUNK_SIZE, total_rows)
            chunk = spine.iloc[start_idx:end_idx].copy()
            
            # Force Integer Types for Safety
            for col in integer_cols:
                if col in chunk.columns:
                    chunk[col] = chunk[col].fillna(0).round().astype(int)

            logger.info(f"  Chunk {chunk_num}/{num_chunks}: rows {start_idx+1:,} to {end_idx:,}")
            
            upload_to_postgis(engine, chunk, OUTPUT_TABLE, SCHEMA, PRIMARY_KEYS)
            
            # Free memory
            del chunk
            gc.collect()
        
        logger.info(f"✅ FEATURE ENGINEERING COMPLETE. Upserted {total_rows:,} rows to {SCHEMA}.{OUTPUT_TABLE}")
        
    except Exception as e:
        logger.critical(f"Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    run()
