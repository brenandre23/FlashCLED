"""
process_spine_and_infrastructure.py
=====================================
REFACTORED v3: Stocks vs Flows Imputation Strategy

Critical Bug Fix (2026-01-24):
- STOCK variables (displacement, prices, nightlights, population) now preserve NaN 
  after forward-fill limit exhaustion. XGBoost learns optimal split direction for 
  missing values natively. Zero-filling stocks creates phantom "resolution" events.
- FLOW variables (events, outages) correctly zero-fill (no event = 0).

Key Changes from v2:
1. STOCK/FLOW SEPARATION: Domain types explicitly classified in features.yaml
2. NO ZERO-FILL FALLBACK for stocks - NaN signals "unknown" to tree models
3. RECENCY FEATURES: Days-since-last-observation for all stock variables
4. AVAILABILITY FLAGS: Generated AFTER imputation (reflects actual usable data)

Imputation Philosophy:
- Flows (conflict, news, outages): Zero-fill immediately (no event = 0)
- Stocks (displacement, prices, environment): Forward-fill with limit, then NaN
- XGBoost handles NaN natively via optimal split direction learning
"""

import sys
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import timedelta
from sqlalchemy import text, inspect

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, apply_forward_fill, upload_to_postgis
from pipeline.processing.utils_processing import (
    SCHEMA,
    OUTPUT_TABLE,
    PRIMARY_KEYS,
    CHUNK_SIZE,
    parse_registry,
    create_table_if_not_exists,
    sanitize_numeric_columns,
    validate_numeric_integrity,
    process_food_prices_spatial
)


# =============================================================================
# STRUCTURAL BREAK DATES (From Data Source Audit v4.0)
# =============================================================================
STRUCTURAL_BREAKS = {
    'viirs': pd.Timestamp('2012-01-28'),       # VIIRS sensor launch
    'gdelt': pd.Timestamp('2015-02-18'),       # GDELT v2 start
    'dynamic_world': pd.Timestamp('2017-03-07'),  # Sentinel-2B launch (updated)
    'iom': pd.Timestamp('2015-01-31'),         # IOM DTM CAR coverage start
    'ioda': pd.Timestamp('2022-02-01'),        # IODA monitoring start
    'econ': pd.Timestamp('2003-12-01'),        # Yahoo Finance reliable start
    'food': pd.Timestamp('2015-01-01'),        # FEWS NET price coverage
}


# =============================================================================
# DOMAIN TYPE CLASSIFICATION (Stocks vs Flows)
# =============================================================================
# This is the canonical source of truth for imputation behavior.
# Can be overridden by features.yaml imputation.domains.<domain>.type
DOMAIN_TYPES = {
    'environmental': 'stock',   # Climate, vegetation, nightlights persist
    'economic': 'stock',        # Prices, indices don't vanish
    'social': 'stock',          # Displacement, population persist
    'demographic': 'stock',     # Population counts persist
    'conflict': 'flow',         # No event recorded = no event occurred
    'news': 'flow',             # No event recorded = no event occurred
    'outage': 'flow',           # No outage detected = no outage
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def process_viirs_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIIRS derived features: Kinetic Delta and strict availability flag.
    
    Target: Post-merge, Pre-upload (after environmental features joined to spine).
    
    Derivations:
    1. viirs_data_available: Dual-guard (date >= 2012-01-28 AND ntl_mean.notna())
       - Prevents leakage if imputation accidentally filled values pre-2012
    2. ntl_kinetic_delta: Peak - Mean, clipped >= 0
       - Isolates transient fire/explosion signals from stable infrastructure
       - NaN propagates naturally (NaN - Value = NaN)
    
    Args:
        df: DataFrame with ntl_mean, ntl_peak columns (post-merge)
    
    Returns:
        df with viirs_data_available and ntl_kinetic_delta columns added
    """
    VIIRS_START = pd.Timestamp('2012-01-28')
    
    # 1. Strict Structural Availability Guard
    # Dual-check: Date threshold AND actual data presence
    # This overrides any prior flag computation to ensure correctness
    if 'ntl_mean' in df.columns:
        date_mask = df['date'] >= VIIRS_START
        data_mask = df['ntl_mean'].notna()
        df['viirs_data_available'] = (date_mask & data_mask).astype(int)
        
        # Log coverage
        pct_available = df['viirs_data_available'].mean() * 100
        logger.info(f"  VIIRS availability (dual-guard): {pct_available:.1f}% of rows")
    
    # 2. Kinetic Delta (Fire/Explosion Proxy)
    # Peak - Mean captures transient spikes above stable baseline
    # Pandas propagates NaN automatically: NaN - Value = NaN
    if {'ntl_peak', 'ntl_mean'}.issubset(df.columns):
        df['ntl_kinetic_delta'] = df['ntl_peak'] - df['ntl_mean']
        # Only clip valid numbers to >= 0, leave NaNs as NaNs
        # (Negative values theoretically shouldn't occur but defensive coding)
        df['ntl_kinetic_delta'] = df['ntl_kinetic_delta'].clip(lower=0)
        
        # Log delta statistics
        valid_deltas = df['ntl_kinetic_delta'].dropna()
        if len(valid_deltas) > 0:
            logger.info(f"  NTL kinetic delta: mean={valid_deltas.mean():.2f}, max={valid_deltas.max():.2f}, non-null={len(valid_deltas):,}")
    
    return df


def compute_availability_flag(spine, data_cols, structural_break_date, flag_name):
    """
    Compute availability flag AFTER imputation.
    
    Logic:
    1. Check structural break date threshold
    2. Check if ANY of the data columns have non-null values
    3. Flag = 1 only if BOTH conditions are met
    
    Args:
        spine: DataFrame with data
        data_cols: List of column names to check for data presence
        structural_break_date: pd.Timestamp of when source became available
        flag_name: Name of the output availability flag column
    
    Returns:
        spine with flag_name column added
    """
    # Date threshold
    date_ok = spine['date'] >= structural_break_date
    
    # Data presence (any column has data)
    existing_cols = [c for c in data_cols if c in spine.columns]
    if existing_cols:
        data_ok = spine[existing_cols].notna().any(axis=1)
    else:
        data_ok = pd.Series(False, index=spine.index)
    
    spine[flag_name] = (date_ok & data_ok).astype(int)
    
    return spine


def apply_domain_imputation(spine: pd.DataFrame, col: str, config: dict, domain: str) -> pd.DataFrame:
    """
    Apply domain-specific imputation based on variable type (stock vs flow).
    
    CRITICAL DISTINCTION:
    - FLOW variables (conflict, news, outages): Zero-fill. No event = 0.
    - STOCK variables (displacement, prices, environment): Forward-fill to limit, then NaN.
      XGBoost learns optimal split direction for missing values during training.
      Zero-filling stocks creates phantom "resolution" events - a critical bug.
    
    Args:
        spine: DataFrame
        col: Column name to impute
        config: Features config dict
        domain: Domain name ('environmental', 'economic', 'social', 'conflict', etc.)
    
    Returns:
        spine with imputed column (NaN preserved for stocks beyond limit)
    """
    if col not in spine.columns:
        return spine
    
    # Get domain configuration
    domain_cfg = config.get('imputation', {}).get('domains', {}).get(domain, {})
    limit = domain_cfg.get('limit')
    if limit is None:
        limit = config.get('imputation', {}).get('defaults', {}).get('limit', 4)
    
    # Determine domain type (stock vs flow)
    domain_type = domain_cfg.get('type', DOMAIN_TYPES.get(domain, 'stock'))
    
    if domain_type == 'flow':
        # FLOW: No event recorded means no event occurred
        # Zero-fill is semantically correct
        spine[col] = spine[col].fillna(0)
        logger.debug(f"  {col}: FLOW domain - zero-filled all NaN")
        
    else:  # stock
        # STOCK: State persists but becomes uncertain over time
        # Forward-fill within validity window, preserve NaN beyond
        if limit > 0:
            spine[col] = spine.groupby('h3_index')[col].ffill(limit=limit)
            filled_count = spine[col].notna().sum()
            total_count = len(spine)
            logger.debug(f"  {col}: STOCK domain - forward-fill (limit={limit}), {filled_count}/{total_count} non-null")
        # CRITICAL: Do NOT zero-fill remaining NaN
        # XGBoost handles NaN natively via learned optimal split direction
    
    return spine


def add_recency_features(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = 'date',
    group_col: str = 'h3_index'
) -> pd.DataFrame:
    """
    Add days-since-last-observation feature for stock variables.
    
    Allows XGBoost to split on data freshness:
    - iom_displacement_sum=10000, iom_displacement_sum_recency_days=5 → Fresh, trust it
    - iom_displacement_sum=10000, iom_displacement_sum_recency_days=180 → Stale, discount it
    
    This provides explicit uncertainty quantification for forward-filled values.
    
    Args:
        df: DataFrame with value column
        value_col: Column to compute recency for
        date_col: Date column name
        group_col: Grouping column (typically h3_index)
    
    Returns:
        df with {value_col}_recency_days column added
    """
    if value_col not in df.columns:
        return df
    
    recency_col = f'{value_col}_recency_days'
    
    # Mark rows with actual observations (not forward-filled)
    # We need to detect this BEFORE forward-fill, so we track original NaN mask
    # Since this is called after imputation, we check for the _original_ observation
    # by looking at where the value first appeared
    
    # Sort to ensure temporal order
    df = df.sort_values([group_col, date_col])
    
    # Create mask of original observations
    # An observation is "original" if it differs from the prior value OR is the first non-null
    df['_prev_val'] = df.groupby(group_col)[value_col].shift(1)
    df['_is_original'] = (
        df[value_col].notna() & 
        (
            df['_prev_val'].isna() |  # First observation
            (df[value_col] != df['_prev_val'])  # Value changed
        )
    )
    
    # For each group, forward-fill the date of the last original observation
    df['_last_obs_date'] = df[date_col].where(df['_is_original'])
    df['_last_obs_date'] = df.groupby(group_col)['_last_obs_date'].ffill()
    
    # Compute days since last observation
    df[recency_col] = (df[date_col] - df['_last_obs_date']).dt.days
    
    # Cleanup temporary columns
    df.drop(columns=['_prev_val', '_is_original', '_last_obs_date'], inplace=True, errors='ignore')
    
    # Where original data is null (never observed), recency should be NaN
    df.loc[df[value_col].isna(), recency_col] = np.nan
    
    logger.info(f"  Added recency feature: {recency_col} (mean={df[recency_col].mean():.1f} days)")
    
    return df


# =============================================================================
# PHASE FUNCTIONS
# =============================================================================

def create_master_spine(engine, start_date, end_date, step_days):
    """Generate the H3 x Date spine with temporal context features."""
    logger.info("PHASE 1: Generating Master Temporal Spine...")
    
    with engine.connect() as conn:
        h3_df = pd.read_sql(f"SELECT h3_index FROM {SCHEMA}.features_static", conn)
    unique_h3 = h3_df['h3_index'].astype('int64').unique()
    logger.info(f"  Loaded {len(unique_h3):,} unique H3 cells.")
    
    freq = f"{step_days}D"
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    logger.info(f"  Generated {len(dates)} temporal steps (Freq: {freq}).")
    
    index = pd.MultiIndex.from_product([unique_h3, dates], names=['h3_index', 'date'])
    spine = pd.DataFrame(index=index).reset_index()
    
    # Temporal context
    spine['year'] = spine['date'].dt.year
    spine['epoch'] = spine.groupby('year')['date'].rank(method='dense').astype(int)
    spine.drop(columns=['year'], inplace=True)
    
    spine['month_sin'] = np.sin(2 * np.pi * spine['date'].dt.month / 12).astype('float32')
    spine['month_cos'] = np.cos(2 * np.pi * spine['date'].dt.month / 12).astype('float32')
    spine['is_dry_season'] = spine['date'].dt.month.isin([11, 12, 1, 2, 3, 4]).astype(int)
    
    return spine


def process_economics(engine, spine, econ_specs, feat_cfg):
    """Process macro-economic indicators with lag and availability flag."""
    logger.info("PHASE 2A: Processing Macro-Economic Indicators...")
    
    inspector = inspect(engine)
    economy_specs = [spec for spec in econ_specs if spec.get('source') == 'Economy']
    if not economy_specs:
        return spine
    if not inspector.has_table("economic_drivers", schema=SCHEMA):
        return spine
    
    raw_cols = list(set([spec['raw'] for spec in economy_specs if spec['raw'] != 'econ_data_available']))
    if not raw_cols:
        return spine
    
    cols_sql = ', '.join(raw_cols)
    econ_df = pd.read_sql(f"SELECT date, {cols_sql} FROM {SCHEMA}.economic_drivers ORDER BY date", engine)
    econ_df['date'] = pd.to_datetime(econ_df['date'])
    
    # Merge with tolerance
    spine = spine.sort_values('date')
    econ_df = econ_df.sort_values('date')
    spine = pd.merge_asof(spine, econ_df, on='date', direction='backward', tolerance=pd.Timedelta(days=14))
    
    # Apply imputation (economic domain = STOCK, no zero-fill fallback)
    for col in raw_cols:
        if col in spine.columns:
            spine = apply_domain_imputation(spine, col, feat_cfg, 'economic')
    
    # Generate transformed columns (lags)
    for spec in economy_specs:
        raw = spec['raw']
        trans = spec.get('transformation', 'none')
        out_col = spec.get('output_col')
        if not out_col:
            continue
        
        if raw == 'econ_data_available':
            continue
        
        if raw not in spine.columns:
            spine[out_col] = np.nan  # Changed from 0.0 to preserve NaN semantics
            continue
        
        if 'lag' in trans:
            # Shift preserves NaN naturally
            spine[out_col] = spine.groupby('h3_index')[raw].shift(1)
        else:
            spine[out_col] = spine[raw]
    
    # Compute availability flag AFTER imputation
    spine = compute_availability_flag(
        spine, raw_cols, STRUCTURAL_BREAKS['econ'], 'econ_data_available'
    )
    
    return spine


def process_food_security(engine, spine, gdf_grid, start_date, end_date, feat_cfg):
    """Process food prices with spatial broadcast, shocks, and availability flag."""
    logger.info("PHASE 2B: Processing Food Prices (Spatial Broadcast)...")
    
    food_df = process_food_prices_spatial(engine, gdf_grid, start_date, end_date)
    
    if food_df.empty:
        logger.warning("Food price spatial broadcast returned empty dataframe.")
        spine['food_price_index'] = np.nan  # Changed from 0.0
        spine['food_data_available'] = 0
        return spine
    
    # Merge food data
    spine = spine.merge(food_df, on=['h3_index', 'date'], how='left')
    
    price_cols = ['price_maize', 'price_rice', 'price_oil', 'price_sorghum', 'price_cassava', 'price_groundnuts']
    
    # Apply imputation (economic domain = STOCK, preserve NaN)
    for col in price_cols:
        if col not in spine.columns:
            spine[col] = np.nan
        spine = apply_domain_imputation(spine, col, feat_cfg, 'economic')
    
    # Add recency features for price columns (helps model discount stale data)
    for col in price_cols:
        if col in spine.columns:
            spine = add_recency_features(spine, col)
    
    # Compute shocks (12-month rolling mean) - only where data exists
    window = 26  # approx 12 months at 14-day steps
    for col in price_cols:
        if col in spine.columns:
            rolling_mean = spine.groupby('h3_index')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            # Shock is ratio to rolling mean; NaN if base data is NaN
            spine[f"{col}_shock"] = (spine[col] / (rolling_mean + 1e-6))
            # Don't fill NaN shocks - let model handle them
    
    # Food price index (mean of available prices, NaN if all NaN)
    existing_price_cols = [c for c in price_cols if c in spine.columns]
    spine['food_price_index'] = spine[existing_price_cols].mean(axis=1, skipna=True)
    # Note: mean with skipna=True returns NaN if all values are NaN
    
    # Add recency for food_price_index
    spine = add_recency_features(spine, 'food_price_index')
    
    # Availability flag AFTER imputation
    spine = compute_availability_flag(
        spine, existing_price_cols, STRUCTURAL_BREAKS['food'], 'food_data_available'
    )
    
    return spine


def process_social_data(engine, spine, social_specs, feat_cfg):
    """Process IOM displacement with publication lag and availability flag."""
    logger.info("PHASE 2C: Processing Social Data (IOM)...")
    
    inspector = inspect(engine)
    iom_spec = next((x for x in social_specs if 'iom_displacement_sum' in x.get('raw', '')), None)
    
    if not iom_spec or not inspector.has_table("iom_displacement_h3", schema=SCHEMA):
        spine['iom_displacement_count_lag1'] = np.nan  # Changed from 0.0
        spine['iom_data_available'] = 0
        return spine
    
    iom_df = pd.read_sql(
        f"SELECT h3_index, date, iom_displacement_sum FROM {SCHEMA}.iom_displacement_h3", 
        engine
    )
    iom_df['date'] = pd.to_datetime(iom_df['date'])
    iom_df['h3_index'] = iom_df['h3_index'].astype('int64')
    
    # Apply publication lag (2 steps = 28 days)
    pub_lag_steps = feat_cfg.get('temporal', {}).get('publication_lags', {}).get('IOM_DTM', 2)
    step_days = feat_cfg.get('temporal', {}).get('step_days', 14)
    lag_days = pub_lag_steps * step_days
    iom_df['date'] = iom_df['date'] + pd.Timedelta(days=lag_days)
    logger.info(f"  Applied {lag_days}-day publication lag to IOM data")
    
    # Merge
    spine = spine.merge(iom_df, on=['h3_index', 'date'], how='left')
    
    # Apply imputation (social domain = STOCK, preserve NaN after limit)
    # CRITICAL FIX: No zero-fill fallback for displacement data
    spine = apply_domain_imputation(spine, 'iom_displacement_sum', feat_cfg, 'social')
    
    # Add recency feature (helps model discount stale displacement data)
    spine = add_recency_features(spine, 'iom_displacement_sum')
    
    # Generate lag (preserves NaN)
    out_col = iom_spec.get('output_col', 'iom_displacement_count_lag1')
    if 'lag' in iom_spec.get('transformation', ''):
        spine[out_col] = spine.groupby('h3_index')['iom_displacement_sum'].shift(1)
    else:
        spine[out_col] = spine['iom_displacement_sum']
    
    # Availability flag AFTER imputation
    spine = compute_availability_flag(
        spine, ['iom_displacement_sum'], STRUCTURAL_BREAKS['iom'], 'iom_data_available'
    )
    
    # Cleanup intermediate column
    if 'iom_displacement_sum' in spine.columns and out_col != 'iom_displacement_sum':
        spine.drop(columns=['iom_displacement_sum'], inplace=True)
    
    return spine


def process_demographics(engine, spine, demo_specs):
    """Process WorldPop population with structural break flag."""
    logger.info("PHASE 2D: Processing Demographics (WorldPop)...")
    
    inspector = inspect(engine)
    pop_spec = next((x for x in demo_specs if x.get('raw') == 'pop_count'), None)
    
    if not pop_spec or not inspector.has_table("population_h3", schema=SCHEMA):
        spine['pop_log'] = np.nan  # Changed from 0.0
        spine['is_worldpop_v1'] = 1
        return spine
    
    pop_df = pd.read_sql(f"SELECT h3_index, year, pop_count FROM {SCHEMA}.population_h3", engine)
    pop_df['h3_index'] = pop_df['h3_index'].astype('int64')
    
    spine['year'] = spine['date'].dt.year
    spine = spine.merge(pop_df, on=['h3_index', 'year'], how='left')
    spine = spine.sort_values(['h3_index', 'date'])
    
    # Forward-fill within each hex (population changes slowly)
    # STOCK variable - do NOT zero-fill
    spine['pop_count'] = spine.groupby('h3_index')['pop_count'].ffill()
    # Note: No .fillna(0) - preserve NaN for pre-coverage or missing hexes
    
    # Transform
    out_col = pop_spec.get('output_col', 'pop_log')
    if 'log1p' in pop_spec.get('transformation', ''):
        spine[out_col] = np.log1p(spine['pop_count'])
    else:
        spine[out_col] = spine['pop_count']
    
    # Structural break flag (V1 vs V2 methodology)
    spine['is_worldpop_v1'] = (spine['date'].dt.year < 2015).astype(int)
    
    # Cleanup
    spine.drop(columns=['year', 'pop_count'], inplace=True)
    
    return spine


def process_environment(engine, spine, env_specs, feat_cfg):
    """Process environmental features from GEE with availability flags."""
    logger.info("PHASE 3A: Processing Environmental Features (GEE)...")
    
    inspector = inspect(engine)
    if not inspector.has_table("environmental_features", schema=SCHEMA):
        logger.warning("environmental_features table not found")
        return spine
    
    # Get available columns
    env_cols = {col['name'] for col in inspector.get_columns("environmental_features", schema=SCHEMA)}
    
    # Raw columns to load (excluding computed ones)
    raw_cols = list(set([item['raw'] for item in env_specs 
                        if item.get('source') not in ['DynamicWorld'] and item['raw'] in env_cols]))
    
    if not raw_cols:
        return spine
    
    db_cols = ', '.join(raw_cols)
    env_df = pd.read_sql(f"SELECT h3_index, date, {db_cols} FROM {SCHEMA}.environmental_features", engine)
    env_df['h3_index'] = env_df['h3_index'].astype('int64')
    env_df['date'] = pd.to_datetime(env_df['date'])
    
    # Merge
    spine = spine.merge(env_df, on=['h3_index', 'date'], how='left')
    
    # Apply imputation (environmental domain = STOCK, preserve NaN after limit)
    for raw in raw_cols:
        # Do NOT forward-fill VIIRS nightlights to avoid misleading carry-over
        if raw.startswith('ntl_'):
            continue
        spine = apply_domain_imputation(spine, raw, feat_cfg, 'environmental')
    
    # Add recency for nightlights (key stock variable)
    if 'ntl_mean' in spine.columns:
        spine = add_recency_features(spine, 'ntl_mean')
    
    # Generate transformed columns
    for spec in env_specs:
        if spec.get('source') == 'DynamicWorld':
            continue  # Handled separately
        
        raw = spec['raw']
        out = spec.get('output_col')
        if not out or raw not in spine.columns:
            continue
        
        trans = spec.get('transformation', 'none')
        if 'anomaly' in trans:
            # Simplified anomaly: raw value (full climatology would go here)
            spine[out] = spine[raw]
        elif 'lag' in trans:
            spine[out] = spine.groupby('h3_index')[raw].shift(1)
        else:
            spine[out] = spine[raw]
    
    # VIIRS availability flag
    if 'ntl_mean' in spine.columns:
        spine = compute_availability_flag(
            spine, ['ntl_mean'], STRUCTURAL_BREAKS['viirs'], 'viirs_data_available'
        )
    else:
        spine['viirs_data_available'] = (spine['date'] >= STRUCTURAL_BREAKS['viirs']).astype(int)
    
    return spine


def process_dynamic_world(engine, spine, feat_cfg):
    """
    Process Dynamic World landcover from landcover_features table.
    
    Maps:
        dw_grass_frac -> landcover_grass
        dw_crops_frac -> landcover_crops
        dw_trees_frac -> landcover_trees
        dw_bare_frac  -> landcover_bare
        dw_built_frac -> landcover_built
    """
    logger.info("PHASE 3B: Processing Dynamic World Landcover...")
    
    inspector = inspect(engine)
    if not inspector.has_table("landcover_features", schema=SCHEMA):
        logger.warning("landcover_features table not found - creating stub columns")
        for col in ['landcover_grass', 'landcover_crops', 'landcover_trees', 'landcover_bare', 'landcover_built']:
            spine[col] = np.nan  # Changed from 0.0
        spine['landcover_data_available'] = 0
        return spine
    
    # Load Dynamic World data
    dw_df = pd.read_sql(
        f"""SELECT h3_index, date, dw_grass_frac, dw_crops_frac, dw_trees_frac, dw_bare_frac, dw_built_frac 
            FROM {SCHEMA}.landcover_features""",
        engine
    )
    dw_df['h3_index'] = dw_df['h3_index'].astype('int64')
    dw_df['date'] = pd.to_datetime(dw_df['date'])
    
    logger.info(f"  Loaded {len(dw_df):,} Dynamic World records")
    
    # Rename columns to match models.yaml expectations
    rename_map = {
        'dw_grass_frac': 'landcover_grass',
        'dw_crops_frac': 'landcover_crops',
        'dw_trees_frac': 'landcover_trees',
        'dw_bare_frac': 'landcover_bare',
        'dw_built_frac': 'landcover_built',
    }
    dw_df = dw_df.rename(columns=rename_map)
    
    # Merge
    spine = spine.merge(dw_df, on=['h3_index', 'date'], how='left')
    
    # Apply imputation (environmental domain = STOCK, preserve NaN)
    landcover_cols = list(rename_map.values())
    for col in landcover_cols:
        if col in spine.columns:
            spine = apply_domain_imputation(spine, col, feat_cfg, 'environmental')
    
    # Availability flag AFTER imputation
    spine = compute_availability_flag(
        spine, landcover_cols, STRUCTURAL_BREAKS['dynamic_world'], 'landcover_data_available'
    )
    
    return spine


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run():
    """Main execution function."""
    engine = get_db_engine()
    configs = load_configs()
    
    if isinstance(configs, tuple):
        data_cfg, feat_cfg = configs[0], configs[1]
    else:
        data_cfg, feat_cfg = configs['data'], configs['features']
    
    start_date = data_cfg['global_date_window']['start_date']
    end_date = data_cfg['global_date_window']['end_date']
    step_days = feat_cfg['temporal']['step_days']
    specs = parse_registry(feat_cfg)
    
    # Log imputation strategy
    logger.info("=" * 60)
    logger.info("IMPUTATION STRATEGY: Stocks vs Flows")
    logger.info("=" * 60)
    for domain, dtype in DOMAIN_TYPES.items():
        cfg_type = feat_cfg.get('imputation', {}).get('domains', {}).get(domain, {}).get('type', dtype)
        cfg_limit = feat_cfg.get('imputation', {}).get('domains', {}).get(domain, {}).get('limit', 'default')
        logger.info(f"  {domain:15s}: type={cfg_type:5s}, limit={cfg_limit}")
    logger.info("=" * 60)
    
    # 1. Create Spine
    spine = create_master_spine(engine, start_date, end_date, step_days)
    
    # 2. Economics
    spine = process_economics(engine, spine, specs['economic'], feat_cfg)
    
    # 3. Food Security
    gdf_grid = gpd.read_postgis(
        f"SELECT h3_index, geometry FROM {SCHEMA}.features_static", 
        engine, geom_col='geometry'
    )
    gdf_grid['h3_index'] = gdf_grid['h3_index'].astype('int64')
    spine = process_food_security(engine, spine, gdf_grid, start_date, end_date, feat_cfg)
    
    # 4. Social (IOM)
    spine = process_social_data(engine, spine, specs['social'], feat_cfg)
    
    # 5. Demographics
    spine = process_demographics(engine, spine, specs['demographic'])
    
    # 6. Environment (GEE)
    spine = process_environment(engine, spine, specs['environmental'], feat_cfg)
    
    # 7. Dynamic World (separate table)
    spine = process_dynamic_world(engine, spine, feat_cfg)
    
    # 8. VIIRS Derived Features (Kinetic Delta + Strict Availability)
    # Must run AFTER environmental merge but BEFORE upload
    logger.info("PHASE 3C: Computing VIIRS Derived Features...")
    spine = process_viirs_derived_features(spine)
    
    # 9. Sanitize and validate NaN percentages against expected values
    # Fully dynamic: discovers source tables, maps columns, computes expected NaN%
    spine = sanitize_numeric_columns(
        spine,
        spine_start=pd.Timestamp(start_date),
        spine_end=pd.Timestamp(end_date),
        step_days=step_days,
        tolerance_pct=5.0,
        engine=engine,
        features_config=feat_cfg  # For column type classification from registry
    )
    
    # 10. Upload (DROP and recreate - this is the context layer)
    logger.info(f"\nUploading Contextual Features: {len(spine):,} rows...")
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{OUTPUT_TABLE} CASCADE"))
    
    total_rows = len(spine)
    num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, total_rows, CHUNK_SIZE):
        chunk = spine.iloc[i:i + CHUNK_SIZE].copy()
        upload_to_postgis(engine, chunk, OUTPUT_TABLE, SCHEMA, PRIMARY_KEYS)
    
    logger.info("\n✅ Contextual Features Uploaded.")
    
    # Log availability flag summary
    logger.info("\nAvailability Flag Summary:")
    for flag in ['econ_data_available', 'food_data_available', 'iom_data_available', 
                 'viirs_data_available', 'landcover_data_available']:
        if flag in spine.columns:
            pct = spine[flag].mean() * 100
            logger.info(f"  {flag}: {pct:.1f}% of rows available")
    
    gc.collect()


if __name__ == "__main__":
    run()
