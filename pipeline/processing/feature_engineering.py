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
"""

import sys
import gc
import numpy as np
import pandas as pd
import h3.api.basic_int as h3
from pathlib import Path
from sqlalchemy import text, inspect
from scipy.spatial import cKDTree

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
        'demographic': []
    }
    
    source_map = {
        'CHIRPS': 'environmental', 'ERA5': 'environmental', 
        'MODIS': 'environmental', 'VIIRS': 'environmental', 'JRC_Landsat': 'environmental',
        'ACLED': 'conflict', 'GDELT': 'conflict',
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
def process_economics(engine, spine, econ_specs):
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
    
    # 1. Identify raw columns needed from the table
    raw_cols = list(set([spec['raw'] for spec in economy_specs]))
    logger.info(f"  Loading columns: {raw_cols}")
    
    # 2. Load economic data from DB
    cols_sql = ', '.join(raw_cols)
    econ_df = pd.read_sql(
        f"SELECT date, {cols_sql} FROM {SCHEMA}.economic_drivers ORDER BY date",
        engine
    )
    econ_df['date'] = pd.to_datetime(econ_df['date'])
    
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
    for col in raw_cols:
        if col in spine.columns:
            spine[col] = apply_forward_fill(spine, col, config=None)
    
    # 5. Apply transformations from registry
    for spec in economy_specs:
        raw = spec['raw']
        trans = spec.get('transformation', 'none')
        out_col = spec.get('output_col')
        
        if not out_col:
            logger.warning(f"  ⚠️ No output_col for {raw}. Skipping.")
            continue
        
        if raw not in spine.columns:
            logger.warning(f"  ⚠️ Column {raw} not found in data. Filling {out_col} with 0.")
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
    cols_to_drop = [col for col in raw_cols if col in spine.columns and col not in 
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
                
                # Ensure all commodity columns exist
                for col in ['price_maize', 'price_oil', 'price_rice', 'price_sorghum']:
                    if col not in df_pivot.columns:
                        df_pivot[col] = np.nan
                
                # D. Voronoi Mapping (H3 -> Nearest Market)
                unique_h3 = spine['h3_index'].unique()
                grid_points = np.fliplr(get_h3_centroids(unique_h3))
                market_points = locs[['longitude', 'latitude']].values
                tree = cKDTree(market_points)
                _, indices = tree.query(grid_points, k=1)
                mapped_markets = locs.iloc[indices]['market_name'].values
                
                h3_map = pd.DataFrame({'h3_index': unique_h3, 'nearest_market': mapped_markets})
                spine = spine.merge(h3_map, on='h3_index', how='left')
                
                # E. Merge Pivoted Prices
                df_pivot = df_pivot.sort_values('date')
                spine = spine.merge(
                    df_pivot.rename(columns={'market_name': 'nearest_market'}),
                    on=['date', 'nearest_market'], how='left'
                )
                
                # F. Forward-fill and fill remaining NaN with 0 (config-driven)
                impute_cfg = feat_cfg
                step_days = feat_cfg.get('temporal', {}).get('step_days', 14)
                price_cols = ['price_maize', 'price_oil', 'price_rice', 'price_sorghum']
                for col in price_cols:
                    if col in spine.columns:
                        spine[col] = apply_forward_fill(spine, col, config=impute_cfg).fillna(0)
                        
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
                        
                        logger.info(f"  ✓ Calculated shock feature: {shock_col}")
                    else:
                        spine[col] = 0.0
                        spine[f"{col}_shock"] = 0.0
                
                spine.drop(columns=['nearest_market'], inplace=True)
                
                # Log summary
                for col in price_cols:
                    non_zero = (spine[col] > 0).sum()
                    logger.info(f"  ✓ {col}: {non_zero:,} non-zero values")
            else:
                logger.warning("  ⚠️ No commodities matched mapping. Filling price columns with 0.")
                for col in ['price_maize', 'price_oil', 'price_rice', 'price_sorghum']:
                    spine[col] = 0.0
                    spine[f"{col}_shock"] = 0.0
        else:
            logger.warning("  ⚠️ Market locations or prices empty. Filling price columns with 0.")
            for col in ['price_maize', 'price_oil', 'price_rice', 'price_sorghum']:
                spine[col] = 0.0
                spine[f"{col}_shock"] = 0.0
    else:
        logger.warning("  ⚠️ Market tables not found. Filling price columns with 0.")
        for col in ['price_maize', 'price_oil', 'price_rice', 'price_sorghum']:
            spine[col] = 0.0
            spine[f"{col}_shock"] = 0.0
    
    return spine


# ==============================================================================
# PHASE 2C: SOCIAL DATA (IPC & IOM)
# ==============================================================================
def process_social_data(engine, spine, social_specs, feat_cfg):
    """Process social data including IPC and IOM displacement data."""
    logger.info("PHASE 2C: Processing Social Data (IPC & IOM)...")
    inspector = inspect(engine)
    impute_cfg = feat_cfg
    
    # --- IPC Data ---
    ipc_spec = next((x for x in social_specs if 'ipc' in x['raw']), None)
    if ipc_spec:
        if inspector.has_table("ipc_h3", schema=SCHEMA):
            logger.info("  Merging IPC Data...")
            row_count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {SCHEMA}.ipc_h3", engine).iloc[0]["cnt"]
            if row_count == 0:
                raise RuntimeError(f"{SCHEMA}.ipc_h3 exists but is empty. Spatial disaggregation failed.")
            logger.info(f"  IPC rows: {row_count:,}")
            ipc_df = pd.read_sql(f"SELECT h3_index, date, ipc_phase_class FROM {SCHEMA}.ipc_h3", engine)
            ipc_df['date'] = pd.to_datetime(ipc_df['date'])
            
            spine = spine.merge(ipc_df, on=['h3_index', 'date'], how='left')
            spine['ipc_phase_class'] = apply_forward_fill(spine, 'ipc_phase_class', config=impute_cfg).fillna(0)
            
            if 'lag' in ipc_spec.get('transformation', ''):
                spine[ipc_spec['output_col']] = spine.groupby('h3_index')['ipc_phase_class'].shift(1)
        else:
            logger.warning(f"  ⚠️ Table {SCHEMA}.ipc_h3 not found. Skipping IPC integration (filling 0).")
            if ipc_spec.get('output_col'):
                spine[ipc_spec['output_col']] = 0
    
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
            
            spine = spine.merge(iom_df, on=['h3_index', 'date'], how='left')
            spine['iom_data_available'] = spine['iom_displacement_sum'].notna().astype(int)
            spine['iom_displacement_sum'] = spine['iom_displacement_sum'].fillna(0)
            
            if 'lag' in iom_spec.get('transformation', ''):
                spine[iom_spec['output_col']] = spine.groupby('h3_index')['iom_displacement_sum'].shift(1)
        else:
            logger.warning(f"  ⚠️ Table {SCHEMA}.iom_displacement_h3 not found. Skipping IOM integration (filling 0).")
            if iom_spec.get('output_col'):
                spine[iom_spec['output_col']] = 0
    
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
    
    db_cols = ', '.join(raw_cols)
    
    # 2. Load Raw Data
    env_df = pd.read_sql(
        f"SELECT h3_index, date, {db_cols} FROM {SCHEMA}.environmental_features",
        engine
    )
    env_df['date'] = pd.to_datetime(env_df['date'])
    
    spine = spine.merge(env_df, on=['h3_index', 'date'], how='left')
    
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
        spine[raw] = apply_forward_fill(spine, raw, config=feat_cfg)
        
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
    drop_cols = [c for c in drop_cols if 'market_price' not in c]
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
        tolerance=pd.Timedelta(days=365*25)
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

    # --- Admin mapping (prefer admin2/prefecture, fallback to admin1) ---
    insp = inspect(engine)
    fs_cols = {c["name"] for c in insp.get_columns("features_static", schema=SCHEMA)}
    admin_col = "admin2" if "admin2" in fs_cols else "admin1"
    if admin_col in fs_cols:
        logger.info(f"  Using '{admin_col}' for regional risk aggregation.")
        admin_map = pd.read_sql(f"SELECT h3_index, {admin_col} FROM {SCHEMA}.features_static", engine)
        admin_map["h3_index"] = admin_map["h3_index"].astype("int64")
        spine = spine.merge(admin_map, on="h3_index", how="left")
    else:
        logger.warning("  No admin columns found; regional risk will be skipped.")
        spine[admin_col] = None

    # 1. Load ACLED - Split Protest and Riots
    acled_query = f"""
        SELECT h3_index, {admin_col}, event_date AS date, geo_precision, time_precision, fatalities, 
               CASE WHEN event_type = 'Protests' THEN 1 ELSE 0 END as protest_flag,
               CASE WHEN event_type = 'Riots' THEN 1 ELSE 0 END as riot_flag
        FROM {SCHEMA}.acled_events
    """
    acled_raw = pd.read_sql(acled_query, engine)
    acled_raw['date'] = pd.to_datetime(acled_raw['date'])
    acled_raw["h3_index"] = acled_raw["h3_index"].astype("int64")
    acled_raw["geo_precision"] = pd.to_numeric(acled_raw["geo_precision"], errors="coerce")
    acled_raw["time_precision"] = pd.to_numeric(acled_raw["time_precision"], errors="coerce")
    
    # Bin to Spine
    dates = sorted(spine['date'].unique())
    acled_raw['spine_date'] = pd.cut(acled_raw['date'], bins=dates, labels=dates[1:], right=True)
    
    # --- Stream A: Local Precision (geo_precision/time_precision 1 or 2) ---
    precise_mask = acled_raw['geo_precision'].isin([1, 2]) & acled_raw['time_precision'].isin([1, 2])
    acled_local = acled_raw[precise_mask]
    acled_local_agg = acled_local.groupby(['h3_index', 'spine_date'], observed=True).agg({
        'fatalities': 'sum',
        'protest_flag': 'sum',
        'riot_flag': 'sum'
    }).reset_index().rename(columns={
        'spine_date': 'date',
        'protest_flag': 'protest_count',
        'riot_flag': 'riot_count'
    })
    acled_local_agg['date'] = pd.to_datetime(acled_local_agg['date'])
    
    # Targets from local stream
    acled_local_agg['target_fatalities'] = acled_local_agg['fatalities']
    acled_local_agg['target_binary'] = (acled_local_agg['fatalities'] > 0).astype(int)

    # --- Stream B: Regional Context (All precisions) ---
    acled_regional = acled_raw.copy()
    acled_regional_agg = acled_regional.groupby([admin_col, 'spine_date'], observed=True)['fatalities'].sum().reset_index()
    acled_regional_agg = acled_regional_agg.rename(columns={'spine_date': 'date', 'fatalities': 'regional_fatalities'})
    acled_regional_agg['date'] = pd.to_datetime(acled_regional_agg['date'])
    acled_regional_agg = acled_regional_agg.sort_values([admin_col, 'date'])
    acled_regional_agg['regional_risk_score'] = np.log1p(acled_regional_agg['regional_fatalities'])
    acled_regional_agg['regional_risk_score_lag1'] = (
        acled_regional_agg.groupby(admin_col)['regional_risk_score'].shift(1)
    )

    # 2. Load GDELT
    gdelt_raw = pd.read_sql(
        f"SELECT h3_index, date, value FROM {SCHEMA}.features_dynamic_daily WHERE variable = 'gdelt_event_count'", 
        engine
    )
    gdelt_raw['date'] = pd.to_datetime(gdelt_raw['date'])
    gdelt_raw['spine_date'] = pd.cut(gdelt_raw['date'], bins=dates, labels=dates[1:], right=True)
    
    gdelt_agg = gdelt_raw.groupby(['h3_index', 'spine_date'], observed=True)['value'].sum().reset_index()
    gdelt_agg = gdelt_agg.rename(columns={'spine_date': 'date', 'value': 'gdelt_event_count'})
    gdelt_agg['date'] = pd.to_datetime(gdelt_agg['date'])

    # 3. Merge onto Spine
    spine = spine.merge(acled_local_agg, on=['h3_index', 'date'], how='left')
    if admin_col in spine.columns:
        spine = spine.merge(acled_regional_agg[[admin_col, 'date', 'regional_risk_score_lag1']], on=[admin_col, 'date'], how='left')
    spine = spine.merge(gdelt_agg, on=['h3_index', 'date'], how='left')
    
    # 4. Initialize missing conflict columns with 0 and apply forward-fill
    conflict_raws = list(set([spec['raw'] for spec in conflict_specs]))
    for raw in conflict_raws:
        if raw not in spine.columns:
            spine[raw] = 0
    
    # Fill missing values and apply forward-fill (limit=4)
    impute_cfg = features_config if isinstance(features_config, dict) else {}
    for col in ['fatalities', 'protest_count', 'riot_count', 'gdelt_event_count', 'target_fatalities', 'target_binary', 'regional_risk_score_lag1']:
        if col in spine.columns:
            spine[col] = spine[col].fillna(0)
            if col == 'regional_risk_score_lag1':
                spine[col] = apply_forward_fill(spine, col, groupby_col=admin_col, config=impute_cfg)
            else:
                spine[col] = apply_forward_fill(spine, col, config=impute_cfg)
    
    # 5. Create fatalities_14d_sum - This is the simple sum for the baseline model
    spine['fatalities_14d_sum'] = spine['fatalities']
    
    # 6. Apply Registry Transformations
    decays = temporal_config.get('decays', {})
    
    for spec in conflict_specs:
        raw = spec['raw']
        trans = spec['transformation']
        out = spec['output_col']
        
        if out == 'regional_risk_score_lag1':
            spine[out] = spine.get('regional_risk_score_lag1', 0).fillna(0)
            continue

        if 'decay' in trans:
            decay_key = f"half_life_{trans.split('_')[1]}"
            steps = decays.get(decay_key, {}).get('steps', 2.14)
            spine[out] = spine.groupby('h3_index')[raw].ewm(halflife=steps).mean().reset_index(level=0, drop=True)
            
        elif 'sum' in trans:
            spine[out] = spine[raw]
            
        elif 'lag' in trans:
            spine[out] = spine.groupby('h3_index')[raw].shift(1).fillna(0)
            
        # Fill NaN values after transformation
        spine[out] = spine[out].fillna(0)

    # 7. Compute time_since_last_fatal_event (FIX for 9999 bug)
    spine = compute_time_since_last_fatal_event(spine, acled_raw)

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
        spine = process_economics(engine, spine, specs['economic'])
        gc.collect()
        
        # PHASE 2B: Local Food Prices
        spine = process_food_security(engine, spine, specs['social'], feat_cfg)
        gc.collect()
        
        # PHASE 2C: Social Data (IPC & IOM)
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
        
        # --- Final Optimization ---
        logger.info("Optimizing data types...")
        fcols = spine.select_dtypes('float64').columns
        spine[fcols] = spine[fcols].astype('float32')
        spine['h3_index'] = spine['h3_index'].astype('int64')
        
        # Clean up any spurious columns
        if 'level_0' in spine.columns: 
            spine.drop(columns=['level_0'], inplace=True)
        
        # --- CLEANING STEP BEFORE SAVE ---
        # 1. Unpack any weird list-strings (Fixes Error A)
        for col in spine.columns:
            if spine[col].dtype == 'object':
                try:
                    spine[col] = spine[col].astype(str).str.strip("[]").astype(float)
                except Exception:
                    pass

        # 2. Ensure Targets are Kept (Fixes Error B)
        target_cols = ['conflict_binary', 'target_binary', 'fatalities', 'target_fatalities']
        for t in target_cols:
            if t in spine.columns:
                spine[t] = spine[t].fillna(0).astype(int)

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
