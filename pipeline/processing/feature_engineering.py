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
5. POPULATION FIX: Added process_demographics to handle WorldPop log1p transformation.
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

from utils import logger, get_db_engine, load_configs, upload_to_postgis

# --- Constants ---
SCHEMA = "car_cewp"
OUTPUT_TABLE = "temporal_features"
PRIMARY_KEYS = ['h3_index', 'date']  # For upsert operations
CHUNK_SIZE = 50000  # For chunked uploads


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
        'demographic': []  # Added for WorldPop
    }
    
    source_map = {
        'CHIRPS': 'environmental', 'ERA5': 'environmental', 
        'MODIS': 'environmental', 'VIIRS': 'environmental', 'JRC_Landsat': 'environmental',
        'ACLED': 'conflict', 'GDELT': 'conflict',
        'YahooFinance': 'economic', 'FEWS_NET': 'social', 'IOM_DTM': 'social',
        'WorldPop': 'demographic'  # Added mapping
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
        
        if not inspector.has_table(OUTPUT_TABLE, schema=SCHEMA):
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
            
        else:
            # --- ADD MISSING COLUMNS TO EXISTING TABLE ---
            logger.info(f"Table {SCHEMA}.{OUTPUT_TABLE} exists. Checking for new columns...")
            
            existing_cols = {c['name'] for c in inspector.get_columns(OUTPUT_TABLE, schema=SCHEMA)}
            new_cols_added = 0
            
            for col in spine_sample.columns:
                if col not in existing_cols:
                    sql_type = _infer_sql_type(spine_sample[col].dtype)
                    alter_sql = f'ALTER TABLE {SCHEMA}.{OUTPUT_TABLE} ADD COLUMN IF NOT EXISTS "{col}" {sql_type}'
                    conn.execute(text(alter_sql))
                    new_cols_added += 1
                    logger.info(f"  Added new column: {col} ({sql_type})")
            
            if new_cols_added == 0:
                logger.info("  No new columns needed.")
            else:
                logger.info(f"  Added {new_cols_added} new columns.")
        
        # Create indexes for performance
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_date 
            ON {SCHEMA}.{OUTPUT_TABLE} (date)
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_h3 
            ON {SCHEMA}.{OUTPUT_TABLE} (h3_index)
        """))
    
    logger.info(f"✓ Table schema verified: {SCHEMA}.{OUTPUT_TABLE}")


# ==============================================================================
# PHASE 2: ECONOMIC & FOOD SECURITY (Hybrid: Config + Base Logic)
# ==============================================================================
def process_economy_and_food(engine, spine, econ_specs, social_specs):
    logger.info("PHASE 2: Processing Economy & Food Security...")
    inspector = inspect(engine)

    # --- 2A. Global Economy (YahooFinance) ---
    gold_spec = next((x for x in econ_specs if x['raw'] == 'commodity_gold_price_usd'), None)
    
    if gold_spec:
        if inspector.has_table("economic_drivers", schema=SCHEMA):
            logger.info("  Merging Global Economy (Gold)...")
            econ_df = pd.read_sql(
                f"SELECT date, commodity_gold_price_usd FROM {SCHEMA}.economic_drivers ORDER BY date", 
                engine
            )
            econ_df['date'] = pd.to_datetime(econ_df['date'])
            
            spine = spine.sort_values('date')
            spine = pd.merge_asof(
                spine, econ_df, on='date', direction='backward', tolerance=pd.Timedelta(days=14)
            )
            spine['commodity_gold_price_usd'] = spine['commodity_gold_price_usd'].ffill(limit=4)
            
            # Apply Transformation (e.g., Lag)
            trans = gold_spec.get('transformation', '')
            out_col = gold_spec.get('output_col')
            if 'lag' in trans:
                spine[out_col] = spine.groupby('h3_index')['commodity_gold_price_usd'].shift(1)
            else:
                spine[out_col] = spine['commodity_gold_price_usd']
        else:
            logger.warning(f"  ⚠️ Table {SCHEMA}.economic_drivers not found. Skipping Gold data.")
            if gold_spec.get('output_col'):
                spine[gold_spec['output_col']] = 0

    # --- 2B. Local Food Prices (Spatial Broadcast) ---
    logger.info("  Processing Local Food Prices (Spatial Voronoi)...")
    
    has_locs = inspector.has_table("market_locations", schema=SCHEMA)
    has_prices = inspector.has_table("food_security", schema=SCHEMA)

    if has_locs and has_prices:
        locs = pd.read_sql(f"SELECT market_id, market_name, latitude, longitude FROM {SCHEMA}.market_locations", engine)
        prices = pd.read_sql(
            f"SELECT date, market AS market_name, value FROM {SCHEMA}.food_security WHERE commodity IN ('Maize', 'Rice', 'Cassava')",
            engine
        )
        
        if not locs.empty and not prices.empty:
            # Voronoi Map
            unique_h3 = spine['h3_index'].unique()
            grid_points = np.fliplr(get_h3_centroids(unique_h3))
            market_points = locs[['longitude', 'latitude']].values
            tree = cKDTree(market_points)
            _, indices = tree.query(grid_points, k=1)
            mapped_markets = locs.iloc[indices]['market_name'].values
            
            h3_map = pd.DataFrame({'h3_index': unique_h3, 'nearest_market': mapped_markets})
            spine = spine.merge(h3_map, on='h3_index', how='left')
            
            # Merge Prices
            prices['date'] = pd.to_datetime(prices['date'])
            daily_prices = prices.groupby(['date', 'market_name'])['value'].mean().reset_index()
            daily_prices = daily_prices.sort_values('date')
            
            spine = spine.merge(
                daily_prices.rename(columns={'value': 'market_price_mean', 'market_name': 'nearest_market'}),
                on=['date', 'nearest_market'], how='left'
            )
            spine['market_price_mean'] = spine.groupby('h3_index')['market_price_mean'].ffill(limit=6).fillna(0)
            spine.drop(columns=['nearest_market'], inplace=True)
    else:
        logger.warning("  ⚠️ Market tables not found. Skipping Food Security features.")
    
    return spine


# ==============================================================================
# PHASE 2B: SOCIAL DATA (IPC & IOM)
# ==============================================================================
def process_social_data(engine, spine, social_specs):
    """Process social data including IPC and IOM displacement data."""
    logger.info("PHASE 2B: Processing Social Data (IPC & IOM)...")
    inspector = inspect(engine)
    
    # --- IPC Data ---
    ipc_spec = next((x for x in social_specs if 'ipc' in x['raw']), None)
    if ipc_spec:
        if inspector.has_table("ipc_h3", schema=SCHEMA):
            logger.info("  Merging IPC Data...")
            ipc_df = pd.read_sql(f"SELECT h3_index, date, ipc_phase_class FROM {SCHEMA}.ipc_h3", engine)
            ipc_df['date'] = pd.to_datetime(ipc_df['date'])
            
            spine = spine.merge(ipc_df, on=['h3_index', 'date'], how='left')
            spine['ipc_phase_class'] = spine.groupby('h3_index')['ipc_phase_class'].ffill(limit=6).fillna(0)
            
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
            # [FIXED] Correct column name is iom_displacement_sum, not displacement_count
            iom_df = pd.read_sql(
                f"SELECT h3_index, date, iom_displacement_sum FROM {SCHEMA}.iom_displacement_h3", 
                engine
            )
            iom_df['date'] = pd.to_datetime(iom_df['date'])
            
            # Merge onto spine
            spine = spine.merge(iom_df, on=['h3_index', 'date'], how='left')
            spine['iom_displacement_sum'] = spine['iom_displacement_sum'].fillna(0)
            
            # Apply lag transformation
            if 'lag' in iom_spec.get('transformation', ''):
                spine[iom_spec['output_col']] = spine.groupby('h3_index')['iom_displacement_sum'].shift(1)
        else:
            logger.warning(f"  ⚠️ Table {SCHEMA}.iom_displacement_h3 not found. Skipping IOM integration (filling 0).")
            if iom_spec.get('output_col'):
                spine[iom_spec['output_col']] = 0
    
    return spine


# ==============================================================================
# PHASE 2C: DEMOGRAPHICS (WorldPop)
# ==============================================================================
def process_demographics(engine, spine, demo_specs):
    """
    Merges annual population data and applies transformations (e.g., log1p).
    """
    logger.info("PHASE 2C: Processing Demographics (WorldPop)...")
    inspector = inspect(engine)
    
    # Check if we need to process population
    pop_spec = next((x for x in demo_specs if x['raw'] == 'pop_count'), None)
    
    if pop_spec and inspector.has_table("population_h3", schema=SCHEMA):
        logger.info("  Merging Population Data (Annual)...")
        
        # 1. Load Population Data
        pop_df = pd.read_sql(
            f"SELECT h3_index, year, pop_count FROM {SCHEMA}.population_h3", 
            engine
        )
        
        # 2. Merge on H3 + Year (Spine already has 'year' temporarily, or we re-derive it)
        spine['year'] = spine['date'].dt.year
        spine = spine.merge(pop_df, on=['h3_index', 'year'], how='left')
        
        # 3. Handle Missing Years (Forward Fill then Fill 0)
        # Sort to ensure fill works chronologically
        spine = spine.sort_values(['h3_index', 'date'])
        spine['pop_count'] = spine.groupby('h3_index')['pop_count'].ffill().fillna(0)
        
        # 4. Apply Transformations
        trans = pop_spec.get('transformation', '')
        out_col = pop_spec.get('output_col')
        
        if 'log1p' in trans:
            spine[out_col] = np.log1p(spine['pop_count'])
        else:
            spine[out_col] = spine['pop_count']
            
        # Cleanup
        spine.drop(columns=['year', 'pop_count'], inplace=True)
        
    elif pop_spec:
        logger.warning(f"  ⚠️ Table {SCHEMA}.population_h3 not found. Filling {pop_spec['output_col']} with 0.")
        if pop_spec.get('output_col'):
            spine[pop_spec['output_col']] = 0.0
            
    return spine


# ==============================================================================
# PHASE 3: ENVIRONMENTAL DATA (Config-Driven Anomalies)
# ==============================================================================
def process_environment(engine, spine, env_specs):
    logger.info("PHASE 3: Processing Environmental Features (Config-Driven)...")
    
    if not env_specs:
        return spine

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
        spine[raw] = spine.groupby('h3_index')[raw].ffill(limit=4)
        
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
def process_conflict(engine, spine, conflict_specs, temporal_config):
    logger.info("PHASE 4: Processing Conflict (Config-Driven)...")
    
    if not conflict_specs:
        return spine

    # 1. Load ACLED - Split Protest and Riots (P1-2)
    acled_query = f"""
        SELECT h3_index, event_date AS date, fatalities, 
               CASE WHEN event_type = 'Protests' THEN 1 ELSE 0 END as protest_flag,
               CASE WHEN event_type = 'Riots' THEN 1 ELSE 0 END as riot_flag
        FROM {SCHEMA}.acled_events
    """
    acled_raw = pd.read_sql(acled_query, engine)
    acled_raw['date'] = pd.to_datetime(acled_raw['date'])
    
    # Bin to Spine
    dates = sorted(spine['date'].unique())
    acled_raw['spine_date'] = pd.cut(acled_raw['date'], bins=dates, labels=dates[1:], right=True)
    
    # Aggregate with split counts (P1-2)
    acled_agg = acled_raw.groupby(['h3_index', 'spine_date'], observed=True).agg({
        'fatalities': 'sum',
        'protest_flag': 'sum',
        'riot_flag': 'sum'
    }).reset_index().rename(columns={
        'spine_date': 'date',
        'protest_flag': 'protest_count',
        'riot_flag': 'riot_count'
    })
    acled_agg['date'] = pd.to_datetime(acled_agg['date'])
    
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
    spine = spine.merge(acled_agg, on=['h3_index', 'date'], how='left')
    spine = spine.merge(gdelt_agg, on=['h3_index', 'date'], how='left')
    
    # 4. Initialize missing conflict columns with 0 and apply forward-fill
    conflict_raws = list(set([spec['raw'] for spec in conflict_specs]))
    for raw in conflict_raws:
        if raw not in spine.columns:
            spine[raw] = 0
    
    # Fill missing values and apply forward-fill (limit=4)
    for col in ['fatalities', 'protest_count', 'riot_count', 'gdelt_event_count']:
        if col in spine.columns:
            spine[col] = spine[col].fillna(0)
            spine[col] = spine.groupby('h3_index')[col].ffill(limit=4)
    
    # 5. Create fatalities_14d_sum (P0-2) - This is the simple sum for the baseline model
    spine['fatalities_14d_sum'] = spine['fatalities']
    
    # 6. Apply Registry Transformations
    decays = temporal_config.get('decays', {})
    
    for spec in conflict_specs:
        raw = spec['raw']
        trans = spec['transformation']
        out = spec['output_col']
        
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

    return spine


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run():
    logger.info("=" * 60)
    logger.info("MASTER FEATURE ENGINEERING (Config-Driven + Idempotent)")
    logger.info("=" * 60)
    
    engine = get_db_engine()
    
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
        
        spine = process_economy_and_food(engine, spine, specs['economic'], specs['social'])
        gc.collect()
        
        spine = process_social_data(engine, spine, specs['social'])
        gc.collect()
        
        # New call for Demographics
        spine = process_demographics(engine, spine, specs['demographic'])
        gc.collect()
        
        spine = process_environment(engine, spine, specs['environmental'])
        gc.collect()
        
        spine = process_conflict(engine, spine, specs['conflict'], feat_cfg['temporal'])
        gc.collect()
        
        # --- Final Optimization ---
        logger.info("Optimizing data types...")
        fcols = spine.select_dtypes('float64').columns
        spine[fcols] = spine[fcols].astype('float32')
        spine['h3_index'] = spine['h3_index'].astype('int64')
        
        # Clean up any spurious columns
        if 'level_0' in spine.columns: 
            spine.drop(columns=['level_0'], inplace=True)
        
        # --- Ensure Table Schema ---
        ensure_output_table_schema(engine, spine)
        
        # --- Upload in Chunks using Upsert (IDEMPOTENT) ---
        total_rows = len(spine)
        num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        logger.info(f"Uploading {total_rows:,} rows in {num_chunks} chunks of {CHUNK_SIZE:,}...")
        
        for chunk_num, start_idx in enumerate(range(0, total_rows, CHUNK_SIZE), start=1):
            end_idx = min(start_idx + CHUNK_SIZE, total_rows)
            chunk = spine.iloc[start_idx:end_idx].copy()
            
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