"""
pipeline/processing/feature_engineering.py
==========================================
Master Feature Engineering Pipeline (Phase 5).
CONFIG-DRIVEN: Controlled strictly by configs/features.yaml.

LOGIC:
1. SPINAL: Generates Master Spine based on config frequency.
2. REGISTRY: Only processes features explicitly defined in features.yaml registry.
3. TRANSFORMS: Dynamic mapping of 'anomaly', 'decay', 'lag' keywords to vector logic.
4. ROBUSTNESS: Checks for table existence; Casts types before merges.

OUTPUT:
- car_cewp.temporal_features (Target for EPR injection and Modeling)
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
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs

# --- Constants ---
SCHEMA = "car_cewp"
OUTPUT_TABLE = "temporal_features"

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
        'social': []
    }
    
    source_map = {
        'CHIRPS': 'environmental', 'ERA5': 'environmental', 
        'MODIS': 'environmental', 'VIIRS': 'environmental', 'JRC_Landsat': 'environmental',
        'ACLED': 'conflict', 'GDELT': 'conflict',
        'YahooFinance': 'economic', 'FEWS_NET': 'social', 'IOM_DTM': 'social'
    }

    for item in registry:
        source = item.get('source')
        category = source_map.get(source, 'other')
        if category in specs:
            specs[category].append(item)
            
    return specs

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
    
    # --- 2C. Social (IPC / IOM) ---
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

    return spine

# ==============================================================================
# PHASE 3: ENVIRONMENTAL DATA (Config-Driven Anomalies)
# ==============================================================================
def process_environment(engine, spine, env_specs):
    logger.info("PHASE 3: Processing Environmental Features (Config-Driven)...")
    
    if not env_specs: return spine

    # 1. Identify Columns to Load
    raw_cols = list(set([item['raw'] for item in env_specs]))
    db_cols = ', '.join(raw_cols)
    
    # 2. Load Raw Data
    env_df = pd.read_sql(
        f"SELECT h3_index, date, {db_cols} FROM {SCHEMA}.environmental_features",
        engine
    )
    env_df['date'] = pd.to_datetime(env_df['date'])
    
    spine = spine.merge(env_df, on=['h3_index', 'date'], how='left')
    
    # 3. Calculate Climatology (Baselines)
    env_df['year'] = env_df['date'].dt.year
    env_df['epoch'] = env_df.groupby('year')['date'].rank(method='dense').astype(int)
    
    for col in raw_cols:
        subset = env_df
        if 'ntl' in col:
            subset = env_df[env_df['year'] >= 2012]
            
        stats = subset.groupby(['h3_index', 'epoch'])[col].agg(['mean', 'std']).reset_index()
        stats.columns = ['h3_index', 'epoch', f"{col}_mean", f"{col}_std"]
        
        spine = spine.merge(stats, on=['h3_index', 'epoch'], how='left')
        
    # 4. Apply Transformations from Registry
    for spec in env_specs:
        raw = spec['raw']
        trans = spec['transformation']
        out = spec['output_col']
        
        spine[raw] = spine.groupby('h3_index')[raw].ffill(limit=4)
        
        if 'anomaly' in trans:
            anom_col = f"{raw}_anom"
            spine[anom_col] = spine[raw] - spine[f"{raw}_mean"]
            spine[out] = spine[anom_col]
            
        elif 'lag' in trans:
             spine[out] = spine.groupby('h3_index')[raw].shift(1)
             
        else:
             spine[out] = spine[raw]

    # Cleanup
    drop_cols = [c for c in spine.columns if c.endswith('_mean') or c.endswith('_std')]
    drop_cols = [c for c in drop_cols if 'market_price' not in c]
    spine.drop(columns=drop_cols, inplace=True)
    
    spine.drop(columns=['epoch'], inplace=True)
    return spine

# ==============================================================================
# PHASE 4: CONFLICT DATA (Config-Driven Decay)
# ==============================================================================
def process_conflict(engine, spine, conflict_specs, temporal_config):
    logger.info("PHASE 4: Processing Conflict (Config-Driven)...")
    
    if not conflict_specs: return spine

    # 1. Load ACLED
    acled_query = f"""
        SELECT h3_index, event_date AS date, fatalities, 
               CASE WHEN event_type IN ('Protests', 'Riots') THEN 1 ELSE 0 END as protest_riot
        FROM {SCHEMA}.acled_events
    """
    acled_raw = pd.read_sql(acled_query, engine)
    acled_raw['date'] = pd.to_datetime(acled_raw['date'])
    
    # Bin to Spine
    dates = sorted(spine['date'].unique())
    acled_raw['spine_date'] = pd.cut(acled_raw['date'], bins=dates, labels=dates[1:], right=True)
    
    # [FIXED] Use observed=True and explicit cast to datetime
    acled_agg = acled_raw.groupby(['h3_index', 'spine_date'], observed=True).agg({
        'fatalities': 'sum', 'protest_riot': 'sum'
    }).reset_index().rename(columns={'spine_date': 'date', 'protest_riot': 'protest_count'})
    acled_agg['date'] = pd.to_datetime(acled_agg['date'])
    
    # 2. Load GDELT
    gdelt_raw = pd.read_sql(f"SELECT h3_index, date, value FROM {SCHEMA}.features_dynamic_daily WHERE variable = 'gdelt_event_count'", engine)
    gdelt_raw['date'] = pd.to_datetime(gdelt_raw['date'])
    gdelt_raw['spine_date'] = pd.cut(gdelt_raw['date'], bins=dates, labels=dates[1:], right=True)
    
    # [FIXED] Use observed=True and explicit cast to datetime
    gdelt_agg = gdelt_raw.groupby(['h3_index', 'spine_date'], observed=True)['value'].sum().reset_index().rename(columns={'spine_date': 'date', 'value': 'gdelt_event_count'})
    gdelt_agg['date'] = pd.to_datetime(gdelt_agg['date'])

    # 3. Merge onto Spine
    spine = spine.merge(acled_agg, on=['h3_index', 'date'], how='left')
    spine = spine.merge(gdelt_agg, on=['h3_index', 'date'], how='left')
    
    # 4. Apply Registry Transformations
    decays = temporal_config.get('decays', {})
    
    for spec in conflict_specs:
        raw = spec['raw']
        trans = spec['transformation']
        out = spec['output_col']
        
        if raw not in spine.columns:
            spine[raw] = 0
        spine[raw] = spine[raw].fillna(0)

        if 'decay' in trans:
            decay_key = f"half_life_{trans.split('_')[1]}"
            steps = decays.get(decay_key, {}).get('steps', 2.14)
            spine[out] = spine.groupby('h3_index')[raw].ewm(halflife=steps).mean().reset_index(level=0, drop=True)
            
        elif 'sum' in trans:
            spine[out] = spine[raw]
            
        elif 'lag' in trans:
            spine[out] = spine.groupby('h3_index')[raw].shift(1).fillna(0)

    return spine

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run():
    logger.info("=" * 60)
    logger.info("MASTER FEATURE ENGINEERING (Config-Driven)")
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
        
        # Execution
        spine = create_master_spine(engine, start_date, end_date, step_days)
        gc.collect()
        
        spine = process_economy_and_food(engine, spine, specs['economic'], specs['social'])
        gc.collect()
        
        spine = process_environment(engine, spine, specs['environmental'])
        gc.collect()
        
        spine = process_conflict(engine, spine, specs['conflict'], feat_cfg['temporal'])
        gc.collect()
        
        # Final Save
        logger.info("Optimization & Upload...")
        fcols = spine.select_dtypes('float64').columns
        spine[fcols] = spine[fcols].astype('float32')
        spine['h3_index'] = spine['h3_index'].astype('int64')
        
        if 'level_0' in spine.columns: spine.drop(columns=['level_0'], inplace=True)
        
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{OUTPUT_TABLE}"))
            
        spine.to_sql(
            OUTPUT_TABLE, engine, schema=SCHEMA, if_exists='replace', index=False, 
            chunksize=10000, method='multi'
        )
        
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{OUTPUT_TABLE} ADD PRIMARY KEY (h3_index, date)"))
            conn.execute(text(f"CREATE INDEX idx_temp_date ON {SCHEMA}.{OUTPUT_TABLE} (date)"))
            
        logger.info(f"✅ FEATURE ENGINEERING COMPLETE. Saved to {SCHEMA}.{OUTPUT_TABLE}")
        
    except Exception as e:
        logger.critical(f"Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        engine.dispose()

if __name__ == "__main__":
    run()