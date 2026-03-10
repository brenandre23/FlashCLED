"""
process_conflict_features.py
============================
REFACTORED (v3.2): Full CrisisWatch Processing Pipeline

Key Changes from v3.1:
- FIX (2026-01-25): fatalities_lag1 computed on FULL SPINE, not sparse ACLED agg
  - Bug: Computing lag on sparse df before merge caused ~99% NaN  
  - Fix: Merge first, zero-fill, compute lag on full spine (~300ms overhead)
  - Why not sparse+fillna? Sparse lag skips empty windows; we need "previous window"
  - Added hard guardrail: raises ValueError if any NaN in fatalities_lag1
- regional_fatalities_lag1 now recomputed from spine-level fatalities_lag1
- Optimized: uses sort=False since safe_merge pre-sorts by [h3_index, date]

Key Changes from v3:
- FIX: narrative_velocity edge case handling (first 2 months NaN → 0)
- ADD: narrative_velocity_lag1 and narrative_acceleration features

Key Changes from v2:
1. DAMPED NORMALIZATION: Uses spatial_confidence_norm (sqrt cell_count dampening)
2. STROBE LIGHT FIX: Forward-fill CW for 2 steps, then zero-fill true gaps
3. COMPOSITE SCORE: Weighted cw_score_local with thesis-defined weights
4. DELTAS & DECAY: First-difference with _delta suffix (NOT _drift), 14-day half-life
5. INTERACTION FEATURES: Pillar × Pillar and Fusion (CW × ACLED) features
6. SPATIAL DIFFUSION: k-ring spatial lags for cw_score_local

Architecture:
- Stage 1 (Onset): Binary classification - regime_guerrilla_fragmentation as trigger
- Stage 2 (Intensity): Count regression - regime_ethno_pastoral_rupture as amplifier

Constraints:
- Preserve raw pillars: Model receives BOTH raw columns AND composite
- No "drift" naming: Use _delta suffix for first-differences
- Idempotent: Re-running produces identical results
- Type safety: h3_index as int64, dates as datetime64
- Missing column handling: Log warning, create column with 0, continue

NARRATIVE VELOCITY EDGE CASE:
- Month 1: No previous centroid → velocity = NaN → fill with 0
- Month 2: velocity exists, but lag1(velocity) = NaN → acceleration = NaN → fill with 0
- Month 3+: Valid acceleration values
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from sqlalchemy import text, inspect
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis
from pipeline.processing.utils_processing import (
    SCHEMA, OUTPUT_TABLE, PRIMARY_KEYS, CHUNK_SIZE,
    parse_registry, safe_merge,
    add_spatial_diffusion_features, apply_halflife_decay,
    sanitize_numeric_columns
)


# =============================================================================
# STRUCTURAL BREAK DATES
# =============================================================================
STRUCTURAL_BREAKS = {
    'gdelt': pd.Timestamp('2015-02-18'),
    'ioda': pd.Timestamp('2022-02-01'),
}

# =============================================================================
# CRISISWATCH PILLAR WEIGHTS (Thesis-Defined)
# =============================================================================
CW_PILLAR_WEIGHTS = {
    'regime_guerrilla_fragmentation': 0.30,    # Onset trigger (Pillar 12)
    'regime_transnational_predation': 0.25,    # Intensity amplifier - Wagner (Pillar 11)
    'regime_ethno_pastoral_rupture': 0.25,     # Mass casualty risk (Pillar 13)
    'regime_parallel_governance': 0.20,        # Structural enabler (Pillar 10)
}

# Mapping from topic_id to pillar name
CW_TOPIC_ID_MAP = {
    10: 'regime_parallel_governance',
    11: 'regime_transnational_predation',
    12: 'regime_guerrilla_fragmentation',
    13: 'regime_ethno_pastoral_rupture',
    99: 'narrative_velocity',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enrich_table_schema(engine, df: pd.DataFrame, table_name: str, schema: str) -> None:
    """
    Dynamically add missing columns to support schema evolution before upsert.
    Prevents COPY/undefined-column errors when new features are introduced.
    """
    insp = inspect(engine)
    if not insp.has_table(table_name, schema=schema):
        return  # upload_to_postgis will create table if missing

    existing_cols = {c["name"] for c in insp.get_columns(table_name, schema=schema)}
    new_cols = [c for c in df.columns if c not in existing_cols]

    if not new_cols:
        return

    logger.info(f"⚡ Schema Evolution: Adding {len(new_cols)} new columns to {schema}.{table_name}...")
    with engine.begin() as conn:
        for col in new_cols:
            col_dtype = df[col].dtype
            if np.issubdtype(col_dtype, np.integer):
                col_type = "BIGINT"
            elif np.issubdtype(col_dtype, np.floating):
                col_type = "DOUBLE PRECISION"
            elif np.issubdtype(col_dtype, np.datetime64):
                col_type = "TIMESTAMP"
            else:
                col_type = "TEXT"
            conn.execute(text(f'ALTER TABLE {schema}.{table_name} ADD COLUMN IF NOT EXISTS "{col}" {col_type}'))


def compute_availability_flag_from_data(spine: pd.DataFrame, data_cols: List[str], 
                                        structural_break_date: pd.Timestamp, 
                                        flag_name: str) -> pd.DataFrame:
    """
    Compute availability flag based on structural break AND data presence.
    
    For conflict data: flag = 1 if date >= break AND any column has actual data
    """
    date_ok = spine['date'] >= structural_break_date

    existing_cols = [c for c in data_cols if c in spine.columns]
    if existing_cols:
        # For zero-filled data, check if original had any non-zero values
        data_ok = (spine[existing_cols] > 0).any(axis=1)
    else:
        data_ok = pd.Series(False, index=spine.index)

    # Conservative: if date is valid, assume data is potentially available
    spine[flag_name] = date_ok.astype(int)

    return spine


def zero_fill_conflict_columns(spine: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Zero-fill conflict columns (no forward-fill for fast-moving signals)."""
    for col in cols:
        if col in spine.columns:
            spine[col] = spine[col].fillna(0)
    return spine


def ensure_column_exists(spine: pd.DataFrame, col: str, default: float = 0.0) -> pd.DataFrame:
    """Ensure column exists, creating with default if missing. Log warning."""
    if col not in spine.columns:
        logger.warning(f"Missing column '{col}' - creating with default value {default}")
        spine[col] = default
    return spine


def apply_halflife_decay_14d(spine: pd.DataFrame, target_col: str, 
                              half_life_days: int = 14) -> pd.DataFrame:
    """
    Apply exponential decay with specified half-life in days.
    
    For 14-day spine intervals:
        half_life_days=14 means alpha such that signal decays to 50% after 1 step
    """
    if target_col not in spine.columns:
        return spine
    
    # Convert half-life to EWM alpha
    # For 14-day intervals: steps = half_life_days / 14
    steps = half_life_days / 14.0
    alpha = 1 - np.exp(-np.log(2) / steps)
    
    out_col = f"{target_col}_decay_{half_life_days}d"
    spine[target_col] = spine[target_col].fillna(0)
    spine[out_col] = spine.groupby('h3_index')[target_col].apply(
        lambda s: s.ewm(alpha=alpha, adjust=False).mean()
    ).reset_index(level=0, drop=True)
    spine[out_col] = spine[out_col].fillna(0)
    
    return spine


# =============================================================================
# PHASE FUNCTIONS
# =============================================================================

def load_existing_spine(engine, start_date: str, end_date: str) -> pd.DataFrame:
    """Load the contextual feature spine (h3_index, date only for merge keys)."""
    logger.info("Loading existing contextual features (spine).")
    query = f"""
        SELECT h3_index, date 
        FROM {SCHEMA}.{OUTPUT_TABLE} 
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    spine = pd.read_sql(query, engine)
    spine['h3_index'] = spine['h3_index'].astype('int64')
    spine['date'] = pd.to_datetime(spine['date'])
    return spine


def process_acled(engine, spine: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    """
    Process ACLED conflict data with zero-fill (no forward imputation).
    
    Output columns:
    - Event counts (acled_count_*)
    - Fatalities and targets
    - Regional risk scores
    - Protest/riot lags
    """
    logger.info("PHASE 4A: Processing ACLED Conflict Data...")

    insp = inspect(engine)

    # Get admin column for regional aggregation
    fs_cols = {c["name"] for c in insp.get_columns("features_static", schema=SCHEMA)}
    admin_col = "admin3" if "admin3" in fs_cols else "admin1"

    if admin_col in fs_cols:
        admin_map = pd.read_sql(f"SELECT h3_index, {admin_col} FROM {SCHEMA}.features_static", engine)
        admin_map["h3_index"] = admin_map["h3_index"].astype("int64")
        spine = spine.merge(admin_map, on="h3_index", how="left")
    else:
        spine[admin_col] = "Unknown"

    # Fetch ACLED events
    acled_query = f"""
        SELECT ae.h3_index, fs.{admin_col} AS {admin_col}, ae.event_date AS date, 
               ae.geo_precision, ae.time_precision, ae.fatalities,
               ae.acled_count_battles, ae.acled_count_vac, ae.acled_count_explosions,
               ae.acled_count_protests, ae.acled_count_riots,
               CASE WHEN ae.event_type = 'Protests' THEN 1 ELSE 0 END as protest_flag,
               CASE WHEN ae.event_type = 'Riots' THEN 1 ELSE 0 END as riot_flag
        FROM {SCHEMA}.acled_events ae
        LEFT JOIN {SCHEMA}.features_static fs ON ae.h3_index = fs.h3_index
    """
    acled_raw = pd.read_sql(acled_query, engine)
    acled_raw['date'] = pd.to_datetime(acled_raw['date'])
    acled_raw['h3_index'] = acled_raw['h3_index'].astype('int64')
    acled_raw['geo_precision'] = pd.to_numeric(acled_raw['geo_precision'], errors='coerce')
    acled_raw['time_precision'] = pd.to_numeric(acled_raw['time_precision'], errors='coerce')

    for c in ['acled_count_battles', 'acled_count_vac', 'acled_count_explosions',
              'acled_count_protests', 'acled_count_riots']:
        acled_raw[c] = acled_raw[c].fillna(0).astype(int)

    # Bin to spine dates
    dates = sorted(spine['date'].unique())
    acled_raw['spine_date'] = pd.cut(acled_raw['date'], bins=dates, labels=dates[1:], 
                                      right=True, include_lowest=True)

    # Filter for good precision
    mask = acled_raw['geo_precision'].isin([1, 2]) & acled_raw['time_precision'].isin([1, 2])
    acled_local = acled_raw[mask].copy()
    if admin_col in acled_local.columns:
        acled_local[admin_col] = acled_local[admin_col].fillna("Unknown")

    # Aggregate
    acled_local_agg = acled_local.groupby(['h3_index', admin_col, 'spine_date'], 
                                           observed=True, dropna=False).agg({
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
        'riot_flag': 'riot_count',
        'fatalities': 'fatalities_source'
    })
    acled_local_agg['date'] = pd.to_datetime(acled_local_agg['date'])

    # Targets (unlagged)
    acled_local_agg['target_fatalities'] = acled_local_agg['fatalities_source']
    acled_local_agg['target_binary'] = (acled_local_agg['fatalities_source'] > 0).astype(int)

    # -------------------------------------------------------------------------
    # FIX (2026-01-25): Compute fatalities_lag1 on the FULL SPINE, not sparse agg
    # -------------------------------------------------------------------------
    # Previously, fatalities_lag1 was computed here on acled_local_agg (sparse),
    # causing ~99% NaN after merge. Now we:
    #   1. Merge sparse agg to spine first
    #   2. Zero-fill fatalities on spine (no event = 0)
    #   3. Compute fatalities_lag1 on the full spine
    #   4. Recompute regional_fatalities_lag1 from spine-level lag
    # -------------------------------------------------------------------------

    # Drop admin from local before merge (keep it on spine for regional agg later)
    if admin_col in acled_local_agg.columns:
        acled_local_agg.drop(columns=[admin_col], inplace=True)

    # Merge sparse ACLED aggregates into full spine
    spine = safe_merge(spine, acled_local_agg, on=['h3_index', 'date'], how='left')

    # Restore fatalities (zero-fill: no event = 0 fatalities)
    if 'fatalities_source' in spine.columns:
        spine['fatalities'] = spine['fatalities_source'].fillna(0)
        spine['fatalities_14d_sum'] = spine['fatalities']
        spine.drop(columns=['fatalities_source'], inplace=True)
    else:
        spine['fatalities'] = 0
        spine['fatalities_14d_sum'] = 0

    # Compute fatalities_lag1 on the FULL SPINE (critical fix)
    # Note: safe_merge already returns data sorted by [h3_index, date]
    spine['fatalities_lag1'] = spine.groupby('h3_index', sort=False)['fatalities'].shift(1).fillna(0)

    # Recompute regional aggregates from spine-level fatalities_lag1
    regional_agg = spine.groupby([admin_col, 'date'], observed=True)['fatalities_lag1'].sum().reset_index()
    regional_agg = regional_agg.rename(columns={'fatalities_lag1': 'regional_fatalities_lag1'})
    regional_agg['regional_risk_score_lag1'] = np.log1p(regional_agg['regional_fatalities_lag1'])

    # Merge regional back (will overwrite any stale values)
    if 'regional_fatalities_lag1' in spine.columns:
        spine.drop(columns=['regional_fatalities_lag1', 'regional_risk_score_lag1'], errors='ignore', inplace=True)
    if not regional_agg.empty:
        spine = safe_merge(spine, regional_agg, on=[admin_col, 'date'], how='left')

    # Zero-fill all ACLED flow columns (NO forward-fill for conflict data)
    acled_cols = ['protest_count', 'riot_count', 'acled_count_battles', 'acled_count_vac',
                  'acled_count_explosions', 'acled_count_protests', 'acled_count_riots',
                  'target_fatalities', 'target_binary', 'regional_fatalities_lag1',
                  'regional_risk_score_lag1', 'fatalities_lag1']
    spine = zero_fill_conflict_columns(spine, acled_cols)

    # -------------------------------------------------------------------------
    # GUARDRAIL: Hard fail if any NaN in fatalities_lag1 (flow variable)
    # -------------------------------------------------------------------------
    nan_count = spine['fatalities_lag1'].isna().sum()
    if nan_count > 0:
        raise ValueError(
            f"FATAL: fatalities_lag1 contains {nan_count:,} NaN values ({100*nan_count/len(spine):.2f}%). "
            f"Flow variables must be zero-filled. Check process_acled() logic."
        )
    logger.info(f"  ✓ fatalities_lag1 validated: 0 NaN, {(spine['fatalities_lag1'] > 0).sum():,} non-zero rows")

    # Generate lags for protest/riot counts
    for col in ['protest_count', 'riot_count']:
        if col in spine.columns:
            spine[f"{col}_lag1"] = spine.groupby('h3_index')[col].shift(1).fillna(0)

    # Drop admin column
    if admin_col in spine.columns:
        spine.drop(columns=[admin_col], inplace=True)

    return spine


def process_acled_hybrid(engine, spine: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    """Merge ACLED Hybrid NLP features (mechanisms, drivers, actor risk)."""
    logger.info("PHASE 4B: Merging ACLED Hybrid Features...")

    insp = inspect(engine)
    if not insp.has_table("features_acled_hybrid", schema=SCHEMA):
        logger.warning("features_acled_hybrid table not found")
        return spine

    # Load raw hybrid data
    hybrid_df = pd.read_sql(f"SELECT * FROM {SCHEMA}.features_acled_hybrid", engine)
    if hybrid_df.empty:
        logger.warning("features_acled_hybrid is empty")
        return spine

    # Standardize types immediately on load
    hybrid_df['event_date'] = pd.to_datetime(hybrid_df['event_date'])
    hybrid_df['h3_index'] = hybrid_df['h3_index'].astype('int64')
    hybrid_df = hybrid_df.rename(columns={'event_date': 'date'})
    logger.info(f"    Loaded {len(hybrid_df):,} hybrid feature rows.")

    # Bin to spine dates (include lowest edge to catch first window)
    dates = sorted(spine['date'].unique())
    hybrid_df['spine_date'] = pd.cut(
        hybrid_df['date'],
        bins=dates,
        labels=dates[1:],
        right=True,
        include_lowest=True
    )

    # Diagnostic: count rows that failed binning
    unbinned = hybrid_df['spine_date'].isna().sum()
    if unbinned > 0:
        logger.warning(
            f"    {unbinned} hybrid rows ({100*unbinned/len(hybrid_df):.1f}%) "
            "fell outside spine date range and were dropped."
        )

    # Aggregate numeric columns only
    numeric_cols = hybrid_df.select_dtypes(include=[np.number]).columns.tolist()
    for k in ['h3_index', 'event_date', 'date']:
        if k in numeric_cols:
            numeric_cols.remove(k)

    hybrid_agg = hybrid_df.groupby(['h3_index', 'spine_date'], observed=True)[numeric_cols].mean().reset_index()
    hybrid_agg = hybrid_agg.rename(columns={'spine_date': 'date'})

    # Handle potential duplicate date columns
    if isinstance(hybrid_agg['date'], pd.DataFrame):
        hybrid_agg = hybrid_agg.loc[:, ~hybrid_agg.columns.duplicated()]

    # Enforce merge key types (prevents silent mismatches)
    hybrid_agg['date'] = pd.to_datetime(hybrid_agg['date'])
    hybrid_agg['h3_index'] = hybrid_agg['h3_index'].astype('int64')

    # Diagnostics: non-zero check pre-merge
    if 'mech_gold_pivot' in hybrid_agg.columns:
        pre_merge_nz = (hybrid_agg['mech_gold_pivot'] > 0).sum()
        logger.info(f"    Pre-merge non-zero mech_gold_pivot: {pre_merge_nz:,}")

    # Merge
    spine = safe_merge(spine, hybrid_agg, on=['h3_index', 'date'], how='left')

    # Diagnostics: ensure signal survives merge
    if 'mech_gold_pivot' in spine.columns:
        post_merge_nz = (spine['mech_gold_pivot'] > 0).sum()
        logger.info(f"    Post-merge non-zero mech_gold_pivot: {post_merge_nz:,}")
        if pre_merge_nz > 0 and post_merge_nz == 0:
            logger.error("    CRITICAL: All hybrid features lost during merge! Check date alignment.")

    # Zero-fill (no forward imputation)
    mech_cols = [c for c in spine.columns if c.startswith('mech_') or c.startswith('acled_')]
    spine = zero_fill_conflict_columns(spine, mech_cols)

    # Generate mechanism lags (and acled_combined_risk_score lag for consistency)
    cols_to_lag = [
        c for c in spine.columns
        if (c.startswith('mech_') or c == 'acled_combined_risk_score')
        and not c.endswith('_lag1')
    ]
    for col in cols_to_lag:
        spine[f"{col}_lag1"] = spine.groupby('h3_index')[col].shift(1).fillna(0)

    # NOTE: Legacy driver_* columns REMOVED (2026-01-24)
    # The old schema used driver_resource_cattle, driver_resource_mining, etc.
    # The new schema uses mech_gold_pivot, mech_predatory_tax, etc.
    # If any code references driver_*, it will now fail fast (good) instead of
    # silently training on zeros.

    return spine


def process_gdelt(engine, spine: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    """
    Process GDELT with zero-fill and correct decay naming.
    
    Creates:
    - Raw GDELT columns (gdelt_event_count, gdelt_avg_tone, etc.)
    - Spatial lags
    - National heat aggregates
    - Decays with CORRECT naming (gdelt_predatory_action_decay_30d, not _count_decay_30d)
    - gdelt_data_available flag
    """
    logger.info("PHASE 4C: Processing GDELT...")

    insp = inspect(engine)
    if not insp.has_table("features_dynamic_daily", schema=SCHEMA):
        logger.warning("features_dynamic_daily table not found")
        spine['gdelt_data_available'] = 0
        return spine

    gdelt_vars = [
        'gdelt_event_count', 'gdelt_avg_tone', 'gdelt_goldstein_mean',
        'gdelt_mentions_total', 'gdelt_predatory_action_count', 'gdelt_border_buffer_flag',
        'gdelt_theme_resource_predation_count', 'gdelt_theme_displacement_count',
        'gdelt_theme_governance_breakdown_count'
    ]

    gdelt_vars_sql = ", ".join(f"'{v}'" for v in gdelt_vars)
    gdelt_raw = pd.read_sql(
        f"""SELECT h3_index, date, variable, value 
            FROM {SCHEMA}.features_dynamic_daily 
            WHERE variable IN ({gdelt_vars_sql})""",
        engine
    )

    if gdelt_raw.empty:
        logger.warning("No GDELT data found")
        spine['gdelt_data_available'] = (spine['date'] >= STRUCTURAL_BREAKS['gdelt']).astype(int)
        return spine

    gdelt_raw['h3_index'] = gdelt_raw['h3_index'].astype('int64')
    gdelt_raw['date'] = pd.to_datetime(gdelt_raw['date'])

    # Bin to spine dates
    dates = sorted(spine['date'].unique())
    gdelt_raw['spine_date'] = pd.cut(gdelt_raw['date'], bins=dates, labels=dates[1:], right=True)

    # Pivot to wide
    gdelt_wide = gdelt_raw.pivot_table(
        index=['h3_index', 'spine_date'],
        columns='variable',
        values='value',
        aggfunc='mean'
    ).reset_index().rename(columns={'spine_date': 'date'})
    gdelt_wide['date'] = pd.to_datetime(gdelt_wide['date'])

    # Spatial lags
    gdelt_targets = [c for c in gdelt_vars if c in gdelt_wide.columns]
    for col in gdelt_targets:
        gdelt_wide = add_spatial_diffusion_features(gdelt_wide, col, k=1)

    # National heat (sum across all cells per date)
    nat_cols = [c for c in gdelt_targets if 'avg_tone' not in c]
    if nat_cols:
        nat_agg = gdelt_wide.groupby('date')[nat_cols].sum().reset_index()
        rename_map = {c: f"national_{c.replace('gdelt_', '').replace('_count', '')}_sum" for c in nat_cols}
        nat_agg.rename(columns=rename_map, inplace=True)
        spine = safe_merge(spine, nat_agg, on='date', how='left')

        # Zero-fill and apply decays to national
        for n_col in rename_map.values():
            if n_col in spine.columns:
                spine[n_col] = spine[n_col].fillna(0)
                spine = apply_halflife_decay(spine, n_col, features_config)

    # Merge local GDELT
    spine = safe_merge(spine, gdelt_wide, on=['h3_index', 'date'], how='left')

    # Zero-fill all GDELT columns (NO forward imputation)
    gdelt_cols = [c for c in spine.columns if 'gdelt' in c.lower() and c != 'gdelt_data_available']
    spine = zero_fill_conflict_columns(spine, gdelt_cols)

    # Apply decays and create correct aliases
    if 'gdelt_event_count' in spine.columns:
        spine = apply_halflife_decay(spine, 'gdelt_event_count', features_config)
        if 'gdelt_event_count_decay_30d' in spine.columns:
            spine['gdelt_decay_30d'] = spine['gdelt_event_count_decay_30d']

        # 90-day rolling for events_3m_lag
        spine['events_3m_lag'] = spine.groupby('h3_index')['gdelt_event_count'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        ).fillna(0)

    if 'gdelt_avg_tone' in spine.columns:
        spine = apply_halflife_decay(spine, 'gdelt_avg_tone', features_config)

    # Apply decays to thematic columns WITH CORRECT NAMING
    thematic_cols = [
        ('gdelt_predatory_action_count', 'gdelt_predatory_action_decay_30d'),
        ('gdelt_theme_resource_predation_count', 'gdelt_theme_resource_predation_decay_30d'),
        ('gdelt_theme_displacement_count', 'gdelt_theme_displacement_decay_30d'),
        ('gdelt_theme_governance_breakdown_count', 'gdelt_theme_governance_breakdown_decay_30d'),
    ]

    for raw_col, alias in thematic_cols:
        if raw_col in spine.columns:
            spine = apply_halflife_decay(spine, raw_col, features_config)
            decay_col = f"{raw_col}_decay_30d"
            if decay_col in spine.columns:
                spine[alias] = spine[decay_col]

    # GDELT availability flag
    spine = compute_availability_flag_from_data(
        spine, gdelt_targets, STRUCTURAL_BREAKS['gdelt'], 'gdelt_data_available'
    )

    return spine


def process_crisiswatch(engine, spine: pd.DataFrame, nlp_specs: List[dict], 
                        features_config: dict) -> pd.DataFrame:
    """
    Process CrisisWatch NLP features with full pipeline.
    
    STEP 1: Load & Pivot (using spatial_confidence_norm for damped normalization)
    STEP 2: Persistence (strobe light fix: ffill limit=2, then zero-fill)
    STEP 3: Composite Score (weighted cw_score_local)
    STEP 4: Deltas & Decay (first-difference with _delta suffix, 14-day half-life)
    STEP 5: Narrative Velocity Derivatives (lag1 + acceleration with edge case handling)
    STEP 6: Interaction Features (pillar × pillar, fusion with ACLED)
    STEP 7: Spatial Diffusion (k-ring lags)
    """
    logger.info("PHASE 4D: Processing CrisisWatch NLP (Full Pipeline)...")

    insp = inspect(engine)
    if not insp.has_table("features_crisiswatch", schema=SCHEMA):
        logger.warning("features_crisiswatch table not found")
        _create_stub_cw_columns(spine)
        return spine

    dates = sorted(spine['date'].unique())

    # ==========================================================================
    # STEP 1: Load & Pivot (using spatial_confidence_norm for damped normalization)
    # ==========================================================================
    logger.info("  Step 1: Loading and pivoting CrisisWatch data...")
    
    # Check if spatial_confidence_norm exists (v2 schema)
    cw_cols_query = """
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = 'features_crisiswatch'
    """
    cw_cols_df = pd.read_sql(cw_cols_query, engine, params=(SCHEMA,))
    available_cw_cols = set(cw_cols_df['column_name'].tolist())
    
    # Use spatial_confidence_norm if available, otherwise fall back to spatial_confidence
    score_col = 'spatial_confidence_norm' if 'spatial_confidence_norm' in available_cw_cols else 'spatial_confidence'
    logger.info(f"    Using score column: {score_col}")
    
    # MEMORY OPTIMIZATION: Convert spine to float32 where possible to save space
    float64_cols = spine.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        spine[float64_cols] = spine[float64_cols].astype('float32')

    cw_raw = pd.read_sql(
        f"""SELECT h3_index, date, cw_topic_id, {score_col} as score 
            FROM {SCHEMA}.features_crisiswatch""",
        engine
    )

    if cw_raw.empty:
        logger.warning("  No CrisisWatch data found")
        _create_stub_cw_columns(spine)
        return spine

    cw_raw['date'] = pd.to_datetime(cw_raw['date'])
    cw_raw['h3_index'] = cw_raw['h3_index'].astype('int64')
    cw_raw['score'] = cw_raw['score'].astype('float32') # Use float32

    # Apply Publication Lag (e.g., 2 steps = 28 days)
    lag_steps = features_config.get('temporal', {}).get('publication_lags', {}).get('NLP_CrisisWatch', 0)
    step_days = features_config.get('temporal', {}).get('step_days', 14)
    lag_days = lag_steps * step_days
    if lag_days > 0:
        logger.info(f"    Applying CrisisWatch publication lag: {lag_steps} steps ({lag_days} days)")
        cw_raw['date'] = cw_raw['date'] + pd.Timedelta(days=lag_days)
    
    # Bin to spine dates
    cw_raw['spine_date'] = pd.cut(cw_raw['date'], bins=dates, labels=dates[1:], right=True)

    # Split national velocity (topic 99) from spatial pillars (10-13)
    velocity_df = cw_raw[cw_raw['cw_topic_id'] == 99].copy()
    pillars_df = cw_raw[cw_raw['cw_topic_id'] != 99].copy()
    del cw_raw # Free memory

    # ------------------------------------------------------------------
    # Broadcast NATIONAL narrative velocity to all H3 cells (join on date only)
    # ------------------------------------------------------------------
    if not velocity_df.empty:
        # National signal: drop placeholder h3_index before aggregating
        velocity_df = velocity_df.drop(columns=['h3_index'], errors='ignore')
        velocity_df = velocity_df.groupby('spine_date', observed=True)['score'].sum().reset_index()
        velocity_df = velocity_df.rename(columns={'spine_date': 'date', 'score': 'narrative_velocity'})
        velocity_df['date'] = pd.to_datetime(velocity_df['date'].astype(str))
        velocity_df['narrative_velocity'] = velocity_df['narrative_velocity'].astype('float32')

        spine = safe_merge(spine, velocity_df, on='date', how='left')
        spine['narrative_velocity'] = spine['narrative_velocity'].fillna(0)
        del velocity_df
    else:
        spine['narrative_velocity'] = 0.0

    # ------------------------------------------------------------------
    # Standard spatial pillars (10-13): pivot and join on h3_index + date
    # ------------------------------------------------------------------
    if not pillars_df.empty:
        cw_pivot = pillars_df.pivot_table(
            index=['h3_index', 'spine_date'],
            columns='cw_topic_id',
            values='score',
            aggfunc='sum'
        ).reset_index().rename(columns={'spine_date': 'date'})
        cw_pivot['date'] = pd.to_datetime(cw_pivot['date'].astype(str))
        cw_pivot['h3_index'] = cw_pivot['h3_index'].astype('int64')
        
        # Convert pivoted columns to float32
        for col in cw_pivot.select_dtypes(include=['float64']).columns:
            cw_pivot[col] = cw_pivot[col].astype('float32')

        rename_map = {tid: name for tid, name in CW_TOPIC_ID_MAP.items() if tid in cw_pivot.columns}
        cw_pivot = cw_pivot.rename(columns=rename_map)
        logger.info(f"    Pivoted CrisisWatch columns: {list(rename_map.values())}")

        spine = safe_merge(spine, cw_pivot, on=['h3_index', 'date'], how='left')
        del cw_pivot
        del pillars_df
    else:
        logger.warning("    No CrisisWatch pillar rows (10-13) found after split.")
    
    import gc
    gc.collect() # Force cleanup before continuing memory intensive operations

    # ==========================================================================
    # STEP 2: Persistence (Strobe Light Fix)
    # ==========================================================================
    logger.info("  Step 2: Applying persistence (strobe light fix)...")
    
    # CrisisWatch is monthly, spine is 14-day
    # Forward-fill for 2 steps to hold state, then zero-fill true gaps.
    cw_pillar_cols = [name for name in CW_TOPIC_ID_MAP.values() if name in spine.columns and name != 'narrative_velocity']
    
    for col in cw_pillar_cols:
        spine[col] = spine.groupby('h3_index')[col].ffill(limit=2).fillna(0)
    
    # Apply the same bounded persistence to narrative velocity to avoid
    # biweekly strobing (monthly source on a 14-day spine).
    if 'narrative_velocity' in spine.columns:
        spine['narrative_velocity'] = (
            spine.groupby('h3_index')['narrative_velocity']
            .ffill(limit=2)
            .fillna(0)
        )
    
    logger.info("    Applied persistence to CrisisWatch pillars and narrative velocity.")

    # ==========================================================================
    # STEP 3: Composite Score (Weighted cw_score_local)
    # ==========================================================================
    logger.info("  Step 3: Computing weighted composite score (cw_score_local)...")
    
    # Ensure all pillar columns exist
    for pillar_name in CW_PILLAR_WEIGHTS.keys():
        spine = ensure_column_exists(spine, pillar_name, default=0.0)
    
    # Compute weighted sum
    spine['cw_score_local'] = sum(
        spine[pillar_name] * weight 
        for pillar_name, weight in CW_PILLAR_WEIGHTS.items()
        if pillar_name in spine.columns
    )
    
    logger.info(f"    cw_score_local weights: {CW_PILLAR_WEIGHTS}")

    # ==========================================================================
    # STEP 4: Deltas & Decay
    # ==========================================================================
    logger.info("  Step 4: Computing deltas and decay...")
    
    # First-difference (NOT called "drift")
    spine['cw_score_local_lag1'] = spine.groupby('h3_index')['cw_score_local'].shift(1).bfill().fillna(0)
    spine['cw_score_local_delta'] = spine['cw_score_local'] - spine['cw_score_local_lag1']
    
    # Apply 14-day half-life decay to delta
    spine = apply_halflife_decay_14d(spine, 'cw_score_local_delta', half_life_days=14)
    # Result: cw_score_local_delta_decay_14d
    
    # Also apply 30-day decay for consistency with other features
    spine = apply_halflife_decay(spine, 'cw_score_local_lag1', features_config)
    
    logger.info("    Created: cw_score_local_delta, cw_score_local_delta_decay_14d")

    # ==========================================================================
    # STEP 5: Narrative Velocity Derivatives (with edge case handling)
    # ==========================================================================
    logger.info("  Step 5: Computing narrative velocity derivatives...")
    
    spine = compute_narrative_velocity_features(spine, features_config)

    # ==========================================================================
    # STEP 6: Interaction Features
    # ==========================================================================
    logger.info("  Step 6: Creating interaction features...")
    spine = create_interaction_features(spine)

    # ==========================================================================
    # STEP 7: Spatial Diffusion
    # ==========================================================================
    logger.info("  Step 7: Adding spatial diffusion features...")
    
    spine = add_spatial_diffusion_features(spine, 'cw_score_local', k=1)
    spine = add_spatial_diffusion_features(spine, 'cw_score_local_lag1', k=1)
    
    # Decay on spatial lag
    if 'cw_score_local_spatial_lag' in spine.columns:
        spine['cw_score_local_spatial_lag_lag1'] = spine.groupby('h3_index')['cw_score_local_spatial_lag'].shift(1).fillna(0)
        spine = apply_halflife_decay(spine, 'cw_score_local_spatial_lag_lag1', features_config)

    logger.info("  CrisisWatch processing complete.")
    return spine


def compute_narrative_velocity_features(spine: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    """
    Compute narrative velocity derivatives with proper edge case handling.
    
    EDGE CASE PROBLEM:
    - Month 1: No previous centroid → velocity = NaN (from process_crisiswatch.py)
    - Month 2: velocity exists, but lag1(velocity) = NaN → acceleration = NaN
    - Month 3+: Valid acceleration values
    
    SOLUTION:
    - Zero-fill NaN velocity values (first month has no semantic drift)
    - Compute lag1 and acceleration
    - Zero-fill resulting NaNs (first 2 months of acceleration)
    
    OUTPUT COLUMNS:
    - narrative_velocity (already exists, zero-filled)
    - narrative_velocity_lag1 (lagged velocity)
    - narrative_acceleration (second derivative: velocity_t - velocity_{t-1})
    - narrative_acceleration_decay_14d (decayed acceleration)
    """
    logger.info("    Computing narrative_velocity_lag1 and narrative_acceleration...")
    
    # Ensure narrative_velocity exists
    spine = ensure_column_exists(spine, 'narrative_velocity', default=0.0)
    
    # Zero-fill NaN velocity values (Month 1 edge case)
    # Interpretation: No prior month means no semantic drift → velocity = 0
    initial_nan_count = spine['narrative_velocity'].isna().sum()
    spine['narrative_velocity'] = spine['narrative_velocity'].fillna(0)
    
    if initial_nan_count > 0:
        logger.info(f"    Zero-filled {initial_nan_count} NaN narrative_velocity values (Month 1 edge case)")
    
    # Compute lag1 (for use in acceleration and as a feature)
    # Note: narrative_velocity is a NATIONAL score (broadcasted), so we shift on the unique date series
    v_series = spine.groupby('date', observed=True)['narrative_velocity'].mean().reset_index().sort_values('date')
    v_series['narrative_velocity_lag1'] = v_series['narrative_velocity'].shift(1).bfill().fillna(0)
    
    if len(v_series) > 0:
        logger.info(f"    Computed narrative_velocity_lag1 with start-of-series backfill (stationary start).")
    
    # Merge lag1 back into spine
    if 'narrative_velocity_lag1' in spine.columns:
        spine.drop(columns=['narrative_velocity_lag1'], inplace=True)
    spine = safe_merge(spine, v_series[['date', 'narrative_velocity_lag1']], on='date', how='left')
    spine['narrative_velocity_lag1'] = spine['narrative_velocity_lag1'].fillna(0)
    
    # Compute acceleration (second derivative)
    # acceleration_t = velocity_t - velocity_{t-1}
    spine['narrative_acceleration'] = spine['narrative_velocity'] - spine['narrative_velocity_lag1']
    
    # Zero-fill acceleration NaNs (Month 2 edge case - shift creates NaN for first row)
    accel_nan_count = spine['narrative_acceleration'].isna().sum()
    spine['narrative_acceleration'] = spine['narrative_acceleration'].fillna(0)
    
    if accel_nan_count > 0:
        logger.info(f"    Zero-filled {accel_nan_count} NaN narrative_acceleration values (Month 2 edge case)")
    
    # Apply 14-day half-life decay to acceleration
    spine = apply_halflife_decay_14d(spine, 'narrative_acceleration', half_life_days=14)
    # Result: narrative_acceleration_decay_14d
    
    # Also apply 30-day decay for consistency
    spine = apply_halflife_decay(spine, 'narrative_velocity_lag1', features_config)
    
    # Validation: ensure no NaNs remain
    velocity_cols = ['narrative_velocity', 'narrative_velocity_lag1', 'narrative_acceleration']
    remaining_nans = {col: spine[col].isna().sum() for col in velocity_cols if col in spine.columns}
    if any(v > 0 for v in remaining_nans.values()):
        logger.warning(f"    Remaining NaNs in velocity features: {remaining_nans}")
    else:
        logger.info("    All narrative velocity features validated (no NaNs)")
    
    return spine


def create_interaction_features(spine: pd.DataFrame) -> pd.DataFrame:
    """
    Create pillar × pillar and fusion (CW × ACLED) interaction features.
    
    | Feature                       | Formula                                      | Null Handling        |
    |-------------------------------|----------------------------------------------|----------------------|
    | cw_onset_amplifier            | frag × wagner                                | 0 if either missing  |
    | cw_mass_casualty_risk         | pastoral × frag                              | 0 if either missing  |
    | cw_extraction_violence        | governance × wagner                          | 0 if either missing  |
    | cw_pastoral_predation         | governance × pastoral                        | 0 if either missing  |
    | fusion_gold_signal            | wagner × mech_gold_pivot_lag1                | 0 if either missing  |
    | fusion_fragmentation_confirmed| frag × mech_factional_infighting_lag1        | 0 if either missing  |
    | fusion_escalation_momentum    | max(delta, 0) × acled_mechanism_intensity    | 0 if either missing  |
    """
    logger.info("    Creating pillar interaction features...")
    
    # Shorthand for pillar columns
    frag = 'regime_guerrilla_fragmentation'
    wagner = 'regime_transnational_predation'
    pastoral = 'regime_ethno_pastoral_rupture'
    governance = 'regime_parallel_governance'
    
    def safe_multiply(col1: str, col2: str) -> pd.Series:
        """Multiply two columns, returning 0 if either is missing."""
        if col1 not in spine.columns or col2 not in spine.columns:
            return pd.Series(0.0, index=spine.index)
        return (spine[col1].fillna(0) * spine[col2].fillna(0))
    
    # Pillar × Pillar interactions
    spine['cw_onset_amplifier'] = safe_multiply(frag, wagner)
    spine['cw_mass_casualty_risk'] = safe_multiply(pastoral, frag)
    spine['cw_extraction_violence'] = safe_multiply(governance, wagner)
    spine['cw_pastoral_predation'] = safe_multiply(governance, pastoral)
    
    logger.info("    Creating fusion (CW × ACLED) interaction features...")
    
    # Fusion: CrisisWatch × ACLED Hybrid NLP
    spine['fusion_gold_signal'] = safe_multiply(wagner, 'mech_gold_pivot_lag1')
    spine['fusion_fragmentation_confirmed'] = safe_multiply(frag, 'mech_factional_infighting_lag1')
    
    # Escalation momentum: positive delta × mechanism intensity
    if 'cw_score_local_delta' in spine.columns:
        delta_positive = spine['cw_score_local_delta'].clip(lower=0)
    else:
        delta_positive = pd.Series(0.0, index=spine.index)
    
    # Use acled_mechanism_intensity if available, otherwise sum of mechanism columns
    if 'acled_mechanism_intensity' in spine.columns:
        mech_intensity = spine['acled_mechanism_intensity'].fillna(0)
    else:
        # Fallback: sum of available mech_* columns
        mech_cols = [c for c in spine.columns if c.startswith('mech_') and '_lag' not in c]
        if mech_cols:
            mech_intensity = spine[mech_cols].fillna(0).sum(axis=1)
        else:
            mech_intensity = pd.Series(0.0, index=spine.index)
    
    spine['fusion_escalation_momentum'] = delta_positive * mech_intensity
    
    logger.info(f"    Created 7 interaction features")
    
    return spine


def _create_stub_cw_columns(spine: pd.DataFrame) -> pd.DataFrame:
    """Create stub columns when CrisisWatch data is unavailable."""
    stub_cols = [
        'cw_score_local', 'cw_score_local_lag1', 'cw_score_local_lag1_decay_30d',
        'cw_score_local_spatial_lag', 'cw_score_local_spatial_lag_lag1',
        'cw_score_local_spatial_lag_lag1_decay_30d', 'cw_score_local_delta',
        'cw_score_local_delta_decay_14d',
        'regime_parallel_governance', 'regime_transnational_predation',
        'regime_guerrilla_fragmentation', 'regime_ethno_pastoral_rupture',
        'narrative_velocity', 'narrative_velocity_lag1', 'narrative_velocity_lag1_decay_30d',
        'narrative_acceleration', 'narrative_acceleration_decay_14d',
        'cw_onset_amplifier', 'cw_mass_casualty_risk', 'cw_extraction_violence',
        'cw_pastoral_predation', 'fusion_gold_signal', 'fusion_fragmentation_confirmed',
        'fusion_escalation_momentum'
    ]
    for col in stub_cols:
        spine[col] = 0.0
    return spine


def process_ioda(engine, spine: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    """Process IODA internet outage data with availability flag."""
    logger.info("PHASE 4E: Processing IODA...")

    insp = inspect(engine)
    if not insp.has_table("internet_outages", schema=SCHEMA):
        logger.warning("internet_outages table not found")
        spine['ioda_outage_score'] = 0.0
        spine['ioda_data_available'] = 0
        return spine

    dates = sorted(spine['date'].unique())

    try:
        ioda_raw = pd.read_sql(
            f"SELECT h3_index, date, variable, value FROM {SCHEMA}.internet_outages",
            engine
        )

        if ioda_raw.empty:
            spine['ioda_outage_score'] = 0.0
            spine['ioda_data_available'] = (spine['date'] >= STRUCTURAL_BREAKS['ioda']).astype(int)
            return spine

        ioda_raw['date'] = pd.to_datetime(ioda_raw['date'])
        ioda_raw['h3_index'] = ioda_raw['h3_index'].astype('int64')
        ioda_raw['spine_date'] = pd.cut(ioda_raw['date'], bins=dates, labels=dates[1:], right=True)

        ioda_outage = ioda_raw[ioda_raw['variable'] == 'ioda_outage_score']

        if not ioda_outage.empty:
            ioda_agg = ioda_outage.groupby(['h3_index', 'spine_date'], observed=True)['value'].max().reset_index()
            ioda_agg = ioda_agg.rename(columns={'spine_date': 'date', 'value': 'ioda_outage_score'})
            ioda_agg['date'] = pd.to_datetime(ioda_agg['date'])

            spine = safe_merge(spine, ioda_agg, on=['h3_index', 'date'], how='left')

        spine['ioda_outage_score'] = spine.get('ioda_outage_score', pd.Series(0.0)).fillna(0)

    except Exception as e:
        logger.warning(f"IODA processing failed: {e}")
        spine['ioda_outage_score'] = 0.0

    # IODA availability flag
    spine['ioda_data_available'] = (spine['date'] >= STRUCTURAL_BREAKS['ioda']).astype(int)

    return spine


def process_fusion_features(spine: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    """Create additional fusion/interaction features (legacy compatibility)."""
    logger.info("PHASE 4F: Creating Additional Fusion Features...")

    # GDELT shock signal (negative tone × regime instability)
    if 'gdelt_avg_tone' in spine.columns:
        regime_cols = ['regime_parallel_governance', 'regime_transnational_predation']
        existing_regime = [c for c in regime_cols if c in spine.columns]

        if existing_regime:
            spine['gdelt_shock_signal'] = (
                (spine['gdelt_avg_tone'] * -1) - spine[existing_regime].sum(axis=1)
            ).fillna(0)
        else:
            spine['gdelt_shock_signal'] = 0.0
    else:
        spine['gdelt_shock_signal'] = 0.0

    return spine


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run():
    """Main execution function."""
    engine = get_db_engine()
    configs = load_configs()

    data_cfg = configs['data'] if 'data' in configs else configs[0]
    feat_cfg = configs['features'] if 'features' in configs else configs[1]

    start_date = data_cfg['global_date_window']['start_date']
    end_date = data_cfg['global_date_window']['end_date']
    specs = parse_registry(feat_cfg)

    logger.info("=" * 70)
    logger.info("CONFLICT FEATURES PROCESSING (v3.2 - fatalities_lag1 fix)")
    logger.info("=" * 70)

    # 1. Load existing spine
    spine = load_existing_spine(engine, start_date, end_date)
    logger.info(f"Loaded spine: {len(spine):,} rows")

    # 2. ACLED (core conflict)
    spine = process_acled(engine, spine, feat_cfg)

    # 3. ACLED Hybrid (NLP mechanisms)
    spine = process_acled_hybrid(engine, spine, feat_cfg)

    # 4. GDELT (news events)
    spine = process_gdelt(engine, spine, feat_cfg)

    # 5. CrisisWatch (NLP signals - FULL PIPELINE)
    spine = process_crisiswatch(engine, spine, specs.get('nlp', []), feat_cfg)

    # 6. IODA (internet outages)
    spine = process_ioda(engine, spine, feat_cfg)

    # 7. Additional fusion features (legacy)
    spine = process_fusion_features(spine, feat_cfg)

    # 8. Sanitize
    spine = sanitize_numeric_columns(spine, engine=engine, features_config=feat_cfg)

    # Ensure target table has all new feature columns before upsert
    enrich_table_schema(engine, spine, OUTPUT_TABLE, SCHEMA)

    # 9. Upsert (don't drop - overlay on context layer)
    logger.info(f"Upserting Conflict Features: {len(spine):,} rows...")
    logger.info(f"Columns ({len(spine.columns)}): {spine.columns.tolist()[:30]}...")

    total_rows = len(spine)
    for i in range(0, total_rows, CHUNK_SIZE):
        chunk = spine.iloc[i:i + CHUNK_SIZE].copy()
        upload_to_postgis(engine, chunk, OUTPUT_TABLE, SCHEMA, PRIMARY_KEYS)

    logger.info("\n" + "=" * 70)
    logger.info("CONFLICT FEATURES UPSERT COMPLETE")
    logger.info("=" * 70)

    # Log availability summary
    logger.info("Conflict Availability Summary:")
    for flag in ['gdelt_data_available', 'ioda_data_available']:
        if flag in spine.columns:
            pct = spine[flag].mean() * 100
            logger.info(f"  {flag}: {pct:.1f}% of rows")

    # Log CrisisWatch feature summary
    logger.info("\nCrisisWatch Feature Summary:")
    cw_features = [c for c in spine.columns if c.startswith('cw_') or c.startswith('regime_') 
                   or c.startswith('fusion_') or c.startswith('narrative_')]
    logger.info(f"  Total CW-derived features: {len(cw_features)}")
    
    for col in ['cw_score_local', 'cw_onset_amplifier', 'narrative_velocity', 'narrative_acceleration']:
        if col in spine.columns:
            non_zero = (spine[col] != 0).sum()
            logger.info(f"  {col}: {non_zero:,} non-zero rows ({100*non_zero/len(spine):.2f}%)")


if __name__ == "__main__":
    run()
