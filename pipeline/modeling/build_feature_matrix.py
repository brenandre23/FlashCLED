# pipeline/modeling/build_feature_matrix.py
"""
build_feature_matrix.py
=======================
Assembles the final Analytical Base Table (ABT) for modeling.
Refactored for Phase 5 (Robust Dynamic Logic).

CRITICAL FIXES:
1. Dynamic Temporal vs Static column selection (No more hardcoding).
2. Dynamic Target Generation based on models.yaml horizons.
3. Strict H3 Type enforcement (BigInt) via centralized helper.
4. FIX: Robust feature selection from registry (avoids NumPy boolean ambiguity).
5. Pre-flight column validation: ensures all required columns exist before running the query.
6. FIX: Target columns (target_1_step, target_2_step, etc.) are now PRESERVED in final output.
7. FIX: ACLED Hybrid uses new mech_* schema (not legacy driver_* columns).
8. FIX (2026-01-25): Date normalization for ACLED hybrid merge - fixes 0% coverage bug.
   - Root cause: hybrid table stores DATE (no time), spine has datetime64 (with time)
   - Solution: Normalize both to midnight before merge using .dt.normalize()

Updated: 2026-01-25
- Fixed ACLED hybrid merge returning 0% non-zero coverage
- Added date alignment diagnostics
- Both spine and hybrid dates now normalized to midnight for consistent merge
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from sqlalchemy import text, Engine
from sqlalchemy.engine import Engine as SqlEngine

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]

if "utils" not in sys.modules:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, SCHEMA, ensure_h3_int64, validate_h3_types


# ==============================================================================
# TARGET COLUMN DEFINITIONS (Critical for sensitivity analysis)
# ==============================================================================
TARGET_COL_PATTERNS = [
    'target_fatalities',
    'target_binary',
    'target_fatalities_',
    'target_binary_',
    'target_1_step',
    'target_2_step',
]

# ==============================================================================
# ACLED HYBRID COLUMNS (New mech_* schema - NOT legacy driver_* columns)
# ==============================================================================
# These are the columns produced by process_acled_hybrid.py
EXPECTED_HYBRID_COLS = [
    # Mechanism scores (0-1 range, quality-weighted)
    "mech_gold_pivot",
    "mech_predatory_tax",
    "mech_factional_infighting",
    "mech_collective_punishment",
    # Uncertainty scores (0-1 range, margin-based)
    "mech_gold_pivot_uncertainty",
    "mech_predatory_tax_uncertainty",
    "mech_factional_infighting_uncertainty",
    "mech_collective_punishment_uncertainty",
    # Aggregate risk scores
    "acled_actor_risk_score",
    "acled_mechanism_intensity",
    "acled_combined_risk_score",
]


def get_required_features(models_cfg: Dict[str, Any]) -> List[str]:
    """Extracts the SET of all feature names required by enabled submodels."""
    required = set()
    for name, cfg in models_cfg["submodels"].items():
        if cfg.get("enabled", False):
            reqs = cfg.get("features", [])
            reqs = [r for r in reqs if isinstance(r, str)]
            if "all_candidates" not in reqs:
                required.update(reqs)
    return list(required)


def get_enabled_features_from_registry(features_cfg: Dict[str, Any]) -> Set[str]:
    """Parses features.yaml registry to find globally enabled features."""
    registry = features_cfg.get("registry", [])
    if not registry:
        logger.info("Registry is empty or missing in features.yaml.")
        return set()

    df = pd.DataFrame(registry)
    
    if "output_col" not in df.columns:
        logger.warning("Feature registry missing 'output_col'. Cannot filter.")
        return set()

    if "enabled" not in df.columns:
        logger.info(f"Registry: All {len(df)} features enabled (no 'enabled' flag detected).")
        return set(df["output_col"].dropna().unique())

    df["enabled"] = df["enabled"].fillna(True)
    enabled_features = df[df["enabled"] == True]["output_col"].tolist()
    
    logger.info(f"Registry: {len(enabled_features)} enabled features out of {len(df)} total.")
    return set(enabled_features)


def check_model_features_against_registry(required_features: List[str], features_cfg: Dict[str, Any]):
    """Validates features requested by models.yaml exist in features.yaml registry."""
    registry = features_cfg.get("registry", [])
    registered_outputs = {item.get("output_col") for item in registry if "output_col" in item}
    missing = [f for f in required_features if f not in registered_outputs]
    if missing:
        logger.debug(f"Features requested but not in registry (likely static): {missing}")


def _table_columns(engine: SqlEngine, schema: str, table: str) -> Set[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
    """)
    df = pd.read_sql(q, engine, params={"schema": schema, "table": table})
    return set(df["column_name"].tolist())


def validate_columns(engine: SqlEngine, table: str, required_cols: List[str]) -> None:
    """Pre-flight validation to ensure all required columns exist."""
    if not required_cols:
        return
    query = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
    """)
    df_cols = pd.read_sql(query, engine, params={"schema": SCHEMA, "table": table})
    existing_cols = set(df_cols["column_name"].tolist())
    missing = [col for col in required_cols if col not in existing_cols]
    if missing:
        # For temporal_features, create missing columns as DOUBLE PRECISION to unblock modeling
        if table == "temporal_features":
            logger.warning(
                f"Missing columns in {SCHEMA}.{table}: {missing}. "
                "Auto-adding as DOUBLE PRECISION (fill will occur downstream)."
            )
            with engine.begin() as conn:
                for col in missing:
                    conn.execute(
                        text(f'ALTER TABLE {SCHEMA}.{table} ADD COLUMN IF NOT EXISTS "{col}" DOUBLE PRECISION;')
                    )
            return
        raise ValueError(
            f"Missing columns in {SCHEMA}.{table}: {missing}. "
            "Ensure these columns are ingested or adjust models.yaml accordingly."
        )


def classify_features(
    required_features: List[str], 
    engine: SqlEngine, 
    taxonomy_alias_map: Optional[Dict[str, str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """Classify features by source table (temporal, static, or hybrid)."""
    temporal_cols = _table_columns(engine, SCHEMA, "temporal_features")
    static_cols = _table_columns(engine, SCHEMA, "features_static")
    hybrid_cols = _table_columns(engine, SCHEMA, "features_acled_hybrid")

    temporal_selects, static_selects, taxonomy_selects, missing = [], [], [], []
    taxonomy_alias_map = taxonomy_alias_map or {}

    for f in required_features:
        if f in temporal_cols:
            temporal_selects.append(f"t.{f}")
        elif f in static_cols:
            static_selects.append(f"s.{f}")
        elif f in hybrid_cols:
            taxonomy_selects.append(f"a.{f}")
        elif f in taxonomy_alias_map and taxonomy_alias_map[f] in hybrid_cols:
            raw_col = taxonomy_alias_map[f]
            taxonomy_selects.append(f"a.{raw_col} as {f}")
        else:
            missing.append(f)

    if missing:
        raise ValueError(
            f"Missing required features (not in temporal_features, features_static, or features_acled_hybrid):\n"
            f"{missing}\n\n"
            "Fix by computing/ingesting these columns or correcting models.yaml."
        )

    return temporal_selects, static_selects, taxonomy_selects


def build_dynamic_query(
    start_date: str, 
    end_date: str, 
    temp_cols: List[str], 
    static_cols: List[str], 
    horizons: List[Dict[str, Any]]
) -> str:
    """Constructs SQL query dynamically handling Targets and Joins."""
    all_selects = temp_cols + static_cols
    select_sql = ",\n        ".join(all_selects)
    
    # Target Generation (Dynamic LEADs)
    target_clauses = []
    for h in horizons:
        steps = h["steps"]
        # Fatality count target
        target_clauses.append(
            f"LEAD(t.fatalities_14d_sum, {steps}) "
            f"OVER (PARTITION BY t.h3_index ORDER BY t.date) "
            f"as target_fatalities_{steps}_step"
        )
        # Binary occurrence target
        target_clauses.append(
            f"CASE WHEN LEAD(t.fatalities_14d_sum, {steps}) "
            f"OVER (PARTITION BY t.h3_index ORDER BY t.date) > 0 THEN 1 ELSE 0 END "
            f"as target_binary_{steps}_step"
        )
        
    target_sql = ",\n        ".join(target_clauses)
    
    sql = f"""
    SELECT
        t.h3_index,
        t.date,
        {select_sql},
        {target_sql}
    FROM {SCHEMA}.temporal_features t
    JOIN {SCHEMA}.features_static s ON t.h3_index = s.h3_index
    WHERE t.date BETWEEN '{start_date}' AND '{end_date}'
    """
    return sql


def identify_target_columns(df: pd.DataFrame, horizons: List[Dict[str, Any]]) -> List[str]:
    """
    Identifies all target columns that should be preserved in the output.
    
    This function ensures that target columns generated by the SQL LEAD functions
    are explicitly tracked and NOT dropped during feature filtering.
    
    Parameters
    ----------
    df : pd.DataFrame
        The feature matrix DataFrame
    horizons : List[Dict]
        List of horizon configurations from models.yaml
        
    Returns
    -------
    List[str]
        List of target column names present in the DataFrame
    """
    target_cols = []
    
    # 1. Add horizon-specific targets
    for h in horizons:
        steps = h["steps"]
        fatality_col = f"target_fatalities_{steps}_step"
        binary_col = f"target_binary_{steps}_step"
        
        if fatality_col in df.columns:
            target_cols.append(fatality_col)
        if binary_col in df.columns:
            target_cols.append(binary_col)
    
    # 2. Add any columns matching legacy patterns
    for col in df.columns:
        for pattern in TARGET_COL_PATTERNS:
            if col.startswith(pattern) and col not in target_cols:
                target_cols.append(col)
    
    # 3. Add base target columns if present
    base_targets = ['target_fatalities', 'target_binary', 'fatalities_14d_sum']
    for col in base_targets:
        if col in df.columns and col not in target_cols:
            target_cols.append(col)
    
    return target_cols


def run(
    configs: Dict[str, Any], 
    engine: SqlEngine, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
) -> None:
    logger.info("=" * 60)
    logger.info("BUILDING FEATURE MATRIX (Fixed Dynamic Version)")
    logger.info("=" * 60)

    validate_h3_types(engine)
    
    data_cfg = configs["data"]
    models_cfg = configs["models"]
    features_cfg = configs["features"]
    
    # Resolve Dates
    if not start_date:
        start_date = data_cfg["global_date_window"]["start_date"]
    if not end_date:
        end_date = data_cfg["global_date_window"]["end_date"]
    
    logger.info(f"Window: {start_date} -> {end_date}")
    
    # Identify Requested Features
    req_feats = get_required_features(models_cfg)
    check_model_features_against_registry(req_feats, features_cfg)
    registry_enabled_set = get_enabled_features_from_registry(features_cfg)
    
    registry_all = set()
    if features_cfg.get("registry"):
        registry_all = {
            item.get("output_col") for item in features_cfg["registry"] if "output_col" in item
        }
        
    filtered_model_feats = []
    for f in req_feats:
        if f in registry_all and f not in registry_enabled_set:
            logger.warning(f"Feature '{f}' requested by models.yaml but DISABLED. Skipping.")
            continue
        filtered_model_feats.append(f)
    
    req_feats = sorted(set(filtered_model_feats) | registry_enabled_set)
    logger.info(f"Including {len(registry_enabled_set)} enabled registry features; total: {len(req_feats)}")

    # Classify Features
    t_cols, s_cols, a_cols = classify_features(req_feats, engine)
    
    logger.info(f"Requested Features: {len(req_feats)}")
    logger.info(f" -> Mapped to Temporal: {len(t_cols)}")
    logger.info(f" -> Mapped to Static:   {len(s_cols)}")
    logger.info(f" -> Mapped to ACLED Hybrid: {len(a_cols)}")
    
    # Pre-flight Column Validation
    temporal_fields = [c.split(".", 1)[1] for c in t_cols]
    static_fields = [c.split(".", 1)[1] for c in s_cols]
    validate_columns(engine, "temporal_features", temporal_fields)
    validate_columns(engine, "features_static", static_fields)
    if a_cols:
        hybrid_fields = []
        for c in a_cols:
            col_part = c.split(".", 1)[1]
            if " as " in col_part:
                col_part = col_part.split(" as ", 1)[0].strip()
            hybrid_fields.append(col_part)
        try:
            validate_columns(engine, "features_acled_hybrid", hybrid_fields)
        except ValueError as e:
            logger.warning(f"features_acled_hybrid missing expected columns; will fill with 0: {e}")

    # Build Query
    horizons = models_cfg["horizons"]
    sql = build_dynamic_query(start_date, end_date, t_cols, s_cols, horizons)
    
    # Stream & Process
    out_path = PATHS["data_proc"] / "feature_matrix.parquet"
    chunk_size = 50000
    
    logger.info("Executing Query (Streaming)...")
    chunks = pd.read_sql(sql, engine, chunksize=chunk_size)
    
    master_df = None
    total_rows = 0
    
    for i, chunk in enumerate(chunks):
        chunk["h3_index"] = chunk["h3_index"].apply(ensure_h3_int64).astype("int64")
        chunk["date"] = pd.to_datetime(chunk["date"])
        
        if master_df is None:
            master_df = chunk
        else:
            master_df = pd.concat([master_df, chunk], ignore_index=True)
            
        total_rows += len(chunk)
        print(f"  Batch {i+1}: {total_rows} rows accumulated...", end="\r")
        
    print()
    
    if master_df is None or master_df.empty:
        logger.error("Query returned no data! Check database population.")
        sys.exit(1)

    # ==================================================================
    # CRITICAL FIX: Identify and preserve target columns (Task 2)
    # ==================================================================
    target_cols = identify_target_columns(master_df, horizons)
    logger.info(f"Identified {len(target_cols)} target columns to preserve: {target_cols}")
    
    # Validate targets exist
    missing_targets = [t for t in target_cols if t not in master_df.columns]
    if missing_targets:
        logger.warning(f"Expected target columns missing from query result: {missing_targets}")

    # ==================================================================
    # ACLED HYBRID FEATURES (New mech_* schema)
    # ==================================================================
    # CHECK: If columns already exist in master_df (from temporal_features), skip redundant merge
    existing_hybrid_cols = [c for c in EXPECTED_HYBRID_COLS if c in master_df.columns]
    
    if len(existing_hybrid_cols) == len(EXPECTED_HYBRID_COLS):
        logger.info("✅ ACLED Hybrid features already present in temporal_features. Skipping redundant merge.")
    else:
        logger.info("Loading Optimized ACLED Hybrid Features (mech_* schema)...")
        
        # Build query for new schema columns
        query_acled = f"""
            SELECT 
                event_date as date, 
                h3_index,
                mech_gold_pivot,
                mech_predatory_tax,
                mech_factional_infighting,
                mech_collective_punishment,
                mech_gold_pivot_uncertainty,
                mech_predatory_tax_uncertainty,
                mech_factional_infighting_uncertainty,
                mech_collective_punishment_uncertainty,
                acled_actor_risk_score,
                acled_mechanism_intensity,
                acled_combined_risk_score
            FROM {SCHEMA}.features_acled_hybrid
        """
        
        try:
            df_acled = pd.read_sql(query_acled, engine)
        except Exception as e:
            logger.error(f"Failed to load ACLED hybrid features: {e}")
            df_acled = pd.DataFrame()
        
        if df_acled.empty:
            logger.warning(f"ACLED hybrid table empty. Filling {len(EXPECTED_HYBRID_COLS)} columns with 0.")
            for col in EXPECTED_HYBRID_COLS:
                if col not in master_df.columns:
                    master_df[col] = 0
        else:
            actual_cols = [c for c in EXPECTED_HYBRID_COLS if c in df_acled.columns]
            missing_cols = [c for c in EXPECTED_HYBRID_COLS if c not in df_acled.columns]
            
            if missing_cols:
                logger.warning(f"Missing ACLED hybrid columns (will fill with 0): {missing_cols}")
            
            logger.info(f"ACLED hybrid: {len(df_acled):,} rows, {len(actual_cols)}/{len(EXPECTED_HYBRID_COLS)} columns")
            
            # CRITICAL FIX: Normalize dates to midnight for consistent merging
            # The hybrid table stores DATE (no time), spine has datetime64 (with time)
            df_acled["date"] = pd.to_datetime(df_acled["date"]).dt.normalize()
            df_acled["h3_index"] = df_acled["h3_index"].astype("int64")
            
            # Also normalize master_df dates for consistent comparison
            master_df["date"] = pd.to_datetime(master_df["date"]).dt.normalize()
            
            # Debug: Log date alignment info
            acled_dates = set(df_acled["date"].unique())
            master_dates = set(master_df["date"].unique())
            date_overlap = len(acled_dates.intersection(master_dates))
            logger.info(f"  Date alignment: {len(acled_dates)} hybrid dates, {len(master_dates)} spine dates, {date_overlap} overlap")
            
            if date_overlap == 0:
                logger.error("CRITICAL: No date overlap between ACLED hybrid and spine!")
                logger.error(f"  Hybrid date range: {df_acled['date'].min()} to {df_acled['date'].max()}")
                logger.error(f"  Spine date range: {master_df['date'].min()} to {master_df['date'].max()}")
                # Sample dates for debugging
                logger.error(f"  Sample hybrid dates: {sorted(list(acled_dates))[:5]}")
                logger.error(f"  Sample spine dates: {sorted(list(master_dates))[:5]}")
            
            acled_date_range = (df_acled["date"].min(), df_acled["date"].max())
            master_date_range = (master_df["date"].min(), master_df["date"].max())
            
            if acled_date_range[0] > master_date_range[0]:
                logger.warning(
                    f"ACLED hybrid starts at {acled_date_range[0].date()}, "
                    f"after feature matrix start {master_date_range[0].date()}."
                )
            
            # Drop columns from master_df that we are about to merge (to avoid _x/_y)
            cols_to_drop = [c for c in df_acled.columns if c in master_df.columns and c not in ["date", "h3_index"]]
            if cols_to_drop:
                logger.info(f"Dropping {len(cols_to_drop)} partial columns from spine before merge to prevent duplicates.")
                master_df.drop(columns=cols_to_drop, inplace=True)

            pre_merge_rows = len(master_df)
            master_df = master_df.merge(df_acled, on=["date", "h3_index"], how="left", validate="1:1")
            post_merge_rows = len(master_df)
            
            if pre_merge_rows != post_merge_rows:
                raise ValueError(
                    f"ACLED hybrid merge changed row count: {pre_merge_rows} -> {post_merge_rows}."
                )
            
            for col in EXPECTED_HYBRID_COLS:
                if col not in master_df.columns:
                    master_df[col] = 0
                else:
                    null_count = master_df[col].isna().sum()
                    if null_count > 0:
                        logger.debug(f"  {col}: filling {null_count:,} nulls with 0")
                    master_df[col] = master_df[col].fillna(0)
            
            coverage_stats = {}
            for col in EXPECTED_HYBRID_COLS:
                non_zero = (master_df[col] != 0).sum()
                coverage_stats[col] = non_zero / len(master_df) * 100
            
            avg_coverage = sum(coverage_stats.values()) / len(coverage_stats)
            logger.info(f"ACLED hybrid merge complete. Average non-zero coverage: {avg_coverage:.1f}%")

    # Distance columns type safety
    dist_cols = [c for c in master_df.columns if c.startswith("dist_")]
    if dist_cols:
        logger.info(f"Enforcing numeric types for distance columns: {dist_cols}")
        MAX_DISTANCE_KM = 500.0
        for col in dist_cols:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")
            null_pct = master_df[col].isna().mean()
            if null_pct == 1.0:
                logger.warning(f"  ⚠ {col} is 100% null — filling with sentinel {MAX_DISTANCE_KM} km.")
                master_df[col] = MAX_DISTANCE_KM
            elif null_pct > 0.5:
                logger.warning(f"  ⚠ {col} has {null_pct:.1%} nulls.")
            master_df[col] = master_df[col].astype("float32")

    logger.info(f"Final Matrix Shape: {master_df.shape}")
    
    # ==================================================================
    # CRITICAL: Verify target columns are still present before saving
    # ==================================================================
    final_target_cols = [t for t in target_cols if t in master_df.columns]
    if len(final_target_cols) < len(target_cols):
        dropped = set(target_cols) - set(final_target_cols)
        logger.error(f"TARGET COLUMNS WERE DROPPED DURING PROCESSING: {dropped}")
        raise ValueError(f"Target columns missing from final output: {dropped}")
    
    logger.info(f"✓ Target columns preserved in output: {final_target_cols}")
    
    # Fill NaN in target columns (downstream analysis expects no NaN)
    for col in final_target_cols:
        null_count = master_df[col].isna().sum()
        if null_count > 0:
            logger.info(f"  Filling {null_count:,} NaN values in {col} with 0")
            master_df[col] = master_df[col].fillna(0)
    
    # Save
    master_df.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")
    
    # Verify saved file contains targets
    saved_df = pd.read_parquet(out_path)
    saved_targets = [c for c in saved_df.columns if any(c.startswith(p) for p in TARGET_COL_PATTERNS)]
    logger.info(f"✓ Verification: Saved parquet contains {len(saved_targets)} target columns")
    
    # Save Meta
    meta = {
        "start": str(start_date),
        "end": str(end_date),
        "rows": len(master_df),
        "features": req_feats,
        "horizons": [h["name"] for h in horizons],
        "target_columns": final_target_cols,  # Include targets in metadata
    }
    with open(PATHS["data_proc"] / "matrix_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    try:
        cfg = load_configs()
        if isinstance(cfg, tuple):
            configs = {"data": cfg[0], "features": cfg[1], "models": cfg[2]}
        else:
            configs = cfg
            
        engine = get_db_engine()
        run(configs, engine)
    except Exception as e:
        logger.error(f"Matrix Build Failed: {e}", exc_info=True)
    finally:
        if "engine" in locals():
            engine.dispose()
