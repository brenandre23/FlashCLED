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
# Determine project root. The original code climbed three directory levels
# from this file, which was incorrect for the current project structure.
# We now go up two levels (from pipeline/modeling to the repository root).
ROOT_DIR = Path(__file__).resolve().parents[2]

# Fallback if standard relative path fails. Append ROOT_DIR to sys.path so
# that the "utils" module can be resolved when running this script directly.
if "utils" not in sys.modules:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, SCHEMA, ensure_h3_int64, validate_h3_types


def get_required_features(models_cfg: Dict[str, Any]) -> List[str]:
    """
    Extracts the SET of all feature names required by enabled submodels.
    """
    required = set()
    for name, cfg in models_cfg["submodels"].items():
        if cfg.get("enabled", False):
            # Handle PCA special case (features might be generated later, 
            # but we need the inputs if listed)
            reqs = cfg.get("features", [])
            # If 'all_candidates' is used, we skip specific validation here 
            # and handle it in the wild card logic, but usually explicit lists are safer.
            if "all_candidates" not in reqs:
                required.update(reqs)
    return list(required)


def get_enabled_features_from_registry(features_cfg: Dict[str, Any]) -> Set[str]:
    """
    Parses features.yaml registry to find globally enabled features.
    
    FIX: Prevents 'ValueError: The truth value of an array with more than one element is ambiguous'
    by using explicit boolean indexing instead of implicit truth checks on NumPy arrays.
    """
    registry = features_cfg.get("registry", [])
    if not registry:
        logger.info("Registry is empty or missing in features.yaml.")
        return set()

    df = pd.DataFrame(registry)
    
    # Ensure strictly necessary columns exist
    if "output_col" not in df.columns:
        logger.warning("Feature registry missing 'output_col'. Cannot filter.")
        return set()

    # If 'enabled' column is missing, assume ALL are enabled by default
    if "enabled" not in df.columns:
        logger.info(f"Registry: All {len(df)} features enabled (no 'enabled' flag detected).")
        return set(df["output_col"].dropna().unique())

    # Fill NAs with True (default to enabled)
    df["enabled"] = df["enabled"].fillna(True)
    
    enabled_features = df[df["enabled"] == True]["output_col"].tolist()
    
    logger.info(f"Registry: {len(enabled_features)} enabled features out of {len(df)} total.")
    if len(enabled_features) < 10:
        logger.debug(f"Sample enabled: {enabled_features}")
        
    return set(enabled_features)


def check_model_features_against_registry(required_features: List[str], features_cfg: Dict[str, Any]):
    """
    Validates that every feature requested by models.yaml is actually defined
    in the features.yaml registry (output_col).
    """
    registry = features_cfg.get("registry", [])
    registered_outputs = {item.get("output_col") for item in registry if "output_col" in item}
    
    missing = [f for f in required_features if f not in registered_outputs]
    
    # Note: Static features (dist_to_road, etc.) won't be in registry.
    # So we don't raise an error here, but we log it for debugging.
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
    """
    Pre-flight validation to ensure all required columns exist in the specified
    table. Queries ``information_schema.columns`` for the given schema and
    table and raises a descriptive ValueError if any columns are missing.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Active database engine.
    table : str
        Name of the table (without schema) to inspect.
    required_cols : Iterable[str]
        Column names that must be present in the table.

    Raises
    ------
    ValueError
        If any columns in ``required_cols`` are absent from the table.
    """
    if not required_cols:
        return
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        """
    )
    df_cols = pd.read_sql(query, engine, params={"schema": SCHEMA, "table": table})
    existing_cols = set(df_cols["column_name"].tolist())
    missing = [col for col in required_cols if col not in existing_cols]
    if missing:
        raise ValueError(
            f"Missing columns in {SCHEMA}.{table}: {missing}. "
            "Ensure these columns are ingested or adjust models.yaml accordingly."
        )


def classify_features(required_features: List[str], engine: SqlEngine) -> Tuple[List[str], List[str]]:
    """
    Decide whether each feature should be pulled from:
      - car_cewp.temporal_features  => t.<col>
      - car_cewp.features_static   => s.<col>

    This prevents misclassification of computed temporal columns like pop_log and EPR-derived fields.
    """
    temporal_cols = _table_columns(engine, SCHEMA, "temporal_features")
    static_cols   = _table_columns(engine, SCHEMA, "features_static")

    temporal_selects, static_selects, missing = [], [], []

    for f in required_features:
        if f in temporal_cols:
            temporal_selects.append(f"t.{f}")
        elif f in static_cols:
            static_selects.append(f"s.{f}")
        else:
            missing.append(f)

    if missing:
        raise ValueError(
            "Missing required features (not found in DB columns of either "
            f"{SCHEMA}.temporal_features or {SCHEMA}.features_static):\n"
            f"{missing}\n\n"
            "Fix by either:\n"
            "  • computing/ingesting these columns before building the feature matrix, OR\n"
            "  • correcting the feature names in models.yaml to match the DB."
        )

    return temporal_selects, static_selects


def build_dynamic_query(
    start_date: str, 
    end_date: str, 
    temp_cols: List[str], 
    static_cols: List[str], 
    horizons: List[Dict[str, Any]]
) -> str:
    """
    Constructs SQL query dynamically handling Targets and Joins.
    """
    # 1. Feature Selection
    all_selects = temp_cols + static_cols
    select_sql = ",\n        ".join(all_selects)
    
    # 2. Target Generation (Dynamic LEADs)
    target_clauses = []
    for h in horizons:
        steps = h["steps"]
        # Name convention: target_{steps}_step
        # We partition by H3 and order by Date
        clause = (
            f"LEAD(t.fatalities_14d_sum, {steps}) "
            f"OVER (PARTITION BY t.h3_index ORDER BY t.date) "
            f"as target_{steps}_step"
        )
        target_clauses.append(clause)
        
    target_sql = ",\n        ".join(target_clauses)
    
    # 3. Final Query
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
    
    # 1. Resolve Dates
    if not start_date:
        start_date = data_cfg["global_date_window"]["start_date"]
    if not end_date:
        end_date = data_cfg["global_date_window"]["end_date"]
    
    logger.info(f"Window: {start_date} -> {end_date}")
    
    # 2. Identify Requested Features
    req_feats = get_required_features(models_cfg)
    
    # 3. Validation: Check against Registry Enabled Flags
    check_model_features_against_registry(req_feats, features_cfg)

    registry_enabled_set = get_enabled_features_from_registry(features_cfg)
    
    # Determine all registry features for conflict detection
    registry_all = set()
    if features_cfg.get("registry"):
        registry_all = {
            item.get("output_col") for item in features_cfg["registry"] if "output_col" in item
        }
        
    # Filter out disabled features
    final_req_feats = []
    for f in req_feats:
        if f in registry_all and f not in registry_enabled_set:
            logger.warning(f"Feature '{f}' requested by models.yaml but DISABLED in features.yaml. Skipping.")
            continue
        final_req_feats.append(f)
    
    req_feats = final_req_feats

    # 4. Classify Features (Temporal vs Static)
    t_cols, s_cols = classify_features(req_feats, engine)
    
    logger.info(f"Requested Features: {len(req_feats)}")
    logger.info(f" -> Mapped to Temporal: {len(t_cols)}")
    logger.info(f" -> Mapped to Static:   {len(s_cols)}")
    
    # 5. Pre-flight Column Validation
    # Validate that all temporal and static columns exist in the database before
    # constructing the large SQL query. Extract the raw column names from
    # the "t." and "s." prefixes used by classify_features.
    temporal_fields = [c.split(".", 1)[1] for c in t_cols]
    static_fields = [c.split(".", 1)[1] for c in s_cols]
    validate_columns(engine, "temporal_features", temporal_fields)
    validate_columns(engine, "features_static", static_fields)

    # 6. Build Query
    horizons = models_cfg["horizons"]
    sql = build_dynamic_query(start_date, end_date, t_cols, s_cols, horizons)
    
    # 7. Stream & Process
    out_path = PATHS["data_proc"] / "feature_matrix.parquet"
    chunk_size = 50000
    
    logger.info("Executing Query (Streaming)...")
    chunks = pd.read_sql(sql, engine, chunksize=chunk_size)
    
    master_df = None
    total_rows = 0
    
    for i, chunk in enumerate(chunks):
        # Enforce H3 BigInt (Critical Fix for compatibility) using centralized helper
        chunk["h3_index"] = chunk["h3_index"].apply(ensure_h3_int64).astype("int64")
        chunk["date"] = pd.to_datetime(chunk["date"])
        
        if master_df is None:
            master_df = chunk
        else:
            master_df = pd.concat([master_df, chunk], ignore_index=True)
            
        total_rows += len(chunk)
        print(f"  Batch {i+1}: {total_rows} rows accumulated...", end="\r")
        
    print()  # Newline
    
    if master_df is None or master_df.empty:
        logger.error("Query returned no data! Check database population.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # TYPE SAFETY: Distance features must be numeric (not object) to
    # avoid model crashes when columns are entirely null/empty (e.g.,
    # dist_to_controlled_mine when no roadblocks exist).
    # ------------------------------------------------------------------
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
    
    # 8. Save
    master_df.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")
    
    # 9. Save Meta
    meta = {
        "start": str(start_date),
        "end": str(end_date),
        "rows": len(master_df),
        "features": req_feats,
        "horizons": [h["name"] for h in horizons],
    }
    with open(PATHS["data_proc"] / "matrix_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    try:
        cfg = load_configs()
        # Handle tuple return if load_configs returns (data, feat, model)
        if isinstance(cfg, tuple):
            configs = {"data": cfg[0], "features": cfg[1], "models": cfg[2]}
        else:
            configs = cfg  # Assuming dict
            
        engine = get_db_engine()
        run(configs, engine)
    except Exception as e:
        logger.error(f"Matrix Build Failed: {e}", exc_info=True)
    finally:
        if "engine" in locals():
            engine.dispose()
