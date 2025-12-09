# pipeline/modeling/build_feature_matrix.py
"""
build_feature_matrix.py
=======================
Assembles the final Analytical Base Table (ABT) for modeling.
Refactored for Phase 5 (Robust Dynamic Logic).

CRITICAL FIXES:
1. Dynamic Temporal vs Static column selection (No more hardcoding).
2. Dynamic Target Generation based on models.yaml horizons.
3. Strict H3 Type enforcement (BigInt).
4. FIX: Robust feature selection from registry (avoids NumPy boolean ambiguity).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import text

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[3] # Adjusted to match nested structure if needed, or use robust lookup
# Fallback if standard relative path fails
if "utils" not in sys.modules:
    # Try 3 levels up (pipeline/modeling/script -> root)
    sys.path.append(str(Path(__file__).resolve().parents[2])) 

from utils import logger, PATHS, get_db_engine, load_configs

SCHEMA = "car_cewp"

def get_required_features(models_cfg):
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

def get_enabled_features_from_registry(features_cfg):
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

    # FIX: Explicit boolean filtering
    # df["enabled"] == True creates a boolean mask safely.
    # df[mask] selects rows.
    # We explicitly cast to boolean to handle 0/1 integers or strings like "true" if necessary,
    # but normally yaml parses true/false as booleans.
    
    # Fill NAs with True (default to enabled)
    df["enabled"] = df["enabled"].fillna(True)
    
    enabled_features = df[df["enabled"] == True]["output_col"].tolist()
    
    logger.info(f"Registry: {len(enabled_features)} enabled features out of {len(df)} total.")
    if len(enabled_features) < 10:
        logger.debug(f"Sample enabled: {enabled_features}")
        
    return set(enabled_features)

def _table_columns(engine, schema: str, table: str) -> set[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
    """)
    df = pd.read_sql(q, engine, params={"schema": schema, "table": table})
    return set(df["column_name"].tolist())

def classify_features(required_features, engine):
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

def build_dynamic_query(start_date, end_date, temp_cols, static_cols, horizons):
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
        clause = f"LEAD(t.fatalities_14d_sum, {steps}) OVER (PARTITION BY t.h3_index ORDER BY t.date) as target_{steps}_step"
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

def run(configs, engine, start_date=None, end_date=None):
    logger.info("="*60)
    logger.info("BUILDING FEATURE MATRIX (Fixed Dynamic Version)")
    logger.info("="*60)
    
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
    # If a feature is in the registry but disabled, we should warn or remove it.
    # Note: Not all features are in the registry (e.g. static ones), so we only check intersection.
    
    registry_enabled_set = get_enabled_features_from_registry(features_cfg)
    
    # If registry has entries, we can check for explicitly disabled ones.
    # To do this, we need the set of ALL registry features first.
    registry_all = set()
    if features_cfg.get("registry"):
        registry_all = {item.get("output_col") for item in features_cfg["registry"] if "output_col" in item}
        
    # Check for conflicts
    final_req_feats = []
    for f in req_feats:
        # If feature is KNOWN to the registry but NOT enabled -> Skip/Error
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
    
    # 5. Build Query
    horizons = models_cfg["horizons"]
    sql = build_dynamic_query(start_date, end_date, t_cols, s_cols, horizons)
    
    # 6. Stream & Process
    out_path = PATHS["data_proc"] / "feature_matrix.parquet"
    chunk_size = 50000
    
    logger.info("Executing Query (Streaming)...")
    chunks = pd.read_sql(sql, engine, chunksize=chunk_size)
    
    master_df = None
    total_rows = 0
    
    for i, chunk in enumerate(chunks):
        # Enforce H3 BigInt (Critical Fix for compatibility)
        chunk['h3_index'] = chunk['h3_index'].astype('int64')
        chunk['date'] = pd.to_datetime(chunk['date'])
        
        if master_df is None:
            master_df = chunk
        else:
            master_df = pd.concat([master_df, chunk], ignore_index=True)
            
        total_rows += len(chunk)
        print(f"  Batch {i+1}: {total_rows} rows accumulated...", end='\r')
        
    print() # Newline
    
    if master_df is None or master_df.empty:
        logger.error("Query returned no data! Check database population.")
        sys.exit(1)

    logger.info(f"Final Matrix Shape: {master_df.shape}")
    
    # 7. Save
    master_df.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")
    
    # 8. Save Meta
    import json
    meta = {
        "start": str(start_date),
        "end": str(end_date),
        "rows": len(master_df),
        "features": req_feats,
        "horizons": [h["name"] for h in horizons]
    }
    with open(PATHS["data_proc"] / "matrix_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    try:
        cfg = load_configs()
        # Handle tuple return if load_configs returns (data, feat, model)
        if isinstance(cfg, tuple):
            configs = {"data": cfg[0], "features": cfg[1], "models": cfg[2]}
        else:
            configs = cfg # Assuming dict
            
        engine = get_db_engine()
        run(configs, engine)
    except Exception as e:
        logger.error(f"Matrix Build Failed: {e}", exc_info=True)
    finally:
        if 'engine' in locals(): engine.dispose()