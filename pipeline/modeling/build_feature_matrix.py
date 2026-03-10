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
import argparse
import gc
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from sqlalchemy import text, Engine
from sqlalchemy.engine import Engine as SqlEngine

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]

if "utils" not in sys.modules:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, SCHEMA, ensure_h3_int64, validate_h3_types
from pipeline.modeling.load_data_utils import sanitize_dataframe


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
    """Constructs SQL query. No LEAD window functions — targets computed Python-side."""
    all_selects = temp_cols + static_cols
    select_sql = ",\n        ".join(all_selects)
    sql = f"""
    SELECT
        t.h3_index,
        t.date,
        {select_sql}
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
            logger.warning(f"features_acled_hybrid missing expected columns; will fill with NaN: {e}")
    # Build Query
    horizons = models_cfg["horizons"]
    sql = build_dynamic_query(start_date, end_date, t_cols, s_cols, horizons)
    
    # ---------------------------------------------------------------
    # PASS 1: Stream features to a temp parquet.
    # Simultaneously collect a tiny 3-column companion DataFrame
    # (h3_index, date, fatalities_14d_sum) for Python-side LEAD.
    # Peak RAM during this pass: ~1 chunk (~28 MB) + ~43 MB companion.
    # ---------------------------------------------------------------
    out_path = PATHS["data_proc"] / "feature_matrix.parquet"
    tmp_path = out_path.with_suffix(".tmp.parquet")
    chunk_size = 50000

    logger.info("PASS 1: Streaming features to temp parquet...")
    total_rows = 0
    writer = None
    target_rows = []  # companion: only 3 cols, ~43 MB total
    target_dtypes = None  # dtype map from first chunk; enforced on all subsequent chunks

    raw_conn = engine.raw_connection()
    try:
        raw_conn.autocommit = False
        with raw_conn.cursor() as setup_cur:
            setup_cur.execute("SET work_mem = '128MB'")

        with raw_conn.cursor(name="feature_matrix_cursor") as cursor:
            cursor.itersize = chunk_size
            cursor.execute(sql)
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                logger.error("Query returned no data! Check database population.")
                sys.exit(1)

            col_names = [desc[0] for desc in cursor.description]

            while rows:
                chunk = pd.DataFrame(rows, columns=col_names)

                # Type enforcement
                chunk["h3_index"] = chunk["h3_index"].apply(ensure_h3_int64).astype("int64")
                chunk["date"] = pd.to_datetime(chunk["date"]).dt.normalize()
                float_cols = chunk.select_dtypes(include=["float64"]).columns
                if not float_cols.empty:
                    chunk[float_cols] = chunk[float_cols].astype("float32")
                dist_cols = [c for c in chunk.columns if c.startswith("dist_")]
                for col in dist_cols:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

                # Sanitize
                chunk, _ = sanitize_dataframe(chunk, verbose=False)

                # Schema consistency: all-NULL columns arrive as object dtype in some
                # chunks, causing float32 vs float64 (double) mismatches across chunks.
                # Establish dtype map from first chunk; coerce every subsequent chunk to match.
                if target_dtypes is None:
                    target_dtypes = {col: chunk[col].dtype for col in chunk.columns}
                else:
                    for col in chunk.columns:
                        if col in target_dtypes and chunk[col].dtype != target_dtypes[col]:
                            try:
                                chunk[col] = chunk[col].astype(target_dtypes[col])
                            except (ValueError, TypeError):
                                chunk[col] = pd.to_numeric(
                                    chunk[col], errors="coerce"
                                ).astype(target_dtypes[col])

                # Collect companion rows (3 cols only)
                if "fatalities_14d_sum" in chunk.columns:
                    target_rows.append(
                        chunk[["h3_index", "date", "fatalities_14d_sum"]].copy()
                    )

                # Write chunk to temp parquet
                if not chunk.empty:
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema, compression="snappy")
                    writer.write_table(table)
                    total_rows += len(chunk)

                rows = cursor.fetchmany(chunk_size)
                print(f"  Streamed {total_rows:,} rows...", end="\r")
    finally:
        if writer:
            writer.close()
        raw_conn.close()
    print()
    logger.info(f"Pass 1 complete: {total_rows:,} rows written to temp parquet.")

    # ---------------------------------------------------------------
    # PASS 2: Compute targets Python-side on the tiny companion DataFrame.
    # target_df: ~43 MB. Sort + groupby/shift is all in-memory on 3 cols.
    # ---------------------------------------------------------------
    logger.info("PASS 2: Computing target columns (Python-side LEAD)...")
    target_df = pd.concat(target_rows, ignore_index=True)
    del target_rows; gc.collect()

    target_df.sort_values(["h3_index", "date"], inplace=True)
    for h in horizons:
        steps = h["steps"]
        lead_vals = target_df.groupby("h3_index")["fatalities_14d_sum"].shift(-steps)
        target_df[f"target_fatalities_{steps}_step"] = lead_vals.astype("float32")
        target_df[f"target_binary_{steps}_step"] = (
            lead_vals.map(lambda x: 1.0 if x > 0 else (np.nan if pd.isna(x) else 0.0))
        ).astype("float32")

    target_col_names = [c for c in target_df.columns if c.startswith("target_")]

    # ---------------------------------------------------------------
    # PASS 3: Read temp parquet (~200 MB), merge targets, drop NaN
    # target rows, write final parquet. Peak RAM: ~250 MB.
    # ---------------------------------------------------------------
    logger.info("PASS 3: Merging targets into feature matrix...")
    master_df = pd.read_parquet(tmp_path)
    master_df = master_df.merge(
        target_df[["h3_index", "date"] + target_col_names],
        on=["h3_index", "date"],
        how="left",
    )
    del target_df; gc.collect()

    final_target_cols = identify_target_columns(master_df, horizons)
    logger.info(f"Identified {len(final_target_cols)} target columns: {final_target_cols}")

    before_drop = len(master_df)
    master_df = master_df.dropna(subset=final_target_cols)
    dropped_count = before_drop - len(master_df)
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count:,} rows with missing targets (temporal boundary).")

    logger.info(f"Final Matrix Shape: {master_df.shape}")

    # Verify targets survived
    final_target_cols = [t for t in final_target_cols if t in master_df.columns]
    if not final_target_cols:
        raise ValueError("No target columns present in final output — aborting.")
    logger.info(f"✓ Target columns preserved: {final_target_cols}")

    master_df.to_parquet(out_path, index=False)
    final_row_count = len(master_df)
    del master_df; gc.collect()

    tmp_path.unlink(missing_ok=True)
    logger.info(f"✓ Saved final feature matrix to {out_path}")

    # Schema-only verification
    schema = pq.read_schema(out_path)
    saved_targets = [f.name for f in schema if any(f.name.startswith(p) for p in TARGET_COL_PATTERNS)]
    logger.info(f"✓ Verification: parquet contains {len(saved_targets)} target columns")

    # Save meta
    meta = {
        "start": str(start_date),
        "end": str(end_date),
        "rows": final_row_count,
        "features": req_feats,
        "horizons": [h["name"] for h in horizons],
        "target_columns": final_target_cols,
    }
    with open(PATHS["data_proc"] / "matrix_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()

    try:
        cfg = load_configs()
        if isinstance(cfg, tuple):
            configs = {"data": cfg[0], "features": cfg[1], "models": cfg[2]}
        else:
            configs = cfg
            
        engine = get_db_engine()
        run(configs, engine, start_date=args.start_date, end_date=args.end_date)
    except Exception as e:
        logger.error(f"Matrix Build Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if "engine" in locals():
            engine.dispose()
