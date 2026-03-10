"""
generate_explanations.py
========================
Batch job to compute SHAP attributions per cell/date/horizon/learner.

Two modes:
1. --shap-export: Per-cell top-K SHAP features for dashboard cell inspector (NEW)
2. Default: Grouped SHAP contributions per feature family for DB ingest (LEGACY)

SHAP Export Mode:
-----------------
Computes SHAP values for sampled rows from feature_matrix.parquet.
Outputs: shap_explanations_{horizon}_{learner}.parquet
Columns: h3_index, date, horizon, learner, top_features (JSON list of {feature, theme, value})

Usage:
    python -m pipeline.analysis.generate_explanations --shap-export
    python -m pipeline.analysis.generate_explanations --shap-export --horizon 3m --learner xgboost
"""

import sys
import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
import shap

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, load_configs, get_db_engine, upload_to_postgis, SCHEMA
from pipeline.modeling.generate_predictions import apply_pca_if_needed


# ---------------------------
# Legacy Feature Group Definitions (for grouped mode)
# ---------------------------
FEATURE_GROUPS: Dict[str, List[str]] = {
    "environment": [
        "chirps_", "era5_", "ndvi", "soil_moisture", "nightlights", "temp", "precip"
    ],
    "past_conflict": [
        "fatalities", "protest", "riot", "regional_risk_score"
    ],
    "gdelt_news": [
        "gdelt_", "crisiswatch_", "acled_topic_"
    ],
    "markets": [
        "price_", "maize", "rice", "cassava", "sorghum", "groundnuts", "oil_price", "gold_price", "sp500", "eur_usd"
    ],
    "displacement": [
        "iom_displacement"
    ],
    "population": [
        "pop_", "population"
    ],
    "distance": [
        "dist_to_", "distance_", "terrain_ruggedness", "elevation", "slope"
    ],
    "admin": [
        "admin1", "admin2", "admin3"
    ],
    "pca": [
        "pca_"
    ],
    "other": [
        ""
    ]
}


def assign_group(feature: str) -> str:
    """Assigns a feature name to a group based on prefix matching."""
    for group, prefixes in FEATURE_GROUPS.items():
        for pref in prefixes:
            if feature.startswith(pref):
                return group
    return "other"


def get_theme_from_config(feature: str, theme_map: Dict[str, List[str]]) -> str:
    """
    Assigns a feature to a theme based on config-driven prefix mapping.
    
    Args:
        feature: Feature name
        theme_map: Dict of theme -> list of prefixes from models.yaml
        
    Returns:
        Theme name or "other" if no match
    """
    feature_lower = feature.lower()
    for theme, prefixes in theme_map.items():
        for prefix in prefixes:
            if feature_lower.startswith(prefix.lower()):
                return theme
    return "other"


def get_model_bundle(horizon: str, learner: str) -> Optional[dict]:
    """Load trained model bundle from disk."""
    model_path = PATHS["models"] / f"two_stage_ensemble_{horizon}_{learner}.pkl"
    if not model_path.exists():
        logger.error(f"Model bundle not found: {model_path}")
        return None
    try:
        bundle = joblib.load(model_path)
        return bundle
    except Exception as e:
        logger.error(f"Failed to load model bundle {model_path}: {e}")
        return None


def compute_shap_values(
    df: pd.DataFrame,
    bundle: dict,
    feature_names: List[str],
    background_size: int = 200
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for the given dataframe using the model bundle's ensemble.
    
    Tries TreeExplainer first (for tree-based models); falls back to KernelExplainer.
    
    Returns:
        SHAP values array of shape (n_samples, n_features) or None if failed
    """
    ensemble = bundle.get("ensemble")
    if ensemble is None:
        logger.error("No ensemble in bundle; cannot compute SHAP.")
        return None

    # Use a small background sample for KernelExplainer
    background = df.sample(min(len(df), background_size), random_state=42)

    # Wrap ensemble.predict to return probabilities only (SHAP expects single output)
    def model_fn(X_input):
        if isinstance(X_input, pd.DataFrame):
            probs, _ = ensemble.predict(X_input)
            return probs
        else:
            X_df = pd.DataFrame(X_input, columns=feature_names)
            probs, _ = ensemble.predict(X_df)
            return probs

    # Attempt generic Explainer first
    try:
        explainer = shap.Explainer(model_fn, background, feature_names=feature_names)
        shap_values = explainer(df)
        return shap_values.values if hasattr(shap_values, "values") else shap_values
    except Exception as e:
        logger.warning(f"Explainer failed ({e}); trying KernelExplainer.")
    
    try:
        explainer = shap.KernelExplainer(model_fn, background)
        shap_values = explainer.shap_values(df, nsamples=200)
        # KernelExplainer returns a list for multi-output; take first
        if isinstance(shap_values, list):
            return shap_values[0]
        return shap_values
    except Exception as e:
        logger.error(f"KernelExplainer failed: {e}")
        return None


def extract_top_k_features(
    shap_row: np.ndarray,
    feature_names: List[str],
    theme_map: Dict[str, List[str]],
    top_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Extract top-K features by absolute SHAP value for a single row.
    
    Args:
        shap_row: 1D array of SHAP values for one sample
        feature_names: List of feature names
        theme_map: Theme mapping from config
        top_k: Number of top features to return
        
    Returns:
        List of dicts: [{feature, theme, value}, ...] sorted by |value| descending
    """
    # Create (feature, value) pairs
    feature_values = list(zip(feature_names, shap_row))
    
    # Sort by absolute value descending
    feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Take top-K
    top_features = []
    for feat, val in feature_values[:top_k]:
        theme = get_theme_from_config(feat, theme_map)
        top_features.append({
            "feature": feat,
            "theme": theme,
            "value": float(val)
        })
    
    return top_features


# =============================================================================
# SHAP EXPORT MODE (NEW)
# =============================================================================

def generate_shap_export(
    horizon: str,
    learner: str,
    sample_ratio: float = 0.1,
    top_k: int = 8,
    background_size: int = 200,
    theme_map: Optional[Dict[str, List[str]]] = None,
    upload_db: bool = False
) -> Optional[Path]:
    """
    Generate per-cell SHAP explanations for dashboard cell inspector.
    
    Outputs parquet with columns:
    - h3_index (int64)
    - date
    - horizon
    - learner
    - top_features (JSON string of [{feature, theme, value}, ...])
    """
    logger.info(f"=== SHAP Export: {horizon} | {learner} ===")
    
    # Load model bundle
    bundle = get_model_bundle(horizon, learner)
    if bundle is None:
        return None
    
    # Load feature matrix
    feature_matrix_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not feature_matrix_path.exists():
        logger.error(f"Feature matrix not found: {feature_matrix_path}")
        return None
    
    df = pd.read_parquet(feature_matrix_path)
    logger.info(f"Loaded feature matrix: {len(df):,} rows")
    
    # Subsample for performance
    if sample_ratio < 1.0:
        n_sample = max(1, int(len(df) * sample_ratio))
        df = df.sample(n_sample, random_state=42)
        logger.info(f"Subsampled to {len(df):,} rows ({sample_ratio*100:.1f}%)")

    # Apply PCA if needed
    df = apply_pca_if_needed(df, bundle)
    
    # Ensure types for IDs
    if 'date' in df.columns and not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
    if 'h3_index' in df.columns:
        df['h3_index'] = df['h3_index'].astype('int64')

    # Identify feature columns (align to model bundle if available)
    id_cols = ["h3_index", "date"]
    # Only exclude raw targets, not lagged features like fatalities_lag1
    target_cols = [c for c in df.columns if c.startswith("target_") or c == "fatalities"]
    bundle_features = bundle.get("feature_names") or bundle.get("feature_cols") or []

    if bundle_features:
        # Keep only features known to the model; add missing as NaN
        missing = [c for c in bundle_features if c not in df.columns]
        for m in missing:
            df[m] = np.nan
        feature_cols = bundle_features
    else:
        feature_cols = [c for c in df.columns if c not in id_cols + target_cols]

    # Compute SHAP values
    logger.info(f"Computing SHAP values for {len(df):,} samples...")
    shap_values = compute_shap_values(
        df[feature_cols],
        bundle,
        feature_names=feature_cols,
        background_size=background_size
    )
    
    if shap_values is None:
        logger.error("SHAP computation failed")
        return None
    
    shap_arr = shap_values.values if hasattr(shap_values, "values") else shap_values
    
    if shap_arr.shape[0] != len(df):
        logger.error(f"SHAP values row count mismatch: {shap_arr.shape[0]} vs {len(df)}")
        return None
    
    logger.info(f"SHAP values computed: {shap_arr.shape}")
    
    # Default theme map if not provided
    if theme_map is None:
        theme_map = {}
    
    # Build output records
    records = []
    for i, (idx, row) in enumerate(df.iterrows()):
        top_features = extract_top_k_features(
            shap_arr[i],
            feature_cols,
            theme_map,
            top_k=top_k
        )
        
        records.append({
            "h3_index": int(row["h3_index"]),
            "date": row["date"],
            "horizon": horizon,
            "learner": learner,
            "top_features": json.dumps(top_features)  # Store as JSON string
        })
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Processed {i + 1}/{len(df)} rows...")
    
    result_df = pd.DataFrame(records)
    
    # Save to parquet
    out_path = PATHS["data_proc"] / f"shap_explanations_{horizon}_{learner}.parquet"
    result_df.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved SHAP explanations to {out_path} ({len(result_df):,} rows)")
    
    # Optionally upload to DB
    if upload_db:
        try:
            engine = get_db_engine()
            upload_to_postgis(
                engine,
                result_df,
                table_name="shap_explanations",
                schema=SCHEMA,
                primary_keys=["h3_index", "date", "horizon", "learner"],
            )
            logger.info(f"✅ Upserted SHAP explanations to {SCHEMA}.shap_explanations")
        except Exception as e:
            logger.error(f"Failed to upload SHAP explanations to DB: {e}")
    
    # Cleanup
    del shap_values, result_df, df
    gc.collect()
    
    return out_path


def run_shap_export(
    horizon_filter: Optional[str] = None,
    learner_filter: Optional[str] = None,
    upload_db: bool = False,
    n_jobs: int = 1
):
    """
    Run SHAP export for all horizon/learner combinations (or filtered subset).
    """
    config = load_configs()
    shap_config = config["models"].get("shap", {})
    
    if not shap_config.get("enabled", True):
        logger.warning("SHAP export is disabled in config")
        return
    
    # Get parameters from config
    sample_ratio = shap_config.get("sample_ratio", 0.1)
    top_k = shap_config.get("top_k", 8)
    background_size = shap_config.get("background_size", 200)
    theme_map = shap_config.get("theme_map", {})
    
    # Determine horizons and learners
    all_horizons = [h['name'] for h in config['models']['horizons']]
    all_learners = list(config['models']['learners'].keys())
    
    # Apply config-level filters
    config_horizons = shap_config.get("horizons")
    config_learners = shap_config.get("learners")
    
    if config_horizons:
        all_horizons = [h for h in all_horizons if h in config_horizons]
    if config_learners:
        all_learners = [l for l in all_learners if l in config_learners]
    
    # Apply CLI-level filters
    if horizon_filter:
        all_horizons = [h for h in all_horizons if h == horizon_filter]
    if learner_filter:
        all_learners = [l for l in all_learners if l == learner_filter]
    
    tasks = [(h, l) for h in all_horizons for l in all_learners]
    
    logger.info(f"🎬 SHAP EXPORT: {len(tasks)} tasks ({len(all_horizons)}H x {len(all_learners)}L)")
    logger.info(f"   sample_ratio={sample_ratio}, top_k={top_k}, background_size={background_size}, n_jobs={n_jobs}")
    
    if n_jobs == 1:
        for horizon, learner in tasks:
            generate_shap_export(
                horizon=horizon,
                learner=learner,
                sample_ratio=sample_ratio,
                top_k=top_k,
                background_size=background_size,
                theme_map=theme_map,
                upload_db=upload_db
            )
    else:
        try:
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs)(
                delayed(generate_shap_export)(
                    horizon=h,
                    learner=l,
                    sample_ratio=sample_ratio,
                    top_k=top_k,
                    background_size=background_size,
                    theme_map=theme_map,
                    upload_db=upload_db
                )
                for h, l in tasks
            )
        except Exception as e:
            logger.error(f"Parallel execution failed ({e}); falling back to sequential.")
            for h, l in tasks:
                generate_shap_export(
                    horizon=h,
                    learner=l,
                    sample_ratio=sample_ratio,
                    top_k=top_k,
                    background_size=background_size,
                    theme_map=theme_map,
                    upload_db=upload_db
                )


# =============================================================================
# LEGACY GROUPED MODE
# =============================================================================

def group_shap_values(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Sum SHAP contributions by feature group (legacy mode).
    Returns DataFrame with columns: group, contribution.
    """
    contributions = {}
    for val, feat in zip(shap_values, feature_names):
        group = assign_group(feat)
        contributions[group] = contributions.get(group, 0.0) + float(val)
    rows = [{"feature_group": g, "contribution": v} for g, v in contributions.items()]
    return pd.DataFrame(rows)


def generate_explanations_for_hl(
    horizon: str,
    learner: str,
    sample_rows: Optional[int] = None,
    upload_db: bool = False,
) -> Optional[Path]:
    """Compute grouped SHAP for a single (horizon, learner) pair (legacy mode)."""
    bundle = get_model_bundle(horizon, learner)
    if bundle is None:
        return None

    # Load features
    df = pd.read_parquet(PATHS["data_proc"] / "feature_matrix.parquet")
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)

    # Rebuild PCA columns as needed
    df = apply_pca_if_needed(df, bundle)

    # Keep track of identifiers
    id_cols = ["h3_index", "date"]
    feature_cols = [c for c in df.columns if c not in id_cols]

    shap_values = compute_shap_values(df[feature_cols], bundle, feature_names=feature_cols)
    if shap_values is None:
        return None

    shap_arr = shap_values.values if hasattr(shap_values, "values") else shap_values
    if shap_arr.shape[0] != len(df):
        logger.error("SHAP values row count does not match dataframe rows.")
        return None

    # Build long-format attribution table
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        grouped = group_shap_values(shap_arr[i], feature_cols)
        for _, g_row in grouped.iterrows():
            records.append({
                "h3_index": row["h3_index"],
                "date": row["date"],
                "horizon": horizon,
                "learner": learner,
                "feature_group": g_row["feature_group"],
                "contribution": g_row["contribution"]
            })

        if (i + 1) % 5000 == 0:
            logger.info(f"Processed SHAP for {i + 1}/{len(df)} rows...")

    result_df = pd.DataFrame(records)
    out_path = PATHS["data_proc"] / f"explanations_{horizon}_{learner}.parquet"
    result_df.to_parquet(out_path, index=False)
    logger.info(f"Saved grouped SHAP explanations to {out_path} ({len(result_df):,} rows)")

    if upload_db:
        try:
            engine = get_db_engine()
            upload_to_postgis(
                engine,
                result_df,
                table_name="explanations",
                schema=SCHEMA,
                primary_keys=["h3_index", "date", "horizon", "learner", "feature_group"],
            )
            logger.info(f"Upserted explanations into {SCHEMA}.explanations")
        except Exception as e:
            logger.error(f"Failed to upload explanations to DB: {e}")

    # Cleanup
    del shap_values, result_df, df
    gc.collect()
    return out_path


def run_legacy_mode(upload_db: bool = False, sample_rows: Optional[int] = None, n_jobs: int = 1):
    """Run legacy grouped SHAP mode."""
    config = load_configs()
    horizons = [h['name'] for h in config['models']['horizons']]
    learners = list(config['models']['learners'].keys())

    tasks = [(h, l) for h in horizons for l in learners]

    if n_jobs == 1:
        for h, l in tasks:
            logger.info(f"=== Generating explanations for {h} | {l} ===")
            generate_explanations_for_hl(h, l, sample_rows=sample_rows, upload_db=upload_db)
    else:
        try:
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs)(
                delayed(generate_explanations_for_hl)(h, l, sample_rows=sample_rows, upload_db=upload_db)
                for h, l in tasks
            )
        except Exception as e:
            logger.error(f"Parallel execution failed ({e}); falling back to sequential.")
            for h, l in tasks:
                generate_explanations_for_hl(h, l, sample_rows=sample_rows, upload_db=upload_db)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explanations.")
    
    # Mode selection
    parser.add_argument(
        "--shap-export",
        action="store_true",
        help="Run SHAP export mode (per-cell top-K features for dashboard)"
    )
    
    # Filters
    parser.add_argument("--horizon", type=str, default=None, help="Filter to specific horizon (e.g., 3m)")
    parser.add_argument("--learner", type=str, default=None, help="Filter to specific learner (e.g., xgboost)")
    
    # Common options
    parser.add_argument("--upload-db", action="store_true", help="Upload results to Postgres")
    
    # Legacy mode options
    parser.add_argument("--sample-rows", type=int, default=None, help="[Legacy] Row subsample for faster SHAP")
    parser.add_argument("--n-jobs", type=int, default=1, help="[Legacy] Parallel workers")
    
    args = parser.parse_args()
    
    if args.shap_export:
        # New SHAP export mode for dashboard
        run_shap_export(
            horizon_filter=args.horizon,
            learner_filter=args.learner,
            upload_db=args.upload_db,
            n_jobs=args.n_jobs
        )
    else:
        # Legacy grouped mode
        run_legacy_mode(
            upload_db=args.upload_db,
            sample_rows=args.sample_rows,
            n_jobs=args.n_jobs
        )
