"""
generate_fast_shap.py
=====================
Optimized SHAP generation using TreeExplainer decomposition.

Strategy:
1. Load TwoStageEnsemble.
2. Extract thematic sub-models (XGBoost/LightGBM).
3. Compute SHAP for each sub-model using fast TreeExplainer.
4. Scale SHAP values by the Meta-Learner (Logistic Regression) coefficients.
5. Aggregate into a single feature importance view.

This approximates the full ensemble explanation but runs ~100x faster.
"""

import sys
import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
import shap
from scipy.special import expit  # Sigmoid function

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, load_configs, get_db_engine
from pipeline.modeling.generate_predictions import apply_pca_if_needed
from pipeline.analysis.generate_explanations import get_model_bundle, extract_top_k_features, get_theme_from_config
from pipeline.modeling.load_data_utils import sanitize_dataframe

# Setup customized logger
fast_logger = logging.getLogger("FastSHAP")
fast_logger.setLevel(logging.INFO)
if not fast_logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - FAST_SHAP - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fast_logger.addHandler(ch)


def force_clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressively clean '[1.23E-1]' strings to floats."""
    import re
    
    # Identify object columns that might contain these strings
    obj_cols = df.select_dtypes(include=['object']).columns
    
    # Pre-compile regex for performance
    # Matches optional [, number, E/e, -, number, optional ]
    pattern = re.compile(r"[\[\]]") 
    
    for col in obj_cols:
        # Skip non-numeric features like 'h3_index' if it's stored as string
        if col in ['h3_index', 'date', 'top_features']:
            continue
            
        try:
            # Check a sample to see if it looks like the bad data
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample, str) and ('[' in sample or 'E' in sample):
                fast_logger.info(f"Cleaning dirty column: {col}")
                # Remove brackets and convert
                df[col] = df[col].astype(str).str.replace(r'[\[\]]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        except Exception:
            continue
            
    return df

def compute_fast_shap(
    df: pd.DataFrame,
    bundle: dict,
    feature_names: List[str]
) -> Optional[np.ndarray]:
    """
    Compute SHAP values by decomposing the ensemble.
    
    Logic:
    SHAP_total ≈ sum( w_theme * SHAP_theme )
    where w_theme is the LogisticRegression coefficient for that theme.
    """
    ensemble = bundle.get("ensemble")
    if ensemble is None:
        fast_logger.error("No ensemble found in bundle.")
        return None

    if not hasattr(ensemble, "meta_binary") or not hasattr(ensemble, "theme_models"):
        fast_logger.error("Ensemble structure not compatible with fast decomposition.")
        return None

    # 1. Get Meta-Learner Weights (Coefficients or Importance)
    # The meta-learner expects inputs in the order of theme_models
    if hasattr(ensemble.meta_binary, "coef_"):
        meta_weights = ensemble.meta_binary.coef_[0]  # Shape (n_themes,)
        fast_logger.info(f"Meta-Learner Weights (Logistic): {meta_weights}")
    elif hasattr(ensemble.meta_binary, "feature_importances_"):
        # For XGBoost, use feature importance (gain) as weights
        # Note: This is an approximation of the stacking interaction
        meta_weights = ensemble.meta_binary.feature_importances_
        fast_logger.info(f"Meta-Learner Weights (XGBoost Importance): {meta_weights}")
    else:
        fast_logger.warning("Unknown meta-learner type. Using uniform weights.")
        meta_weights = np.ones(len(ensemble.theme_models))
    
    # Initialize total SHAP matrix (rows x features)
    total_shap_values = pd.DataFrame(0.0, index=df.index, columns=feature_names)
    
    # 2. Iterate through each theme
    for i, theme in enumerate(ensemble.theme_models):
        theme_name = theme.get("name", f"theme_{i}")
        weight = meta_weights[i]
        
        # Skip if weight is negligible (optimization)
        if abs(weight) < 1e-4:
            continue
            
        # Get the underlying base model (Classifier)
        # We focus on the Binary Classifier (Onset) as it drives the primary risk probability
        model = theme["binary_model"]
        theme_features = theme["features"]
        
        # Filter input data for this theme
        available_feats = [f for f in theme_features if f in df.columns]
        if not available_feats:
            continue
            
        X_theme = df[available_feats].fillna(0)
        
        # 3. Run Fast TreeExplainer on Base Model
        try:
            # TreeExplainer is fast for XGBoost/LightGBM
            explainer = shap.TreeExplainer(model)
            # check_additivity=False to handle minor precision issues in XGBoost
            shap_values = explainer.shap_values(X_theme, check_additivity=False)
            
            # If binary, shap_values might be list, or (n, features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Take positive class
            
            # 4. Scale by Meta-Weight and Add to Total
            # shap_values is (n_samples, n_theme_features)
            weighted_shap = shap_values * weight
            
            # Add to the corresponding columns in the total matrix
            for col_idx, col_name in enumerate(available_feats):
                if col_name in total_shap_values.columns:
                    total_shap_values[col_name] += weighted_shap[:, col_idx]
                    
        except Exception as e:
            fast_logger.warning(f"Failed to explain theme '{theme_name}': {e}")
            continue

    return total_shap_values.values


def generate_fast_shap_export(
    horizon: str,
    learner: str,
    sample_ratio: float = 0.1,
    top_k: int = 8,
):
    """
    Main driver for fast SHAP generation.
    """
    fast_logger.info(f"=== Starting Fast SHAP Generation: {horizon} | {learner} ===")
    
    # Load bundle
    bundle = get_model_bundle(horizon, learner)
    if bundle is None:
        return

    # Load Data
    feature_matrix_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not feature_matrix_path.exists():
        fast_logger.error("Feature matrix not found.")
        return
        
    df = pd.read_parquet(feature_matrix_path)
    fast_logger.info(f"Loaded {len(df):,} rows.")
    
    # Aggressive cleaning
    df = force_clean_strings(df)
    
    # Subsample
    if sample_ratio < 1.0:
        n_sample = int(len(df) * sample_ratio)
        df = df.sample(n_sample, random_state=42)
        fast_logger.info(f"Subsampled to {len(df):,} rows ({sample_ratio*100}%)")

    # Apply PCA if needed
    df = apply_pca_if_needed(df, bundle)
    
    # Prepare Feature List
    id_cols = ["h3_index", "date"]
    feature_cols = [c for c in df.columns if c not in id_cols and not c.startswith("target_")]
    
    # Load Theme Map
    config = load_configs()
    theme_map = config["models"].get("shap", {}).get("theme_map", {})

    # Compute Fast SHAP
    fast_logger.info("Computing decomposed SHAP values...")
    shap_matrix = compute_fast_shap(df, bundle, feature_cols)
    
    if shap_matrix is None:
        fast_logger.error("SHAP computation failed.")
        return

    # Extract Top-K
    fast_logger.info("Extracting top contributors...")
    records = []
    
    # Pre-compute date strings to speed up loop
    if 'date' in df.columns:
        date_strs = df['date'].astype(str).values
    
    h3_values = df['h3_index'].values
    
    for i in range(len(df)):
        top_features = extract_top_k_features(
            shap_matrix[i],
            feature_cols,
            theme_map,
            top_k=top_k
        )
        
        records.append({
            "h3_index": int(h3_values[i]),
            "date": date_strs[i] if i < len(date_strs) else None,
            "horizon": horizon,
            "learner": learner,
            "top_features": json.dumps(top_features)
        })
        
        if (i + 1) % 10000 == 0:
            fast_logger.info(f"Processed {i+1}/{len(df)} rows...")

    # Save
    out_df = pd.DataFrame(records)
    out_path = PATHS["data_proc"] / f"shap_explanations_{horizon}_{learner}_fast.parquet"
    out_df.to_parquet(out_path, index=False)
    
    fast_logger.info(f"✅ Saved FAST SHAP explanations to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default=None, help="Filter horizon (e.g. 14d). If None, runs all.")
    parser.add_argument("--learner", type=str, default=None, help="Filter learner (e.g. xgboost). If None, runs all.")
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    # Load config to get lists
    config = load_configs()
    all_horizons = [h['name'] for h in config['models']['horizons']]
    all_learners = list(config['models']['learners'].keys())
    
    # Apply filters
    horizons = [args.horizon] if args.horizon else all_horizons
    learners = [args.learner] if args.learner else all_learners
    
    fast_logger.info(f"Running Fast SHAP for: Horizons={horizons}, Learners={learners}")
    
    for h in horizons:
        for l in learners:
            generate_fast_shap_export(h, l, args.sample_ratio)
