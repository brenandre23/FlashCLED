"""
analyze_subtheme_shap.py
========================
Dedicated Micro-Level Explainability Script.
Calculates SHAP values for each sub-model in the Two-Stage Ensemble.

NOTE: This script is computationally intensive. It uses a sample of data
to generate summary plots for each theme (Conflict, Economics, Terrain, etc.).

FIXES APPLIED:
1. Aggressive numeric coercion that checks ALL columns regardless of dtype.
2. PCA reconstruction from model bundle.
3. Graceful handling of constant/empty features.
"""

import sys
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# --- 1. Path Setup & Imports ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from utils import logger, PATHS, load_configs
    from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble
    from pipeline.modeling.load_data_utils import sanitize_dataframe
except ImportError as e:
    print(f"CRITICAL: Could not import project modules. Run from root. Error: {e}")
    sys.exit(1)

# --- Configuration ---
MODEL_FILENAME = "two_stage_ensemble_14d_xgboost.pkl"
OUTPUT_DIR = ROOT_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SHAP Settings
SAMPLE_SIZE = 2000  # Higher than importance script, but still manageable
TEST_DATE_CUTOFF = "2020-12-31"


def load_model_bundle(model_name: str):
    """Loads the full model bundle (Ensemble + PCA + Scaler)."""
    model_path = PATHS["models"] / model_name
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        sys.exit(1)

    logger.info(f"Loading model: {model_path.name}")
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        ensemble = bundle.get("ensemble", bundle)
        return ensemble, bundle
    return bundle, {"ensemble": bundle}


def get_theme_names(configs) -> List[str]:
    """Infers theme names from models.yaml configuration order."""
    submodels = configs["models"]["submodels"]
    return list(submodels.keys())


def _apply_pca_if_needed(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Reconstruct PCA components using artifacts in the bundle (if present).
    
    CRITICAL: This function KEEPS all original columns and ADDS PCA components.
    Other themes need the raw features; only broad_pca uses the PCA components.
    """
    if "pca" not in bundle:
        return df

    logger.info("Reconstructing PCA features for sub-theme SHAP...")
    pca = bundle["pca"]
    scaler = bundle["pca_scaler"]
    pca_inputs = bundle["pca_input_features"]
    pca_cols = bundle["pca_component_names"]

    # Ensure all inputs exist
    missing = [c for c in pca_inputs if c not in df.columns]
    if missing:
        logger.warning(f"PCA reconstruction: {len(missing)} inputs missing (e.g., {missing[:3]}). Filling with 0.")
        for c in missing:
            df[c] = 0.0

    X_ordered = df[pca_inputs].fillna(0)
    X_scaled = scaler.transform(X_ordered)
    comps = pca.transform(X_scaled)

    pca_df = pd.DataFrame(comps, columns=pca_cols, index=df.index)
    
    # CRITICAL FIX: Keep ALL original columns and ADD PCA components
    # Other themes need the raw features; only broad_pca uses PCA components
    return pd.concat([df, pca_df], axis=1)


def run_theme_shap_analysis(ensemble, bundle: dict, df_test: pd.DataFrame, theme_names: List[str]):
    """Iterates through each theme model, calculates SHAP, and saves a plot."""
    logger.info(f"Starting SHAP Analysis on {len(df_test)} samples...")

    # Rebuild PCA columns if the bundle contains them
    df_test = _apply_pca_if_needed(df_test, bundle)

    for i, theme_model_dict in enumerate(ensemble.theme_models):
        theme_name = theme_names[i] if i < len(theme_names) else f"theme_{i}"
        logger.info(f"--- Processing Theme: {theme_name.upper()} ---")

        feature_cols = theme_model_dict["features"]
        missing = [c for c in feature_cols if c not in df_test.columns]
        X_theme = df_test.copy()
        if missing:
            logger.warning(
                f"Skipping {theme_name}: Missing {len(missing)} columns (e.g., {missing[:3]}). Filling with 0."
            )
            for c in missing:
                X_theme[c] = 0

        X_theme = X_theme[feature_cols]
        # Defense-in-depth: strip any bracketed scientific notation that survived
        # sanitize_dataframe() (e.g., '[9.735312E0]' stored as TEXT in DB)
        for _col in X_theme.select_dtypes(include='object').columns:
            X_theme[_col] = X_theme[_col].astype(str).str.replace(r'[\[\]]', '', regex=True)
        X_theme = X_theme.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Skip if all features are constant (no variance to explain)
        if (X_theme.nunique(dropna=False) <= 1).all():
            logger.warning(f"   Skipping {theme_name}: all features constant after fills.")
            continue

        # Guard against sparse themes where the subsample has too few non-zero rows.
        # At ~0.15% event density, a 2000-row random sample may contain < 3 positive rows,
        # causing SHAP's KernelExplainer to crash with an empty-array index error.
        _MIN_POSITIVE_ROWS = 10
        _n_positive = (X_theme != 0).any(axis=1).sum()
        if _n_positive < _MIN_POSITIVE_ROWS:
            logger.warning(
                f"   Skipping {theme_name}: only {_n_positive} non-zero rows in sample "
                f"(need >= {_MIN_POSITIVE_ROWS}). Theme too sparse for SHAP at this sample size."
            )
            continue

        # CRITICAL FIX: Do NOT drop constant columns. 
        # The model expects the full feature vector. SHAP will handle constants naturally (0 importance).
        
        model = theme_model_dict["regress_model"]

        # Try fast TreeExplainer; fall back to model-agnostic Explainer if the serialized booster
        # cannot be parsed by SHAP (common with version-skewed XGBoost pickles).
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_theme, check_additivity=False)
        except Exception as e:
            logger.warning(
                f"   TreeExplainer failed for {theme_name} ({e}); "
                "falling back to model-agnostic Explainer (slower)."
            )
            try:
                # KernelExplainer fallback
                explainer = shap.Explainer(model.predict, X_theme)
                shap_values = explainer(X_theme, silent=True).values
            except Exception as e2:
                logger.error(f"   Failed to calculate SHAP for {theme_name}: {e2}")
                continue

        if shap_values is None or (isinstance(shap_values, np.ndarray) and shap_values.size == 0):
            logger.warning(f"   Skipping {theme_name}: SHAP returned empty array.")
            continue

        plt.figure(figsize=(10, 8))
        plt.title(f"Feature Impact: {theme_name.upper()}", fontsize=14)
        shap.summary_plot(shap_values, X_theme, show=False)

        out_filename = f"shap_summary_{theme_name}.png"
        out_path = OUTPUT_DIR / out_filename

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"   Plot saved to {out_filename}")


def main():
    logger.info("=== SHAP SUB-THEME ANALYSIS START ===")

    cfgs = load_configs()
    if isinstance(cfgs, tuple):
        configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
    else:
        configs = cfgs

    matrix_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not matrix_path.exists():
        logger.error("Feature matrix not found.")
        sys.exit(1)

    logger.info("Loading feature matrix...")
    df = pd.read_parquet(matrix_path)
    
    # Use centralized sanitization
    df, stats = sanitize_dataframe(df, verbose=False)
    if stats['columns_cleaned'] > 0:
        logger.info(f"  Sanitized {stats['columns_cleaned']} columns using load_data_utils.")

    # Apply Pruning Contract
    pruned_path = PATHS["configs"] / "pruned_features.yaml"
    if pruned_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(pruned_path.read_text()) or {}
            active_features = cfg.get("active_features", []) or []
            if active_features:
                logger.info(f"  Filtering to {len(active_features)} active features (Pruning Contract).")
                cols_to_keep = ["h3_index", "date"] + [f for f in active_features if f in df.columns]
                df = df[cols_to_keep]
        except Exception as exc:
            logger.warning(f"Failed to apply pruning registry: {exc}")

    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    test_df = df[df["date"] > pd.Timestamp(TEST_DATE_CUTOFF)].copy()
    test_sample = (
        test_df.sample(n=SAMPLE_SIZE, random_state=42) if len(test_df) > SAMPLE_SIZE else test_df
    )
    logger.info(f"Data Loaded. Using {len(test_sample)} rows for explanation.")

    ensemble, bundle = load_model_bundle(MODEL_FILENAME)
    theme_names = get_theme_names(configs)

    run_theme_shap_analysis(ensemble, bundle, test_sample, theme_names)

    logger.info("=== SHAP ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
