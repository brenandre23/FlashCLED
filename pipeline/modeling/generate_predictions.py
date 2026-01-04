"""
generate_predictions.py

Usage:
    python generate_predictions.py 14d

This script:
- Loads the canonical feature matrix Parquet
- Loads a trained TwoStageEnsemble bundle for the requested horizon
- Reconstructs PCA features if required
- Generates predictions (probability, magnitude, risk)
- Saves results to CSV and to Postgres (predictions_latest)

FIXES:
- Automatically detects learner suffix (e.g. _xgboost, _lightgbm) to match train_models.py output.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from sqlalchemy.engine import Engine as SqlEngine

# Make sure root is on path so we can import utils
file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parent.parent.parent))

from utils import logger, PATHS, get_db_engine, load_configs, SCHEMA


def load_feature_matrix() -> pd.DataFrame:
    """Load canonical feature matrix used for prediction."""
    feature_matrix_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not feature_matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {feature_matrix_path}. "
            "Make sure build_feature_matrix.py wrote a canonical file."
        )
    logger.info(f"Loading feature matrix from: {feature_matrix_path}")
    df = pd.read_parquet(feature_matrix_path)
    
    if df.empty:
        logger.warning('Feature matrix is empty - downstream features may be missing')
        
    return df


def apply_pca_if_needed(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """
    If the model bundle contains PCA information, recreate the PCA components.
    """
    pca = bundle.get("pca")
    scaler = bundle.get("pca_scaler")
    # Accept both new and legacy keys for backward compatibility
    input_features = bundle.get("pca_input_features") or bundle.get("pca_input")
    component_names = bundle.get("pca_component_names") or bundle.get("pca_cols")

    if pca is None or scaler is None or not input_features:
        logger.info("No PCA configuration in bundle – skipping PCA reconstruction.")
        return df

    logger.info("Reconstructing PCA features for 'broad' model.")

    missing = [c for c in input_features if c not in df.columns]
    if missing:
        # Fill missing PCA inputs with 0 to prevent crash during prediction
        logger.warning(f"Missing PCA inputs: {missing}. Filling with 0.")
        for c in missing:
            df[c] = 0.0

    X_input = df[input_features].fillna(0.0)
    X_scaled = scaler.transform(X_input)
    X_pca = pca.transform(X_scaled)

    n_components = X_pca.shape[1]
    if not component_names or len(component_names) != n_components:
        component_names = [f"pca_{i+1}" for i in range(n_components)]

    df_out = df.copy()
    for idx, name in enumerate(component_names):
        df_out[name] = X_pca[:, idx]

    logger.info(f"Added {n_components} PCA components to feature matrix.")
    return df_out


def resolve_model_path(horizon: str) -> Path:
    """
    Finds the model file, handling the learner suffix (xgboost/lightgbm).
    Prioritizes XGBoost if both exist.
    """
    model_dir = PATHS["root"] / "models"
    
    # 1. Try XGBoost (Default preference)
    p_xgb = model_dir / f"two_stage_ensemble_{horizon}_xgboost.pkl"
    if p_xgb.exists():
        return p_xgb
        
    # 2. Try LightGBM
    p_lgb = model_dir / f"two_stage_ensemble_{horizon}_lightgbm.pkl"
    if p_lgb.exists():
        return p_lgb
        
    # 3. Try legacy generic name
    p_generic = model_dir / f"two_stage_ensemble_{horizon}.pkl"
    if p_generic.exists():
        return p_generic
        
    raise FileNotFoundError(
        f"Could not find model for horizon '{horizon}' in {model_dir}. "
        f"Checked: {p_xgb.name}, {p_lgb.name}, {p_generic.name}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python generate_predictions.py <horizon> (e.g., 14d, 1m, 3m)")
        sys.exit(1)

    horizon = sys.argv[1]
    logger.info(f"=== GENERATING PREDICTIONS FOR HORIZON: {horizon} ===")

    # Load feature matrix
    df = load_feature_matrix()

    # Ensure we have identifiers
    for col in ["h3_index", "date"]:
        if col not in df.columns:
            raise ValueError(f"Feature matrix is missing required column: {col}")

    # Load model bundle (Auto-resolving learner name)
    model_path = resolve_model_path(horizon)
    logger.info(f"Loading model bundle from: {model_path}")
    bundle = joblib.load(model_path)

    if "ensemble" not in bundle:
        raise ValueError(
            f"Loaded bundle at {model_path} does not contain key 'ensemble'."
        )

    ensemble = bundle["ensemble"]

    # Rebuild PCA features if the model uses PCA
    df_features = apply_pca_if_needed(df, bundle)

    X = df_features

    logger.info("Running ensemble.predict() on feature matrix...")
    pred_binary, pred_fatalities = ensemble.predict(X)

    # Combined risk = probability * magnitude
    risk_score = pred_binary * pred_fatalities

    # Assemble prediction DataFrame
    pred_df = pd.DataFrame(
        {
            "h3_index": df["h3_index"].values,
            "date": df["date"].values,
            f"prob_conflict_{horizon}": pred_binary,
            f"expected_fatalities_{horizon}": pred_fatalities,
            f"risk_score_{horizon}": risk_score,
        }
    )

    # Save to CSV
    output_csv = PATHS["data_proc"] / f"predictions_{horizon}.csv"
    logger.info(f"Saving predictions to CSV: {output_csv}")
    pred_df.to_csv(output_csv, index=False)

    # Write to Postgres
    try:
        engine = get_db_engine()
    except Exception as e:
        logger.error(f"Failed to obtain DB engine, skipping DB write: {e}")
    else:
        logger.info(f"Writing predictions to {SCHEMA}.predictions_latest (replace)...")
        # Ensure H3 is BigInt
        pred_df["h3_index"] = pred_df["h3_index"].astype("int64")
        
        pred_df.to_sql(
            "predictions_latest",
            engine,
            schema=SCHEMA,
            if_exists="replace",
            index=False,
        )
        logger.info("DB write complete.")

    logger.info("=== PREDICTION GENERATION COMPLETE ===")


if __name__ == "__main__":
    main()
