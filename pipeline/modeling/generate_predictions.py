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
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Make sure root is on path so we can import utils
file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parent.parent.parent))  # adjust if needed

from utils import logger, PATHS, get_db_engine, load_configs  # noqa: E402

# Keep consistent with build_feature_matrix.py
SCHEMA = "car_cewp"


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
        logger.warning('df is empty - downstream features may be missing')
    if df.empty:
        logger.warning('df is empty - downstream features may be missing')
    if df.empty:
        logger.warning('df is empty - downstream features may be missing')
    return df


def apply_pca_if_needed(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    If the model bundle contains PCA information, recreate the PCA components
    on the new feature matrix.

    Expects bundle keys:
        - 'pca': fitted PCA object (or None)
        - 'pca_scaler': fitted scaler used before PCA
        - 'pca_input_features': list of original feature names for PCA
        - 'pca_component_names': names used for PCA component columns
    """
    pca = bundle.get("pca")
    scaler = bundle.get("pca_scaler")
    input_features = bundle.get("pca_input_features")
    component_names = bundle.get("pca_component_names")

    if pca is None or scaler is None or not input_features:
        logger.info("No PCA configuration in bundle – skipping PCA reconstruction.")
        return df

    logger.info("Reconstructing PCA features for 'broad' model.")

    missing = [c for c in input_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing PCA input features in feature matrix: {missing}"
        )

    X_input = df[input_features].fillna(0.0)
    X_scaled = scaler.transform(X_input)
    X_pca = pca.transform(X_scaled)

    n_components = X_pca.shape[1]
    if not component_names or len(component_names) != n_components:
        # Fallback: name components generically if names are missing/mismatched
        component_names = [f"pca_{i+1}" for i in range(n_components)]

    df_out = df.copy()
    for idx, name in enumerate(component_names):
        df_out[name] = X_pca[:, idx]

    logger.info(f"Added {n_components} PCA components to feature matrix.")
    return df_out


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python generate_predictions.py <horizon> (e.g., 14d, 1m, 3m)")
        sys.exit(1)

    horizon = sys.argv[1]
    logger.info(f"=== GENERATING PREDICTIONS FOR HORIZON: {horizon} ===")

    # Load configs (if you want to use them later, e.g., thresholds)
    try:
        data_config, features_config, models_config = load_configs()
    except Exception:
        # If load_configs returns a single dict in your version, adapt as needed
        logger.warning("load_configs() signature differs; not using configs directly.")
        data_config = features_config = models_config = None

    # Load feature matrix
    df = load_feature_matrix()

    # Ensure we have identifiers
    for col in ["h3_index", "date"]:
        if col not in df.columns:
            raise ValueError(f"Feature matrix is missing required column: {col}")

    # Load model bundle
    model_dir = PATHS["root"] / "models"
    model_path = model_dir / f"two_stage_ensemble_{horizon}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {model_path}. "
            f"Make sure train_models.py saved two_stage_ensemble_{horizon}.pkl"
        )

    logger.info(f"Loading model bundle from: {model_path}")
    bundle = joblib.load(model_path)

    if "ensemble" not in bundle:
        raise ValueError(
            f"Loaded bundle at {model_path} does not contain key 'ensemble'."
        )

    ensemble = bundle["ensemble"]

    # Rebuild PCA features if the model uses PCA
    df_features = apply_pca_if_needed(df, bundle)

    # For prediction, we can just pass the full feature frame; TwoStageEnsemble
    # will slice the needed columns via theme['features'] internally.
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
