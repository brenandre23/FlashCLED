"""
train_single_model.py
=====================
ISOLATED WORKER (Hard Exit Version).
Includes 'Skip if Exists' logic and forces OS-level process termination.
"""
import argparse
import sys
import os
import gc
from pathlib import Path
import pandas as pd
import joblib

# Add root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, load_configs, PATHS
from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble
from pipeline.modeling.conformal_prediction import (
    BinConditionalConformalPredictor,
    create_default_fatality_bins,
)


def process_pca_subsampled(df, config):
    """Fits PCA if enabled (Same as before)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    pca_cfg = config.get("submodels", {}).get("broad_pca", {})
    if not pca_cfg.get("enabled", False):
        return df, {}

    logger.info("🎨 Fitting Broad PCA (Subsampled)...")

    input_features = []
    for name, sub in config["submodels"].items():
        if name != "broad_pca" and sub.get("enabled"):
            input_features.extend(sub.get("features", []))

    valid_inputs = [f for f in set(input_features) if f in df.columns]

    if not valid_inputs:
        return df, {}

    try:
        sample_size = 300000
        if len(df) > sample_size:
            df_sample = (
                df[valid_inputs]
                .sample(n=sample_size, random_state=42)
                .fillna(0)
                .astype(np.float32)
            )
        else:
            df_sample = df[valid_inputs].fillna(0).astype(np.float32)

        scaler = StandardScaler()
        X_sample_scaled = scaler.fit_transform(df_sample.values)

        variance = pca_cfg.get("variance_retention", 0.9)
        pca = PCA(n_components=variance, random_state=42)
        pca.fit(X_sample_scaled)

        pca_cols = [f"pca_{i+1}" for i in range(pca.n_components_)]
        pca_results = []
        chunk_size = 100000

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            X_chunk = (
                df[valid_inputs]
                .iloc[start:end]
                .fillna(0)
                .astype(np.float32)
                .values
            )
            X_chunk = scaler.transform(X_chunk)
            X_chunk_pca = pca.transform(X_chunk)
            pca_results.append(
                pd.DataFrame(
                    X_chunk_pca, columns=pca_cols, index=df.index[start:end]
                )
            )

        pca_df = pd.concat(pca_results, axis=0)
        df_out = pd.concat([df, pca_df], axis=1)

        config["submodels"]["broad_pca"]["features"] = pca_cols
        bundle = {
            "pca": pca,
            "pca_scaler": scaler,
            "pca_input_features": valid_inputs,
            "pca_component_names": pca_cols,
        }
        return df_out, bundle

    except Exception as e:
        logger.error(f"❌ PCA Failed: {e}")
        return df, {}


def build_theme_models(config, learner_name):
    """Factory for initializing sub-models."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        XGBClassifier = XGBRegressor = None
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError:
        LGBMClassifier = LGBMRegressor = None

    theme_models = []
    learner_cfg = config["learners"][learner_name]
    learner_type = learner_cfg["type"]
    base_params = learner_cfg["params"].copy()

    if learner_type == "xgboost":
        Cls, Reg = XGBClassifier, XGBRegressor
        base_params["n_jobs"] = 1
    elif learner_type == "lightgbm":
        Cls, Reg = LGBMClassifier, LGBMRegressor
        base_params["num_threads"] = 1
        base_params["verbose"] = -1

    hurdle_cfg = config.get("hurdle_params", {})
    scale_pos_weight = hurdle_cfg.get("classifier_scale_pos_weight", 1.0)

    for name, info in config["submodels"].items():
        if not info.get("enabled", False):
            continue
        if name == "broad_pca" and not info.get("features"):
            continue

        clf_params = base_params.copy()
        clf_params["scale_pos_weight"] = scale_pos_weight

        theme_models.append(
            {
                "name": name,
                "features": info["features"],
                "binary_model": Cls(**clf_params),
                "regress_model": Reg(**base_params),
            }
        )
    return theme_models


def run_single_training(horizon, learner_name):
    # 1. SKIP LOGIC: Check if model already exists
    filename = f"two_stage_ensemble_{horizon}_{learner_name}.pkl"
    output_path = PATHS["models"] / filename

    if output_path.exists():
        logger.info(f"⏭️ [WORKER] Skipping {filename} (Already exists)")
        return

    logger.info(f"🟢 [WORKER PID {os.getpid()}] Starting: {horizon} | {learner_name}")

    try:
        _, _, models_cfg = load_configs()
        horizon_cfg = next(h for h in models_cfg["horizons"] if h["name"] == horizon)
        lookahead_steps = horizon_cfg["steps"]

        logger.info("📦 Loading Feature Matrix...")
        df = pd.read_parquet(PATHS["data_proc"] / "feature_matrix.parquet")

        target_col = f"target_fatalities_{lookahead_steps}_step"
        if target_col not in df.columns:
            logger.warning(f"⚠️ {target_col} missing. Reconstructing...")
            df.sort_values(["h3_index", "date"], inplace=True)
            source = "fatalities" if "fatalities" in df.columns else "fatalities_14d_sum"
            df[target_col] = df.groupby("h3_index")[source].shift(-lookahead_steps)
            df.dropna(subset=[target_col], inplace=True)

        split_date = pd.to_datetime("2024-01-01")
        train_mask = pd.to_datetime(df["date"]) < split_date
        df = df[train_mask].reset_index(drop=True)
        gc.collect()

        df, pca_bundle = process_pca_subsampled(df, models_cfg)
        y_reg = df[target_col]
        y_bin = (y_reg > 0).astype(int)

        logger.info(f"🏭 Building models for {learner_name}...")
        theme_models = build_theme_models(models_cfg, learner_name)

        ensemble = TwoStageEnsemble(
            theme_models=theme_models,
            calibration_method=models_cfg["calibration"]["method"],
            calibration_fraction=models_cfg["calibration"]["fraction"],
        )
        ensemble.fit(df, y_bin, y_reg)

        logger.info("🔮 Fitting Conformal Intervals...")
        _, cal_preds_fatal = ensemble.predict(df)
        bccp = BinConditionalConformalPredictor(
            bins=create_default_fatality_bins(),
            alpha=models_cfg["conformal_prediction"]["alpha"],
        )
        bccp.fit(y_reg.values, cal_preds_fatal)

        save_dict = {
            "ensemble": ensemble,
            "bccp": bccp,
            "config": horizon_cfg,
            "learner": learner_name,
        }
        save_dict.update(pca_bundle)

        joblib.dump(save_dict, output_path)
        logger.info(f"✅ [WORKER] Success: {filename}")

    except Exception as e:
        logger.error(f"❌ [WORKER] Failed: {e}")
        os._exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--learner", required=True)
    args = parser.parse_args()

    run_single_training(args.horizon, args.learner)

    # Hard exit to avoid lingering threads/handles that can hang WSL
    os._exit(0)
