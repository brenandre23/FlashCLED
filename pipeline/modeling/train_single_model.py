"""
train_single_model.py
=====================
ISOLATED WORKER (Hard Exit Version).
Includes 'Skip if Exists' logic and forces OS-level process termination.

Updated: 2026-01-25
Changes: 
1. Fixed BCCP Data Leakage (Fit on Cal set only).
2. Updated RMSE metrics to use Unconditional Expectation.
3. Aligned split logic with TwoStageEnsemble.
"""
import argparse
import sys
import os
import gc
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_poisson_deviance

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


# =============================================================================
# FEATURE IMPORTANCE MONITORING
# =============================================================================

def extract_feature_importance(theme_models: List[Dict], importance_type: str = "total_gain") -> Dict[str, float]:
    """Extract aggregated feature importance across all theme models."""
    importance_agg: Dict[str, float] = {}
    
    for theme in theme_models:
        for model_key in ['binary_model', 'regress_model']:
            model = theme.get(model_key)
            if model is None:
                continue
                
            # XGBoost
            if hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    imp = booster.get_score(importance_type=importance_type)
                    for feat, score in imp.items():
                        importance_agg[feat] = importance_agg.get(feat, 0.0) + score
                except Exception:
                    pass
                    
            # LightGBM
            elif hasattr(model, 'feature_importances_') and hasattr(model, 'feature_name_'):
                try:
                    for feat, score in zip(model.feature_name_, model.feature_importances_):
                        importance_agg[feat] = importance_agg.get(feat, 0.0) + score
                except Exception:
                    pass
    
    return importance_agg


def analyze_nlp_feature_contribution(
    importance: Dict[str, float],
    nlp_features: List[str],
    min_threshold_pct: float = 5.0,
) -> Dict[str, Any]:
    """Analyze whether NLP features are being discovered by the model."""
    if not importance:
        return {"error": "No feature importance available"}
    
    total_importance = sum(importance.values())
    if total_importance == 0:
        return {"error": "Total importance is zero"}
    
    # Calculate NLP contribution
    nlp_importance = {k: v for k, v in importance.items() if k in nlp_features}
    nlp_total = sum(nlp_importance.values())
    nlp_pct = 100.0 * nlp_total / total_importance
    
    # Get top features overall
    sorted_all = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_all[:10]
    
    # Get top NLP features
    sorted_nlp = sorted(nlp_importance.items(), key=lambda x: x[1], reverse=True)
    top_nlp = sorted_nlp[:5] if sorted_nlp else []
    
    # Check for dominance by lag features
    lag_features = [f for f in importance.keys() if 'lag' in f.lower() or 'fatalities' in f.lower()]
    lag_importance = sum(importance.get(f, 0) for f in lag_features)
    lag_pct = 100.0 * lag_importance / total_importance
    
    result = {
        "nlp_contribution_pct": nlp_pct,
        "lag_contribution_pct": lag_pct,
        "top_10_features": [(f, round(s, 2)) for f, s in top_10],
        "top_nlp_features": [(f, round(s, 2)) for f, s in top_nlp],
        "nlp_features_found": len(nlp_importance),
        "nlp_features_monitored": len(nlp_features),
        "alert": nlp_pct < min_threshold_pct,
    }
    
    return result


def log_feature_importance_report(analysis: Dict[str, Any], horizon: str, learner: str) -> None:
    """Log a formatted feature importance report."""
    logger.info("=" * 60)
    logger.info(f"FEATURE IMPORTANCE REPORT: {horizon} | {learner}")
    logger.info("=" * 60)
    
    if "error" in analysis:
        logger.warning(f"⚠️ {analysis['error']}")
        return
    
    # NLP contribution
    nlp_pct = analysis["nlp_contribution_pct"]
    lag_pct = analysis["lag_contribution_pct"]
    
    if analysis["alert"]:
        logger.warning(f"⚠️ NLP features contributing only {nlp_pct:.1f}% (threshold: 5%)")
        logger.warning("   Consider increasing colsample_bytree or reviewing feature engineering.")
    else:
        logger.info(f"✅ NLP features contributing {nlp_pct:.1f}% (healthy)")
    
    logger.info(f"   Autoregressive (lag) features: {lag_pct:.1f}%")
    logger.info(f"   NLP features found: {analysis['nlp_features_found']}/{analysis['nlp_features_monitored']}")
    
    # Top 10 overall
    logger.info("\n📊 Top 10 Features (by total_gain):")
    for i, (feat, score) in enumerate(analysis["top_10_features"], 1):
        marker = "🔵" if any(nlp in feat for nlp in ['cw_', 'mech_', 'regime_', 'narrative_', 'gdelt_shock']) else "⚪"
        logger.info(f"   {i:2d}. {marker} {feat}: {score:.1f}")
    
    # Top NLP features
    if analysis["top_nlp_features"]:
        logger.info("\n🧠 Top NLP Features:")
        for feat, score in analysis["top_nlp_features"]:
            logger.info(f"       {feat}: {score:.1f}")
    
    logger.info("=" * 60)


# =============================================================================
# PCA PROCESSING
# =============================================================================

def process_pca_subsampled(df, config):
    """Fits PCA if enabled."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

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


# =============================================================================
# MODEL FACTORY
# =============================================================================

def build_theme_models(config, learner_name):
    """Factory for initializing sub-models with stochastic gradient boosting."""
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
    reg_params = base_params.copy()

    if learner_type == "xgboost":
        Cls, Reg = XGBClassifier, XGBRegressor
        base_params["n_jobs"] = 1
        reg_params["n_jobs"] = 1
        base_params["objective"] = "binary:logistic"
        base_params["eval_metric"] = "logloss"
        reg_params["objective"] = "count:poisson"
        reg_params["eval_metric"] = "poisson-nloglik"
        
        logger.info(f"XGBoost Stochastic Params: subsample={base_params.get('subsample', 1.0)}, "
                    f"colsample_bytree={base_params.get('colsample_bytree', 1.0)}")
        
    elif learner_type == "lightgbm":
        Cls, Reg = LGBMClassifier, LGBMRegressor
        base_params["num_threads"] = 1
        base_params["verbose"] = -1
        reg_params["num_threads"] = 1
        reg_params["verbose"] = -1
        base_params["objective"] = "binary"
        reg_params["objective"] = "poisson"
        
        logger.info(f"LightGBM Stochastic Params: subsample={base_params.get('subsample', 1.0)}, "
                    f"colsample_bytree={base_params.get('colsample_bytree', 1.0)}")

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
                "regress_model": Reg(**reg_params),
            }
        )
    return theme_models


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run_single_training(horizon, learner_name):
    # 1. SKIP LOGIC
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
            calibration_config=models_cfg["calibration"],  # Pass entire calibration config
        )
        # FIT the ensemble (computes and stores train_idx_ and cal_idx_ internally)
        ensemble.fit(df, y_bin, y_reg)

        # =================================================================
        # USE ENSEMBLE-DEFINED CALIBRATION INDICES (Single Source of Truth)
        # =================================================================
        # DEFENSIVE ASSERTIONS: Ensure calibration indices are valid
        assert hasattr(ensemble, "cal_idx_"), "Calibration indices missing from ensemble"
        assert hasattr(ensemble, "train_idx_"), "Training indices missing from ensemble"
        assert len(ensemble.cal_idx_) > 0, "Empty calibration set"
        assert len(set(ensemble.cal_idx_) & set(ensemble.train_idx_)) == 0, \
            "Train/Cal indices overlap! Data leakage detected."
        
        cal_idx = ensemble.cal_idx_
        
        # Isolate the Calibration Set using ensemble-defined indices
        df_cal = df.iloc[cal_idx]
        y_reg_cal = y_reg.iloc[cal_idx]
        
        logger.info(f"Using ensemble-defined calibration set: {len(cal_idx):,} rows")
        
        # Predict on Calibration Set ONLY
        # Returns: prob, expected_fatalities (unconditional)
        _, expected_fatalities_cal = ensemble.predict(df_cal)

        # =================================================================
        # FEATURE IMPORTANCE MONITORING
        # =================================================================
        monitoring_cfg = models_cfg.get("feature_monitoring", {})
        if monitoring_cfg.get("enabled", False):
            importance = extract_feature_importance(theme_models)
            nlp_features = monitoring_cfg.get("nlp_features", [])
            min_threshold = monitoring_cfg.get("min_nlp_importance_pct", 5.0)
            
            analysis = analyze_nlp_feature_contribution(
                importance, nlp_features, min_threshold
            )
            log_feature_importance_report(analysis, horizon, learner_name)
            pca_bundle["feature_importance_analysis"] = analysis
            pca_bundle["feature_importance_raw"] = importance

        # =================================================================
        # METRICS & CONFORMAL PREDICTION (On Calibration Set)
        # =================================================================
        logger.info("🔮 Fitting Conformal Intervals (Calibration Set Only)...")

        # Metrics now calculated on Calibration Set using Unconditional Expectation
        y_true_clip = np.clip(y_reg_cal.values, 0, None)
        y_pred_clip = np.clip(expected_fatalities_cal, 0, None)
        
        rmse = np.sqrt(mean_squared_error(y_true_clip, y_pred_clip))
        try:
            mpd = mean_poisson_deviance(y_true_clip, np.maximum(y_pred_clip, 1e-9))
        except ValueError:
            mpd = float("nan")
            
        logger.info(f"Intensity metrics (Cal Set) -> RMSE: {rmse:.4f}, Mean Poisson Deviance: {mpd:.4f}")

        # BCCP Fit on Calibration Set
        y_reg_log_cal = np.log1p(y_reg_cal.values)
        preds_log_cal = np.log1p(expected_fatalities_cal)

        bccp = BinConditionalConformalPredictor(
            bins=create_default_fatality_bins(),
            alpha=models_cfg["conformal_prediction"]["alpha"],
            log_scale=True,
        )
        bccp.fit(y_reg_log_cal, preds_log_cal)

        # =================================================================
        # SAVE MODEL
        # =================================================================
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
        import traceback
        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", required=True)
    parser.add_argument("--learner", required=True)
    args = parser.parse_args()

    run_single_training(args.horizon, args.learner)
    os._exit(0)