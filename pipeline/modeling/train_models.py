"""
train_models.py
===============
Trains the Two-Stage Hurdle Ensemble.
Methodology: Hardcoded Objectives (Poisson/Recall) + Dynamic Class Weighting.
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, load_configs
from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

def get_train_test_split(df, train_cutoff):
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
    cutoff = pd.to_datetime(train_cutoff)
    train = df[df['date'] <= cutoff].copy()
    test = df[df['date'] > cutoff].copy()
    return train, test

def get_learner_constructors(model_type):
    if model_type == "xgboost":
        if xgb is None: raise ImportError("XGBoost not installed.")
        return xgb.XGBClassifier, xgb.XGBRegressor
    elif model_type == "lightgbm":
        if lgb is None: raise ImportError("LightGBM not installed.")
        return lgb.LGBMClassifier, lgb.LGBMRegressor
    else:
        raise ValueError(f"Unknown model type: {model_type}")

PCA_EXCLUDE = {
    'target_fatalities', 'target_binary', 'fatalities_14d_sum',
    'target_1_step', 'target_2_step', 'target_6_step',
    'conflict_binary'
}

def build_theme_models(submodels_cfg, learner_cfg, dynamic_weight):
    """
    Constructs base learners.
    Injects Methodology (Objectives) + Dynamic Weight.
    """
    theme_models = []
    model_type = learner_cfg["type"]
    hyperparams = learner_cfg["params"].copy()  # Pure hyperparameters from YAML
    # Guard against duplicate verbose argument (can be present in YAML)
    hyperparams.pop("verbose", None)
    
    ClsClassifier, ClsRegressor = get_learner_constructors(model_type)
    
    for name, cfg in submodels_cfg.items():
        if not cfg["enabled"]: continue
        
        # METHODOLOGY INJECTION (The Thesis Logic)
        if model_type == "xgboost":
            clf = ClsClassifier(
                **hyperparams,
                objective='binary:logistic',
                eval_metric='aucpr',
                scale_pos_weight=dynamic_weight,  # Runtime Calculation
                n_jobs=-1
            )
            reg = ClsRegressor(
                **hyperparams,
                objective='count:poisson',        # Thesis Requirement
                eval_metric='poisson-nloglik',
                n_jobs=-1
            )
        else: # LightGBM
            clf = ClsClassifier(
                **hyperparams,
                objective='binary',
                metric='average_precision',
                scale_pos_weight=dynamic_weight,  # Runtime Calculation
                n_jobs=-1, verbose=-1
            )
            reg = ClsRegressor(
                **hyperparams,
                objective='poisson',              # Thesis Requirement
                metric='poisson',
                n_jobs=-1, verbose=-1
            )
        
        theme_models.append({
            "name": name,
            "features": cfg["features"],
            "binary_model": clf,
            "regress_model": reg
        })
        
    return theme_models


def validate_training_data(df, models_cfg, horizon):
    """
    Pre-training validation. Raises ValueError on failure.
    """
    errors = []
    target_col = f"target_{horizon['steps']}_step"

    if target_col not in df.columns:
        errors.append(f"Missing target column: {target_col}")
    elif df[target_col].isna().all():
        errors.append(f"Target column {target_col} is entirely null")

    required_features = set()
    for name, cfg in models_cfg["submodels"].items():
        if cfg.get("enabled", False) and name != "broad_pca":
            required_features.update(cfg.get("features", []))

    missing = [f for f in required_features if f not in df.columns]
    if missing:
        errors.append(f"Missing features: {missing}")

    all_null = [f for f in required_features if f in df.columns and df[f].isna().all()]
    if all_null:
        errors.append(f"All-null features: {all_null}")

    high_null = []
    for f in required_features:
        if f in df.columns:
            null_pct = df[f].isna().mean()
            if null_pct > 0.5:
                high_null.append(f"{f} ({null_pct:.1%})")
    if high_null:
        logger.warning(f"High null rate features: {high_null}")

    if errors:
        raise ValueError(
            "Training data validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
    logger.info(f"✓ Training data validated: {len(required_features)} features OK for horizon {horizon['name']}")

def run():
    logger.info("="*60)
    logger.info("MODEL TRAINING (Dynamic Weighting + Poisson)")
    logger.info("="*60)
    
    configs = load_configs()
    data_cfg = configs["data"]
    models_cfg = configs["models"]
    
    matrix_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not matrix_path.exists():
        logger.error("Feature matrix not found.")
        return
        
    df = pd.read_parquet(matrix_path)
    train_df, test_df = get_train_test_split(df, data_cfg["split"]["train_cutoff"])
    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # --- PCA Pre-processing ---
    pca_bundle = {} 
    pca_config = models_cfg["submodels"].get("broad_pca")
    if pca_config and pca_config.get("enabled"):
        logger.info("Building PCA features...")
        all_inputs = set()
        for m_name, m_cfg in models_cfg["submodels"].items():
            if m_name != "broad_pca" and m_cfg.get("enabled"):
                    all_inputs.update(m_cfg["features"])
        all_inputs = all_inputs - PCA_EXCLUDE
        input_list = [f for f in all_inputs if f in train_df.columns]
        logger.info(f"PCA inputs: {len(input_list)} features (excluded {len(PCA_EXCLUDE)} target-adjacent cols)")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_df[input_list].fillna(0))
        pca = PCA(n_components=pca_config.get("variance_retention", 0.90), random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
        
        train_df = pd.concat([train_df, pd.DataFrame(X_pca, columns=pca_cols, index=train_df.index)], axis=1)
        X_test_pca = pca.transform(scaler.transform(test_df[input_list].fillna(0)))
        test_df = pd.concat([test_df, pd.DataFrame(X_test_pca, columns=pca_cols, index=test_df.index)], axis=1)
        
        models_cfg["submodels"]["broad_pca"]["features"] = pca_cols
        pca_bundle = {
            "pca": pca,
            "pca_scaler": scaler,
            "pca_input_features": input_list,
            "pca_component_names": pca_cols
        }

    # --- Training Loop ---
    horizons = models_cfg["horizons"]
    learners = models_cfg.get("learners", {})
    model_dir = PATHS["models"]
    model_dir.mkdir(exist_ok=True)
    results = []

    for h in horizons:
        name = h["name"]
        target_col = f"target_{h['steps']}_step"
        logger.info(f"\n=== Horizon: {name} ===")
        
        train_h = train_df.dropna(subset=[target_col])
        if train_h.empty: continue

        validate_training_data(train_df, models_cfg, h)
        
        X_train = train_h
        y_binary = (train_h[target_col] > 0).astype(int)
        y_reg = train_h[target_col]
        
        # Dynamic Weight Calculation
        n_pos = y_binary.sum()
        n_neg = len(y_binary) - n_pos
        if n_pos < 10: continue
            
        calc_weight = n_neg / n_pos
        final_weight = min(calc_weight, 10000.0) # Safety Cap
        logger.info(f"  Dynamic Scale Weight: {final_weight:.2f} (Prev: {n_pos/len(y_binary):.4%})")

        for learner_name, learner_cfg in learners.items():
            try:
                themes = build_theme_models(models_cfg["submodels"], learner_cfg, dynamic_weight=final_weight)
                ensemble = TwoStageEnsemble(theme_models=themes, n_folds=5)
                ensemble.fit(X_train, y_binary, y_reg)
                
                joblib.dump({"ensemble": ensemble, **pca_bundle}, model_dir / f"two_stage_ensemble_{name}_{learner_name}.pkl")
                
                # Simple Eval
                test_h = test_df.dropna(subset=[target_col])
                if not test_h.empty:
                    probs, _ = ensemble.predict(test_h)
                    auc = roc_auc_score((test_h[target_col] > 0).astype(int), probs)
                    logger.info(f"    {learner_name} Test AUC: {auc:.4f}")
                    results.append({"horizon": name, "learner": learner_name, "auc": auc})
            except Exception as e:
                logger.error(f"Failed {learner_name}: {e}")

    if results:
        res_df = pd.DataFrame(results)
        print("\n=== BENCHMARK RESULTS ===\n", res_df.pivot(index="horizon", columns="learner", values="auc"))

if __name__ == "__main__":
    run()
