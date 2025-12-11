"""
train_models.py
===============
Trains the Two-Stage Hurdle Ensemble for each defined horizon.
Updated for Multi-Learner Benchmarking (XGBoost vs LightGBM).
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, load_configs
from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble

# Optional imports for learner factories
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def get_train_test_split(df, train_cutoff):
    """Splits matrix based on date cutoff."""
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
        
    cutoff = pd.to_datetime(train_cutoff)
    train = df[df['date'] <= cutoff].copy()
    test = df[df['date'] > cutoff].copy()
    return train, test


def get_learner_constructors(model_type):
    """Factory function to get the correct class constructors."""
    if model_type == "xgboost":
        if xgb is None: raise ImportError("XGBoost not installed.")
        return xgb.XGBClassifier, xgb.XGBRegressor
    elif model_type == "lightgbm":
        if lgb is None: raise ImportError("LightGBM not installed.")
        return lgb.LGBMClassifier, lgb.LGBMRegressor
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_theme_models(submodels_cfg, learner_cfg):
    """
    Constructs base learners using the specific learner config.
    """
    theme_models = []
    model_type = learner_cfg["type"]
    params = learner_cfg["params"]
    
    # Get the correct classes (XGB or LGB)
    ClsClassifier, ClsRegressor = get_learner_constructors(model_type)
    
    for name, cfg in submodels_cfg.items():
        if not cfg["enabled"]: continue
        
        # Instantiate specific model type
        # Note: XGB uses objective='binary:logistic', LGB uses objective='binary'
        if model_type == "xgboost":
            clf = ClsClassifier(**params, objective='binary:logistic')
            reg = ClsRegressor(**params, objective='reg:squarederror')
        else:
            # LightGBM objectives
            clf = ClsClassifier(**params, objective='binary')
            reg = ClsRegressor(**params, objective='regression')
        
        theme_models.append({
            "name": name,
            "features": cfg["features"],
            "binary_model": clf,
            "regress_model": reg
        })
        
    return theme_models


def run():
    logger.info("="*60)
    logger.info("MODEL TRAINING BENCHMARK (XGBoost vs LightGBM)")
    logger.info("="*60)
    
    # 1. Load Data & Configs
    configs = load_configs()
    data_cfg = configs["data"]
    models_cfg = configs["models"]
    
    matrix_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not matrix_path.exists():
        logger.error("Feature matrix not found. Run build_feature_matrix.py first.")
        return
        
    df = pd.read_parquet(matrix_path)
    logger.info(f"Loaded Matrix: {len(df)} rows")
    
    # 2. Split Data
    train_cutoff = data_cfg["split"]["train_cutoff"]
    train_df, test_df = get_train_test_split(df, train_cutoff)
    logger.info(f"Train Cutoff: {train_cutoff}")
    logger.info(f"Train Size: {len(train_df)} | Test Size: {len(test_df)}")

    # =========================================================
    # PRE-PROCESSING: Handle PCA / "Broad" Theme Logic
    # (Moved OUTSIDE the loop to avoid re-calculating per horizon)
    # =========================================================
    pca_bundle = {} 
    pca_config = models_cfg["submodels"].get("broad_pca")
    
    if pca_config and pca_config.get("enabled"):
        logger.info("Building PCA features for 'broad_pca' theme...")
        
        # A. Gather all input features from OTHER themes
        all_inputs = set()
        for m_name, m_cfg in models_cfg["submodels"].items():
            if m_name != "broad_pca" and m_cfg.get("enabled"):
                    all_inputs.update(m_cfg["features"])
        
        input_list = list(all_inputs)
        
        # B. Fit PCA on Training Data
        X_in = train_df[input_list].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_in)
        
        # Retain 90% variance (or config value)
        variance = pca_config.get("variance_retention", 0.90)
        pca = PCA(n_components=variance, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # C. Create Column Names
        n_components = X_pca.shape[1]
        pca_cols = [f"pca_{i+1}" for i in range(n_components)]
        logger.info(f"PCA retained {n_components} components (Variance: {variance})")
        
        # D. Inject PCA columns back into DataFrames
        train_df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=train_df.index)
        train_df = pd.concat([train_df, train_df_pca], axis=1)
        
        # Transform Test Data (using fitted scaler/pca)
        X_test_in = test_df[input_list].fillna(0)
        X_test_scaled = scaler.transform(X_test_in)
        X_test_pca = pca.transform(X_test_scaled)
        test_df_pca = pd.DataFrame(X_test_pca, columns=pca_cols, index=test_df.index)
        test_df = pd.concat([test_df, test_df_pca], axis=1)
        
        # E. Update the Config dynamically
        models_cfg["submodels"]["broad_pca"]["features"] = pca_cols
        
        # Store for saving
        pca_bundle = {
            "pca": pca,
            "pca_scaler": scaler,
            "pca_input_features": input_list,
            "pca_component_names": pca_cols
        }

    # 3. Iterate Horizons AND Learners
    horizons = models_cfg["horizons"]
    
    # Fallback if 'learners' is missing (backward compatibility)
    if "learners" not in models_cfg:
        logger.warning("'learners' not found in models.yaml. Using 'base_learner' as 'xgboost' default.")
        learners = {
            "xgboost": {
                "type": models_cfg["base_learner"]["model_type"],
                "params": models_cfg["base_learner"]["xgb_params"]
            }
        }
    else:
        learners = models_cfg["learners"]
    
    model_dir = PATHS["models"]
    model_dir.mkdir(exist_ok=True)
    
    # Store results for comparison
    results = []

    for h in horizons:
        name = h["name"]
        steps = h["steps"]
        target_col = f"target_{steps}_step"
        
        logger.info(f"\n=== Horizon: {name} ===")
        
        # Prepare Data
        train_h = train_df.dropna(subset=[target_col])
        if train_h.empty: continue
        
        X_train = train_h
        y_raw = train_h[target_col]
        y_binary = (y_raw > 0).astype(int)
        y_reg = y_raw
        
        if y_binary.sum() < 10:
            logger.warning(f"Too few positive cases for {name}. Skipping.")
            continue
        
        # --- LOOP THROUGH LEARNERS ---
        for learner_name, learner_cfg in learners.items():
            logger.info(f"  Training {learner_name}...")
            
            try:
                # Build Theme Models for THIS learner
                themes = build_theme_models(models_cfg["submodels"], learner_cfg)
                
                # Initialize Ensemble
                ensemble = TwoStageEnsemble(theme_models=themes, n_folds=5)
                
                # Fit
                ensemble.fit(X_train, y_binary, y_reg)
                
                # Save with learner name in filename
                # e.g., two_stage_ensemble_14d_xgboost.pkl
                out_file = model_dir / f"two_stage_ensemble_{name}_{learner_name}.pkl"
                
                full_bundle = {"ensemble": ensemble, **pca_bundle}
                joblib.dump(full_bundle, out_file)
                
                # Evaluate on Test Set
                test_h = test_df.dropna(subset=[target_col])
                if not test_h.empty:
                    probs, preds = ensemble.predict(test_h)
                    try:
                        auc = roc_auc_score((test_h[target_col] > 0).astype(int), probs)
                        logger.info(f"    {learner_name} Test AUC: {auc:.4f}")
                        results.append({
                            "horizon": name,
                            "learner": learner_name,
                            "auc": auc
                        })
                    except Exception as e:
                        logger.warning(f"AUC Calc failed: {e}")
            
            except Exception as e:
                logger.error(f"Failed to train {learner_name}: {e}")

    # Print Summary Table
    if results:
        logger.info("\n=== BENCHMARK RESULTS ===")
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            print(res_df.pivot(index="horizon", columns="learner", values="auc"))

if __name__ == "__main__":
    run()