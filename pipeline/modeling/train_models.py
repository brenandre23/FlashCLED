"""
train_models.py
===============
Trains the Two-Stage Hurdle Ensemble for each defined horizon.
Refactored for Phase 5:
- Uses step-based horizons (target_X_step)
- Respects train/test splits from config
- Implements PCA "Broad" theme logic
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
# Import the TwoStageEnsemble class (unchanged, just imported)
from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble
import xgboost as xgb

def get_train_test_split(df, train_cutoff):
    """Splits matrix based on date cutoff."""
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
        
    cutoff = pd.to_datetime(train_cutoff)
    train = df[df['date'] <= cutoff].copy()
    test = df[df['date'] > cutoff].copy()
    return train, test

def build_theme_models(models_cfg):
    """
    Constructs the base learners for the ensemble based on submodels config.
    """
    theme_models = []
    base_params = models_cfg["base_learner"]["xgb_params"]
    
    for name, cfg in models_cfg["submodels"].items():
        if not cfg["enabled"]: continue
        
        # Instantiate fresh models for this theme
        clf = xgb.XGBClassifier(**base_params, objective='binary:logistic')
        reg = xgb.XGBRegressor(**base_params, objective='reg:squarederror')
        
        theme_models.append({
            "name": name,
            "features": cfg["features"],
            "binary_model": clf,
            "regress_model": reg
        })
        
    return theme_models

def run():
    logger.info("="*60)
    logger.info("MODEL TRAINING (Phase 5)")
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
    pca_bundle = {} # Store objects to save later
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

    # 3. Iterate Horizons
    horizons = models_cfg["horizons"] 
    model_dir = PATHS["models"]
    model_dir.mkdir(exist_ok=True)
    
    for h in horizons:
        name = h["name"]
        steps = h["steps"]
        target_col = f"target_{steps}_step"
        
        logger.info(f"\n--- Training Horizon: {name} (Steps: {steps}) ---")
        
        # Prepare Target Vectors
        train_h = train_df.dropna(subset=[target_col])
        
        if train_h.empty:
            logger.warning(f"No training data for {name} (check target column names).")
            continue

        X_train = train_h 
        y_raw = train_h[target_col]
        y_binary = (y_raw > 0).astype(int)
        y_reg = y_raw 
        
        if y_binary.sum() < 10:
            logger.warning("Too few positive cases. Skipping.")
            continue
            
        # Build Ensemble
        themes = build_theme_models(models_cfg)
        ensemble = TwoStageEnsemble(
            theme_models=themes,
            n_folds=5
        )
        
        # Fit
        ensemble.fit(X_train, y_binary, y_reg)
        
        # Save Bundle (Ensemble + PCA objects)
        out_file = model_dir / f"two_stage_ensemble_{name}.pkl"
        
        full_bundle = {
            "ensemble": ensemble,
            **pca_bundle  # Unpack PCA objects into the file
        }
        
        joblib.dump(full_bundle, out_file)
        logger.info(f"Saved model bundle to {out_file}")
        
        # Quick Evaluation on Test Set
        test_h = test_df.dropna(subset=[target_col])
        if not test_h.empty:
            probs, preds = ensemble.predict(test_h)
            try:
                auc = roc_auc_score((test_h[target_col]>0).astype(int), probs)
                logger.info(f"Test AUC: {auc:.4f}")
            except Exception:
                logger.info("Could not calc AUC")

    logger.info("\nTraining Complete.")

if __name__ == "__main__":
    run()