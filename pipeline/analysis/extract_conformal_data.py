"""
extract_conformal_data.py
=========================
Extracts real Conformal Prediction (BCCP) diagnostics by evaluating 
the model on the test set. This provides the 'Empirical Coverage' 
needed for Figure 5.6B.
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Path setup
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from utils import logger, PATHS

# Parameters
TEST_DATE_CUTOFF = "2021-01-01"
MODEL_PATH = PATHS["models"] / "two_stage_ensemble_14d_xgboost.pkl"
FEATURE_MATRIX_PATH = PATHS["data_proc"] / "feature_matrix.parquet"
OUTPUT_PATH = PATHS["root"] / "analysis" / "conformal_diagnostics.csv"

def run_conformal_evaluation():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    print(f"Loading bundle: {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    ensemble = bundle.get('ensemble')
    bccp = bundle.get('bccp')
    
    if not bccp:
        print("No BCCP object in bundle.")
        return

    if not FEATURE_MATRIX_PATH.exists():
        print(f"Feature matrix not found: {FEATURE_MATRIX_PATH}")
        return

    print(f"Loading feature matrix: {FEATURE_MATRIX_PATH}...")
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to test set
    test_df = df[df['date'] >= pd.Timestamp(TEST_DATE_CUTOFF)].copy()
    print(f"Test set size: {len(test_df):,} rows")

    # Get target and predictions
    # We need the target on the scale BCCP expects (usually log1p if log_scale=True)
    target_col = "target_1_step" 
    if target_col not in test_df.columns:
        # Try finding any target column
        candidates = [c for c in test_df.columns if "target" in c and "1_step" in c]
        if candidates:
            target_col = candidates[0]
        else:
            # Fallback to fatalities_14d_sum if needed
            target_col = "fatalities_14d_sum"
    
    print(f"Using target column: {target_col}")
    
    # Drop rows where target is NaN (future dates)
    test_df = test_df.dropna(subset=[target_col])
    
    # Generate point predictions
    print("Generating point predictions...")
    # BCCP expects predictions on log1p scale if it was fitted that way
    # Most TwoStageEnsemble models return (prob, intensity)
    # Intensity is already E[y|y>0] on log1p scale? 
    # Actually, ensemble.predict(X) returns (probs, fatal_preds)
    # fatal_preds is expected fatalities E[y] = P(y>0) * E[y|y>0]
    
    # For BCCP evaluation, we need to know what scale it was fitted on.
    # Looking at conformal_prediction.py, it has a log_scale attribute.
    
    try:
        # Use only needed features
        all_features = set()
        for theme in ensemble.theme_models:
            all_features.update(theme.get('features', []))
        feature_cols = [f for f in all_features if f in test_df.columns]
        
        X_test = test_df[feature_cols].fillna(0)
        probs, fatal_preds = ensemble.predict(X_test)
        
        # BCCP expects input on log1p scale if bccp.log_scale is True
        if bccp.log_scale:
            y_pred_bccp = np.log1p(fatal_preds)
            y_true_bccp = np.log1p(test_df[target_col].values)
        else:
            y_pred_bccp = fatal_preds
            y_true_bccp = test_df[target_col].values
            
        print("Evaluating coverage...")
        results = bccp.evaluate_coverage(y_true_bccp, y_pred_bccp)
        
        # Format for Figure 5.6B
        # Expected: Bin_ID, Empirical_Coverage, Target_Coverage
        data = []
        for bin_idx, coverage in results['coverage_by_bin'].items():
            data.append({
                "Bin_ID": bin_idx + 1,
                "Range": f"Bin {bin_idx}", # Could be more descriptive
                "Empirical_Coverage": coverage,
                "Target_Coverage": results['target_coverage'],
                "Count": results['counts_by_bin'].get(bin_idx, 0)
            })
            
        output_df = pd.DataFrame(data)
        output_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved real diagnostics to {OUTPUT_PATH}")
        print(output_df)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_conformal_evaluation()