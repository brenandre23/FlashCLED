"""
analyze_sensitivity.py
======================
Experiment to assess the effect of class-weighting on sensitivity.

We vary the XGBoost `scale_pos_weight` via multipliers:
  - 0.0 -> disable weighting (use 1.0)
  - 1.0 -> standard weighting (n_neg / n_pos)
  - 2.0 -> high sensitivity (double the standard weight)

For the 14d horizon only, we:
  1) Load feature_matrix.parquet.
  2) Train/Test split on date (train <= 2020-12-31, test > 2020-12-31).
  3) Fit a temporary TwoStageEnsemble (XGBoost learners) per weight setting.
  4) Evaluate Recall @ 10% FPR and Precision.
  5) Save a sensitivity trade-off plot to analysis/sensitivity_experiment.png.

Output: prints a Markdown table with metrics.

FIXES (2026-01-25):
- Pre-flight validation: Skip submodels with 0 valid features in the data
- Graceful handling: Log warnings for missing features instead of crashing
- Guard against empty DataFrames being passed to XGBoost
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, precision_score

# Project imports
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, load_configs  # noqa: E402
from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble  # noqa: E402

try:
    import xgboost as xgb  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise ImportError("XGBoost is required for this experiment.") from e


# ==============================================================================
# CONSTANTS
# ==============================================================================
MIN_FEATURES_PER_SUBMODEL = 1  # Minimum features required to include a submodel
MODEL_FILENAME = "two_stage_ensemble_14d_xgboost.pkl"


def apply_pca_if_needed(df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """
    Reconstruct PCA components using artifacts in the saved model bundle, if present.
    """
    if not isinstance(model_bundle, dict) or "pca" not in model_bundle:
        return df

    pca = model_bundle["pca"]
    scaler = model_bundle["pca_scaler"]
    pca_inputs = model_bundle["pca_input_features"]
    pca_cols = model_bundle["pca_component_names"]

    missing = [c for c in pca_inputs if c not in df.columns]
    if missing:
        logger.warning(f"PCA reconstruction: {len(missing)} inputs missing (e.g., {missing[:3]}). Filling with 0.")
        for c in missing:
            df[c] = 0.0

    X_ordered = df[pca_inputs].fillna(0)
    comps = pca.transform(scaler.transform(X_ordered))
    pca_df = pd.DataFrame(comps, columns=pca_cols, index=df.index)
    return pd.concat([df, pca_df], axis=1)


def get_train_test_split(df: pd.DataFrame, cutoff: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by date cutoff."""
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
    cutoff_dt = pd.to_datetime(cutoff)
    train = df[df["date"] <= cutoff_dt].copy()
    test = df[df["date"] > cutoff_dt].copy()
    return train, test


def validate_submodel_features(
    submodels_cfg: Dict[str, Any],
    available_columns: Set[str],
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Validate which features exist in the data for each submodel.
    
    Returns:
        validated_submodels: Dict of submodels with only valid features
        diagnostics: Dict mapping submodel name -> list of missing features
    
    Raises:
        ValueError: If ALL submodels have 0 valid features (cannot proceed)
    """
    validated_submodels = {}
    diagnostics = {}
    
    for name, cfg in submodels_cfg.items():
        if not cfg.get("enabled", False):
            continue
        
        requested_features = cfg.get("features", [])
        if not requested_features:
            logger.warning(f"Submodel '{name}' has no features configured. Skipping.")
            diagnostics[name] = ["(no features configured)"]
            continue
        
        # Filter to only features that exist in the data
        valid_features = [f for f in requested_features if f in available_columns]
        missing_features = [f for f in requested_features if f not in available_columns]
        
        if missing_features:
            logger.warning(
                f"Submodel '{name}': {len(missing_features)}/{len(requested_features)} "
                f"features missing from data: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
            )
            diagnostics[name] = missing_features
        
        if len(valid_features) < MIN_FEATURES_PER_SUBMODEL:
            logger.warning(
                f"Submodel '{name}' has only {len(valid_features)} valid features "
                f"(minimum: {MIN_FEATURES_PER_SUBMODEL}). SKIPPING this submodel."
            )
            continue
        
        # Create validated config with only valid features
        validated_cfg = cfg.copy()
        validated_cfg["features"] = valid_features
        validated_submodels[name] = validated_cfg
        
        logger.info(
            f"Submodel '{name}': {len(valid_features)}/{len(requested_features)} features valid"
        )
    
    # Fatal check: at least one submodel must be usable
    if not validated_submodels:
        raise ValueError(
            "No submodels have valid features in the data. Cannot proceed with sensitivity analysis.\n"
            f"Diagnostics: {diagnostics}"
        )
    
    return validated_submodels, diagnostics


def build_theme_models_xgb(
    submodels_cfg: Dict[str, Any],
    learner_cfg: Dict[str, Any],
    weight: float,
    available_columns: Set[str],
) -> List[Dict[str, Any]]:
    """
    Construct XGBoost base learners with injected scale_pos_weight.
    
    CRITICAL FIX: Only includes submodels with features that exist in the data.
    This prevents empty DataFrame errors in XGBoost.
    
    Args:
        submodels_cfg: Submodel configurations from models.yaml
        learner_cfg: XGBoost learner configuration
        weight: scale_pos_weight value to inject
        available_columns: Set of column names available in the feature matrix
    
    Returns:
        List of theme model dicts with validated features
    """
    # First validate features
    validated_submodels, diagnostics = validate_submodel_features(
        submodels_cfg, available_columns
    )
    
    if diagnostics:
        logger.info("=== Feature Validation Diagnostics ===")
        for name, missing in diagnostics.items():
            if missing:
                logger.info(f"  {name}: {len(missing)} missing features")
    
    theme_models: List[Dict[str, Any]] = []
    hyper = learner_cfg["params"].copy()
    
    # Remove parameters that will be set explicitly or cause issues
    hyper.pop("scale_pos_weight", None)
    hyper.pop("verbose", None)
    hyper.pop("n_jobs", None)
    
    ClsClassifier, ClsRegressor = xgb.XGBClassifier, xgb.XGBRegressor

    for name, cfg in validated_submodels.items():
        features = cfg["features"]
        
        # Double-check: should never be empty due to validation above
        if not features:
            logger.error(f"BUG: Submodel '{name}' passed validation but has no features!")
            continue
        
        clf = ClsClassifier(
            **hyper,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=weight,
            n_jobs=-1,
        )
        reg = ClsRegressor(
            **hyper,
            objective="count:poisson",
            eval_metric="poisson-nloglik",
            n_jobs=-1,
        )

        theme_models.append({
            "name": name,
            "features": features,
            "binary_model": clf,
            "regress_model": reg,
        })
        
        logger.debug(f"Built theme model '{name}' with {len(features)} features")

    logger.info(f"Built {len(theme_models)} theme models for ensemble")
    return theme_models


def recall_precision_at_fpr(
    y_true: np.ndarray,
    prob: np.ndarray,
    fpr_target: float = 0.10,
) -> Tuple[float, float]:
    """
    Compute recall at the highest threshold with FPR <= fpr_target,
    and the corresponding precision.
    
    Returns:
        (recall, precision) tuple
    """
    # Guard against edge cases
    if len(y_true) == 0 or len(prob) == 0:
        logger.warning("Empty arrays passed to recall_precision_at_fpr")
        return 0.0, 0.0
    
    if y_true.sum() == 0:
        logger.warning("No positive samples in y_true")
        return 0.0, 0.0
    
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    valid_idxs = np.where(fpr <= fpr_target)[0]
    
    if valid_idxs.size == 0:
        logger.warning(f"No thresholds found with FPR <= {fpr_target}")
        return 0.0, 0.0

    idx = valid_idxs[np.argmax(tpr[valid_idxs])]
    thr = thresholds[idx]
    preds = (prob >= thr).astype(int)

    rec = tpr[idx]
    
    # Precision guard: if no positives predicted, precision_score raises
    if preds.sum() == 0:
        prec = 0.0
    else:
        prec = precision_score(y_true, preds, zero_division=0)
    
    return rec, prec


def main():
    """Main sensitivity analysis workflow."""
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS: Class Weighting Experiment")
    logger.info("=" * 60)
    
    # ------------------------------------------------------------------
    # Config & data loading
    # ------------------------------------------------------------------
    cfg = load_configs()
    if isinstance(cfg, tuple):
        data_cfg, features_cfg, models_cfg = cfg
    else:
        data_cfg, features_cfg, models_cfg = cfg["data"], cfg["features"], cfg["models"]

    # Horizon lookup (14d)
    horizon_name = "14d"
    horizon = next((h for h in models_cfg["horizons"] if h["name"] == horizon_name), None)
    if not horizon:
        raise ValueError(f"Horizon '{horizon_name}' not found in models.yaml.")
    
    # Target column
    target_col = f"target_fatalities_{horizon['steps']}_step"
    logger.info(f"Target column: {target_col}")

    # Load feature matrix
    fm_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not fm_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {fm_path}")
    
    logger.info(f"Loading feature matrix from {fm_path}...")
    df = pd.read_parquet(fm_path)
    logger.info(f"Feature matrix shape: {df.shape}")
    logger.info(f"Available columns: {len(df.columns)}")

    # Reconstruct PCA components if the saved model bundle contains PCA
    bundle_path = PATHS["models"] / MODEL_FILENAME
    if bundle_path.exists():
        try:
            bundle = joblib.load(bundle_path)
            df = apply_pca_if_needed(df, bundle if isinstance(bundle, dict) else {})
            logger.info("Applied PCA reconstruction for sensitivity analysis.")
        except Exception as e:  # pragma: no cover
            logger.warning(f"PCA reconstruction skipped (could not load bundle): {e}")

    if target_col not in df.columns:
        # Check for alternative target column names
        available_targets = [c for c in df.columns if c.startswith("target_")]
        raise ValueError(
            f"Target column '{target_col}' not found in feature matrix.\n"
            f"Available target columns: {available_targets}"
        )

    # Get set of available columns for validation
    available_columns = set(df.columns)
    
    # ------------------------------------------------------------------
    # Train/Test split
    # ------------------------------------------------------------------
    train_df, test_df = get_train_test_split(df, cutoff="2020-12-31")
    
    # Drop rows with missing targets
    train_df = train_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])
    
    logger.info(f"Train set: {len(train_df):,} rows (after dropping NaN targets)")
    logger.info(f"Test set: {len(test_df):,} rows (after dropping NaN targets)")
    
    if len(train_df) == 0:
        raise ValueError("Training set is empty after dropping NaN targets!")
    if len(test_df) == 0:
        raise ValueError("Test set is empty after dropping NaN targets!")

    # Prepare targets
    y_train_bin = (train_df[target_col] > 0).astype(int)
    y_train_reg = train_df[target_col]
    y_test_bin = (test_df[target_col] > 0).astype(int)

    # Class balance statistics
    n_pos = y_train_bin.sum()
    n_neg = len(y_train_bin) - n_pos
    base_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    
    logger.info(f"Training class balance: {n_pos:,} positive / {n_neg:,} negative")
    logger.info(f"Base scale_pos_weight: {base_weight:.2f}")

    # Data for evaluation
    X_train = train_df
    X_test = test_df

    # Learner/submodel configs
    learner_cfg = models_cfg["learners"]["xgboost"]
    submodels_cfg = models_cfg["submodels"]

    # ------------------------------------------------------------------
    # Pre-flight feature validation
    # ------------------------------------------------------------------
    logger.info("\n=== Pre-flight Feature Validation ===")
    try:
        validated_submodels, feature_diagnostics = validate_submodel_features(
            submodels_cfg, available_columns
        )
        logger.info(f"Validated {len(validated_submodels)} submodels for experiment")
    except ValueError as e:
        logger.error(f"Feature validation failed: {e}")
        raise

    # ------------------------------------------------------------------
    # Run sensitivity experiment
    # ------------------------------------------------------------------
    weight_multipliers = [0.0, 1.0, 2.0]
    results = []

    for mult in weight_multipliers:
        # Compute final weight
        final_weight = 1.0 if mult == 0 else base_weight * mult
        logger.info(f"\n{'='*50}")
        logger.info(f"Weight Multiplier: {mult} | scale_pos_weight: {final_weight:.2f}")
        logger.info(f"{'='*50}")

        try:
            # Build theme models with validated features
            themes = build_theme_models_xgb(
                submodels_cfg,
                learner_cfg,
                final_weight,
                available_columns,
            )
            
            if not themes:
                logger.error(f"No valid theme models for multiplier {mult}. Skipping.")
                results.append({
                    "weight_multiplier": mult,
                    "scale_pos_weight": final_weight,
                    "recall_at_10fpr": np.nan,
                    "precision_at_thresh": np.nan,
                    "error": "No valid theme models",
                })
                continue
            
            # Build and fit ensemble
            ensemble = TwoStageEnsemble(theme_models=themes, n_folds=5)
            ensemble.fit(X_train, y_train_bin, y_train_reg)

            # Predict
            probs, _ = ensemble.predict(X_test)
            
            # Evaluate
            rec, prec = recall_precision_at_fpr(y_test_bin.values, probs, fpr_target=0.10)
            
            logger.info(f"Results: Recall@10%FPR={rec:.4f}, Precision={prec:.4f}")

            results.append({
                "weight_multiplier": mult,
                "scale_pos_weight": final_weight,
                "recall_at_10fpr": rec,
                "precision_at_thresh": prec,
            })
            
        except Exception as e:
            logger.error(f"Error for multiplier {mult}: {e}", exc_info=True)
            results.append({
                "weight_multiplier": mult,
                "scale_pos_weight": final_weight,
                "recall_at_10fpr": np.nan,
                "precision_at_thresh": np.nan,
                "error": str(e),
            })

    # ------------------------------------------------------------------
    # Output table (Markdown)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    print("\n| weight_multiplier | scale_pos_weight | recall@10%FPR | precision |")
    print("|---|---|---|---|")
    for r in results:
        recall_str = f"{r['recall_at_10fpr']:.4f}" if not np.isnan(r.get('recall_at_10fpr', np.nan)) else "N/A"
        prec_str = f"{r['precision_at_thresh']:.4f}" if not np.isnan(r.get('precision_at_thresh', np.nan)) else "N/A"
        print(
            f"| {r['weight_multiplier']:.1f} | {r['scale_pos_weight']:.2f} | "
            f"{recall_str} | {prec_str} |"
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    # Filter out failed runs for plotting
    valid_results = [r for r in results if not np.isnan(r.get('recall_at_10fpr', np.nan))]
    
    if not valid_results:
        logger.warning("No valid results to plot!")
        return
    
    out_dir = PATHS["analysis"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitivity_experiment.png"

    plt.figure(figsize=(7, 5))
    for r in valid_results:
        plt.scatter(
            r["recall_at_10fpr"],
            r["precision_at_thresh"],
            s=120,
            label=f"×{r['weight_multiplier']}"
        )
        plt.text(
            r["recall_at_10fpr"] + 0.002,
            r["precision_at_thresh"] + 0.002,
            f"×{r['weight_multiplier']}",
            fontsize=9,
        )
    
    plt.xlabel("Recall @ 10% FPR")
    plt.ylabel("Precision")
    plt.title("Sensitivity Trade-off (scale_pos_weight multipliers)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    logger.info(f"Sensitivity plot saved to {out_path}")
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SENSITIVITY ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    # Report any submodels that were skipped
    if feature_diagnostics:
        logger.info("\nSubmodels with missing features:")
        for name, missing in feature_diagnostics.items():
            logger.info(f"  {name}: {len(missing)} features unavailable")


if __name__ == "__main__":
    main()
