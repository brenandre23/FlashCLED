"""
analyze_model_selection.py
==========================
Model Architecture Justification Analysis for Thesis.

Purpose:
    Compare the Two-Stage Stacked Ensemble against:
    1. Individual Level 1 Sub-models (e.g., Conflict History alone)
    2. XGBoost vs LightGBM variants
    
    Generates ROC/PR curves and summary statistics to justify
    the complexity of the ensemble architecture.

Output:
    - analysis/model_selection_curves.png
    - analysis/model_selection_metrics.csv
    - Console summary table
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    brier_score_loss, f1_score, accuracy_score, classification_report
)
from tabulate import tabulate

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine
from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble


# ==============================================================================
# CONFIGURATION
# ==============================================================================
HORIZON = "14d"  # Forecast horizon to analyze
TEST_SPLIT_DATE = "2020-12-31"  # Data after this date is test set
FEATURE_MATRIX_PATH = PATHS["data"] / "processed" / "feature_matrix.parquet"
MODELS_DIR = PATHS["models"]
ANALYSIS_DIR = PATHS["root"] / "analysis"

# Target columns
TARGET_BINARY = "conflict_binary"
TARGET_FATALITIES = "fatalities_14d_sum"

# Ensure analysis directory exists
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_feature_matrix(path: Path) -> pd.DataFrame:
    """
    Load the feature matrix parquet file.
    
    Returns:
        DataFrame with features and targets
    """
    if not path.exists():
        # Try alternative paths
        alt_paths = [
            PATHS["data_proc"] / "feature_matrix.parquet",
            PATHS["data"] / "feature_matrix.parquet",
            ROOT_DIR / "feature_matrix.parquet"
        ]
        for alt in alt_paths:
            if alt.exists():
                path = alt
                break
        else:
            raise FileNotFoundError(
                f"Feature matrix not found at {path} or alternative locations.\n"
                f"Run build_feature_matrix.py first."
            )
    
    logger.info(f"Loading feature matrix from: {path}")
    df = pd.read_parquet(path)
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def split_train_test(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets based on date threshold.
    
    Args:
        df: Full feature matrix
        split_date: ISO date string (e.g., "2020-12-31")
        
    Returns:
        (train_df, test_df)
    """
    df['date'] = pd.to_datetime(df['date'])
    split_dt = pd.to_datetime(split_date)
    
    train = df[df['date'] <= split_dt].copy()
    test = df[df['date'] > split_dt].copy()
    
    logger.info(f"Train/Test Split (date > {split_date}):")
    logger.info(f"  Train: {len(train):,} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    logger.info(f"  Test:  {len(test):,} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    
    return train, test


def load_ensemble_model(horizon: str, learner_type: str) -> Dict[str, Any]:
    """
    Load a trained TwoStageEnsemble pickle.
    
    Args:
        horizon: "14d", "1m", or "3m"
        learner_type: "xgboost" or "lightgbm"
        
    Returns:
        Dictionary containing 'ensemble' key with the model instance
    """
    filename = f"two_stage_ensemble_{horizon}_{learner_type}.pkl"
    path = MODELS_DIR / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    logger.info(f"Loading model: {filename}")
    model_dict = joblib.load(path)
    
    return model_dict


# PCA reconstruction helper (mirrors generate_predictions.py)
def apply_pca_if_needed(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    pca = bundle.get("pca")
    input_features = bundle.get("pca_input_features")
    pca_cols = bundle.get("pca_component_names")
    scaler = bundle.get("pca_scaler")

    if not pca or not input_features or not pca_cols:
        return df

    df = df.copy()
    for col in input_features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[input_features].fillna(0).values
    if scaler is not None:
        X = scaler.transform(X)
    comps = pca.transform(X)
    pca_df = pd.DataFrame(comps, columns=pca_cols, index=df.index)
    return pd.concat([df, pca_df], axis=1)


# ==============================================================================
# PREDICTION HELPERS
# ==============================================================================
def get_level1_predictions(
    ensemble: TwoStageEnsemble,
    X: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Extract predictions from each Level 1 (theme) sub-model.
    
    Args:
        ensemble: Fitted TwoStageEnsemble instance
        X: Feature dataframe
        
    Returns:
        Dict mapping theme name to probability predictions
    """
    predictions = {}
    
    for i, theme in enumerate(ensemble.theme_models):
        # Get theme name from features or use index
        theme_name = theme.get('name', f'Theme_{i}')
        
        # Infer theme name from features if not explicitly set
        if theme_name.startswith('Theme_'):
            features = theme.get('features', [])
            if any('conflict' in f.lower() or 'fatalities' in f.lower() for f in features):
                theme_name = 'conflict_history'
            elif any('price' in f.lower() or 'gold' in f.lower() for f in features):
                theme_name = 'economics'
            elif any('elevation' in f.lower() or 'slope' in f.lower() or 'dist_to' in f.lower() for f in features):
                theme_name = 'terrain'
            elif any('precip' in f.lower() or 'temp' in f.lower() or 'ndvi' in f.lower() for f in features):
                theme_name = 'environmental'
            elif any('epr' in f.lower() or 'ethnic' in f.lower() for f in features):
                theme_name = 'epr'
            else:
                theme_name = f'submodel_{i}'
        
        # Get feature subset
        feature_cols = theme.get('features', [])
        missing_cols = [c for c in feature_cols if c not in X.columns]
        
        if missing_cols:
            logger.warning(f"  Theme '{theme_name}' missing {len(missing_cols)} features. Skipping.")
            continue
        
        X_theme = X[feature_cols]
        
        # Get binary classifier predictions
        clf = theme.get('binary_model')
        if clf is not None:
            try:
                probs = clf.predict_proba(X_theme)[:, 1]
            except AttributeError:
                probs = clf.predict(X_theme)
            
            predictions[theme_name] = probs
    
    return predictions


def get_ensemble_predictions(
    ensemble: TwoStageEnsemble,
    X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get full ensemble (Level 2) predictions.
    
    Args:
        ensemble: Fitted TwoStageEnsemble instance
        X: Feature dataframe
        
    Returns:
        (probabilities, expected_fatalities)
    """
    return ensemble.predict(X)


# ==============================================================================
# METRICS CALCULATION
# ==============================================================================
def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        name: Model name for reporting
        
    Returns:
        Dict of metric names to values
    """
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning(f"  {name}: Only one class in y_true. Metrics may be unreliable.")
        return {
            'model': name,
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'brier_score': np.nan,
            'f1_optimal': np.nan,
            'accuracy_optimal': np.nan,
            'threshold_optimal': np.nan
        }
    
    # Core metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    # Find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Metrics at optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    f1_optimal = f1_score(y_true, y_pred_optimal, zero_division=0)
    acc_optimal = accuracy_score(y_true, y_pred_optimal)
    
    return {
        'model': name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'f1_optimal': f1_optimal,
        'accuracy_optimal': acc_optimal,
        'threshold_optimal': optimal_threshold
    }


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compare all models and return metrics dataframe.
    
    Args:
        y_true: Ground truth labels
        predictions: Dict mapping model name to predictions
        
    Returns:
        DataFrame with one row per model and metrics as columns
    """
    results = []
    
    for name, y_prob in predictions.items():
        metrics = calculate_classification_metrics(y_true, y_prob, name)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.sort_values('roc_auc', ascending=False).reset_index(drop=True)
    
    return df


# ==============================================================================
# PLOTTING
# ==============================================================================
def plot_roc_pr_curves(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    title_suffix: str = ""
) -> None:
    """
    Generate combined ROC and PR curve plot.
    
    Args:
        y_true: Ground truth labels
        predictions: Dict mapping model name to predictions
        output_path: Path to save the figure
        title_suffix: Optional suffix for plot titles
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color palette - distinguish ensemble from submodels
    colors = {
        'XGBoost Ensemble': '#1f77b4',
        'LightGBM Ensemble': '#2ca02c',
        'conflict_history': '#ff7f0e',
        'economics': '#d62728',
        'terrain': '#9467bd',
        'environmental': '#8c564b',
        'epr': '#e377c2',
    }
    
    # Linestyles - solid for ensembles, dashed for submodels
    linestyles = {
        'XGBoost Ensemble': '-',
        'LightGBM Ensemble': '-',
    }
    
    # === ROC Curve ===
    ax1 = axes[0]
    
    for name, y_prob in predictions.items():
        if len(np.unique(y_true)) < 2:
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        color = colors.get(name, plt.cm.tab10(hash(name) % 10))
        ls = linestyles.get(name, '--')
        lw = 2.5 if 'Ensemble' in name else 1.5
        alpha = 1.0 if 'Ensemble' in name else 0.7
        
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', 
                 color=color, linestyle=ls, linewidth=lw, alpha=alpha)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'ROC Curve Comparison{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.grid(True, alpha=0.3)
    
    # === Precision-Recall Curve ===
    ax2 = axes[1]
    
    # Baseline: proportion of positive class
    baseline = y_true.mean()
    
    for name, y_prob in predictions.items():
        if len(np.unique(y_true)) < 2:
            continue
            
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        color = colors.get(name, plt.cm.tab10(hash(name) % 10))
        ls = linestyles.get(name, '--')
        lw = 2.5 if 'Ensemble' in name else 1.5
        alpha = 1.0 if 'Ensemble' in name else 0.7
        
        ax2.plot(recall, precision, label=f'{name} (AP={ap:.3f})',
                 color=color, linestyle=ls, linewidth=lw, alpha=alpha)
    
    ax2.axhline(y=baseline, color='gray', linestyle=':', linewidth=1, 
                alpha=0.7, label=f'Random (AP={baseline:.3f})')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'Precision-Recall Curve Comparison{title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved curves to: {output_path}")


def plot_metric_comparison_bar(
    metrics_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate bar chart comparing key metrics across models.
    
    Args:
        metrics_df: DataFrame with model metrics
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    models = metrics_df['model'].values
    
    # Colors
    colors = []
    for m in models:
        if 'Ensemble' in m:
            colors.append('#1f77b4' if 'XGBoost' in m else '#2ca02c')
        else:
            colors.append('#d3d3d3')
    
    # ROC-AUC
    ax1 = axes[0]
    bars = ax1.barh(models, metrics_df['roc_auc'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('ROC-AUC', fontsize=11)
    ax1.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlim([0.5, 1.0])
    for bar, val in zip(bars, metrics_df['roc_auc']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                 va='center', fontsize=9)
    
    # PR-AUC
    ax2 = axes[1]
    bars = ax2.barh(models, metrics_df['pr_auc'], color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('PR-AUC (Average Precision)', fontsize=11)
    ax2.set_title('PR-AUC Comparison', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, metrics_df['pr_auc']):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                 va='center', fontsize=9)
    
    # Brier Score (lower is better)
    ax3 = axes[2]
    bars = ax3.barh(models, metrics_df['brier_score'], color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Brier Score (lower is better)', fontsize=11)
    ax3.set_title('Brier Score Comparison', fontsize=12, fontweight='bold')
    
    # Use scientific notation for x-axis
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    for bar, val in zip(bars, metrics_df['brier_score']):
        # Smaller offset and scientific notation for labels
        # Reduced offset from 0.01 to 0.002 to bring numbers closer
        ax3.text(val + (metrics_df['brier_score'].max() * 0.002), 
                 bar.get_y() + bar.get_height()/2, 
                 f'{val:.2e}', 
                 va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved bar chart to: {output_path}")


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
def main(horizon: str = HORIZON) -> pd.DataFrame:
    """
    Run the full model selection analysis.
    
    Args:
        horizon: Forecast horizon to analyze ("14d", "1m", "3m")
        
    Returns:
        DataFrame with comparison metrics
    """
    logger.info("=" * 70)
    logger.info(f"MODEL SELECTION ANALYSIS - Horizon: {horizon}")
    logger.info("=" * 70)
    
    # 1. Load Data
    logger.info("\n[1/5] Loading feature matrix...")
    df = load_feature_matrix(FEATURE_MATRIX_PATH)
    
    # 2. Split Train/Test
    logger.info("\n[2/5] Splitting train/test data...")
    train_df, test_df = split_train_test(df, TEST_SPLIT_DATE)
    
    # Prepare test targets
    if TARGET_BINARY not in test_df.columns:
        # Derive binary target from fatalities
        if TARGET_FATALITIES in test_df.columns:
            test_df[TARGET_BINARY] = (test_df[TARGET_FATALITIES] > 0).astype(int)
        else:
            raise ValueError(f"Neither {TARGET_BINARY} nor {TARGET_FATALITIES} found in data.")
    
    y_test = test_df[TARGET_BINARY].values
    
    logger.info(f"  Test set class distribution: {y_test.mean()*100:.2f}% positive")
    
    # 3. Load Models
    logger.info("\n[3/5] Loading trained models...")
    all_predictions = {}
    
    for learner_type in ['xgboost', 'lightgbm']:
        try:
            model_dict = load_ensemble_model(horizon, learner_type)
            ensemble = model_dict.get('ensemble')
            # Reconstruct PCA features if present in the bundle
            test_df_with_pca = apply_pca_if_needed(test_df, model_dict)
            
            if ensemble is None:
                logger.warning(f"  No 'ensemble' key in {learner_type} pickle. Skipping.")
                continue
            
            # Get Level 2 (full ensemble) predictions
            logger.info(f"\n  Getting predictions for {learner_type.upper()}...")
            
            try:
                prob, _ = get_ensemble_predictions(ensemble, test_df_with_pca)
                ensemble_name = f"{learner_type.upper()} Ensemble"
                all_predictions[ensemble_name] = prob
                logger.info(f"    ✓ Ensemble predictions: {len(prob):,} samples")
            except Exception as e:
                logger.error(f"    ✗ Ensemble prediction failed: {e}")
            
            # Get Level 1 (sub-model) predictions - only for first learner to avoid clutter
            if learner_type == 'xgboost':
                try:
                    submodel_preds = get_level1_predictions(ensemble, test_df_with_pca)
                    for name, preds in submodel_preds.items():
                        all_predictions[name] = preds
                        logger.info(f"    ✓ Sub-model '{name}': {len(preds):,} samples")
                except Exception as e:
                    logger.warning(f"    ⚠ Sub-model extraction failed: {e}")
                    
        except FileNotFoundError as e:
            logger.warning(f"  Model not found: {e}")
        except Exception as e:
            logger.error(f"  Error loading {learner_type}: {e}")
    
    if not all_predictions:
        raise RuntimeError("No models could be loaded. Ensure models are trained first.")
    
    # 4. Calculate Metrics
    logger.info("\n[4/5] Calculating comparison metrics...")
    metrics_df = compare_models(y_test, all_predictions)
    
    # 5. Generate Outputs
    logger.info("\n[5/5] Generating outputs...")
    
    # Save metrics CSV
    metrics_path = ANALYSIS_DIR / "model_selection_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"  ✓ Saved metrics to: {metrics_path}")
    
    # Generate ROC/PR curves
    curves_path = ANALYSIS_DIR / "model_selection_curves.png"
    plot_roc_pr_curves(y_test, all_predictions, curves_path, f" ({horizon} Horizon)")
    
    # Generate bar chart
    bars_path = ANALYSIS_DIR / "model_selection_bars.png"
    plot_metric_comparison_bar(metrics_df, bars_path)
    
    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    
    # Format for display
    display_df = metrics_df.copy()
    display_df['roc_auc'] = display_df['roc_auc'].apply(lambda x: f"{x:.4f}")
    display_df['pr_auc'] = display_df['pr_auc'].apply(lambda x: f"{x:.4f}")
    display_df['brier_score'] = display_df['brier_score'].apply(lambda x: f"{x:.4f}")
    display_df['f1_optimal'] = display_df['f1_optimal'].apply(lambda x: f"{x:.4f}")
    display_df['threshold_optimal'] = display_df['threshold_optimal'].apply(lambda x: f"{x:.3f}")
    
    print("\n" + tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Key findings
    logger.info("\n" + "-" * 70)
    logger.info("KEY FINDINGS:")
    logger.info("-" * 70)
    
    best_model = metrics_df.iloc[0]['model']
    best_roc = metrics_df.iloc[0]['roc_auc']
    
    # Find best submodel
    submodel_df = metrics_df[~metrics_df['model'].str.contains('Ensemble')]
    if not submodel_df.empty:
        best_submodel = submodel_df.iloc[0]['model']
        best_submodel_roc = submodel_df.iloc[0]['roc_auc']
        
        # Calculate improvement
        ensemble_df = metrics_df[metrics_df['model'].str.contains('Ensemble')]
        if not ensemble_df.empty:
            ensemble_roc = ensemble_df.iloc[0]['roc_auc']
            improvement = (ensemble_roc - best_submodel_roc) / best_submodel_roc * 100
            
            logger.info(f"  • Best overall model: {best_model} (ROC-AUC: {best_roc:.4f})")
            logger.info(f"  • Best single sub-model: {best_submodel} (ROC-AUC: {best_submodel_roc:.4f})")
            logger.info(f"  • Ensemble improvement over best sub-model: {improvement:+.2f}%")
    
    # XGBoost vs LightGBM
    xgb_row = metrics_df[metrics_df['model'] == 'XGBoost Ensemble']
    lgb_row = metrics_df[metrics_df['model'] == 'LightGBM Ensemble']
    
    if not xgb_row.empty and not lgb_row.empty:
        xgb_roc = xgb_row.iloc[0]['roc_auc']
        lgb_roc = lgb_row.iloc[0]['roc_auc']
        diff = xgb_roc - lgb_roc
        
        winner = "XGBoost" if diff > 0 else "LightGBM"
        logger.info(f"  • XGBoost vs LightGBM: {winner} wins by {abs(diff):.4f} ROC-AUC")
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"  Outputs saved to: {ANALYSIS_DIR}")
    logger.info("=" * 70)
    
    return metrics_df


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Model Selection Analysis for Two-Stage Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_model_selection.py                    # Default: 14d horizon
    python analyze_model_selection.py --horizon 1m      # Analyze 1-month horizon
    python analyze_model_selection.py --horizon 3m      # Analyze 3-month horizon
        """
    )
    
    parser.add_argument(
        '--horizon', '-H',
        type=str,
        default=HORIZON,
        choices=['14d', '1m', '3m'],
        help='Forecast horizon to analyze (default: 14d)'
    )
    
    parser.add_argument(
        '--test-split',
        type=str,
        default=TEST_SPLIT_DATE,
        help='Test set split date in ISO format (default: 2020-12-31)'
    )
    
    args = parser.parse_args()
    
    # Update globals if CLI args provided
    if args.test_split != TEST_SPLIT_DATE:
        TEST_SPLIT_DATE = args.test_split
    
    try:
        results = main(horizon=args.horizon)
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)
