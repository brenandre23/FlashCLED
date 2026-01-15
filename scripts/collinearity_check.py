"""
pipeline/processing/collinearity_check.py
=========================================
Variance Inflation Factor (VIF) and correlation analysis for feature selection.

This module implements collinearity detection to identify redundant features
that may destabilize model coefficients and SHAP values.

Usage:
    python -m pipeline.processing.collinearity_check

Outputs:
    - data/processed/vif_analysis.csv
    - data/processed/correlation_matrix.parquet
    - data/processed/feature_clusters.json
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Suppress convergence warnings for VIF calculation
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, load_configs


def calculate_vif(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for all numeric features.
    
    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity with other predictors.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric features (will filter to numeric columns)
    threshold : float
        VIF threshold for flagging high collinearity (default: 5.0)
        - VIF = 1: No correlation
        - VIF 1-5: Moderate correlation
        - VIF > 5: High correlation (problematic)
        - VIF > 10: Severe correlation (definitely problematic)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, VIF, is_high_vif
        
    Notes
    -----
    Uses the formula: VIF_i = 1 / (1 - R²_i)
    where R²_i is from regressing feature i on all other features.
    
    For large feature sets, this can be slow. Consider sampling or
    using correlation-based pre-filtering.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Filter to numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target columns and identifiers
    exclude_patterns = ['target_', 'h3_index', 'date', 'year', 'epoch']
    feature_cols = [
        c for c in numeric_cols 
        if not any(pattern in c for pattern in exclude_patterns)
    ]
    
    if len(feature_cols) < 2:
        logger.warning("VIF requires at least 2 features")
        return pd.DataFrame(columns=['feature', 'VIF', 'is_high_vif'])
    
    logger.info(f"Calculating VIF for {len(feature_cols)} features...")
    
    # Prepare data (drop NaN rows)
    X = df[feature_cols].dropna()
    
    if len(X) < len(feature_cols):
        logger.warning(
            f"Insufficient rows ({len(X)}) for VIF calculation with {len(feature_cols)} features. "
            "Results may be unreliable."
        )
    
    # Calculate VIF for each feature
    vif_data = []
    for i, col in enumerate(feature_cols):
        try:
            vif_value = variance_inflation_factor(X.values, i)
            vif_data.append({
                'feature': col,
                'VIF': vif_value,
                'is_high_vif': vif_value > threshold
            })
        except Exception as e:
            logger.warning(f"VIF calculation failed for {col}: {e}")
            vif_data.append({
                'feature': col,
                'VIF': np.nan,
                'is_high_vif': False
            })
        
        if (i + 1) % 20 == 0:
            logger.debug(f"  Processed {i + 1}/{len(feature_cols)} features...")
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    # Log summary
    high_vif_count = vif_df['is_high_vif'].sum()
    if high_vif_count > 0:
        logger.warning(
            f"⚠️ {high_vif_count} features have VIF > {threshold} (high collinearity):\n"
            f"{vif_df[vif_df['is_high_vif']][['feature', 'VIF']].head(10).to_string()}"
        )
    else:
        logger.info(f"✓ No features exceed VIF threshold of {threshold}")
    
    return vif_df


def calculate_correlation_matrix(
    df: pd.DataFrame, 
    method: str = 'spearman',
    threshold: float = 0.8
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Calculate correlation matrix and identify highly correlated feature pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric features
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'
    threshold : float
        Correlation threshold for flagging pairs (default: 0.8)
        
    Returns
    -------
    Tuple[pd.DataFrame, List[Tuple[str, str, float]]]
        - Full correlation matrix
        - List of (feature1, feature2, correlation) for pairs above threshold
    """
    # Filter to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    exclude_patterns = ['target_', 'h3_index', 'date', 'year', 'epoch']
    feature_cols = [
        c for c in numeric_cols 
        if not any(pattern in c for pattern in exclude_patterns)
    ]
    
    logger.info(f"Computing {method} correlation matrix for {len(feature_cols)} features...")
    
    corr_matrix = df[feature_cols].corr(method=method)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i+1:]:
            corr_value = corr_matrix.loc[col1, col2]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append((col1, col2, corr_value))
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if high_corr_pairs:
        logger.warning(
            f"⚠️ {len(high_corr_pairs)} feature pairs have |correlation| >= {threshold}:\n"
            f"Top 5:\n" + 
            "\n".join([f"  {p[0]} <-> {p[1]}: {p[2]:.3f}" for p in high_corr_pairs[:5]])
        )
    else:
        logger.info(f"✓ No feature pairs exceed correlation threshold of {threshold}")
    
    return corr_matrix, high_corr_pairs


def cluster_correlated_features(
    corr_matrix: pd.DataFrame, 
    threshold: float = 0.7,
    method: str = 'complete'
) -> Dict[int, List[str]]:
    """
    Cluster features based on correlation using hierarchical clustering.
    
    Useful for identifying groups of redundant features where one representative
    could be selected from each cluster.
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix from calculate_correlation_matrix()
    threshold : float
        Distance threshold for clustering (1 - correlation)
    method : str
        Linkage method: 'complete', 'average', 'single', 'ward'
        
    Returns
    -------
    Dict[int, List[str]]
        Dictionary mapping cluster ID to list of feature names in that cluster
    """
    # Clean correlation matrix to ensure finiteness
    corr_clean = corr_matrix.replace([np.inf, -np.inf], np.nan)
    np.fill_diagonal(corr_clean.values, 1.0)
    corr_clean = corr_clean.fillna(0.0)
    
    # Convert correlation to distance (1 - |correlation|)
    distance_matrix = 1 - corr_clean.abs()
    
    # Ensure symmetry and fill diagonal
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix.values, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(condensed, method=method)
    
    # Cut tree at threshold
    cluster_labels = hierarchy.fcluster(linkage_matrix, threshold, criterion='distance')
    
    # Group features by cluster
    clusters = {}
    for feature, cluster_id in zip(corr_matrix.columns, cluster_labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(feature)
    
    # Filter to clusters with multiple features (redundancy)
    redundant_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    
    if redundant_clusters:
        logger.info(f"Found {len(redundant_clusters)} feature clusters with potential redundancy:")
        for cluster_id, features in redundant_clusters.items():
            logger.info(f"  Cluster {cluster_id}: {features[:5]}{'...' if len(features) > 5 else ''}")
    
    return clusters


def suggest_features_to_remove(
    vif_df: pd.DataFrame,
    high_corr_pairs: List[Tuple[str, str, float]],
    feature_importance: Optional[Dict[str, float]] = None,
    vif_threshold: float = 10.0
) -> List[str]:
    """
    Suggest features to remove based on VIF and correlation analysis.
    
    Strategy:
    1. For features with VIF > threshold, mark for removal
    2. For highly correlated pairs, keep the one with higher importance (if available)
       or the first alphabetically
    3. Return deduplicated list of features to remove
    
    Parameters
    ----------
    vif_df : pd.DataFrame
        VIF analysis results from calculate_vif()
    high_corr_pairs : List[Tuple[str, str, float]]
        Highly correlated feature pairs from calculate_correlation_matrix()
    feature_importance : Optional[Dict[str, float]]
        Feature importance scores (higher = more important, keep these)
    vif_threshold : float
        VIF threshold for removal (default: 10.0, more conservative than detection)
        
    Returns
    -------
    List[str]
        List of feature names suggested for removal
    """
    to_remove = set()
    
    # 1. High VIF features
    high_vif_features = vif_df[vif_df['VIF'] > vif_threshold]['feature'].tolist()
    to_remove.update(high_vif_features)
    
    # 2. From correlated pairs, remove the less important one
    if feature_importance is None:
        feature_importance = {}
    
    for feat1, feat2, corr in high_corr_pairs:
        # Skip if one is already marked for removal
        if feat1 in to_remove or feat2 in to_remove:
            continue
            
        imp1 = feature_importance.get(feat1, 0)
        imp2 = feature_importance.get(feat2, 0)
        
        if imp1 >= imp2:
            to_remove.add(feat2)
        else:
            to_remove.add(feat1)
    
    removal_list = sorted(to_remove)
    
    logger.info(
        f"Suggested {len(removal_list)} features for removal:\n"
        f"  - {len(high_vif_features)} due to high VIF\n"
        f"  - {len(removal_list) - len(high_vif_features)} due to high correlation"
    )
    
    return removal_list


def run_full_analysis(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    vif_threshold: float = 5.0,
    corr_threshold: float = 0.8,
    removal_vif_threshold: float = 10.0
) -> Dict:
    """
    Run complete collinearity analysis pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    output_dir : Path, optional
        Directory for output files (default: data/processed/)
    vif_threshold : float
        VIF threshold for flagging (default: 5.0)
    corr_threshold : float
        Correlation threshold for flagging pairs (default: 0.8)
    removal_vif_threshold : float
        VIF threshold for removal suggestions (default: 10.0)
        
    Returns
    -------
    Dict
        Analysis results including VIF, correlations, clusters, and removal suggestions
    """
    if output_dir is None:
        output_dir = PATHS.get("data_proc", Path("data/processed"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("COLLINEARITY ANALYSIS")
    logger.info("=" * 60)
    
    # 1. VIF Analysis
    logger.info("\n--- VIF Analysis ---")
    vif_df = calculate_vif(df, threshold=vif_threshold)
    vif_df.to_csv(output_dir / "vif_analysis.csv", index=False)
    
    # 2. Correlation Analysis
    logger.info("\n--- Correlation Analysis ---")
    corr_matrix, high_corr_pairs = calculate_correlation_matrix(
        df, method='spearman', threshold=corr_threshold
    )
    corr_matrix.to_parquet(output_dir / "correlation_matrix.parquet")
    
    # 3. Feature Clustering
    logger.info("\n--- Feature Clustering ---")
    clusters = cluster_correlated_features(corr_matrix, threshold=0.3)
    with open(output_dir / "feature_clusters.json", 'w') as f:
        # Convert to JSON-serializable format
        clusters_json = {str(k): v for k, v in clusters.items()}
        json.dump(clusters_json, f, indent=2)
    
    # 4. Removal Suggestions
    logger.info("\n--- Removal Suggestions ---")
    removal_list = suggest_features_to_remove(
        vif_df, high_corr_pairs, 
        vif_threshold=removal_vif_threshold
    )
    
    # Save removal suggestions
    with open(output_dir / "suggested_removals.json", 'w') as f:
        json.dump({
            'features_to_remove': removal_list,
            'vif_threshold': removal_vif_threshold,
            'corr_threshold': corr_threshold
        }, f, indent=2)
    
    # Summary
    results = {
        'vif_analysis': vif_df,
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'feature_clusters': clusters,
        'suggested_removals': removal_list,
        'summary': {
            'total_features': len(vif_df),
            'high_vif_count': int(vif_df['is_high_vif'].sum()),
            'high_corr_pairs_count': len(high_corr_pairs),
            'redundant_clusters': len([c for c in clusters.values() if len(c) > 1]),
            'suggested_removal_count': len(removal_list)
        }
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"  Total features analyzed: {results['summary']['total_features']}")
    logger.info(f"  High VIF features: {results['summary']['high_vif_count']}")
    logger.info(f"  High correlation pairs: {results['summary']['high_corr_pairs_count']}")
    logger.info(f"  Redundant clusters: {results['summary']['redundant_clusters']}")
    logger.info(f"  Suggested removals: {results['summary']['suggested_removal_count']}")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 60)
    
    return results


def run():
    """Main entry point for standalone execution."""
    # Load feature matrix
    matrix_path = PATHS.get("data_proc", Path("data/processed")) / "feature_matrix.parquet"
    
    if not matrix_path.exists():
        logger.error(f"Feature matrix not found: {matrix_path}")
        logger.info("Run build_feature_matrix.py first.")
        sys.exit(1)
    
    logger.info(f"Loading feature matrix from {matrix_path}...")
    df = pd.read_parquet(matrix_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Run analysis
    results = run_full_analysis(df)
    
    return results


if __name__ == "__main__":
    run()
