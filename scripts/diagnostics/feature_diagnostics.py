"""
feature_diagnostics.py

RAM-aware diagnostics for collinearity, redundancy, and stability.
Does NOT drop features or train models; produces evidence for human judgment.

UPDATED: Now supports automatic exclusion of structural break flags via
environment variables or CLI args. Structural break flags are binary
data availability indicators that add noise to VIF/correlation analysis.

Usage:
    # Via main.py (recommended):
    python main.py --run-diagnostics-only
    python main.py --run-diagnostics-only --include-structural-breaks
    python main.py --run-diagnostics-only --diagnostic-exclude-cols "col1,col2"
    
    # Standalone:
    python -m scripts.diagnostics.feature_diagnostics --exclude-structural-breaks
"""

import argparse
import os
from pathlib import Path
import sys
import random
from typing import List, Tuple, Optional, Set, Dict, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import kendalltau, entropy
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

file_path = Path(__file__).resolve()
ROOT = file_path.parents[2]
sys.path.insert(0, str(ROOT))

from utils import logger, PATHS, load_configs


def get_diagnostic_options_from_env() -> dict:
    """
    Read diagnostic options from environment variables.
    These are set by main.py when running in diagnostic modes.
    """
    exclude_structural = os.environ.get("CEWP_DIAG_EXCLUDE_STRUCTURAL", "True").lower() == "true"
    extra_exclusions_str = os.environ.get("CEWP_DIAG_EXTRA_EXCLUSIONS", "")
    extra_exclusions = [c.strip() for c in extra_exclusions_str.split(",") if c.strip()]
    sample_frac_str = os.environ.get("CEWP_DIAG_SAMPLE_FRAC", "")
    sample_frac = float(sample_frac_str) if sample_frac_str else None
    
    return {
        "exclude_structural_breaks": exclude_structural,
        "extra_exclusions": extra_exclusions,
        "sample_frac": sample_frac,
    }


def filter_diagnostic_columns(
    df: pd.DataFrame,
    exclude_structural_breaks: bool = True,
    extra_exclusions: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Filter DataFrame columns for diagnostic analysis.
    
    Removes:
    - Non-numeric columns
    - Target columns
    - Identifier columns (h3_index, date, etc.)
    - Structural break flags (optional)
    - Extra exclusions
    
    Returns:
        Tuple of (filtered_df, excluded_columns)
    """
    try:
        from pipeline.common.diagnostic_utils import (
            get_structural_break_flags,
            get_standard_exclusion_patterns
        )
        use_diagnostic_utils = True
    except ImportError:
        logger.warning("diagnostic_utils not found; using fallback exclusion logic")
        use_diagnostic_utils = False
    
    original_cols = set(df.columns)
    excluded = set()
    
    # 1. Keep only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    numeric_cols = set(numeric_df.columns)
    excluded.update(original_cols - numeric_cols)
    
    # 2. Exclude target columns
    target_cols = {c for c in numeric_cols if c.startswith("target_")}
    excluded.update(target_cols)
    
    # 3. Exclude identifiers
    id_cols = {"h3_index", "year", "epoch"}
    excluded.update(id_cols & numeric_cols)
    
    # 4. Exclude structural break flags
    if exclude_structural_breaks:
        if use_diagnostic_utils:
            configs = load_configs()
            features_cfg = configs["features"]
            structural_flags = get_structural_break_flags(features_cfg)
        else:
            # Fallback hardcoded list
            structural_flags = {
                "is_worldpop_v1",
                "iom_data_available",
                "econ_data_available",
                "ioda_data_available",
                "landcover_data_available",
                "viirs_data_available",
                "gdelt_data_available",
                "food_data_available",
            }
        excluded.update(structural_flags & numeric_cols)
    
    # 5. Extra exclusions
    if extra_exclusions:
        excluded.update(set(extra_exclusions) & numeric_cols)
    
    # Final column list
    final_cols = [c for c in numeric_df.columns if c not in excluded]

    # Ensure 'date' is preserved if it exists in the original dataframe
    if 'date' in df.columns and 'date' not in final_cols:
        final_cols.append('date')
    
    if verbose:
        logger.info(f"Diagnostic column filtering:")
        logger.info(f"  Original columns: {len(original_cols)}")
        logger.info(f"  Numeric columns: {len(numeric_cols)}")
        logger.info(f"  Excluded total: {len(excluded)}")
        logger.info(f"  Final diagnostic columns: {len(final_cols)}")
        
        if exclude_structural_breaks:
            excluded_flags = sorted(excluded & (structural_flags if exclude_structural_breaks else set()))
            if excluded_flags:
                logger.info(f"  Structural break flags excluded: {excluded_flags}")
    
    return df[final_cols].copy(), excluded


def stratified_sample(df: pd.DataFrame, horizon_steps: List[int], max_pos: int = 5000, neg_ratio: float = 3.0,
                      spatial_frac: float = 0.2) -> pd.DataFrame:
    """
    Temporal (early/mid/late) + outcome stratified + spatial subsample to preserve structure.
    """
    df = df.copy()
    
    # Check if date column exists
    if "date" not in df.columns:
        logger.warning("'date' column not found; skipping temporal stratification")
        return df.sample(n=min(len(df), max_pos * 10), random_state=42)
    
    df["period_bin"] = pd.qcut(df["date"].rank(method="first"), [0, 0.25, 0.75, 1], labels=["early", "mid", "late"])

    samples = []
    for steps in horizon_steps:
        tgt = f"target_binary_{steps}_step"
        if tgt not in df.columns:
            continue
        for period in ["early", "mid", "late"]:
            block = df[df["period_bin"] == period]
            pos = block[block[tgt] == 1]
            neg = block[block[tgt] == 0]
            n_pos = min(max_pos, len(pos))
            n_neg = min(int(neg_ratio * n_pos), len(neg))
            pos_s = pos.sample(n=n_pos, random_state=42) if n_pos > 0 else pos
            neg_s = neg.sample(n=n_neg, random_state=42) if n_neg > 0 else neg
            subset = pd.concat([pos_s, neg_s])
            # Spatial thinning via H3 hash
            if "h3_index" in subset.columns and spatial_frac < 1.0:
                subset = subset[subset["h3_index"].astype(int) % int(1/spatial_frac) == 0]
            samples.append(subset)

    if not samples:
        return df

    out = pd.concat(samples).drop_duplicates().sort_values(["date", "h3_index"] if "date" in df.columns and "h3_index" in df.columns else df.columns[:2])
    return out


def compute_pairwise(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute pairwise Spearman correlations and flag high correlations."""
    corr = df[numeric_cols].corr(method="spearman")
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            if abs(rho) >= 0.6:
                kt, _ = kendalltau(df[cols[i]], df[cols[j]])
                pairs.append({"feat_a": cols[i], "feat_b": cols[j], "spearman": float(rho), "kendall": float(kt)})
    # Mutual information on flagged pairs
    if pairs:
        for p in pairs:
            mi = mutual_info_regression(
                df[[p["feat_a"]]].fillna(0).astype(np.float32),
                df[p["feat_b"]].fillna(0).astype(np.float32),
                discrete_features=False
            )[0]
            p["mi"] = float(mi)
    return pd.DataFrame(pairs).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def compute_vif_cond_pca(df: pd.DataFrame, cols: List[str]) -> Tuple[float, float, float, float, float]:
    """Compute VIF, condition number, and PCA variance for a set of columns."""
    if len(cols) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    X = df[cols].fillna(0).astype(np.float32)
    vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    cond_num = float(np.linalg.cond(X.values))
    pca = PCA()
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    pc1 = float(var_ratio[0]) if len(var_ratio) > 0 else 0.0
    pc2 = float(var_ratio[1]) if len(var_ratio) > 1 else 0.0
    return float(np.nanmax(vif_vals)), float(np.nanmedian(vif_vals)), cond_num, pc1, pc2


def temporal_stability(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Analyze temporal stability of pairwise correlations across time periods."""
    if "date" not in df.columns:
        logger.warning("'date' column not found; skipping temporal stability analysis")
        return pd.DataFrame()
    
    df = df.copy()
    df["period_bin"] = pd.qcut(df["date"].rank(method="first"), [0, 0.25, 0.75, 1], labels=["early", "mid", "late"])
    rows = []
    for _, row in pairs.iterrows():
        feats = (row["feat_a"], row["feat_b"])
        vals = {}
        for period in ["early", "mid", "late"]:
            sub = df[df["period_bin"] == period]
            rho = sub[list(feats)].corr(method="spearman").iloc[0, 1]
            vals[period] = float(rho)
        label = "stable"
        spread = max(vals.values()) - min(vals.values())
        if any(np.sign(vals[p]) != np.sign(list(vals.values())[0]) for p in vals):
            label = "unstable"
        elif spread > 0.4:
            label = "regime-dependent"
        rows.append({"feat_a": feats[0], "feat_b": feats[1], **{f"rho_{k}": v for k, v in vals.items()}, "stability": label})
    return pd.DataFrame(rows)


def interpretability_checks(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Compute interpretability metrics for each feature."""
    records = []
    corr_matrix = df[numeric_cols].corr(method="spearman")
    
    for col in numeric_cols:
        series = df[col].dropna()
        std = float(series.std())
        variance = float(series.var())
        ent = float(entropy(series.value_counts(normalize=True), base=2)) if len(series) > 0 else 0.0
        max_rho = float(corr_matrix[col].drop(col).abs().max())
        records.append({
            "feature": col,
            "std": std,
            "variance": variance,
            "entropy": ent,
            "max_abs_rho": max_rho,
            "max_abs_spearman": max_rho  # Alias for compatibility
        })
    return pd.DataFrame(records)


def load_optimized_matrix(
    parquet_path: Path,
    exclude_structural_breaks: bool = True,
    extra_exclusions: Optional[List[str]] = None,
    sample_frac: Optional[float] = None
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Memory-efficient loading of feature matrix.
    1. Inspects schema to identify numeric columns and columns to exclude.
    2. Loads only necessary columns.
    3. If sample_frac is set, reads a random subset of row groups.
    """
    if not parquet_path.exists():
        logger.error(f"Feature matrix not found: {parquet_path}")
        sys.exit(1)

    logger.info(f"Inspecting Parquet schema: {parquet_path}")
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema
    
    # 1. Identify columns to load
    # We want numeric columns, plus 'date' and 'h3_index' for stratification/metadata.
    # We exclude targets from diagnostics, but we might need them for stratified sampling?
    # stratified_sample uses 'target_binary_*'. Let's keep them for now and filter later if needed,
    # OR we can implement stratification here.
    # To be safe and save memory, let's keep potential stratification columns.
    
    all_cols = schema.names
    
    # Determine exclusions based on name patterns (replicating filter_diagnostic_columns logic mostly)
    # But we need to check types too.
    
    # Helper to check if type is numeric
    def is_numeric_type(col_name):
        # PyArrow schema type check
        field = schema.column(all_cols.index(col_name))
        # logical_type can be None, check physical
        pt = field.physical_type
        return pt in ['INT32', 'INT64', 'FLOAT', 'DOUBLE']

    # Get structural flags if needed
    structural_flags = set()
    if exclude_structural_breaks:
        try:
            from pipeline.common.diagnostic_utils import get_structural_break_flags
            configs = load_configs()
            structural_flags = get_structural_break_flags(configs["features"])
        except ImportError:
            # Fallback
            structural_flags = {
                "is_worldpop_v1", "iom_data_available", "econ_data_available",
                "ioda_data_available", "landcover_data_available", "viirs_data_available",
                "gdelt_data_available", "food_data_available"
            }

    cols_to_load = []
    excluded_log = set()

    essential_cols = {'date', 'h3_index', 'year', 'epoch'}
    
    for col in all_cols:
        # Always keep essential cols if they exist
        if col in essential_cols:
            cols_to_load.append(col)
            continue
            
        # Exclude structural flags
        if exclude_structural_breaks and col in structural_flags:
            excluded_log.add(col)
            continue
            
        # Exclude extra exclusions
        if extra_exclusions and col in extra_exclusions:
            excluded_log.add(col)
            continue
            
        # Check if numeric
        if is_numeric_type(col):
            cols_to_load.append(col)
        else:
            # Non-numeric (and not essential) -> Exclude
            excluded_log.add(col)

    logger.info(f"  Schema columns: {len(all_cols)}")
    logger.info(f"  Columns to load: {len(cols_to_load)}")
    logger.info(f"  Excluded (pre-load): {len(excluded_log)}")

    # 2. Row Group Sampling
    if sample_frac and 0 < sample_frac < 1.0:
        num_groups = pf.num_row_groups
        # We want approx sample_frac of rows.
        # Assuming row groups are roughly equal size.
        n_groups_to_read = max(1, int(num_groups * sample_frac))
        
        # Randomly select row group indices
        # Use sorted indices for potentially better sequential read perf
        group_indices = sorted(random.sample(range(num_groups), n_groups_to_read))
        
        logger.info(f"  Sampling: Reading {n_groups_to_read}/{num_groups} row groups ({sample_frac:.1%})")
        
        # Read specific row groups and columns
        df = pf.read_row_groups(group_indices, columns=cols_to_load).to_pandas()
        
    else:
        # Load full file (but only selected columns)
        logger.info("  Loading full dataset (selected columns)...")
        df = pf.read(columns=cols_to_load).to_pandas()

    logger.info(f"  Loaded DataFrame shape: {df.shape}")
    
    return df, excluded_log


def main(diag_opts: Optional[Dict[str, Any]] = None):
    # If called directly, provide default args for standalone execution
    if diag_opts is None:
        parser = argparse.ArgumentParser(description="RAM-aware feature diagnostics (no dropping).")
        parser.add_argument("--parquet", type=Path, default=None, help="Optional feature matrix path.")
        parser.add_argument("--max-pos", type=int, default=5000, help="Max positives per period per horizon.")
        parser.add_argument("--neg-ratio", type=float, default=3.0, help="Negatives per positive cap.")
        parser.add_argument("--spatial-frac", type=float, default=0.2, help="Fraction of H3s to keep in spatial thinning.")
        parser.add_argument(
            "--exclude-structural-breaks",
            action="store_true",
            default=None, # Set to None so it can be overwritten by diag_opts
            help="Exclude structural break flags from diagnostics."
        )
        parser.add_argument(
            "--include-structural-breaks",
            action="store_true",
            help="Include structural break flags in diagnostics (overrides default exclusion)."
        )
        parser.add_argument(
            "--exclude-cols",
            type=str,
            default=None,
            help="Comma-separated list of additional columns to exclude."
        )
        cli_args = parser.parse_args()

        exclude_structural = False
        if cli_args.include_structural_breaks:
            exclude_structural = False
        elif cli_args.exclude_structural_breaks is not None:
            exclude_structural = cli_args.exclude_structural_breaks
        else:
            exclude_structural = True # Default for standalone run if no flags set

        diag_opts = {
            "exclude_structural_breaks": exclude_structural,
            "extra_exclusions": [c.strip() for c in (cli_args.exclude_cols or "").split(",") if c.strip()],
            "sample_frac": cli_args.diagnostic_sample_frac if hasattr(cli_args, 'diagnostic_sample_frac') else None, # Assuming this is not present in standalone
            "parquet_path": cli_args.parquet,
            "max_pos": cli_args.max_pos,
            "neg_ratio": cli_args.neg_ratio,
            "spatial_frac": cli_args.spatial_frac,
        }
    
    # Load configs
    configs = load_configs()
    models_cfg = configs["models"] if isinstance(configs, dict) else configs[2]
    features_cfg = configs["features"] if isinstance(configs, dict) else configs[1]
    horizons = models_cfg.get("horizons", [])
    horizon_steps = [h["steps"] for h in horizons if "steps" in h]

    # Use diag_opts passed from orchestrator or derived from CLI
    exclude_structural = diag_opts["exclude_structural_breaks"]
    extra_exclusions = diag_opts["extra_exclusions"]
    sample_frac = diag_opts["sample_frac"]
    parquet_path = diag_opts.get("parquet_path") or PATHS["data_proc"] / "feature_matrix.parquet"
    max_pos = diag_opts.get("max_pos", 5000)
    neg_ratio = diag_opts.get("neg_ratio", 3.0)
    spatial_frac = diag_opts.get("spatial_frac", 0.2)
    
    # Load feature matrix (Optimized)
    df, excluded_cols_preload = load_optimized_matrix(
        parquet_path,
        exclude_structural_breaks=exclude_structural,
        extra_exclusions=extra_exclusions,
        sample_frac=sample_frac
    )

    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # 1. Stratified sample (Needs targets to work!)
    # We pass the FULL df (with targets) so stratification can find positive/negative examples
    sample_full = stratified_sample(
        df, 
        horizon_steps, 
        max_pos=max_pos, 
        neg_ratio=neg_ratio, 
        spatial_frac=spatial_frac
    )
    logger.info(f"\nDiagnostic sample size: {len(sample_full):,} rows (downsampled from {len(df):,})")

    # 2. Filter columns for diagnostics (Remove targets/metadata from the SAMPLE)
    logger.info(f"\nFinalizing column selection...")
    sample, excluded_cols_post = filter_diagnostic_columns(
        sample_full,
        exclude_structural_breaks=exclude_structural,
        extra_exclusions=extra_exclusions,
        verbose=False
    )
    
    # Merge exclusion logs
    excluded_cols = excluded_cols_preload | excluded_cols_post
    logger.info(f"  Total excluded columns: {len(excluded_cols)}")

    # Preserve date for temporal stratification/stability while keeping it out of correlation/VIF
    if "date" in df.columns and "date" not in sample.columns:
        sample = sample.join(sample_full["date"])

    # Get numeric columns for analysis
    numeric_cols = sample.select_dtypes(include=[int, float]).columns.tolist()
    if 'date' in numeric_cols:
        numeric_cols.remove('date') # Exclude date from numeric analysis
    sample[numeric_cols] = sample[numeric_cols].astype(np.float32)

    # Pairwise dependence
    logger.info("\nComputing pairwise correlations...")
    pair_df = compute_pairwise(sample, numeric_cols)
    logger.info(f"  Found {len(pair_df)} pairs with |rho| >= 0.6")

    # Multivariate redundancy per theme
    logger.info("\nComputing theme-level redundancy metrics...")
    theme_rows = []
    submodels = models_cfg.get("submodels", {})
    for name, cfg in submodels.items():
        if not cfg.get("enabled"):
            continue
        feats = [f for f in cfg.get("features", []) if f in numeric_cols]
        if len(feats) < 2:
            logger.debug(f"  Skipping theme '{name}': only {len(feats)} features in diagnostic set")
            continue
        max_vif, med_vif, cond_num, pc1, pc2 = compute_vif_cond_pca(sample, feats)
        theme_rows.append({
            "theme": name, 
            "n_features": len(feats),
            "max_vif": max_vif, 
            "median_vif": med_vif, 
            "cond_num": cond_num, 
            "pc1_var": pc1, 
            "pc2_var": pc2
        })
    theme_df = pd.DataFrame(theme_rows)
    logger.info(f"  Analyzed {len(theme_df)} themes")

    # Temporal stability on flagged pairs
    logger.info("\nAnalyzing temporal stability...")
    stability_df = temporal_stability(sample, pair_df.head(50)) if not pair_df.empty else pd.DataFrame()
    logger.info(f"  Analyzed {len(stability_df)} pair stability profiles")

    # Interpretability checks
    logger.info("\nComputing interpretability metrics...")
    interp_df = interpretability_checks(sample, numeric_cols)
    logger.info(f"  Analyzed {len(interp_df)} features")

    # Save outputs
    out_dir = Path(__file__).resolve().parents[2] / "analysis" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pair_df.to_csv(out_dir / "pairwise_dependence.csv", index=False)
    theme_df.to_csv(out_dir / "theme_redundancy.csv", index=False)
    stability_df.to_csv(out_dir / "temporal_stability.csv", index=False)
    interp_df.to_csv(out_dir / "interpretability_checks.csv", index=False)
    
    # Save metadata about exclusions
    meta_df = pd.DataFrame({
        "excluded_column": sorted(excluded_cols),
        "reason": ["structural_break_flag" if "avail" in c.lower() or "data_available" in c.lower() or c.startswith("is_") 
                   else "identifier" if c in {"h3_index", "date", "year", "epoch"}
                   else "target" if c.startswith("target_")
                   else "extra_exclusion" if c in (extra_exclusions or [])
                   else "non_numeric" for c in sorted(excluded_cols)]
    })
    meta_df.to_csv(out_dir / "excluded_columns.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Output directory: {out_dir}")
    logger.info(f"  Files generated:")
    logger.info(f"    - pairwise_dependence.csv ({len(pair_df)} pairs)")
    logger.info(f"    - theme_redundancy.csv ({len(theme_df)} themes)")
    logger.info(f"    - temporal_stability.csv ({len(stability_df)} pairs)")
    logger.info(f"    - interpretability_checks.csv ({len(interp_df)} features)")
    logger.info(f"    - excluded_columns.csv ({len(excluded_cols)} excluded)")


def analyze_sub_ensemble_correlations(df: pd.DataFrame, models_cfg: dict, output_dir: Path):
    """
    Computes a correlation matrix for each enabled sub-model's feature set.
    Saves CSV and heatmap per submodel.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    logger.info("PHASE 6: Analyzing Sub-Ensemble Correlations...")

    # Identify target columns
    target_cols = [c for c in df.columns if any(p in c for p in ['target_fatalities', 'target_binary'])]

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, config in models_cfg.get("submodels", {}).items():
        if not config.get("enabled", True):
            continue

        sub_features = [f for f in config.get("features", []) if f in df.columns]
        if not sub_features:
            logger.warning(f"  No valid features for sub-model: {model_name}")
            continue

        analysis_cols = sub_features + target_cols
        corr_matrix = df[analysis_cols].corr()

        file_prefix = f"corr_{model_name.replace(' ', '_').lower()}"
        corr_matrix.to_csv(output_dir / f"{file_prefix}.csv")

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='RdYlGn', center=0)
        plt.title(f"Correlation Matrix: {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{file_prefix}_heatmap.png")
        plt.close()

        logger.info(f"  ✓ {model_name}: {len(sub_features)} features analyzed.")
