"""
feature_diagnostics.py

RAM-aware diagnostics for collinearity, redundancy, and stability.
Does NOT drop features or train models; produces evidence for human judgment.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, entropy
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

file_path = Path(__file__).resolve()
ROOT = file_path.parents[2]
sys.path.insert(0, str(ROOT))

from utils import logger, PATHS, load_configs
from pipeline.common.feature_loader import build_required_columns, load_feature_matrix


def stratified_sample(df: pd.DataFrame, horizon_steps: List[int], max_pos: int = 5000, neg_ratio: float = 3.0,
                      spatial_frac: float = 0.2) -> pd.DataFrame:
    """
    Temporal (early/mid/late) + outcome stratified + spatial subsample to preserve structure.
    """
    df = df.copy()
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
            subset = subset[subset["h3_index"].astype(int) % int(1/spatial_frac) == 0] if spatial_frac < 1.0 else subset
            samples.append(subset)

    if not samples:
        return df

    out = pd.concat(samples).drop_duplicates().sort_values(["date", "h3_index"])
    return out


def compute_pairwise(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
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
    records = []
    for col in numeric_cols:
        series = df[col].dropna()
        std = float(series.std())
        ent = float(entropy(series.value_counts(normalize=True), base=2)) if len(series) > 0 else 0.0
        max_rho = float(df[numeric_cols].corr(method="spearman")[col].drop(col).abs().max())
        records.append({"feature": col, "std": std, "entropy": ent, "max_abs_spearman": max_rho})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="RAM-aware feature diagnostics (no dropping).")
    parser.add_argument("--parquet", type=Path, default=None, help="Optional feature matrix path.")
    parser.add_argument("--max-pos", type=int, default=5000, help="Max positives per period per horizon.")
    parser.add_argument("--neg-ratio", type=float, default=3.0, help="Negatives per positive cap.")
    parser.add_argument("--spatial-frac", type=float, default=0.2, help="Fraction of H3s to keep in spatial thinning.")
    args = parser.parse_args()

    configs = load_configs()
    models_cfg = configs["models"] if isinstance(configs, dict) else configs[2]
    horizons = models_cfg.get("horizons", [])
    horizon_steps = [h["steps"] for h in horizons if "steps" in h]

    required_cols = build_required_columns(models_cfg, bundle=None, extra=[f"target_binary_{h['steps']}_step" for h in horizons if "steps" in h])
    df = load_feature_matrix(required_cols, parquet_path=args.parquet)
    df["date"] = pd.to_datetime(df["date"])

    sample = stratified_sample(df, horizon_steps, max_pos=args.max_pos, neg_ratio=args.neg_ratio, spatial_frac=args.spatial_frac)
    logger.info(f"Diagnostic sample size: {len(sample):,} rows")

    # Prepare numeric cols (exclude identifiers/targets)
    numeric_cols = sample.select_dtypes(include=[int, float]).columns.tolist()
    target_cols = [c for c in numeric_cols if c.startswith("target_")]
    numeric_cols = [c for c in numeric_cols if c not in target_cols and c != "h3_index"]
    sample[numeric_cols] = sample[numeric_cols].astype(np.float32)

    # Pairwise dependence
    pair_df = compute_pairwise(sample, numeric_cols)

    # Multivariate redundancy per theme
    theme_rows = []
    submodels = models_cfg.get("submodels", {})
    for name, cfg in submodels.items():
        if not cfg.get("enabled"):
            continue
        feats = [f for f in cfg.get("features", []) if f in numeric_cols]
        max_vif, med_vif, cond_num, pc1, pc2 = compute_vif_cond_pca(sample, feats)
        theme_rows.append({"theme": name, "max_vif": max_vif, "median_vif": med_vif, "cond_num": cond_num, "pc1_var": pc1, "pc2_var": pc2})
    theme_df = pd.DataFrame(theme_rows)

    # Temporal stability on flagged pairs
    stability_df = temporal_stability(sample, pair_df.head(50)) if not pair_df.empty else pd.DataFrame()

    # Interpretability checks
    interp_df = interpretability_checks(sample, numeric_cols)

    out_dir = Path(__file__).resolve().parents[2] / "analysis" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(out_dir / "pairwise_dependence.csv", index=False)
    theme_df.to_csv(out_dir / "theme_redundancy.csv", index=False)
    stability_df.to_csv(out_dir / "temporal_stability.csv", index=False)
    interp_df.to_csv(out_dir / "interpretability_checks.csv", index=False)

    logger.info(f"Diagnostics complete. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
