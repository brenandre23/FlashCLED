"""
detect_redundant_features.py

Purpose: Identify highly correlated features and suggest a pruning list
based solely on the predictor matrix (no targets) prior to inference.
"""

import yaml
import pandas as pd
from pathlib import Path
import sys

file_path = Path(__file__).resolve()
ROOT = file_path.parents[2]
sys.path.insert(0, str(ROOT))

from utils import PATHS, load_configs, logger


def build_required_columns(models_cfg):
    required = {"h3_index", "date"}
    submodels = models_cfg.get("submodels", {})
    for _, cfg in submodels.items():
        if cfg.get("enabled"):
            required.update(cfg.get("features", []))
    return required


def main():
    configs = load_configs()
    models_cfg = configs["models"] if isinstance(configs, dict) else configs[2]

    feature_matrix_path = PATHS["data_proc"] / "feature_matrix.parquet"
    required_cols = build_required_columns(models_cfg)

    logger.info(f"Loading sample for redundancy analysis from {feature_matrix_path}")
    df = pd.read_parquet(feature_matrix_path, columns=list(required_cols))
    if df.empty:
        logger.error("Feature matrix is empty; aborting redundancy analysis.")
        return

    sample_n = min(200_000, len(df))
    df_sample = df.sample(n=sample_n, random_state=42)

    numeric_cols = df_sample.select_dtypes(include=[int, float]).columns
    df_sample[numeric_cols] = df_sample[numeric_cols].astype("float32")

    corr_matrix = df_sample[numeric_cols].corr(method="spearman")

    redundant_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr_matrix.iloc[i, j]
            if abs(rho) > 0.95:
                redundant_pairs.append((cols[i], cols[j], float(rho)))

    features_to_drop = set()
    for f1, f2, rho in redundant_pairs:
        if "_lag_" in f1 and "_lag_" not in f2:
            features_to_drop.add(f1)
        else:
            features_to_drop.add(f2)

    out_path = ROOT / "analysis" / "diagnostics" / "redundant_features.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump({"features_to_drop": sorted(features_to_drop)}, f)

    logger.info(
        f"Redundancy analysis complete. "
        f"Found {len(redundant_pairs)} high-corr pairs; "
        f"suggest dropping {len(features_to_drop)} features. "
        f"Saved to {out_path}"
    )


if __name__ == "__main__":
    main()
