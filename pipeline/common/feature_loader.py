"""
Shared helpers for loading column-pruned feature matrices for inference/analysis.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Set, Iterable

import pandas as pd

from utils import PATHS, logger


def build_required_columns(models_cfg: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None,
                           extra: Optional[Iterable[str]] = None) -> Set[str]:
    """
    Build the minimal required column set for inference/analysis based on enabled submodels,
    optional PCA inputs, and any extra columns (e.g., targets) requested.
    """
    required: Set[str] = {"h3_index", "date"}

    submodels = models_cfg.get("submodels", {})
    for _, cfg in submodels.items():
        if cfg.get("enabled"):
            required.update(cfg.get("features", []))

    if bundle:
        pca_inputs = bundle.get("pca_input_features") or bundle.get("pca_input") or []
        required.update(pca_inputs)

    if extra:
        required.update(extra)

    return required


def load_feature_matrix(required_cols: Set[str], parquet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the feature matrix with column pruning. Falls back to the canonical path if none provided.
    """
    feature_matrix_path = parquet_path or PATHS["data_proc"] / "feature_matrix.parquet"
    if not feature_matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {feature_matrix_path}. "
            "Make sure build_feature_matrix.py wrote a canonical file."
        )
    logger.info(f"Loading feature matrix from: {feature_matrix_path} (cols={len(required_cols)})")
    df = pd.read_parquet(feature_matrix_path, columns=list(required_cols))

    if df.empty:
        logger.warning("Feature matrix is empty - downstream analyses may be missing inputs.")

    return df
