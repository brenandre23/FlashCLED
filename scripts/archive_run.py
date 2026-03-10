"""
archive_run.py
==============
Snapshots all outputs from the current pipeline run into a versioned directory.
Run this immediately after each training + prediction + analysis cycle completes,
before changing configs and starting the next run.

Usage:
    python scripts/archive_run.py --label xgboost_meta
    python scripts/archive_run.py --label logistic_meta

Outputs are saved to:
    data/runs/{label}/
        predictions/    — all predictions_*.parquet files
        metrics/        — comparison_metrics.csv, temporal_auc_by_year.csv,
                          model_selection_metrics.csv, conformal_diagnostics.csv
        models/         — two_stage_ensemble_*.pkl files
        analysis/       — SHAP plots, macro importance, model selection figures
        config_snapshot.yaml  — copy of models.yaml at time of archiving
"""

import shutil
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import logger, PATHS


def archive_run(label: str):
    dest = ROOT / "data" / "runs" / label
    dest.mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0

    def cp(src: Path, sub: str):
        nonlocal copied, skipped
        if not src.exists():
            logger.warning(f"  SKIP (not found): {src.name}")
            skipped += 1
            return
        target_dir = dest / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target_dir / src.name)
        logger.info(f"  ✓ {sub}/{src.name}")
        copied += 1

    logger.info(f"=== Archiving run: '{label}' → data/runs/{label}/ ===")

    # 1. Predictions
    for f in sorted(PATHS["data_proc"].glob("predictions_*.parquet")):
        cp(f, "predictions")

    # 2. Metrics
    for name in [
        "comparison_metrics.csv",
        "temporal_auc_by_year.csv",
    ]:
        cp(PATHS["data_proc"] / "analysis" / name, "metrics")

    for name in [
        "model_selection_metrics.csv",
        "conformal_diagnostics.csv",
    ]:
        cp(ROOT / "analysis" / name, "metrics")

    # 3. Models
    for f in sorted(PATHS["models"].glob("two_stage_ensemble_*.pkl")):
        cp(f, "models")

    # 4. Key analysis plots
    for f in sorted((ROOT / "analysis").glob("*.png")):
        cp(f, "analysis")

    # 5. Config snapshot
    cfg_src = PATHS["configs"] / "models.yaml"
    if cfg_src.exists():
        shutil.copy2(cfg_src, dest / "config_snapshot.yaml")
        logger.info(f"  ✓ config_snapshot.yaml")
        copied += 1

    logger.info(f"\nDone. {copied} files archived, {skipped} skipped.")
    logger.info(f"Path: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True,
                        help="Run label, e.g. 'xgboost_meta' or 'logistic_meta'")
    args = parser.parse_args()
    archive_run(args.label)
