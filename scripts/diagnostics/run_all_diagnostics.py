"""
run_all_diagnostics.py

Convenience wrapper to run the full diagnostics stack:
1) feature_diagnostics.py (writes CSVs)
2) visualize_diagnostics.py (writes figures)
3) detect_redundant_features.py (writes config/redundant_features.yaml)
4) sub-ensemble correlations (per submodel)

Assumes the feature matrix exists at data/processed/feature_matrix.parquet.
"""

import sys
from pathlib import Path

import pandas as pd

file_path = Path(__file__).resolve()
ROOT = file_path.parents[2]
sys.path.insert(0, str(ROOT))

from utils import logger, PATHS, load_configs  # noqa: E402


def main():
    try:
        logger.info("Running diagnostics: feature_diagnostics -> visualize_diagnostics -> detect_redundant_features -> sub-ensemble correlations")

        # 1) Core diagnostics (CSV outputs)
        from scripts.diagnostics import feature_diagnostics

        feature_diagnostics.main()

        # 2) Visualization (figures)
        from scripts.diagnostics import visualize_diagnostics

        visualize_diagnostics.main()

        # 3) Redundancy analysis (pruning suggestions)
        from scripts.diagnostics import detect_redundant_features

        detect_redundant_features.main()

        # 4) Sub-ensemble correlations
        cfgs = load_configs()
        df = pd.read_parquet(PATHS["data_proc"] / "feature_matrix.parquet")
        out_dir = (PATHS.get("analysis", PATHS["root"] / "analysis") / "diagnostics" / "sub_ensemble_corrs")
        feature_diagnostics.analyze_sub_ensemble_correlations(df=df, models_cfg=cfgs["models"], output_dir=out_dir)

        logger.info("Diagnostics pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Diagnostics pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
