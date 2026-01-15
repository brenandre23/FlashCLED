"""
run_all_diagnostics.py

Convenience wrapper to run the full diagnostics stack:
1) feature_diagnostics.py (writes CSVs)
2) visualize_diagnostics.py (writes figures)
3) detect_redundant_features.py (writes config/redundant_features.yaml)

Assumes the feature matrix exists at data/processed/feature_matrix.parquet.
"""

import sys
from pathlib import Path

file_path = Path(__file__).resolve()
ROOT = file_path.parents[2]
sys.path.insert(0, str(ROOT))

from utils import logger  # noqa: E402


def main():
    try:
        logger.info("Running diagnostics: feature_diagnostics -> visualize_diagnostics -> detect_redundant_features")

        # 1) Core diagnostics (CSV outputs)
        from scripts.diagnostics import feature_diagnostics

        feature_diagnostics.main()

        # 2) Visualization (figures)
        from scripts.diagnostics import visualize_diagnostics

        visualize_diagnostics.main()

        # 3) Redundancy analysis (pruning suggestions)
        from scripts.diagnostics import detect_redundant_features

        detect_redundant_features.main()

        logger.info("Diagnostics pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Diagnostics pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
