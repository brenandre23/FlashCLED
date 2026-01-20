"""
pipeline/common
===============
Shared utilities for the CEWP pipeline.

Modules:
- feature_loader: Load and filter feature matrices for inference/analysis
- diagnostic_utils: Utilities for diagnostic analysis with column exclusions
"""

from .feature_loader import build_required_columns, load_feature_matrix
from .diagnostic_utils import (
    get_structural_break_flags,
    get_diagnostic_exclusions,
    filter_diagnostic_columns,
    load_feature_matrix_for_diagnostics,
    summarize_structural_flags,
)

__all__ = [
    "build_required_columns",
    "load_feature_matrix",
    "get_structural_break_flags",
    "get_diagnostic_exclusions",
    "filter_diagnostic_columns",
    "load_feature_matrix_for_diagnostics",
    "summarize_structural_flags",
]
