"""
scripts/diagnostics
===================
Diagnostic scripts for feature analysis and quality checks.

Modules:
- feature_diagnostics: RAM-aware diagnostics for collinearity, redundancy, and stability
- visualize_diagnostics: Publication-quality visualization of diagnostic results
- detect_redundant_features: Feature redundancy detection
- run_all_diagnostics: Run all diagnostic scripts in sequence
"""

from . import feature_diagnostics
from . import visualize_diagnostics

__all__ = ["feature_diagnostics", "visualize_diagnostics"]
