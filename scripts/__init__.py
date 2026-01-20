"""
scripts
=======
Utility scripts for the CEWP pipeline.

Modules:
- collinearity_check: VIF and correlation analysis for feature selection
- clean_cache: Cache management utilities
- build_graph: Graph construction utilities
"""

from . import collinearity_check

__all__ = ["collinearity_check"]
