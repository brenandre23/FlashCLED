"""
pipeline/common/diagnostic_utils.py
===================================
Utilities for feature diagnostics with configurable column exclusions.

This module provides centralized logic for:
1. Identifying structural break flags (data availability indicators)
2. Filtering columns for diagnostic analysis
3. Loading clean feature matrices for collinearity/VIF analysis

Structural break flags are binary (0/1) columns that indicate whether
data from a specific source was available at a given time. They're
essential for XGBoost training (teaching the model about data availability)
but create noise in diagnostic analysis (VIF, correlation, etc.).
"""

import re
from pathlib import Path
from typing import List, Set, Dict, Any, Optional, Union
import pandas as pd

import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import logger, PATHS, load_configs


# =============================================================================
# DEFAULT STRUCTURAL BREAK FLAGS
# These are columns that indicate data availability periods and should
# typically be excluded from diagnostic analysis (VIF, correlation, etc.)
# =============================================================================

DEFAULT_STRUCTURAL_BREAK_FLAGS: List[str] = [
    # WorldPop version flag
    "is_worldpop_v1",
    
    # Data availability flags
    "iom_data_available",
    "econ_data_available",
    "ioda_data_available",
    "landcover_data_available",
    "viirs_data_available",
    "gdelt_data_available",
    "food_data_available",
    
    # Any future availability flags should follow this pattern
]

# Regex patterns for auto-detecting structural break flags
STRUCTURAL_FLAG_PATTERNS: List[str] = [
    r"^is_[a-z]+_v\d+$",           # is_worldpop_v1, is_acled_v2, etc.
    r".*_data_available.*",        # any_data_available, data_available_flag
    r".*_avail(able)?$",            # landcover_data_available, legacy *_avail, etc.
    r"^is_[a-z]+_available$",      # legacy patterns
]


def get_structural_break_flags(features_cfg: Optional[Dict[str, Any]] = None) -> Set[str]:
    """
    Get the complete set of structural break flag column names.
    
    Combines:
    1. Hardcoded defaults (known flags)
    2. Config-defined flags (from features.yaml diagnostics section)
    
    Parameters
    ----------
    features_cfg : dict, optional
        Features configuration from features.yaml. If None, loads from disk.
        
    Returns
    -------
    Set[str]
        Set of column names to exclude as structural break flags
    """
    flags = set(DEFAULT_STRUCTURAL_BREAK_FLAGS)
    
    # Load config if not provided
    if features_cfg is None:
        try:
            configs = load_configs()
            features_cfg = configs["features"]
        except Exception as e:
            logger.warning(f"Could not load features config: {e}. Using defaults only.")
            return flags
    
    # Add config-defined flags
    diag_cfg = features_cfg.get("diagnostics", {})
    config_flags = diag_cfg.get("structural_break_flags", [])
    if config_flags:
        flags.update(config_flags)
    
    return flags


def detect_structural_flags_in_dataframe(df: pd.DataFrame) -> Set[str]:
    """
    Auto-detect structural break flags in a DataFrame based on patterns.
    
    Uses regex patterns to identify columns that look like data availability
    indicators. Useful for discovering flags that weren't explicitly configured.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to scan for structural break flags
        
    Returns
    -------
    Set[str]
        Set of detected column names matching structural flag patterns
    """
    detected = set()
    
    for col in df.columns:
        for pattern in STRUCTURAL_FLAG_PATTERNS:
            if re.match(pattern, col, re.IGNORECASE):
                detected.add(col)
                break
                
    return detected


def get_diagnostic_exclusions(
    features_cfg: Optional[Dict[str, Any]] = None,
    include_structural_breaks: bool = True,
    extra_exclusions: Optional[List[str]] = None
) -> Set[str]:
    """
    Get the complete set of columns to exclude from diagnostic analysis.
    
    Combines:
    1. Structural break flags (if include_structural_breaks=True)
    2. Config-defined exclusions (from features.yaml diagnostics section)
    3. Extra exclusions passed as argument
    4. Standard exclusions (targets, identifiers)
    
    Parameters
    ----------
    features_cfg : dict, optional
        Features configuration from features.yaml
    include_structural_breaks : bool
        If True, exclude structural break flags (default: True)
    extra_exclusions : list, optional
        Additional column names to exclude
        
    Returns
    -------
    Set[str]
        Set of column names to exclude from diagnostics
    """
    exclusions = set()
    
    # Load config if not provided
    if features_cfg is None:
        try:
            configs = load_configs()
            features_cfg = configs["features"]
        except Exception as e:
            logger.warning(f"Could not load features config: {e}")
            features_cfg = {}
    
    # 1. Structural break flags
    if include_structural_breaks:
        exclusions.update(get_structural_break_flags(features_cfg))
    
    # 2. Config-defined exclusions
    diag_cfg = features_cfg.get("diagnostics", {})
    config_exclusions = diag_cfg.get("exclude_from_diagnostics", [])
    if config_exclusions:
        exclusions.update(config_exclusions)
    
    # 3. Extra exclusions
    if extra_exclusions:
        exclusions.update(extra_exclusions)
    
    return exclusions


def get_standard_exclusion_patterns() -> List[str]:
    """
    Get regex patterns for columns that should always be excluded from
    feature diagnostics (identifiers, targets, metadata).
    
    Returns
    -------
    List[str]
        List of regex patterns
    """
    return [
        r"^target_",       # All target columns
        r"^h3_index$",     # Spatial identifier
        r"^date$",         # Temporal identifier
        r"^year$",         # Year column
        r"^epoch$",        # Epoch identifier
        r"^geometry$",     # Geometry column
        r"^admin\d+",      # Admin columns
    ]


def filter_diagnostic_columns(
    df: pd.DataFrame,
    exclude_structural_breaks: bool = True,
    extra_exclusions: Optional[List[str]] = None,
    features_cfg: Optional[Dict[str, Any]] = None,
    auto_detect_flags: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter a DataFrame to remove columns not suitable for diagnostic analysis.
    
    This is the main entry point for preparing data for VIF/correlation analysis.
    It removes:
    - Non-numeric columns
    - Structural break flags (optional)
    - Config-defined exclusions
    - Target columns
    - Identifier columns
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    exclude_structural_breaks : bool
        If True, remove structural break flag columns (default: True)
    extra_exclusions : list, optional
        Additional column names to exclude
    features_cfg : dict, optional
        Features configuration (loaded if not provided)
    auto_detect_flags : bool
        If True, also detect structural flags via regex patterns
    verbose : bool
        If True, log information about excluded columns
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only diagnostic-ready columns
    """
    original_cols = set(df.columns)
    
    # 1. Start with numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    numeric_cols = set(numeric_df.columns)
    non_numeric = original_cols - numeric_cols
    
    # 2. Get exclusions
    exclusions = get_diagnostic_exclusions(
        features_cfg=features_cfg,
        include_structural_breaks=exclude_structural_breaks,
        extra_exclusions=extra_exclusions
    )
    
    # 3. Auto-detect additional structural flags
    if auto_detect_flags:
        detected = detect_structural_flags_in_dataframe(df)
        if detected:
            exclusions.update(detected)
    
    # 4. Apply standard exclusion patterns
    pattern_exclusions = set()
    for col in numeric_cols:
        for pattern in get_standard_exclusion_patterns():
            if re.match(pattern, col, re.IGNORECASE):
                pattern_exclusions.add(col)
                break
    exclusions.update(pattern_exclusions)
    
    # 5. Filter columns
    cols_to_exclude = exclusions & numeric_cols
    final_cols = [c for c in numeric_df.columns if c not in exclusions]
    
    if verbose:
        logger.info(f"Diagnostic column filtering:")
        logger.info(f"  Original columns: {len(original_cols)}")
        logger.info(f"  Numeric columns: {len(numeric_cols)}")
        logger.info(f"  Excluded (non-numeric): {len(non_numeric)}")
        logger.info(f"  Excluded (structural breaks): {len(cols_to_exclude & get_structural_break_flags(features_cfg))}")
        logger.info(f"  Excluded (patterns/config): {len(cols_to_exclude - get_structural_break_flags(features_cfg))}")
        logger.info(f"  Final diagnostic columns: {len(final_cols)}")
        
        # List excluded structural break flags
        excluded_flags = sorted(cols_to_exclude & get_structural_break_flags(features_cfg))
        if excluded_flags:
            logger.info(f"  Structural break flags excluded: {excluded_flags}")
    
    return df[final_cols].copy()


def load_feature_matrix_for_diagnostics(
    parquet_path: Optional[Path] = None,
    exclude_structural_breaks: bool = True,
    extra_exclusions: Optional[List[str]] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load feature matrix with diagnostic filtering applied.
    
    Convenience function that loads the parquet file and applies
    diagnostic column filtering in one step.
    
    Parameters
    ----------
    parquet_path : Path, optional
        Path to feature matrix parquet. Defaults to standard location.
    exclude_structural_breaks : bool
        If True, exclude structural break flags (default: True)
    extra_exclusions : list, optional
        Additional columns to exclude
    sample_frac : float, optional
        If provided, sample this fraction of rows (for large datasets)
    random_state : int
        Random seed for sampling
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame ready for diagnostic analysis
    """
    # Resolve path
    if parquet_path is None:
        parquet_path = PATHS.get("data_proc", Path("data/processed")) / "feature_matrix.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {parquet_path}\n"
            "Run build_feature_matrix.py first."
        )
    
    logger.info(f"Loading feature matrix from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Optional sampling
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)
        logger.info(f"  Sampled: {len(df):,} rows ({sample_frac*100:.0f}%)")
    
    # Apply diagnostic filtering
    df_filtered = filter_diagnostic_columns(
        df,
        exclude_structural_breaks=exclude_structural_breaks,
        extra_exclusions=extra_exclusions
    )
    
    return df_filtered


def summarize_structural_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of structural break flags in a DataFrame.
    
    Useful for understanding the data availability landscape before
    deciding which columns to exclude.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing potential structural break flags
        
    Returns
    -------
    pd.DataFrame
        Summary with columns: flag_name, unique_values, coverage_pct, first_nonzero_date
    """
    flags = get_structural_break_flags()
    present_flags = [f for f in flags if f in df.columns]
    
    summaries = []
    for flag in present_flags:
        series = df[flag]
        coverage = (series == 1).mean() * 100
        
        # Try to find first non-zero date
        first_nonzero = None
        if "date" in df.columns:
            nonzero_mask = series == 1
            if nonzero_mask.any():
                first_nonzero = df.loc[nonzero_mask, "date"].min()
        
        summaries.append({
            "flag_name": flag,
            "unique_values": sorted(series.dropna().unique().tolist()),
            "coverage_pct": round(coverage, 1),
            "first_nonzero_date": first_nonzero
        })
    
    return pd.DataFrame(summaries)


# =============================================================================
# CLI UTILITIES
# =============================================================================

def parse_exclusion_list(exclusion_str: Optional[str]) -> List[str]:
    """
    Parse a comma-separated string of column names into a list.
    
    Parameters
    ----------
    exclusion_str : str, optional
        Comma-separated column names (e.g., "col1,col2,col3")
        
    Returns
    -------
    List[str]
        List of column names, empty if input is None/empty
    """
    if not exclusion_str:
        return []
    return [c.strip() for c in exclusion_str.split(",") if c.strip()]
