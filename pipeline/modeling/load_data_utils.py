"""
pipeline/modeling/load_data_utils.py
=====================================
Utility helpers for pruning to load a training panel.

Reads the existing feature_matrix.parquet, filters by date window,
sanitizes problematic values, and returns a clean DataFrame.

UPDATES (2026-01-28):
- Enhanced sanitization for bracketed scientific notation
- Robust handling of string "nan", "None", null variants  
- Pre-emptive type coercion for all numeric-candidate columns
- Detailed logging of sanitization actions
- Validation of critical columns before returning
"""

from pathlib import Path
from typing import List, Optional, Tuple, Set
import pandas as pd
import numpy as np
import re

from utils import PATHS, logger
import yaml


# =============================================================================
# SANITIZATION CONSTANTS
# =============================================================================

# Patterns that indicate stringified numeric values
BRACKETED_PATTERN = re.compile(r'^\[(.+)\]$')  # Matches [5E-1], [1.234], etc.
SCIENTIFIC_PATTERN = re.compile(r'^-?[\d.]+[eE][+-]?\d+$')  # Matches 5E-1, 1.23E+05

# String representations of null/missing
NULL_STRINGS = frozenset(['nan', 'NaN', 'NAN', 'None', 'null', 'NULL', '', 'NA', 'N/A', 'n/a', '.'])

# Columns that should NEVER be converted to numeric (keep as object/category)
NON_NUMERIC_COLUMNS = frozenset([
    'h3_index',  # Keep as int64, not float
    'admin0', 'admin1', 'admin2',  # Geographic names
    'country', 'region', 'district',
])

# Features that should NEVER be pruned or dropped (Structural Whitelist)
STRUCTURAL_FLAGS = [
    "is_worldpop_v1", "iom_data_available", "econ_data_available",
    "ioda_data_available", "landcover_data_available",
    "viirs_data_available", "gdelt_data_available", "food_data_available"
]


# =============================================================================
# CORE SANITIZATION FUNCTIONS
# =============================================================================

def _clean_bracketed_value(val: str) -> Optional[float]:
    """
    Extract numeric value from bracketed string like '[5E-1]'.
    
    Returns:
        float if parseable, None otherwise
    """
    if pd.isna(val):
        return None
    
    val_str = str(val).strip()
    
    # Check for bracketed pattern
    match = BRACKETED_PATTERN.match(val_str)
    if match:
        inner = match.group(1).strip()
        try:
            return float(inner)
        except ValueError:
            return None
    
    return None


def _is_likely_numeric_column(series: pd.Series, sample_size: int = 100) -> bool:
    """
    Heuristic check if an object column should be numeric.
    
    Samples non-null values and checks if majority parse as numbers.
    """
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return False
    
    # Filter out known null strings
    sample = sample[~sample.astype(str).isin(NULL_STRINGS)]
    if sample.empty:
        return False
    
    # Try to parse as numeric
    numeric_count = 0
    for val in sample:
        val_str = str(val).strip()
        
        # Check bracketed
        if BRACKETED_PATTERN.match(val_str):
            numeric_count += 1
            continue
        
        # Check direct numeric parse
        try:
            float(val_str)
            numeric_count += 1
        except ValueError:
            pass
    
    # If >70% parse as numeric, likely a numeric column
    return numeric_count / len(sample) > 0.7


def sanitize_column(series: pd.Series, col_name: str) -> Tuple[pd.Series, dict]:
    """
    Sanitize a single column, converting stringified numerics to proper floats.
    
    Returns:
        (cleaned_series, stats_dict)
    """
    stats = {
        'original_dtype': str(series.dtype),
        'bracketed_cleaned': 0,
        'null_strings_cleaned': 0,
        'coercion_failures': 0,
        'final_dtype': None
    }
    
    if series.dtype == 'object':
        # Always attempt bracket-stripping for object columns before probing with sample.
        # Bracketed scientific-notation values ([9.735312E0]) can be sparse enough to
        # miss the 100-row sample, causing them to survive into TreeExplainer and crash.
        cleaned = series.astype(str).str.strip()

        # Strip brackets unconditionally (safe: non-bracketed values are unaffected)
        bracketed_mask = cleaned.str.match(r'^\[.*\]$', na=False)
        if bracketed_mask.any():
            stats['bracketed_cleaned'] = int(bracketed_mask.sum())
            cleaned = cleaned.str.replace(r'^\[', '', regex=True)
            cleaned = cleaned.str.replace(r'\]$', '', regex=True)

        # Replace null strings with actual NaN
        null_mask = cleaned.isin(NULL_STRINGS)
        if null_mask.any():
            stats['null_strings_cleaned'] = int(null_mask.sum())
            cleaned = cleaned.replace(list(NULL_STRINGS), np.nan)

        # Attempt numeric conversion
        result = pd.to_numeric(cleaned, errors='coerce')

        # Accept conversion only if at least 50% of non-null values survived
        # (protects genuine string columns like admin names from being zeroed out)
        original_non_null = series.notna().sum()
        result_non_null = result.notna().sum()
        if original_non_null > 0 and result_non_null < original_non_null * 0.5:
            stats['final_dtype'] = 'object (kept)'
            stats['bracketed_cleaned'] = 0
            stats['null_strings_cleaned'] = 0
            return series, stats

        stats['coercion_failures'] = int(original_non_null - result_non_null - stats['null_strings_cleaned'])
        stats['final_dtype'] = str(result.dtype)

        return result, stats
    
    # Already numeric - just return as-is
    stats['final_dtype'] = str(series.dtype)
    return series, stats


def sanitize_dataframe(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Sanitize entire DataFrame, fixing stringified numeric issues.
    
    Returns:
        (cleaned_df, summary_stats)
    """
    all_stats = {}
    
    # Identify object columns that might need cleaning
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    object_cols = [c for c in object_cols if c not in NON_NUMERIC_COLUMNS]
    
    cleaned_count = 0
    for col in object_cols:
        cleaned_series, stats = sanitize_column(df[col], col)
        
        # Only update if something changed
        if stats['bracketed_cleaned'] > 0 or stats['null_strings_cleaned'] > 0:
            df[col] = cleaned_series
            all_stats[col] = stats
            cleaned_count += 1
            
            if verbose and (stats['bracketed_cleaned'] > 0 or stats['coercion_failures'] > 0):
                logger.info(
                    f"  Sanitized '{col}': {stats['bracketed_cleaned']} bracketed, "
                    f"{stats['null_strings_cleaned']} null strings, "
                    f"{stats['coercion_failures']} coercion failures"
                )
    
    summary = {
        'columns_cleaned': cleaned_count,
        'total_object_cols': len(object_cols),
        'details': all_stats
    }
    
    return df, summary


# =============================================================================
# PRUNING HELPERS
# =============================================================================

def load_pruned_feature_set(full_feature_list: List[str], pruned_path: Path = PATHS["root"] / "configs" / "pruned_features.yaml") -> List[str]:
    """
    Intersect a feature list with configs/pruned_features.yaml if it exists.
    
    Args:
        full_feature_list: Features requested by models.yaml for a theme.
        pruned_path: Path to pruning registry.
    Returns:
        Filtered list (or original if registry missing/unreadable).
    """
    if not pruned_path.exists():
        return full_feature_list

    try:
        cfg = yaml.safe_load(pruned_path.read_text()) or {}
        pruned = set(cfg.get("active_features", []))
        filtered = [f for f in full_feature_list if f in pruned]
        logger.info(f"[PRUNING] Applied registry: {len(full_feature_list)} -> {len(filtered)} features (file: {pruned_path})")
        return filtered
    except Exception as exc:
        logger.warning(f"[PRUNING] Failed to load {pruned_path}: {exc}. Using full feature list.")
        return full_feature_list


# =============================================================================
# MAIN LOADER FUNCTION
# =============================================================================

def load_training_data(
    engine: Optional[object],  # kept for API compatibility; not used
    start_date: str,
    end_date: str,
    targets: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load training data for pruning:
    - reads feature_matrix.parquet from data/processed
    - filters to the specified date window
    - sanitizes stringified numeric values
    - keeps numeric columns plus requested targets
    
    Parameters
    ----------
    engine : Optional[object]
        Database engine (not used, kept for API compatibility)
    start_date : str
        Start date for training window (inclusive)
    end_date : str
        End date for training window (inclusive)  
    targets : List[str]
        List of target column names to include
    verbose : bool
        Whether to log detailed sanitization info
        
    Returns
    -------
    pd.DataFrame
        Clean training data panel
    """
    fm_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not fm_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {fm_path}")

    logger.info(f"Pruning loader: reading {fm_path}")
    df = pd.read_parquet(fm_path)
    logger.info(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if "date" not in df.columns:
        raise ValueError("Feature matrix is missing 'date' column required for pruning.")

    # =========================================================================
    # PHASE 1: Sanitize stringified values BEFORE any filtering
    # =========================================================================
    logger.info("Pruning loader: Sanitizing stringified values...")
    df, sanitize_stats = sanitize_dataframe(df, verbose=verbose)
    
    if sanitize_stats['columns_cleaned'] > 0:
        logger.info(f"  Sanitized {sanitize_stats['columns_cleaned']} columns")
    else:
        logger.info("  No sanitization needed")

    # =========================================================================
    # PHASE 2: Date filtering
    # =========================================================================
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    
    # Check for date parse failures
    date_nulls = df["date"].isna().sum()
    if date_nulls > 0:
        logger.warning(f"  {date_nulls} rows have unparseable dates - will be excluded")
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
    df = df.loc[mask].copy()
    
    logger.info(f"  Date filtered to {start_date} - {end_date}: {len(df):,} rows")

    # =========================================================================
    # PHASE 3: Column selection
    # =========================================================================
    # Keep: numeric columns + explicitly requested targets + date + h3_index
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Always include these structural columns
    structural_cols = ['date', 'h3_index']
    
    # Build final column list (preserving order, no duplicates)
    keep_cols = []
    seen = set()
    
    for col in structural_cols + numeric_cols + targets:
        if col in df.columns and col not in seen:
            keep_cols.append(col)
            seen.add(col)
    
    # Check for missing targets
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        logger.warning(f"  Targets missing from feature matrix: {missing_targets}")
        # Try to find similar columns
        for mt in missing_targets:
            similar = [c for c in df.columns if mt.split('_')[0] in c][:3]
            if similar:
                logger.info(f"    Similar to '{mt}': {similar}")

    # Ensure structural flags are preserved even if non-numeric/low variance
    for flag in STRUCTURAL_FLAGS:
        if flag in df.columns and flag not in keep_cols:
            keep_cols.append(flag)

    df = df[keep_cols].copy()
    
    # =========================================================================
    # PHASE 4: Final validation
    # =========================================================================
    # Ensure h3_index is int64 (not object or float)
    if 'h3_index' in df.columns:
        df['h3_index'] = df['h3_index'].astype('int64')
    
    # Report final dtype composition
    dtype_counts = df.dtypes.apply(lambda x: x.name).value_counts()
    logger.info(f"  Final dtypes: {dict(dtype_counts)}")
    
    # Warn about remaining object columns
    remaining_objects = df.select_dtypes(include=['object']).columns.tolist()
    if remaining_objects:
        logger.warning(f"  Remaining object columns (excluded from candidates): {remaining_objects}")

    logger.info(f"Pruning loader: filtered to {len(df):,} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# UTILITIES FOR EXTERNAL USE
# =============================================================================

def get_numeric_candidates(df: pd.DataFrame, exclude: Optional[Set[str]] = None) -> List[str]:
    """
    Get list of numeric columns suitable as pruning candidates.
    
    Excludes: h3_index, date, and any columns in exclude set.
    """
    exclude = exclude or set()
    exclude.update(['h3_index', 'date'])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def validate_training_data(df: pd.DataFrame, targets: List[str]) -> dict:
    """
    Validate loaded training data for common issues.
    
    Returns dict with validation results.
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check minimum rows
    if len(df) < 1000:
        results['warnings'].append(f"Very few rows: {len(df)}")
    
    # Check date coverage
    if 'date' in df.columns:
        date_range = (df['date'].max() - df['date'].min()).days
        if date_range < 365:
            results['warnings'].append(f"Short date range: {date_range} days")
    
    # Check target availability
    for target in targets:
        if target not in df.columns:
            results['errors'].append(f"Missing target: {target}")
            results['valid'] = False
        else:
            non_zero = (df[target] != 0).sum()
            if non_zero < 20:
                results['warnings'].append(f"Very few positive cases in {target}: {non_zero}")
    
    # Check for constant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    if constant_cols:
        results['warnings'].append(f"{len(constant_cols)} constant columns detected")
    
    return results
