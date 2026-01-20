# CEWP Diagnostic Filtering System

## Overview

The CEWP pipeline includes a comprehensive diagnostic filtering system that automatically handles **structural break flags** - binary data availability indicators that are essential for XGBoost training but add noise to diagnostic analysis (VIF, correlation, etc.).

## Quick Start

```bash
# Build the feature matrix first
python main.py --skip-modeling --skip-analysis

# Run diagnostics (structural breaks excluded by default)
python main.py --run-diagnostics-only

# Run collinearity check only
python main.py --run-collinearity-only
```

## What Are Structural Break Flags?

Structural break flags are columns that indicate whether data from a specific source was available at a given time:

| Flag | Source | Threshold Date | Description |
|------|--------|----------------|-------------|
| `is_worldpop_v1` | WorldPop | 2015-01-01 | V1 (census-adjusted) vs V2 (constrained) methodology |
| `iom_data_available` | IOM DTM | 2015-01-31 | IOM Displacement Tracking Matrix coverage start |
| `econ_data_available` | Yahoo Finance | 2003-12-01 | Economic indicators coverage start |
| `ioda_data_available` | IODA | 2022-02-01 | Internet outage detection coverage start |
| `landcover_avail` | Dynamic World | 2015-06-27 | Sentinel-2 landcover coverage start |
| `is_viirs_available` | VIIRS | 2012-01-28 | Nighttime lights coverage start |
| `gdelt_data_available` | GDELT | 2015-01-01 | GDELT v2 coverage start |
| `food_data_available` | FEWS NET | varies | Food security price data coverage |

### Why They Cause Problems in Diagnostics

These flags are **essential for XGBoost training** (teaching the model about data availability windows) but cause issues in diagnostic analysis:

1. **Inflate VIF artificially** - Binary flags with high prevalence correlate spuriously with many continuous features
2. **Pollute correlation matrices** - Show up as "highly correlated" with unrelated features due to temporal coincidence
3. **Distort feature clustering** - Create artificial clusters based on data availability rather than meaningful relationships
4. **Mislead redundancy detection** - May be flagged as "redundant" when they serve different purposes

### Why They're Still Important for Modeling

During model training, these flags teach XGBoost:
- When data sources become available
- How to handle missing data periods
- How to weight predictions based on data completeness

**The solution:** Exclude them from diagnostics but keep them in the modeling pipeline.

## Usage

### Via main.py (Recommended)

```bash
# Prerequisites: Build feature matrix first
python main.py --skip-modeling --skip-analysis

# Run diagnostics with structural breaks EXCLUDED (default)
python main.py --run-diagnostics-only

# Run diagnostics with structural breaks INCLUDED (for comparison)
python main.py --run-diagnostics-only --include-structural-breaks

# Exclude additional columns beyond structural breaks
python main.py --run-diagnostics-only --diagnostic-exclude-cols "col1,col2,col3"

# Faster analysis with sampling
python main.py --run-diagnostics-only --diagnostic-sample-frac 0.3

# Run collinearity check only
python main.py --run-collinearity-only

# List all configured structural break flags
python main.py --list-structural-breaks
```

### Standalone Scripts

```bash
# Feature diagnostics with exclusions
python -m scripts.diagnostics.feature_diagnostics --exclude-structural-breaks

# Feature diagnostics with extra exclusions
python -m scripts.diagnostics.feature_diagnostics --exclude-cols "col1,col2"

# Collinearity check with exclusions and custom thresholds
python -m scripts.collinearity_check \
  --exclude-structural-breaks \
  --vif-threshold 5.0 \
  --corr-threshold 0.8

# Specify custom parquet file
python -m scripts.diagnostics.feature_diagnostics \
  --parquet data/processed/custom_matrix.parquet
```

### Programmatic Usage

```python
from pipeline.common.diagnostic_utils import (
    filter_diagnostic_columns,
    load_feature_matrix_for_diagnostics,
    get_structural_break_flags,
    summarize_structural_flags
)

# Load pre-filtered matrix (easiest approach)
df = load_feature_matrix_for_diagnostics(
    exclude_structural_breaks=True,
    extra_exclusions=["some_column"],
    sample_frac=0.3
)

# Or filter an existing DataFrame
df_filtered = filter_diagnostic_columns(
    df,
    exclude_structural_breaks=True,
    extra_exclusions=["some_column"]
)

# List configured flags
flags = get_structural_break_flags()
print(f"Structural break flags: {flags}")

# Get coverage summary from a DataFrame
summary = summarize_structural_flags(df)
print(summary)
```

## Output Files

### Diagnostics Mode (`--run-diagnostics-only`)

Generated in `analysis/diagnostics/`:

| File | Description |
|------|-------------|
| `pairwise_dependence.csv` | Feature pairs with \|ρ\| ≥ 0.6 (Spearman, Kendall, MI) |
| `theme_redundancy.csv` | Per-theme VIF, condition number, PCA variance |
| `temporal_stability.csv` | Correlation stability across early/mid/late periods |
| `interpretability_checks.csv` | Per-feature variance, entropy, max correlation |
| `excluded_columns.csv` | List of columns excluded and reason why |

### Collinearity Mode (`--run-collinearity-only`)

Generated in `data/processed/`:

| File | Description |
|------|-------------|
| `vif_analysis.csv` | Full VIF analysis for all features |
| `correlation_matrix.parquet` | Complete Spearman correlation matrix |
| `feature_clusters.json` | Hierarchical clustering of correlated features |
| `suggested_removals.json` | Features suggested for removal |

## Configuration

Structural break flags and diagnostic thresholds are configured in `configs/features.yaml`:

```yaml
diagnostics:
  # Structural break flags: Binary (0/1) columns indicating data availability
  # Excluded from diagnostics by default when using --run-diagnostics-only
  structural_break_flags:
    - is_worldpop_v1              # WorldPop V1 vs V2 methodology
    - iom_data_available          # IOM DTM coverage start (2015-01-31)
    - econ_data_available         # Yahoo Finance coverage start (2003-12-01)
    - ioda_data_available         # IODA monitoring start (2022-02-01)
    - landcover_avail             # Dynamic World start (2015-06-27)
    - is_viirs_available          # VIIRS coverage start (2012-01-28)
    - gdelt_data_available        # GDELT v2 coverage start (2015)
    - food_data_available         # FEWS NET food price coverage
  
  # Additional columns to always exclude from diagnostics
  exclude_from_diagnostics: []
  
  # Thresholds for diagnostic flagging
  thresholds:
    vif_warning: 5.0              # VIF > 5 = moderate collinearity
    vif_severe: 10.0              # VIF > 10 = severe collinearity
    correlation_warning: 0.7      # |rho| > 0.7 = moderate correlation
    correlation_severe: 0.9       # |rho| > 0.9 = severe correlation
    condition_number: 30.0        # Condition number > 30 = ill-conditioned
```

### Adding New Flags

To add a new structural break flag:

1. Add it to `configs/features.yaml` under `diagnostics.structural_break_flags`
2. The diagnostic utilities will automatically exclude it

### Excluding Other Columns

To exclude columns that aren't structural break flags:

1. Add to `configs/features.yaml` under `diagnostics.exclude_from_diagnostics`
2. Or use `--diagnostic-exclude-cols` at runtime

## Typical Workflow

### Before (Manual Process)

```bash
# 1. Build feature matrix
python main.py --skip-modeling

# 2. Manually identify structural break flags
# 3. Open Python, load matrix, drop columns manually
# 4. Run diagnostics on filtered DataFrame
# 5. Identify more columns to drop
# 6. Repeat until satisfied
# 7. Finally run model
```

### After (Automated)

```bash
# 1. Build feature matrix (skip modeling for faster iteration)
python main.py --skip-modeling --skip-analysis

# 2. Run diagnostics (structural breaks auto-excluded)
python main.py --run-diagnostics-only

# 3. Review outputs, optionally exclude more columns
python main.py --run-diagnostics-only --diagnostic-exclude-cols "noisy_col1,noisy_col2"

# 4. Run modeling
python main.py --skip-static --skip-dynamic --skip-features
```

## Important Notes

1. **Structural break flags are still used in modeling** - They're only excluded from diagnostic analysis, not from the final model training.

2. **The exclusion is configurable** - Edit `features.yaml` to add/remove flags without code changes.

3. **Auto-detection available** - The system can auto-detect potential flags via regex patterns (columns matching `*_data_available*`, `is_*_v*`, etc.).

4. **The `excluded_columns.csv` output** documents exactly what was excluded and why, for full transparency.

5. **Use `--include-structural-breaks`** to compare diagnostics with and without the flags if you want to verify the filtering is helping.

## API Reference

### `get_structural_break_flags(features_cfg=None)`

Returns the set of structural break flag column names from config.

### `filter_diagnostic_columns(df, exclude_structural_breaks=True, extra_exclusions=None)`

Filters a DataFrame for diagnostic analysis. Removes:
- Non-numeric columns
- Target columns (`target_*`)
- Identifier columns (`h3_index`, `date`, etc.)
- Structural break flags (if `exclude_structural_breaks=True`)
- Extra exclusions

### `load_feature_matrix_for_diagnostics(parquet_path=None, exclude_structural_breaks=True, extra_exclusions=None, sample_frac=None)`

Convenience function that loads and filters in one step.

### `summarize_structural_flags(df)`

Returns a DataFrame summarizing structural break flag coverage (unique values, coverage %, first non-zero date).
