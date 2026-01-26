# CEWP Usage Guide

Practical examples for running the Conflict Early Warning Pipeline.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Phase Control](#phase-control)
- [Date Range Specification](#date-range-specification)
- [Diagnostic Modes](#diagnostic-modes)
- [Advanced Workflows](#advanced-workflows)
- [Output Interpretation](#output-interpretation)
- [Command Reference](#command-reference)
- [Common Issues and Solutions](#common-issues-and-solutions)

---

## Basic Usage

### Full Pipeline Execution

Run all phases for the default date range (configured in `configs/data.yaml`):

```bash
python main.py
```

**Expected runtime:** 2-4 hours  
**Output:** Feature matrices, trained models, predictions in `data/processed/`

### Quick Test Run

Test the pipeline with a small date range:

```bash
python main.py --start-date 2024-01-01 --end-date 2024-03-31
```

**Expected runtime:** 20-30 minutes

---

## Phase Control

The pipeline has 5 phases that can be selectively enabled/disabled:

### 1. Skip Individual Phases

```bash
# Skip static data ingestion (useful if already run)
python main.py --skip-static

# Skip dynamic data ingestion
python main.py --skip-dynamic

# Skip feature engineering
python main.py --skip-features

# Skip modeling and prediction
python main.py --skip-modeling

# Skip post-run analysis
python main.py --skip-analysis
```

### 2. Run Only Specific Phases

```bash
# Run only static ingestion
python main.py --skip-dynamic --skip-features --skip-modeling --skip-analysis

# Run only modeling (assumes feature matrix exists)
python main.py --skip-static --skip-dynamic --skip-features --skip-analysis

# Run everything except analysis (faster iteration)
python main.py --skip-analysis
```

### 3. Combine Multiple Flags

```bash
# Common: Re-run feature engineering and modeling only
python main.py \
  --skip-static \
  --skip-dynamic \
  --skip-analysis
```

### Incremental Ingestion & Force-Refresh

The pipeline uses an **incremental-by-default** strategy for all dynamic data sources to minimize API load and processing time.

- **Default Behavior:** For each source, the pipeline queries the database for the `MAX(date)` and only fetches new records between that date and the requested `--end-date`.
- **Force Refresh (`--no-incremental`):** Use this flag to override the incremental logic and refetch the entire window from the configured `start_date`.
- **Overlap Sources:** To ensure data integrity for sources with late-arriving or reprocessed data, the following sources include a mandatory 14-day safety overlap during incremental fetches:
  - GEE Environmental features
  - Dynamic World land cover
  - IOM DTM displacement
  - GDELT (7-day buffer)

---

## Date Range Specification

### Override Config Dates

```bash
# Specify custom date window
python main.py \
  --start-date 2020-01-01 \
  --end-date 2023-12-31
```

### Recent Data Only

```bash
# Last 2 years of data
python main.py \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

### Historical Analysis

```bash
# Pre-COVID period
python main.py \
  --start-date 2015-01-01 \
  --end-date 2019-12-31
```

---

## Diagnostic Modes

The pipeline includes specialized diagnostic modes for analyzing feature quality before model training. These modes **automatically exclude structural break flags** (data availability indicators) that add noise to VIF/correlation analysis.

### Prerequisites

**Important:** Diagnostic modes require an existing feature matrix. You must run the pipeline first:

```bash
# Option 1: Full pipeline
python main.py

# Option 2: Build matrix without modeling (faster)
python main.py --skip-modeling --skip-analysis
```

### Run Full Diagnostics

```bash
# Run all diagnostics (VIF, correlation, stability analysis)
# Structural break flags excluded by default
python main.py --run-diagnostics-only
```

Output files in `analysis/diagnostics/`:
- `pairwise_dependence.csv` - Feature pairs with |ρ| ≥ 0.6
- `theme_redundancy.csv` - Per-theme VIF, condition number, PCA variance
- `temporal_stability.csv` - Correlation stability across time periods
- `interpretability_checks.csv` - Per-feature variance, entropy, max correlation
- `excluded_columns.csv` - Columns excluded and reasons

### Run Collinearity Check Only

```bash
# Just VIF and correlation analysis
python main.py --run-collinearity-only
```

Output files in `data/processed/`:
- `vif_analysis.csv` - Full VIF analysis
- `correlation_matrix.parquet` - Complete correlation matrix
- `feature_clusters.json` - Hierarchical clustering of correlated features
- `suggested_removals.json` - Features flagged for potential removal

### Control Structural Break Filtering

```bash
# Include structural break flags (for comparison)
python main.py --run-diagnostics-only --include-structural-breaks

# Exclude additional columns beyond structural breaks
python main.py --run-diagnostics-only --diagnostic-exclude-cols "cw_score_local,gdelt_goldstein_mean"

# Use sampling for faster analysis on large datasets
python main.py --run-diagnostics-only --diagnostic-sample-frac 0.3
```

### List Structural Break Flags

```bash
# Show all configured structural break flags and their coverage
python main.py --list-structural-breaks
```

Example output:
```
============================================================
STRUCTURAL BREAK FLAGS
============================================================

Total configured flags: 10

  - econ_data_available
  - food_data_available
  - gdelt_data_available
  - ioda_data_available
  - iom_data_available
  - is_viirs_available
  - is_worldpop_v1
  - landcover_avail

------------------------------------------------------------
Checking feature matrix for structural break flags...
  Present in matrix: 7
    ✓ econ_data_available
    ✓ ioda_data_available
    ✓ iom_data_available
    ...

Coverage summary:
  econ_data_available: 85.2% coverage (values: [0, 1])
  iom_data_available: 72.3% coverage (values: [0, 1])
  ...
```

### Standalone Diagnostic Scripts

```bash
# Run feature diagnostics directly
python -m scripts.diagnostics.feature_diagnostics --exclude-structural-breaks

# Run collinearity check with custom thresholds
python -m scripts.collinearity_check \
  --vif-threshold 5.0 \
  --corr-threshold 0.8 \
  --exclude-structural-breaks

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

# Load pre-filtered matrix
df = load_feature_matrix_for_diagnostics(
    exclude_structural_breaks=True,
    extra_exclusions=["col1", "col2"],
    sample_frac=0.5
)

# Or filter an existing DataFrame
df_filtered = filter_diagnostic_columns(
    df,
    exclude_structural_breaks=True,
    extra_exclusions=["col1"]
)

# List configured flags
flags = get_structural_break_flags()
print(f"Structural break flags: {flags}")

# Get coverage summary
summary = summarize_structural_flags(df)
print(summary)
```

### Configuration

Structural break flags are configured in `configs/features.yaml`:

```yaml
diagnostics:
  structural_break_flags:
    - is_worldpop_v1
    - iom_data_available
    - econ_data_available
    - ioda_data_available
    - landcover_avail
    - is_viirs_available
    - gdelt_data_available
    - food_data_available
  
  exclude_from_diagnostics: []  # Add columns here
  
  thresholds:
    vif_warning: 5.0
    vif_severe: 10.0
    correlation_warning: 0.7
    correlation_severe: 0.9
```

### Why Filter Structural Break Flags?

Structural break flags are binary (0/1) columns indicating data availability periods:

| Flag | Purpose |
|------|--------|
| `is_worldpop_v1` | WorldPop V1 vs V2 methodology (pre-2015) |
| `iom_data_available` | IOM DTM coverage start (2015-01-31) |
| `econ_data_available` | Yahoo Finance coverage (2003-12-01) |
| `ioda_data_available` | IODA monitoring start (2022-02-01) |
| `landcover_avail` | Dynamic World start (2015-06-27) |
| `is_viirs_available` | VIIRS coverage start (2012-01-28) |

These flags are **essential for XGBoost training** (teaching the model when data becomes available) but:

1. **Inflate VIF artificially** - Binary flags with high prevalence correlate spuriously with many features
2. **Pollute correlation matrices** - Show up as "highly correlated" with unrelated features
3. **Distort clustering** - Create artificial feature clusters based on data availability

By default, diagnostic modes exclude them to give you a cleaner picture of actual feature relationships. Use `--include-structural-breaks` to include them when needed.

---

## Advanced Workflows

### Scenario 1: Update Data Only

You have existing models but want to refresh with latest data:

```bash
# Step 1: Ingest new data
python main.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-04 \
  --skip-modeling \
  --skip-analysis

# Step 2: Generate predictions with existing models
python pipeline/modeling/generate_predictions.py 14d
python pipeline/modeling/generate_predictions.py 1m
python pipeline/modeling/generate_predictions.py 3m
```

### Scenario 2: Retrain Models with Different Features

```bash
# Step 1: Edit configs/features.yaml to enable/disable features
nano configs/features.yaml

# Step 2: Rebuild feature matrix
python main.py \
  --skip-static \
  --skip-dynamic

# Step 3: Verify features
python -c "import pandas as pd; df = pd.read_parquet('data/processed/feature_matrix.parquet'); print(df.columns.tolist())"
```

### Scenario 3: Experiment with Model Hyperparameters

```bash
# Step 1: Edit configs/models.yaml
nano configs/models.yaml

# Step 2: Retrain models only
python main.py \
  --skip-static \
  --skip-dynamic \
  --skip-features
```

### Scenario 4: Database Reset and Fresh Start

```bash
# Step 1: Reset database schema
python main.py --reset-schema

# Step 2: Run full pipeline
python main.py
```

**⚠️ Warning:** `--reset-schema` deletes ALL data in the `car_cewp` schema.

### Scenario 5: Validation Before Full Run

```bash
# Check prerequisites without running pipeline
python main.py --validate-only

# Example output:
# ✅ Validation passed for planned execution path.
# OR
# ❌ Validation failed. Missing table: car_cewp.features_static
```

---

## Output Interpretation

### Feature Matrix

Location: `data/processed/feature_matrix.parquet`

```bash
# Inspect structure
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/feature_matrix.parquet')
print(f'Shape: {df.shape}')
print(f'Date range: {df.date.min()} to {df.date.max()}')
print(f'H3 cells: {df.h3_index.nunique()}')
print(f'Features: {len(df.columns) - 2}')  # Minus h3_index and date
"
```

### Predictions

Locations:
- `data/processed/predictions_14d.csv` - 14-day forecast
- `data/processed/predictions_1m.csv` - 1-month forecast
- `data/processed/predictions_3m.csv` - 3-month forecast

```bash
# View sample predictions
python -c "
import pandas as pd
df = pd.read_csv('data/processed/predictions_14d.csv')
print(df.nlargest(10, 'predicted_fatalities')[['h3_index', 'date', 'predicted_fatalities', 'probability_conflict']])
"
```

### Model Artifacts

Location: `models/`

```bash
ls -lh models/
# two_stage_ensemble_14d_xgboost.pkl
# two_stage_ensemble_1m_xgboost.pkl
# two_stage_ensemble_3m_xgboost.pkl
```

### Analysis Outputs

Location: `analysis/`

Generated plots include:
- `macro_feature_importance.png` - Top-level feature importance
- `subtheme_shap_*.png` - SHAP values by feature category
- `model_comparison_*.png` - Model performance metrics
- `spatial_residuals_*.png` - Geographic error patterns

---

## Command Reference

### All Available Flags

```bash
python main.py --help

Options:
  --start-date TEXT              Override start date (YYYY-MM-DD)
  --end-date TEXT                Override end date (YYYY-MM-DD)
  --reset-schema                 Drop and recreate database schema
  --skip-static                  Skip static data ingestion
  --skip-dynamic                 Skip dynamic data ingestion
  --skip-gdelt-themes            Skip GDELT themes fetch
  --skip-features                Skip feature engineering
  --skip-modeling                Skip model training and predictions
  --skip-analysis                Skip post-run analysis
  --stop-after-features          Run feature engineering, write schema summary, exit
  --stop-after-feature-matrix    Build feature matrix, then exit before modeling
  --step [acled_hybrid|build_matrix]
                                 Run a single pipeline step and exit
  --validate-only                Check prerequisites without running
  
  # Diagnostic Options
  --run-diagnostics-only         Run VIF/correlation diagnostics and exit
  --run-collinearity-only        Run collinearity check only and exit
  --exclude-structural-breaks    Exclude data availability flags from diagnostics (default)
  --include-structural-breaks    Include structural break flags in diagnostics
  --diagnostic-exclude-cols TEXT Comma-separated columns to exclude from diagnostics
  --diagnostic-sample-frac FLOAT Sample fraction for diagnostics (0.0-1.0)
  --list-structural-breaks       List all structural break flags and exit
  
  --help                         Show this message and exit
```

### Individual Script Execution

Most pipeline components can be run standalone:

```bash
# Database initialization
python init_db.py

# Individual ingestion scripts
python pipeline/ingestion/fetch_acled.py
python pipeline/ingestion/fetch_gee_server_side.py

# Feature engineering
python pipeline/processing/feature_engineering.py

# Build feature matrix
python pipeline/modeling/build_feature_matrix.py

# Train models
python pipeline/modeling/train_models.py

# Generate predictions
python pipeline/modeling/generate_predictions.py 14d

# Run analysis
python pipeline/analysis/analyze_feature_importance.py
```

---

## Common Issues and Solutions

### Issue: "Feature matrix not found"

**Cause:** You skipped too many phases or are running diagnostics without building the matrix first.

**Solution:**
```bash
# Build the feature matrix
python main.py --skip-modeling --skip-analysis

# Or rebuild just the matrix from existing DB
python main.py --step build_matrix
```

### Issue: "Feature matrix not found" when running --run-diagnostics-only

**Cause:** Diagnostic modes require an existing feature matrix. They don't build one from scratch.

**Solution:**
```bash
# Step 1: Build the pipeline first
python main.py --skip-modeling --skip-analysis

# Step 2: Then run diagnostics
python main.py --run-diagnostics-only
```

### Issue: "Missing required features" when building feature matrix

**Cause:** The database tables haven't been populated yet.

**Solution:** Run the full pipeline from the beginning:
```bash
python main.py
```

### Issue: "Model file not found"

**Solution:** You need to train models first:
```bash
python main.py --skip-static --skip-dynamic --skip-features
```

### Issue: Pipeline crashes during modeling

**Solution:** Reduce date range to lower memory usage:
```bash
python main.py --start-date 2023-01-01 --end-date 2024-01-01
```

### Issue: "Google Earth Engine quota exceeded"

**Solution:** GEE has daily limits. Wait 24 hours or use service account for higher quota.

### Issue: VIF values seem artificially high

**Cause:** Structural break flags may be included in the analysis.

**Solution:** Ensure you're excluding them (default behavior):
```bash
python main.py --run-diagnostics-only

# Or explicitly exclude
python main.py --run-collinearity-only --exclude-structural-breaks
```

### Issue: Diagnostics taking too long

**Solution:** Use sampling for faster iteration:
```bash
python main.py --run-diagnostics-only --diagnostic-sample-frac 0.3
```

---

## Tips and Best Practices

### Incremental Development

```bash
# Test changes quickly with small date range
python main.py --start-date 2024-11-01 --end-date 2024-11-30 --skip-analysis
```

### Monitoring Progress

```bash
# Watch logs in real-time
tail -f logs/pipeline.log
```

### Database Inspection

```bash
# Check what's been ingested
psql -d car_cewp -c "SELECT COUNT(*) FROM car_cewp.acled_events;"
psql -d car_cewp -c "SELECT MIN(date), MAX(date) FROM car_cewp.temporal_features;"
```

### Configuration Validation

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/data.yaml'))"
```
