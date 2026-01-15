# CEWP Usage Guide

Practical examples for running the Conflict Early Warning Pipeline.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Phase Control](#phase-control)
- [Date Range Specification](#date-range-specification)
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

## Advanced Workflows

### Scenario 1: Update Data Only

You have existing models but want to refresh with latest data:

```bash
# Step 1: Ingest new data
python main.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
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
  --start-date TEXT      Override start date (YYYY-MM-DD)
  --end-date TEXT        Override end date (YYYY-MM-DD)
  --reset-schema         Drop and recreate database schema
  --skip-static          Skip static data ingestion
  --skip-dynamic         Skip dynamic data ingestion
  --skip-features        Skip feature engineering
  --skip-modeling        Skip model training and predictions
  --skip-analysis        Skip post-run analysis
  --validate-only        Check prerequisites without running
  --help                 Show this message and exit
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

**Solution:** You skipped too many phases. Run:
```bash
python main.py --skip-static --skip-dynamic
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
