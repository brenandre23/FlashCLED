# RUNBOOK.md - Canonical Operations

## 1. Environment Setup
```bash
# Initialize DB extensions and schema
python init_db.py

# Install dependencies (pinned where possible)
pip install -r requirements.txt
```

## 2. Pipeline Execution
- **Full Run:** `python main.py`
- **Skip Ingestion:** `python main.py --skip-static --skip-dynamic`
- **Modeling Only:** `python main.py --skip-static --skip-dynamic --skip-features`
- **Diagnostics Mode:** `python main.py --run-diagnostics-only`

## 3. Thesis Figure Regeneration
```bash
# Step 1: Refresh analysis data
python pipeline/analysis/analyze_predictions.py

# Step 2: Generate all SVG/PNG panels
python generate_thesis_figures.py
```

## 4. Maintenance & Audit
- **Check Data Gaps:** `/audit` (runs `scripts/audit_data_availability.py`)
- **Check Collinearity:** `python main.py --run-collinearity-only`
- **Clean Cache:** `python scripts/clean_cache.py`

## 5. Diagnostics & Post-Hoc Analysis
- **Conflict Onset Diagnostic (v2):** `conda run -n geo_env python scripts/diagnostics/onset_diagnostic.py`
  - Analyses hard-onset detection, lead-time distribution, never-flagged blind spots, and structural signal dip.
  - Add `--plot` to export figures to `Overleaf/Newest Figures/`.
- **Precision-Recall Maps:** `conda run -n geo_env python pipeline/analysis/generate_precision_recall_maps.py`
  - Generates Figure 5.8 diagnostic maps (TP / FN / Residuals) from current predictions parquet.

## 6. Deployment (WSL/Linux)
- Ensure PostgreSQL/PostGIS is running.
- GEE Service Account JSON path must be in `.env`.
