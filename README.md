# Conflict Early Warning & Prediction System (CEWP) 🇨🇫

A production-grade geospatial machine learning pipeline for forecasting sub-national conflict in the **Central African Republic (CAR)**.

The system ingests multi-source data (satellite, economic, political, NLP-derived), engineers **141 features** (unique columns pre-pruning) on a hexagonal grid (**H3 resolution 5, ~10km cells**), and predicts conflict probability and fatality magnitude using a **Two-Stage Hurdle Ensemble** with calibrated uncertainty quantification.

---

## 🚀 Quick Start

Get the CEWP pipeline running in under 10 minutes.

### Prerequisites
- Python 3.10 or 3.11
- PostgreSQL 13+ with PostGIS and H3 extensions
- Google Earth Engine account (for satellite data)
- 16GB+ RAM recommended for full pipeline

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/CEWP-CAR.git
   cd CEWP-CAR
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up database:**
   ```bash
   # Create database
   createdb car_cewp
   
   # Enable extensions
   psql -d car_cewp -c "CREATE EXTENSION postgis; CREATE EXTENSION h3; CREATE EXTENSION h3_postgis;"
   ```

5. **Configure credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and API keys
   ```

### First Run

Run the initialization and a small test:

```bash
# Initialize database schema
python init_db.py

# Run static phase only (fastest test)
python main.py --start-date 2020-01-01 --end-date 2020-12-31 --skip-dynamic --skip-features --skip-modeling
```

### Expected Output

You should see:
```
=============================================================
   ORCHESTRATING CEWP PIPELINE
   Window: 2020-01-01 -> 2020-12-31
=============================================================
▶ STARTING: PHASE 1: STATIC INGESTION
✔ COMPLETED: PHASE 1: STATIC INGESTION (45.2s)
✅ PIPELINE EXECUTION SUCCESSFUL
```

**Next Steps:** See [docs/INSTALL.md](docs/INSTALL.md) for complete setup including Google Earth Engine authentication and data source configuration.

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [INSTALL.md](docs/INSTALL.md) | Full installation guide (Python, DB, GEE, API keys, manual downloads) |
| [USAGE.md](docs/USAGE.md) | CLI examples and pipeline workflows |
| [DATABASE_SETUP.md](docs/DATABASE_SETUP.md) | PostgreSQL schema and performance tuning |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker production deployment |
| [CEWP_Thesis_Overview.pdf](docs/CEWP_Thesis_Overview.pdf) | Complete thesis methodology |
| [CEWP_Data_Source_Audit.pdf](docs/CEWP_Data_Source_Audit.pdf) | Data source specifications |

---

## 🧭 What This Repo Does

- Builds an **H3 grid** (3,407 cells at resolution 5) + static geography layers (terrain, rivers, roads, settlements, mines)
- Ingests dynamic time-series data from **21+ data sources** (ACLED, GDELT, IODA, WorldPop, GEE, FEWS NET, Yahoo Finance)
- Extracts **NLP features** from ACLED event narratives and CrisisWatch monthly reports
- Runs **master feature engineering** on a **14-day temporal spine** (anomalies, shocks, decays, spatial diffusion)
- Trains a **Two-Stage Hurdle Ensemble** (9 thematic sub-models + meta-learners)
- Produces **calibrated probabilities** with **uncertainty quantification** via BCCP

---

## 🏗️ Pipeline Architecture

Orchestrated by `main.py`:

| Phase | Description | Key Modules |
| --- | --- | --- |
| **1. Static Ingestion** | Generate H3 grid and process invariant geography (terrain, rivers, roads, settlements, mines). | `create_h3_grid`, `fetch_dem`, `fetch_rivers`, `fetch_mines` |
| **2. Dynamic Ingestion** | Fetch time-series data from APIs (ACLED, GDELT, IODA, WorldPop, GEE, FEWS NET, Yahoo Finance). | `fetch_acled`, `fetch_gee_server_side`, `fetch_food_security`, `fetch_ioda` |
| **3. NLP Processing** | Extract semantic themes and conflict drivers from ACLED narratives. | `process_acled_hybrid.py` |
| **4. Feature Engineering** | 14-day spine, climatological anomalies, price shocks, conflict decay, spatial diffusion. | `feature_engineering.py` |
| **5. Modeling** | Build ABT + train Two-Stage Hurdle Ensemble (classifier + regressor). | `build_feature_matrix`, `train_models` |
| **6. Calibration** | Sigmoid (Platt) calibration + BCCP prediction intervals. Isotonic remains optional. | `calibrate_models` |
| **7. Inference** | Generate risk forecasts with uncertainty quantification. | `generate_predictions` |

---

## 📁 Project Structure

```text
.
├── configs/                 # YAML config files (pipeline control panel)
│   ├── data.yaml            # data sources, URLs, date windows
│   ├── features.yaml        # feature registry (111 features), lags, decay rates
│   └── models.yaml          # hyperparameters, target horizons, calibration settings
├── docs/                    # Documentation
│   ├── INSTALL.md           # Full installation guide
│   ├── USAGE.md             # CLI examples and workflows
│   ├── DATABASE_SETUP.md    # PostgreSQL configuration
│   ├── DEPLOYMENT.md        # Docker deployment
│   └── *.pdf                # Thesis documents
├── data/                    # Data storage (ignored by git)
│   ├── raw/                 # Manual downloads (ACLED.csv, EPR-2021.csv)
│   ├── processed/           # Intermediate outputs (parquet/geotiffs)
│   └── cache/               # API caches (GDELT, GeoEPR)
├── pipeline/
│   ├── ingestion/           # Fetch raw data
│   ├── processing/          # Feature engineering & cleaning
│   └── modeling/            # ABT building, training, calibration, prediction
├── main.py                  # Orchestrator
├── init_db.py               # DB init (extensions, H3 types)
└── requirements.txt         # Python dependencies
```

---

## 📊 Data Sources (21+)

| Category | Sources | Key Features |
| --- | --- | --- |
| **Environmental** | CHIRPS, ERA5, MODIS, VIIRS, JRC Water, Dynamic World | Precipitation, temperature, NDVI anomalies, Integrated NTL (Stability/Kinetic/Staleness), surface water, landcover |
| **Conflict & Events** | ACLED, GDELT | Event counts, fatalities, protest/riot indicators, media tone |
| **ACLED Hybrid NLP** | ACLED notes field | 8 semantic themes + 5 explicit drivers (semi-supervised) |
| **Socio-Political** | EPR, IOM DTM, IODA | Ethnic exclusion, displacement, internet outages |
| **Economic** | Yahoo Finance, WFP Markets | Gold/oil prices, local market prices, price shocks |
| **Infrastructure** | GRIP4, HydroRIVERS, IPIS, OSM | Distance to roads, rivers, mines, settlements |
| **Demographics** | WorldPop | Population count and density |
| **Temporal** | Generated | Seasonal features (month_sin, month_cos, is_dry_season) |

---

## 🧠 Model Architecture

### Two-Stage Hurdle Ensemble

**Stage 1:** 9 thematic sub-models (each with XGBoost/LightGBM base learners)  
**Stage 2:** Meta-learners aggregate predictions through stacking  
**Stage 3:** Sigmoid (Platt) calibration maps raw scores to calibrated probabilities (isotonic optional)  
**Stage 4:** BCCP provides prediction intervals with guaranteed coverage (fit on log-scale, served on count-scale)

**Theme grouping note:**  
- `conflict_history` uses the curated ACLED dataset (location-aware, structured peace/violence categories).  
- `news_ops` bundles GDELT media signals, CrisisWatch scores, and IODA outage indicators alongside ACLED hybrid drivers.

### Prediction Horizons

| Horizon | Steps | Use Case |
| --- | --- | --- |
| **14-day** | 1 step | Tactical response |
| **1-month** | 2 steps | Operational planning |
| **3-month** | 6 steps | Strategic allocation |

### Output

```
risk_score = calibrated_probability × expected_fatalities
```

---

## 🔄 Structural Break Handling

The pipeline explicitly tracks data availability shifts:

| Flag | Threshold | Purpose |
| --- | --- | --- |
| `is_worldpop_v1` | Pre-2015 | Distinguishes census-adjusted (V1) vs constrained (V2) population |
| `iom_data_available` | Pre-2015-01-31 | IOM DTM data coverage start |
| `econ_data_available` | Pre-2003-12-01 | Yahoo Finance coverage start |
| `ioda_data_available` | Pre-2022-02-01 | IODA internet monitoring start |
| `landcover_avail` | Pre-2015-06-27 | Dynamic World landcover start |
| `is_viirs_available` | Pre-2012-01-28 | VIIRS nighttime lights coverage start |
| `gdelt_data_available` | Pre-2015 | GDELT v2 coverage start |
| `food_data_available` | varies | FEWS NET food price coverage |

> **Note:** These flags are **essential for model training** (teaching XGBoost about data availability) but **add noise to diagnostic analysis** (VIF, correlation). The diagnostic modes automatically exclude them by default. See [Diagnostic Modes](#-diagnostic-modes) below.

---

## 🔬 Diagnostic Modes

The pipeline includes specialized diagnostic modes for feature quality analysis, with **automatic filtering of structural break flags** to avoid noise in VIF/correlation analysis.

### Quick Reference

```bash
# Run full diagnostics (VIF, correlation, stability) - excludes structural break flags by default
python main.py --run-diagnostics-only

# Run collinearity check only
python main.py --run-collinearity-only

# Include structural break flags in diagnostics (for comparison)
python main.py --run-diagnostics-only --include-structural-breaks

# Exclude additional columns
python main.py --run-diagnostics-only --diagnostic-exclude-cols "col1,col2,col3"

# List all structural break flags
python main.py --list-structural-breaks
```

### Prerequisites

Diagnostic modes require an **existing feature matrix**. Run the pipeline first:

```bash
# Build feature matrix (skip modeling for faster iteration)
python main.py --skip-modeling --skip-analysis

# Then run diagnostics
python main.py --run-diagnostics-only
```

### Diagnostic Outputs

Generated in `analysis/diagnostics/`:

| File | Description |
|------|-------------|
| `pairwise_dependence.csv` | Feature pairs with \|ρ\| ≥ 0.6 |
| `theme_redundancy.csv` | Per-theme VIF, condition number, PCA variance |
| `temporal_stability.csv` | Correlation stability across time periods |
| `interpretability_checks.csv` | Per-feature variance, entropy, max correlation |
| `excluded_columns.csv` | Columns excluded from diagnostics and why |
| `vif_analysis.csv` | Full VIF analysis (collinearity check) |
| `correlation_matrix.parquet` | Complete correlation matrix |
| `suggested_removals.json` | Features flagged for potential removal |

### Configuration

Structural break flags are configured in `configs/features.yaml`:

```yaml
diagnostics:
  structural_break_flags:
    - is_worldpop_v1
    - iom_data_available
    - econ_data_available
    # ... see features.yaml for full list
  
  thresholds:
    vif_warning: 5.0
    vif_severe: 10.0
    correlation_warning: 0.7
    correlation_severe: 0.9
```

See [docs/DIAGNOSTIC_FILTERING.md](docs/DIAGNOSTIC_FILTERING.md) for complete documentation.

---

## 📈 Feature Summary

**Total Features: 141 (unique columns pre-pruning)**

| Category | Count |
| --- | --- |
| Environmental | 29 |
| Conflict | 20 |
| ACLED Hybrid NLP | 13 |
| NLP & Narrative (v2.0) | 27 |
| Economic | 20 |
| Socio-Political | 13 |
| Infrastructure | 12 |
| Demographics | 4 |
| Temporal Context | 3 |

---

## 📉 Evaluation Metrics

Model performance is assessed on operational utility rather than raw accuracy (due to ~2% positive class rate):

| Metric | Description |
| --- | --- |
| **Top-10% Recall** | Primary operational metric — % of actual conflict captured if intervention targets top 10% highest-risk cells |
| **PR-AUC** | Discrimination capability under class imbalance |
| **Brier Score** | Calibration quality (lower is better) |
| **RMSE** | Intensity prediction accuracy (absolute error on counts) |
| **Mean Poisson Deviance** | Intensity fit on count scale; respects heteroscedasticity of conflict counts (lower is better) |
| **Coverage** | BCCP interval reliability |

---

## 🔬 Research Validation

The pipeline includes a dedicated diagnostic module to validate three core research questions:

| RQ | Objective | Validation Method |
|----|-----------|-------------------|
| **RQ1** | Spatial Granularity | SHAP gradient analysis (<5km) & Within-Admin Variance |
| **RQ2** | Operational Tempo | Precision-Recall comparison (14d vs 30d) & Time-to-Detection |
| **RQ3** | Multi-Modal Fusion | SHAP Interaction (Hard × Soft) & Ablation Study (F1 Gain) |

Run the validation suite:
```bash
python research_questions_diagnostic.py
```

---

## 🛑 Limitations

- **Reporting bias:** Relies on ACLED/GDELT, which depend on media coverage; remote areas may be under-represented
- **Temporal resolution:** 14-day windows are optimized for strategic planning but may miss rapid escalations
- **Proxy indicators:** Variables like nighttime lights are proxies for economic activity and can include sensor noise
- **Causal mechanisms:** Current features capture correlates, not causes, limiting policy interpretability

---

## ⚠️ Quotas & Limits

- **Google BigQuery:** GDELT ingestion uses BigQuery; free tier (1TB/month) applies
- **Google Earth Engine:** Requires a GEE-enabled Google Cloud Project
- **FEWS NET API:** Requires `FEWS_NET_TOKEN` for IPC and market price data

---

## ✅ Notes

- The `data/` directory is ignored by Git by design
- Logs should not be committed (see `.gitignore`)
- H3 indices are stored as `BIGINT` (int64) for compatibility

### Temporal Lag Handling

The pipeline distinguishes two independent lag mechanisms:

| Lag Type | Purpose | Applied At |
|----------|---------|------------|
| **Publication Lag** | Accounts for data release delays (when data becomes available) | Ingestion/storage |
| **Analytical Lag** | Prevents temporal leakage (model sees only prior periods) | Feature engineering |

**Publication lags** shift timestamps at ingestion: GEE +14 days, Food Prices +56 days, ACLED NLP +14 days, ACLED counts +14 days (post-merge).

**Analytical lags** use `LAG()`/`shift()` for features and `LEAD()` for targets downstream in feature engineering.

A feature can have both—e.g., GEE data has a 14-day publication lag at ingestion AND an analytical lag when used as a model feature.

---

## 📧 Contact

For inquiries regarding the CEWP methodology or deployment, please contact:
**Brenan Andre** - [brenan.andre23@gmail.com](mailto:brenan.andre23@gmail.com)

---

## 📜 License

[Specify license]
