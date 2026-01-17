# Conflict Early Warning & Prediction System (CEWP) 🇨🇫

A production-grade geospatial machine learning pipeline for forecasting sub-national conflict in the **Central African Republic (CAR)**.

The system ingests multi-source data (satellite, economic, political, NLP-derived), engineers **111 features** on a hexagonal grid (**H3 resolution 5, ~10km cells**), and predicts conflict probability and fatality magnitude using a **Two-Stage Hurdle Ensemble** with calibrated uncertainty quantification.

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
- Extracts **13 NLP features** from ACLED event narratives using semi-supervised semantic projection
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
| **Environmental** | CHIRPS, ERA5, MODIS, VIIRS, JRC Water, Dynamic World | Precipitation, temperature, NDVI anomalies, nighttime lights, surface water, landcover |
| **Conflict & Events** | ACLED, GDELT | Event counts, fatalities, protest/riot indicators, media tone |
| **ACLED Hybrid NLP** | ACLED notes field | 8 semantic themes + 5 explicit drivers (semi-supervised) |
| **Socio-Political** | EPR, IOM DTM, FEWS NET IPC, IODA | Ethnic exclusion, displacement, food security phases, internet outages |
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

---

## 📈 Feature Summary

**Total Features: 111** (45 raw + 66 transformed)

| Category | Count |
| --- | --- |
| Environmental | 26 |
| Conflict | 20 |
| ACLED Hybrid NLP | 13 |
| Economic | 20 |
| Socio-Political | 14 |
| Infrastructure | 12 |
| Demographics | 5 |
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

## 📜 License

[Specify license]
