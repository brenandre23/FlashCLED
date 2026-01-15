# Conflict Early Warning & Prediction System (CEWP) 🇨🇫

A production-grade geospatial machine learning pipeline for forecasting sub-national conflict in the **Central African Republic (CAR)**.

The system ingests multi-source data (satellite, economic, political), engineers features on a hexagonal grid (**H3**), and predicts conflict probability and fatality magnitude using a **two-stage hurdle model**.

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

**Next Steps:** See [INSTALL.md](INSTALL.md) for complete setup including Google Earth Engine authentication and data source configuration.

---

## 📄 Key documents (PDF)

- **Thesis overview:** [`docs/CEWP_Thesis_Overview.pdf`](docs/CEWP_Thesis_Overview.pdf)
- **Data source audit:** [`docs/CEWP_Data_Source_Audit.pdf`](docs/CEWP_Data_Source_Audit.pdf)

---

## 🧭 What this repo does

- Builds an **H3 grid** + static geography layers (terrain, rivers, roads, settlements)
- Ingests dynamic time-series data (ACLED, GDELT, IODA, WorldPop, GEE, FEWS NET)
- Runs **master feature engineering** on a **14-day temporal spine** (anomalies, shocks, staleness/decay, spatial diffusion)
- Trains an **XGBoost ensemble OR LGB** (classifier + regressor)

---

## 🏗️ Pipeline architecture

Orchestrated by `main.py`:

| Phase | Description | Key modules |
| --- | --- | --- |
| **1. Static ingestion** | Generate H3 grid and process invariant geography (terrain, rivers, roads, settlements). | `create_h3_grid`, `fetch_dem`, `fetch_rivers` |
| **2. Dynamic ingestion** | Fetch time-series data from APIs (ACLED, GDELT, IODA, WorldPop, GEE, FEWS NET). | `fetch_acled`, `fetch_dynamic_event`, `ingest_food_security` |
| **3. Processing** | Master feature engineering: 14-day spine, climatological anomalies, price shocks, conflict decay, spatial diffusion. | `pipeline/processing/feature_engineering.py`, `calculate_epr_features` |
| **4. Modeling** | Build ABT + train the two-stage model (classifier + regressor). | `build_feature_matrix`, `train_models` |

---

## 📁 Project structure

```text
.
├── configs/                 # YAML config files (pipeline control panel)
│   ├── data.yaml            # data sources, URLs, date windows
│   ├── features.yaml        # feature registry, lags, decay rates
│   └── models.yaml          # hyperparameters + target horizons
├── docs/                    # PDFs + documentation (recommended)
│   ├── CEWP_Thesis_Overview.pdf
│   └── CEWP_Data_Source_Audit.pdf
├── data/                    # data storage (ignored by git)
│   ├── raw/                 # place manual downloads here
│   ├── processed/           # intermediate outputs (parquet/geotiffs)
│   └── cache/               # API caches (GDELT, GeoEPR)
├── pipeline/
│   ├── ingestion/           # fetch raw data
│   ├── processing/          # master feature engineering & cleaning
│   └── modeling/            # ABT building, training, prediction
├── main.py                  # orchestrator
├── init_db.py               # DB init (extensions, H3 types)
└── setup.py                 # python dependencies
```

---

## 📊 Modeling approach

This system uses a **two-stage hurdle model**:

1) **Binary classifier (XGBoost OR LGB)**  
   Predicts probability of conflict (e.g., `fatalities > 0`).

2) **Regressor (XGBoost OR LGB)**  
   Predicts fatalities conditional on conflict occurring.

**Output:**  
`risk_score = probability * expected_fatalities`

---

## 📉 Evaluation metrics

Model performance is assessed on operational utility rather than raw accuracy (due to class imbalance):

- **Top-10% Recall** (primary operational metric): % of actual conflict captured if intervention targets the top 10% highest-risk cells
- **PR-AUC**: discrimination capability under class imbalance
- **Brier score**: calibration / reliability of probability estimates

---

## 🛑 Limitations

- **Reporting bias:** relies on ACLED/GDELT, which depend on media/reporting coverage; remote areas may be under-represented
- **Temporal resolution:** 14-day windows are optimized for strategic planning but may miss rapid escalations
- **Proxy indicators:** variables like nighttime lights are proxies for economic activity and can include sensor noise (e.g., clouds), though the pipeline applies imputation and anomaly logic

---

## ⚠️ Quotas & limits

- **Google BigQuery:** GDELT ingestion uses BigQuery; free tier scanning limits may apply (caching reduces cost)
- **Google Earth Engine:** requires a GEE-enabled Google Cloud Project

---

## ✅ Notes

- The `data/` directory is ignored by Git by design.
- Logs should not be committed (see `.gitignore`).
