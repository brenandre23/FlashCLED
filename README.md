# Conflict Early Warning & Prediction System (CEWP) 🇨🇫

A production-grade geospatial machine learning pipeline for forecasting sub-national conflict in the **Central African Republic (CAR)**.

The system ingests multi-source data (satellite, economic, political), engineers features on a hexagonal grid (**H3**), and predicts conflict probability and fatality magnitude using a **two-stage hurdle model**.

---

## 📄 Key documents (PDF)

- **Thesis overview:** [`docs/CEWP_Thesis_Overview.pdf`](docs/CEWP_Thesis_Overview.pdf)
- **Data source audit:** [`docs/CEWP_Data_Source_Audit.pdf`](docs/CEWP_Data_Source_Audit.pdf)

---

## 🧭 What this repo does

- Builds an **H3 grid** + static geography layers (terrain, rivers, roads, settlements)
- Ingests dynamic time-series data (ACLED, GDELT, IODA, WorldPop, GEE, FEWS NET)
- Runs **master feature engineering** on a **14-day temporal spine** (anomalies, shocks, staleness/decay, spatial diffusion)
- Trains an **XGBoost ensemble** (classifier + regressor)

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

## 🚀 Quick start

### 1) Prerequisites
- **Python 3.10+** (Conda recommended)
- **PostgreSQL 13+** with **PostGIS** enabled
- **GDAL** (required for terrain processing)  
  - Ubuntu: `sudo apt-get install gdal-bin`  
  - Windows: install via OSGeo4W or Conda (`conda install gdal`)

### 2) Install
```bash
git clone https://github.com/brenandre23/FlashCLED.git
cd FlashCLED
pip install -e .
```

### 3) Configure secrets
Create a `.env` file in the root fplder

```ini
DB_HOST=localhost
DB_PORT=5432
DB_NAME=car_cewp
DB_USER=postgres
DB_PASS=yourpassword

ACLED_EMAIL=your_email
# ... (see data instructions for full list)
```

### 4) Manual data prerequisites (required)
⚠️ Due to licensing/auth restrictions, a few files must be downloaded manually and placed in `data/raw/` before running.

Expected directory:
```text
data/
└── raw/
    ├── acled.csv
    ├── EPR-2021.csv
    ├── wbgCAFadmin1.geojson
    └── wbgCAFadmin3.geojson
```

#### Administrative boundaries (World Bank vs OCHA “Admin 3” naming fix)
**Key mismatch:**
- In the **World Bank / GADM-style** hierarchy: **Admin 1 = Region**, **Admin 2 = Prefecture**
- In **OCHA/HDX COD**: **Admin 1 = Prefecture**, **Admin 2 = Sub-prefecture**

**Compatibility hack used by this pipeline:**  
We take **OCHA Admin 2 (Sub-prefectures)** and store it under the filename the pipeline expects for “Admin 3”.

**Action:**
1) Download **OCHA Admin 1 (Prefectures)** → convert to GeoJSON if needed → save as **`wbgCAFadmin1.geojson`**  
2) Download **OCHA Admin 2 (Sub-prefectures)** → convert to GeoJSON if needed → save/rename as **`wbgCAFadmin3.geojson`**

> Note: If you download Shapefiles (`.shp`), convert them to GeoJSON before saving/renaming.

### 5) Run the full pipeline
```bash
python main.py
```

Optional: partial runs (debug)
```bash
# Skip static data (if grid/rivers already exist)
python main.py --skip-static

# Run ONLY modeling (assumes database is populated)
python main.py --skip-static --skip-dynamic --skip-features
```

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

1) **Binary classifier (XGBoost)**  
   Predicts probability of conflict (e.g., `fatalities > 0`).

2) **Regressor (XGBoost)**  
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
