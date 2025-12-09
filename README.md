# Conflict Early Warning & Prediction System (CEWP) 🇨🇫

A production-grade geospatial machine learning pipeline for forecasting sub-national conflict in the Central African Republic (CAR). The system ingests multi-source data (satellite, economic, political), engineers features on a hexagonal grid (H3), and predicts conflict probability and fatality magnitude using a Two-Stage Hurdle Model.

## 🏗️ Architecture

The pipeline is modular and orchestrated by `main.py`.

| Phase | Description | Key Modules |
| :--- | :--- | :--- |
| **1. Static Ingestion** | Generates the H3 grid and processes invariant geography (Terrain, Rivers, Roads, Settlements). | `create_h3_grid`, `fetch_dem`, `fetch_rivers` |
| **2. Dynamic Ingestion** | Fetches time-series data from APIs (ACLED, GDELT, IODA, WorldPop, GEE, FEWS NET). | `fetch_acled`, `fetch_dynamic_event`, `ingest_food_security` |
| **3. Processing** | **Master Feature Engineering**: Creates a 14-day temporal spine, computes climatological anomalies, price shocks, and conflict decay vectors. | `feature_engineering` (Master), `calculate_epr_features` |
| **4. Modeling** | Builds the Analytical Base Table (ABT) and trains an XGBoost Ensemble (Classifier + Regressor). | `build_feature_matrix`, `train_models` |

## 🚀 Quick Start

### 1. Prerequisites
* **Python 3.10+** (Recommended via Conda)
* **PostgreSQL 13+** with **PostGIS** extension enabled.
* **System Libraries:** `gdal-bin` (required for terrain processing).
  * *Ubuntu:* `sudo apt-get install gdal-bin`
  * *Windows:* Install via OSGeo4W or Conda (`conda install gdal`).

### 2. Installation
Clone the repo and install it in "editable" mode. This automatically installs all Python dependencies listed in `setup.py`.

```bash
git clone [https://github.com/YourUsername/car_cewp.git](https://github.com/YourUsername/car_cewp.git)
cd car_cewp
pip install -e .

3. Configuration
Create a .env file in the root directory with your secrets (see .env.example or documentation below):

Ini, TOML

DB_HOST=localhost
DB_PORT=5432
DB_NAME=car_cewp
DB_USER=postgres
DB_PASS=yourpassword
ACLED_EMAIL=your_email
# ... (see Data Instructions for full list)
4. Data Setup
⚠️ Crucial Step: This pipeline relies on 4 manually downloaded files (ACLED, EPR, Admin Boundaries) that cannot be automated due to licensing. 👉 Read DATA_INSTRUCTIONS.md and place these files in data/raw/ before running.

5. Running the Pipeline
Run the full end-to-end process:

Bash

python main.py
Partial Runs (for debugging):

Bash

# Skip static data (if grid/rivers already exist)
python main.py --skip-static

# Run ONLY modeling (assumes database is populated)
python main.py --skip-static --skip-dynamic --skip-features
📁 Project Structure
Plaintext

.
├── configs/                 # YAML Configuration files (Control panel for the pipeline)
│   ├── data.yaml            # Data sources, URLs, and date windows
│   ├── features.yaml        # Feature registry, lag definitions, decay rates
│   └── models.yaml          # Model hyperparameters and target horizons
├── data/                    # Data storage (Ignored by Git)
│   ├── raw/                 # PLACE MANUAL DOWNLOADS HERE
│   ├── processed/           # Intermediate outputs (Parquet/GeoTiffs)
│   └── cache/               # API Caches (GDELT, GeoEPR)
├── pipeline/
│   ├── ingestion/           # Scripts to fetch raw data
│   ├── processing/          # Master Feature Engineering & Cleaning
│   └── modeling/            # Matrix building, Training, Prediction
├── main.py                  # Master Orchestrator
├── init_db.py               # Database initialization (Extensions, H3 types)
└── setup.py                 # Dependency definition
📊 Modeling Approach
The system uses a Two-Stage Hurdle Model:

Binary Classifier (XGBoost): Predicts probability of conflict (Any fatalities > 0).

Regressor (XGBoost): Predicts magnitude (Number of fatalities) conditional on conflict occurring.

Output: risk_score = probability * expected_fatalities.

⚠️ Quotas & Limits
Google BigQuery: The GDELT ingestion uses BigQuery. The free sandbox tier has a 1TB/month scanning limit. The script includes caching to minimize costs.

Earth Engine: Requires a GEE-enabled Google Cloud Project.


---

### **File 2: DATA_INSTRUCTIONS.md**
Place this in the root of your repository. It explicitly handles the "Manual Pre-requisites."

```markdown
# 📂 Data Retrieval Instructions

While the CEWP pipeline automates 90% of data collection (GDELT, WorldPop, GEE, FEWS NET APIs), **strict licensing and authentication requirements** prevent us from automating the following datasets.

You must manually download these **4 files** and place them in the `data/raw/` directory.

### 🛑 Required Directory Structure
Ensure your `data/` folder looks like this before running `main.py`:

```text
data/
└── raw/
    ├── acled.csv                <-- Manual Download
    ├── EPR-2021.csv             <-- Manual Download
    ├── wbgCAFadmin1.geojson     <-- Manual Download
    ├── wbgCAFadmin3.geojson     <-- Manual Download
    └── (Other files will be auto-generated here)
1. ACLED Conflict Data
File: acled.csv

Source: ACLED Data Export Tool

Instructions:

Register for a free account.

Select Central African Republic.

Select Date Range: 2000-01-01 to Present.

Click "Export".

Rename the downloaded file to acled.csv.

2. Ethnic Power Relations (EPR)
File: EPR-2021.csv

Source: ETH Zürich EPR Core Dataset

Instructions:

Go to the "Core" dataset section.

Download the CSV version (usually EPR-2021.csv or similar).

Ensure the filename is exactly EPR-2021.csv.

### 3. Administrative Boundaries (Admin 1 & 3)
**Required Files:** `wbgCAFadmin1.geojson` and `wbgCAFadmin3.geojson`

**Source:** [Humanitarian Data Exchange (HDX) - OCHA Central African Republic Administrative Boundaries](https://data.humdata.org/dataset/cod-ab-caf)

**Instructions:**
1.  **Visit the URL:** Go to https://data.humdata.org/dataset/cod-ab-caf
2.  **Download Admin 1 (Prefectures):**
    * Look for the file named `caf_admbnda_adm1_1m_gov_20201216_shp.zip` (or the `shp` zip for Admin 1).
    * Extract the `.shp` file.
    * Convert it to GeoJSON (using QGIS or Python).
    * **Rename the resulting file to:** `wbgCAFadmin1.geojson`
3.  **Download Admin 2 (Sub-prefectures):**
    * **CRITICAL NOTE:** The pipeline configuration expects the sub-prefecture file to be named "admin3". You must download the **Admin 2** dataset but rename it to **admin3**.
    * Look for the file named `caf_admbnda_adm2_1m_gov_20201216_shp.zip` (or the `shp` zip for Admin 2).
    * Extract the `.shp` file.
    * Convert it to GeoJSON.
    * **Rename the resulting file to:** `wbgCAFadmin3.geojson`
4.  **Place both files in:** `data/raw/`

✅ Automated Datasets (No Action Required)
The following are handled automatically by the pipeline if you have set up your .env keys correctly:

Market Locations & Prices: Fetched via FEWS NET API (GeoJSON).
Population: Downloaded from WorldPop.
Terrain (DEM): Fetched from Copernicus/Sentinel Hub.
Environmental: Fetched from Google Earth Engine (ERA5, CHIRPS).
Roads: Downloaded from GRIP4 (PBL).
Mines: Fetched from IPIS WFS server.