# CEWP Installation Guide

Complete installation and setup instructions for the Conflict Early Warning Pipeline.

## Table of Contents
- [System Requirements](#system-requirements)
- [Python Environment Setup](#python-environment-setup)
- [Database Setup](#database-setup)
- [Google Earth Engine Setup](#google-earth-engine-setup)
- [Configuration Files](#configuration-files)
- [API Keys & Credentials](#api-keys--credentials)
- [Manual Data Downloads](#manual-data-downloads)
- [Admin Boundary Configuration](#admin-boundary-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS:** Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11 with WSL2
- **Python:** 3.10 or 3.11 (3.12 not yet supported)
- **RAM:** 16GB (32GB recommended for full pipeline)
- **Storage:** 50GB free space (20GB for data, 30GB for database)
- **PostgreSQL:** Version 13 or higher

### Required PostgreSQL Extensions
- PostGIS 3.0+
- H3 PostgreSQL (via [h3-pg](https://github.com/zachasme/h3-pg))
- H3-PostGIS bridge extension

### External Accounts Needed
- [Google Earth Engine](https://earthengine.google.com/) (free for research/education)
- [ACLED](https://acleddata.com/) API access (free tier available)
- [IOM DTM](https://dtm.iom.int/data/api) API access (request required)
- [FEWS NET](https://fdw.fews.net/) Data Warehouse access
- [Sentinel Hub](https://www.sentinel-hub.com/) (for Copernicus DEM)

---

## Python Environment Setup

### Step 1: Install Python 3.10/3.11

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

**macOS (via Homebrew):**
```bash
brew install python@3.10
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) or use [pyenv-win](https://github.com/pyenv-win/pyenv-win)

### Step 2: Create Virtual Environment

```bash
cd /path/to/CEWP-CAR
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Expected install time:** 5-10 minutes

### Step 4: Verify Installation

```bash
python -c "import geopandas, h3, earthengine, xgboost; print('✓ Core dependencies installed')"
```

---

## Database Setup

### Step 1: Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt install postgresql-13 postgresql-contrib-13 postgresql-13-postgis-3
sudo apt install postgresql-13-h3  # May need to build from source
```

**macOS:**
```bash
brew install postgresql postgis
```

**Windows:**
Use [EnterpriseDB installer](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) or Docker

### Step 2: Install H3 Extension

The H3 extension must be compiled and installed manually:

```bash
# Clone h3-pg repository
git clone https://github.com/zachasme/h3-pg.git
cd h3-pg

# Build and install
make
sudo make install

# Verify
psql -c "SELECT h3_get_version();"
```

### Step 3: Create Database

```bash
# Start PostgreSQL service
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE car_cewp;
CREATE USER cewp_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE car_cewp TO cewp_user;
\c car_cewp
CREATE EXTENSION postgis;
CREATE EXTENSION h3;
CREATE EXTENSION h3_postgis;
EOF
```

### Step 4: Configure Database Connection

Edit `.env` file:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=car_cewp
DB_USER=cewp_user
DB_PASS=your_secure_password
```

### Step 5: Initialize Schema

```bash
python init_db.py
```

**Expected output:**
```
--- STARTING DATABASE INITIALIZATION (Step 0) ---
Initializing extensions...
Extensions verified: postgis, h3, h3_postgis
✓ features_static is already compliant (BIGINT).
--- DATABASE INITIALIZATION COMPLETE ---
```

---

## Google Earth Engine Setup

### Step 1: Create GEE Account

1. Go to [https://earthengine.google.com/](https://earthengine.google.com/)
2. Sign up using your institutional email (for research access)
3. Wait for approval (usually 24-48 hours)

### Step 2: Install and Authenticate

```bash
# The earthengine-api is already in requirements.txt
# Authenticate with your Google account
earthengine authenticate

# Follow the browser prompt to authorize
# Copy the authorization code back to terminal
```

### Step 3: Test Authentication

```bash
python -c "import ee; ee.Initialize(); print('✓ GEE authenticated')"
```

### Step 4: Configure Service Account (Optional, for Production)

For automated/scheduled runs, use a service account:

1. Create service account in [Google Cloud Console](https://console.cloud.google.com/)
2. Download JSON key file
3. Save to `GoogleKeys/gee-service-account.json`
4. Authenticate:
   ```bash
   earthengine authenticate --service-account gee-service-account@project.iam.gserviceaccount.com --key-file GoogleKeys/gee-service-account.json
   ```

---

## Configuration Files

### Step 1: Copy Example Environment File

```bash
cp .env.example .env
```

### Step 2: Review Configuration YAMLs

The pipeline is controlled by three config files in `configs/`:

**configs/data.yaml:**
- Date ranges for data ingestion
- API endpoints and credentials
- File paths for raw data

**configs/features.yaml:**
- Feature engineering parameters
- H3 resolution (default: 5 ≈ 10km hexagons)
- Temporal window (default: 14 days)
- Registry of all 109 features to compute

**configs/models.yaml:**
- Model architectures (XGBoost, LightGBM)
- Forecast horizons (14d, 1m, 3m)
- Hyperparameters for training

**Action required:** Review `configs/data.yaml` and update:
```yaml
global_date_window:
  start_date: "2000-01-01"  # Adjust to your needs
  end_date: "2025-12-31"    # Adjust to your needs
```

---

## API Keys & Credentials

Create a `.env` file in the project root with the following credentials:

```bash
# --- Database ---
DB_HOST=localhost
DB_PORT=5432
DB_NAME=car_cewp
DB_USER=cewp_user
DB_PASS=your_password

# --- ACLED (Required) ---
# Register at: https://developer.acleddata.com/
ACLED_EMAIL=your_email
ACLED_KEY=your_api_key

# --- IOM DTM (Required for displacement data) ---
# Request access at: https://dtm.iom.int/data/api
IOM_PRIMARY_KEY=your_key

# --- IPC-CH API (Required for food security) ---
# Request access at: https://www.ipcinfo.org/ipc-country-analysis/api/
IPC_API_KEY=your_key

# --- Sentinel Hub (Required for Copernicus DEM) ---
# Register at: https://www.sentinel-hub.com/
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret

# --- FEWS NET (Required for market prices) ---
# Register at: https://fdw.fews.net/
FEWS_NET_EMAIL=your_email
FEWS_NET_PASSWORD=your_password

# --- Google Cloud (Required for BigQuery/Earth Engine) ---
# Path to your JSON service account key
GOOGLE_APPLICATION_CREDENTIALS="C:/path/to/gcp_key.json"
```

---

## Manual Data Downloads

Due to licensing restrictions and API limits, several datasets must be downloaded manually.

### Directory Structure

Ensure your project contains the following structure:
```
CEWP-CAR/
├── data/
│   ├── raw/
│   │   ├── acled.csv                 # Manual download
│   │   ├── EPR-2021.csv              # Manual download
│   │   ├── wbgCAFadmin1.geojson      # Manual download
│   │   └── wbgCAFadmin3.geojson      # Manual download (see Section 8)
```

### A. ACLED Data

1. Go to the [ACLED Export Tool](https://acleddata.com/data-export-tool/)
2. Set filters:
   - **Country:** Central African Republic
   - **Event Date:** 2000-01-01 to Present
3. Export the file and rename it to `acled.csv`
4. Place in `data/raw/`

### B. Ethnic Power Relations (EPR)

1. Go to the [ETH Zürich EPR Dataset](https://icr.ethz.ch/data/epr/)
2. Download the **Core Dataset (CSV)** - Version 2021
3. Rename to `EPR-2021.csv`
4. Place in `data/raw/`

> **Note:** The GeoEPR polygons are handled automatically by the pipeline's caching system, but the Core CSV is required manually.

---

## Admin Boundary Configuration

⚠️ **CRITICAL CONFIGURATION STEP**

The pipeline relies on a specific mapping between OCHA/HDX standards and World Bank standards for administrative boundaries.

### The Problem

| Standard | Admin 1 | Admin 2 |
|----------|---------|---------|
| World Bank | Region | Prefecture |
| OCHA/HDX | Prefecture | Sub-prefecture |

### The Solution

To resolve this mismatch, perform the following file renaming:

1. Download **OCHA Admin 1 (Prefectures)** boundaries for CAR (GeoJSON)
   - Save as: `wbgCAFadmin1.geojson`

2. Download **OCHA Admin 2 (Sub-prefectures)** boundaries for CAR (GeoJSON)
   - Save as: `wbgCAFadmin3.geojson`

> **Why?** The pipeline's `spatial_disaggregation.py` logic looks for an "Admin 3" file to find sub-prefectures. Saving the Admin 2 file with this name bridges the gap.

### Download Links

- [HDX CAR Admin Boundaries](https://data.humdata.org/dataset/cod-ab-caf)

---

## Verification

### Full Pipeline Test

Run a small test to verify everything works:

```bash
# Test with 1 year of data
python main.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --skip-analysis

# Expected runtime: 30-60 minutes
# Expected output: Feature matrix with ~130k rows
```

### Check Outputs

```bash
# Verify database tables
psql -d car_cewp -c "\dt car_cewp.*"

# Should show tables:
# - features_static
# - temporal_features
# - acled_events
# - environmental_features
# - (and ~15 more)

# Verify feature matrix
ls -lh data/processed/feature_matrix.parquet

# Should be ~50-200MB depending on date range
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'h3'"

**Solution:**
```bash
pip install h3==3.7.6
```

### Issue: "could not connect to server: Connection refused"

**Solution:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start if stopped
sudo systemctl start postgresql

# Verify port
psql -c "SHOW port;"
```

### Issue: "ERROR: extension 'h3' is not available"

**Solution:** H3 extension must be manually compiled. See [Database Setup Step 2](#step-2-install-h3-extension).

### Issue: "Earth Engine initialization failed"

**Solution:**
```bash
# Re-authenticate
earthengine authenticate --force

# Check credentials
python -c "import ee; ee.Initialize(project='your-project-id'); print('OK')"
```

### Issue: "Memory error during feature engineering"

**Solution:**
- Reduce date range in `configs/data.yaml`
- Increase system swap space
- Use a machine with more RAM (32GB recommended)

### Issue: WSL-specific database connection errors

**Solution:**
```bash
# In WSL, PostgreSQL on Windows requires host IP
# Find Windows host IP:
ip route show | awk '/default/ {print $3}'

# Update .env:
DB_HOST=172.x.x.x  # Use IP from above, not 'localhost'
```

---

## Quotas & Limits

- **Google BigQuery (GDELT):** The pipeline queries the public GDELT dataset. This falls under the BigQuery free tier (1TB/month), but repeated full-history ingestion may incur costs. The pipeline implements caching to minimize this.

- **Google Earth Engine:** Requires an enabled GEE account linked to your Google Cloud Project.

---

## Next Steps

After successful installation:

1. **Run full pipeline:** Execute `python main.py` for complete workflow
2. **Explore outputs:** Check `data/processed/` for feature matrices and predictions
3. **Customize:** Modify `configs/*.yaml` to experiment with different parameters
4. **Usage guide:** See [USAGE.md](USAGE.md) for CLI examples and workflows

For questions or issues not covered here, please [open an issue](https://github.com/YOUR_USERNAME/CEWP-CAR/issues).
