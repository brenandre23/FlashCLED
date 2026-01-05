# Data Configuration & Setup Guide 🗄️

This document outlines the strict data prerequisites required to run the Conflict Early Warning Pipeline (CEWP). Due to licensing restrictions and API limits, several steps must be completed manually.

1. Environment Variables (.env)

Create a .env file in the root directory. You will need API keys for the services listed below.

# --- Database ---
DB_HOST=localhost
DB_PORT=5433
DB_NAME=car_cewp
DB_USER=postgres
DB_PASS=your_password

# --- API Keys ---
# ACLED (Account required): [https://developer.acleddata.com/](https://developer.acleddata.com/)
ACLED_EMAIL=your_email
ACLED_PASSWORD=your_password

# IOM DTM (Request access): [https://dtm.iom.int/data/api](https://dtm.iom.int/data/api)
IOM_PRIMARY_KEY=your_key

# Sentinel Hub (Copernicus DEM): [https://www.sentinel-hub.com/](https://www.sentinel-hub.com/)
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret

# FEWS NET (Data Warehouse): [https://fdw.fews.net/](https://fdw.fews.net/)
FEWS_NET_EMAIL=your_email
FEWS_NET_PASSWORD=your_password

# Google Cloud (BigQuery/Earth Engine)
# Path to your JSON service account key
GOOGLE_APPLICATION_CREDENTIALS="C:/path/to/gcp_key.json"


# 2. Directory Structure

Ensure your project root contains a data/raw folder. The pipeline expects files to be placed here exactly as named.

FlashCLED/
├── data/
│   ├── raw/
│   │   ├── acled.csv                <-- Manual Download
│   │   ├── EPR-2021.csv             <-- Manual Download
│   │   ├── wbgCAFadmin1.geojson     <-- Manual Download
│   │   └── wbgCAFadmin3.geojson     <-- Manual Download (See Section 3)


# 3. Manual Downloads

A. ACLED Data

Go to the ACLED Export Tool.

# Filters:

# Country: Central African Republic
Event Date: 2000-01-01 to Present
Export the file and rename it to acled.csv.
Place in data/raw/.

# B. Ethnic Power Relations (EPR)
Go to the ETH Zürich EPR Dataset.
Download the Core Dataset (CSV) (Version 2021).
Rename to EPR-2021.csv.
Place in data/raw/.
Note: The GeoEPR polygons are handled automatically by the pipeline's caching system, but the Core CSV is required manually.

# C. Administrative Boundaries (The "Admin 3" Fix)

⚠️ CRITICAL CONFIGURATION STEP
The pipeline relies on a specific mapping between OCHA/HDX standards and World Bank standards.

World Bank Hierarchy: Admin 1 = Region, Admin 2 = Prefecture.
OCHA/HDX Hierarchy: Admin 1 = Prefecture, Admin 2 = Sub-prefecture.

To resolve this mismatch, you must perform the following file renaming hack:
Download OCHA Admin 1 (Prefectures) boundaries for CAR (GeoJSON).
Save as: wbgCAFadmin1.geojson
Download OCHA Admin 2 (Sub-prefectures) boundaries for CAR (GeoJSON).
Save as: wbgCAFadmin3.geojson

Why? The pipeline's legacy spatial_disaggregation.py logic looks for an "Admin 3" file to find sub-prefectures. Saving the Admin 2 file with this name bridges the gap.

# 4. Quotas & Limits

Google BigQuery (GDELT): The pipeline queries the public GDELT dataset. This falls under the BigQuery free tier (1TB/month), but repeated full-history ingestion may incur costs. The pipeline implements caching to minimize this.
Google Earth Engine: Requires an enabled GEE account linked to your Google Cloud Project.