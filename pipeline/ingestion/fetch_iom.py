"""
fetch_iom.py
================
Purpose: Fetch IOM DTM (Displacement Tracking Matrix) data for CAR.
Output: 
  1. iom_displacement_data.csv in data/processed/
  2. iom_dtm_raw table in database

AUDIT FIXES:
1. DATA INTEGRITY: Uploads raw data to DB (backup/audit).
2. AGGREGATION: Sums metrics by Admin Area to prevent PK violations.
3. BUG FIX: Creates table schema before upload to prevent UndefinedTable error.
"""
import os
import sys
import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import text

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import PATHS, logger, load_configs, get_db_engine, upload_to_postgis

# --- Configuration ---
DEFAULT_TARGET_COUNTRY = "Central African Republic"
OUTPUT_FILENAME = "iom_displacement_data.csv"
TABLE_NAME = "iom_dtm_raw"
SCHEMA = "car_cewp"
DEFAULT_API_BASE_URL = "https://api.dtm.iom.int/v3"

class IOMClientManual:
    """
    Local implementation of the IOM DTM API client (Fallback).
    """
    def __init__(self, api_key, base_url=DEFAULT_API_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def _get_data(self, endpoint, params):
        url = f"{self.base_url}/{endpoint}"
        try:
            logger.info(f"Manual Request: {url}")
            response = requests.get(url, headers=self.headers, params=params, timeout=120)
            if response.status_code == 404:
                return pd.DataFrame()
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'result' in data:
                return pd.DataFrame(data['result'])
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Manual API Request failed: {e}")
            return pd.DataFrame()

    def get_idp_admin2_data(self, CountryName):
        return self._get_data("idp/admin2", {"CountryName": CountryName})

    def get_idp_admin1_data(self, CountryName):
        return self._get_data("idp/admin1", {"CountryName": CountryName})
    
    def get_idp_admin_data(self, CountryName, AdminLevel):
        return self._get_data(f"idp/admin{AdminLevel}", {"CountryName": CountryName})

def setup_client(data_config):
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("IOM_PRIMARY_KEY") or os.getenv("IOM_SECONDARY_KEY")
    if not api_key:
        logger.error(f"IOM_PRIMARY_KEY not found in {env_path}")
        raise ValueError("Missing IOM API Key")
    
    iom_cfg = data_config.get("iom_dtm", {})
    base_url = iom_cfg.get("api_endpoint", DEFAULT_API_BASE_URL)

    try:
        from dtmapi import DTMApi
        logger.info("Using official 'dtmapi' package.")
        return DTMApi(subscription_key=api_key) 
    except ImportError:
        logger.warning("'dtmapi' package not found. Using local fallback client.")
        return IOMClientManual(api_key=api_key, base_url=base_url)

def resolve_column(df, candidates):
    col_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        cand_clean = cand.lower().strip()
        if cand_clean in col_map:
            return col_map[cand_clean]
    return None

def standardize_columns(df, granularity):
    if df.empty: return df
    
    mapping_targets = {
        'admin1': ['admin1Name', 'adm1_name', 'admin1', 'province', 'state'],
        'admin2': ['admin2Name', 'adm2_name', 'admin2', 'district', 'county'],
        'date': ['reportingDate', 'date', 'report_date', 'surveyDate'],
        'individuals': ['numPresentIdpInd', 'individuals', 'ind', 'population', 'number_idps'],
        'households': ['numPresentIdpHh', 'households', 'hh', 'families']
    }
    
    rename_dict = {}
    for target, aliases in mapping_targets.items():
        actual_col = resolve_column(df, aliases)
        if actual_col:
            rename_dict[actual_col] = target
        else:
            if target in ['admin1', 'individuals', 'date']:
                logger.warning(f"  Missing mandatory column '{target}'.")

    df = df.rename(columns=rename_dict)
    
    if 'admin2' not in df.columns: df['admin2'] = 'None'
    if 'households' not in df.columns: df['households'] = 0
    if 'individuals' not in df.columns: df['individuals'] = 0
        
    df['Granularity'] = granularity
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors="coerce").dt.strftime('%Y-%m-%d')
    else:
        df['date'] = pd.to_datetime('today').strftime('%Y-%m-%d')
        
    keep_cols = ['date', 'admin1', 'admin2', 'individuals', 'households', 'Granularity']
    return df[keep_cols]

def fetch_data(client, country_name):
    logger.info(f"Fetching IOM DTM data for: {country_name}...")

    # Attempt 1: Admin 2
    try:
        df = client.get_idp_admin2_data(CountryName=country_name)
    except Exception:
        try: df = client.get_idp_admin_data(CountryName=country_name, AdminLevel=2)
        except: df = pd.DataFrame()

    if not df.empty:
        logger.info(f" Success! Found {len(df):,} Admin 2 records.")
        return standardize_columns(df, 'Admin 2')

    # Attempt 2: Admin 1
    logger.info("Admin 2 empty/failed. Attempting Admin 1...")
    try:
        df = client.get_idp_admin1_data(CountryName=country_name)
    except Exception:
        try: df = client.get_idp_admin_data(CountryName=country_name, AdminLevel=1)
        except: df = pd.DataFrame()
    
    if not df.empty:
        logger.info(f" Success! Found {len(df):,} Admin 1 records.")
        return standardize_columns(df, 'Admin 1')
            
    return pd.DataFrame()

def ensure_raw_table_exists(engine):
    logger.info(f"Verifying schema for {SCHEMA}.{TABLE_NAME}...")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                date DATE NOT NULL,
                admin1 TEXT NOT NULL,
                admin2 TEXT NOT NULL,
                individuals INTEGER,
                households INTEGER,
                "Granularity" TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, admin1, admin2, "Granularity")
            );
        """))

def main():
    try:
        output_path = PATHS["data_proc"] / OUTPUT_FILENAME
        data_config, _, _ = load_configs()
        target_country = data_config.get("iom_dtm", {}).get("country_name", DEFAULT_TARGET_COUNTRY)
        engine = get_db_engine()

        client = setup_client(data_config)
        df = fetch_data(client, target_country)
        
        if not df.empty:
            # 1. Deduplicate / Aggregate (The FIX)
            logger.info("Aggregating duplicates by Admin boundary...")
            df_agg = df.groupby(
                ['date', 'admin1', 'admin2', 'Granularity'], 
                as_index=False
            )[['individuals', 'households']].sum()
            
            # 2. Save CSV
            df_agg.to_csv(output_path, index=False)
            logger.info(f" Data saved to: {output_path}")
            
            # 3. DB Upload
            ensure_raw_table_exists(engine)
            logger.info(f"Uploading IOM data to {SCHEMA}.{TABLE_NAME}...")
            
            upload_to_postgis(
                engine, 
                df_agg, 
                TABLE_NAME, 
                SCHEMA, 
                primary_keys=["date", "admin1", "admin2", "Granularity"] 
            )
            logger.info(f"  Rows: {len(df_agg)}")
        else:
            logger.warning("No data extracted from IOM API.")
            
    except Exception as e:
        logger.error(f"IOM Fetch failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()