"""
fetch_iom.py
================
Purpose: Fetch IOM DTM (Displacement Tracking Matrix) data for CAR.
Output: iom_displacement_data.csv in data/processed/

FIXES:
1. Robust Column Mapping: Handles case-insensitivity and multiple aliases 
   (fixing the spatial_disaggregation crash).
2. Schema Guarantee: Ensures output CSV always has required columns.
3. Fallback Client: Handles missing 'dtmapi' package gracefully.
"""
import os
import sys
import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import PATHS, logger, load_configs

# --- Configuration ---
DEFAULT_TARGET_COUNTRY = "Central African Republic"
OUTPUT_FILENAME = "iom_displacement_data.csv"
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
            
            # Handle API wrapping results in 'result' key or returning list directly
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

def setup_client(data_config):
    """Load credentials and initialize API client."""
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("IOM_PRIMARY_KEY")
    if not api_key:
        # Check secondary key
        api_key = os.getenv("IOM_SECONDARY_KEY")
        
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
    """
    Case-insensitive search for a column name from a list of candidates.
    Returns the actual column name in the DF, or None.
    """
    # Create a map of {lowercase_name: actual_name}
    col_map = {c.lower().strip(): c for c in df.columns}
    
    for cand in candidates:
        cand_clean = cand.lower().strip()
        if cand_clean in col_map:
            return col_map[cand_clean]
    return None

def standardize_columns(df, granularity):
    """
    Robustly maps API columns to the Project Schema using fuzzy matching.
    Target Cols: date, admin1, admin2, households, individuals, Granularity
    """
    if df.empty: return df
    
    # 1. Define Aliases for each target column
    mapping_targets = {
        'admin1': ['admin1Name', 'adm1_name', 'admin1', 'province', 'state'],
        'admin2': ['admin2Name', 'adm2_name', 'admin2', 'district', 'county'],
        'date': ['reportingDate', 'date', 'report_date', 'surveyDate'],
        'individuals': ['numPresentIdpInd', 'individuals', 'ind', 'population', 'number_idps'],
        'households': ['numPresentIdpHh', 'households', 'hh', 'families']
    }
    
    rename_dict = {}
    found_cols = []

    # 2. Resolve columns
    for target, aliases in mapping_targets.items():
        actual_col = resolve_column(df, aliases)
        if actual_col:
            rename_dict[actual_col] = target
            found_cols.append(target)
        else:
            # If mandatory columns are missing, log warning
            if target in ['admin1', 'individuals', 'date']:
                logger.warning(f"  Missing mandatory column '{target}'. Candidates checked: {aliases}")

    # 3. Rename
    df = df.rename(columns=rename_dict)
    
    # 4. Fill Missing / Defaults
    if 'admin2' not in df.columns:
        df['admin2'] = 'None'
    if 'households' not in df.columns:
        df['households'] = 0
    if 'individuals' not in df.columns:
        # Critical failure prevention: if no individuals found, create 0s so pipeline continues
        logger.error("  CRITICAL: 'individuals' count not found in IOM data. Filling with 0.")
        df['individuals'] = 0
        
    df['Granularity'] = granularity
    
    # 5. Format Date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors="coerce").dt.date
    else:
        # Fallback date if missing (rare)
        logger.warning("  No date column found. Using today.")
        df['date'] = pd.to_datetime('today').date()
        
    # 6. Final Selection
    keep_cols = ['date', 'admin1', 'admin2', 'individuals', 'households', 'Granularity']
    return df[keep_cols]

def fetch_data(client, country_name):
    """Fetch displacement data with granularity fallback."""
    logger.info(f"Fetching IOM DTM data for: {country_name}...")

    # Attempt 1: Admin 2 (High Granularity)
    logger.info("Attempting Admin 2 (Sub-prefecture)...")
    try:
        df = client.get_idp_admin2_data(CountryName=country_name)
    except Exception:
        # Handle variations in client methods
        try:
            df = client.get_idp_admin_data(CountryName=country_name, AdminLevel=2)
        except:
            df = pd.DataFrame()

    if not df.empty:
        logger.info(f" Success! Found {len(df):,} Admin 2 records.")
        return standardize_columns(df, 'Admin 2')

    # Attempt 2: Admin 1 (Low Granularity)
    logger.info("Admin 2 empty/failed. Attempting Admin 1 (Prefecture)...")
    try:
        df = client.get_idp_admin1_data(CountryName=country_name)
    except Exception:
        try:
             df = client.get_idp_admin_data(CountryName=country_name, AdminLevel=1)
        except:
            df = pd.DataFrame()
    
    if not df.empty:
        logger.info(f" Success! Found {len(df):,} Admin 1 records.")
        return standardize_columns(df, 'Admin 1')
            
    return pd.DataFrame()

def main():
    try:
        output_path = PATHS["data_proc"] / OUTPUT_FILENAME

        # Load Configs
        data_config, _, _ = load_configs()
        target_country = data_config.get("iom_dtm", {}).get("country_name", DEFAULT_TARGET_COUNTRY)

        # Setup & Fetch
        client = setup_client(data_config)
        df = fetch_data(client, target_country)
        
        # Save
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f" Data saved to: {output_path}")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Rows: {len(df)}")
        else:
            logger.warning("No data could be extracted from IOM API (All attempts empty).")
            
    except Exception as e:
        logger.error(f"IOM Fetch failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()