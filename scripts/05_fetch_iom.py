"""
06_fetch_iom.py
================
Purpose: Fetch IOM DTM (Displacement Tracking Matrix) data for CAR.
Output: iom_displacement_data.csv in data/processed/

API Reference: https://displacement.iom.int/data/api
"""
import os
import sys
import pandas as pd
from dtmapi import DTMApi
from dotenv import load_dotenv

# --- Import Centralized Utilities ---
# Ensure we can import utils regardless of where this script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
from utils import PATHS, logger

# --- Configuration ---
TARGET_COUNTRY_NAME = "Central African Republic"
OUTPUT_FILENAME = "iom_displacement_data.csv"

def setup_api():
    """Load credentials and initialize DTM API client."""
    # Load .env from the Project Root (robust path)
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("IOM_PRIMARY_KEY")
    if not api_key:
        logger.error(f"IOM_PRIMARY_KEY not found in {env_path}")
        raise ValueError("Missing IOM API Key")
    
    return DTMApi(subscription_key=api_key)

def fetch_data(api):
    """
    Fetch displacement data with fallback logic (Admin 2 -> Admin 1).
    """
    logger.info(f"Fetching IOM DTM data for: {TARGET_COUNTRY_NAME}...")

    # Attempt 1: Admin 2 (Sub-prefecture level - Granular)
    try:
        logger.info("Attempting to fetch Admin 2 (Sub-prefecture) data...")
        df = api.get_idp_admin2_data(CountryName=TARGET_COUNTRY_NAME)
        
        if not df.empty:
            logger.info(f" Success! Found {len(df):,} Admin 2 records.")
            df['Granularity'] = 'Admin 2'
            return df
        else:
            logger.warning("No Admin 2 data found. Falling back...")

    except Exception as e:
        logger.warning(f"Admin 2 request failed: {e}. Falling back...")

    # Attempt 2: Admin 1 (Prefecture level - Coarse)
    try:
        logger.info("Attempting to fetch Admin 1 (Prefecture) data...")
        df = api.get_idp_admin1_data(CountryName=TARGET_COUNTRY_NAME)
        
        if not df.empty:
            logger.info(f" Success! Found {len(df):,} Admin 1 records.")
            df['Granularity'] = 'Admin 1'
            return df
            
    except Exception as e:
        logger.error(f"Admin 1 request also failed: {e}")
    
    return pd.DataFrame()

def main():
    try:
        # 1. Setup
        api = setup_api()
        
        # 2. Fetch
        df = fetch_data(api)
        
        # 3. Save
        if not df.empty:
            output_path = PATHS["data_proc"] / OUTPUT_FILENAME
            df.to_csv(output_path, index=False)
            logger.info(f" Data saved to: {output_path}")
            logger.info(f"  Columns: {list(df.columns)}")
        else:
            logger.warning("No data could be extracted from IOM API.")
            # We don't exit(1) here because we might want the pipeline to continue 
            # with other data, depending on strictness.
            
    except Exception as e:
        logger.error(f"IOM Fetch failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()