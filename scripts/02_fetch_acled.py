"""
03_fetch_acled.py
==================
Purpose: Fetch ACLED conflict data using OAuth2 (matching curl command logic).
Output: 'car_cewp.acled_events' table (with H3 indices).

Logic:
1. Authenticate via OAuth2 (Email/Pass).
2. Download data (Paginated).
3. Save to Parquet Cache.
4. Upload to PostGIS & Optimize (Spatial Indexing).
"""
import sys
import requests
import pandas as pd
import geopandas as gpd
from sqlalchemy import text
from pathlib import Path
from dotenv import load_dotenv

# --- Import Centralized Utilities ---
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))
import utils
from utils import logger, PATHS

SCHEMA = "car_cewp"
TABLE_NAME = "acled_events"
TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"

def setup_credentials():
    """Load ACLED credentials from .env"""
    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)
    
    email = utils.os.getenv("ACLED_EMAIL")
    pwd = utils.os.getenv("ACLED_PASSWORD")
    
    if not email or not pwd:
        raise ValueError(f"Missing ACLED credentials in {env_path}")
    return email, pwd

def get_oauth_token(email, password):
    """Authenticate using Password Grant."""
    logger.info("Authenticating with ACLED (OAuth2)...")
    
    payload = {
        "username": email,
        "password": password,
        "grant_type": "password",
        "client_id": "acled"
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    try:
        response = requests.post(TOKEN_URL, data=payload, headers=headers, timeout=30)
        response.raise_for_status()
        token = response.json().get('access_token')
        logger.info(" Token received.")
        return token
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        if 'response' in locals(): logger.error(f"Response: {response.text}")
        sys.exit(1)

def fetch_acled_data(data_config, token):
    """Fetch data with pagination and caching."""
    # Check Cache first
    cache_path = PATHS["data_raw"] / "acled_car.parquet"
    if cache_path.exists():
        logger.info(f" Found cached ACLED data at {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Fetching fresh ACLED data...")
    
    # Get params from YAML
    acled_cfg = data_config["acled"]
    country = acled_cfg["query_params"]["country"]
    # Note: API expects region as a single string or list, handled below
    
    all_results = []
    page = 1
    
    while True:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        params = {
            "country": country,
            "page": page,
            "limit": 5000
        }
        
        try:
            r = requests.get(READ_URL, params=params, headers=headers, timeout=60)
            
            if r.status_code == 403:
                logger.critical("⛔ 403 FORBIDDEN. Please accept Terms of Use at acleddata.com/login")
                sys.exit(1)
                
            r.raise_for_status()
            data = r.json()
            
            if not data.get("success"):
                logger.error(f"API Error: {data.get('error')}")
                break
                
            results = data.get("data", [])
            if not results: break
            
            all_results.extend(results)
            logger.info(f"  Page {page}: {len(results)} events")
            
            if len(results) < 5000: break
            page += 1
            
        except Exception as e:
            logger.error(f"Fetch failed on page {page}: {e}")
            sys.exit(1)
            
    df = pd.DataFrame(all_results)
    logger.info(f" Fetched {len(df):,} total events")
    
    # Save Cache
    df