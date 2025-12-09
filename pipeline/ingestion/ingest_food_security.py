"""
ingest_food_security.py
=======================
Consolidated Food Security Ingestion.
UPDATED: Fetches Market Locations from FEWS NET GeoJSON API.
"""

import sys
import os
import pandas as pd
import requests
import geopandas as gpd
from sqlalchemy import text
from pathlib import Path

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, upload_to_postgis, load_configs, get_db_engine

SCHEMA = "car_cewp"
TABLE_PRICES = "food_security"
TABLE_IPC = "ipc_phases"
TABLE_LOCATIONS = "market_locations"

# --- AUTHENTICATION HELPER ---
def get_fews_token():
    username = os.getenv("FEWS_NET_EMAIL")
    password = os.getenv("FEWS_NET_PASSWORD")

    if not username or not password:
        logger.warning("❌ Missing FEWS_NET_EMAIL or FEWS_NET_PASSWORD in .env.")
        return None

    auth_url = "https://fdw.fews.net/api-token-auth/"
    payload = {"username": username, "password": password}

    try:
        response = requests.post(auth_url, json=payload, timeout=30)
        response.raise_for_status()
        token = response.json().get("token")
        if token:
            logger.info("✓ Authentication successful.")
            return token
        return None
    except Exception as e:
        logger.error(f"❌ FEWS NET Authentication failed: {e}")
        return None

def ensure_tables_exist(engine):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE_LOCATIONS};"))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE_PRICES};"))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE_IPC};"))

        conn.execute(text(f"""
            CREATE TABLE {SCHEMA}.{TABLE_LOCATIONS} (
                market_id INTEGER PRIMARY KEY,
                market_name TEXT,
                latitude FLOAT,
                longitude FLOAT,
                admin1 TEXT,
                admin2 TEXT
            );
        """))

        conn.execute(text(f"""
            CREATE TABLE {SCHEMA}.{TABLE_PRICES} (
                date DATE NOT NULL,
                market TEXT NOT NULL,
                commodity TEXT NOT NULL,
                indicator TEXT NOT NULL,
                value FLOAT,
                currency TEXT,
                unit TEXT,
                source TEXT,
                PRIMARY KEY (date, market, commodity, indicator)
            );
        """))

        conn.execute(text(f"""
            CREATE TABLE {SCHEMA}.{TABLE_IPC} (
                date DATE NOT NULL,
                admin1 TEXT,
                admin2 TEXT,
                phase INTEGER,
                population INTEGER,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, admin1, admin2)
            );
        """))

def fetch_all_pages(url, params, headers, label="Data"):
    all_results = []
    while url:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            if r.status_code != 200:
                logger.warning(f"  {label} endpoint returned {r.status_code}")
                break
            
            data = r.json()
            if isinstance(data, dict) and "results" in data:
                all_results.extend(data["results"])
                url = data.get("next")
                params = None 
            else:
                if isinstance(data, list): all_results.extend(data)
                elif isinstance(data, dict): all_results.append(data)
                url = None
        except Exception as e:
            logger.error(f"  Error fetching {label}: {e}")
            break
    return all_results

# --- NEW: REMOTE GEOJSON FETCH ---
def fetch_market_locations_remote(engine, data_config):
    """
    Fetches market locations from the FEWS NET GeoJSON endpoint defined in data.yaml.
    """
    url = data_config.get("fews_net", {}).get("markets_geojson_url")
    
    if not url:
        logger.error("  ❌ Missing 'markets_geojson_url' in data.yaml (fews_net section).")
        return {}

    logger.info(f"Fetching Market Locations from: {url}")
    
    try:
        # Fetch GeoJSON directly into GeoPandas
        gdf = gpd.read_file(url)
        
        if gdf.empty:
            logger.warning("  GeoJSON returned no features.")
            return {}

        # Parse Geometry (Centroids)
        # GeoJSON is usually Polygon or Point. If Polygon, get centroid.
        gdf['centroid'] = gdf.geometry.centroid
        gdf['longitude'] = gdf['centroid'].x
        gdf['latitude'] = gdf['centroid'].y
        
        # Map Columns (Adjust these keys based on the API response structure)
        # FEWS NET GeoJSON usually has 'name', 'id' (or 'fnid'), 'admin_1', etc.
        # Check available columns if this fails.
        
        # Common FEWS NET keys: 'name', 'geographic_unit_name', 'fnid'
        # We try to find the best match for ID and Name
        
        if 'id' not in gdf.columns and 'fnid' in gdf.columns:
            gdf = gdf.rename(columns={'fnid': 'id'})
            
        # Normalize columns
        rename_map = {
            'name': 'market_name',
            'admin_1': 'admin1',
            'admin_2': 'admin2'
        }
        
        # If standard keys missing, try 'geographic_unit_name'
        if 'name' not in gdf.columns and 'geographic_unit_name' in gdf.columns:
            rename_map['geographic_unit_name'] = 'market_name'
            
        gdf = gdf.rename(columns=rename_map)
        
        # Ensure ID is integer (FEWS IDs can be strings like "CF2001")
        # If they are strings, we map them to a hash or numeric ID for our DB
        # Our schema expects INTEGER market_id.
        # Strategy: Extract digits if mixed, or hash if string.
        
        def safe_id(val):
            try:
                # Try keeping just numbers
                return int(''.join(filter(str.isdigit, str(val))))
            except:
                return abs(hash(str(val))) % (10 ** 8) # Fallback hash

        gdf['market_id'] = gdf['id'].apply(safe_id)
        
        # Select & Clean
        target_cols = ['market_id', 'market_name', 'latitude', 'longitude', 'admin1', 'admin2']
        
        for col in target_cols:
            if col not in gdf.columns:
                gdf[col] = "Unknown"
        
        out_df = pd.DataFrame(gdf[target_cols])
        out_df = out_df.drop_duplicates(subset=['market_id'])
        
        # Upload
        upload_to_postgis(engine, out_df, TABLE_LOCATIONS, SCHEMA, primary_keys=['market_id'])
        logger.info(f"✓ Uploaded {len(out_df)} market locations from API.")
        
        return pd.Series(out_df.market_name.values, index=out_df.market_id).to_dict()

    except Exception as e:
        logger.error(f"Failed to fetch/parse Market GeoJSON: {e}")
        return {}

def fetch_fews_ipc(data_config, auth_token):
    if not auth_token: return pd.DataFrame()
    
    url = "https://fdw.fews.net/api/ipcphase/"
    start_date = data_config.get("global_date_window", {}).get("start_date", "2015-01-01")
    end_date = data_config.get("global_date_window", {}).get("end_date", "2025-12-31")
    
    params = {"country_code": "CF", "start_date": start_date, "end_date": end_date, "scenario": "CS", "format": "json"}
    headers = {"Authorization": f"JWT {auth_token}", "Content-Type": "application/json"}
    
    results = fetch_all_pages(url, params, headers, label="IPC Phases")
    records = []
    
    for item in results:
        date = item.get("reporting_date") or item.get("projection_start") or item.get("start_date")
        phase = item.get("value")
        adm1 = item.get("geographic_unit_name")
        if not adm1:
            geo_obj = item.get("geographic_unit")
            if isinstance(geo_obj, dict): adm1 = geo_obj.get("name")
        
        if date and phase is not None:
            records.append({
                "date": pd.to_datetime(date).date(),
                "admin1": str(adm1 or "Unknown").strip(),
                "admin2": "None",
                "phase": int(float(phase)),
                "population": 0,
                "source": "FEWS_NET_API"
            })
    return pd.DataFrame(records)

def fetch_fews_market_prices(data_config, market_lookup, auth_token):
    if not auth_token: return pd.DataFrame()
    
    url = "https://fdw.fews.net/api/marketpricefacts/"
    start_date = data_config.get("global_date_window", {}).get("start_date", "2015-01-01")
    end_date = data_config.get("global_date_window", {}).get("end_date", "2025-12-31")
    
    params = {"country_code": "CF", "start_date": start_date, "end_date": end_date, "format": "json"}
    headers = {"Authorization": f"JWT {auth_token}", "Content-Type": "application/json"}
    
    data = fetch_all_pages(url, params, headers, label="Price Facts")
    records = []
    
    for p in data:
        m_id = p.get("market")
        date = p.get("period_date") or p.get("start_date")
        val = p.get("value")
        
        # ID matching from lookup
        # API returns ID, we matched name in lookup
        # If ID logic differs, we fallback to name string match if possible?
        # Usually FEWS NET uses the same ID across endpoints.
        
        m_name = market_lookup.get(m_id, f"Unknown_{m_id}")
        ds_name = p.get("dataseries_name", "")
        commodity = ds_name.split(",")[0].strip() if ds_name else "Unknown"
        
        if date and val is not None:
            records.append({
                "date": pd.to_datetime(date).date(),
                "market": m_name,
                "commodity": commodity,
                "indicator": "market_price",
                "value": float(val),
                "currency": "XAF",
                "unit": "kg",
                "source": "FEWS_NET_API"
            })
    return pd.DataFrame(records)

# ---------------------------------------------------------
# RUN FUNCTION
# ---------------------------------------------------------
def run(configs, engine):
    """
    Main entry point called by orchestrator.
    """
    logger.info("=" * 60)
    logger.info("FOOD SECURITY INGESTION")
    logger.info("=" * 60)
    
    # 1. Auth
    token = get_fews_token()
    if not token:
        logger.error("Skipping Food Security (Auth Failed).")
        return

    # 2. Setup
    ensure_tables_exist(engine)
    
    # 3. Load Locations (NOW FROM API via DATA.YAML)
    market_lookup = fetch_market_locations_remote(engine, configs["data"])
    
    # 4. Fetch IPC
    df_ipc = fetch_fews_ipc(configs["data"], token)
    if not df_ipc.empty:
        df_ipc = df_ipc.sort_values('date').drop_duplicates(subset=["date", "admin1"], keep='last')
        upload_to_postgis(engine, df_ipc, TABLE_IPC, SCHEMA, ["date", "admin1", "admin2"])
        logger.info(f"✓ IPC: {len(df_ipc)} rows.")
    
    # 5. Fetch Prices
    df_prices = fetch_fews_market_prices(configs["data"], market_lookup, token)
    if not df_prices.empty:
        df_prices = df_prices.drop_duplicates(subset=["date", "market", "commodity"])
        upload_to_postgis(engine, df_prices, TABLE_PRICES, SCHEMA, ["date", "market", "commodity", "indicator"])
        logger.info(f"✓ Prices: {len(df_prices)} rows.")

def main():
    try:
        cfgs = load_configs()
        configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]} if isinstance(cfgs, tuple) else cfgs
        engine = get_db_engine()
        run(configs, engine)
    finally:
        if 'engine' in locals(): engine.dispose()

if __name__ == "__main__":
    main()