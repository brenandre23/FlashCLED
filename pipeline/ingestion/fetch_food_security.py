"""
ingest_food_security.py
=======================
Consolidated Food Security Ingestion.

UPDATES:
- FIXED: Parsing logic now handles both (lat,lon) and [lat,lon] centroid formats.
- PUBLICATION LAG: Config-driven via data.yaml fews_net.publication_lag_steps.
  FEWS NET prices have ~45-60 day publication delay (4 steps × 14 days = 56 days).
"""

import sys
import os
import pandas as pd
import requests
import geopandas as gpd
from io import BytesIO
from sqlalchemy import text
from pathlib import Path

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, upload_to_postgis, load_configs, get_db_engine

SCHEMA = "car_cewp"
TABLE_PRICES = "food_security"
TABLE_LOCATIONS = "market_locations"


def get_publication_lag_days(data_cfg, features_cfg):
    """
    Calculate publication lag in days from config.
    
    Reads:
      - data.yaml:     fews_net.publication_lag_steps (default: 4)
      - features.yaml: temporal.step_days (default: 14)
    
    Returns:
      int: Number of days to shift stored dates forward
    """
    step_days = features_cfg.get('temporal', {}).get('step_days', 14)
    lag_steps = data_cfg.get('fews_net', {}).get('publication_lag_steps', 4)
    lag_days = lag_steps * step_days
    
    logger.info(f"Food Security publication lag: {lag_steps} steps × {step_days} days = {lag_days} days")
    return lag_days


# --- AUTHENTICATION HELPER ---
def get_fews_token():
    username = os.environ.get("FEWS_NET_EMAIL")
    password = os.environ.get("FEWS_NET_PASSWORD")
    if not username or not password:
        raise EnvironmentError("FEWS_NET_EMAIL and FEWS_NET_PASSWORD must be set in .env")

    auth_url = "https://fdw.fews.net/api-token-auth/"
    response = requests.post(auth_url, json={"username": username, "password": password}, timeout=30)
    response.raise_for_status()
    return response.json().get("token")


def ensure_tables_exist(engine):
    logger.info("Verifying Food Security tables...")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        
        # Locations
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_LOCATIONS} (
                market_id INTEGER PRIMARY KEY,
                market_name TEXT,
                latitude FLOAT, longitude FLOAT,
                admin1 TEXT, admin2 TEXT
            );
        """))
        
        # Prices
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_PRICES} (
                date DATE NOT NULL,
                market TEXT NOT NULL,
                commodity TEXT NOT NULL,
                indicator TEXT NOT NULL,
                value FLOAT,
                currency TEXT, unit TEXT, source TEXT,
                PRIMARY KEY (date, market, commodity, indicator)
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


def fetch_market_locations_remote(engine, data_config):
    url = data_config.get("fews_net", {}).get("markets_geojson_url")
    if not url: return {}

    logger.info(f"Fetching Market Locations from: {url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        if "features" not in data or not data["features"]: return {}
        
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        if not gdf.crs: gdf.set_crs(epsg=4326, inplace=True)
            
    except Exception as e:
        logger.error(f"  ❌ Failed to parse Market GeoJSON: {e}")
        return {}

    # Cleaning
    if 'id' not in gdf.columns and 'fnid' in gdf.columns: gdf = gdf.rename(columns={'fnid': 'id'})
    
    def safe_id(val):
        try: return int(''.join(filter(str.isdigit, str(val))))
        except: return abs(hash(str(val))) % (10 ** 8)
    
    gdf['market_id'] = gdf['id'].apply(safe_id)
    
    rename_map = {'name': 'market_name', 'geographic_unit_name': 'market_name', 'admin_1': 'admin1', 'admin_2': 'admin2'}
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    # Fix Geometries (ROBUST PARSING)
    if 'centroid' in gdf.columns:
        def parse_coord(x):
            try:
                # Remove both () and [] and split by comma
                clean = str(x).replace('(','').replace(')','').replace('[','').replace(']','')
                parts = clean.split(',')
                if len(parts) >= 2:
                    return [float(parts[0]), float(parts[1])]
            except:
                pass
            return [None, None]

        coords = gdf['centroid'].apply(parse_coord)
        gdf['longitude'] = coords.apply(lambda x: x[0])
        gdf['latitude'] = coords.apply(lambda x: x[1])
    else:
        gdf['longitude'] = gdf.geometry.centroid.x
        gdf['latitude'] = gdf.geometry.centroid.y

    # Fix Geography & Fusion
    mask_bangui = gdf['market_name'].astype(str).str.contains("Bangui|Combattant", case=False, na=False)
    gdf.loc[mask_bangui, 'market_name'] = "Bangui"

    # Build Lookup
    market_lookup = {}
    for _, row in gdf.iterrows():
        market_lookup[row['id']] = row['market_name']       # ID -> Name
        market_lookup[row['market_id']] = row['market_name'] # Int -> Name
        market_lookup[row['market_name']] = row['market_name'] # Name -> Name (Self)

    # Upload
    final_df = gdf[['market_id', 'market_name', 'latitude', 'longitude', 'admin1', 'admin2']].drop_duplicates('market_name')
    upload_to_postgis(engine, final_df, TABLE_LOCATIONS, SCHEMA, primary_keys=['market_id'])
    
    return market_lookup


def fetch_fews_market_prices(data_config, features_config, market_lookup, auth_token):
    """
    Fetch market prices and apply publication lag.
    
    FEWS NET prices are collected monthly but published with ~45-60 day delay.
    We shift dates forward by the configured lag to ensure operational validity.
    """
    if not auth_token: return pd.DataFrame()
    
    # Get publication lag from config
    lag_days = get_publication_lag_days(data_config, features_config)
    
    url = "https://fdw.fews.net/api/marketpricefacts/"
    start_date = data_config.get("global_date_window", {}).get("start_date", "2015-01-01")
    end_date = data_config.get("global_date_window", {}).get("end_date", "2025-12-31")
    
    params = {"country_code": "CF", "start_date": start_date, "end_date": end_date, "format": "json"}
    headers = {"Authorization": f"JWT {auth_token}", "Content-Type": "application/json"}
    
    data = fetch_all_pages(url, params, headers, label="Price Facts")
    records = []
    
    for p in data:
        m_id = p.get("market")
        m_name = market_lookup.get(m_id) or market_lookup.get(str(m_id)) or str(m_id)
        
        date = p.get("period_date") or p.get("start_date")
        val = p.get("value")
        
        # --- ROBUST PARSING LOGIC ---
        commodity = None
        
        # 1. Try explicit fields first
        for field in ["commodity", "commodity_name", "product", "product_name"]:
            if p.get(field):
                commodity = str(p.get(field)).strip()
                break
        
        # 2. Fallback to dataseries_name (Handling "Market, Product" format)
        if not commodity and p.get("dataseries_name"):
            ds_parts = [x.strip() for x in str(p.get("dataseries_name")).split(",")]
            
            if len(ds_parts) > 1 and (ds_parts[0] == m_name or ds_parts[0] in market_lookup.values()):
                commodity = ds_parts[1]
            else:
                commodity = ds_parts[0]

        if not commodity: commodity = "Unknown"
        
        if date and val is not None:
            # PUBLICATION LAG: Shift date forward by lag_days
            # Prices observed on 'date' become "available" at (date + lag_days)
            original_date = pd.to_datetime(date)
            lagged_date = original_date + pd.Timedelta(days=lag_days)
            
            records.append({
                "date": lagged_date.date(),
                "market": m_name,  
                "commodity": commodity,
                "indicator": "market_price",
                "value": float(val),
                "currency": "XAF", "unit": "kg", "source": "FEWS_NET_API"
            })
            
    if not records: return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # Average across fused markets (e.g. Bangui markets)
    df_fused = df.groupby(['date', 'market', 'commodity', 'indicator'], as_index=False).agg({
        'value': 'mean', 'currency': 'first', 'unit': 'first', 'source': 'first'
    })
    
    logger.info(f"Loaded {len(df_fused)} price records (with {lag_days}-day publication lag applied).")
    logger.info(f"Sample commodities: {df_fused['commodity'].unique()[:5]}")
    return df_fused


def run(configs, engine):
    logger.info("FOOD SECURITY INGESTION (With Publication Lag)")
    token = get_fews_token()
    ensure_tables_exist(engine)
    
    # Extract both data and features configs
    data_cfg = configs.get("data", configs)
    features_cfg = configs.get("features", {})
    
    market_lookup = fetch_market_locations_remote(engine, data_cfg)
    
    # Prices - now with publication lag
    df_prices = fetch_fews_market_prices(data_cfg, features_cfg, market_lookup, token)
    if not df_prices.empty:
        df_prices = df_prices.drop_duplicates(subset=["date", "market", "commodity"])
        upload_to_postgis(engine, df_prices, TABLE_PRICES, SCHEMA, ["date", "market", "commodity", "indicator"])
        logger.info(f"✓ Prices: {len(df_prices)} rows.")
    else:
        logger.warning("No Market Price data loaded.")


def main():
    try:
        cfgs = load_configs()
        if isinstance(cfgs, tuple):
            configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
        else:
            configs = cfgs
        engine = get_db_engine()
        run(configs, engine)
    finally:
        if 'engine' in locals(): engine.dispose()


if __name__ == "__main__":
    main()