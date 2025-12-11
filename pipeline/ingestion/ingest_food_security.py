"""
ingest_food_security.py
=======================
Consolidated Food Security Ingestion.

FEATURES:
1. MARKET LOCATIONS (API): 
   - Fetches from FEWS NET GeoJSON API.
   - [FIX] Debugs raw response to catch empty returns.
   - [FIX] Parses string centroids "(lon, lat)".
   - [STRATEGY] Fuses "Bangui, Marché Combattant" into "Bangui" to create a single spatial node.
   - [PATCH] Fixes missing Admin 1 for Birao and Bossangoa.
2. MARKET PRICES (API):
   - Fetches from FEWS NET API.
   - [STRATEGY] Averages prices for fused markets (Bangui) to create a robust signal.
3. IPC PHASES (API): 
   - Fetches from FEWS NET API /ipcphase/.

OPTIMIZATIONS:
- Dynamic Authentication: Hits /api-token-auth/ to get a fresh token.
- Upsert Logic: Uses upload_to_postgis to prevent data loss.
- [AUDIT FIX] Idempotent Schema: Removed destructive DROP TABLE; added Schema Evolution.
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
TABLE_IPC = "ipc_phases"
TABLE_LOCATIONS = "market_locations"

# --- AUTHENTICATION HELPER ---
def get_fews_token():
    """
    Acquire a FEWS NET API authentication token.
    """
    username = os.environ.get("FEWS_NET_EMAIL")
    password = os.environ.get("FEWS_NET_PASSWORD")

    if not username or not password:
        error_msg = "FEWS_NET_EMAIL and FEWS_NET_PASSWORD must be set in .env."
        logger.error(f"❌ {error_msg}")
        raise EnvironmentError(error_msg)

    auth_url = "https://fdw.fews.net/api-token-auth/"
    payload = {"username": username, "password": password}

    try:
        response = requests.post(auth_url, json=payload, timeout=30)
        response.raise_for_status()
        token = response.json().get("token")
        if token:
            logger.info("✓ Authentication successful.")
            return token
        raise RuntimeError("Authentication returned no token.")
    except Exception as e:
        logger.error(f"❌ FEWS NET Authentication failed: {e}")
        raise

def ensure_tables_exist(engine):
    """
    Ensures tables exist without destroying data.
    Implements Schema Evolution via ALTER TABLE ADD COLUMN IF NOT EXISTS.
    """
    logger.info("Verifying Food Security tables (Idempotent Check)...")
    
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))

        # 1. Market Locations
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_LOCATIONS} (
                market_id INTEGER PRIMARY KEY,
                market_name TEXT,
                latitude FLOAT,
                longitude FLOAT,
                admin1 TEXT,
                admin2 TEXT
            );
        """))
        # Schema Evolution for Locations
        for col, dtype in [
            ("market_name", "TEXT"), 
            ("latitude", "FLOAT"), 
            ("longitude", "FLOAT"), 
            ("admin1", "TEXT"), 
            ("admin2", "TEXT")
        ]:
             conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE_LOCATIONS} ADD COLUMN IF NOT EXISTS {col} {dtype}"))

        # 2. Market Prices
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_PRICES} (
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
        # Schema Evolution for Prices
        for col, dtype in [
            ("value", "FLOAT"), 
            ("currency", "TEXT"), 
            ("unit", "TEXT"), 
            ("source", "TEXT")
        ]:
             conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE_PRICES} ADD COLUMN IF NOT EXISTS {col} {dtype}"))

        # 3. IPC Phases
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_IPC} (
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
        # Schema Evolution for IPC
        for col, dtype in [
            ("phase", "INTEGER"), 
            ("population", "INTEGER"), 
            ("source", "TEXT"), 
            ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        ]:
             conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE_IPC} ADD COLUMN IF NOT EXISTS {col} {dtype}"))

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

# --- CORE LOGIC: REMOTE GEOJSON FETCH ---
def fetch_market_locations_remote(engine, data_config):
    """
    Fetches market locations from FEWS NET GeoJSON endpoint.
    Performs critical cleaning:
    1. Parses string centroids.
    2. Patches missing Admin 1 (Birao, Bossangoa).
    3. Fuses 'Marché Combattant' into 'Bangui'.
    
    Returns:
        market_lookup (dict): Mapping of {original_market_id: final_market_name}
    """
    url = data_config.get("fews_net", {}).get("markets_geojson_url")
    if not url:
        logger.error("  ❌ Missing 'markets_geojson_url' in data.yaml.")
        return {}

    logger.info(f"Fetching Market Locations from: {url}")
    
    # 1. DEBUG: Raw Response Inspection
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw_text = r.text
        
        logger.info(f"  Raw Response Preview (first 500 chars):\n{raw_text[:500]}")
        
        if not raw_text.strip():
            logger.error("  ❌ API returned empty response body.")
            return {}
            
    except Exception as e:
        logger.error(f"  ❌ Failed to download raw data: {e}")
        return {}

    # 2. Parse GeoJSON
    try:
        # Load directly from the bytes content we just fetched
        gdf = gpd.read_file(BytesIO(r.content))
        
        if gdf.empty:
            logger.warning("  ⚠️ GeoJSON returned valid structure but 0 features.")
            return {}
            
        logger.info(f"  Raw features loaded: {len(gdf)}")

    except Exception as e:
        logger.error(f"  ❌ Failed to parse GeoJSON: {e}")
        return {}

    # 3. Data Cleaning & Normalization
    try:
        # A. ID Mapping
        # FEWS NET uses 'fnid' or 'id'. We need an integer ID for our DB.
        if 'id' not in gdf.columns and 'fnid' in gdf.columns:
            gdf = gdf.rename(columns={'fnid': 'id'})
            
        def safe_id(val):
            """Extract integers from string IDs (e.g. 'CF2001' -> 2001)."""
            try:
                digits = ''.join(filter(str.isdigit, str(val)))
                return int(digits) if digits else abs(hash(str(val))) % (10 ** 8)
            except:
                return 0
        
        gdf['market_id'] = gdf['id'].apply(safe_id)

        # B. Column Renaming
        rename_map = {
            'name': 'market_name',
            'geographic_unit_name': 'market_name',
            'admin_1': 'admin1',
            'admin_2': 'admin2'
        }
        # Apply rename only for columns that exist
        actual_rename = {k: v for k, v in rename_map.items() if k in gdf.columns}
        gdf = gdf.rename(columns=actual_rename)

        # C. Parse Centroid (String -> Lat/Lon)
        # Format is usually "(lon, lat)" e.g. "(20.66871, 5.76101)"
        if 'centroid' in gdf.columns:
            def parse_centroid(val):
                try:
                    # Remove parens and split
                    clean = str(val).replace('(', '').replace(')', '')
                    parts = [float(x) for x in clean.split(',')]
                    # API Format check: FEWS NET is typically (Lon, Lat)
                    # CAR Bounds: Lon ~14-27, Lat ~2-11.
                    # Example: 20.6 (Lon), 5.7 (Lat). So index 0 is Lon, 1 is Lat.
                    return parts[0], parts[1]
                except:
                    return None, None

            coords = gdf['centroid'].apply(parse_centroid)
            gdf['longitude'] = coords.apply(lambda x: x[0])
            gdf['latitude'] = coords.apply(lambda x: x[1])
        else:
            # Fallback to geometry if centroid column missing
            gdf['longitude'] = gdf.geometry.centroid.x
            gdf['latitude'] = gdf.geometry.centroid.y

        # D. Manual Patching (Geography)
        # Fix Birao and Bossangoa missing Admin 1
        mask_birao = gdf['market_name'].astype(str).str.contains("Birao", case=False, na=False)
        mask_bossangoa = gdf['market_name'].astype(str).str.contains("Bossangoa", case=False, na=False)
        
        # Apply patches only where admin1 is missing/unknown
        if 'admin1' not in gdf.columns: gdf['admin1'] = None
        
        gdf.loc[mask_birao & (gdf['admin1'].isna()), 'admin1'] = "Vakaga"
        gdf.loc[mask_bossangoa & (gdf['admin1'].isna()), 'admin1'] = "Ouham"
        
        # E. Market Fusion (Bangui)
        # Strategy: Rename "Bangui, Marché Combattant" -> "Bangui"
        # This allows us to group them later.
        mask_combattant = gdf['market_name'].astype(str).str.contains("Combattant", case=False, na=False)
        gdf.loc[mask_combattant, 'market_name'] = "Bangui"
        
        # Also ensure standard "Bangui" is clean
        mask_bangui = gdf['market_name'].astype(str).str.contains("Bangui", case=False, na=False)
        gdf.loc[mask_bangui, 'market_name'] = "Bangui"

        # 4. Generate Lookup Dict (CRITICAL STEP)
        # We need a map of {Original_ID -> Final_Name}
        # Even if we drop a row in the next step, we must remember its ID maps to "Bangui"
        market_lookup = pd.Series(
            gdf.market_name.values, 
            index=gdf.id  # Use original string ID for lookup from Price API
        ).to_dict()
        
        # Add integer ID lookup too, just in case
        market_lookup.update(pd.Series(
            gdf.market_name.values, 
            index=gdf.market_id
        ).to_dict())

        # 5. Deduplicate for Spatial Registry
        # We only want ONE spatial point for "Bangui" in the database.
        # We prefer the one that isn't Combattant if possible (Central Market), or just the first one.
        # Since we renamed them all to "Bangui", duplicates now exist.
        
        out_df = gdf.drop_duplicates(subset=['market_name'], keep='first').copy()
        
        # Select Final Columns
        target_cols = ['market_id', 'market_name', 'latitude', 'longitude', 'admin1', 'admin2']
        for c in target_cols:
            if c not in out_df.columns: out_df[c] = "Unknown"
            
        final_df = out_df[target_cols]

        # 6. Upload
        upload_to_postgis(engine, final_df, TABLE_LOCATIONS, SCHEMA, primary_keys=['market_id'])
        logger.info(f"✓ Uploaded {len(final_df)} unique market locations.")
        
        return market_lookup

    except Exception as e:
        logger.error(f"  ❌ Data cleaning/parsing failed: {e}", exc_info=True)
        return {}

def fetch_fews_market_prices(data_config, market_lookup, auth_token):
    """
    Fetches prices and applies Market Fusion logic (Averaging).
    """
    if not auth_token: return pd.DataFrame()
    
    url = "https://fdw.fews.net/api/marketpricefacts/"
    start_date = data_config.get("global_date_window", {}).get("start_date", "2015-01-01")
    end_date = data_config.get("global_date_window", {}).get("end_date", "2025-12-31")
    
    params = {"country_code": "CF", "start_date": start_date, "end_date": end_date, "format": "json"}
    headers = {"Authorization": f"JWT {auth_token}", "Content-Type": "application/json"}
    
    data = fetch_all_pages(url, params, headers, label="Price Facts")
    records = []
    
    for p in data:
        # Original Market ID from API (e.g., 1082 or 'CF2001')
        m_id = p.get("market")
        
        # Resolve Name using Lookup (This maps Combattant ID -> "Bangui")
        m_name = market_lookup.get(m_id)
        if not m_name:
            # Try integer version
            try: m_name = market_lookup.get(int(m_id))
            except: pass
            
        if not m_name:
            m_name = f"Unknown_{m_id}"

        date = p.get("period_date") or p.get("start_date")
        val = p.get("value")
        ds_name = p.get("dataseries_name", "")
        commodity = ds_name.split(",")[0].strip() if ds_name else "Unknown"
        
        if date and val is not None:
            records.append({
                "date": pd.to_datetime(date).date(),
                "market": m_name,  # Unified Name
                "commodity": commodity,
                "indicator": "market_price",
                "value": float(val),
                "currency": "XAF",
                "unit": "kg",
                "source": "FEWS_NET_API"
            })
            
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # --- PRICE FUSION STRATEGY ---
    # Group by [Date, Market, Commodity] and AVERAGE the value.
    # This merges "Bangui Central" and "Bangui Combattant" prices into one signal.
    logger.info("Fusing duplicate market prices (Averaging strategy)...")
    
    df_fused = df.groupby(['date', 'market', 'commodity', 'indicator'], as_index=False).agg({
        'value': 'mean',          # <--- The fusion happens here
        'currency': 'first',
        'unit': 'first',
        'source': 'first'
    })
    
    return df_fused

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
    try:
        token = get_fews_token()
    except Exception:
        logger.error("Skipping Food Security (Auth Failed).")
        return

    # 2. Setup (Idempotent)
    ensure_tables_exist(engine)
    
    # 3. Load Locations (Now returns robust lookup)
    market_lookup = fetch_market_locations_remote(engine, configs["data"])
    
    # 4. Fetch IPC
    df_ipc = fetch_fews_ipc(configs["data"], token)
    if not df_ipc.empty:
        df_ipc = df_ipc.sort_values('date').drop_duplicates(subset=["date", "admin1"], keep='last')
        upload_to_postgis(engine, df_ipc, TABLE_IPC, SCHEMA, ["date", "admin1", "admin2"])
        logger.info(f"✓ IPC: {len(df_ipc)} rows.")
    
    # 5. Fetch Prices (Uses Lookup + Fusion)
    df_prices = fetch_fews_market_prices(configs["data"], market_lookup, token)
    
    if not df_prices.empty:
        # Final deduplication just in case
        df_prices = df_prices.drop_duplicates(subset=["date", "market", "commodity"])
        upload_to_postgis(engine, df_prices, TABLE_PRICES, SCHEMA, ["date", "market", "commodity", "indicator"])
        logger.info(f"✓ Prices: {len(df_prices)} rows (Fused).")
    else:
        logger.warning("No Market Price data loaded.")

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