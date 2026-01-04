"""
fetch_dynamic_event.py
=======================
GDELT + IODA Ingestion (Spatially Aware)

FIXES APPLIED:
1. DATE RANGE FIX: Now ignores 'test_year' during ingestion and strictly uses 
   'global_date_window' (2017-2025) to ensure full history is fetched.
2. Idempotency: Uses upload_to_postgis for UPSERTs.
3. Resilience: Retains BigQuery retry logic.
4. CACHE OPTIMIZATION: Smart incremental caching for GDELT (only fetches missing dates).
"""

import sys
import requests
import pandas as pd
import geopandas as gpd
import logging
import h3.api.basic_int as h3
from h3 import LatLngPoly
import unicodedata
import re
import json
import time
from sqlalchemy import text
from google.cloud import bigquery
from google.api_core import exceptions
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# --- Import Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import ensure_h3_int64, load_configs, get_db_engine, PATHS, logger, upload_to_postgis

logger = logging.getLogger(__name__)

SCHEMA = "car_cewp"
DYNAMIC_TABLE = "features_dynamic_daily"
GDELT_FIPS_MAP = {'CF': 'CT'}

# -----------------------------------------------------------------------------
# 1. SETUP & UTILS
# -----------------------------------------------------------------------------

def create_dynamic_table(engine):
    """Recreate table with BIGINT H3 index."""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{DYNAMIC_TABLE} (
                h3_index BIGINT NOT NULL,
                date DATE NOT NULL,
                variable TEXT NOT NULL,
                value FLOAT,
                PRIMARY KEY (h3_index, date, variable)
            );
            CREATE INDEX IF NOT EXISTS idx_dyn_date ON {SCHEMA}.{DYNAMIC_TABLE} (date);
            CREATE INDEX IF NOT EXISTS idx_dyn_var ON {SCHEMA}.{DYNAMIC_TABLE} (variable);
            CREATE INDEX IF NOT EXISTS idx_dyn_h3 ON {SCHEMA}.{DYNAMIC_TABLE} (h3_index);
        """))
    logger.info(f"✓ Table {SCHEMA}.{DYNAMIC_TABLE} verified")

def get_date_range(engine, config):
    """
    FIXED: Strictly uses global_date_window for ingestion.
    Ignores split/test_year because ingestion must cover the whole timeline.
    """
    data_config = config.get('data', {})
    
    # Defaults if config is missing
    default_start = '2017-01-01'
    default_end = datetime.now().strftime('%Y-%m-%d')

    global_start = data_config.get('global_date_window', {}).get('start_date', default_start)
    global_end = data_config.get('global_date_window', {}).get('end_date', default_end)
    
    start_dt = datetime.strptime(str(global_start), "%Y-%m-%d")
    end_dt = datetime.strptime(str(global_end), "%Y-%m-%d")

    return start_dt, end_dt

def normalize_name(name):
    """Normalize admin names for matching (strip accents, case, special chars)."""
    if not isinstance(name, str): return ""
    n = name.lower().strip()
    n = unicodedata.normalize('NFKD', n).encode('ASCII', 'ignore').decode('utf-8')
    n = re.sub(r'[^a-z0-9]', '', n) # Remove all non-alphanumeric
    return n

def save_gdelt_cache(df, cache_path):
    """Save GDELT data to cache for fallback use."""
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info(f"Saved GDELT cache: {cache_path} ({len(df):,} rows)")

def load_gdelt_cache(cache_path, start_date, end_date):
    """Load cached GDELT data for fallback."""
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            cached_data = df[mask].copy()
            if not cached_data.empty:
                logger.warning(f"⚠️  Using CACHED GDELT data: {len(cached_data):,} rows from {cache_path}")
                logger.warning("  Note: This may be outdated. Check BigQuery credentials and network.")
                return cached_data
        except Exception as e:
            logger.error(f"Failed to load GDELT cache: {e}")
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. BIGQUERY CLIENT WITH RETRY AND FALLBACK
# -----------------------------------------------------------------------------

def validate_bigquery_credentials():
    """Check if BigQuery credentials are available."""
    try:
        from google.auth import default
        credentials, project = default()
        if not project:
            logger.warning("⚠️  No Google Cloud project found. Using Application Default Credentials.")
        return True
    except Exception as e:
        logger.error(f"❌ BigQuery credentials validation failed: {e}")
        return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        exceptions.ServiceUnavailable,
        exceptions.InternalServerError,
        exceptions.TooManyRequests,
        exceptions.GatewayTimeout,
        ConnectionError,
        TimeoutError
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def create_bigquery_client_with_retry():
    """Create BigQuery client with retry for transient failures."""
    try:
        client = bigquery.Client()
        # Test the client with a simple query
        test_query = "SELECT 1 as test"
        client.query(test_query).result()
        logger.info("✓ BigQuery client initialized and tested successfully")
        return client
    except exceptions.Forbidden as e:
        logger.critical(f"❌ BigQuery permission denied: {e}")
        raise RuntimeError(f"BigQuery permissions insufficient: {e}")
    except exceptions.BadRequest as e:
        logger.critical(f"❌ BigQuery configuration error: {e}")
        raise RuntimeError(f"BigQuery configuration error: {e}")
    except Exception as e:
        logger.error(f"BigQuery client initialization failed: {e}")
        raise

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type((
        exceptions.ServiceUnavailable,
        exceptions.InternalServerError,
        exceptions.TooManyRequests,
        exceptions.GatewayTimeout
    ))
)
def execute_bigquery_query(client, sql, job_config):
    """Execute BigQuery query with retry logic."""
    logger.info(f"Executing BigQuery query: {len(sql)} chars")
    return client.query(sql, job_config=job_config).to_dataframe()

# -----------------------------------------------------------------------------
# 3. GDELT INGESTION WITH SMART INCREMENTAL CACHING
# -----------------------------------------------------------------------------

def fetch_gdelt_from_bigquery_incremental(config, start_date, end_date):
    """
    Fetch GDELT data from BigQuery for a specific date range.
    Wrapper around the original BigQuery logic for incremental fetching.
    """
    # Validate credentials before attempting
    if not validate_bigquery_credentials():
        logger.warning("⚠️  BigQuery credentials not available")
        return pd.DataFrame()
    
    # Try to create client
    try:
        client = create_bigquery_client_with_retry()
    except Exception as e:
        logger.error(f"❌ BigQuery client creation failed: {e}")
        return pd.DataFrame()
    
    # Prepare parameters
    data_config = config.get('data', {})
    features_config = config.get('features', {})
    iso3 = data_config.get('gdelt', {}).get('country_iso3', 'CF')
    bq_dataset = data_config.get('gdelt', {}).get('bq_dataset', 'gdelt-bq.gdeltv2.events')
    resolution = features_config.get('spatial', {}).get('h3_resolution', 5)
    
    fips_code = GDELT_FIPS_MAP.get(iso3, iso3)
    
    # Convert dates to integer format for BigQuery
    if isinstance(start_date, date):
        s_int = int(start_date.strftime("%Y%m%d"))
        e_int = int(end_date.strftime("%Y%m%d"))
    else:
        s_int = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
        e_int = int(pd.to_datetime(end_date).strftime("%Y%m%d"))
    
    logger.info(f"Querying BigQuery for dates: {s_int} to {e_int}")
    
    sql = f"""
        SELECT SQLDATE, Actor1Geo_Lat as lat, Actor1Geo_Long as lon, 
               EventRootCode, GoldsteinScale, NumMentions, AvgTone
        FROM `{bq_dataset}`
        WHERE (Actor1Geo_CountryCode = @fips OR ActionGeo_CountryCode = @fips)
          AND SQLDATE BETWEEN @s_int AND @e_int
          AND Actor1Geo_Lat BETWEEN -90 AND 90 
          AND Actor1Geo_Long BETWEEN -180 AND 180
        ORDER BY SQLDATE
        LIMIT 1000000  -- Safety limit
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("fips", "STRING", fips_code),
            bigquery.ScalarQueryParameter("s_int", "INT64", s_int),
            bigquery.ScalarQueryParameter("e_int", "INT64", e_int),
        ]
    )
    
    try:
        df = execute_bigquery_query(client, sql, job_config)
    except Exception as e:
        logger.error(f"❌ BigQuery query failed: {e}")
        return pd.DataFrame()
    
    if df.empty:
        logger.warning("BigQuery returned empty results")
        return pd.DataFrame()
    
    logger.info(f"  Raw GDELT Rows fetched: {len(df)}")
    
    # Process data with H3 Type Safety
    try:
        # Calculate H3 & Enforce Int64
        # We import here to ensure the library is available within the scope
        import h3.api.basic_int as h3_int
        
        def get_safe_h3(lat, lon):
            try:
                # Basic Int API returns integer (possibly unsigned in some versions)
                val = h3_int.latlng_to_cell(lat, lon, resolution)
                # Force signed int64 for PostgreSQL compatibility
                return ensure_h3_int64(val)
            except:
                return None

        # Apply safe conversion
        df['h3_index'] = [get_safe_h3(r.lat, r.lon) for r in df.itertuples()]
        
        # Drop invalid cells and cast
        df = df.dropna(subset=['h3_index'])
        df['h3_index'] = df['h3_index'].astype('int64')

        df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
        
        # Aggregate multiple metrics
        agg = df.groupby(['h3_index', 'date']).agg(
            gdelt_event_count=('EventRootCode', 'count'),
            gdelt_goldstein_mean=('GoldsteinScale', 'mean'),
            gdelt_mentions_total=('NumMentions', 'sum'),
            gdelt_avg_tone=('AvgTone', 'mean')
        ).reset_index()
        
        result = agg.melt(id_vars=['h3_index', 'date'], 
                          var_name='variable', 
                          value_name='value').dropna()
        
        logger.info(f"✓ GDELT processed: {len(result):,} feature rows")
        return result
        
    except Exception as e:
        logger.error(f"❌ GDELT data processing failed: {e}")
        return pd.DataFrame()

def fetch_gdelt_with_fallback(config, start_date, end_date):
    """
    Fetch GDELT data from BigQuery with cache-first incremental strategy.
    
    Strategy:
    1. Check for existing full cache file
    2. If cache exists and covers requested range, use it entirely
    3. If cache exists but is outdated, only fetch missing recent data
    4. Merge, deduplicate, and save updated cache
    5. Fall back to cache if BigQuery fails
    """
    logger.info("Fetching GDELT data (cache-first incremental)...")
    
    # Setup cache - use a single file for all data (easier for incremental updates)
    cache_dir = PATHS["cache"] / "gdelt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "gdelt_full_cache.parquet"
    if cache_file.exists():
        age_days = (datetime.now().date() - datetime.fromtimestamp(cache_file.stat().st_mtime).date()).days
        if age_days > 7:
            logger.warning(f"⚠️  GDELT cache {cache_file} is {age_days} days old; refreshing window if needed.")
    
    # Convert dates to datetime.date for consistent comparison
    start_date_date = start_date.date() if isinstance(start_date, datetime) else pd.to_datetime(start_date).date()
    end_date_date = end_date.date() if isinstance(end_date, datetime) else pd.to_datetime(end_date).date()
    
    # Step 1: Check for existing cache
    if cache_file.exists():
        try:
            logger.info(f"Loading existing GDELT cache: {cache_file}")
            df_cache = pd.read_parquet(cache_file)
            
            # Ensure date column is in proper format
            if 'date' in df_cache.columns:
                df_cache['date'] = pd.to_datetime(df_cache['date'])
                
                # Get the maximum date in cache
                if not df_cache.empty:
                    max_cache_date = df_cache['date'].max().date()
                    logger.info(f"Cache contains data up to: {max_cache_date}")
                    
                    # Step 2: Check if cache already covers requested range
                    if max_cache_date >= end_date_date:
                        logger.info(f"✓ Cache fully covers requested range ({start_date_date} to {end_date_date})")
                        
                        # Filter cache to requested date range
                        mask = (df_cache['date'].dt.date >= start_date_date) & \
                               (df_cache['date'].dt.date <= end_date_date)
                        df_filtered = df_cache[mask].copy()
                        
                        if not df_filtered.empty:
                            logger.info(f"✓ Using cached GDELT data: {len(df_filtered):,} rows")
                            return df_filtered
                        else:
                            logger.warning("Cache exists but empty for requested date range")
                    else:
                        # Step 3: Cache is outdated - calculate what we need to fetch
                        logger.info(f"Cache outdated. Missing data from {max_cache_date + timedelta(days=1)} to {end_date_date}")
                        
                        # Filter cache for existing data in requested range
                        mask = (df_cache['date'].dt.date >= start_date_date) & \
                               (df_cache['date'].dt.date <= end_date_date)
                        df_existing = df_cache[mask].copy()
                        
                        # Calculate fetch range (only missing dates)
                        fetch_start_date = max_cache_date + timedelta(days=1)
                        
                        # Don't fetch if start is after end
                        if fetch_start_date > end_date_date:
                            logger.info("No missing dates to fetch")
                            return df_existing
                        
                        # Fetch only missing data
                        logger.info(f"Fetching missing data from {fetch_start_date} to {end_date_date}")
                        df_new = fetch_gdelt_from_bigquery_incremental(
                            config, fetch_start_date, end_date_date
                        )
                        
                        if df_new.empty:
                            logger.warning("No new GDELT data fetched from BigQuery")
                            return df_existing
                        
                        # Step 4: Merge and deduplicate
                        logger.info("Merging cached and new data...")
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        
                        # Deduplicate (same primary keys as table)
                        before_dedup = len(df_combined)
                        df_combined = df_combined.drop_duplicates(
                            subset=['h3_index', 'date', 'variable'], 
                            keep='last'
                        )
                        after_dedup = len(df_combined)
                        
                        if before_dedup != after_dedup:
                            logger.info(f"Removed {before_dedup - after_dedup} duplicate rows")
                        
                        # Save updated cache
                        logger.info("Saving updated cache...")
                        df_combined.to_parquet(cache_file, index=False)
                        logger.info(f"✓ Updated cache saved: {len(df_combined):,} total rows")
                        
                        return df_combined
                else:
                    logger.warning("Cache file exists but is empty")
            else:
                logger.warning("Cache file exists but missing 'date' column")
                
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    # Step 5: No cache or cache error - fetch full range from BigQuery
    logger.info("No valid cache found. Fetching full date range from BigQuery...")
    df_full = fetch_gdelt_from_bigquery_incremental(config, start_date_date, end_date_date)
    
    if not df_full.empty:
        # Save initial cache
        logger.info(f"Saving initial cache with {len(df_full):,} rows...")
        df_full.to_parquet(cache_file, index=False)
        logger.info(f"✓ Initial cache saved: {cache_file}")
    
    return df_full

# -----------------------------------------------------------------------------
# 4. IODA INGESTION (SUBNATIONAL OPTIMIZED) - MINIMAL CHANGES
# -----------------------------------------------------------------------------

def get_admin1_h3_map(data_config, features_config, engine):
    """Creates a mapping: {Normalized_Admin_Name: [List of H3 Indices]}"""
    logger.info("  Building Admin 1 -> H3 Mapping...")
    resolution = features_config["spatial"]["h3_resolution"]
    
    admin_path = PATHS["root"] / data_config["admin_boundaries"]["admin1_path"]
    if not admin_path.exists():
        logger.warning("  Admin 1 Shapefile not found. Falling back to country-level only.")
        return {}
        
    gdf = gpd.read_file(admin_path)
    mapping = {}
    
    for _, row in gdf.iterrows():
        name = row.get('NAM_1') or row.get('admin1Name') or row.get('name')
        if not name: continue
        
        norm_name = normalize_name(name)
        
        try:
            poly = row.geometry
            if poly.geom_type == 'MultiPolygon':
                polys = list(poly.geoms)
            else:
                polys = [poly]
            
            cells = set()
            for p in polys:
                exterior = [(y, x) for x, y in p.exterior.coords]
                holes = [[(y, x) for x, y in i.coords] for i in p.interiors]
                cells.update(h3.polygon_to_cells(LatLngPoly(exterior, *holes), resolution))
            
            mapping[norm_name] = list(cells)
        except Exception:
            continue
            
    logger.info(f"  Mapped {len(mapping)} subnational regions.")
    return mapping

def fetch_ioda_events(entity_type, entity_code, start_ts, end_ts, max_retries=3):
    """Helper to call IODA API with retry logic."""
    url = 'https://api.ioda.inetintel.cc.gatech.edu/v2/outages/events'
    params = {
        "entityType": entity_type,
        "entityCode": entity_code,
        "from": start_ts,
        "until": end_ts
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data.get('data', []) if isinstance(data, dict) else data
            elif r.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"IODA rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"IODA API call failed: {e}")
            continue
    
    return []

def fetch_ioda_with_fallback(config, engine, start_date, end_date):
    """Fetch IODA Outages with Subnational Granularity and fallback."""
    logger.info("Fetching IODA Internet Outages (Subnational)...")
    
    data_config = config['data']
    features_config = config['features']
    country_code = data_config.get('commodities', {}).get('ioda_entity_code', 'CF')
    
    start_ts = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
    
    records = []
    
    # Load full grid
    full_grid_df = pd.read_sql(f"SELECT h3_index FROM {SCHEMA}.features_static", engine)
    if full_grid_df.empty:
        logger.warning("  Grid empty. Cannot map IODA.")
        return pd.DataFrame()
    full_h3_set = set(full_grid_df['h3_index'].astype(int).tolist())
    
    # Get regional mapping
    region_map = get_admin1_h3_map(data_config, features_config, engine)
    
    # Discover regions
    meta_url = "https://api.ioda.inetintel.cc.gatech.edu/v2/entities/query"
    regions_to_fetch = []
    
    try:
        mr = requests.get(meta_url, params={"entityType": "region", "relatedTo": f"country/{country_code}"}, timeout=30)
        if mr.status_code == 200:
            meta_data = mr.json().get('data', [])
            for item in meta_data:
                regions_to_fetch.append((item['code'], item['name']))
    except Exception as e:
        logger.warning(f"  IODA Metadata lookup failed: {e}. Using static region mapping.")

    logger.info(f"  Found {len(regions_to_fetch)} IODA regions to query.")

    # Country-level events
    country_events = fetch_ioda_events("country", country_code, start_ts, end_ts)
    logger.info(f"  Country-level events: {len(country_events)}")
    
    for evt in country_events:
        try:
            d = datetime.fromtimestamp(evt['start'], tz=timezone.utc).date()
            if start_date.date() <= d <= end_date.date():
                # Apply to EVERY cell in the country
                for cell in full_h3_set:
                    records.append({
                        'h3_index': cell,
                        'date': d,
                        'variable': 'ioda_outage_detected',
                        'value': 1.0
                    })
        except: continue

    # Region-level events
    for r_code, r_name in regions_to_fetch:
        norm_r_name = normalize_name(r_name)
        target_cells = region_map.get(norm_r_name)
        
        # Fuzzy matching fallback
        if not target_cells:
            for k, v in region_map.items():
                if k in norm_r_name or norm_r_name in k:
                    target_cells = v
                    break
        
        if not target_cells:
            continue
            
        events = fetch_ioda_events("region", r_code, start_ts, end_ts)
        if events:
            logger.info(f"    Region '{r_name}': {len(events)} events -> {len(target_cells)} cells")
            
        for evt in events:
            try:
                d = datetime.fromtimestamp(evt['start'], tz=timezone.utc).date()
                if start_date.date() <= d <= end_date.date():
                    for cell in target_cells:
                        if cell in full_h3_set:
                            records.append({
                                'h3_index': cell,
                                'date': d,
                                'variable': 'ioda_outage_detected',
                                'value': 1.0
                            })
            except: continue

    if records:
        result = pd.DataFrame(records)
        # Deduplicate
        result = result.drop_duplicates(subset=['h3_index', 'date', 'variable'])
        logger.info(f"✓ IODA processed: {len(result):,} outage rows")
        return result
    else:
        logger.warning("⚠️  No IODA events found")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 5. MAIN ORCHESTRATION WITH ERROR HANDLING
# -----------------------------------------------------------------------------

def run(configs_bundle, engine):
    logger.info("=" * 80)
    logger.info("DYNAMIC EVENT INGESTION (GDELT + IODA Subnational)")
    logger.info("=" * 80)
    
    try:
        create_dynamic_table(engine)
        start_date, end_date = get_date_range(engine, configs_bundle)
        
        # 1. GDELT with smart incremental caching
        logger.info("\n--- GDELT Ingestion ---")
        df_gdelt = fetch_gdelt_with_fallback(configs_bundle, start_date, end_date)
        
        if not df_gdelt.empty:
            # FIX: Use upload_to_postgis for idempotency (UPSERT)
            logger.info(f"Uploading {len(df_gdelt):,} GDELT rows...")
            upload_to_postgis(
                engine, 
                df_gdelt, 
                DYNAMIC_TABLE, 
                SCHEMA, 
                primary_keys=['h3_index', 'date', 'variable']
            )
            logger.info(f"✓ GDELT: Uploaded {len(df_gdelt):,} rows")
        else:
            logger.warning("⚠️  GDELT: No data available (including cache)")
        
        # 2. IODA
        logger.info("\n--- IODA Ingestion ---")
        df_ioda = fetch_ioda_with_fallback(configs_bundle, engine, start_date, end_date)
        
        if not df_ioda.empty:
            # FIX: Use upload_to_postgis for idempotency (UPSERT)
            logger.info(f"Uploading {len(df_ioda):,} IODA rows...")
            upload_to_postgis(
                engine, 
                df_ioda, 
                DYNAMIC_TABLE, 
                SCHEMA, 
                primary_keys=['h3_index', 'date', 'variable']
            )
            logger.info(f"✓ IODA: Uploaded {len(df_ioda):,} rows")
        else:
            logger.warning("⚠️  IODA: No data available")
        
        # 3. Summary
        logger.info("\n" + "=" * 80)
        logger.info("DYNAMIC EVENT INGESTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
        logger.info(f"GDELT Rows: {len(df_gdelt):,}")
        logger.info(f"IODA Rows:  {len(df_ioda):,}")
        
        # Check if we have any data at all
        with engine.connect() as conn:
            total_rows = conn.execute(
                text(f"SELECT COUNT(*) FROM {SCHEMA}.{DYNAMIC_TABLE} WHERE date BETWEEN :start AND :end"),
                {"start": start_date.date(), "end": end_date.date()}
            ).scalar()
            
        if total_rows == 0:
            logger.warning("⚠️  WARNING: No dynamic event data loaded for this period!")
            logger.warning("    This may affect model performance. Check logs for errors.")
        else:
            logger.info(f"✓ Total dynamic features in database: {total_rows:,}")
        
    except Exception as e:
        logger.critical(f"❌ Dynamic event ingestion failed: {e}", exc_info=True)
        # Don't raise if we want pipeline to continue, but log critically
        # raise  # Uncomment if you want pipeline to stop on failure

if __name__ == '__main__':
    try:
        data_config, features_config, models_config = load_configs()
        engine = get_db_engine()
        configs = {'data': data_config, 'features': features_config, 'models': models_config}
        run(configs, engine)
    except Exception as e:
        logger.critical(f"❌ fetch_dynamic_event.py execution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.dispose()
