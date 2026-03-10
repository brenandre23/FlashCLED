"""
fetch_gdelt_dual.py
===================
GDELT CSV ingestion with dual-sensor processing (Events + GKG Themes).

UPDATES (2026-01-24):
- Uses centralized get_incremental_window helper
- Supports --no-incremental flag for full refresh
- Adds overlap buffer for late-arriving data
"""

import os
import sys
import re
import requests
import zipfile
import io
import pandas as pd
import h3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import multiprocessing
from datetime import timedelta

# -----------------------
# Project imports
# -----------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, upload_to_postgis, SCHEMA, load_configs, ensure_h3_int64, get_incremental_window

load_dotenv(ROOT_DIR / ".env")

# -----------------------
# Configuration
# -----------------------
H3_RES = int(os.getenv("GDELT_H3_RES", "5"))
COUNTRY_CODE = os.getenv("GDELT_COUNTRY_CODE", "CF").upper()
DEFAULT_START = "2015-02-18"  # GDELT v2 start
TARGET_TABLE = "features_dynamic_daily"
MASTER_LIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# Overlap buffer for late-arriving GDELT data
OVERLAP_BUFFER_DAYS = 7

# -----------------------
# Global Buffers & Session (Initialized per process)
# -----------------------
SESSION = None

def get_session():
    """Create a session with retry logic."""
    global SESSION
    if SESSION is None:
        SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=5)
        SESSION.mount('http://', adapter)
        SESSION.mount('https://', adapter)
    return SESSION

CAMEO_PREDATORY_CODES = ['10', '11', '17', '18', '19']

THEME_CLUSTERS = {
    "resource_predation": {"NATURAL_RESOURCES", "MINING", "SMUGGLING", "TAX_FNCACT", "EXTORTION", "BLACK_MARKET"},
    "displacement": {"REFUGEES", "DISPLACEMENT", "FAMINE", "FOOD_SECURITY"},
    "governance_breakdown": {"CORRUPTION", "UNGOVERNED", "FAILED_STATE", "REBELLION", "COUP"}
}

METRIC_MAP = {
    "event_count": "gdelt_event_count",
    "goldstein_mean": "gdelt_goldstein_mean",
    "mentions_total": "gdelt_mentions_total",
    "avg_tone_mean": "gdelt_avg_tone",
    "predatory_action_count": "gdelt_predatory_action_count",
    "border_buffer_flag": "gdelt_border_buffer_flag",
    "theme_resource_predation_count": "gdelt_theme_resource_predation_count",
    "theme_displacement_count": "gdelt_theme_displacement_count",
    "theme_governance_breakdown_count": "gdelt_theme_governance_breakdown_count"
}

# Approximate Buffer Box (CAR + 50km)
# Derived from data.yaml bbox [14.0, 2.0, 28.0, 11.5]
BUFFER_MIN_LON, BUFFER_MIN_LAT = 13.5, 1.5
BUFFER_MAX_LON, BUFFER_MAX_LAT = 28.5, 12.0

def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()

def _get_config_dates():
    try:
        cfg = load_configs()
        data_cfg = cfg[0] if isinstance(cfg, tuple) else cfg.get("data", {})
        gw = data_cfg.get("global_date_window", {})
        return gw.get("start_date"), gw.get("end_date")
    except:
        return None, None

def process_file_stream(url, ftype, timestamp, buffer_wkt=None):
    """Worker function for threaded processing."""
    # buffer_wkt is legacy/unused now that we use simple BBox checks.
    records = []
    session = get_session()
    
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200: return []
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                if ftype == 'export':
                    df = pd.read_csv(f, sep='\t', header=None, usecols=[26, 30, 31, 33, 51, 53, 56, 57], on_bad_lines='skip', low_memory=False)
                    target_ccs = {'CF', 'CD', 'SU', 'TD', 'SD'}
                    df = df[df[53].isin(target_ccs)]
                    
                    for _, item in df.iterrows():
                        try:
                            lat, lon, cc = float(item[56]), float(item[57]), item[53]
                            keep, is_buffer = (cc == 'CF'), 0
                            
                            # Simple BBox check for neighbors
                            if not keep:
                                if BUFFER_MIN_LON <= lon <= BUFFER_MAX_LON and BUFFER_MIN_LAT <= lat <= BUFFER_MAX_LAT:
                                    keep, is_buffer = True, 1
                            
                            if keep:
                                code = str(item[26])
                                weight = 1.0 if item[51] in (3,4) else (0.5 if item[51]==5 else (0.25 if item[51]==1 else 0.75))
                                records.append({
                                    "h3_index": h3.latlng_to_cell(lat, lon, H3_RES),
                                    "date": timestamp,
                                    "event_count": 1,
                                    "goldstein_sum": (float(item[30]) if pd.notnull(item[30]) else 0.0),
                                    "mentions_total": (int(item[31]) if pd.notnull(item[31]) else 0),
                                    "avg_tone_sum": (float(item[33]) if pd.notnull(item[33]) else 0.0),
                                    "border_buffer_flag": is_buffer,
                                    "predatory_action_count": (weight if code.startswith(tuple(CAMEO_PREDATORY_CODES)) else 0.0)
                                })
                        except: continue

                elif ftype == 'gkg':
                    df = pd.read_csv(f, sep='\t', header=None, usecols=[7, 9], on_bad_lines='skip', low_memory=False)
                    mask = df[9].astype(str).str.contains(r'#CF#|#CD#|#SU#|#TD#|#SD#', regex=True)
                    df = df[mask]
                    for _, item in df.iterrows():
                        themes = set(str(item[7]).split(';'))
                        clusters = {f"theme_{k}_count": 1 for k, v in THEME_CLUSTERS.items() if not themes.isdisjoint(v)}
                        if not clusters: continue
                        
                        for loc in str(item[9]).split(';'):
                            p = loc.split('#')
                            if len(p) < 6: continue
                            try:
                                lat, lon, cc = float(p[4]), float(p[5]), p[2]
                                keep, is_buffer = (cc == 'CF'), 0
                                
                                # Simple BBox check
                                if not keep:
                                    # Check country code match first to avoid false positives in other countries
                                    if cc in ['CD', 'SU', 'TD', 'SD']:
                                        if BUFFER_MIN_LON <= lon <= BUFFER_MAX_LON and BUFFER_MIN_LAT <= lat <= BUFFER_MAX_LAT:
                                            keep, is_buffer = True, 1
                                
                                if keep:
                                    rec = {"h3_index": h3.latlng_to_cell(lat, lon, H3_RES), "date": timestamp, "border_buffer_flag": is_buffer}
                                    rec.update(clusters)
                                    records.append(rec)
                            except: continue
    except Exception:
        return []
    return records

def run(start_date=None, end_date=None, incremental=True, engine=None, force_full=False):
    """
    Main GDELT ingestion function.
    
    Args:
        start_date: Override start date (YYYY-MM-DD)
        end_date: Override end date (YYYY-MM-DD)
        incremental: If True (default), fetch only new data since MAX(date)
        engine: SQLAlchemy engine
        force_full: If True, force full refresh (overrides incremental)
    """
    if engine is None:
        engine = get_db_engine()
    
    c_start, c_end = _get_config_dates()
    
    # Determine effective incremental mode
    effective_incremental = incremental and not force_full
    
    # Get requested end date from config if not provided
    requested_end = end_date or c_end or pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # CLI start date takes precedence
    cli_start_set = start_date is not None
    
    # -------------------------------------------------------------------------
    # INCREMENTAL LOADING: Use centralized helper
    # -------------------------------------------------------------------------
    if effective_incremental and not cli_start_set:
        start, end = get_incremental_window(
            engine=engine,
            table=TARGET_TABLE,
            date_col="date",
            requested_end_date=requested_end,
            default_start_date=DEFAULT_START,
            force_full=force_full,
            schema=SCHEMA
        )
        
        if start is None:
            logger.info("✅ GDELT data already up to date. No fetch needed.")
            return
        
        # Apply overlap buffer for late-arriving data
        start = pd.to_datetime(start) - timedelta(days=OVERLAP_BUFFER_DAYS)
        end = pd.to_datetime(end)
        
        logger.info(f"Incremental fetch: {start.date()} to {end.date()} (includes {OVERLAP_BUFFER_DAYS}-day overlap buffer)")
    else:
        # Use provided dates or defaults
        start = _parse_date(start_date or c_start or DEFAULT_START)
        end = _parse_date(requested_end)
        logger.info(f"Full fetch: {start.date()} to {end.date()}")

    if start > end:
        logger.info("Start date is after end date. Nothing to fetch.")
        return

    logger.info(f"Scanning GDELT files from {start.date()} to {end.date()}...")
    try:
        r = requests.get(MASTER_LIST_URL)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch master list: {e}")
        return

    files = []
    for line in r.text.splitlines():
        p = line.split()
        if len(p) < 3: continue
        url = p[2]
        if "export" not in url and "gkg" not in url: continue
        ts = pd.to_datetime(url.split('/')[-1].split('.')[0], format="%Y%m%d%H%M%S", errors='coerce')
        if ts and start <= ts <= end + pd.Timedelta(days=1):
            files.append({"url": url, "type": "export" if "export" in url else "gkg", "timestamp": ts.normalize()})
    
    if not files:
        logger.info("No new files found in GDELT master list.")
        return

    # Re-enable ProcessPoolExecutor now that Shapely is gone
    workers = max(16, multiprocessing.cpu_count() * 2) 
    
    BATCH_SIZE = 200
    
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]
        all_records = []
        
        logger.info(f"🚀 Launching {workers} workers for batch {i//BATCH_SIZE + 1} ({(len(files)//BATCH_SIZE)+1}) - {len(batch)} files...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_file_stream, f['url'], f['type'], f['timestamp']): f for f in batch}
            
            desc = f"Batch {i//BATCH_SIZE + 1}/{(len(files)//BATCH_SIZE)+1} ({batch[0]['timestamp'].date()})"
            for future in tqdm(as_completed(futures), total=len(batch), desc=desc, unit="file"):
                res = future.result()
                if res: all_records.extend(res)

        if not all_records: continue

        # Aggregate & Upload
        df = pd.DataFrame(all_records)
        cols = ['event_count', 'mentions_total', 'goldstein_sum', 'avg_tone_sum', 'predatory_action_count'] + [f"theme_{k}_count" for k in THEME_CLUSTERS.keys()]
        for c in cols: 
            if c not in df.columns: df[c] = 0.0
        
        agg = {c: 'sum' for c in cols}
        agg['border_buffer_flag'] = 'max'
        df = df.groupby(['h3_index', 'date']).agg(agg).reset_index()
        df['goldstein_mean'] = df['goldstein_sum'] / df['event_count'].replace(0, 1)
        df['avg_tone_mean'] = df['avg_tone_sum'] / df['event_count'].replace(0, 1)
        
        df_long = df.melt(id_vars=['h3_index', 'date'], value_vars=[c for c in METRIC_MAP.keys() if c in df.columns or c.endswith('mean')], var_name='variable', value_name='value')
        df_long['variable'] = df_long['variable'].map(METRIC_MAP)
        df_long = df_long.dropna(subset=['variable']).query("value != 0")
        
        # Use centralized H3 type conversion (DRY - single source of truth in utils.py)
        df_long['h3_index'] = df_long['h3_index'].apply(ensure_h3_int64)
        upload_to_postgis(engine, df_long, TARGET_TABLE, SCHEMA, primary_keys=['h3_index', 'date', 'variable'])
        
    logger.info("✅ GDELT CSV Fetch Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--incremental", action="store_true", default=True, help="Incremental mode (default)")
    parser.add_argument("--no-incremental", action="store_true", help="Force full refresh")
    parser.add_argument("--full", action="store_true", help="Alias for --no-incremental")
    args = parser.parse_args()
    
    force_full = args.no_incremental or args.full
    
    run(
        start_date=args.start_date, 
        end_date=args.end_date, 
        incremental=not force_full,
        force_full=force_full
    )