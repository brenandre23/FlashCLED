"""
fetch_ioda.py
=============
Source: IODA API (Georgia Tech)
Strategy: Event-Based Outage Detection

CRITICAL API NOTE (2024):
-------------------------
The IODA API no longer provides raw time-series signals via /v2/signals endpoint.
Only discrete outage EVENTS are available via /v2/outages/events.

This script converts discrete outage events into a continuous connectivity index
by aggregating event scores and durations within 14-day temporal windows.

DATA LIMITATIONS FOR CAR:
- Data starts 2022 (no earlier data available)
- Only national-level events are meaningful (regional data only exists for Bangui)
- ~270 events over 2022-2024 = sparse signal

Output Variable: ioda_outage_score
- Higher values = more/worse outages in the period
- 0 = no detected outages

ALIGNMENT & CONFIG:
- Dates: Pulled from data.yaml (global_date_window)
- Spines: Pulled from features.yaml (temporal.alignment_date, step_days)
"""

import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# --- SETUP ---
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
try:
    from utils import logger, get_db_engine, upload_to_postgis, SCHEMA, load_configs
except ImportError:
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    SCHEMA = "car_cewp"

    def load_configs():
        return None

load_dotenv()

# --- CONSTANTS ---
IODA_COUNTRY_CODE = "CF"
TABLE_NAME = "internet_outages"
VARIABLE_NAME = "ioda_outage_score"

# IODA data for CAR starts 2022
IODA_MIN_YEAR = 2022

# API Configuration
API_BASE = "https://api.ioda.inetintel.cc.gatech.edu/v2"
API_TIMEOUT = 60
MAX_RETRIES = 3


# -----------------------------------------------------------------------------
# 1. GRID LOADER
# -----------------------------------------------------------------------------


def get_full_grid_set(engine):
    """Fetches ALL valid H3 cells for CAR."""
    try:
        query = f"SELECT DISTINCT h3_index FROM {SCHEMA}.features_static"
        df = pd.read_sql(query, engine)
        return list(df["h3_index"].unique())
    except Exception as e:
        logger.error(f"Could not load static grid: {e}")
        return []


# -----------------------------------------------------------------------------
# 2. IODA API CLIENT
# -----------------------------------------------------------------------------


def api_request(endpoint, params, timeout=API_TIMEOUT):
    """Make API request with retry logic."""
    url = f"{API_BASE}/{endpoint}"
    
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("data", [])
            elif r.status_code == 404:
                return []
            else:
                logger.warning(f"API returned {r.status_code} for {endpoint}")
        except requests.exceptions.Timeout:
            logger.warning(f"API timeout (attempt {attempt + 1}/{MAX_RETRIES})")
        except Exception as e:
            logger.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    
    return []


def fetch_outage_events(start_ts, end_ts):
    """
    Fetches national-level outage events from IODA API.
    
    Each event has:
    - start: Unix timestamp
    - duration: Seconds
    - score: Severity score
    - datasource: bgp, gtr, ping-slash24, etc.
    """
    params = {
        "entityType": "country",
        "entityCode": IODA_COUNTRY_CODE,
        "from": start_ts,
        "until": end_ts,
    }
    return api_request("outages/events", params)


# -----------------------------------------------------------------------------
# 3. TEMPORAL SPINE & AGGREGATION
# -----------------------------------------------------------------------------


def generate_spine(start_date, end_date, align_date, step_days):
    """
    Generate a list of (window_start, window_end) tuples aligned to the global spine.
    """
    align_dt = pd.Timestamp(align_date)
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    # Find the first aligned window that contains or follows start_dt
    delta = (start_dt - align_dt).days
    periods_since_origin = delta // step_days
    first_window_start = align_dt + timedelta(days=periods_since_origin * step_days)
    
    if first_window_start < start_dt:
        first_window_start += timedelta(days=step_days)
    
    windows = []
    current = first_window_start
    while current < end_dt:
        window_end = current + timedelta(days=step_days)
        windows.append((current, min(window_end, end_dt)))
        current = window_end
    
    return windows


def aggregate_events_to_windows(events, windows):
    """
    Aggregate discrete outage events into temporal windows.
    
    For each window, computes the sum of severity scores (weighted by overlap).
    """
    records = []
    
    for window_start, window_end in windows:
        w_start_ts = window_start.timestamp()
        w_end_ts = window_end.timestamp()
        
        total_score = 0.0
        
        for event in events:
            event_start = event["start"]
            event_end = event_start + event.get("duration", 0)
            
            # Check if event overlaps with window
            if event_end > w_start_ts and event_start < w_end_ts:
                # Weight score by fraction of event in this window
                overlap_start = max(event_start, w_start_ts)
                overlap_end = min(event_end, w_end_ts)
                overlap_duration = overlap_end - overlap_start
                
                event_duration = event.get("duration", 1)
                fraction = overlap_duration / event_duration if event_duration > 0 else 1.0
                
                total_score += event.get("score", 0) * fraction
        
        records.append({
            "date": window_start.date(),
            "value": total_score,
        })
    
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------------------------


def run(configs=None, engine=None):
    """Main entry point for IODA ingestion."""
    
    # Load Configs
    if configs is None:
        configs = load_configs()
    
    if configs is None:
        logger.error("Could not load configs. Ensure data.yaml and features.yaml exist.")
        return

    # --- EXTRACT CONFIGURATION ---
    try:
        data_cfg = configs.data if hasattr(configs, "data") else configs.get("data", {})
    except Exception:
        data_cfg = {}
    try:
        feat_cfg = configs.features if hasattr(configs, "features") else configs.get("features", {})
    except Exception:
        feat_cfg = {}

    global_start_str = data_cfg.get("global_date_window", {}).get("start_date")
    global_end_str = data_cfg.get("global_date_window", {}).get("end_date")
    align_date = feat_cfg.get("temporal", {}).get("alignment_date")
    step_days = feat_cfg.get("temporal", {}).get("step_days", 14)

    if not all([global_start_str, global_end_str, align_date]):
        logger.error("Missing required config: global_date_window or temporal.alignment_date")
        return

    # --- CLAMP DATES FOR IODA ---
    g_start_dt = pd.to_datetime(global_start_str)
    g_end_dt = pd.to_datetime(global_end_str)
    
    # IODA data for CAR starts 2022
    effective_start = max(g_start_dt, pd.Timestamp(f"{IODA_MIN_YEAR}-01-01"))
    effective_end = g_end_dt

    logger.info("Starting IODA Ingestion.")
    logger.info(f"   Config window: {g_start_dt.date()} to {g_end_dt.date()}")
    logger.info(f"   Effective window: {effective_start.date()} to {effective_end.date()} (IODA data starts {IODA_MIN_YEAR})")
    logger.info(f"   Temporal alignment: {step_days} days from {align_date}")

    engine = engine or get_db_engine()

    # A. LOAD GRID
    full_grid = get_full_grid_set(engine)
    if not full_grid:
        logger.error("Static grid is empty. Run features_static ingestion first.")
        return
    
    logger.info(f"   Loaded {len(full_grid):,} H3 cells.")

    # B. GENERATE TEMPORAL SPINE
    windows = generate_spine(effective_start, effective_end, align_date, step_days)
    logger.info(f"   Generated {len(windows)} temporal windows.")

    if not windows:
        logger.warning("No temporal windows to process.")
        return

    # C. FETCH EVENTS
    s_ts = int(effective_start.timestamp())
    e_ts = int(effective_end.timestamp())
    
    logger.info("   Fetching outage events from IODA API...")
    events = fetch_outage_events(s_ts, e_ts)
    logger.info(f"   Found {len(events)} outage events.")

    # D. AGGREGATE TO WINDOWS
    agg = aggregate_events_to_windows(events, windows)
    
    if agg.empty:
        logger.warning("No data aggregated. Creating zero-filled baseline.")
        agg = pd.DataFrame({
            "date": [w[0].date() for w in windows],
            "value": [0.0] * len(windows),
        })
    
    non_zero = (agg["value"] > 0).sum()
    logger.info(f"   Aggregated to {len(agg)} windows ({non_zero} with outages).")

    # E. BROADCAST TO ALL CELLS
    n_dates = len(agg)
    n_cells = len(full_grid)
    
    dates_vec = np.tile(agg["date"].values, n_cells)
    vals_vec = np.tile(agg["value"].values, n_cells)
    cells_vec = np.repeat(full_grid, n_dates)
    
    result_df = pd.DataFrame({
        "h3_index": cells_vec,
        "date": dates_vec,
        "value": vals_vec,
    })

    # F. FINALIZE AND UPLOAD
    result_df["variable"] = VARIABLE_NAME
    result_df["date"] = pd.to_datetime(result_df["date"]).dt.date
    result_df["h3_index"] = result_df["h3_index"].astype("int64")
    result_df["value"] = result_df["value"].astype("float64")

    logger.info(f"   Uploading {len(result_df):,} rows...")

    upload_to_postgis(
        engine, 
        result_df, 
        TABLE_NAME, 
        SCHEMA, 
        primary_keys=["h3_index", "date", "variable"]
    )

    logger.info("✓ IODA Ingestion Complete.")


if __name__ == "__main__":
    run()
