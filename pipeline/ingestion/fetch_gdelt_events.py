"""
fetch_gdelt_events.py
=====================

DAILY GDELT (Client-Side Processing) -> H3 aggregation -> Postgres.

Architecture:
1. Query BigQuery for RAW daily events (filtered by Country).
   - This avoids "Function not found" errors with H3/UDFs in BigQuery.
2. Download small dataframe to Python.
3. Apply H3 indexing (h3-py) and Actor Regex flags locally.
4. Aggregate to (h3_index, date) and upload.

This is robust, cost-effective, and fully compatible with modern H3 v4 libraries.
"""

import os
import sys
import re
import pandas as pd
import h3
from pathlib import Path
from datetime import date as date_cls
from google.cloud import bigquery
from dotenv import load_dotenv

# -----------------------
# Project imports
# -----------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, upload_to_postgis, SCHEMA, load_configs

load_dotenv(ROOT_DIR / ".env")

# -----------------------
# Configuration
# -----------------------
H3_RES = int(os.getenv("GDELT_H3_RES", "7"))
COUNTRY_CODE = os.getenv("GDELT_COUNTRY_CODE", "CF").upper()
DEFAULT_START = os.getenv("GDELT_START_DATE", "2015-02-18")
DEFAULT_END = os.getenv("GDELT_END_DATE", "")

# Chunking: Download 30 days at a time to keep memory usage low
CHUNK_DAYS = int(os.getenv("GDELT_CHUNK_DAYS", "30")) 
DEFAULT_INCREMENTAL = os.getenv("GDELT_INCREMENTAL", "true").lower() in ("1", "true", "yes", "y")
TARGET_TABLE = os.getenv("GDELT_TARGET_TABLE", "features_dynamic_daily")


# -----------------------
# Variable Mapping
# -----------------------
METRIC_MAP = {
    "event_count": "gdelt_event_count",
    "goldstein_mean": "gdelt_goldstein_mean",
    "mentions_total": "gdelt_mentions_total",
    "avg_tone_mean": "gdelt_avg_tone",
    "weighted_event_count": "gdelt_weighted_event_count",
    "actor_diversity": "gdelt_actor_diversity",
    
    # Actor Flags
    "wagner_presence": "gdelt_actor_wagner_presence",
    "rebel_cpc": "gdelt_actor_rebel_cpc",
    "rebel_upc": "gdelt_actor_rebel_upc",
    "rebel_3r": "gdelt_actor_rebel_3r",
    "militia_antibalaka": "gdelt_actor_militia_antibalaka",
    "state_forces": "gdelt_actor_state_forces",
    "peacekeepers": "gdelt_actor_peacekeepers",
}

# -----------------------
# Actor Regex Patterns
# -----------------------
ACTOR_REGEX = {
    "wagner_presence": r"(WAGNER|AFRICA\s+CORPS|RUSSIAN\s+MERCENAR|RUSSIAN\s+INSTRUCTOR)",
    "rebel_cpc": r"(CPC|COALITION\s+OF\s+PATRIOTS|BOZIZE)",
    "rebel_upc": r"(UPC|ALI\s+DARASSA|UNION\s+FOR\s+PEACE)",
    "rebel_3r": r"(3R|RETURN\s+RECLAMATION|RETURN\s+RECLAMATION\s+REHABILITATION)",
    "militia_antibalaka": r"(ANTI-?BALAKA|ANTIBALAKA)",
    "state_forces": r"(FACA|NATIONAL\s+ARMY|TOUADERA|RWANDAN)",
    "peacekeepers": r"(MINUSCA|UN\s+PEACEKEEPING|BLUE\s+HELMETS?)",
}


def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()

def _utc_today() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()

def _config_end_date() -> str | None:
    try:
        cfg = load_configs()
        data_cfg = cfg[0] if isinstance(cfg, tuple) else getattr(cfg, "data", {})
        return (data_cfg or {}).get("global_date_window", {}).get("end_date")
    except Exception as e:
        logger.warning(f"Could not read end_date from data.yaml: {e}")
        return None

def _get_last_loaded_date(engine) -> pd.Timestamp | None:
    try:
        q = f"SELECT MAX(date) AS max_date FROM {SCHEMA}.{TARGET_TABLE}"
        df = pd.read_sql(q, engine)
        if df.empty or pd.isna(df.loc[0, "max_date"]):
            return None
        return pd.to_datetime(df.loc[0, "max_date"]).normalize()
    except Exception as e:
        logger.warning(f"Could not read last loaded date: {e}")
        return None

def _date_windows(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int):
    cur = start
    delta = pd.Timedelta(days=max(1, int(chunk_days)) - 1)
    while cur <= end:
        win_end = min(cur + delta, end)
        yield cur, win_end
        cur = win_end + pd.Timedelta(days=1)

def _get_weight(gtype):
    """
    Spatial Confidence Weighting:
    3/4 (Point/City) -> 1.0
    5 (ADM1)         -> 0.5
    1 (Country)      -> 0.25
    Others           -> 0.75
    """
    if gtype in (3, 4): return 1.0
    if gtype == 5: return 0.5
    if gtype == 1: return 0.25
    return 0.75

def run(
    start_date: str | None = None,
    end_date: str | None = None,
    incremental: bool = DEFAULT_INCREMENTAL,
    chunk_days: int = CHUNK_DAYS,
    engine=None,
):
    engine = engine or get_db_engine()

    start = _parse_date(start_date or DEFAULT_START)
    if end_date:
        end = _parse_date(end_date)
    elif DEFAULT_END:
        end = _parse_date(DEFAULT_END)
    else:
        cfg_end = _config_end_date()
        end = _parse_date(cfg_end) if cfg_end else _utc_today()

    if incremental:
        last = _get_last_loaded_date(engine)
        if last:
            start = max(start, last + pd.Timedelta(days=1))

    if start > end:
        logger.info(f"Nothing to fetch: start {start.date()} > end {end.date()}")
        return

    logger.info("🚀 Starting GDELT Fetch (Client-Side Logic)")
    logger.info(f"   Country: {COUNTRY_CODE} | H3_RES: {H3_RES}")
    logger.info(f"   Range: {start.date()} -> {end.date()}")

    client = bigquery.Client()
    
    # Standard SQL to get RAW events (Partition Pruned)
    sql_raw = """
        SELECT
            PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS event_date,
            ActionGeo_Lat AS lat,
            ActionGeo_Long AS lon,
            ActionGeo_Type AS geo_type,
            GoldsteinScale,
            NumMentions,
            AvgTone,
            Actor1Name,
            Actor2Name,
            UPPER(CONCAT(IFNULL(Actor1Name, ''), ' ', IFNULL(Actor2Name, ''))) AS all_text
        FROM `gdelt-bq.gdeltv2.events`
        WHERE SQLDATE BETWEEN @start_int AND @end_int
          AND ActionGeo_CountryCode = @country_code
          AND ActionGeo_Lat IS NOT NULL
          AND ActionGeo_Long IS NOT NULL
    """

    total_uploaded = 0
    total_windows = 0

    for win_start, win_end in _date_windows(start, end, chunk_days):
        total_windows += 1
        logger.info(f"   Window {total_windows}: {win_start.date()} -> {win_end.date()}")

        # Convert to Integer for efficient Partition Filtering
        start_int = int(win_start.strftime("%Y%m%d"))
        end_int = int(win_end.strftime("%Y%m%d"))

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("country_code", "STRING", COUNTRY_CODE),
                bigquery.ScalarQueryParameter("start_int", "INTEGER", start_int),
                bigquery.ScalarQueryParameter("end_int", "INTEGER", end_int),
            ]
        )

        try:
            df = client.query(sql_raw, job_config=job_config).to_dataframe()
        except Exception as e:
            logger.error(f"   ❌ BigQuery Error: {e}")
            continue

        if df.empty:
            logger.info("   (no events found in window)")
            continue

        # ---------------------------------------------------------
        # CLIENT-SIDE PROCESSING (Python)
        # ---------------------------------------------------------
        
        # 1. H3 Mapping (using h3-py v4 syntax)
        df["h3_index"] = df.apply(lambda x: h3.latlng_to_cell(x.lat, x.lon, H3_RES), axis=1)

        # 2. Weights
        df["weight"] = df["geo_type"].map(_get_weight)

        # 3. Actor Flags (Regex)
        # We pre-compile regexes for speed
        for col, pattern in ACTOR_REGEX.items():
            regex = re.compile(pattern)
            # If match, score is equal to the event weight. Else 0.
            df[col] = df.apply(lambda x: x.weight if regex.search(x.all_text) else 0.0, axis=1)

        # 4. Aggregation Rules
        agg_rules = {
            "GoldsteinScale": "mean",
            "NumMentions": "sum",
            "AvgTone": "mean",
            "weight": "sum",       # Becomes weighted_event_count
            "Actor1Name": "nunique", # Becomes actor_diversity
        }
        # Add actor columns (summing their weighted scores)
        for col in ACTOR_REGEX.keys():
            agg_rules[col] = "sum"

        # Group by H3 AND Date (ensure we respect daily granularity)
        df_grouped = df.groupby(["h3_index", "event_date"]).agg(agg_rules).reset_index()

        # Add simple event counts (COUNT(*))
        counts = df.groupby(["h3_index", "event_date"]).size().reset_index(name="event_count")
        df_final = pd.merge(df_grouped, counts, on=["h3_index", "event_date"])

        # Rename columns to match DB
        df_final.rename(columns={
            "event_date": "date",
            "GoldsteinScale": "goldstein_mean",
            "NumMentions": "mentions_total",
            "AvgTone": "avg_tone_mean",
            "weight": "weighted_event_count",
            "Actor1Name": "actor_diversity"
        }, inplace=True)

        # ---------------------------------------------------------
        # UPLOAD
        # ---------------------------------------------------------

        # Melt to long format
        value_vars = [c for c in df_final.columns if c not in ("h3_index", "date")]
        df_long = df_final.melt(
            id_vars=["h3_index", "date"],
            value_vars=value_vars,
            var_name="variable",
            value_name="value"
        )

        # Map variable names
        df_long["variable"] = df_long["variable"].map(METRIC_MAP)
        
        # Clean up
        df_long = df_long.dropna(subset=["variable", "value"])
        df_long = df_long[df_long["value"] != 0]

        if df_long.empty:
            continue

        # ---------------------------------------------------------
        # DB COMPATIBILITY: Convert Hex to Signed Int64
        # ---------------------------------------------------------
        # Postgres BIGINT is signed, so we must convert the hex 
        # to a base-16 int and ensure it's treated as a signed 64-bit.
        def to_signed_int64(hex_str):
            unsigned = int(hex_str, 16)
            # If the value is larger than the max signed 64-bit int, 
            # wrap it into the negative range (canonical for Postgres BIGINT)
            if unsigned > 0x7FFFFFFFFFFFFFFF:
                return unsigned - 0x10000000000000000
            return unsigned

        df_long['h3_index'] = df_long['h3_index'].apply(to_signed_int64)

        # Upsert
        upload_to_postgis(
            engine, 
            df_long, 
            TARGET_TABLE, 
            SCHEMA, 
            primary_keys=["h3_index", "date", "variable"]
        )
        total_uploaded += len(df_long)
        logger.info(f"   ✅ Uploaded {len(df_long):,} rows.")

    logger.info(f"✅ GDELT fetch complete. Total uploaded feature-rows: {total_uploaded:,}")

if __name__ == "__main__":
    run()
