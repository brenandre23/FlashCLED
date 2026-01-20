"""
fetch_acled.py
==============
ACLED Conflict Event Ingestion (restored to full schema + 6 event types).
"""
import sys
from pathlib import Path

import h3.api.basic_int as h3
import pandas as pd
from sqlalchemy import inspect, text

# --- Setup Imports ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import PATHS, get_db_engine, load_configs, logger, upload_to_postgis  # noqa: E402



SCHEMA = "car_cewp"
TABLE_NAME = "acled_events"

create_table_sql = """
CREATE TABLE IF NOT EXISTS car_cewp.acled_events (
    event_id_cnty VARCHAR(50) PRIMARY KEY,
    event_date DATE NOT NULL,
    year INTEGER,
    time_precision INTEGER,
    event_type VARCHAR(100),
    sub_event_type VARCHAR(100),
    geo_precision INTEGER,
    
    -- 6 Event Type Counts
    acled_count_battles INTEGER DEFAULT 0,
    acled_count_vac INTEGER DEFAULT 0,
    acled_count_explosions INTEGER DEFAULT 0,
    acled_count_protests INTEGER DEFAULT 0,
    acled_count_riots INTEGER DEFAULT 0,
    acled_count_strategic_developments INTEGER DEFAULT 0,  -- 6th Type
    
    actor1 VARCHAR(255),
    actor2 VARCHAR(255),
    latitude FLOAT,
    longitude FLOAT,
    location VARCHAR(255),
    admin1 VARCHAR(255),
    admin2 VARCHAR(255),
    fatalities INTEGER,
    notes TEXT,
    h3_index BIGINT,
    conflict_binary INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_acled_date ON car_cewp.acled_events (event_date);
CREATE INDEX IF NOT EXISTS idx_acled_h3 ON car_cewp.acled_events (h3_index);
"""


def ensure_table_structure(engine):
    """
    Create the ACLED table if missing and patch in any new columns.
    """
    logger.info(f"Ensuring schema for {SCHEMA}.{TABLE_NAME}...")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(create_table_sql))

    inspector = inspect(engine)
    if not inspector.has_table(TABLE_NAME, schema=SCHEMA):
        logger.warning(f"Table {SCHEMA}.{TABLE_NAME} not visible after creation attempt.")
        return

    existing_cols = {col["name"] for col in inspector.get_columns(TABLE_NAME, schema=SCHEMA)}
    required_cols = {
        "geo_precision": "INTEGER",
        "time_precision": "INTEGER",
        "admin1": "VARCHAR(255)",
        "admin2": "VARCHAR(255)",
        "acled_count_battles": "INTEGER DEFAULT 0",
        "acled_count_vac": "INTEGER DEFAULT 0",
        "acled_count_explosions": "INTEGER DEFAULT 0",
        "acled_count_protests": "INTEGER DEFAULT 0",
        "acled_count_riots": "INTEGER DEFAULT 0",
        "acled_count_strategic_developments": "INTEGER DEFAULT 0",
    }

    with engine.begin() as conn:
        for col, dtype in required_cols.items():
            if col not in existing_cols:
                logger.warning(f"Column '{col}' missing in DB. Adding it now...")
                try:
                    conn.execute(text(f"ALTER TABLE {SCHEMA}.{TABLE_NAME} ADD COLUMN {col} {dtype}"))
                except Exception as e:
                    logger.warning(f"Could not add column {col}: {e}")


def process_data(df: pd.DataFrame, resolution: int = 5) -> pd.DataFrame:
    """
    Clean raw ACLED data, map to H3, derive event counts, and sanitize text fields.
    """
    initial_count = len(df)
    logger.info(f"  Raw ACLED rows: {initial_count:,}")

    # Standardize date and drop invalid rows
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    logger.info(f"  Rows after date cleaning: {len(df):,} (Dropped {initial_count - len(df):,})")
    if df.empty:
        return df

    # Latitude/Longitude to H3
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")

    def get_h3(lat, lon, res):
        try:
            return h3.latlng_to_cell(lat, lon, res)
        except Exception:
            return 0

    df["h3_index"] = df.apply(lambda row: get_h3(row["latitude"], row["longitude"], resolution), axis=1)
    df = df[df["h3_index"] != 0]
    logger.info(f"  Rows after H3 mapping: {len(df):,}")
    if df.empty:
        return df

    # Numeric hygiene
    df["h3_index"] = df["h3_index"].astype("int64")
    df["fatalities"] = pd.to_numeric(df.get("fatalities"), errors="coerce").fillna(0).astype(int)
    df["geo_precision"] = pd.to_numeric(df.get("geo_precision"), errors="coerce").fillna(0).astype(int)
    df["time_precision"] = pd.to_numeric(df.get("time_precision"), errors="coerce").fillna(0).astype(int)
    df["year"] = df.get("year", pd.Series(dtype=int))
    if df["year"].isna().any():
        df["year"] = df["event_date"].dt.year

    # 1. Calculate All 6 Event Types Manually
    df["acled_count_battles"] = (df["event_type"] == "Battles").astype(int)
    df["acled_count_vac"] = (df["event_type"] == "Violence against civilians").astype(int)
    df["acled_count_explosions"] = (df["event_type"] == "Explosions/Remote violence").astype(int)
    df["acled_count_protests"] = (df["event_type"] == "Protests").astype(int)
    df["acled_count_riots"] = (df["event_type"] == "Riots").astype(int)
    df["acled_count_strategic_developments"] = (df["event_type"] == "Strategic developments").astype(int)

    # 2. Conflict Binary
    df["conflict_binary"] = (df["fatalities"] > 0).astype(int)

    # 3. ID Generation (preserve ACLED-provided IDs; avoid synthetic H3/date IDs)
    if "event_id_cnty" in df.columns:
        df["event_id_cnty"] = df["event_id_cnty"].astype(str).str.strip()
    elif "data_id" in df.columns:
        # Fallback to data_id if present
        df["event_id_cnty"] = df["data_id"].astype(str).str.strip()
    else:
        # Last-resort synthetic ID when ACLED ID is missing (unlikely)
        logger.warning("  'event_id_cnty' missing; generating synthetic IDs.")
        df["event_id_cnty"] = (
            df["event_date"].dt.strftime("%Y%m%d") + "_" +
            df["actor1"].astype(str).str.strip() + "_" +
            df["location"].astype(str).str.strip() + "_" +
            df.groupby(["event_date", "location"]).cumcount().astype(str)
        )

    # 4. Text Sanitization (Prevent SQL errors)
    text_cols = ["actor1", "actor2", "admin1", "admin2", "location", "notes", "sub_event_type", "event_type"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    ordered_cols = [
        "event_id_cnty",
        "event_date",
        "year",
        "time_precision",
        "event_type",
        "sub_event_type",
        "geo_precision",
        "acled_count_battles",
        "acled_count_vac",
        "acled_count_explosions",
        "acled_count_protests",
        "acled_count_riots",
        "acled_count_strategic_developments",
        "actor1",
        "actor2",
        "latitude",
        "longitude",
        "location",
        "admin1",
        "admin2",
        "fatalities",
        "notes",
        "h3_index",
        "conflict_binary",
    ]

    available_cols = [c for c in ordered_cols if c in df.columns]
    return df[available_cols].copy()


def run(configs, engine):
    logger.info("=" * 60)
    logger.info("ACLED INGESTION (Full Event Types)")
    logger.info("=" * 60)

    resolution = configs["features"]["spatial"]["h3_resolution"]

    acled_path = PATHS["data_raw"] / "acled.csv"
    if not acled_path.exists():
        logger.error(f"ACLED file not found at {acled_path}")
        return

    logger.info(f"Loading {acled_path}...")
    df = pd.read_csv(acled_path, low_memory=False)

    ensure_table_structure(engine)

    df_clean = process_data(df, resolution)

    if df_clean.empty:
        logger.warning("No valid events found after processing! Check date formats or coordinate columns.")
        return

    logger.info(f"Upserting {len(df_clean):,} events to {SCHEMA}.{TABLE_NAME}...")
    upload_to_postgis(
        engine,
        df_clean,
        TABLE_NAME,
        SCHEMA,
        primary_keys=["event_id_cnty"],
    )
    logger.info("ACLED Ingestion Complete.")


if __name__ == "__main__":
    cfg = load_configs()
    eng = get_db_engine()
    run(cfg, eng)
