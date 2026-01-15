"""
fetch_ioda.py
=============
Source: IODA API (Georgia Tech)
Variable: Active Probing (ping_slash24_up)
Logic: Hybrid Overlay (National Baseline + Regional Precision)

ALIGNMENT & CONFIG:
- Dates: Pulled from data.yaml (global_date_window)
- Spines: Pulled from features.yaml (temporal.alignment_date, step_days)
- Logic: Resamples raw 10-min data -> 14-day averages aligned to 2000-01-01.
"""

import sys
import time
import requests
import pandas as pd
import unicodedata
import re
from pathlib import Path
from datetime import datetime, timezone
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
    SCHEMA = "public"

    def load_configs():
        return None

load_dotenv()

# --- CONSTANTS ---
IODA_COUNTRY_CODE = "CF"
TABLE_NAME = "internet_outages"
VARIABLE_NAME = "ioda_connectivity_index"
IODA_MIN_YEAR = 2014  # API returns 404s before this year

# -----------------------------------------------------------------------------
# 1. GEOGRAPHIC UTILS
# -----------------------------------------------------------------------------


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    n = unicodedata.normalize("NFKD", n).encode("ASCII", "ignore").decode("utf-8")
    return re.sub(r"[^a-z0-9]", "", n)


def get_full_grid_set(engine):
    """Fetches ALL valid H3 cells for CAR."""
    try:
        query = f"SELECT DISTINCT h3_index FROM {SCHEMA}.features_static"
        df = pd.read_sql(query, engine)
        return list(df["h3_index"].unique())
    except Exception as e:
        logger.error(f"Could not load static grid: {e}")
        return []


def get_admin_map_from_static(engine):
    """
    Maps admin2 names in features_static -> List of H3 cells.
    Uses direct equality on admin2 names (normalized) and drops NULL/NaN rows.
    """
    try:
        df = pd.read_sql(f"SELECT admin2, h3_index FROM {SCHEMA}.features_static", engine)
    except Exception as e:
        logger.error(f"Could not load admin2 mapping from features_static: {e}")
        return {}, {}

    if "admin2" not in df.columns:
        logger.error("Column admin2 not found on features_static.")
        return {}, {}

    df = df[df["admin2"].notna()]
    df = df[df["admin2"].astype(str).str.lower() != "nan"]  # Ignore stray literal NaN entry
    if df.empty:
        logger.warning("No admin2 values found on features_static.")
        return {}, {}

    df["admin2_norm"] = df["admin2"].apply(normalize_name)
    mapping = df.groupby("admin2_norm")["h3_index"].apply(list).to_dict()
    display_names = df.groupby("admin2_norm")["admin2"].apply(lambda s: sorted(set(s))).to_dict()
    return mapping, display_names


# -----------------------------------------------------------------------------
# 2. IODA API CLIENT
# -----------------------------------------------------------------------------


def fetch_raw_signal(entity_type, entity_code, start_ts, end_ts):
    """
    Fetches RAW time-series (10-min resolution) for 'active_probing'.
    """
    url = "https://api.ioda.inetintel.cc.gatech.edu/v2/signals/raw"
    params = {
        "entityType": entity_type,
        "entityCode": entity_code,
        "signal": "active_probing",
        "from": start_ts,
        "until": end_ts,
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data.get("data", {}).get("values", [])  # Returns [[ts, val], ...]
            time.sleep(1)
        except Exception as e:
            logger.warning(f"API retry {attempt + 1}: {e}")
            time.sleep(2)
    return []


def get_ioda_regions(country_code):
    """Gets list of available sub-regions (Bangui, Ouham, etc.) defined by IODA."""
    url = "https://api.ioda.inetintel.cc.gatech.edu/v2/entities/query"
    try:
        r = requests.get(
            url, params={"entityType": "region", "relatedTo": f"country/{country_code}"}, timeout=20
        )
        if r.status_code == 200:
            return [(x["code"], x["name"]) for x in r.json().get("data", [])]
    except Exception as e:
        logger.warning(f"Could not fetch IODA regions: {e}")
    return []


# -----------------------------------------------------------------------------
# 3. SPINE ALIGNMENT
# -----------------------------------------------------------------------------


def process_to_spine(raw_values, start_ts, end_ts, align_date, step_days):
    """
    Converts raw IODA samples -> 14-Day Averages aligned to global spine.
    """
    if not raw_values:
        return None

    df = pd.DataFrame(raw_values, columns=["timestamp", "val"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("dt", inplace=True)

    # --- CRITICAL: Use 'origin' from features.yaml ---
    resampled = df.resample(f"{step_days}D", origin=pd.Timestamp(align_date)).mean()

    # Trim to requested year window
    s_dt = datetime.fromtimestamp(start_ts)
    e_dt = datetime.fromtimestamp(end_ts)
    resampled = resampled[(resampled.index >= s_dt) & (resampled.index <= e_dt)]

    if resampled.empty:
        return None

    return resampled.reset_index().rename(columns={"dt": "date"})


# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------------------------


def run(configs=None, engine=None):
    # Load Configs
    if configs is None:
        configs = load_configs()

    # --- EXTRACT CONFIGURATION ---
    # 1. Dates from data.yaml
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

    # 2. Spine from features.yaml
    align_date = feat_cfg.get("temporal", {}).get("alignment_date")
    step_days = feat_cfg.get("temporal", {}).get("step_days")

    # --- CLAMP DATES FOR IODA ---
    # IODA API starts ~2014. If config starts 2000, we clamp to 2014 to save API calls.
    # But we keep alignment_date=2000 to ensure bins match.
    g_start_dt = pd.to_datetime(global_start_str)
    g_end_dt = pd.to_datetime(global_end_str)

    start_year = max(g_start_dt.year, IODA_MIN_YEAR)
    end_year = g_end_dt.year

    logger.info("Starting IODA Hybrid Ingestion.")
    logger.info(f"   Window: {start_year}-{end_year} (Config requested: {g_start_dt.year})")
    logger.info(f"   Alignment: {step_days} days from {align_date}")

    engine = engine or get_db_engine()

    # A. PREPARE GEOGRAPHY
    full_grid = get_full_grid_set(engine)
    if not full_grid:
        logger.error("Static grid is empty. Run features_static ingestion first.")
        return

    admin_map, admin_display_names = get_admin_map_from_static(engine)
    if not admin_map:
        logger.error("Admin2 mapping missing; cannot apply regional overlays.")
        return

    ioda_regions = get_ioda_regions(IODA_COUNTRY_CODE)
    logger.info(f"   Found {len(ioda_regions)} IODA-defined regions.")
    unmatched_regions = []

    # B. PROCESS YEAR BY YEAR
    for year in range(start_year, end_year + 1):
        logger.info(f"   Processing Year {year}...")

        # Timestamps for API
        s_ts = int(datetime(year, 1, 1).replace(tzinfo=timezone.utc).timestamp())
        e_ts = int(datetime(year, 12, 31).replace(tzinfo=timezone.utc).timestamp())

        # 1. FETCH NATIONAL BASELINE
        raw_country = fetch_raw_signal("country", IODA_COUNTRY_CODE, s_ts, e_ts)
        df_country = process_to_spine(raw_country, s_ts, e_ts, align_date, step_days)

        if df_country is None:
            logger.info(f"      No data for {year}. Skipping.")
            continue

        # 2. BROADCAST NATIONAL BASELINE (In Memory)
        # Create Cartesian Product: (All Cells) x (All Time Steps in Year)
        n_dates = len(df_country)
        n_cells = len(full_grid)

        dates_vec = df_country["date"].repeat(n_cells).values
        vals_vec = df_country["val"].repeat(n_cells).values
        cells_vec = full_grid * n_dates

        year_df = pd.DataFrame({"h3_index": cells_vec, "date": dates_vec, "value": vals_vec})

        # 3. REGIONAL OVERLAY (The Hybrid Step)
        regions_applied = 0
        for r_code, r_name in ioda_regions:
            norm_name = normalize_name(r_name)
            target_cells = admin_map.get(norm_name)
            if not target_cells:
                unmatched_regions.append(r_name)
                continue

            # Fetch & Resample
            raw_region = fetch_raw_signal("region", r_code, s_ts, e_ts)
            df_region = process_to_spine(raw_region, s_ts, e_ts, align_date, step_days)

            if df_region is not None:
                # Expand region data to its specific cells
                r_dates = df_region["date"].repeat(len(target_cells)).values
                r_vals = df_region["val"].repeat(len(target_cells)).values
                r_cells = target_cells * len(df_region)

                region_overlay = pd.DataFrame(
                    {"h3_index": r_cells, "date": r_dates, "val_region": r_vals}
                )

                # Merge & Overwrite
                year_df = year_df.merge(region_overlay, on=["h3_index", "date"], how="left")

                mask = year_df["val_region"].notna()
                year_df.loc[mask, "value"] = year_df.loc[mask, "val_region"]

                year_df.drop(columns=["val_region"], inplace=True)
                regions_applied += 1

        # 4. UPLOAD
        year_df["variable"] = VARIABLE_NAME
        year_df["date"] = pd.to_datetime(year_df["date"]).dt.date

        logger.info(f"      Saving {len(year_df):,} rows (Regions overlaid: {regions_applied})...")

        upload_to_postgis(
            engine, year_df, TABLE_NAME, SCHEMA, primary_keys=["h3_index", "date", "variable"]
        )

        del year_df, df_country

    if unmatched_regions:
        logger.warning(
            f"No admin2 match for IODA regions: {sorted(set(unmatched_regions))}"
        )
    extra_static = set(admin_map.keys()) - {normalize_name(name) for _, name in ioda_regions}
    if extra_static:
        extras_display = sorted(
            {
                name
                for key, names in admin_display_names.items()
                if key in extra_static
                for name in names
            }
        )
        logger.info(
            "Admin2 values present in features_static without an IODA region match: "
            f"{extras_display}"
        )
    logger.info("IODA ingestion complete.")


if __name__ == "__main__":
    run()
