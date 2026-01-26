"""
fetch_gee_server_side.py
========================
Purpose: Server-side environmental data aggregation.

UPDATES:
- Uses centralized 'init_gee' from utils.py for silent Service Account auth.
- DYNAMIC END DATE: Automatically trims query to data availability.
- PUBLICATION LAG: Config-driven via data.yaml publication_lag_steps.
"""

import sys
import ee
import pandas as pd
import time
import json
import gc
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import (
    logger,
    get_db_engine,
    load_configs,
    upload_to_postgis,
    ensure_h3_int64,
    init_gee,
    get_incremental_window,
)

# --- Constants ---
SCHEMA = "car_cewp"
GRID_TABLE = "features_static"
TARGET_TABLE = "environmental_features"
DATA_DIR = ROOT_DIR / "data" / "raw"

MAX_WORKERS = 12
H3_BATCH_SIZE = 300
MAX_RETRIES = 5
BATCH_SLEEP = 1


def get_publication_lag_days(data_cfg, features_cfg):
    """
    Calculate publication lag in days from config.
    
    Reads:
      - data.yaml:     gee.publication_lag_steps (default: 1)
      - features.yaml: temporal.step_days (default: 14)
    
    Returns:
      int: Number of days to shift stored dates forward
    """
    step_days = features_cfg.get('temporal', {}).get('step_days', 14)
    lag_steps = data_cfg.get('gee', {}).get('publication_lag_steps', 1)
    lag_days = lag_steps * step_days
    
    logger.info(f"Publication lag: {lag_steps} steps × {step_days} days = {lag_days} days")
    return lag_days


# -------------------------------------------------------------------------
# 1. DATABASE & TYPE HELPERS
# -------------------------------------------------------------------------

def ensure_environmental_table_exists(engine, schema="car_cewp"):
    from sqlalchemy import text
    create_sql = text(f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    CREATE TABLE IF NOT EXISTS {schema}.environmental_features (
        h3_index BIGINT NOT NULL,
        date DATE NOT NULL,
        precip_mean_depth_mm FLOAT,
        temp_mean FLOAT,
        dew_mean FLOAT,
        soil_moisture_mean FLOAT,
        ndvi_max FLOAT,
        ntl_mean FLOAT,
        ntl_peak FLOAT,
        ntl_stale_days FLOAT,
        ntl_trust_frac FLOAT,
        water_local_mean FLOAT,
        water_local_max FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (h3_index, date)
    );
    """)
    with engine.begin() as conn:
        conn.execute(create_sql)
        # Ensure new VIIRS columns exist on legacy tables
        conn.execute(text(f'ALTER TABLE {schema}.environmental_features ADD COLUMN IF NOT EXISTS "ntl_peak" FLOAT;'))
        conn.execute(text(f'ALTER TABLE {schema}.environmental_features ADD COLUMN IF NOT EXISTS "ntl_stale_days" FLOAT;'))
        conn.execute(text(f'ALTER TABLE {schema}.environmental_features ADD COLUMN IF NOT EXISTS "ntl_trust_frac" FLOAT;'))
    logger.info(f"✓ Table {schema}.environmental_features is ready.")


def get_h3_grid_data(engine):
    logger.info("Loading H3 Grid from PostGIS...")
    sql = f"SELECT to_hex(h3_index::bigint) AS h3_id, ST_AsGeoJSON(geometry) AS geom FROM {SCHEMA}.{GRID_TABLE}"
    df = pd.read_sql(sql, engine)
    grid_data = [{"h3_id": r["h3_id"], "geom": json.loads(r["geom"])} for _, r in df.iterrows()]
    logger.info(f"Loaded {len(grid_data)} H3 cells.")
    return grid_data


def make_ee_feature_collection(grid_subset):
    features = [ee.Feature(ee.Geometry(c['geom']), {"h3_index": c['h3_id']}) for c in grid_subset]
    return ee.FeatureCollection(features)


# -------------------------------------------------------------------------
# 2. DATE LOGIC & DYNAMIC LIMITS
# -------------------------------------------------------------------------

def get_collection_last_date(collection_id):
    """Queries GEE to find the absolute last available date for a collection."""
    try:
        last_img = ee.ImageCollection(collection_id).limit(1, 'system:time_start', False).first()
        last_ts = last_img.get('system:time_start').getInfo()
        if last_ts:
            return datetime.fromtimestamp(last_ts / 1000.0, tz=timezone.utc).replace(tzinfo=None)
    except Exception as e:
        logger.warning(f"Could not verify end date for {collection_id}: {e}")
    return None


def build_14day_spine(start_year: int, end_year: int):
    epoch = datetime(2000, 1, 1)
    req_start = datetime(start_year, 1, 1)
    req_end = datetime(end_year, 12, 31)
    spine = []
    curr = epoch
    while curr < req_start:
        curr += timedelta(days=14)
    while curr <= req_end:
        spine.append((curr, curr + timedelta(days=13)))
        curr += timedelta(days=14)
    return spine


def window_strs(start_dt: datetime, end_dt: datetime):
    return start_dt.strftime("%Y-%m-%d"), (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")


def get_prop(props: dict, keys, default=None):
    for k in keys:
        if k in props and props[k] is not None:
            return props[k]
    return default


# -------------------------------------------------------------------------
# 3. GEE PROCESSING LOGIC
# -------------------------------------------------------------------------

def get_water_image(start_dt, end_dt):
    mid = start_dt + (end_dt - start_dt) / 2
    if mid.year < 2022:
        m_start = datetime(mid.year, mid.month, 1).strftime("%Y-%m-%d")
        m_end = (datetime(mid.year, mid.month, 1) + timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d")
        jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory").filterDate(m_start, m_end)
        return ee.Image(ee.Algorithms.If(jrc.size().gt(0), jrc.first().select("water").eq(2), ee.Image.constant(0))).rename("water").toByte()

    s_str, e_str = window_strs(start_dt, end_dt)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    landsat = l8.merge(l9).filterDate(s_str, e_str)

    def prep(img):
        qa = img.select("QA_PIXEL")
        clear = (
            qa.bitwiseAnd(1 << 3).eq(0)
            .And(qa.bitwiseAnd(1 << 4).eq(0))
            .And(qa.bitwiseAnd(1 << 5).eq(0))
        )
        return img.normalizedDifference(["SR_B3", "SR_B6"]).gt(0.1).rename("water").updateMask(clear)

    return (
        ee.Image(
            ee.Algorithms.If(
                landsat.size().gt(0),
                landsat.map(prep).max().unmask(0),
                ee.Image.constant(0),
            )
        )
        .rename("water")
        .toByte()
    )


def get_viirs_integrated(start_date, end_date):
    """
    Fetches VIIRS NTL with Stability, Kinetic, and Staleness signals.
    
    Signal Decomposition:
    - Stability (ntl_mean): Gap-filled mean for infrastructure/development baseline
    - Kinetic (ntl_peak): Raw max for fires/explosions/transient events
    - Staleness (ntl_stale_days): Median days since last HQ retrieval (uncertainty)
    
    Uses NASA/VIIRS/002/VNP46A2 (VNP46A2 Black Marble product).
    Returns masked NULLs for pre-2012 (VIIRS launch: 2012-01-28).
    """
    # 1. Access Collection (NASA VNP46A2)
    viirs_col = ee.ImageCollection("NASA/VIIRS/002/VNP46A2").filterDate(start_date, end_date)

    # 2. STABILITY SIGNAL (Infrastructure / Development)
    # Gap-Filled Mean: Smooth baseline, handles clouds, creates continuity.
    signal_stable = viirs_col.select("Gap_Filled_DNB_BRDF_Corrected_NTL").mean().rename("ntl_mean")

    # 3. KINETIC SIGNAL (Fires / Explosions)
    # Raw Band Max: Captures short-term intensity spikes that gap-filling erases.
    # Essential for catching conflict events like arson.
    signal_peak = viirs_col.select("DNB_BRDF_Corrected_NTL").max().rename("ntl_peak")

    # 4. UNCERTAINTY SIGNAL (Data Age)
    # 'Latest_High_Quality_Retrieval' = Days since last good observation.
    # Use MEDIAN to avoid skew from single bad pixels (outlier-resistant).
    # High value = Old data (Carry-forward risk).
    stale_days = viirs_col.select("Latest_High_Quality_Retrieval").median().rename("ntl_stale_days")

    # 5. TRUST FRACTION (Quality Coverage)
    # Mandatory_Quality_Flag == 0 means high quality.
    def measure_trust(img):
        return img.select("Mandatory_Quality_Flag").eq(0).rename("ntl_trust_flag") \
                 .copyProperties(img, ["system:time_start"])
    trust_frac = viirs_col.map(measure_trust).mean().rename("ntl_trust_frac")

    # 5. PRE-2012 GUARD (Masking)
    # If the collection is empty (pre-2012), return NULLs so the binary flag takes over.
    # Essential for structural breaks - prevents zero-fill leakage.
    dummy = ee.Image.constant(0).selfMask().rename("ntl_mean") \
            .addBands(ee.Image.constant(0).selfMask().rename("ntl_peak")) \
            .addBands(ee.Image.constant(0).selfMask().rename("ntl_stale_days")) \
            .addBands(ee.Image.constant(0).selfMask().rename("ntl_trust_frac"))

    return ee.Algorithms.If(
        viirs_col.size().gt(0),
        signal_stable.addBands(signal_peak).addBands(stale_days).addBands(trust_frac),
        dummy
    )


def process_single_batch(batch_cells, s_str, e_excl_str, collections_cfg, start_dt, end_dt):
    chirps_id = collections_cfg.get("chirps", "UCSB-CHG/CHIRPS/DAILY")
    era5_id = collections_cfg.get("era5", "ECMWF/ERA5_LAND/HOURLY")
    modis_id = collections_cfg.get("modis", "MODIS/061/MCD43A4")
    viirs_id = collections_cfg.get("viirs", "NASA/VIIRS/002/VNP46A2")

    batch_fc = make_ee_feature_collection(batch_cells)

    for attempt in range(MAX_RETRIES):
        try:
            # 1. Coarse
            chirps = ee.ImageCollection(chirps_id).filterDate(s_str, e_excl_str).select("precipitation").sum().rename("precip_depth_mm")
            era5 = ee.ImageCollection(era5_id).filterDate(s_str, e_excl_str).mean().select(["temperature_2m", "dewpoint_temperature_2m", "volumetric_soil_water_layer_1"], ["temp", "dew", "soil"])
            coarse_res = chirps.addBands(era5).reduceRegions(collection=batch_fc, reducer=ee.Reducer.mean(), scale=5000, tileScale=4)

            # 2. Fine (VIIRS at 500m scale, tileScale=8 to prevent timeouts)
            viirs_img = get_viirs_integrated(s_str, e_excl_str)

            def add_ndvi(img):
                return img.normalizedDifference(["Nadir_Reflectance_Band2", "Nadir_Reflectance_Band1"]).rename("ndvi")

            modis = ee.ImageCollection(modis_id).filterDate(s_str, e_excl_str).map(add_ndvi).select("ndvi")
            modis_img = ee.Image(ee.Algorithms.If(modis.size().gt(0), modis.max(), ee.Image.constant(0))).rename("ndvi_max")
            water_img = get_water_image(start_dt, end_dt).rename("water_local_mean")
            
            # VIIRS native resolution is ~500m; use scale=500, tileScale=8 to prevent GEE timeouts
            fine_res = viirs_img.addBands(modis_img).addBands(water_img).reduceRegions(collection=batch_fc, reducer=ee.Reducer.mean(), scale=500, tileScale=8)

            # 3. Water Max
            water_max_res = get_water_image(start_dt, end_dt).rename("water_bin").reduceRegions(collection=batch_fc, reducer=ee.Reducer.max().setOutputs(["water_local_max"]), scale=30, tileScale=4)

            return (coarse_res.getInfo(), fine_res.getInfo(), water_max_res.getInfo())

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "too many requests" in err:
                time.sleep((attempt + 1) * 10)
            elif "timeout" in err or "remote end closed" in err:
                time.sleep((attempt + 1) * 5)
            else:
                time.sleep(2)
    raise RuntimeError("Worker failed")


# -------------------------------------------------------------------------
# 4. MAIN ORCHESTRATOR
# -------------------------------------------------------------------------

def process_year_batch(year: int, all_cells: list, collections_cfg: dict, windows, engine, lag_days: int):
    """
    Process a year's worth of windows with publication lag applied.
    
    lag_days: Number of days to shift stored date forward (from config)
    """
    logger.info(f"--- Processing {len(windows)} windows for Year {year} (Parallel: {MAX_WORKERS} threads) ---")
    logger.info(f"    Publication lag: {lag_days} days")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"CEWP_Env_Res5_Year_{year}.csv"
    header_written = out_path.exists()

    cell_batches = [all_cells[i : i + H3_BATCH_SIZE] for i in range(0, len(all_cells), H3_BATCH_SIZE)]

    for start_dt, end_dt in tqdm(windows, desc=f"Year {year}"):
        s_str, e_excl_str = window_strs(start_dt, end_dt)
        daily_rows = []

        work_items = [(batch, s_str, e_excl_str, collections_cfg, start_dt, end_dt) for batch in cell_batches]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(process_single_batch, *args): i for i, args in enumerate(work_items)}

            for future in as_completed(future_to_batch):
                try:
                    c_data, f_data, w_data = future.result()
                    if not (c_data and f_data and w_data): continue

                    c_map = {f["properties"]["h3_index"]: f["properties"] for f in c_data.get("features", [])}
                    w_map = {f["properties"]["h3_index"]: f["properties"].get("water_local_max") for f in w_data.get("features", [])}

                    for feat in f_data.get("features", []):
                        props = feat.get("properties", {})
                        h3_idx = props.get("h3_index")
                        c_props = c_map.get(h3_idx, {})

                        # PUBLICATION LAG: Shift stored date forward by lag_days
                        # Data from window [start_dt, end_dt] becomes "available" at (start_dt + lag_days)
                        lagged_date = start_dt + timedelta(days=lag_days)

                        daily_rows.append({
                            "h3_index": h3_idx,
                            "date": lagged_date.strftime("%Y-%m-%d"),
                            "precip_mean_depth_mm": get_prop(c_props, ["precip_depth_mm", "precip_depth_mm_mean"]),
                            "temp_mean": get_prop(c_props, ["temp", "temp_mean"]),
                            "dew_mean": get_prop(c_props, ["dew", "dew_mean"]),
                            "soil_moisture_mean": get_prop(c_props, ["soil", "soil_mean"]),
                            "ndvi_max": get_prop(props, ["ndvi_max", "ndvi_max_mean", "ndvi"]),
                            "ntl_mean": get_prop(props, ["ntl_mean", "ntl_mean_mean", "ntl"]),
                            "ntl_peak": get_prop(props, ["ntl_peak", "ntl_peak_mean"]),
                            "ntl_stale_days": get_prop(props, ["ntl_stale_days", "ntl_stale_days_mean"]),
                            "water_local_mean": get_prop(props, ["water_local_mean", "water_local_mean_mean", "water"]),
                            "water_local_max": w_map.get(h3_idx),
                        })
                except Exception as e:
                    logger.error(f"Batch failed: {e}")

        if daily_rows:
            df = pd.DataFrame(daily_rows)
            df["h3_index"] = df["h3_index"].apply(ensure_h3_int64)
            df = df.dropna(subset=["h3_index"])
            df["h3_index"] = df["h3_index"].astype("int64")

            mode = "a" if header_written else "w"
            df.to_csv(out_path, mode=mode, header=not header_written, index=False)
            header_written = True

            if engine:
                upload_to_postgis(engine, df, TARGET_TABLE, SCHEMA, primary_keys=["h3_index", "date"])

        gc.collect()


def run(configs, engine, args=None):
    """
    Incremental GEE fetch with 14-day safety overlap for late-arriving scenes.
    """
    data_cfg = configs["data"]
    features_cfg = configs["features"]
    project_id = data_cfg["gee"]["project_id"]

    requested_end = data_cfg["global_date_window"].get("end_date", datetime.now().strftime("%Y-%m-%d"))
    default_start = data_cfg["global_date_window"].get("start_date", "2000-01-01")

    # Determine missing window (no overlap) then apply 14-day buffer
    force_full = getattr(args, "no_incremental", False) if args is not None else False
    start, end = get_incremental_window(
        engine=engine,
        table=TARGET_TABLE,
        date_col="date",
        requested_end_date=requested_end,
        default_start_date=default_start,
        force_full=force_full,
        schema=SCHEMA,
    )

    if start is None:
        logger.info("✅ Environmental features up to date; no GEE fetch needed.")
        return

    buffer_start = pd.to_datetime(start) - timedelta(days=14)
    buffer_end = pd.to_datetime(end)
    start_year = buffer_start.year
    end_year = buffer_end.year

    ensure_environmental_table_exists(engine)
    
    # --- GET PUBLICATION LAG FROM CONFIG ---
    lag_days = get_publication_lag_days(data_cfg, features_cfg)
    
    # --- AUTHENTICATE ---
    init_gee(project_id)

    era5_id = data_cfg["gee"]["collections"].get("era5", "ECMWF/ERA5_LAND/HOURLY")
    logger.info(f"Checking data availability for {era5_id}...")
    real_end_date = get_collection_last_date(era5_id)

    if real_end_date:
        logger.info(f"✓ Latest ERA5 data found: {real_end_date.strftime('%Y-%m-%d')}")
        cutoff_date = real_end_date
    else:
        logger.warning("Could not determine ERA5 end date. Using requested end date.")
        cutoff_date = buffer_end

    all_cells = get_h3_grid_data(engine)

    for year in range(start_year, end_year + 1):
        full_spine = build_14day_spine(year, year)
        valid_spine = [
            w for w in full_spine
            if (w[0] >= buffer_start) and (w[0] <= buffer_end) and (w[0] <= cutoff_date)
        ]

        if not valid_spine:
            logger.info(f"Skipping Year {year} (no windows within buffer/end/cutoff)")
            continue

        out_path = DATA_DIR / f"CEWP_Env_Res5_Year_{year}.csv"
        processed = set()
        if out_path.exists():
            # Sync existing CSV to DB to ensure DB isn't missing prior exports
            try:
                df_csv = pd.read_csv(out_path)
                if not df_csv.empty:
                    df_csv["h3_index"] = df_csv["h3_index"].apply(ensure_h3_int64)
                    upload_to_postgis(
                        engine=engine,
                        df=df_csv,
                        table_name=TARGET_TABLE,
                        schema=SCHEMA,
                        primary_keys=["date", "h3_index"],
                    )
                    logger.info(f"✅ Synced existing CSV for year {year} to PostGIS ({len(df_csv)} rows).")
                processed = set(pd.to_datetime(df_csv["date"]).dt.strftime("%Y-%m-%d")) if not df_csv.empty else set()
            except Exception:
                processed = set()

        # Check against LAGGED dates (what we would store)
        windows_to_run = []
        for w in valid_spine:
            lagged_date = (w[0] + timedelta(days=lag_days)).strftime("%Y-%m-%d")
            if lagged_date not in processed:
                windows_to_run.append(w)

        if windows_to_run:
            process_year_batch(year, all_cells, data_cfg["gee"]["collections"], windows_to_run, engine, lag_days)
        else:
            logger.info(f"Year {year} complete (nothing new within buffered window).")


if __name__ == "__main__":
    cfgs = load_configs()
    cfg_dict = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]} if isinstance(cfgs, tuple) else cfgs
    run(cfg_dict, get_db_engine())
