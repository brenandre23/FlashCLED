"""
calculate_epr_features.py
========================
Purpose: TIME-VARYING EPR features mapped onto the 14-day modeling spine.

FIXES APPLIED:
- [CRITICAL-1] Fixed config loading (data_cfg vs features_cfg).
- [MINOR-3] Replaced deprecated h3.polyfill with h3.polygon_to_cells.
- Uses correct table name 'epr_core'.
"""

import sys
import math
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import h3.api.basic_int as h3
from h3 import LatLngPoly 
from sqlalchemy import text

# --- Import Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis

SCHEMA = "car_cewp"
POLY_TABLE = "geoepr_polygons"
CORE_TABLE = "epr_core"
TARGET_TABLE = "temporal_features"

CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _mode_or_unknown(series: pd.Series, unknown: str = "UNKNOWN") -> str:
    if series is None or series.empty:
        return unknown
    m = series.mode(dropna=True)
    if m is None or m.empty:
        return unknown
    return str(m.iloc[0])


def _shannon_entropy(values: pd.Series) -> float:
    """
    Normalized Shannon entropy for categorical values.
    Returns 0.0 when there is 0/1 category present.
    """
    if values is None or values.empty:
        return 0.0
    counts = values.value_counts(dropna=True)
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    ent = -(p * np.log(p)).sum()
    return float(ent / np.log(len(counts)))


def ensure_columns(engine, schema: str, table: str, cols: dict):
    with engine.begin() as conn:
        for col, sql_type in cols.items():
            conn.execute(text(f'ALTER TABLE "{schema}"."{table}" '
                              f'ADD COLUMN IF NOT EXISTS "{col}" {sql_type};'))
    logger.info(f"✓ Ensured {len(cols)} EPR columns exist in {schema}.{table}")


# -------------------------------------------------------------
# Load data with logging
# -------------------------------------------------------------
# In pipeline/processing/calculate_epr_features.py

def load_geoepr_polygons(engine) -> gpd.GeoDataFrame:
    logger.info("=" * 60)
    logger.info("Loading GeoEPR polygons...")
    
    # 1. Inspect table columns to handle naming variations
    insp = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = :schema AND table_name = :table
    """)
    with engine.connect() as conn:
        cols = pd.read_sql(insp, conn, params={"schema": SCHEMA, "table": POLY_TABLE})['column_name'].tolist()
    
    # 2. Determine Group Name column
    # Priority: group_name -> group -> name
    if 'group_name' in cols:
        name_col = 'group_name'
    elif 'group' in cols:
        name_col = 'group'
    else:
        logger.warning("Could not find 'group_name' or 'group' column. Using 'gwgroupid' as label.")
        name_col = 'gwgroupid::text as group_name' # Fallback

    # 3. Determine Group ID column
    # Priority: gwgroupid -> groupid
    id_col = 'gwgroupid' if 'gwgroupid' in cols else 'groupid'

    logger.info(f"  Using columns: ID='{id_col}', Name='{name_col}'")

    q = f"""
        SELECT
            {id_col}::bigint AS gwgroupid,
            {name_col} AS group_name,
            geometry
        FROM {SCHEMA}.{POLY_TABLE}
        WHERE geometry IS NOT NULL
    """
    
    gdf = gpd.read_postgis(q, engine, geom_col="geometry")
    
    if gdf.empty:
        raise RuntimeError(f"No GeoEPR polygons found in {SCHEMA}.{POLY_TABLE}")
    
    logger.info(f"✓ Loaded {len(gdf):,} GeoEPR polygon groups")
    
    if gdf.crs is not None and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        logger.info(f"  Converting CRS from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)
    
    return gdf


def load_epr_core(engine, min_year: int, max_year: int) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info(f"Loading EPR core data (years {min_year}-{max_year})...")
    
    q = f"""
        SELECT
            gwgroupid::bigint AS gwgroupid,
            year::int AS year,
            status::text AS status,
            status_numeric::int AS status_numeric,
            COALESCE(is_excluded, 0)::int AS is_excluded,
            COALESCE(is_included, 0)::int AS is_included,
            COALESCE(is_discriminated, 0)::int AS is_discriminated,
            COALESCE(has_autonomy, 0)::int AS has_autonomy
        FROM {SCHEMA}.{CORE_TABLE}
        WHERE year BETWEEN {int(min_year)} AND {int(max_year)}
    """
    df = pd.read_sql(q, engine)
    
    if df.empty:
        raise RuntimeError(f"No EPR core data found in {SCHEMA}.{CORE_TABLE}")
    
    logger.info(f"✓ Loaded {len(df):,} EPR group-year records")
    return df


def load_temporal_keys(engine, start_date: str, end_date: str) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info(f"Loading temporal spine keys from {SCHEMA}.{TARGET_TABLE}...")
    
    q = f"""
        SELECT h3_index::bigint AS h3_index, date::date AS date
        FROM {SCHEMA}.{TARGET_TABLE}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    df = pd.read_sql(q, engine)
    
    if df.empty:
        raise RuntimeError(
            f"temporal_features has no rows in {start_date} to {end_date}. "
            "Run calculate_temporal_features first."
        )
    
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)
    
    return df


# -------------------------------------------------------------
# Spatial membership (cached)
# -------------------------------------------------------------
def build_group_h3_membership(gdf_poly: gpd.GeoDataFrame, resolution: int) -> pd.DataFrame:
    """
    Returns DataFrame: [gwgroupid, h3_index]
    """
    cache_path = CACHE_DIR / f"geoepr_group_h3_membership_res{resolution}.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached GeoEPR membership: {cache_path}")
        df = pd.read_parquet(cache_path)
        df["gwgroupid"] = df["gwgroupid"].astype("int64")
        df["h3_index"] = df["h3_index"].astype("int64")
        return df

    logger.info("=" * 60)
    logger.info(f"Building GeoEPR -> H3 membership (resolution={resolution})...")

    rows = []
    
    for idx, r in gdf_poly.iterrows():
        gid = int(r["gwgroupid"])
        geom = r["geometry"]
        if geom is None:
            continue

        try:
            cells = set()
            geom_type = getattr(geom, "geom_type", None)

            # [MINOR-3 FIX] Use polygon_to_cells instead of polyfill
            if geom_type == "Polygon":
                exterior = [(y, x) for x, y in geom.exterior.coords]
                holes = [[(y, x) for x, y in interior.coords] for interior in geom.interiors]
                cells |= set(h3.polygon_to_cells(LatLngPoly(exterior, *holes), resolution))
            elif geom_type == "MultiPolygon":
                for part in geom.geoms:
                    exterior = [(y, x) for x, y in part.exterior.coords]
                    holes = [[(y, x) for x, y in interior.coords] for interior in part.interiors]
                    cells |= set(h3.polygon_to_cells(LatLngPoly(exterior, *holes), resolution))
            
            for c in cells:
                rows.append({"gwgroupid": gid, "h3_index": int(c)})
        except Exception as e:
            logger.warning(f"  Failed to polyfill group {gid}: {e}")
            continue

    if not rows:
        raise RuntimeError("Failed to polyfill any GeoEPR polygons.")

    df = pd.DataFrame(rows).drop_duplicates()
    df["gwgroupid"] = df["gwgroupid"].astype("int64")
    df["h3_index"] = df["h3_index"].astype("int64")

    df.to_parquet(cache_path, index=False)
    logger.info(f"✓ Built and cached {len(df):,} group-cell memberships")
    
    return df


# -------------------------------------------------------------
# Feature aggregation (h3, year)
# -------------------------------------------------------------
def aggregate_epr_by_h3_year(membership: pd.DataFrame, epr_core: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating EPR features to (h3_index, year)...")

    df = membership.merge(epr_core, on="gwgroupid", how="inner")
    
    if df.empty:
        raise RuntimeError("Membership x EPR core join produced 0 rows.")

    g = df.groupby(["h3_index", "year"], as_index=False)

    out = g.agg(
        ethnic_group_count=("gwgroupid", "nunique"),
        epr_excluded_groups_count=("is_excluded", "sum"),
        epr_included_groups_count=("is_included", "sum"),
        epr_discriminated_groups_count=("is_discriminated", "sum"),
        epr_autonomy_groups_count=("has_autonomy", "sum"),
        epr_status_mean=("status_numeric", "mean"),
    )

    out["epr_power_status_mode"] = g["status"].apply(_mode_or_unknown).values
    out["epr_status_entropy"] = g["status"].apply(_shannon_entropy).values
    out["epr_horizontal_inequality"] = out["epr_status_entropy"]

    out["h3_index"] = out["h3_index"].astype("int64")
    out["year"] = out["year"].astype(int)

    return out


# -------------------------------------------------------------
# Upsert into temporal_features
# -------------------------------------------------------------
def upsert_into_temporal_features(engine, temporal_keys: pd.DataFrame, epr_by_year: pd.DataFrame):
    logger.info("Merging EPR features onto temporal spine...")

    merged = temporal_keys.merge(
        epr_by_year,
        on=["h3_index", "year"],
        how="left"
    )

    fill_zero_cols = [
        "ethnic_group_count",
        "epr_excluded_groups_count",
        "epr_included_groups_count",
        "epr_discriminated_groups_count",
        "epr_autonomy_groups_count",
    ]
    for c in fill_zero_cols:
        merged[c] = merged[c].fillna(0).astype(int)

    merged["epr_status_mean"] = merged["epr_status_mean"].fillna(0.0).astype(float)
    merged["epr_status_entropy"] = merged["epr_status_entropy"].fillna(0.0).astype(float)
    merged["epr_horizontal_inequality"] = merged["epr_horizontal_inequality"].fillna(0.0).astype(float)
    merged["epr_power_status_mode"] = merged["epr_power_status_mode"].fillna("UNKNOWN").astype(str)

    out_cols = [
        "h3_index",
        "date",
        "ethnic_group_count",
        "epr_excluded_groups_count",
        "epr_included_groups_count",
        "epr_discriminated_groups_count",
        "epr_autonomy_groups_count",
        "epr_power_status_mode",
        "epr_status_mean",
        "epr_status_entropy",
        "epr_horizontal_inequality",
    ]
    out = merged[out_cols].copy()

    logger.info("  Ensuring EPR columns exist in database...")
    ensure_columns(engine, SCHEMA, TARGET_TABLE, {
        "ethnic_group_count": "INTEGER",
        "epr_excluded_groups_count": "INTEGER",
        "epr_included_groups_count": "INTEGER",
        "epr_discriminated_groups_count": "INTEGER",
        "epr_autonomy_groups_count": "INTEGER",
        "epr_power_status_mode": "TEXT",
        "epr_status_mean": "DOUBLE PRECISION",
        "epr_status_entropy": "DOUBLE PRECISION",
        "epr_horizontal_inequality": "DOUBLE PRECISION",
    })

    logger.info(f"  Upserting {len(out):,} rows into {SCHEMA}.{TARGET_TABLE}...")
    chunk_size = 250_000
    for i in range(0, len(out), chunk_size):
        chunk = out.iloc[i:i + chunk_size].copy()
        upload_to_postgis(engine, chunk, TARGET_TABLE, SCHEMA, ["h3_index", "date"])

    logger.info("✅ EPR features successfully upserted into temporal_features")


def main():
    logger.info("STEP 3.3: EPR FEATURE ENGINEERING (TIME-VARYING)")
    engine = None
    try:
        # [CRITICAL-1 FIX] Correct config loading (features_cfg)
        data_cfg, features_cfg, _ = load_configs()
        engine = get_db_engine()

        start_date = data_cfg["global_date_window"]["start_date"]
        end_date = data_cfg["global_date_window"]["end_date"]
        min_year = int(start_date[:4])
        max_year = int(end_date[:4])

        # [CRITICAL-1 FIX] Load resolution from features config
        resolution = int(features_cfg["spatial"]["h3_resolution"])

        # 1) Load keys
        temporal_keys = load_temporal_keys(engine, start_date, end_date)

        # 2) Load polygons + core
        gdf_poly = load_geoepr_polygons(engine)
        df_core = load_epr_core(engine, min_year=min_year, max_year=max_year)

        # 3) Build or load cached membership
        membership = build_group_h3_membership(gdf_poly, resolution)

        # 4) Aggregate per (h3, year)
        epr_by_year = aggregate_epr_by_h3_year(membership, df_core)

        # 5) Upsert
        upsert_into_temporal_features(engine, temporal_keys, epr_by_year)

        logger.info("✅ EPR FEATURE CALCULATION COMPLETE")

    except Exception as e:
        logger.error(f"❌ EPR Calculation Failed: {e}", exc_info=True)
        raise
    finally:
        if engine:
            engine.dispose()


if __name__ == "__main__":
    main()