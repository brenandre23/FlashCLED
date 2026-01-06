"""
pipeline/processing/spatial_disaggregation.py
=============================================
Spatial disaggregation from admin boundaries to H3 grid.

UPDATES:
1. IOM FIX: Reads from 'iom_dtm_raw' (was iom_displacement).
2. COLUMN FIX: Maps 'individuals' -> displacement_sum, 'reporting_date' -> date.
3. IPC REMOVED: Removed all food security disaggregation logic.
4. ADMIN MAPPING: Preserves IOM Admin2 -> WBG Admin3 mapping logic.
5. OVERLAP FIX: Deduplicates sjoin results to handle overlapping admin polygons.
"""

import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from sqlalchemy import text, inspect
from typing import Dict, List, Optional, Tuple

# --- Setup paths ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis, SCHEMA

# --- Constants ---
POPULATION_TABLE = "population_h3"
STATIC_TABLE = "features_static"
IOM_SOURCE_TABLE = "iom_dtm_raw"  # Updated to match fetch_iom.py


def ensure_admin_columns(engine) -> None:
    """
    Ensure admin1/admin2/admin3 columns exist on features_static.
    """
    with engine.begin() as conn:
        for col in ["admin1", "admin2", "admin3"]:
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{STATIC_TABLE} ADD COLUMN IF NOT EXISTS {col} VARCHAR(255);"))


def _infer_pcode_col(gdf: gpd.GeoDataFrame, level: str) -> Optional[str]:
    """Infer the p-code column name from a GeoDataFrame."""
    candidates = []
    
    if level == "admin1":
        candidates = ["ADM1_PCODE", "adm1_pcode", "PCODE", "pcode", "ADM1_CODE"]
    elif level == "admin2":
        candidates = ["ADM2_PCODE", "adm2_pcode", "PCODE", "pcode", "ADM2_CODE"]
    elif level == "admin3":
        # CRITICAL: Admin3 often uses adm2_pcode in WBG data due to naming inconsistencies
        candidates = ["adm2_pcode", "ADM3_PCODE", "adm3_pcode", "PCODE", "pcode", "ADM3_CODE"]
    
    for col in candidates:
        if col in gdf.columns:
            logger.info(f"  Using p-code column: {col}")
            return col
    return None


def _infer_name_col(gdf: gpd.GeoDataFrame, level: str) -> Optional[str]:
    """Infer the admin name column from a GeoDataFrame."""
    candidates = []
    
    if level == "admin1":
        candidates = ["ADM1_REF", "ADM1_NAME", "adm1_name", "NAME_1", "NAM_1", "name"]
    elif level == "admin2":
        candidates = ["ADM2_REF", "ADM2_NAME", "adm2_name", "NAME_2", "NAM_2", "name"]
    elif level == "admin3":
        # CRITICAL: Admin3 often uses adm2_ref_name or adm2_name in WBG data
        candidates = ["adm2_ref_name", "adm2_name", "ADM3_REF", "ADM3_NAME", "adm3_name", "NAME_3", "NAM_3", "name"]
    
    for col in candidates:
        if col in gdf.columns:
            logger.info(f"  Using name column: {col}")
            return col
    return None


def normalize_name(name: str) -> str:
    """Normalize admin unit names for fuzzy matching."""
    if pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Remove common prefixes
    prefixes = ["prefecture de ", "prefecture ", "sous-prefecture de ", "sous-prefecture "]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Remove accents
    accent_map = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'ô': 'o', 'ö': 'o',
        'û': 'u', 'ü': 'u',
        'ç': 'c', 'î': 'i', 'ï': 'i'
    }
    for accented, plain in accent_map.items():
        name = name.replace(accented, plain)
    
    # Remove punctuation and extra spaces
    name = ''.join(c if c.isalnum() else ' ' for c in name)
    name = ' '.join(name.split())
    
    return name


def build_admin_h3_map(engine, data_cfg: Dict, features_cfg: Dict) -> Dict[str, Dict]:
    """
    Build mapping from admin units (admin1, admin2, admin3) to H3 cells.
    Includes admin3 support for IOM sub-prefecture mapping.
    """
    logger.info("Building Admin→H3 mapping...")
    
    admin_map = {
        "admin1": {"by_pcode": {}, "by_name": {}},
        "admin2": {"by_pcode": {}, "by_name": {}},
        "admin3": {"by_pcode": {}, "by_name": {}}
    }
    
    # Load H3 grid
    h3_gdf = gpd.read_postgis(
        f"SELECT h3_index, geometry FROM {SCHEMA}.features_static",
        engine,
        geom_col="geometry"
    )
    h3_gdf["h3_index"] = h3_gdf["h3_index"].astype("int64")
    
    # Load population data for weighting
    inspector = inspect(engine)
    if inspector.has_table(POPULATION_TABLE, schema=SCHEMA):
        pop_df = pd.read_sql(
            f"SELECT h3_index, year, pop_count FROM {SCHEMA}.{POPULATION_TABLE}",
            engine
        )
        pop_df["h3_index"] = pop_df["h3_index"].astype("int64")
        
        # Use most recent year's population
        latest_year = pop_df["year"].max()
        pop_latest = pop_df[pop_df["year"] == latest_year][["h3_index", "pop_count"]]
        h3_gdf = h3_gdf.merge(pop_latest, on="h3_index", how="left")
        h3_gdf["pop_count"] = h3_gdf["pop_count"].fillna(0)
    else:
        logger.warning(f"Population table {SCHEMA}.{POPULATION_TABLE} not found. Using uniform weights.")
        h3_gdf["pop_count"] = 1.0
    
    geodetic_crs = features_cfg["spatial"]["crs"]["geodetic"]
    h3_gdf = h3_gdf.to_crs(geodetic_crs)
    
    # Process each admin level
    admin_levels = ["admin1", "admin2", "admin3"]
    
    for level in admin_levels:
        path_key = f"{level}_path"
        admin_path = Path(data_cfg["admin_boundaries"][path_key])
        
        if not admin_path.exists():
            logger.warning(f"  {level} boundary file not found: {admin_path}")
            continue
        
        logger.info(f"  Loading {level} boundaries from {admin_path.name}...")
        admin_gdf = gpd.read_file(admin_path)
        admin_gdf = admin_gdf.to_crs(geodetic_crs)
        
        pcode_col = _infer_pcode_col(admin_gdf, level)
        name_col = _infer_name_col(admin_gdf, level)
        
        if not pcode_col and not name_col:
            logger.error(f"  Cannot identify pcode or name columns for {level}. Skipping.")
            continue
        
        # Spatial join
        joined = gpd.sjoin(
            h3_gdf[["h3_index", "pop_count", "geometry"]],
            admin_gdf,
            how="inner",
            predicate="intersects"
        )
        
        if pcode_col:
            for pcode, group in joined.groupby(pcode_col):
                admin_map[level]["by_pcode"][str(pcode)] = group["h3_index"].tolist()
        
        if name_col:
            for name, group in joined.groupby(name_col):
                normalized = normalize_name(name)
                admin_map[level]["by_name"][normalized] = group["h3_index"].tolist()
        
        logger.info(f"  Mapped {level}: {len(admin_map[level]['by_pcode'])} pcodes")
    
    return admin_map


def enrich_features_static_admins(engine, data_cfg, features_cfg):
    """
    Assigns Admin 1, 2, and 3 memberships to H3 cells based on LARGEST AREAL OVERLAP.
    FIXED: Handles index preservation during merge to avoid KeyError.
    """
    logger.info("Enriching features_static with admin names (Area-Weighted)...")
    
    # 1. Load H3 Grid
    h3_gdf = gpd.read_postgis(f"SELECT h3_index, geometry FROM {SCHEMA}.{STATIC_TABLE}", engine, geom_col="geometry")
    if h3_gdf.empty:
        logger.warning("No H3 grid found in features_static.")
        return
    
    # Set index initially to align with updates later
    h3_gdf.set_index("h3_index", inplace=True)
    
    levels = ["admin1", "admin2", "admin3"]
    admin_cols = features_cfg.get("spatial", {}).get("admin_columns", {})
    
    for level in levels:
        name_col = admin_cols.get(level)
        if not name_col:
            continue
            
        admin_path = ROOT_DIR / data_cfg["admin_boundaries"].get(f"{level}_path", "")
        if not admin_path.exists():
            continue
            
        logger.info(f"  Processing {level} (Source: {admin_path.name})...")
        admin_gdf = gpd.read_file(admin_path)
        
        if admin_gdf.crs != h3_gdf.crs:
            admin_gdf = admin_gdf.to_crs(h3_gdf.crs)

        # --- CORE LOGIC ---
        
        # 1. Spatial Join (find all admins that touch each cell)
        # Result index is still 'h3_index' here
        joined = gpd.sjoin(h3_gdf, admin_gdf[[name_col, "geometry"]], how="inner", predicate="intersects")
        
        # FIX 1: Reset index so 'h3_index' becomes a regular column.
        # This prevents the subsequent merge from deleting the H3 IDs.
        joined = joined.reset_index()
        
        # 2. Merge to get admin polygon geometry (for area calc)
        # Note: We merge on the name column
        joined = joined.merge(admin_gdf[[name_col, "geometry"]], on=name_col, suffixes=('', '_admin'))
        
        # 3. Calculate Intersection Area
        joined['overlap_area'] = joined['geometry'].intersection(joined['geometry_admin']).area
        
        # 4. Sort by H3 Index and Area (Largest first)
        # FIX 2: Sort by "h3_index" (the column), not "index"
        joined = joined.sort_values(by=["h3_index", "overlap_area"], ascending=[True, False])
        
        # 5. Dedup: Group by H3 Index and take the first (largest area)
        # FIX 3: Group by "h3_index"
        winner = joined.groupby("h3_index").first()
        
        # Map the winner back to the master dataframe
        # Since 'winner' and 'h3_gdf' both have h3_index as index, this aligns perfectly
        h3_gdf[level] = winner[name_col]
        
        logger.info(f"    Mapped {level}: {h3_gdf[level].notna().sum()} / {len(h3_gdf)} cells")

    # 3. Write updates to Database
    logger.info("Updating database records...")
    df_upload = h3_gdf[levels].reset_index()
    
    with engine.begin() as conn:
        conn.execute(text("CREATE TEMP TABLE admin_updates (h3_index BIGINT, admin1 TEXT, admin2 TEXT, admin3 TEXT)"))
        
        # Chunked insert for stability
        chunk_size = 5000
        for i in range(0, len(df_upload), chunk_size):
            chunk = df_upload.iloc[i : i + chunk_size]
            data_dict = chunk.to_dict(orient='records')
            conn.execute(
                text("INSERT INTO admin_updates VALUES (:h3_index, :admin1, :admin2, :admin3)"),
                data_dict
            )
            
        conn.execute(text(f"""
            UPDATE {SCHEMA}.{STATIC_TABLE} t
            SET admin1 = u.admin1,
                admin2 = u.admin2,
                admin3 = u.admin3
            FROM admin_updates u
            WHERE t.h3_index = u.h3_index
        """))
        conn.execute(text("DROP TABLE admin_updates"))
        
    logger.info("✓ features_static admin enrichment complete.")


def distribute_iom(engine, admin_map: Dict[str, Dict], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Disaggregate IOM DTM data from 'iom_dtm_raw' to H3 grid.
    
    UPDATES:
    - Reads from iom_dtm_raw (fetch_iom.py output)
    - Maps 'reporting_date' -> date
    - Maps 'individuals' -> iom_displacement_sum
    """
    logger.info(f"Disaggregating IOM data from {IOM_SOURCE_TABLE}...")
    
    inspector = inspect(engine)
    if not inspector.has_table(IOM_SOURCE_TABLE, schema=SCHEMA):
        logger.warning(f"Table {SCHEMA}.{IOM_SOURCE_TABLE} not found. Skipping IOM.")
        return pd.DataFrame()
    
    # Query adapted to new schema from fetch_iom.py
    query = f"""
        SELECT 
            reporting_date as date,
            admin1_pcode,
            admin1_name,
            admin2_pcode,
            admin2_name,
            individuals as count
        FROM {SCHEMA}.{IOM_SOURCE_TABLE}
        WHERE reporting_date BETWEEN '{start_date}' AND '{end_date}'
          AND individuals > 0
    """
    
    iom_df = pd.read_sql(query, engine)
    
    if iom_df.empty:
        logger.warning("No IOM records found in date range.")
        return pd.DataFrame()
    
    logger.info(f"  Loaded {len(iom_df):,} IOM records")
    
    # Load population weights for distribution
    pop_df = pd.read_sql(
        f"SELECT h3_index, year, pop_count FROM {SCHEMA}.{POPULATION_TABLE}",
        engine
    )
    latest_year = pop_df["year"].max()
    pop_weights = pop_df[pop_df["year"] == latest_year].set_index("h3_index")["pop_count"].to_dict()
    
    def _get_h3_list(level: str, pcode: str, name: str) -> List[int]:
        if pd.notna(pcode) and str(pcode) in admin_map[level]["by_pcode"]:
            return admin_map[level]["by_pcode"][str(pcode)]
        if pd.notna(name):
            normalized = normalize_name(name)
            if normalized in admin_map[level]["by_name"]:
                return admin_map[level]["by_name"][normalized]
        return []
    
    disagg_records = []
    unmapped_count = 0
    
    for _, row in iom_df.iterrows():
        date = row["date"]
        count = row["count"]
        
        # MAPPING STRATEGY:
        # IOM "admin2" (sub-prefecture) -> Maps to WBG "admin3"
        # IOM "admin1" (prefecture) -> Maps to WBG "admin2"
        
        h3_list = []
        
        # Try Admin 2 (Sub-Prefecture) first
        if pd.notna(row["admin2_pcode"]) or pd.notna(row["admin2_name"]):
            h3_list = _get_h3_list("admin3", row["admin2_pcode"], row["admin2_name"])
            
        # Fallback to Admin 1 (Prefecture)
        if not h3_list and (pd.notna(row["admin1_pcode"]) or pd.notna(row["admin1_name"])):
            h3_list = _get_h3_list("admin2", row["admin1_pcode"], row["admin1_name"])
            
        if not h3_list:
            unmapped_count += 1
            continue
            
        # Population weighting
        h3_pops = {h3: pop_weights.get(h3, 1.0) for h3 in h3_list}
        total_pop = sum(h3_pops.values())
        
        if total_pop > 0:
            for h3_idx, pop in h3_pops.items():
                disagg_records.append({
                    "h3_index": h3_idx,
                    "date": date,
                    "iom_displacement_sum": count * (pop / total_pop)
                })
        else:
            # Uniform fallback
            weight = 1.0 / len(h3_list)
            for h3_idx in h3_list:
                disagg_records.append({
                    "h3_index": h3_idx,
                    "date": date,
                    "iom_displacement_sum": count * weight
                })

    if unmapped_count > 0:
        logger.warning(f"  Could not map {unmapped_count:,} IOM records to H3.")

    if not disagg_records:
        return pd.DataFrame()

    # Aggregate by Cell + Date
    result_df = pd.DataFrame(disagg_records)
    result_df = result_df.groupby(["h3_index", "date"], as_index=False)["iom_displacement_sum"].sum()
    result_df["h3_index"] = result_df["h3_index"].astype("int64")
    result_df["date"] = pd.to_datetime(result_df["date"])
    
    logger.info(f"  Disaggregated to {len(result_df):,} H3-date records")
    return result_df


def run(configs: Dict, engine):
    """
    Execute spatial disaggregation pipeline.
    ONLY runs IOM disaggregation. IPC has been removed.
    """
    logger.info("=" * 60)
    logger.info("SPATIAL DISAGGREGATION (IOM → H3)")
    logger.info("=" * 60)
    
    if isinstance(configs, tuple):
        data_cfg, features_cfg = configs[0], configs[1]
    else:
        data_cfg = configs.get("data", configs)
        features_cfg = configs.get("features", configs)
    
    start_date = data_cfg["global_date_window"]["start_date"]
    end_date = data_cfg["global_date_window"]["end_date"]
    
    try:
        # 0. Enrich features_static with admin names for downstream aggregations
        enrich_features_static_admins(engine, data_cfg, features_cfg)

        # 1. Build Map
        admin_map = build_admin_h3_map(engine, data_cfg, features_cfg)
        
        # 2. Disaggregate IOM
        iom_h3 = distribute_iom(engine, admin_map, start_date, end_date)
        
        if not iom_h3.empty:
            logger.info(f"Uploading {len(iom_h3):,} IOM H3 records...")
            upload_to_postgis(
                engine,
                iom_h3,
                "iom_displacement_h3",
                SCHEMA,
                primary_keys=["h3_index", "date"]
            )
        else:
            logger.warning("No IOM data produced.")
        
        logger.info("=" * 60)
        logger.info("✓ DISAGGREGATION COMPLETE")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Spatial disaggregation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        cfg = load_configs()
        engine = get_db_engine()
        run(cfg, engine)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if "engine" in locals():
            engine.dispose()
