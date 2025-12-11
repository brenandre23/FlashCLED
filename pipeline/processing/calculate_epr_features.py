"""
calculate_epr_features.py
========================
Purpose: TIME-VARYING EPR features mapped onto the 14-day modeling spine.

LOGIC CHANGES:
1. UMBRELLA EXCLUSION: Explicitly filters out "Northern groups" and other umbrella terms
   to prevent double-counting with disaggregated subgroups (e.g., Runga, Goula).
2. TEMPORAL FILTER: Excludes groups that ceased to exist before 2000.
3. EFFICIENCY: Calculates stats at (H3, Year) level before broadcasting to 14-day spine.
4. ROBUSTNESS: Strict Int64 H3 typing and IDEMPOTENT upserts.
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import h3.api.basic_int as h3
from h3 import LatLngPoly
from sqlalchemy import text
from pathlib import Path

# --- Import Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis, ensure_h3_int64

# --- Configuration & Constants ---
SCHEMA = "car_cewp"
POLY_TABLE = "geoepr_polygons"
CORE_TABLE = "epr_core"
TARGET_TABLE = "temporal_features"

# Groups to exclude entirely to prevent double-counting/multicollinearity
EXCLUDED_GROUP_NAMES = [
    "Northern groups",  # Major umbrella group in CAR, overlaps with Runga/Goula/etc.
    "Muslims",          # Religious umbrella, often redundant with ethnic markers
]

# Numeric mapping for Mean Status calculation
# Based on political power access
STATUS_SCORES = {
    'DOMINANT': 4,
    'SENIOR PARTNER': 3,
    'JUNIOR PARTNER': 2,
    'POWERLESS': 1,
    'DISCRIMINATED': 0,
    'SELF-EXCLUSION': 1, # Treated similar to powerless in terms of access
    'IRRELEVANT': -2     # Special case, often filtered out or treated as noise
}

# -------------------------------------------------------------
# 1. Data Loading & Filtering
# -------------------------------------------------------------

def load_and_filter_polygons(engine):
    """
    Loads ethnic polygons and applies UMBRELLA EXCLUSION logic.
    """
    logger.info("Loading GeoEPR polygons...")
    
    # Handle column naming variations (group_name vs group)
    try:
        q_cols = text(f"SELECT * FROM {SCHEMA}.{POLY_TABLE} LIMIT 0")
        with engine.connect() as conn:
            cols = pd.read_sql(q_cols, conn).columns
        
        name_col = 'group_name' if 'group_name' in cols else '"group"'
        
        query = f"""
            SELECT gwgroupid, {name_col} as group_name, geometry, "to" as to_year
            FROM {SCHEMA}.{POLY_TABLE}
            WHERE geometry IS NOT NULL
        """
        gdf = gpd.read_postgis(query, engine, geom_col="geometry")
        
    except Exception as e:
        logger.error(f"Failed to load polygons: {e}")
        raise

    initial_count = len(gdf)
    
    # --- LOGIC 1: Exclude Umbrella Groups ---
    # Filter by name
    gdf = gdf[~gdf['group_name'].isin(EXCLUDED_GROUP_NAMES)].copy()
    
    # --- LOGIC 2: Temporal Validity ---
    # Exclude groups that ceased existing before the study period (2000)
    # "to_year" in GeoEPR indicates when the polygon validity ends.
    gdf = gdf[gdf['to_year'] >= 2000].copy()
    
    logger.info(f"✓ Filtered Polygons: {len(gdf)}/{initial_count} kept.")
    logger.info(f"  (Removed 'Northern groups', 'Muslims', and pre-2000 entities)")
    
    return gdf.to_crs(epsg=4326)


def load_epr_core_status(engine, start_year=2000, end_year=2025):
    """
    Loads yearly status data for the modeling window.
    """
    logger.info(f"Loading EPR Core status ({start_year}-{end_year})...")
    
    query = f"""
        SELECT gwgroupid, year, status, status_numeric
        FROM {SCHEMA}.{CORE_TABLE}
        WHERE year BETWEEN {start_year} AND {end_year}
    """
    df = pd.read_sql(query, engine)
    return df


def load_temporal_spine_keys(engine, start_date, end_date):
    """
    Get unique (h3_index, date) pairs to broadcast features onto.
    """
    logger.info("Loading target temporal spine keys...")
    query = f"""
        SELECT h3_index, date 
        FROM {SCHEMA}.{TARGET_TABLE}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    df = pd.read_sql(query, engine)
    
    # Ensure H3 Types
    df["h3_index"] = df["h3_index"].apply(ensure_h3_int64).astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    
    return df

# -------------------------------------------------------------
# 2. Spatial Mapping (Polygon -> H3)
# -------------------------------------------------------------

def map_groups_to_h3(gdf, resolution=5):
    """
    Performs H3 Polyfill to map each ethnic group to H3 cells.
    Returns: DataFrame [gwgroupid, h3_index]
    """
    logger.info(f"Mapping ethnic polygons to H3 (Res {resolution})...")
    
    mapping_rows = []
    
    for _, row in gdf.iterrows():
        gid = row['gwgroupid']
        geom = row['geometry']
        
        try:
            cells = set()
            if geom.geom_type == 'Polygon':
                exterior = [(y, x) for x, y in geom.exterior.coords]
                holes = [[(y, x) for x, y in i.coords] for i in geom.interiors]
                cells.update(h3.polygon_to_cells(LatLngPoly(exterior, *holes), resolution))
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    exterior = [(y, x) for x, y in poly.exterior.coords]
                    holes = [[(y, x) for x, y in i.coords] for i in poly.interiors]
                    cells.update(h3.polygon_to_cells(LatLngPoly(exterior, *holes), resolution))
            
            for cell in cells:
                # Ensure signed int64
                h3_int = ensure_h3_int64(cell)
                if h3_int is not None:
                    mapping_rows.append({'gwgroupid': gid, 'h3_index': h3_int})
                    
        except Exception as e:
            continue

    if not mapping_rows:
        # It's possible no polygons map if they are small or outside boundaries
        logger.warning("No H3 cells mapped from Ethnic Polygons.")
        return pd.DataFrame(columns=['gwgroupid', 'h3_index'])
        
    df_map = pd.DataFrame(mapping_rows)
    df_map['gwgroupid'] = df_map['gwgroupid'].astype(int)
    df_map['h3_index'] = df_map['h3_index'].astype('int64')
    
    logger.info(f"✓ Mapped {len(gdf)} groups to {len(df_map)} cell-group pairs.")
    return df_map

# -------------------------------------------------------------
# 3. Aggregation Logic
# -------------------------------------------------------------

def calculate_shannon_entropy(status_series):
    """Calculates Shannon Entropy (diversity) of status codes."""
    if status_series.empty:
        return 0.0
    counts = status_series.value_counts()
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)

def compute_yearly_h3_stats(df_map, df_status):
    """
    Joins Spatial Map with Temporal Status and aggregates by (H3, Year).
    """
    logger.info("Computing EPR statistics per (H3, Year)...")
    
    if df_map.empty:
        return pd.DataFrame()

    # Join: (H3, Group) x (Group, Year, Status) -> (H3, Group, Year, Status)
    merged = df_map.merge(df_status, on='gwgroupid', how='inner')
    
    if merged.empty:
        logger.warning("No overlap between spatial groups and status data.")
        return pd.DataFrame()

    # Pre-calculate boolean flags for fast summing
    merged['is_excluded'] = merged['status'].isin(['DISCRIMINATED', 'POWERLESS']).astype(int)
    merged['is_discriminated'] = (merged['status'] == 'DISCRIMINATED').astype(int)
    
    # Map status to numeric score
    merged['status_score'] = merged['status'].map(STATUS_SCORES).fillna(-2)

    # Aggregation
    grouped = merged.groupby(['h3_index', 'year'])
    
    stats = grouped.agg(
        ethnic_group_count=('gwgroupid', 'nunique'),
        epr_excluded_groups_count=('is_excluded', 'sum'),
        epr_discriminated_groups_count=('is_discriminated', 'sum'),
        epr_status_mean=('status_score', 'mean')
    ).reset_index()

    # Calculate Entropy (slower, apply separately)
    # We group by [h3, year] and apply entropy to 'status'
    entropy_df = grouped['status'].apply(calculate_shannon_entropy).reset_index(name='epr_status_entropy')
    
    # Merge entropy back
    stats = stats.merge(entropy_df, on=['h3_index', 'year'])
    
    # Alias for interpretability
    stats['epr_horizontal_inequality'] = stats['epr_status_entropy']

    return stats

# -------------------------------------------------------------
# 4. Main Execution
# -------------------------------------------------------------

def run():
    logger.info("="*60)
    logger.info("EPR FEATURE ENGINEERING (Robust Umbrella Exclusion)")
    logger.info("="*60)
    
    engine = None
    try:
        data_cfg, feat_cfg, _ = load_configs()
        engine = get_db_engine()
        
        # 1. Config
        start_date = data_cfg["global_date_window"]["start_date"]
        end_date = data_cfg["global_date_window"]["end_date"]
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        resolution = feat_cfg["spatial"]["h3_resolution"]
        
        # 2. Load Inputs
        gdf_poly = load_and_filter_polygons(engine)
        df_status = load_epr_core_status(engine, start_year, end_year)
        
        # 3. Spatial Map
        df_map = map_groups_to_h3(gdf_poly, resolution)
        
        # 4. Compute Yearly Stats
        df_yearly = compute_yearly_h3_stats(df_map, df_status)
        
        if df_yearly.empty:
            logger.warning("No yearly stats computed. Exiting.")
            return

        # 5. Broadcast to 14-Day Spine
        logger.info("Broadcasting yearly stats to 14-day temporal spine...")
        df_spine = load_temporal_spine_keys(engine, start_date, end_date)
        
        # Merge on (h3_index, year)
        final_df = df_spine.merge(df_yearly, on=['h3_index', 'year'], how='left')
        
        # Fill NaNs (Cells with no ethnic groups mapped)
        fill_values = {
            'ethnic_group_count': 0,
            'epr_excluded_groups_count': 0,
            'epr_discriminated_groups_count': 0,
            'epr_status_mean': -2.0,       # Irrelevant
            'epr_status_entropy': 0.0,
            'epr_horizontal_inequality': 0.0
        }
        final_df.fillna(fill_values, inplace=True)
        
        # --- FIX: Explicitly cast count columns to Integer to avoid "0.0" error ---
        int_cols = [
            'ethnic_group_count', 
            'epr_excluded_groups_count', 
            'epr_discriminated_groups_count'
        ]
        for c in int_cols:
            final_df[c] = final_df[c].astype(int)

        # 6. Database Upsert
        # Ensure schema
        cols_to_upload = [
            'h3_index', 'date', 
            'ethnic_group_count', 'epr_excluded_groups_count', 
            'epr_discriminated_groups_count', 'epr_status_mean', 
            'epr_status_entropy', 'epr_horizontal_inequality'
        ]
        
        # Create columns if missing (Schema Evolution)
        with engine.begin() as conn:
            for col in cols_to_upload[2:]: # Skip keys
                dtype = 'INTEGER' if 'count' in col else 'FLOAT'
                conn.execute(text(f"""
                    ALTER TABLE {SCHEMA}.{TARGET_TABLE} 
                    ADD COLUMN IF NOT EXISTS {col} {dtype}
                """))
        
        logger.info(f"Upserting {len(final_df):,} rows to {SCHEMA}.{TARGET_TABLE}...")
        
        # Chunked upload for memory safety
        chunk_size = 100000
        total = len(final_df)
        for i in range(0, total, chunk_size):
            chunk = final_df[cols_to_upload].iloc[i:i+chunk_size]
            upload_to_postgis(
                engine, 
                chunk, 
                TARGET_TABLE, 
                SCHEMA, 
                primary_keys=['h3_index', 'date']
            )
            print(f"  Processed {min(i+chunk_size, total)}/{total}", end='\r')
            
        logger.info("\n✅ EPR Features Calculation Complete.")

    except Exception as e:
        logger.critical(f"EPR Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine: engine.dispose()

def main():
    run()

if __name__ == "__main__":
    main()