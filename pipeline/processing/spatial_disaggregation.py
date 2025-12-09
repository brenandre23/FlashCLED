"""
spatial_disaggregation.py
=========================
Distributes Admin-Level Data (IOM DTM + IPC) to H3 Grid Cells.

FIXES APPLIED:
- [CRITICAL] Replaces "CREATE IF NOT EXISTS" with "DROP + CREATE" to enforce 
  Primary Key constraints required for ON CONFLICT upserts.
- [MAJOR] Includes IPC disaggregation logic.
"""

import sys
import re
import unicodedata
from pathlib import Path
import geopandas as gpd
import pandas as pd
from sqlalchemy import text

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, upload_to_postgis

SCHEMA = "car_cewp"

# ---------------------------------------------------------------------
# 1. HELPER: NORMALIZATION & MAPPING
# ---------------------------------------------------------------------
def normalize_name(name):
    """Standardizes Admin Names (lowercase, no accents, no suffixes)."""
    if pd.isna(name) or str(name).lower() in ["nan", "none", ""]:
        return "none"
        
    s = str(name).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s = re.sub(r"[-_]", " ", s)
    s = s.replace("sub prefecture", "").replace("prefecture", "")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

SUBPREFECTURE_MAPPING = {
    "kaga bandoro": "kaga bandoro", "mbres": "mbres", "ndele": "ndele",
    "bamingui": "bamingui", "bossangoa": "bossangoa", "markounda": "markounda",
    "nana bakassa": "nana bakassa", "nanga boguila": "nanga boguila",
    "bouca": "bouca", "batangafo": "batangafo", "kabo": "kabo",
    "bozum": "bozoum", "bozoum": "bozoum", "bocaranga": "bocaranga",
    "koui": "koui", "paoua": "paoua", "ngaoundaye": "ngaoundaye",
    "bimbo": "bimbo", "begoua": "begoua", "boali": "boali", "damara": "damara",
    "bogangolo": "bogangolo", "yaloke": "yaloke", "bossembele": "bossembele",
    "mbaiki": "mbaiki", "boda": "boda", "mongoumba": "mongoumba",
    "boganangone": "boganangone", "boganda": "boganda", "berberati": "berberati",
    "gamboula": "gamboula", "dede makouba": "dede makouba", "carnot": "carnot",
    "amada gaza": "amada gaza", "gadzi": "gadzi", "bouar": "bouar",
    "baboua": "baboua", "baoro": "baoro", "abba": "abba", "nola": "nola",
    "bambio": "bambio", "bayanga": "bayanga", "sibut": "sibut", "dekoa": "dekoa",
    "mala": "mala", "ndjoukou": "ndjoukou", "bambari": "bambari",
    "grimari": "grimari", "kouango": "kouango", "ippy": "ippy", "bakala": "bakala",
    "mobaye": "mobaye", "alindao": "alindao", "kembe": "kembe",
    "mingala": "mingala", "satema": "satema", "zangba": "zangba", "bria": "bria",
    "ouadda": "ouadda", "yalinga": "yalinga", "bangassou": "bangassou",
    "rafai": "rafai", "gambo": "gambo", "ouango": "ouango", "bakouma": "bakouma",
    "obo": "obo", "zemio": "zemio", "bambouti": "bambouti", "djema": "djehma",
    "djemah": "djehma", "birao": "birao", "ouanda djalle": "ouanda djalle",
    "bangui centre": "bangui", "bangui fleuve": "bangui", "bangui kagas": "bangui",
    "bangui rapide": "bangui", "bangui rapides": "bangui",
    "sido": "kabo", "nana outa": "nana bakassa",
    "ndim": "ngaoundaye", "taley": "paoua", "tale": "paoua",
    "ouandja": "ouadda", "ouandja kotto": "ouadda", "sam ouandja": "ouadda"
}

def _apply_mappings(df, col_name):
    if col_name in df.columns:
        df[col_name] = df[col_name].replace(SUBPREFECTURE_MAPPING)
    return df

# ---------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------
def load_admin_boundaries(data_config, target_crs):
    admin_cfg = data_config.get("admin_boundaries", {})
    
    path1 = ROOT_DIR / admin_cfg.get("admin1_path", "data/raw/wbgCAFadmin1.geojson")
    if not path1.exists(): raise FileNotFoundError(f"Admin 1 not found: {path1}")
    
    gdf1 = gpd.read_file(path1)
    col1 = next((c for c in gdf1.columns if c.lower() in ["adm1_name", "nam_1", "name_1", "admin1name"]), None)
    if not col1: raise ValueError(f"No Admin 1 name column found in {path1}")
    
    gdf1['admin1_key'] = gdf1[col1].apply(normalize_name)
    gdf1 = gdf1[['admin1_key', 'geometry']].to_crs(target_crs)
    gdf1['admin_level'] = 1

    path_sub = ROOT_DIR / "data/raw/wbgCAFadmin3.geojson"
    if not path_sub.exists(): raise FileNotFoundError(f"Sub-prefecture file not found: {path_sub}")
    
    gdf_sub = gpd.read_file(path_sub)
    col_sub = "adm2_name"
    if col_sub not in gdf_sub.columns:
        col_sub = next((c for c in gdf_sub.columns if c.lower() in ["nam_2", "name_2", "adm2_name"]), None)
    
    gdf_sub['admin2_key'] = gdf_sub[col_sub].apply(normalize_name)
    gdf_sub = gdf_sub[['admin2_key', 'geometry']].to_crs(target_crs)
    gdf_sub['admin_level'] = 2 
    
    return gdf1, gdf_sub

def load_h3_grid(engine, target_crs):
    query = f"SELECT h3_index, geometry FROM {SCHEMA}.features_static"
    gdf = gpd.read_postgis(query, engine, geom_col="geometry")
    return gdf.to_crs(target_crs)

def load_iom_data():
    path = PATHS['data_proc'] / "iom_displacement_data.csv"
    if not path.exists(): return None
    
    df = pd.read_csv(path)
    col_a1 = next((c for c in df.columns if "admin1" in c.lower()), None)
    col_a2 = next((c for c in df.columns if "admin2" in c.lower()), None)
    
    if not col_a1 or not col_a2: return None

    df['admin1_key'] = df[col_a1].apply(normalize_name)
    df['admin2_key'] = df[col_a2].apply(normalize_name)
    df = _apply_mappings(df, 'admin2_key')
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# ---------------------------------------------------------------------
# 3. SPATIAL DISAGGREGATION LOGIC
# ---------------------------------------------------------------------
def distribute_iom(engine, data_config, features_config):
    target_crs = features_config.get('spatial', {}).get('crs', {}).get('metric', 'EPSG:32634')
    logger.info(f"Starting IOM Disaggregation using {target_crs}...")
    
    # Load Data
    gdf_pref, gdf_subpref = load_admin_boundaries(data_config, target_crs)
    df_iom = load_iom_data()
    gdf_h3 = load_h3_grid(engine, target_crs)
    
    if df_iom is None: 
        logger.warning("No IOM data found.")
        return

    # Calculate Area for Density Checks
    gdf_h3['h3_area'] = gdf_h3.geometry.area

    dates = sorted(df_iom['date'].unique())
    logger.info(f"Processing {len(dates)} dates...")
    
    all_results = []
    unmatched_subpref = set()

    for d in dates:
        day_data = df_iom[df_iom['date'] == d].copy()
        
        # --- ATTEMPT 1: Match IOM Admin 2 -> Sub-prefecture Shapefile ---
        merged_sub = gdf_subpref.merge(day_data, on='admin2_key', how='inner')
        
        # Track what failed
        matched_keys = set(merged_sub['admin2_key'].unique())
        failed_rows = day_data[~day_data['admin2_key'].isin(matched_keys)].copy()
        unmatched_subpref.update(failed_rows['admin2_key'].unique())

        # --- ATTEMPT 2: Fallback to Prefecture (Admin 1) ---
        fallback_data = []
        if not failed_rows.empty:
            grp = failed_rows.groupby('admin1_key')['individuals'].sum().reset_index()
            merged_pref = gdf_pref.merge(grp, on='admin1_key', how='inner')
            fallback_data = [merged_pref]

        # Combine streams
        to_process = [merged_sub] + fallback_data
        combined_poly = pd.concat(to_process, ignore_index=True)
        
        if combined_poly.empty:
            continue

        # --- SPATIAL INTERSECTION ---
        # 1. Density = People / Polygon Area
        combined_poly['poly_area'] = combined_poly.geometry.area
        combined_poly['density'] = combined_poly['individuals'] / combined_poly['poly_area']
        
        # 2. Overlay H3 on Polygons
        intersection = gpd.overlay(gdf_h3, combined_poly, how='intersection')
        
        # 3. Fragment Pop = Density * Fragment Area
        intersection['frag_pop'] = intersection['density'] * intersection.geometry.area
        
        # 4. Sum up fragments per H3 cell
        daily_h3 = intersection.groupby('h3_index')['frag_pop'].sum().reset_index()
        daily_h3['date'] = d
        
        all_results.append(daily_h3)

    if unmatched_subpref:
        logger.warning(f"FALLBACK: {len(unmatched_subpref)} IOM Sub-prefectures mapped to Prefecture level.")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.rename(columns={'frag_pop': 'iom_displacement_sum'}, inplace=True)
        final_df['source'] = 'IOM_DTM'
        
        logger.info(f"Generated {len(final_df)} H3 records. Uploading...")
        
        # [CRITICAL FIX] Drop table to guarantee Primary Key creation
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.iom_displacement_h3"))
            conn.execute(text(f"""
                CREATE TABLE {SCHEMA}.iom_displacement_h3 (
                    h3_index BIGINT, 
                    date DATE, 
                    iom_displacement_sum FLOAT, 
                    source TEXT, 
                    PRIMARY KEY (h3_index, date)
                )
            """))
            
        upload_to_postgis(engine, final_df, "iom_displacement_h3", SCHEMA, ["h3_index", "date"])
        logger.info("IOM Disaggregation Complete.")
    else:
        logger.warning("No spatial data generated.")

# ---------------------------------------------------------------------
# [MAJOR-1 FIX] NEW: IPC DISAGGREGATION
# ---------------------------------------------------------------------
def disaggregate_ipc_to_h3(engine, data_config, features_config):
    logger.info("Starting IPC (Admin 1 -> H3) Disaggregation...")
    target_crs = features_config.get('spatial', {}).get('crs', {}).get('metric', 'EPSG:32634')
    
    # 1. Load Data
    gdf_pref, _ = load_admin_boundaries(data_config, target_crs)
    gdf_h3 = load_h3_grid(engine, target_crs)
    
    # Load IPC Data from DB
    ipc_query = f"SELECT date, admin1, phase FROM {SCHEMA}.ipc_phases"
    try:
        df_ipc = pd.read_sql(ipc_query, engine)
    except Exception as e:
        logger.warning(f"Could not load IPC data: {e}. Skipping IPC disaggregation.")
        return

    if df_ipc.empty:
        logger.warning("IPC table empty. Skipping.")
        return

    df_ipc['admin1_key'] = df_ipc['admin1'].apply(normalize_name)
    
    # 2. Join IPC to Admin Boundaries
    merged = gdf_pref.merge(df_ipc, on='admin1_key', how='inner')
    
    if merged.empty:
        logger.warning("No matches between IPC Admin names and Shapefile.")
        return

    # 3. Spatial Intersection
    logger.info("Performing spatial overlay (H3 x Admin1)...")
    intersection = gpd.overlay(gdf_h3, merged[['geometry', 'date', 'phase', 'admin1_key']], how='intersection')
    
    # 4. Aggregation (Max Phase per Cell)
    logger.info("Aggregating to H3...")
    final_df = intersection.groupby(['h3_index', 'date'])['phase'].max().reset_index()
    final_df.rename(columns={'phase': 'ipc_phase_class'}, inplace=True)
    
    # 5. Upload
    logger.info(f"Generated {len(final_df)} IPC H3 records. Uploading to 'ipc_h3'...")
    
    # [CRITICAL FIX] Drop table to guarantee Primary Key creation
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.ipc_h3"))
        conn.execute(text(f"""
            CREATE TABLE {SCHEMA}.ipc_h3 (
                h3_index BIGINT, 
                date DATE, 
                ipc_phase_class INTEGER, 
                PRIMARY KEY (h3_index, date)
            )
        """))
    
    upload_to_postgis(engine, final_df, "ipc_h3", SCHEMA, ["h3_index", "date"])
    logger.info("IPC Disaggregation Complete.")


def run(configs, engine):
    distribute_iom(engine, configs['data'], configs['features'])
    disaggregate_ipc_to_h3(engine, configs['data'], configs['features'])

def main():
    try:
        cfgs = load_configs()
        if isinstance(cfgs, tuple):
            configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
        else:
            configs = cfgs
        engine = get_db_engine()
        run(configs, engine)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()