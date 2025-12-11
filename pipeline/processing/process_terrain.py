"""
process_terrain.py
==================
Vectorized Terrain Processing.
OPTIMIZATION: 
- Memory Safe: Processes H3 grid in spatial chunks to prevent OOM errors.
- CRS Alignment: Forces H3 Grid to match DEM projection.
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from sqlalchemy import text
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis, PATHS

SCHEMA = "car_cewp"
STATIC_TABLE = "features_static"
CHUNK_SIZE = 50000  # Process 50k cells at a time

def iter_h3_grid_chunks(engine):
    """Yields chunks of the H3 grid."""
    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE}")).scalar()
        
    logger.info(f"  Processing {total:,} cells in chunks of {CHUNK_SIZE:,}...")
    
    for offset in range(0, total, CHUNK_SIZE):
        query = f"""
            SELECT h3_index, geometry 
            FROM {SCHEMA}.{STATIC_TABLE} 
            ORDER BY h3_index 
            LIMIT {CHUNK_SIZE} OFFSET {offset}
        """
        gdf = gpd.read_postgis(query, engine, geom_col='geometry')
        if not gdf.empty:
            if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
            yield gdf

def rasterize_chunk(gdf, template_ds):
    """
    Rasterize a specific H3 chunk.
    """
    # CRS Check
    if template_ds.crs:
        if gdf.crs != template_ds.crs:
            gdf = gdf.to_crs(template_ds.crs)

    # Use h3_index as the value (mapped to 1..N for unique ID in raster)
    # We create a local map: {local_id: real_h3_index}
    h3_indices = gdf['h3_index'].values
    local_ids = np.arange(1, len(h3_indices) + 1, dtype=np.int32)
    h3_map = dict(zip(local_ids, h3_indices))
    
    shapes = ((geom, lid) for geom, lid in zip(gdf.geometry, local_ids))
    
    burned = features.rasterize(
        shapes=shapes,
        out_shape=template_ds.shape,
        transform=template_ds.transform,
        fill=0,
        dtype=rasterio.int32
    )
    
    return burned, h3_map

def calculate_chunk_stats(burned, h3_map, raster_inputs):
    """
    Calculate stats for a single chunk of H3 cells against all rasters.
    """
    chunk_results = []
    
    # Flatten the mask for this chunk
    flat_mask = burned.ravel()
    valid_pixels = flat_mask > 0
    
    if not np.any(valid_pixels):
        return pd.DataFrame()

    valid_mask_ids = flat_mask[valid_pixels]
    
    # Base DataFrame for this chunk's results
    # We will merge raster stats onto this
    # Start with unique IDs found in the raster intersection
    unique_ids = np.unique(valid_mask_ids)
    res_df = pd.DataFrame({'local_id': unique_ids})
    res_df['h3_index'] = res_df['local_id'].map(h3_map)
    
    for col_name, r_path in raster_inputs.items():
        if not Path(r_path).exists(): continue
        
        with rasterio.open(r_path) as src:
            data = src.read(1)
            # Handle NoData
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            
            flat_data = data.ravel()
            valid_data = flat_data[valid_pixels]
            
            # DF for grouping
            df_pixels = pd.DataFrame({'id': valid_mask_ids, 'val': valid_data})
            
            # Filter masked values
            if np.ma.is_masked(valid_data):
                df_pixels = df_pixels[~valid_data.mask]
            
            # GroupBy Mean
            means = df_pixels.groupby('id')['val'].mean().reset_index()
            means.columns = ['local_id', col_name]
            
            res_df = res_df.merge(means, on='local_id', how='left')

    return res_df.drop(columns=['local_id'])

def calculate_zonal_stats_vectorized(dem_path, slope_path, tri_path, engine):
    logger.info("Calculating Zonal Stats (Memory Safe Chunking)...")
    
    # Schema check: ONLY add columns, do NOT try to add a Primary Key again.
    with engine.begin() as conn:
        # REMOVED: conn.execute(text(f"ALTER TABLE {SCHEMA}.{STATIC_TABLE} ADD PRIMARY KEY (h3_index);"))
        conn.execute(text(f"""
            ALTER TABLE {SCHEMA}.{STATIC_TABLE} 
            ADD COLUMN IF NOT EXISTS elevation_mean FLOAT,
            ADD COLUMN IF NOT EXISTS slope_mean FLOAT,
            ADD COLUMN IF NOT EXISTS terrain_ruggedness_index FLOAT;
        """))

    # Raster Config
    raster_inputs = {
        "elevation_mean": dem_path,
        "slope_mean": slope_path,
        "terrain_ruggedness_index": tri_path
    }
    
    # Validate Rasters
    if not Path(dem_path).exists():
        logger.error(f"DEM not found: {dem_path}")
        return

    # Process Chunks
    with rasterio.open(dem_path) as dem_src:
        for gdf_chunk in iter_h3_grid_chunks(engine):
            
            # 1. Rasterize this chunk's polygons
            burned, h3_map = rasterize_chunk(gdf_chunk, dem_src)
            
            # 2. Compute stats
            df_results = calculate_chunk_stats(burned, h3_map, raster_inputs)
            
            # 3. Upload immediately
            if not df_results.empty:
                df_results['h3_index'] = df_results['h3_index'].astype('int64')
                upload_to_postgis(engine, df_results, STATIC_TABLE, SCHEMA, ['h3_index'])
    
    logger.info("Terrain processing complete.")

def main():
    engine = None
    try:
        engine = get_db_engine()
        load_configs() # Just to verify paths exist
        
        dem_path = PATHS["data_proc"] / "copernicus_dem_90m.tif"
        slope_path = PATHS["data_proc"] / "slope_car.tif"
        tri_path = PATHS["data_proc"] / "tri_car.tif"
        
        calculate_zonal_stats_vectorized(dem_path, slope_path, tri_path, engine)
        
    except Exception as e:
        logger.error(f"Terrain failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine: engine.dispose()

if __name__ == "__main__":
    main()