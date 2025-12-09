"""
process_terrain.py
==================
Vectorized Terrain Processing.
CRITICAL FIX (Phase 5):
- CRS Alignment: Forces H3 Grid to match DEM projection before rasterization.
- Prevents "empty stats" bugs caused by projection mismatches.
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

def rasterize_h3_grid(gdf, template_ds):
    """
    Rasterize H3 geometries to match the DEM resolution.
    CRITICAL: Ensures Vector CRS matches Raster CRS.
    """
    logger.info("  Rasterizing H3 Grid to match DEM...")
    
    # 1. CRS Check & Reprojection
    # We must match the raster's projection exactly, otherwise they won't overlap.
    if template_ds.crs:
        raster_crs = template_ds.crs
        if gdf.crs != raster_crs:
            logger.info(f"    Reprojecting H3 Grid from {gdf.crs} to {raster_crs}...")
            gdf = gdf.to_crs(raster_crs)
    else:
        logger.warning("    DEM has no CRS defined! Assuming match with Grid.")

    h3_indices = gdf['h3_index'].values
    # Map pixel value back to H3 Index
    # We use i+1 as the 'id' in the raster, because 0 is usually background
    h3_map = {i+1: h3 for i, h3 in enumerate(h3_indices)}
    
    # Generator for rasterization
    shapes = ((geom, i+1) for i, geom in enumerate(gdf.geometry))
    
    burned = features.rasterize(
        shapes=shapes,
        out_shape=template_ds.shape,
        transform=template_ds.transform,
        fill=0,
        dtype=rasterio.int32
    )
    
    return burned, h3_map

def calculate_zonal_stats_vectorized(dem_path, slope_path, tri_path, engine):
    """Calculate terrain statistics using vectorized raster operations."""
    logger.info("Calculating Zonal Stats (Vectorized)...")
    logger.info("  Ensuring database schema integrity...")
    
    # Force connection reset
    engine.dispose()
    
    # Add primary key - separate transaction
    try:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{STATIC_TABLE} ADD PRIMARY KEY (h3_index);"))
    except Exception:
        pass
    
    # Add columns - separate transaction  
    try:
        with engine.begin() as conn:
            conn.execute(text(f"""
                ALTER TABLE {SCHEMA}.{STATIC_TABLE} 
                ADD COLUMN IF NOT EXISTS elevation_mean FLOAT,
                ADD COLUMN IF NOT EXISTS slope_mean FLOAT,
                ADD COLUMN IF NOT EXISTS terrain_ruggedness_index FLOAT;
            """))
    except Exception as e:
        logger.warning(f"  Column addition failed (may already exist): {e}")
    
    # Checkpoint - separate read transaction
    with engine.connect() as conn:
        count = conn.execute(text(f"""
            SELECT COUNT(*) FROM {SCHEMA}.{STATIC_TABLE} 
            WHERE terrain_ruggedness_index IS NOT NULL
        """)).scalar()
        
        if count and count > 0:
            logger.info(f"  ✓ Terrain statistics already exist for {count} cells. Skipping calculation.")
            return

    # Load H3 Grid
    logger.info("  Loading H3 Grid geometries...")
    gdf = gpd.read_postgis(
        f"SELECT h3_index, geometry FROM {SCHEMA}.{STATIC_TABLE}", 
        engine, 
        geom_col='geometry'
    )
    
    if gdf.empty:
        logger.warning("No H3 cells found in features_static table.")
        return

    # Open DEM to create the Master H3 Mask
    if not Path(dem_path).exists():
        logger.error(f"DEM file not found at {dem_path}")
        return

    with rasterio.open(dem_path) as src:
        # This function now handles the CRS reprojection internally
        h3_mask, h3_map = rasterize_h3_grid(gdf, src)

    # Process each raster
    results = []
    raster_inputs = {
        "elevation_mean": dem_path,
        "slope_mean": slope_path,
        "terrain_ruggedness_index": tri_path
    }

    for col_name, r_path in raster_inputs.items():
        if not Path(r_path).exists():
            logger.warning(f"  Missing raster {r_path}, skipping {col_name}")
            continue

        logger.info(f"  Processing {col_name}...")
        with rasterio.open(r_path) as src:
            data = src.read(1)
            
            # Handle NoData
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            
            # Flatten arrays for vectorized grouping
            flat_mask = h3_mask.ravel()
            flat_data = data.ravel()
            
            # Filter where we have a valid H3 ID (mask > 0)
            valid_pixels = flat_mask > 0
            
            valid_mask = flat_mask[valid_pixels]
            valid_data = flat_data[valid_pixels]
            
            # Create DataFrame for GroupBy
            df_pixels = pd.DataFrame({'id': valid_mask, 'val': valid_data})
            
            # Remove masked values (NoData in raster)
            if np.ma.is_masked(valid_data):
                # .data gives raw, .mask gives boolean. We want where mask is False.
                # If valid_data is a MaskedArray, valid_data.mask is the mask.
                # However, flattening often converts to standard array with fill values if not careful.
                # If it is indeed a MaskedArray:
                real_data_mask = ~valid_data.mask
                df_pixels = df_pixels[real_data_mask]
            
            # Calculate Mean per H3 ID
            means = df_pixels.groupby('id')['val'].mean()
            
            # Map ID back to H3 Index
            mapped_means = means.index.map(h3_map)
            
            series = pd.Series(means.values, index=mapped_means, name=col_name)
            results.append(series)

    # Merge and upload
    logger.info("  Merging results...")
    if not results:
        logger.warning("No terrain statistics calculated.")
        return

    final_df = pd.concat(results, axis=1)
    final_df.index.name = 'h3_index'
    final_df = final_df.reset_index()
    
    logger.info(f"  Uploading {len(final_df)} rows...")
    upload_to_postgis(engine, final_df, STATIC_TABLE, SCHEMA, ['h3_index'])
    logger.info("  ✓ Terrain processing complete.")

def main():
    """Entry point for standalone execution."""
    engine = None
    try:
        engine = get_db_engine()
        data_cfg, feat_cfg, _ = load_configs()
        
        dem_path = PATHS["data_proc"] / "copernicus_dem_90m.tif"
        slope_path = PATHS["data_proc"] / "slope_car.tif"
        tri_path = PATHS["data_proc"] / "tri_car.tif"
        
        calculate_zonal_stats_vectorized(dem_path, slope_path, tri_path, engine)
        
    except Exception as e:
        logger.error(f"Terrain processing failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    main()