"""
analyze_spatial_residuals.py
============================
Generates spatial residual maps for thesis Figure 5.7.

Logic:
1. Loads 14d predictions and ground truth from feature matrix.
2. Calculates residuals (Predicted - Actual).
3. Aggregates spatially (Mean Residual per H3 cell).
4. Plots a choropleth map of spatial bias.

Output:
- analysis/spatial_residuals_14d.png
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import h3

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import logger, PATHS, ensure_h3_int64

OUTPUT_DIR = PATHS["analysis"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load predictions and merge with ground truth."""
    logger.info("Loading predictions and targets...")
    
    # 1. Load Feature Matrix (Targets)
    fm_path = PATHS["data_proc"] / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.error(f"Feature matrix not found: {fm_path}")
        return None
    
    target_col = "target_fatalities_1_step" # 14d horizon target
    cols = ["h3_index", "date", target_col]
    
    try:
        df_target = pd.read_parquet(fm_path, columns=cols)
        df_target["h3_index"] = df_target["h3_index"].apply(ensure_h3_int64)
    except Exception as e:
        logger.error(f"Failed to load feature matrix: {e}")
        return None

    # 2. Load Predictions
    pred_path = PATHS["data_proc"] / "predictions_14d_xgboost.parquet"
    if not pred_path.exists():
        # Fallback to generic
        pred_path = PATHS["data_proc"] / "predictions_14d.parquet"
        if not pred_path.exists():
            logger.error("No 14d predictions found.")
            return None
    
    logger.info(f"Loading predictions from {pred_path.name}")
    df_pred = pd.read_parquet(pred_path)
    
    if "h3_index" not in df_pred.columns:
        logger.error("Predictions missing h3_index")
        return None
        
    df_pred["h3_index"] = df_pred["h3_index"].apply(ensure_h3_int64)
    
    # Resolve prediction column
    pred_col = "pred_fatalities_xgboost"
    if pred_col not in df_pred.columns:
        if "predicted_fatalities" in df_pred.columns:
            pred_col = "predicted_fatalities"
        elif "pred_fatalities" in df_pred.columns:
            pred_col = "pred_fatalities"
        else:
            logger.error(f"Prediction column not found. Available: {df_pred.columns.tolist()}")
            return None

    # 3. Merge
    if not np.issubdtype(df_pred["date"].dtype, np.datetime64):
        df_pred["date"] = pd.to_datetime(df_pred["date"])
    if not np.issubdtype(df_target["date"].dtype, np.datetime64):
        df_target["date"] = pd.to_datetime(df_target["date"])
        
    df = df_pred.merge(df_target, on=["h3_index", "date"], how="inner")
    
    if df.empty:
        logger.warning("Merge result is empty (no matching dates/hexes).")
        return None
        
    df["y_true"] = df[target_col].fillna(0)
    df["y_pred"] = df[pred_col].fillna(0)
    
    return df

def generate_geometry(h3_indices):
    """Generate Polygon geometries for H3 indices."""
    polys = []
    for h in h3_indices:
        try:
            # h3-py v4 API
            h_str = h3.int_to_str(h)
            # cell_to_boundary returns ((lat, lon), ...)
            boundary_latlon = h3.cell_to_boundary(h_str)
            # Swap to (lon, lat) for Shapely/GeoJSON
            boundary_lonlat = [(p[1], p[0]) for p in boundary_latlon]
            polys.append(Polygon(boundary_lonlat))
        except Exception as e:
            # Only log the first error to avoid spamming
            if len(polys) == 0:
                logger.error(f"Geometry generation failed for {h}: {e}")
            polys.append(None)
    return polys

def plot_residuals(df):
    """Calculate mean residuals and plot map."""
    logger.info("Calculating spatial residuals...")
    
    # Calculate Residual: Pred - True (Positive = Over-prediction, Negative = Under-prediction)
    df["residual"] = df["y_pred"] - df["y_true"]
    
    # Filter for relevant cells: either true conflict OR predicted conflict > threshold
    # This prevents the map from being dominated by zero-zero correct negatives
    relevant_mask = (df["y_true"] > 0) | (df["y_pred"] > 0.05)
    df_filtered = df[relevant_mask].copy()
    
    if df_filtered.empty:
        logger.warning("No relevant conflict cells found for residual plotting.")
        return

    # Aggregate by H3
    gdf_agg = df_filtered.groupby("h3_index")["residual"].mean().reset_index()
    gdf_agg.rename(columns={"residual": "mean_residual"}, inplace=True)
    
    logger.info(f"Generating geometries for {len(gdf_agg)} active cells...")
    gdf_agg["geometry"] = generate_geometry(gdf_agg["h3_index"].values)
    gdf = gpd.GeoDataFrame(gdf_agg, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.dropna(subset=["geometry"])
    
    if gdf.empty:
        logger.error("No valid geometries created.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Fixed color scale limits as requested [-1.5, 1.5]
    vmin, vmax = -1.5, 1.5
    
    # Plot background for context (optional, or just white)
    # gdf.plot(ax=ax, color='lightgrey', alpha=0.1) 
    
    gdf.plot(
        column="mean_residual",
        cmap="coolwarm",  # Diverging: Blue (Under) to Red (Over)
        linewidth=0.1,
        edgecolor="black",
        legend=True,
        legend_kwds={'label': "Mean Residual (Pred - True)", 'shrink': 0.6},
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )
    
    ax.set_title("Spatial Residuals (Active Conflict Zones)\nBlue = Under-prediction | Red = Over-prediction", fontweight="bold")
    ax.axis("off")
    
    out_path = OUTPUT_DIR / "spatial_residuals_14d.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved residual map to {out_path}")
    plt.close()

def main():
    logger.info("Starting Spatial Residual Analysis...")
    df = load_data()
    if df is not None:
        plot_residuals(df)
    else:
        logger.warning("Skipping residual plotting due to missing data.")

if __name__ == "__main__":
    main()