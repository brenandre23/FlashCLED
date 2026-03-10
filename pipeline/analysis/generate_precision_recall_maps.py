"""
generate_precision_recall_maps.py
=================================
Generates dynamic diagnostic maps for thesis Figure 5.8 (SMenchen Feedback).

1. True Positives Map (Captured in Top 10%)
2. False Negatives Map (Missed outside Top 10%)
3. Relative Residual Map (% Error)
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

OUTPUT_DIR = PATHS["analysis"] / "research_diag"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load predictions and merge with ground truth."""
    logger.info("Loading predictions and targets...")
    
    # 1. Load Feature Matrix (Targets)
    fm_path = PATHS["data_proc"] / "feature_matrix.parquet"
    target_col = "target_fatalities_1_step"
    df_target = pd.read_parquet(fm_path, columns=["h3_index", "date", target_col])
    df_target["h3_index"] = df_target["h3_index"].apply(ensure_h3_int64)

    # 2. Load Predictions
    pred_path = PATHS["data_proc"] / "predictions_14d_xgboost.parquet"
    if not pred_path.exists():
        pred_path = PATHS["data_proc"] / "predictions_14d.parquet"
    
    df_pred = pd.read_parquet(pred_path)
    df_pred["h3_index"] = df_pred["h3_index"].apply(ensure_h3_int64)
    
    # Identify probability and prediction columns
    prob_col = "conflict_prob" if "conflict_prob" in df_pred.columns else "prob"
    pred_col = "predicted_fatalities" if "predicted_fatalities" in df_pred.columns else "y_pred"

    # 3. Merge
    df_pred["date"] = pd.to_datetime(df_pred["date"])
    df_target["date"] = pd.to_datetime(df_target["date"])
    df = df_pred.merge(df_target, on=["h3_index", "date"], how="inner")
    
    df["y_true"] = df[target_col].fillna(0)
    df["y_pred"] = df[pred_col].fillna(0)
    df["prob"] = df[prob_col].fillna(0)
    
    return df

def generate_geometry(h3_indices):
    """Generate Polygon geometries for H3 indices."""
    polys = []
    for h in h3_indices:
        try:
            h_str = h3.int_to_str(h)
            boundary_latlon = h3.cell_to_boundary(h_str)
            boundary_lonlat = [(p[1], p[0]) for p in boundary_latlon]
            polys.append(Polygon(boundary_lonlat))
        except:
            polys.append(None)
    return polys

def plot_diagnostic_maps(df):
    """Generates the three requested diagnostic maps."""
    
    # Calculate Top 10% threshold
    threshold = df["prob"].quantile(0.90)
    logger.info(f"Top 10% Probability Threshold: {threshold:.4f}")
    
    # Classify outcomes
    df["is_conflict"] = df["y_true"] > 0
    df["in_top_10"] = df["prob"] >= threshold
    
    # 1. True Positives (Conflict AND in Top 10%)
    df["is_tp"] = df["is_conflict"] & df["in_top_10"]
    # 2. False Negatives (Conflict AND NOT in Top 10%)
    df["is_fn"] = df["is_conflict"] & (~df["in_top_10"])
    # 3. Relative Residual
    df["rel_residual"] = (df["y_pred"] - df["y_true"]) / (df["y_true"] + 1)

    # Spatial Aggregation
    tp_agg = df.groupby("h3_index")["is_tp"].sum().reset_index()
    fn_agg = df.groupby("h3_index")["is_fn"].sum().reset_index()
    res_agg = df.groupby("h3_index")["rel_residual"].mean().reset_index()

    # Create GeoDataFrames
    def to_gdf(agg_df, col):
        # Only plot cells with activity to avoid clutter
        agg_df = agg_df[agg_df[col] != 0].copy()
        agg_df["geometry"] = generate_geometry(agg_df["h3_index"].values)
        gdf = gpd.GeoDataFrame(agg_df, geometry="geometry", crs="EPSG:4326").dropna()
        return gdf

    gdf_tp = to_gdf(tp_agg, "is_tp")
    gdf_fn = to_gdf(fn_agg, "is_fn")
    gdf_res = to_gdf(res_agg, "rel_residual")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # TP Map
    gdf_tp.plot(column="is_tp", cmap="Greens", ax=axes[0], legend=True, 
                legend_kwds={'label': "True Positive Count (Captured in Top 10%)"})
    axes[0].set_title("A. True Positives (Captured)", fontweight="bold")
    
    # FN Map
    gdf_fn.plot(column="is_fn", cmap="OrRd", ax=axes[1], legend=True,
                legend_kwds={'label': "False Negative Count (Missed outside Top 10%)"})
    axes[1].set_title("B. False Negatives (Missed)", fontweight="bold")
    
    # Relative Residual Map
    gdf_res.plot(column="rel_residual", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2], legend=True,
                 legend_kwds={'label': "Relative Bias (Error / Intensity)"})
    axes[2].set_title("C. Relative Spatial Bias", fontweight="bold")

    for ax in axes:
        ax.axis("off")

    out_path = OUTPUT_DIR / "rq3_spatial_performance_diag.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved diagnostic maps to {out_path}")
    plt.close()

if __name__ == "__main__":
    data = load_data()
    plot_diagnostic_maps(data)
