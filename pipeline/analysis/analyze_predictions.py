"""
analyze_predictions.py
======================
Comprehensive analysis of conflict prediction models.
HYBRID VERSION: Robust Architecture (New) + Thesis Metrics (Old).

Features:
1. ROBUST: Drops NaN targets (future dates) to prevent crashes.
2. POISSON: Calculates Mean Poisson Deviance (Critical for Count Models).
3. CONDITIONAL: Calculates RMSE specifically for conflict events.
4. TEMPORAL: Breaks down performance by year.
5. FIXED: Auto-detects column names to prevent KeyErrors on 14d/1m horizon mismatches.
"""

import sys
import warnings
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_squared_error, mean_poisson_deviance
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
try:
    from utils import logger, PATHS
except ImportError:
    # Fallback if utils not found, for standalone debugging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("CEWP")
    PATHS = {"data_proc": Path("C:/Users/Brenan/Desktop/Thesis/Scratch")} # Adjust if needed

HORIZONS = {
    "14d": {"steps": 1, "label": "2-Week", "days": 14, "color": "#2ecc71"},
    "1m":  {"steps": 2, "label": "4-Week", "days": 28, "color": "#3498db"},
    "3m":  {"steps": 6, "label": "12-Week", "days": 84, "color": "#9b59b6"},
}
LEARNERS = ["xgboost", "lightgbm"]
OUTPUT_DIR = PATHS["data_proc"] / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Utils ---
def get_target_col(horizon): 
    return f"target_{HORIZONS[horizon]['steps']}_step"

# --- Metrics ---
def resolve_target_column(available_cols, steps: int) -> Optional[str]:
    """Pick the best available target column for a given horizon."""
    candidates = [
        f"target_fatalities_{steps}_step",
        f"target_{steps}_step",
        f"target_binary_{steps}_step",
        "target_fatalities",
        "fatalities_14d_sum",  # fallback
    ]
    for c in candidates:
        if c in available_cols:
            return c
    return None


def resolve_prediction_columns(available_cols, key):
    """
    Detect probability and fatalities columns with robust fallbacks.
    """
    # 1. Detect Probabilities
    prob_col = f"pred_prob_{key}"
    if prob_col not in available_cols:
        # Try finding anything with 'prob' AND the learner key
        cands = [c for c in available_cols if key in c and 'prob' in c]
        if cands:
            prob_col = cands[0]
        elif "conflict_prob" in available_cols:
            prob_col = "conflict_prob"
        elif "risk_score" in available_cols:
            prob_col = "risk_score"
        else:
            prob_col = None

    # 2. Detect Fatalities
    fatal_col = f"pred_fatalities_{key}"
    if fatal_col not in available_cols:
        # Try finding anything with 'fatal' AND the learner key
        cands = [c for c in available_cols if key in c and 'fatal' in c]
        if cands:
            fatal_col = cands[0]
        elif "predicted_fatalities" in available_cols:
            fatal_col = "predicted_fatalities"
        elif "fatalities" in available_cols:
            fatal_col = "fatalities"
        else:
            fatal_col = None
            
    return prob_col, fatal_col


def compute_metrics(df, horizon, key):
    """
    Computes performance metrics, ignoring rows where Ground Truth is unknown.
    Includes robust column detection to prevent KeyErrors.
    """
    available_cols = df.columns.tolist()

    # --- Target column detection ---
    steps = HORIZONS[horizon]["steps"]
    target_col = resolve_target_column(available_cols, steps)

    # --- 1. Dynamic Column Detection ---
    prob_col, fatal_col = resolve_prediction_columns(available_cols, key)

    # --- 2. Validation Check ---
    missing = []
    if not target_col: missing.append(f"target for {horizon}")
    if not prob_col: missing.append(f"Prob Col for {key}")
    if not fatal_col: missing.append(f"Fatal Col for {key}")

    if missing:
        logger.warning(f"  ⚠️ Skipping {key} ({horizon}). Missing columns: {missing}")
        return None

    # --- 3. ROBUST FILTER: Drop rows where data is missing ---
    valid_df = df.dropna(subset=[target_col, prob_col, fatal_col])

    if valid_df.empty:
        logger.warning(f"  ⚠️ No valid ground truth data for {horizon} {key} after dropna. Skipping.")
        return None

    y_true_binary = (valid_df[target_col] > 0).astype(int)
    y_true_fatal = valid_df[target_col]
    y_prob = valid_df[prob_col]
    y_pred_fatal = valid_df[fatal_col]

    # --- Standard Binary Metrics ---
    try: roc = roc_auc_score(y_true_binary, y_prob)
    except: roc = 0.5
        
    try: pr_auc = average_precision_score(y_true_binary, y_prob)
    except: pr_auc = 0.0
        
    brier = brier_score_loss(y_true_binary, y_prob)
    
    # --- Thesis Regression Metrics ---
    # 1. Global RMSE
    rmse = np.sqrt(mean_squared_error(y_true_fatal, y_pred_fatal))

    # 2. Conditional RMSE (Error ONLY when conflict actually happened)
    conflict_mask = y_true_binary == 1
    if conflict_mask.sum() > 0:
        rmse_cond = np.sqrt(mean_squared_error(
            y_true_fatal[conflict_mask], 
            y_pred_fatal[conflict_mask]
        ))
        mae_cond = np.mean(np.abs(y_true_fatal[conflict_mask] - y_pred_fatal[conflict_mask]))
    else:
        rmse_cond = np.nan
        mae_cond = np.nan

    # 3. Poisson Deviance
    try:
        y_t_safe = np.maximum(y_true_fatal, 0)
        y_p_safe = np.maximum(y_pred_fatal, 1e-6) # Avoid log(0)
        poisson_dev = mean_poisson_deviance(y_t_safe, y_p_safe)
    except Exception:
        poisson_dev = np.nan

    # 4. Recall at 10%
    k = int(len(valid_df) * 0.10)
    if k > 0 and y_true_binary.sum() > 0:
        top_k_idx = np.argsort(y_prob.values)[-k:]
        recall_top_10 = y_true_binary.iloc[top_k_idx].sum() / y_true_binary.sum()
    else:
        recall_top_10 = 0.0

    return {
        "ROC AUC": roc,
        "PR AUC": pr_auc,
        "Brier Score": brier,
        "RMSE": rmse,
        "RMSE (Events)": rmse_cond,
        "MAE (Events)": mae_cond,
        "Poisson Dev": poisson_dev,
        "Recall@10%": recall_top_10
    }

def analyze_temporal_performance(df, horizon, key):
    """
    Breaks down AUC by Year. Includes same column detection logic.
    """
    available_cols = df.columns.tolist()
    steps = HORIZONS[horizon]["steps"]
    target_col = resolve_target_column(available_cols, steps)
    
    # Quick detection
    prob_col, _ = resolve_prediction_columns(available_cols, key)

    if not target_col or not prob_col: return []
    
    valid_df = df.dropna(subset=[target_col, prob_col])
    if valid_df.empty: return []

    valid_df["year"] = valid_df["date"].dt.year
    results = []
    
    for year, group in valid_df.groupby("year"):
        y_true = (group[target_col] > 0).astype(int)
        if y_true.sum() < 2 or len(y_true) < 10: continue
            
        try:
            auc = roc_auc_score(y_true, group[prob_col])
            results.append({"Year": year, "Horizon": horizon, "Learner": key, "AUC": auc})
        except:
            pass
            
    return results

# --- Single-Pass Engine ---
def run_analysis_single_pass() -> Tuple[Dict, Dict, list]:
    results = {}
    sampled_data = {}
    temporal_results = []
    
    data_dir = PATHS["data_proc"]
    files = list(data_dir.glob("predictions_*.csv")) + list(data_dir.glob("predictions_*.parquet"))
    
    if not files:
        logger.error(f"No prediction files found in {data_dir}")
        return {}, {}, []

    # Load targets from feature matrix to join with predictions
    fm_path = data_dir / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.error(f"Feature matrix not found at {fm_path}; cannot attach targets for analysis.")
        return {}, {}, []
    df_labels_full = pd.read_parquet(fm_path)
    target_cols_all = ["h3_index", "date"] + [
        c for c in df_labels_full.columns
        if c.startswith("target_") or c.startswith("fatalities_")
    ]
    df_labels = df_labels_full[target_cols_all]
    if not np.issubdtype(df_labels["date"].dtype, np.datetime64):
        df_labels["date"] = pd.to_datetime(df_labels["date"])

    # Identify variants (baseline vs weighted)
    variants = set([""])
    for f in files:
        if "baseline" in f.name: variants.add("baseline")
        if "weighted" in f.name: variants.add("weighted")
    variants = sorted(list(variants))

    for horizon in HORIZONS:
        results[horizon] = {}
        sampled_data[horizon] = {}
        logger.info(f"\n=== Analyzing {HORIZONS[horizon]['label']} ===")
        
        for learner in LEARNERS:
            for variant in variants:
                suffix = f"_{variant}" if variant else ""
                # Try to find file - allow for partial matching if naming is inconsistent
                expected_fname_csv = f"predictions_{horizon}_{learner}{suffix}.csv"
                expected_fname_parquet = f"predictions_{horizon}_{learner}{suffix}.parquet"
                
                # Check for explicit file, or fallback to main predictions_{horizon}.csv if that's how you structure it
                fpath = data_dir / expected_fname_parquet
                if not fpath.exists():
                    fpath = data_dir / expected_fname_csv
                if not fpath.exists():
                    fpath = data_dir / f"predictions_{horizon}.parquet"
                if not fpath.exists():
                    fpath = data_dir / f"predictions_{horizon}.csv"
                if not fpath.exists():
                    continue
                
                key = f"{learner} ({variant})" if variant else learner
                # logger.info(f"  > Processing {key} from {fpath.name}...")
                
                try:
                    if fpath.suffix == ".parquet":
                        df = pd.read_parquet(fpath)
                    else:
                        df = pd.read_csv(fpath, parse_dates=["date"])
                        if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
                            df["date"] = pd.to_datetime(df["date"])
                except Exception as e:
                    logger.error(f"Could not read {fpath}: {e}")
                    continue

                # Attach targets from feature matrix
                if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
                    df["date"] = pd.to_datetime(df["date"])
                df = df.merge(df_labels, on=["h3_index", "date"], how="left")

                # OOS filter: evaluate only on held-out test period.
                # train_single_model.py hardcodes split_date = 2024-01-01.
                # Metrics computed on the full 2000-2025 range are contaminated by
                # in-sample predictions (inflates ROC-AUC to ~0.99).
                OOS_CUTOFF = pd.to_datetime("2024-01-01")
                df_oos = df[df["date"] >= OOS_CUTOFF]

                # 1. Compute Main Metrics (OOS period only)
                metric_res = compute_metrics(df_oos, horizon, key)
                if metric_res:
                    results[horizon][key] = metric_res
                    logger.info(f"    ✔ Metrics calculated for {key} (OOS: >={OOS_CUTOFF.date()})")

                # 2. Compute Temporal Metrics (full range — already groups by year)
                temp_res = analyze_temporal_performance(df, horizon, key)
                temporal_results.extend(temp_res)

                # 3. Smart Sampling for Plots
                target_col = resolve_target_column(df.columns.tolist(), HORIZONS[horizon]["steps"])
                prob_col, fatal_col = resolve_prediction_columns(df.columns.tolist(), key)

                if target_col and fatal_col:
                    df_valid = df.dropna(subset=[target_col, fatal_col])
                    if not df_valid.empty:
                        df_conf = df_valid[df_valid[target_col] > 0]
                        df_peace = df_valid[df_valid[target_col] == 0]
                        
                        if len(df_peace) > 10000:
                            df_peace = df_peace.sample(n=10000, random_state=42)
                        
                        df_small = pd.concat([df_conf, df_peace])
                        
                        # Store in sampled data
                        sampled_data[horizon][key] = df_small.copy()
                
                del df
                gc.collect()

    return results, sampled_data, temporal_results

# --- Plots ---
def plot_thesis_intensity_hexbin(data, save_path):
    horizon = "14d"
    if horizon not in data: return
    keys = list(data[horizon].keys())
    
    base_key = next((k for k in keys if "baseline" in k), keys[0] if keys else None)
    new_key = next((k for k in keys if "weighted" in k), keys[1] if len(keys)>1 else None)
    
    if not base_key or not new_key: return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    configs = [(axes[0], data[horizon][base_key], base_key, "Baseline (Ridge)"),
               (axes[1], data[horizon][new_key], new_key, "New Model (Poisson)")]
    
    for ax, df, key, title in configs:
        # Robust column fetch for plotting
        avail = df.columns.tolist()
        target_col = resolve_target_column(avail, HORIZONS[horizon]["steps"])
        _, fatal_col = resolve_prediction_columns(avail, key)
        
        if not target_col or not fatal_col: continue
        
        mask = df[target_col] > 0
        y_true = df.loc[mask, target_col]
        y_pred = df.loc[mask, fatal_col]
        if len(y_true) < 5: continue

        hb = ax.hexbin(y_true, y_pred, gridsize=30, cmap='inferno', bins='log', mincnt=1, xscale='log', yscale='log')
        ax.plot([1, 5000], [1, 5000], 'w--', alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual Fatalities (Log)"); ax.set_xlim(1, 2000); ax.set_ylim(0.01, 2000)
        cb = fig.colorbar(hb, ax=ax); cb.set_label('Density (log10)')

    axes[0].set_ylabel("Predicted Fatalities (Log)")
    plt.suptitle("Thesis Intensity Result: Ridge vs Poisson", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_fatality_scatter(data, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx]
        if horizon not in data: 
            ax.set_visible(False)
            continue
            
        ax.plot([0.1, 5000], [0.1, 5000], 'k--', alpha=0.3)
        for learner_key, df in data[horizon].items():
            # Robust column fetch for plotting
            avail = df.columns.tolist()
            target_col = resolve_target_column(avail, HORIZONS[horizon]["steps"])
            _, fatal_col = resolve_prediction_columns(avail, learner_key)
            
            if not target_col or not fatal_col: continue
            
            y_pred = df[fatal_col].values; y_true = df[target_col].values
            color = "#e74c3c" if "weighted" in learner_key else ("#3498db" if "baseline" in learner_key else "gray")
            marker = "x" if "weighted" in learner_key else "o"
            ax.scatter(y_true, y_pred, alpha=0.4, s=15, label=learner_key, color=color, marker=marker, edgecolors='none')
            
        ax.set_title(f"{HORIZONS[horizon]['label']}"); ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.1, 2000); ax.set_ylim(0.1, 2000); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_horizon_decay(results, save_path):
    """
    Cross-horizon accuracy decay chart.
    Shows how ROC-AUC and PR-AUC drop off from 14d → 1m → 3m,
    with one line per learner. Thesis-quality output.
    """
    horizon_labels = [HORIZONS[h]["label"] for h in HORIZONS]
    horizon_keys = list(HORIZONS.keys())
    horizon_days = [HORIZONS[h]["days"] for h in HORIZONS]

    # Collect per-learner, per-horizon metrics
    learner_roc = {}
    learner_pr  = {}
    all_learners = set()
    for h in horizon_keys:
        for learner, r in results.get(h, {}).items():
            if r is None:
                continue
            all_learners.add(learner)
            learner_roc.setdefault(learner, {})[h] = r.get("ROC AUC")
            learner_pr.setdefault(learner, {})[h]  = r.get("PR AUC")

    if not all_learners:
        return

    # Compute no-skill PR-AUC baseline from 14d results (prevalence)
    # Use the first learner that has 14d data as a proxy
    noskill_pr = None
    for lrn in all_learners:
        if "14d" in learner_pr.get(lrn, {}):
            # Approximate: no-skill PR-AUC ≈ event prevalence
            # We don't store prevalence directly, so mark as None and skip the line
            break

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"]
    markers = ["o", "s", "^", "D", "v"]

    for ax, metric_dict, ylabel, title in [
        (axes[0], learner_roc, "ROC-AUC",  "Onset Detection — ROC-AUC by Horizon"),
        (axes[1], learner_pr,  "PR-AUC",   "Onset Detection — PR-AUC by Horizon"),
    ]:
        for i, learner in enumerate(sorted(all_learners)):
            ys = [metric_dict.get(learner, {}).get(h) for h in horizon_keys]
            # Only plot if at least two horizons have data
            valid = [(x, y) for x, y in zip(horizon_days, ys) if y is not None]
            if len(valid) < 2:
                continue
            xs_v, ys_v = zip(*valid)
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            ax.plot(xs_v, ys_v, marker=m, color=c, linewidth=2, markersize=7,
                    label=learner, zorder=3)
            # Annotate each point
            for xv, yv in zip(xs_v, ys_v):
                ax.annotate(f"{yv:.3f}", (xv, yv),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=c)

        ax.set_xticks(horizon_days)
        ax.set_xticklabels(horizon_labels, fontsize=10)
        ax.set_xlabel("Forecast Horizon", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="lower left")

    plt.suptitle("Forecast Accuracy Decay Across Horizons", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved horizon decay chart → {save_path.name}")


def build_comparison_table(results):
    rows = []
    for horizon, learners in results.items():
        for learner, r in learners.items():
            if r is None: continue
            rows.append({
                "Horizon": HORIZONS[horizon]["label"], 
                "Learner": learner,
                "ROC_AUC": r.get("ROC AUC"), 
                "PR_AUC": r.get("PR AUC"),
                "Recall_10pct": r.get("Recall@10%"), 
                "Poisson_Dev": r.get("Poisson Dev"),
                "RMSE": r.get("RMSE"),
                "RMSE_Events": r.get("RMSE (Events)"),
                "MAE_Events": r.get("MAE (Events)")
            })
    
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Horizon", "Learner"])

def main():
    logger.info("="*80); logger.info("CONFLICT EARLY WARNING - OPTIMIZED ANALYSIS"); logger.info("="*80)
    results, sampled_data, temporal_res = run_analysis_single_pass()
    
    # Check if we have any results
    has_results = any(len(res) > 0 for res in results.values())
    if not has_results: 
        logger.warning("No valid results found to analyze.")
        return
    
    # 1. Main Table
    df_comp = build_comparison_table(results)
    if not df_comp.empty:
        print("\nSUMMARY METRICS:\n", df_comp.to_string(index=False))
        df_comp.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
        
    # 2. Temporal Table
    temporal_results = pd.DataFrame(temporal_res)
    if not temporal_results.empty:
        temporal_results.to_csv(OUTPUT_DIR / "temporal_auc_by_year.csv", index=False)
        logger.info("Saved temporal analysis to temporal_auc_by_year.csv")
    
    # 3. Plots
    logger.info("Generating plots from in-memory samples...")
    plot_thesis_intensity_hexbin(sampled_data, OUTPUT_DIR / "thesis_intensity.png")
    plot_fatality_scatter(sampled_data, OUTPUT_DIR / "fatality_scatter.png")
    plot_horizon_decay(results, OUTPUT_DIR / "horizon_accuracy_decay.png")

    logger.info(f"Done. Outputs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
