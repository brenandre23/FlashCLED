"""
generate_predictions.py
=======================
ROBUST PREDICTION ORCHESTRATOR.
Iterates through Horizons AND Learners via subprocesses.
"""
import sys
import argparse
import subprocess
import joblib
import gc
import pandas as pd
import numpy as np
from pathlib import Path

# --- Import Utils ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, PATHS, get_db_engine, load_configs, SCHEMA, upload_to_postgis


def apply_pca_if_needed(df, bundle):
    pca = bundle.get("pca")
    if not pca:
        return df

    input_features = bundle.get("pca_input_features")
    if not input_features:
        return df

    # Ensure all required inputs exist
    for c in input_features:
        if c not in df.columns:
            df[c] = 0.0

    # Fixed column order
    X_raw = df[input_features]

    # Robust reconstruction components
    scaler = bundle.get("pca_scaler")
    imputer = bundle.get("pca_imputer")
    pca_component_names = bundle.get("pca_component_names")
    
    logger.info(f"🎨 Reconstructing {len(pca_component_names)} PCA components in chunks...")
    
    pca_results = []
    chunk_size = 200000  # Process in smaller chunks to save RAM
    
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        X_chunk = X_raw.iloc[start:end].values
        
        # 1. Impute
        if imputer:
            X_chunk = imputer.transform(X_chunk)
        else:
            X_chunk = np.nan_to_num(X_chunk, nan=0.0)
            
        # 2. Scale
        X_chunk = scaler.transform(X_chunk)
        
        # 3. PCA
        X_comps = pca.transform(X_chunk)
        
        pca_results.append(
            pd.DataFrame(
                X_comps, 
                columns=pca_component_names, 
                index=df.index[start:end]
            )
        )
        
    pca_df = pd.concat(pca_results, axis=0)
    return pd.concat([df, pca_df], axis=1)


import traceback

def generate_single_run(horizon_name, learner_name):
    """WORKER FUNCTION"""
    logger.info(f"🚀 [Worker] Predicting: {horizon_name} | {learner_name}")

    try:
        filename = f"two_stage_ensemble_{horizon_name}_{learner_name}.pkl"
        model_path = PATHS["models"] / filename

        if not model_path.exists():
            logger.warning(f"⚠️ Model missing: {filename}. Skipping.")
            sys.exit(0)

        bundle = joblib.load(model_path)
        ensemble = bundle["ensemble"]
        bccp = bundle.get("bccp")

        df = pd.read_parquet(PATHS["data_proc"] / "feature_matrix.parquet")
        df = apply_pca_if_needed(df, bundle)

        probs, fatalities = ensemble.predict(df)
        
        # --- SAFETY CLIP (Prevent Metric Explosion) ---
        # Caps fatalities at 500 (beyond historical max for CAR H3 cells)
        fatalities = np.clip(fatalities, 0, 500)

        pred_df = df[["h3_index", "date"]].copy()
        pred_df["horizon"] = horizon_name
        pred_df["learner"] = learner_name
        pred_df["conflict_prob"] = probs
        pred_df["predicted_fatalities"] = fatalities

        if bccp:
            # Check if BCCP was trained on log scale
            use_log = getattr(bccp, "log_scale", True)
            
            if use_log:
                fatalities_input = np.log1p(fatalities)
                intervals = bccp.predict_intervals(fatalities_input)
                # Transform back to original scale
                pred_df["fatalities_lower"] = np.expm1(np.maximum(0, intervals.lower))
                pred_df["fatalities_upper"] = np.clip(np.expm1(np.maximum(0, intervals.upper)), 0, 1000)
            else:
                intervals = bccp.predict_intervals(fatalities)
                pred_df["fatalities_lower"] = np.maximum(0, intervals.lower)
                pred_df["fatalities_upper"] = np.clip(intervals.upper, 0, 1000)
            
            pred_df["risk_score"] = pred_df["fatalities_upper"]
        else:
            pred_df["fatalities_lower"] = 0
            pred_df["fatalities_upper"] = fatalities
            pred_df["risk_score"] = fatalities

        # --- VECTORIZED ADAPTIVE THRESHOLDING (Tiered Percentage Logic) ---
        # Calculate percentile rank per date group (0.0 = highest risk, 1.0 = lowest)
        pred_df["_risk_pct"] = pred_df.groupby("date")["risk_score"].rank(pct=True, ascending=False)
        
        # Assign Tiers
        # Critical: Top 5%, High: Top 15%, Elevated: Top 30%
        pred_df["risk_tier"] = pd.cut(
            pred_df["_risk_pct"],
            bins=[0, 0.05, 0.15, 0.30, 1.0],
            labels=["1_critical", "2_high", "3_elevated", "4_baseline"],
            include_lowest=True
        )
        
        # Map boolean priority flag to the "Critical" tier for dashboard compatibility
        pred_df["is_priority_target"] = pred_df["risk_tier"] == "1_critical"
        
        pred_df.drop(columns=["_risk_pct"], inplace=True)
        
        logger.info(f"📍 Adaptive Tiers: Assigned 5%/15%/30% risk buckets for horizon {horizon_name}")

        out_path = PATHS["data_proc"] / f"predictions_{horizon_name}_{learner_name}.parquet"
        pred_df.to_parquet(out_path, index=False)
        logger.info(f"✅ Saved predictions to {out_path}")

        engine = get_db_engine()
        upload_to_postgis(
            engine,
            pred_df,
            table_name="predictions",
            schema=SCHEMA,
            primary_keys=["h3_index", "date", "horizon", "learner"],
        )
        logger.info(f"✅ Upserted predictions to {SCHEMA}.predictions ({len(pred_df):,} rows)")

    except Exception as e:
        logger.error(f"❌ Worker crashed for {horizon_name}-{learner_name}: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    finally:
        if 'df' in locals(): del df
        if 'ensemble' in locals(): del ensemble
        gc.collect()


def run_orchestrator():
    """ORCHESTRATOR FUNCTION"""
    config = load_configs()
    horizons = [h['name'] for h in config['models']['horizons']]
    learners = list(config['models']['learners'].keys())

    logger.info(f"🎬 PREDICTION ORCHESTRATOR: {len(horizons)}H x {len(learners)}L")

    for h in horizons:
        for l in learners:
            logger.info(f"\n🔄 PREDICTING: {h} | {l}")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--horizon", h, "--learner", l]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                logger.error(f"❌ Failed: {h}-{l}")


def main():
    run_orchestrator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str)
    parser.add_argument("--learner", type=str)
    args = parser.parse_args()

    if args.horizon and args.learner:
        generate_single_run(args.horizon, args.learner)
    else:
        run_orchestrator()
