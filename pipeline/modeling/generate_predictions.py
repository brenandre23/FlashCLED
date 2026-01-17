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

    for c in input_features:
        if c not in df.columns:
            df[c] = 0.0

    scaler = bundle.get("pca_scaler")
    X_scaled = scaler.transform(df[input_features].fillna(0).values)
    X_comps = pca.transform(X_scaled)

    pca_df = pd.DataFrame(X_comps, columns=bundle.get("pca_component_names"), index=df.index)
    return pd.concat([df, pca_df], axis=1)


def generate_single_run(horizon_name, learner_name):
    """WORKER FUNCTION"""
    logger.info(f"🚀 [Worker] Predicting: {horizon_name} | {learner_name}")

    filename = f"two_stage_ensemble_{horizon_name}_{learner_name}.pkl"
    model_path = PATHS["models"] / filename

    if not model_path.exists():
        logger.warning(f"⚠️ Model missing: {filename}. Skipping.")
        sys.exit(0)

    try:
        bundle = joblib.load(model_path)
        ensemble = bundle["ensemble"]
        bccp = bundle.get("bccp")
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
        sys.exit(1)

    df = pd.read_parquet(PATHS["data_proc"] / "feature_matrix.parquet")
    df = apply_pca_if_needed(df, bundle)

    probs, fatalities = ensemble.predict(df)

    pred_df = df[["h3_index", "date"]].copy()
    pred_df["horizon"] = horizon_name
    pred_df["learner"] = learner_name
    pred_df["conflict_prob"] = probs
    pred_df["predicted_fatalities"] = fatalities

    if bccp:
        # Transform Poisson count predictions to log-space for BCCP intervals, then back
        fatalities_log = np.log1p(fatalities)
        intervals = bccp.predict_intervals(fatalities_log)
        pred_df["fatalities_lower"] = np.expm1(np.maximum(0, intervals.lower))
        pred_df["fatalities_upper"] = np.expm1(np.maximum(0, intervals.upper))
        pred_df["risk_score"] = pred_df["fatalities_upper"]
    else:
        pred_df["fatalities_lower"] = 0
        pred_df["fatalities_upper"] = fatalities
        pred_df["risk_score"] = fatalities

    # Persist to disk (parquet) to avoid DB sync issues
    out_path = PATHS["data_proc"] / f"predictions_{horizon_name}_{learner_name}.parquet"
    pred_df.to_parquet(out_path, index=False)
    logger.info(f"✅ Saved predictions to {out_path}")

    # Also persist to Postgres for centralized access
    engine = get_db_engine()
    upload_to_postgis(
        engine,
        pred_df,
        table_name="predictions",
        schema=SCHEMA,
        primary_keys=["h3_index", "date", "horizon", "learner"],
    )
    logger.info(f"✅ Upserted predictions to {SCHEMA}.predictions ({len(pred_df):,} rows)")

    del df, ensemble, bundle
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
