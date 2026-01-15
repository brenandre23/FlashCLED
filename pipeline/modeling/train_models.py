"""
train_models.py
===============
ROBUST ORCHESTRATOR.
Launches training tasks as separate system processes to guarantee memory cleanup.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, load_configs


def run(configs=None):
    if configs is None:
        _, _, configs = load_configs()

    horizons = configs["horizons"]
    learners = list(configs["learners"].keys())

    logger.info("=" * 60)
    logger.info("TRAINING ORCHESTRATOR (Process Isolation Mode)")
    logger.info(f"   Targets: {len(horizons)} horizons x {len(learners)} learners")
    logger.info("=" * 60)

    total_start = time.time()

    for h in horizons:
        h_name = h["name"]
        for l_name in learners:
            logger.info(f"\n[ORCHESTRATOR] Launching subprocess: {h_name} - {l_name}")
            start_time = time.time()

            cmd = [
                sys.executable,
                str(ROOT_DIR / "pipeline/modeling/train_single_model.py"),
                "--horizon",
                h_name,
                "--learner",
                l_name,
            ]

            result = subprocess.run(cmd)
            duration = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"Subprocess failed (Code {result.returncode}). Stopping pipeline.")
                sys.exit(1)
            else:
                logger.info(f"[ORCHESTRATOR] Subprocess finished in {duration/60:.1f}m. RAM clean.")

    logger.info(f"\n[ORCHESTRATOR] All models trained in {(time.time() - total_start)/60:.1f}m.")


if __name__ == "__main__":
    run()
