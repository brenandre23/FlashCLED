import subprocess
import sys
import logging
from pathlib import Path

# ---------------------------------------------------------
# Import Centralized Utilities
# ---------------------------------------------------------
# Get the absolute path to the 'scripts' folder
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir / "scripts"

# CRITICAL FIX: Use insert(0, ...) to force Python to look here FIRST
# This prevents conflicts if you have a library named 'utils' installed
sys.path.insert(0, str(scripts_dir))

try:
    from utils import load_configs, PATHS, logger
except ImportError as e:
    # Print the actual error to help debug if it persists
    print(f"CRITICAL ERROR: Could not import 'utils'. Python Error: {e}")
    print(f"Make sure {scripts_dir / 'utils.py'} exists.")
    sys.exit(1)

# ---------------------------------------------------------
# Pipeline Definition
# ---------------------------------------------------------
# Each script handles a single, focused responsibility.
PIPELINE_SCRIPTS = [
    "01_build_static_features.py",
    "02_fetch_acled.py",             
    "03_fetch_temporal_roads_HYBRID.py",
    "04_calculate_distances.py",     
    "05_fetch_iom.py",
    "06_calculate_terrain.py",
    "07_fetch_population.py",
    "08_build_feature_matrix.py",
]

def run_script(script_name):
    """
    Runs a single python script using subprocess.
    Uses centralized PATHS to locate the file.
    """
    # Robustly find the script
    script_path = PATHS["scripts"] / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
        
    logger.info(f"\n{'='*60}\n>>> RUNNING: {script_name}\n{'='*60}")
    
    # sys.executable ensures we use the same Python interpreter (virtualenv)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)], 
            capture_output=False, 
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"--- FAILED: {script_name} ---")
            logger.error(f"Exit Code: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Subprocess failed to launch {script_name}: {e}")
        return False
    
    logger.info(f"--- SUCCESS: {script_name} ---")
    return True

def main():
    """Main execution of the H3-optimized pipeline."""
    logger.info("STARTING H3-OPTIMIZED PIPELINE...")
    logger.info(f"Root Directory: {PATHS['root']}")

    # 1. Validation: Load Configs
    try:
        data_config, features_config, models_config = load_configs()
        logger.info("Configuration files loaded and validated.")
    except Exception as e:
        logger.critical(f"Pipeline aborted: Invalid configuration. {e}")
        sys.exit(1)

    # 2. Execution: Run Scripts sequentially
    for script in PIPELINE_SCRIPTS:
        success = run_script(script)
        if not success:
            logger.critical(f"Pipeline HALTED due to error in {script}.")
            sys.exit(1)
            
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY.")
    logger.info("All static features engineered and uploaded to PostGIS.")
    logger.info(f"Logs available in: {PATHS['logs']}")
    logger.info("="*60)

if __name__ == "__main__":
    main()