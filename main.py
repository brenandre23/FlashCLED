import sys
import argparse
import warnings
import time
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, Any

from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine

# Suppress Google Cloud Python version FutureWarning
warnings.filterwarnings(
    "ignore",
    message="You are using a Python version",
    category=FutureWarning,
    module="google.api_core._python_version_support"
)

# ---------------------------------------------------------
# 1) PATH SETUP & IMPORTS
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import logger, load_configs, get_db_engine

# --- Init ---
import init_db

# --- Ingestion ---
from pipeline.ingestion import (
    create_h3_grid,
    fetch_population,
    fetch_acled,
    fetch_dynamic_event,
    fetch_gee_server_side,
    fetch_mines,
    fetch_dem,
    fetch_grip4_roads,
    fetch_rivers,
    fetch_geoepr,
    fetch_epr_core,
    fetch_iom,
    fetch_settlements,
    ingest_economy,
    ingest_food_security,
)

# --- Processing ---
from pipeline.processing import (
    process_terrain,
    calculate_static_distances,
    spatial_disaggregation,
    feature_engineering as master_fe,
    calculate_epr_features,
)


# --- Modeling ---
from pipeline.modeling import (
    build_feature_matrix,
    train_models,
    generate_predictions,
)


# ---------------------------------------------------------
# 2) UTILITIES & CONTEXTS
# ---------------------------------------------------------

@contextmanager
def pipeline_stage(name: str):
    """Context manager to log stage execution and timing."""
    start_time = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"▶ STARTING: {name}")
    logger.info(f"{'='*60}")
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"✔ COMPLETED: {name} ({duration:.2f}s)")
    except Exception as e:
        logger.error(f"❌ FAILED: {name} ({time.time() - start_time:.2f}s)")
        raise e


def get_date_window(args: argparse.Namespace, data_config: Dict[str, Any]) -> Tuple[datetime.date, datetime.date]:
    """Resolve date window from CLI args or YAML config."""
    s_str = args.start_date
    e_str = args.end_date

    if not s_str or not e_str:
        dyn_conf = data_config.get("global_date_window", {})
        s_str = s_str or dyn_conf.get("start_date", "2000-01-01")
        e_str = e_str or dyn_conf.get("end_date", datetime.now().strftime("%Y-%m-%d"))

    try:
        start_date = datetime.strptime(s_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(e_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")

    if end_date < start_date:
        raise ValueError(f"end_date ({end_date}) is before start_date ({start_date}).")

    return start_date, end_date


# ---------------------------------------------------------
# 3) PIPELINE ORCHESTRATOR
# ---------------------------------------------------------

class CEWPPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.engine: Optional[Engine] = None
        self.configs: Dict[str, Any] = {}
        
        # Load Configs
        data_cfg, features_cfg, models_cfg = load_configs()
        self.configs = {"data": data_cfg, "features": features_cfg, "models": models_cfg}
        
        # Resolve Dates
        self.start_date, self.end_date = get_date_window(args, data_cfg)

    def setup(self):
        """Initialize DB connection and schema."""
        self.engine = get_db_engine()
        
        if self.args.reset_schema:
            self._reset_schema()

        # Init DB (Extensions, Tables, H3 types)
        init_db.run()

    def _reset_schema(self, schema="car_cewp"):
        logger.warning(f"🚨 RESET REQUESTED: Dropping schema {schema} CASCADE")
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS h3;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS h3_postgis;"))
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE;"))
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        logger.info(f"✅ Schema reset complete: {schema}")

    def _require_tables(self, tables: list, schema="car_cewp"):
        insp = inspect(self.engine)
        missing = [t for t in tables if not insp.has_table(t, schema=schema)]
        if missing:
            raise RuntimeError(f"Missing required tables in {schema}: {missing}")

    def run_static_phase(self):
        if self.args.skip_static:
            logger.info("Skipping Phase 1: Static Ingestion")
            return

        with pipeline_stage("PHASE 1: STATIC INGESTION"):
            # Grid & Pop
            create_h3_grid.main()
            self._require_tables(["features_static"])
            fetch_population.main()

            # Physical Geography
            fetch_dem.run(self.configs, self.engine)
            process_terrain.main()

            # Hydrology & Infrastructure
            fetch_rivers.run(self.configs, self.engine)
            fetch_grip4_roads.main()
            fetch_settlements.main()

            # Mines & EPR
            fetch_mines.main()
            fetch_geoepr.run()
            fetch_epr_core.run(self.configs, self.engine)

            # Distances
            calculate_static_distances.main()

    def run_dynamic_phase(self):
        if self.args.skip_dynamic:
            logger.info("Skipping Phase 2: Dynamic Ingestion")
            return

        with pipeline_stage("PHASE 2: DYNAMIC INGESTION"):
            # Conflict Events
            fetch_acled.run(self.configs, self.engine)
            fetch_dynamic_event.run(self.configs, self.engine)

            # Socio-Economic
            ingest_food_security.run(self.configs, self.engine)
            ingest_economy.main()  # Uses main() wrapper in source
            fetch_iom.main()       # Uses main() wrapper in source

            # Spatial Disaggregation
            logger.info(">> Spatial Disaggregation (Admin -> H3)")
            spatial_disaggregation.run(self.configs, self.engine)

            # Environment (GEE)
            fetch_gee_server_side.run(self.configs, self.engine)

    def run_feature_engineering_phase(self):
        if self.args.skip_features:
            logger.info("Skipping Phase 3: Feature Engineering")
            return

        with pipeline_stage("PHASE 3: FEATURE ENGINEERING"):
            # 3.1 Master Feature Script
            master_fe.run()

            # 3.2 EPR Features (Sidecar)
            calculate_epr_features.main()

    def run_modeling_phase(self):
        if self.args.skip_modeling:
            logger.info("Skipping Phase 4: Modeling")
            return

        with pipeline_stage("PHASE 4: MODELING"):
            # 4.1 Build ABT
            build_feature_matrix.run(
                self.configs, 
                self.engine, 
                str(self.start_date), 
                str(self.end_date)
            )

            # 4.2 Train Models
            train_models.run()

            # 4.3 Generate Predictions (Per Horizon)
            horizons = self.configs["models"].get("horizons", [{"name": "14d"}])
            for h in horizons:
                horizon_name = h["name"]
                logger.info(f"   > Predicting Horizon: {horizon_name}")
                # Hack: generate_predictions expects args via sys.argv
                sys.argv = ["generate_predictions.py", horizon_name]
                generate_predictions.main()

    def execute(self):
        try:
            self.setup()
            
            logger.info("=" * 60)
            logger.info("   ORCHESTRATING CEWP PIPELINE")
            logger.info(f"   Window: {self.start_date} -> {self.end_date}")
            logger.info("=" * 60)

            self.run_static_phase()
            self.run_dynamic_phase()
            self.run_feature_engineering_phase()
            self.run_modeling_phase()

            logger.info("\n" + "=" * 60)
            logger.info("✅ PIPELINE EXECUTION SUCCESSFUL")
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.warning("\nPipeline interrupted by user.")
            sys.exit(130)
        except Exception as e:
            logger.critical(f"\n❌ PIPELINE FAILED: {str(e)}", exc_info=True)
            sys.exit(1)
        finally:
            if self.engine:
                self.engine.dispose()


# ---------------------------------------------------------
# 4) ENTRY POINT
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CEWP Master Pipeline Orchestrator")
    
    # Dates
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    
    # Flags
    parser.add_argument("--reset-schema", action="store_true", help="Hard reset schema before running.")
    parser.add_argument("--skip-static", action="store_true", help="Skip static ingestion.")
    parser.add_argument("--skip-dynamic", action="store_true", help="Skip dynamic ingestion.")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering.")
    parser.add_argument("--skip-modeling", action="store_true", help="Skip modeling & predictions.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = CEWPPipeline(args)
    pipeline.execute()