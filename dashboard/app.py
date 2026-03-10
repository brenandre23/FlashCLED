"""
dashboard/app.py
================
FlashCLED: Production-Grade Conflict Early Warning Dashboard Backend.

Three-Map Architecture with Strict Data Lineage:
1. Predictions API - car_cewp.predictions ONLY
2. Temporal Features API - car_cewp.temporal_features ONLY  
3. Static Features API - car_cewp.features_static ONLY

No mixing of data sources. No mock/fallback data generation.

CRITICAL: All H3 indices returned to frontend use canonical H3 string format
(e.g., "855a5a1bfffffff") for Deck.gl H3HexagonLayer compatibility.

Run with: gunicorn -w 4 -b 0.0.0.0:8000 app:app
"""

import os
import sys
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, List, Dict, Any

from flask import Flask, jsonify, send_from_directory, Response, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sqlalchemy import text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
import h3

try:
    import orjson
except ImportError:
    import json as orjson
import json
import yaml

# --- Configuration & Environment ---

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

def load_aggregation_config():
    """Load scientific aggregation rules from research_diagnostic.yaml."""
    config_path = ROOT_DIR / "configs" / "research_diagnostic.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get("aggregation", {})
        except Exception as e:
            print(f"Error loading aggregation config: {e}")
    return {}

AGGREGATION_CONFIG = load_aggregation_config()
AGG_METHODS = AGGREGATION_CONFIG.get("methods", {})
AGG_THRESHOLD = AGGREGATION_CONFIG.get("partial_coverage_threshold", 0.5)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FlashCLED")

SCHEMA = os.environ.get("DB_SCHEMA", "car_cewp")
TOP_CONTRIBUTIONS_LIMIT = 3

# --- Feature Allowlists (Strict Validation) ---

TEMPORAL_FEATURES_ALLOWLIST = {
    # Environmental/Remote sensing
    "chirps_precip_anomaly", "era5_temp_anomaly", "era5_soil_moisture_anomaly",
    "ndvi_anomaly", "ntl_mean",
    # VIIRS Advanced
    "ntl_peak", "ntl_stale_days", "ntl_kinetic_delta",
    # Land Cover (Dynamic World)
    "landcover_grass", "landcover_trees", "landcover_crops", "landcover_bare", "landcover_built",
    # Conflict
    "fatalities_14d_sum", "fatalities_1m_lag", "protest_count_lag1",
    "riot_count_lag1", "regional_risk_score_lag1",
    # GDELT
    "gdelt_event_count", "gdelt_avg_tone", "gdelt_goldstein_mean", "gdelt_mentions_total",
    # Fusion / Interactions
    "cw_onset_amplifier", "cw_mass_casualty_risk", "cw_extraction_violence",
    "cw_pastoral_predation", "fusion_gold_signal", "fusion_fragmentation_confirmed",
    "fusion_escalation_momentum",
    # CrisisWatch & Regime Pillars
    "cw_score_local", "regime_parallel_governance", "regime_transnational_predation",
    "regime_guerrilla_fragmentation", "regime_ethno_pastoral_rupture", "narrative_velocity_lag1",
    # ACLED Hybrid Mechanisms
    "mech_gold_pivot_lag1", "mech_predatory_tax_lag1",
    "mech_factional_infighting_lag1", "mech_collective_punishment_lag1",
    # Market
    "price_maize", "price_rice", "price_oil", "price_sorghum", "price_cassava", "price_groundnuts",
    # Macroeconomics
    "gold_price_usd_lag1", "oil_price_usd_lag1", "sp500_index_lag1", "eur_usd_rate_lag1",
    # Displacement
    "iom_displacement_count_lag1",
    # EPR
    "epr_excluded_groups_count", "epr_discriminated_groups_count", "epr_status_mean", "ethnic_group_count",
    # Population
    "pop_log"
}

STATIC_FEATURES_ALLOWLIST = {
    # Distance features
    "dist_to_capital", "dist_to_border", "dist_to_city", "dist_to_road", "dist_to_river",
    "dist_to_market_km", "dist_to_diamond_mine", "dist_to_gold_mine", "dist_to_large_mine",
    "dist_to_controlled_mine", "dist_to_large_gold_mine",
    # Geographic
    "elevation_mean", "slope_mean", "terrain_ruggedness_index",
    # Admin (codes stored as admin1/admin2/admin3)
    "admin1", "admin2", "admin3"
}

# --- Database Connection Factory ---

def get_db_engine() -> Engine:
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5433")
    db_name = os.environ.get("DB_NAME", "thesis_db")
    db_user = os.environ.get("DB_USER", "postgres")
    db_pass = os.environ.get("DB_PASS", "")

    url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=40,
        pool_timeout=30,
        pool_pre_ping=True,
        connect_args={"options": f"-c search_path={SCHEMA},public"}
    )

# --- Flask Initialization ---

DASHBOARD_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")
CORS(app)

# =============================================================================
# H3 CONVERSION HELPERS (CRITICAL FOR DECK.GL RENDERING)
# =============================================================================

def to_h3_str(h3_val) -> Optional[str]:
    """
    Convert DB h3_index (BIGINT/int/decimal-string) to canonical H3 string.
    Required for Deck.gl H3HexagonLayer rendering.
    
    Returns lowercase hex string like "855a5a1bfffffff" or None if invalid.
    """
    if h3_val is None or pd.isna(h3_val):
        return None
    
    try:
        # Case 1: Already a string
        if isinstance(h3_val, str):
            s = h3_val.strip().lower()
            
            # Already a valid H3 hex string
            if h3.is_valid_cell(s):
                return s
            
            # Decimal string (e.g., "600844665696026624")
            if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                return h3.int_to_str(int(s))
            
            # Hex string without 0x prefix (try parsing as hex)
            try:
                return h3.int_to_str(int(s, 16))
            except ValueError:
                return None
        
        # Case 2: Integer or numpy integer
        if isinstance(h3_val, (int, np.integer)):
            return h3.int_to_str(int(h3_val))
        
        return None
        
    except Exception as e:
        logger.debug(f"H3 conversion failed for {h3_val}: {e}")
        return None


def to_h3_int(h3_val) -> Optional[int]:
    """
    Convert canonical H3 string or decimal string to BIGINT for DB queries.
    
    Accepts:
    - Canonical H3 string: "855a5a1bfffffff"
    - Decimal string: "600844665696026624"
    - Integer: 600844665696026624
    
    Returns integer suitable for PostgreSQL BIGINT queries.
    """
    if h3_val is None or pd.isna(h3_val):
        return None
    
    try:
        # Case 1: Already an integer
        if isinstance(h3_val, (int, np.integer)):
            return int(h3_val)
        
        # Case 2: String
        s = str(h3_val).strip().lower()
        
        # Valid H3 hex string -> convert to int
        if h3.is_valid_cell(s):
            return h3.str_to_int(s)
        
        # Decimal string
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        
        # Try parsing as hex
        try:
            return int(s, 16)
        except ValueError:
            return None
            
    except Exception as e:
        logger.debug(f"H3 int conversion failed for {h3_val}: {e}")
        return None


# Legacy helper - kept for compatibility but NOT used for Deck.gl output
def ensure_signed_h3(h3_val) -> Optional[int]:
    """
    LEGACY: Converts H3 formats to signed 64-bit int.
    DO NOT use for Deck.gl visualization output - use to_h3_str() instead.
    """
    try:
        if pd.isna(h3_val):
            return None
        
        val = 0
        if isinstance(h3_val, (int, np.integer)):
            val = int(h3_val)
        elif isinstance(h3_val, str):
            s = h3_val.strip()
            if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                val = int(s)
            elif h3.is_valid_cell(s.lower()):
                val = h3.str_to_int(s.lower())
            else:
                val = int(s, 16)
        else:
            return None

        if val > 0x7FFFFFFFFFFFFFFF:
            val = val - 0x10000000000000000
        return val
    except Exception:
        return None


def fetch_table_data(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Executes a SQL query and returns a Pandas DataFrame."""
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})
    except Exception as e:
        logger.error(f"SQL Error: {e}\nQuery: {query}")
        return pd.DataFrame()
    finally:
        engine.dispose()

def table_exists_check(table_name: str) -> bool:
    """Checks if a table exists in the schema."""
    query = f"SELECT to_regclass('{SCHEMA}.{table_name}')"
    df = fetch_table_data(query)
    return not df.empty and df.iloc[0, 0] is not None

def column_exists(table_name: str, column_name: str) -> bool:
    """Checks if a column exists in a table."""
    query = """
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = :s AND table_name = :t AND column_name = :c
    """
    df = fetch_table_data(query, {"s": SCHEMA, "t": table_name, "c": column_name})
    return not df.empty

@lru_cache(maxsize=1)
def prediction_interval_columns() -> Dict[str, bool]:
    """Cache interval-column availability on predictions table."""
    return {
        "fatalities_lower": column_exists("predictions", "fatalities_lower"),
        "fatalities_upper": column_exists("predictions", "fatalities_upper"),
    }

# --- Endpoints: System & Health ---

@app.route("/")
def index():
    return send_from_directory(str(DASHBOARD_DIR), "index.html")

@app.route("/api/health")
def health_check():
    """Health check: Verifies DB connection and core tables."""
    status = {"status": "unhealthy", "checks": {}}
    try:
        fetch_table_data("SELECT 1")
        status["checks"]["database"] = "ok"
        status["checks"]["predictions"] = "found" if table_exists_check("predictions") else "missing"
        status["checks"]["features_static"] = "found" if table_exists_check("features_static") else "missing"
        status["checks"]["temporal_features"] = "found" if table_exists_check("temporal_features") else "missing"
        status["checks"]["h3_library"] = h3.__version__
        status["status"] = "healthy"
        return jsonify(status)
    except Exception as e:
        status["error"] = str(e)
        return jsonify(status), 503

# =============================================================================
# DATES ENDPOINTS (Separate by data source)
# =============================================================================

@app.route("/api/dates/predictions")
def get_prediction_dates():
    """Returns sorted ASCENDING unique dates from car_cewp.predictions."""
    if not table_exists_check("predictions"):
        return jsonify({
            "error": "predictions table missing",
            "table": f"{SCHEMA}.predictions"
        }), 500

    horizon = request.args.get("horizon")
    learner = request.args.get("learner")

    filters = []
    params = {}
    if horizon:
        filters.append("horizon = :h")
        params["h"] = horizon
    if learner:
        filters.append("learner = :l")
        params["l"] = learner

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"SELECT DISTINCT date FROM {SCHEMA}.predictions {where_clause} ORDER BY date ASC"
    df = fetch_table_data(query, params)
    
    if df.empty:
        return jsonify({"dates": [], "warning": "No dates found in predictions table"})
    
    # Ensure strict YYYY-MM-DD format to match cube keys
    return jsonify({"dates": [pd.to_datetime(d).strftime('%Y-%m-%d') for d in df['date'].tolist()]})

@app.route("/api/dates/temporal")
def get_temporal_dates():
    """Returns sorted ASCENDING unique dates from car_cewp.temporal_features."""
    if not table_exists_check("temporal_features"):
        return jsonify({
            "error": "temporal_features table missing",
            "table": f"{SCHEMA}.temporal_features"
        }), 500
    
    query = f"SELECT DISTINCT date FROM {SCHEMA}.temporal_features ORDER BY date ASC"
    df = fetch_table_data(query)
    
    if df.empty:
        return jsonify({"dates": [], "warning": "No dates found in temporal_features table"})
    
    # Ensure strict YYYY-MM-DD format
    return jsonify({"dates": [pd.to_datetime(d).strftime('%Y-%m-%d') for d in df['date'].tolist()]})

# =============================================================================
# MAP 1: PREDICTIONS (car_cewp.predictions ONLY)
# =============================================================================

@app.route("/api/predictions")
def get_predictions():
    """
    MAP 1 DATA SOURCE: Returns H3 or Admin prediction rows for visualization.
    
    Query Params:
    - date: YYYY-MM-DD
    - horizon: 14d | 1m | 3m
    - learner: xgboost | lightgbm
    - level: h3 | region | prefecture | subprefecture
    """
    if not table_exists_check("predictions"):
        return jsonify({"error": "predictions table missing"}), 500
    
    target_date = request.args.get('date')
    horizon = request.args.get('horizon', '14d')
    learner = request.args.get('learner')
    level = request.args.get('level', 'h3').lower()
    include_forecast = request.args.get('include_forecast_date', 'false').lower() in ("1", "true", "yes", "y")

    horizon_interval = {"14d": "14 days", "1m": "28 days", "2m": "56 days", "3m": "84 days"}.get(horizon)

    # Base filter logic
    where_filters = ["p.horizon = :h"]
    params = {"h": horizon}
    if learner:
        where_filters.append("p.learner = :l")
        params["l"] = learner
    
    if target_date:
        where_filters.append("p.date = :d")
        params["d"] = target_date
    else:
        # Get latest date subquery
        latest_subquery = f"SELECT MAX(date) FROM {SCHEMA}.predictions WHERE horizon = :h {'AND learner = :l' if learner else ''}"
        where_filters.append(f"p.date = ({latest_subquery})")

    where_clause = " AND ".join(where_filters)

    try:
        col_flags = prediction_interval_columns()
        lower_expr = "p.fatalities_lower" if col_flags["fatalities_lower"] else "NULL::double precision"
        upper_expr = "p.fatalities_upper" if col_flags["fatalities_upper"] else "NULL::double precision"

        if level == 'h3':
            query = f"""
                SELECT p.h3_index,
                       p.conflict_prob AS risk_score,
                       p.predicted_fatalities AS fatalities,
                       {lower_expr} AS fatalities_lower,
                       {upper_expr} AS fatalities_upper,
                       p.is_priority_target,
                       p.conflict_prob * p.predicted_fatalities AS expected_fatalities
                       {", (p.date + INTERVAL '" + horizon_interval + "') AS forecast_date" if include_forecast and horizon_interval else ""}
                FROM {SCHEMA}.predictions p
                WHERE {where_clause}
            """
            df = fetch_table_data(query, params)
            if df.empty: return jsonify([])
            
            df['hex'] = df['h3_index'].apply(to_h3_str)
            df = df.dropna(subset=['hex'])
            
            return jsonify([{
                "hex": row['hex'],
                "risk": float(row['risk_score']) if pd.notna(row['risk_score']) else None,
                "fatalities": float(row['fatalities']) if pd.notna(row['fatalities']) else None,
                "fatalities_lower": float(row['fatalities_lower']) if pd.notna(row.get('fatalities_lower')) else None,
                "fatalities_upper": float(row['fatalities_upper']) if pd.notna(row.get('fatalities_upper')) else None,
                "uncertainty_width": (
                    float(row['fatalities_upper'] - row['fatalities_lower'])
                    if pd.notna(row.get('fatalities_upper')) and pd.notna(row.get('fatalities_lower'))
                    else None
                ),
                "is_priority": bool(row['is_priority_target']) if 'is_priority_target' in row else False,
                "expected_fatalities": float(row['expected_fatalities']) if pd.notna(row['expected_fatalities']) else None,
                **({"forecast_date": str(row["forecast_date"])} if include_forecast and "forecast_date" in df.columns else {})
            } for _, row in df.iterrows()])
        
        else:
            # Administrative Aggregation
            admin_col = {"region": "admin1", "prefecture": "admin2", "subprefecture": "admin3"}.get(level)
            if not admin_col:
                return jsonify({"error": f"Invalid level: {level}"}), 400
            
            # Query with Join to features_static for admin labels
            # We use H3-based aggregation but return representative H3 cells for Deck.gl polygons
            # Or better: return the admin name and value, let frontend handle polygon rendering
            # For now, we return H3 indices that represent the center of the admin unit to keep Deck.gl logic simple
            
            # 1. Get total H3 count per admin unit for coverage check
            coverage_query = f"SELECT {admin_col}, COUNT(*) as total_cells FROM {SCHEMA}.features_static GROUP BY {admin_col}"
            coverage_df = fetch_table_data(coverage_query)
            
            # 2. Get predictions and join
            query = f"""
                SELECT s.{admin_col},
                       p.conflict_prob,
                       p.predicted_fatalities,
                       {lower_expr} AS fatalities_lower,
                       {upper_expr} AS fatalities_upper,
                       p.conflict_prob * p.predicted_fatalities AS expected_fatalities,
                       s.h3_index
                FROM {SCHEMA}.predictions p
                JOIN {SCHEMA}.features_static s ON p.h3_index = s.h3_index
                WHERE {where_clause}
            """
            raw_df = fetch_table_data(query, params)
            if raw_df.empty: return jsonify([])

            # 3. Perform Scientific Aggregation & Broadcast
            agg_map = {}
            for name, group in raw_df.groupby(admin_col):
                # Coverage check
                matches = coverage_df[coverage_df[admin_col] == name]['total_cells']
                total_cells = matches.iloc[0] if not matches.empty else len(group)
                
                coverage = len(group) / total_cells
                if coverage < AGG_THRESHOLD:
                    continue
                
                agg_map[name] = {
                    "risk": float(group['conflict_prob'].mean()),
                    "fatalities": float(group['predicted_fatalities'].mean()),
                    "expected_fatalities": float(group['expected_fatalities'].sum()),  # Sum for total expected burden
                    "fatalities_lower": float(group['fatalities_lower'].sum()) if 'fatalities_lower' in group else None,
                    "fatalities_upper": float(group['fatalities_upper'].sum()) if 'fatalities_upper' in group else None,
                }
            
            # Broadcast values back to all H3 cells
            results = []
            for _, row in raw_df.iterrows():
                name = row[admin_col]
                if name in agg_map:
                    stats = agg_map[name]
                    results.append({
                        "hex": to_h3_str(row['h3_index']),
                        "risk": stats['risk'],
                        "fatalities": stats['fatalities'],
                        "fatalities_lower": stats['fatalities_lower'],
                        "fatalities_upper": stats['fatalities_upper'],
                        "uncertainty_width": (
                            float(stats['fatalities_upper'] - stats['fatalities_lower'])
                            if stats.get('fatalities_upper') is not None and stats.get('fatalities_lower') is not None
                            else None
                        ),
                        "expected_fatalities": stats['expected_fatalities'],
                        "admin_name": name
                    })
            
            return jsonify(results)

    except Exception as e:
        logger.error(f"Prediction fetch failed: {e}")
        return jsonify({"error": "Failed to fetch predictions", "details": str(e)}), 500


@app.route("/api/prediction_explanations")
def get_prediction_explanations():
    """
    Returns grouped SHAP-style contributions for a single cell/date/horizon/learner.
    Expects 'hex' param (canonical H3 string). Optional: date (YYYY-MM-DD), horizon, learner.
    """
    hex_str = request.args.get("hex")
    horizon = request.args.get("horizon", "3m")
    learner = request.args.get("learner", "xgboost")

    if not table_exists_check("explanations"):
        return jsonify({
            "hex": hex_str,
            "horizon": horizon,
            "learner": learner,
            "explanations": [],
            "warning": "explanations table missing"
        }), 200

    if not hex_str:
        return jsonify({"error": "Missing 'hex' parameter"}), 400

    h3_int = to_h3_int(hex_str)
    if h3_int is None:
        return jsonify({"error": "Invalid H3 index"}), 400

    target_date = request.args.get("date")

    try:
        if target_date:
            query = f"""
                SELECT date, horizon, learner, feature_group, contribution
                FROM {SCHEMA}.explanations
                WHERE h3_index = :h
                  AND date = :d
                  AND horizon = :hor
                  AND learner = :learner
            """
            params = {"h": h3_int, "d": target_date, "hor": horizon, "learner": learner}
        else:
            query = f"""
                WITH latest AS (
                    SELECT MAX(date) AS max_date
                    FROM {SCHEMA}.explanations
                    WHERE h3_index = :h AND horizon = :hor AND learner = :learner
                )
                SELECT e.date, e.horizon, e.learner, e.feature_group, e.contribution
                FROM {SCHEMA}.explanations e, latest
                WHERE e.h3_index = :h
                  AND e.horizon = :hor
                  AND e.learner = :learner
                  AND e.date = latest.max_date
            """
            params = {"h": h3_int, "hor": horizon, "learner": learner}
        df = fetch_table_data(query, params)
    except Exception as e:
        logger.error(f"Explanation fetch failed: {e}")
        return jsonify({"error": "Failed to fetch explanations", "details": str(e)}), 500

    if df.empty:
        return jsonify({"hex": hex_str, "horizon": horizon, "learner": learner, "explanations": []})

    df = _top_contribution_df(df)

    result = []
    for _, row in df.iterrows():
        result.append({
            "group": row["feature_group"],
            "contribution": float(row["contribution"]) if pd.notna(row["contribution"]) else None
        })

    return jsonify({
        "hex": hex_str,
        "horizon": df["horizon"].iloc[0] if "horizon" in df.columns else horizon,
        "learner": df["learner"].iloc[0] if "learner" in df.columns else learner,
        "date": str(df["date"].iloc[0]) if "date" in df.columns and not df.empty else (target_date if target_date else None),
        "explanations": result
    })


@app.route("/api/analytics/prediction/hex/<h3_index>/explanations")
def get_prediction_explanations_analytics(h3_index):
    """
    Analytics-friendly wrapper for explanations that matches the predictions analytics pattern.
    Accepts H3 index in any format (canonical string, decimal string, or int).
    Optional query params: date, horizon (default 3m), learner (default xgboost).
    """
    h3_int = to_h3_int(h3_index)
    if h3_int is None:
        return jsonify({"error": "Invalid H3 Index"}), 400

    horizon = request.args.get("horizon", "3m")
    learner = request.args.get("learner", "xgboost")
    target_date = request.args.get("date")

    if not table_exists_check("explanations"):
        return jsonify({
            "h3_index": to_h3_str(h3_int) or h3_index,
            "horizon": horizon,
            "learner": learner,
            "date": target_date if target_date else None,
            "explanations": [],
            "data_missing": True,
            "warning": "explanations table missing"
        }), 200

    try:
        if target_date:
            query = f"""
                SELECT date, horizon, learner, feature_group, contribution
                FROM {SCHEMA}.explanations
                WHERE h3_index = :h
                  AND date = :d
                  AND horizon = :hor
                  AND learner = :learner
            """
            params = {"h": h3_int, "d": target_date, "hor": horizon, "learner": learner}
        else:
            query = f"""
                WITH latest AS (
                    SELECT MAX(date) AS max_date
                    FROM {SCHEMA}.explanations
                    WHERE h3_index = :h AND horizon = :hor AND learner = :learner
                )
                SELECT e.date, e.horizon, e.learner, e.feature_group, e.contribution
                FROM {SCHEMA}.explanations e, latest
                WHERE e.h3_index = :h
                  AND e.horizon = :hor
                  AND e.learner = :learner
                  AND e.date = latest.max_date
            """
            params = {"h": h3_int, "hor": horizon, "learner": learner}

        df = fetch_table_data(query, params)
    except Exception as e:
        logger.error(f"Explanation analytics fetch failed: {e}")
        return jsonify({"error": "Failed to fetch explanations", "details": str(e)}), 500

    if df.empty:
        return jsonify({
            "h3_index": to_h3_str(h3_int) or h3_index,
            "horizon": horizon,
            "learner": learner,
            "date": target_date if target_date else None,
            "explanations": [],
            "data_missing": True
        })

    df = _top_contribution_df(df)

    h3_str = to_h3_str(h3_int) or h3_index
    date_val = target_date if target_date else (df["date"].iloc[0] if "date" in df.columns else None)
    resolved_horizon = df["horizon"].iloc[0] if "horizon" in df.columns else horizon
    resolved_learner = df["learner"].iloc[0] if "learner" in df.columns else learner

    return jsonify({
        "h3_index": h3_str,
        "horizon": resolved_horizon,
        "learner": resolved_learner,
        "date": str(date_val) if date_val is not None else None,
        "explanations": [
            {
                "group": row["feature_group"],
                "contribution": float(row["contribution"]) if pd.notna(row["contribution"]) else None
            }
            for _, row in df.iterrows()
        ],
        "data_missing": False
    })

# =============================================================================
# SHAP EXPLANATIONS FOR CELL INSPECTOR (NEW)
# =============================================================================
# Loads per-cell SHAP explanations from parquet artifacts
# Generated by: python -m pipeline.analysis.generate_explanations --shap-export

# Cache for SHAP DataFrames indexed by (horizon, learner)
_shap_cache: Dict[tuple, pd.DataFrame] = {}


def _top_contribution_df(df: pd.DataFrame, value_col: str = "contribution", limit: int = TOP_CONTRIBUTIONS_LIMIT) -> pd.DataFrame:
    """Return the top-N rows by absolute contribution value."""
    if df.empty or value_col not in df.columns:
        return df.head(limit)
    out = df.copy()
    out["_abs_contribution"] = pd.to_numeric(out[value_col], errors="coerce").abs()
    out = out.sort_values("_abs_contribution", ascending=False).drop(columns=["_abs_contribution"])
    return out.head(limit)


def _top_feature_items(items: Any, limit: int = TOP_CONTRIBUTIONS_LIMIT) -> List[Dict[str, Any]]:
    """Return top-N SHAP feature objects by absolute value when present."""
    if not isinstance(items, list):
        return []
    ranked: List[tuple] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        val = item.get("value")
        try:
            score = abs(float(val))
        except (TypeError, ValueError):
            score = -1.0
        ranked.append((score, item))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in ranked[:limit]]


def _load_shap_data(horizon: str, learner: str, mode: str = "standard") -> Optional[pd.DataFrame]:
    """
    Load SHAP explanations parquet for a given horizon/learner.
    Mode: 'standard' (full permutation) or 'fast' (tree decomposition).
    """
    cache_key = (horizon, learner, mode)
    if cache_key in _shap_cache:
        return _shap_cache[cache_key]
    
    # Construct filename
    suffix = "_fast.parquet" if mode == "fast" else ".parquet"
    shap_path = ROOT_DIR / "data" / "processed" / f"shap_explanations_{horizon}_{learner}{suffix}"
    
    if not shap_path.exists():
        # Fallback: if 'fast' requested but missing, try standard? No, be explicit.
        logger.warning(f"SHAP parquet not found: {shap_path}")
        return None

    try:
        df = pd.read_parquet(shap_path)
        
        # Set multi-index for fast lookup
        if 'h3_index' in df.columns and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # Ensure h3 is int
            if df['h3_index'].dtype == 'object':
                df['h3_index'] = df['h3_index'].apply(to_h3_int)
            
            df.set_index(['h3_index', 'date'], inplace=True)
            df.sort_index(inplace=True)
            
        _shap_cache[cache_key] = df
        logger.info(f"Loaded SHAP data ({mode}): {shap_path} ({len(df):,} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to load SHAP data from {shap_path}: {e}")
        return None


@app.route("/api/predictions/shap")
def get_predictions_shap():
    """
    Returns per-cell SHAP explanations for the cell inspector.
    
    Query Params:
    - hex: H3 index
    - date: YYYY-MM-DD
    - horizon: 14d | 1m | 3m
    - learner: xgboost | lightgbm
    - mode: standard | fast (default: fast)
    """
    hex_str = request.args.get("hex")
    if not hex_str:
        return jsonify({"error": "Missing 'hex' parameter"}), 400
    
    h3_int = to_h3_int(hex_str)
    if h3_int is None:
        return jsonify({"error": "Invalid H3 index"}), 400
    
    horizon = request.args.get("horizon", "3m")
    learner = request.args.get("learner", "xgboost")
    mode = request.args.get("mode", "fast")  # Default to fast now
    target_date = request.args.get("date")
    
    # Load SHAP data
    shap_df = _load_shap_data(horizon, learner, mode)
    
    if shap_df is None:
        return jsonify({
            "hex": to_h3_str(h3_int) or hex_str,
            "date": target_date,
            "horizon": horizon,
            "learner": learner,
            "mode": mode,
            "top_features": [],
            "data_missing": True,
            "warning": f"SHAP data ({mode}) not available. Run generation script."
        }), 200
    
    try:
        # Get all rows for this H3 cell
        if h3_int in shap_df.index.get_level_values('h3_index'):
            cell_data = shap_df.xs(h3_int, level='h3_index')
        else:
            # Cell not in SHAP data (might have been filtered by sample_ratio)
            return jsonify({
                "hex": to_h3_str(h3_int) or hex_str,
                "date": target_date,
                "horizon": horizon,
                "learner": learner,
                "top_features": [],
                "data_missing": True,
                "warning": "Cell not in SHAP sample (try increasing sample_ratio in config)"
            }), 200
        
        # Filter by date if specified, otherwise get latest
        if target_date:
            target_dt = pd.to_datetime(target_date)
            if target_dt in cell_data.index:
                row = cell_data.loc[target_dt]
            else:
                # Date not found for this cell
                available_dates = sorted(cell_data.index.tolist())
                return jsonify({
                    "hex": to_h3_str(h3_int) or hex_str,
                    "date": target_date,
                    "horizon": horizon,
                    "learner": learner,
                    "top_features": [],
                    "data_missing": True,
                    "warning": f"Date {target_date} not found. Available: {str(available_dates[:5])}..."
                }), 200
        else:
            # Get latest date
            latest_date = cell_data.index.max()
            row = cell_data.loc[latest_date]
            target_date = str(latest_date.date()) if hasattr(latest_date, 'date') else str(latest_date)
        
        # Parse top_features JSON
        top_features_json = row['top_features']
        if isinstance(top_features_json, str):
            top_features = json.loads(top_features_json)
        else:
            top_features = top_features_json
        top_features = _top_feature_items(top_features)
        
        return jsonify({
            "hex": to_h3_str(h3_int) or hex_str,
            "date": target_date,
            "horizon": horizon,
            "learner": learner,
            "top_features": top_features,
            "data_missing": False
        })
        
    except Exception as e:
        logger.error(f"SHAP lookup failed: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to fetch SHAP explanations",
            "details": str(e)
        }), 500


@app.route("/api/predictions/shap/dates")
def get_shap_dates():
    """
    Returns available dates in SHAP data for a given horizon/learner.
    Useful for frontend to know what dates are available.
    """
    horizon = request.args.get("horizon", "3m")
    learner = request.args.get("learner", "xgboost")
    
    shap_df = _load_shap_data(horizon, learner)
    
    if shap_df is None:
        return jsonify({
            "dates": [],
            "warning": f"SHAP data not available for {horizon}/{learner}"
        })
    
    dates = sorted(shap_df.index.get_level_values('date').unique())
    return jsonify({
        "dates": [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates],
        "count": len(dates)
    })


@app.route("/api/predictions/shap/status")
def get_shap_status():
    """
    Returns availability status of SHAP data files for all horizon/learner combinations.
    """
    horizons = ["14d", "1m", "3m"]
    learners = ["xgboost", "lightgbm"]
    
    status = {}
    for h in horizons:
        status[h] = {}
        for l in learners:
            shap_path = ROOT_DIR / "data" / "processed" / f"shap_explanations_{h}_{l}.parquet"
            if shap_path.exists():
                try:
                    # Get file info without loading full DataFrame
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(shap_path)
                    num_rows = pf.metadata.num_rows
                    status[h][l] = {
                        "available": True,
                        "rows": num_rows,
                        "path": str(shap_path)
                    }
                except Exception:
                    status[h][l] = {
                        "available": True,
                        "rows": "unknown",
                        "path": str(shap_path)
                    }
            else:
                status[h][l] = {
                    "available": False,
                    "path": str(shap_path)
                }
    
    return jsonify({
        "status": status,
        "generate_command": "python -m pipeline.analysis.generate_explanations --shap-export"
    })


@app.route("/api/predictions/cube")
def get_predictions_cube():
    """
    Full spatio-temporal predictions history for animation.
    Uses ONLY car_cewp.predictions.
    
    CRITICAL: Returns canonical H3 strings for Deck.gl compatibility.
    """
    if not table_exists_check("predictions"):
        return jsonify({
            "error": "predictions table missing",
            "table": f"{SCHEMA}.predictions"
        }), 500

    horizon = request.args.get("horizon", "14d")
    learner = request.args.get("learner")
    include_forecast = request.args.get('include_forecast_date', 'false').lower() in ("1", "true", "yes", "y")
    horizon_interval = {
        "14d": "14 days",
        "1m": "28 days",
        "2m": "56 days",
        "3m": "84 days",
    }.get(horizon, None)

    col_flags = prediction_interval_columns()
    lower_expr = "fatalities_lower" if col_flags["fatalities_lower"] else "NULL::double precision"
    upper_expr = "fatalities_upper" if col_flags["fatalities_upper"] else "NULL::double precision"

    query = f"""
        SELECT 
            to_char(date, 'YYYY-MM-DD') as date,
            h3_index,
            conflict_prob AS risk,
            predicted_fatalities AS fatal,
            {lower_expr} AS fatal_lower,
            {upper_expr} AS fatal_upper,
            is_priority_target AS priority,
            conflict_prob * predicted_fatalities AS expected_fatal
            {" , to_char(date + INTERVAL '" + horizon_interval + "', 'YYYY-MM-DD') AS forecast_date" if include_forecast and horizon_interval else ""}
        FROM {SCHEMA}.predictions
        WHERE date > CURRENT_DATE - INTERVAL '5 years'
          AND horizon = :h
          {"AND learner = :l" if learner else ""}
        ORDER BY date ASC
    """
    
    params = {"h": horizon}
    if learner:
        params["l"] = learner

    df = fetch_table_data(query, params)
    
    if df.empty:
        return jsonify([])

    # Convert to canonical H3 strings for Deck.gl
    df['hex'] = df['h3_index'].apply(to_h3_str)
    df = df.dropna(subset=['hex'])
    
    result = []
    for _, row in df.iterrows():
        result.append({
            "date": row['date'],
            "h3_index": row['hex'],  # Canonical H3 string
            "risk": float(row['risk']) if pd.notna(row['risk']) else None,
            "fatal": float(row['fatal']) if pd.notna(row['fatal']) else None,
            "fatal_lower": float(row['fatal_lower']) if pd.notna(row.get('fatal_lower')) else None,
            "fatal_upper": float(row['fatal_upper']) if pd.notna(row.get('fatal_upper')) else None,
            "priority": bool(row['priority']) if 'priority' in row else False,
            "expected_fatal": float(row['expected_fatal']) if pd.notna(row['expected_fatal']) else None,
            **({"forecast_date": row["forecast_date"]} if include_forecast and "forecast_date" in df.columns else {})
        })

    return Response(orjson.dumps(result), mimetype='application/json')

# =============================================================================
# MAP 2: TEMPORAL FEATURES (car_cewp.temporal_features ONLY)
# =============================================================================

@app.route("/api/temporal_feature")
def get_temporal_feature():
    """
    MAP 2 DATA SOURCE: Returns H3 or Admin values for ONE selected temporal feature.
    
    Query Params:
    - feature: feature_name
    - date: YYYY-MM-DD
    - level: h3 | region | prefecture | subprefecture
    """
    feature = request.args.get('feature')
    target_date = request.args.get('date')
    level = request.args.get('level', 'h3').lower()
    
    if not feature:
        return jsonify({"error": "Missing 'feature' parameter"}), 400
    
    if feature not in TEMPORAL_FEATURES_ALLOWLIST:
        return jsonify({"error": f"Unknown or disallowed feature: {feature}"}), 400
    
    if not table_exists_check("temporal_features"):
        return jsonify({"error": "temporal_features table missing"}), 500
    
    # 1. Base Query logic
    where_filters = []
    params = {}
    if target_date:
        where_filters.append("t.date = :d")
        params["d"] = target_date
    else:
        where_filters.append(f"t.date = (SELECT MAX(date) FROM {SCHEMA}.temporal_features WHERE {feature} IS NOT NULL)")

    where_clause = " WHERE " + " AND ".join(where_filters) if where_filters else ""

    try:
        if level == 'h3':
            query = f"SELECT h3_index, {feature} as value FROM {SCHEMA}.temporal_features t {where_clause}"
            df = fetch_table_data(query, params)
            if df.empty: return jsonify([])
            
            df['hex'] = df['h3_index'].apply(to_h3_str)
            df = df.dropna(subset=['hex'])
            return jsonify([{"hex": r['hex'], "value": float(r['value']) if pd.notna(r['value']) else None} for _, r in df.iterrows()])
        
        else:
            admin_col = {"region": "admin1", "prefecture": "admin2", "subprefecture": "admin3"}.get(level)
            if not admin_col: return jsonify({"error": f"Invalid level: {level}"}), 400
            
            # Coverage stats
            cov_query = f"SELECT {admin_col}, COUNT(*) as total_cells FROM {SCHEMA}.features_static GROUP BY {admin_col}"
            cov_df = fetch_table_data(cov_query)
            
            # Query
            query = f"""
                SELECT s.{admin_col}, t.{feature}, s.h3_index
                FROM {SCHEMA}.temporal_features t
                JOIN {SCHEMA}.features_static s ON t.h3_index = s.h3_index
                {where_clause}
                ORDER BY s.h3_index
            """
            raw_df = fetch_table_data(query, params)
            if raw_df.empty: return jsonify([])

            method = AGG_METHODS.get(feature, AGG_METHODS.get("default_method", "mean"))
            
            agg_map = {}
            for name, group in raw_df.groupby(admin_col):
                matches = cov_df[cov_df[admin_col] == name]['total_cells']
                total_cells = matches.iloc[0] if not matches.empty else len(group)
                
                coverage = len(group) / total_cells
                if coverage < AGG_THRESHOLD: continue
                
                val = group[feature].mean() if method == "mean" else group[feature].median() if method == "median" else group[feature].sum()
                agg_map[name] = {"value": float(val) if pd.notna(val) else None, "coverage": float(coverage)}

            # Broadcast
            results = []
            for _, row in raw_df.iterrows():
                name = row[admin_col]
                if name in agg_map:
                    results.append({
                        "hex": to_h3_str(row['h3_index']),
                        "value": agg_map[name]["value"],
                        "coverage": agg_map[name]["coverage"],
                        "admin_name": name
                    })
            
            return jsonify(results)

    except Exception as e:
        logger.error(f"Temporal feature fetch failed: {e}")
        return jsonify({"error": "Failed to fetch temporal feature", "details": str(e)}), 500

@app.route("/api/temporal_features/list")
def list_temporal_features():
    """Returns the allowlisted temporal features grouped by category."""
    return jsonify({
        "features": {
            "Environmental": [
                "chirps_precip_anomaly", "era5_temp_anomaly", "era5_soil_moisture_anomaly",
                "ndvi_anomaly", "ntl_mean"
            ],
            "Conflict": [
                "fatalities_14d_sum", "fatalities_1m_lag", "protest_count_lag1",
                "riot_count_lag1", "regional_risk_score_lag1"
            ],
            "GDELT": [
                "gdelt_event_count", "gdelt_avg_tone", "gdelt_goldstein_mean", "gdelt_mentions_total"
            ],
            "CrisisWatch": [
                "cw_score_local", "regime_parallel_governance", "regime_transnational_predation",
                "regime_guerrilla_fragmentation", "regime_ethno_pastoral_rupture", "narrative_velocity_lag1"
            ],
            "ACLED Hybrid": [
                "mech_gold_pivot_lag1", "mech_predatory_tax_lag1",
                "mech_factional_infighting_lag1", "mech_collective_punishment_lag1"
            ],
            "Market": [
                "price_maize", "price_rice", "price_oil", "price_sorghum", "price_cassava", "price_groundnuts"
            ],
            "Macroeconomics": [
                "gold_price_usd_lag1", "oil_price_usd_lag1", "sp500_index_lag1", "eur_usd_rate_lag1"
            ],
            "Displacement": ["iom_displacement_count_lag1"],
            "EPR": [
                "epr_excluded_groups_count", "epr_discriminated_groups_count",
                "epr_status_mean", "ethnic_group_count"
            ],
            "Population": ["pop_log"]
        }
    })

# =============================================================================
# MAP 3: STATIC FEATURES (car_cewp.features_static ONLY)
# =============================================================================

@app.route("/api/static_feature")
def get_static_feature():
    """
    MAP 3 DATA SOURCE: Returns H3-level values for ONE selected static feature.
    Uses ONLY car_cewp.features_static. No predictions/temporal mixing.
    
    CRITICAL: Returns canonical H3 strings for Deck.gl compatibility.
    """
    feature = request.args.get('feature')
    
    if not feature:
        return jsonify({"error": "Missing 'feature' parameter"}), 400
    
    if feature not in STATIC_FEATURES_ALLOWLIST:
        return jsonify({
            "error": f"Unknown or disallowed feature: {feature}",
            "allowed_features": sorted(list(STATIC_FEATURES_ALLOWLIST))
        }), 400
    
    if not table_exists_check("features_static"):
        return jsonify({
            "error": "features_static table missing",
            "table": f"{SCHEMA}.features_static"
        }), 500
    
    if not column_exists("features_static", feature):
        return jsonify({
            "error": f"Column '{feature}' not found in features_static table",
            "warning": "Feature may not be computed yet"
        }), 404
    
    try:
        query = f"""
            SELECT h3_index, {feature} as value
            FROM {SCHEMA}.features_static
        """
        df = fetch_table_data(query)
    except Exception as e:
        logger.error(f"Static feature fetch failed: {e}")
        return jsonify({"error": "Failed to fetch static feature", "details": str(e)}), 500

    if df.empty:
        return jsonify([])

    # Convert to canonical H3 strings for Deck.gl
    df['hex'] = df['h3_index'].apply(to_h3_str)
    df = df.dropna(subset=['hex'])

    result = []
    for _, row in df.iterrows():
        result.append({
            "hex": row['hex'],
            "value": float(row['value']) if pd.notna(row['value']) else None
        })

    return jsonify(result)

@app.route("/api/static_features/list")
def list_static_features():
    """Returns the allowlisted static features grouped by category."""
    return jsonify({
        "features": {
            "Distance": [
                "dist_to_capital", "dist_to_border", "dist_to_city", "dist_to_road", "dist_to_river",
                "dist_to_market_km", "dist_to_diamond_mine", "dist_to_gold_mine",
                "dist_to_large_mine", "dist_to_controlled_mine", "dist_to_large_gold_mine"
            ],
            "Geographic": ["elevation_mean", "slope_mean", "terrain_ruggedness_index"]
        }
    })

@app.route("/api/analytics/static/hex/<h3_index>")
def get_static_hex_snapshot(h3_index):
    """
    Return all static feature values for a single hex (one row from features_static).
    """
    h3_int = to_h3_int(h3_index)
    if h3_int is None:
        return jsonify({"error": "Invalid H3 Index"}), 400

    if not table_exists_check("features_static"):
        return jsonify({
            "error": "features_static table missing",
            "table": f"{SCHEMA}.features_static"
        }), 500

    cols = sorted(list(STATIC_FEATURES_ALLOWLIST))
    select_cols = ", ".join(cols)

    try:
        query = text(f"""
            SELECT {select_cols}
            FROM {SCHEMA}.features_static
            WHERE h3_index = :h3
            LIMIT 1
        """)
        df = fetch_table_data(query, params={"h3": h3_int})
    except Exception as e:
        logger.error(f"Static hex snapshot failed: {e}")
        return jsonify({"error": "Failed to fetch static snapshot", "details": str(e)}), 500

    if df.empty:
        return jsonify({"error": "Hex not found"}), 404

    row = df.iloc[0].to_dict()
    return jsonify({
        "hex": h3_index,
        "values": {k: (None if pd.isna(v) else v) for k, v in row.items()}
    })

# =============================================================================
# ANALYTICS ENDPOINTS (Split by domain - no mixing)
# =============================================================================

@app.route("/api/analytics/prediction/hex/<h3_index>")
def get_prediction_analytics(h3_index):
    """
    Prediction history for a single hex.
    Uses ONLY car_cewp.predictions.
    
    Accepts H3 index in any format (canonical string, decimal string, or int).
    """
    # Convert input to integer for DB query
    h3_int = to_h3_int(h3_index)
    if h3_int is None:
        return jsonify({"error": "Invalid H3 Index"}), 400

    horizon = request.args.get("horizon", "3m")
    learner = request.args.get("learner")

    if not table_exists_check("predictions"):
        return jsonify({
            "error": "predictions table missing",
            "table": f"{SCHEMA}.predictions",
            "data_missing": True
        }), 500

    col_flags = prediction_interval_columns()
    lower_expr = "fatalities_lower" if col_flags["fatalities_lower"] else "NULL::double precision"
    upper_expr = "fatalities_upper" if col_flags["fatalities_upper"] else "NULL::double precision"

    query = f"""
        SELECT date, conflict_prob AS risk, predicted_fatalities AS fatalities,
               {lower_expr} AS fatalities_lower,
               {upper_expr} AS fatalities_upper,
               conflict_prob * predicted_fatalities AS expected_fatalities
        FROM {SCHEMA}.predictions 
        WHERE h3_index = :h3 
          AND horizon = :h
          {"AND learner = :l" if learner else ""}
        ORDER BY date ASC
    """
    params = {"h3": h3_int, "h": horizon}
    if learner:
        params["l"] = learner
    df = fetch_table_data(query, params)
    
    # Convert back to canonical string for response
    h3_str = to_h3_str(h3_int) or h3_index
    
    if df.empty:
        return jsonify({
            "h3_index": h3_str,
            "history": {
                "dates": [],
                "risk": [],
                "fatalities": [],
                "fatalities_lower": [],
                "fatalities_upper": [],
                "expected_fatalities": []
            },
            "data_missing": True,
            "warning": "No prediction history found for this hex"
        })

    return jsonify({
        "h3_index": h3_str,
        "history": {
            "dates": [str(d) for d in df['date']],
            "risk": [float(v) if pd.notna(v) else None for v in df['risk']],
            "fatalities": [float(v) if pd.notna(v) else None for v in df['fatalities']],
            "fatalities_lower": [float(v) if pd.notna(v) else None for v in df['fatalities_lower']],
            "fatalities_upper": [float(v) if pd.notna(v) else None for v in df['fatalities_upper']],
            "expected_fatalities": [float(v) if pd.notna(v) else None for v in df['expected_fatalities']]
        },
        "data_missing": False
    })

@app.route("/api/analytics/temporal/hex/<h3_index>")
def get_temporal_analytics(h3_index):
    """
    Temporal feature history for a single hex OR admin unit.
    
    Accepts:
    - h3_index: Canonical string (e.g. "855a5a1bfffffff")
    - admin_name: If level param is provided, this matches the admin name.
    
    Query Params:
    - feature: feature_name
    - level: h3 | region | prefecture | subprefecture
    """
    feature = request.args.get('feature')
    level = request.args.get('level', 'h3').lower()
    
    if not feature or feature not in TEMPORAL_FEATURES_ALLOWLIST:
        return jsonify({"error": "Missing or invalid 'feature' parameter"}), 400

    if not table_exists_check("temporal_features"):
        return jsonify({"error": "temporal_features table missing", "data_missing": True}), 500

    try:
        if level == 'h3':
            h3_int = to_h3_int(h3_index)
            if h3_int is None: return jsonify({"error": "Invalid H3 Index"}), 400
            
            query = f"SELECT date, {feature} as value FROM {SCHEMA}.temporal_features WHERE h3_index = :h3 ORDER BY date ASC"
            df = fetch_table_data(query, {"h3": h3_int})
            h3_str = to_h3_str(h3_int) or h3_index
        else:
            admin_col = {"region": "admin1", "prefecture": "admin2", "subprefecture": "admin3"}.get(level)
            if not admin_col: return jsonify({"error": f"Invalid level: {level}"}), 400
            
            # Scientific Aggregation Rule
            method = AGG_METHODS.get(feature, AGG_METHODS.get("default_method", "mean"))
            
            if method == "mean":
                agg_expr = f"AVG(t.{feature})"
            elif method == "sum":
                agg_expr = f"SUM(t.{feature})"
            elif method == "median":
                agg_expr = f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.{feature})"
            else:
                agg_expr = f"AVG(t.{feature})" # Fallback

            query = f"""
                SELECT t.date, {agg_expr} as value
                FROM {SCHEMA}.temporal_features t
                JOIN {SCHEMA}.features_static s ON t.h3_index = s.h3_index
                WHERE s.{admin_col} = :name
                GROUP BY t.date
                ORDER BY t.date ASC
            """
            df = fetch_table_data(query, {"name": h3_index})
            h3_str = h3_index

        if df.empty:
            return jsonify({"h3_index": h3_str, "feature": feature, "history": {"dates": [], "values": []}, "data_missing": True})

        return jsonify({
            "h3_index": h3_str,
            "feature": feature,
            "history": {
                "dates": [str(d) for d in df['date']],
                "values": [float(v) if pd.notna(v) else None for v in df['value']]
            },
            "data_missing": False
        })

    except Exception as e:
        logger.error(f"Temporal analytics failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analytics/temporal/summary")
def get_temporal_summary():
    """
    Returns per-date Max/Min for a feature across the PEER GROUP.
    
    If level=h3: Max/Min of all H3 cells.
    If level=region: Max/Min of all REGIONS (aggregated first).
    """
    feature = request.args.get('feature')
    level = request.args.get('level', 'h3').lower()
    
    if not feature or feature not in TEMPORAL_FEATURES_ALLOWLIST:
        return jsonify({"error": "Missing or invalid 'feature' parameter"}), 400

    if not table_exists_check("temporal_features"):
        return jsonify({"error": "temporal_features table missing"}), 500

    try:
        if level == 'h3':
            query = f"""
                SELECT date, MAX({feature}) AS max_val, MIN({feature}) AS min_val
                FROM {SCHEMA}.temporal_features
                GROUP BY date
                ORDER BY date ASC
            """
        else:
            admin_col = {"region": "admin1", "prefecture": "admin2", "subprefecture": "admin3"}.get(level)
            if not admin_col: return jsonify({"error": f"Invalid level: {level}"}), 400
            
            # Scientific Aggregation Rule
            method = AGG_METHODS.get(feature, AGG_METHODS.get("default_method", "mean"))
            
            if method == "mean":
                agg_expr = f"AVG(t.{feature})"
            elif method == "sum":
                agg_expr = f"SUM(t.{feature})"
            elif method == "median":
                agg_expr = f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.{feature})"
            else:
                agg_expr = f"AVG(t.{feature})"

            # Two-stage query: 
            # 1. Aggregate per admin unit per date
            # 2. Find Max/Min across admin units per date
            query = f"""
                WITH admin_agg AS (
                    SELECT t.date, s.{admin_col}, {agg_expr} as val
                    FROM {SCHEMA}.temporal_features t
                    JOIN {SCHEMA}.features_static s ON t.h3_index = s.h3_index
                    GROUP BY t.date, s.{admin_col}
                )
                SELECT date, MAX(val) as max_val, MIN(val) as min_val
                FROM admin_agg
                GROUP BY date
                ORDER BY date ASC
            """

        df = fetch_table_data(query)
        if df.empty:
            return jsonify({"dates": [], "max": [], "min": []})
            
        return jsonify({
            "dates": [str(d) for d in df['date']],
            "max": [float(v) if pd.notna(v) else None for v in df['max_val']],
            "min": [float(v) if pd.notna(v) else None for v in df['min_val']]
        })
    except Exception as e:
        logger.error(f"Temporal summary fetch failed: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch temporal summary"}), 500

# =============================================================================
# EVENTS (Separate endpoint, optional layer)
# =============================================================================

@app.route("/api/events")
def get_events():
    limit = min(int(request.args.get('limit', 500)), 2000)
    
    if not table_exists_check("acled_events"):
        return jsonify({"type": "FeatureCollection", "features": []})
        
    query = f"""
        SELECT event_id_cnty, event_date, event_type, fatalities, latitude, longitude 
        FROM {SCHEMA}.acled_events 
        ORDER BY event_date DESC 
        LIMIT :limit
    """
    
    df = fetch_table_data(query, {"limit": limit})
    features = []
    
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "properties": {
                "id": row['event_id_cnty'],
                "type": row['event_type'],
                "date": str(row['event_date']),
                "fatalities": row['fatalities']
            },
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
            }
        })

    return jsonify({"type": "FeatureCollection", "features": features})

@app.route("/api/admin/parent")
def get_admin_parent():
    """Returns the parent prefecture for a given subprefecture."""
    sub = request.args.get('subprefecture')
    if not sub: return jsonify({"error": "Missing subprefecture"}), 400
    
    query = f"SELECT DISTINCT admin2 FROM {SCHEMA}.features_static WHERE admin3 = :sub LIMIT 1"
    df = fetch_table_data(query, {"sub": sub})
    if df.empty: return jsonify({"prefecture": None})
    return jsonify({"prefecture": df.iloc[0, 0]})

# =============================================================================
# DEV TOOLS
# =============================================================================

@app.route("/api/features/columns")
def get_columns():
    table = request.args.get('table', 'temporal_features')
    allowed_tables = {'temporal_features', 'features_static', 'predictions'}
    
    if table not in allowed_tables:
        return jsonify({"error": "Table not allowed"}), 400
    
    if not table_exists_check(table):
        return jsonify({"error": "Table not found"}), 404
        
    query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = :s AND table_name = :t
    """
    df = fetch_table_data(query, {"s": SCHEMA, "t": table})
    return jsonify(df.to_dict(orient='records'))

@app.route("/api/debug/h3_test")
def h3_test():
    """Debug endpoint to verify H3 conversion is working correctly."""
    test_cases = [
        600844665696026624,  # Typical BIGINT from DB
        "600844665696026624",  # Decimal string
        "855a5a1bfffffff",  # Canonical H3 string
    ]
    
    results = []
    for tc in test_cases:
        h3_str = to_h3_str(tc)
        h3_int = to_h3_int(tc)
        results.append({
            "input": str(tc),
            "input_type": type(tc).__name__,
            "to_h3_str": h3_str,
            "to_h3_int": h3_int,
            "is_valid": h3.is_valid_cell(h3_str) if h3_str else False
        })
    
    return jsonify({
        "h3_version": h3.__version__,
        "test_results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
