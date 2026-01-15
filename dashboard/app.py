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

# --- Configuration & Environment ---

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FlashCLED")

SCHEMA = os.environ.get("DB_SCHEMA", "car_cewp")

# --- Feature Allowlists (Strict Validation) ---

TEMPORAL_FEATURES_ALLOWLIST = {
    # Environmental/Remote sensing
    "chirps_precip_anomaly", "era5_temp_anomaly", "era5_soil_moisture_anomaly",
    "ndvi_anomaly", "nightlights_intensity",
    # Conflict
    "fatalities_14d_sum", "fatalities_1m_lag", "protest_count_lag1",
    "riot_count_lag1", "regional_risk_score_lag1",
    # GDELT
    "gdelt_event_count", "gdelt_avg_tone", "gdelt_goldstein_mean", "gdelt_mentions_total",
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
    
    return jsonify({"dates": [str(d) for d in df['date'].tolist()]})

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
    
    return jsonify({"dates": [str(d) for d in df['date'].tolist()]})

# =============================================================================
# MAP 1: PREDICTIONS (car_cewp.predictions ONLY)
# =============================================================================

@app.route("/api/predictions")
def get_predictions():
    """
    MAP 1 DATA SOURCE: Returns H3 prediction rows for visualization.
    Uses ONLY car_cewp.predictions. No temporal/static feature mixing.
    
    CRITICAL: Returns canonical H3 strings for Deck.gl compatibility.
    """
    if not table_exists_check("predictions"):
        return jsonify({
            "error": "predictions table missing",
            "table": f"{SCHEMA}.predictions"
        }), 500
    
    target_date = request.args.get('date')
    horizon = request.args.get('horizon', '3m')
    learner = request.args.get('learner')  # optional filter; if None returns all learners
    include_forecast = request.args.get('include_forecast_date', 'false').lower() in ("1", "true", "yes", "y")

    # Map horizon to SQL interval for forecasted calendar date
    horizon_interval = {
        "14d": "14 days",
        "1m": "1 month",
        "3m": "3 months",
    }.get(horizon, None)

    try:
        if target_date:
            query = f"""
                SELECT p.h3_index,
                       p.conflict_prob AS risk_score,
                       p.predicted_fatalities AS fatalities,
                       p.conflict_prob * p.predicted_fatalities AS expected_fatalities
                       {" , (p.date + INTERVAL '" + horizon_interval + "') AS forecast_date" if include_forecast and horizon_interval else ""}
                FROM {SCHEMA}.predictions p
                WHERE p.date = :d AND p.horizon = :h
                {"AND p.learner = :l" if learner else ""}
            """
            params = {"d": target_date, "h": horizon}
            if learner:
                params["l"] = learner
            df = fetch_table_data(query, params)
        else:
            # Get latest date's predictions
            query = f"""
                WITH latest AS (
                    SELECT MAX(date) AS max_date
                    FROM {SCHEMA}.predictions
                    WHERE horizon = :h {"AND learner = :l" if learner else ""}
                )
                SELECT p.h3_index,
                       p.conflict_prob AS risk_score,
                       p.predicted_fatalities AS fatalities,
                       p.conflict_prob * p.predicted_fatalities AS expected_fatalities
                       {" , (p.date + INTERVAL '" + horizon_interval + "') AS forecast_date" if include_forecast and horizon_interval else ""}
                FROM {SCHEMA}.predictions p, latest
                WHERE p.date = latest.max_date AND p.horizon = :h {"AND p.learner = :l" if learner else ""}
            """
            params = {"h": horizon}
            if learner:
                params["l"] = learner
            df = fetch_table_data(query, params)
    except Exception as e:
        logger.error(f"Prediction fetch failed: {e}")
        return jsonify({"error": "Failed to fetch predictions", "details": str(e)}), 500

    if df.empty:
        return jsonify([])

    # Convert to canonical H3 strings for Deck.gl
    df['hex'] = df['h3_index'].apply(to_h3_str)
    df = df.dropna(subset=['hex'])

    result = []
    for _, row in df.iterrows():
        result.append({
            "hex": row['hex'],
            "risk": float(row['risk_score']) if pd.notna(row['risk_score']) else None,
            "fatalities": float(row['fatalities']) if pd.notna(row['fatalities']) else None,
            "expected_fatalities": float(row['expected_fatalities']) if pd.notna(row['expected_fatalities']) else None,
            **({"forecast_date": str(row["forecast_date"])} if include_forecast and "forecast_date" in df.columns else {})
        })

    return jsonify(result)


@app.route("/api/prediction_explanations")
def get_prediction_explanations():
    """
    Returns grouped SHAP-style contributions for a single cell/date/horizon/learner.
    Expects 'hex' param (canonical H3 string). Optional: date (YYYY-MM-DD), horizon, learner.
    """
    if not table_exists_check("explanations"):
        return jsonify({
            "error": "explanations table missing",
            "table": f"{SCHEMA}.explanations"
        }), 500

    hex_str = request.args.get("hex")
    if not hex_str:
        return jsonify({"error": "Missing 'hex' parameter"}), 400

    h3_int = to_h3_int(hex_str)
    if h3_int is None:
        return jsonify({"error": "Invalid H3 index"}), 400

    horizon = request.args.get("horizon", "3m")
    learner = request.args.get("learner", "xgboost")
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
            "error": "explanations table missing",
            "table": f"{SCHEMA}.explanations",
            "data_missing": True
        }), 500

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

    horizon = request.args.get("horizon", "3m")
    learner = request.args.get("learner")
    include_forecast = request.args.get('include_forecast_date', 'false').lower() in ("1", "true", "yes", "y")
    horizon_interval = {
        "14d": "14 days",
        "1m": "1 month",
        "3m": "3 months",
    }.get(horizon, None)

    query = f"""
        SELECT 
            to_char(date, 'YYYY-MM-DD') as date,
            h3_index,
            conflict_prob AS risk,
            predicted_fatalities AS fatal,
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
    MAP 2 DATA SOURCE: Returns H3-level values for ONE selected temporal feature.
    Uses ONLY car_cewp.temporal_features. No predictions mixing.
    
    CRITICAL: Returns canonical H3 strings for Deck.gl compatibility.
    """
    feature = request.args.get('feature')
    target_date = request.args.get('date')
    
    if not feature:
        return jsonify({"error": "Missing 'feature' parameter"}), 400
    
    if feature not in TEMPORAL_FEATURES_ALLOWLIST:
        return jsonify({
            "error": f"Unknown or disallowed feature: {feature}",
            "allowed_features": sorted(list(TEMPORAL_FEATURES_ALLOWLIST))
        }), 400
    
    if not table_exists_check("temporal_features"):
        return jsonify({
            "error": "temporal_features table missing",
            "table": f"{SCHEMA}.temporal_features"
        }), 500
    
    if not column_exists("temporal_features", feature):
        return jsonify({
            "error": f"Column '{feature}' not found in temporal_features table",
            "warning": "Feature may not be computed yet"
        }), 404
    
    try:
        if target_date:
            query = f"""
                SELECT h3_index, {feature} as value
                FROM {SCHEMA}.temporal_features
                WHERE date = :d
            """
            df = fetch_table_data(query, {"d": target_date})
        else:
            # Get latest date where the feature is not null
            query = f"""
                WITH latest AS (
                    SELECT MAX(date) as max_date
                    FROM {SCHEMA}.temporal_features
                    WHERE {feature} IS NOT NULL
                )
                SELECT t.h3_index, t.{feature} as value
                FROM {SCHEMA}.temporal_features t, latest
                WHERE t.date = latest.max_date
            """
            df = fetch_table_data(query)
    except Exception as e:
        logger.error(f"Temporal feature fetch failed: {e}")
        return jsonify({"error": "Failed to fetch temporal feature", "details": str(e)}), 500

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

@app.route("/api/temporal_features/list")
def list_temporal_features():
    """Returns the allowlisted temporal features grouped by category."""
    return jsonify({
        "features": {
            "Environmental": [
                "chirps_precip_anomaly", "era5_temp_anomaly", "era5_soil_moisture_anomaly",
                "ndvi_anomaly", "nightlights_intensity"
            ],
            "Conflict": [
                "fatalities_14d_sum", "fatalities_1m_lag", "protest_count_lag1",
                "riot_count_lag1", "regional_risk_score_lag1"
            ],
            "GDELT": [
                "gdelt_event_count", "gdelt_avg_tone", "gdelt_goldstein_mean", "gdelt_mentions_total"
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
            "Geographic": ["elevation_mean", "slope_mean", "terrain_ruggedness_index"],
            "Administrative": ["admin1", "admin2", "admin3"]
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

    query = f"""
        SELECT date, conflict_prob AS risk, predicted_fatalities AS fatalities,
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
            "history": {"dates": [], "risk": [], "fatalities": [], "expected_fatalities": []},
            "data_missing": True,
            "warning": "No prediction history found for this hex"
        })

    return jsonify({
        "h3_index": h3_str,
        "history": {
            "dates": [str(d) for d in df['date']],
            "risk": [float(v) if pd.notna(v) else None for v in df['risk']],
            "fatalities": [float(v) if pd.notna(v) else None for v in df['fatalities']],
            "expected_fatalities": [float(v) if pd.notna(v) else None for v in df['expected_fatalities']]
        },
        "data_missing": False
    })

@app.route("/api/analytics/temporal/hex/<h3_index>")
def get_temporal_analytics(h3_index):
    """
    Temporal feature history for a single hex.
    Uses ONLY car_cewp.temporal_features.
    
    Accepts H3 index in any format (canonical string, decimal string, or int).
    """
    # Convert input to integer for DB query
    h3_int = to_h3_int(h3_index)
    if h3_int is None:
        return jsonify({"error": "Invalid H3 Index"}), 400

    feature = request.args.get('feature')
    if not feature or feature not in TEMPORAL_FEATURES_ALLOWLIST:
        return jsonify({
            "error": "Missing or invalid 'feature' parameter",
            "allowed_features": sorted(list(TEMPORAL_FEATURES_ALLOWLIST))
        }), 400

    if not table_exists_check("temporal_features"):
        return jsonify({
            "error": "temporal_features table missing",
            "table": f"{SCHEMA}.temporal_features",
            "data_missing": True
        }), 500

    if not column_exists("temporal_features", feature):
        return jsonify({
            "error": f"Column '{feature}' not found",
            "data_missing": True
        }), 404

    query = f"""
        SELECT date, {feature} as value
        FROM {SCHEMA}.temporal_features 
        WHERE h3_index = :h3 
        ORDER BY date ASC
    """
    df = fetch_table_data(query, {"h3": h3_int})
    
    # Convert back to canonical string for response
    h3_str = to_h3_str(h3_int) or h3_index
    
    if df.empty:
        return jsonify({
            "h3_index": h3_str,
            "feature": feature,
            "history": {"dates": [], "values": []},
            "data_missing": True,
            "warning": "No temporal data found for this hex"
        })

    return jsonify({
        "h3_index": h3_str,
        "feature": feature,
        "history": {
            "dates": [str(d) for d in df['date']],
            "values": [float(v) if pd.notna(v) else None for v in df['value']]
        },
        "data_missing": False
    })

@app.route("/api/analytics/temporal/summary")
def get_temporal_summary():
    """
    Returns per-date max/min for a temporal feature across all H3 cells.
    """
    feature = request.args.get('feature')
    if not feature or feature not in TEMPORAL_FEATURES_ALLOWLIST:
        return jsonify({
            "error": "Missing or invalid 'feature' parameter",
            "allowed_features": sorted(list(TEMPORAL_FEATURES_ALLOWLIST))
        }), 400

    if not table_exists_check("temporal_features"):
        return jsonify({
            "error": "temporal_features table missing",
            "table": f"{SCHEMA}.temporal_features",
            "data_missing": True
        }), 500

    if not column_exists("temporal_features", feature):
        return jsonify({
            "error": f"Column '{feature}' not found",
            "data_missing": True
        }), 404

    query = f"""
        SELECT date, MAX({feature}) AS max_val, MIN({feature}) AS min_val
        FROM {SCHEMA}.temporal_features
        GROUP BY date
        ORDER BY date ASC
    """
    try:
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
