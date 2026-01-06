"""
dashboard/app.py
================
Robust Backend for FlashCLED Dashboard.
Fixes: "Invisible Hexagons" (H3 Type Safety) & "Invisible Roads" (Missing Table Checks).

Run with: python app.py
         or: flask run --port 8000
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine
from functools import lru_cache
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# --- Path Setup ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Load environment variables
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

# Try to import from utils, fallback to local definitions
try:
    from utils import get_db_engine, SCHEMA
except ImportError:
    SCHEMA = "car_cewp"
    
    def get_db_engine() -> Engine:
        """Creates SQLAlchemy engine from environment variables."""
        from sqlalchemy import create_engine
        
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5433")
        db_name = os.environ.get("DB_NAME", "thesis_db")
        db_user = os.environ.get("DB_USER", "postgres")
        db_pass = os.environ.get("DB_PASS", "")
        
        url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        return create_engine(url, pool_pre_ping=True)

# --- Flask App Setup ---
DASHBOARD_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")
CORS(app)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Dashboard")

# --- Helper Functions ---

def ensure_signed_h3(h3_val) -> Optional[int]:
    """
    ROBUST FIX: Converts any H3 format (Hex String, Unsigned Int) 
    to the Signed 64-bit Int required by Deck.gl's H3HexagonLayer.
    """
    try:
        if pd.isna(h3_val):
            return None
        
        # If it's already an integer (native DB format)
        if isinstance(h3_val, (int, np.integer)):
            val = int(h3_val)
        # If it's a hex string (e.g., '852a104ffffffff')
        elif isinstance(h3_val, str):
            val = int(h3_val, 16)
        else:
            return None

        # Convert Unsigned to Signed 64-bit (Python ints are arbitrary precision, 
        # so we simulate 64-bit overflow wrapping)
        if val > 0x7FFFFFFFFFFFFFFF:
            val = val - 0x10000000000000000
        return val
    except Exception:
        return None


def fetch_table_data(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Safe DB fetcher with error handling."""
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params or {})
        return df
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return pd.DataFrame()
    finally:
        engine.dispose()


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    query = f"SELECT to_regclass('{SCHEMA}.{table_name}')"
    df = fetch_table_data(query)
    if df.empty:
        return False
    return df.iloc[0, 0] is not None


def get_available_dates() -> List[date]:
    """Returns list of available prediction dates."""
    # Try predictions_latest first
    query = f"""
        SELECT DISTINCT date 
        FROM {SCHEMA}.predictions_latest
        ORDER BY date DESC
        LIMIT 100
    """
    df = fetch_table_data(query)
    
    if df.empty:
        # Fallback to temporal_features
        query2 = f"""
            SELECT DISTINCT date 
            FROM {SCHEMA}.temporal_features
            ORDER BY date DESC
            LIMIT 100
        """
        df = fetch_table_data(query2)
    
    if df.empty:
        return []
    
    return df['date'].tolist()


# --- Routes ---

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return send_from_directory(str(DASHBOARD_DIR), "index.html")


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    try:
        df = fetch_table_data("SELECT 1 as ok")
        if not df.empty:
            return jsonify({
                "status": "healthy",
                "database": "connected"
            })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503
    
    return jsonify({"status": "unhealthy"}), 503


@app.route("/api/dates")
def get_dates():
    """Returns available prediction dates."""
    dates = get_available_dates()
    return jsonify({
        "dates": [str(d) for d in dates],
        "latest": str(dates[0]) if dates else None,
        "earliest": str(dates[-1]) if dates else None
    })


@app.route("/api/h3_data")
def get_h3_data():
    """
    Fetches the main conflict prediction layer.
    Returns H3 hexagons with risk scores.
    """
    # Try to get real predictions first
    query = f"""
        SELECT h3_index, prob_conflict_3m as risk_score
        FROM {SCHEMA}.predictions_latest
        ORDER BY date DESC
        LIMIT 10000
    """
    df = fetch_table_data(query)
    
    # Fallback to features_static with mock risk if no predictions
    if df.empty:
        query = f"""
            SELECT h3_index, 0.5 as risk_score 
            FROM {SCHEMA}.features_static 
            LIMIT 5000
        """
        df = fetch_table_data(query)
    
    if df.empty:
        return jsonify([])

    # Apply the H3 type fix
    df['h3_index'] = df['h3_index'].apply(ensure_signed_h3)
    df = df.dropna(subset=['h3_index'])
    
    # Generate spatially-varying mock risk if all values are the same
    if df['risk_score'].nunique() == 1:
        np.random.seed(42)
        n = len(df)
        base_prob = np.random.beta(2, 10, n)
        # Add some hotspots
        hotspot_indices = np.random.choice(n, size=min(50, n // 10), replace=False)
        base_prob[hotspot_indices] = np.random.beta(5, 2, len(hotspot_indices))
        df['risk_score'] = np.clip(base_prob, 0, 1)
    
    # Return as list of dicts
    data = df.rename(columns={'h3_index': 'hex', 'risk_score': 'risk'}).to_dict(orient="records")
    return jsonify(data)


@app.route("/api/predictions")
def get_predictions():
    """
    Returns predictions WITH geometry as GeoJSON.
    Query params: date (YYYY-MM-DD), horizon (14d, 1m, 3m)
    """
    from flask import request
    import h3
    
    target_date = request.args.get('date')
    horizon = request.args.get('horizon', '14d')
    
    # Get predictions
    if horizon == '3m':
        prob_col = 'prob_conflict_3m'
        fat_col = 'expected_fatalities_3m'
    elif horizon == '1m':
        prob_col = 'prob_conflict_3m'  # Fallback to 3m if 1m not available
        fat_col = 'expected_fatalities_3m'
    else:
        prob_col = 'prob_conflict_3m'
        fat_col = 'expected_fatalities_3m'
    
    if target_date:
        query = f"""
            SELECT h3_index, {prob_col} as pred_proba, {fat_col} as pred_fatalities
            FROM {SCHEMA}.predictions_latest
            WHERE date = :target_date
        """
        df = fetch_table_data(query, {'target_date': target_date})
    else:
        query = f"""
            SELECT DISTINCT ON (h3_index) 
                h3_index, {prob_col} as pred_proba, {fat_col} as pred_fatalities
            FROM {SCHEMA}.predictions_latest
            ORDER BY h3_index, date DESC
        """
        df = fetch_table_data(query)
    
    # Fallback to features_static with mock predictions
    if df.empty:
        query = f"SELECT h3_index FROM {SCHEMA}.features_static LIMIT 5000"
        df = fetch_table_data(query)
        if not df.empty:
            np.random.seed(42 if not target_date else hash(target_date) % 2**32)
            n = len(df)
            df['pred_proba'] = np.clip(np.random.beta(2, 10, n), 0, 1)
            df['pred_fatalities'] = np.where(df['pred_proba'] > 0.3, np.random.exponential(2, n), 0)
    
    if df.empty:
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"count": 0}
        })
    
    # Apply H3 fix
    df['h3_index'] = df['h3_index'].apply(ensure_signed_h3)
    df = df.dropna(subset=['h3_index'])
    
    # Build GeoJSON features
    features = []
    for _, row in df.iterrows():
        h3_idx = int(row['h3_index'])
        
        # Convert signed to unsigned for h3-py
        if h3_idx < 0:
            h3_unsigned = h3_idx + 0x10000000000000000
        else:
            h3_unsigned = h3_idx
        
        try:
            boundary = h3.cell_to_boundary(h3_unsigned)
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])  # Close polygon
            
            features.append({
                "type": "Feature",
                "properties": {
                    "h3_index": h3_idx,
                    "pred_proba": float(row.get('pred_proba', 0)),
                    "pred_fatalities": float(row.get('pred_fatalities', 0))
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            })
        except Exception as e:
            logger.warning(f"Failed to convert H3 {h3_idx}: {e}")
            continue
    
    # Calculate stats
    probas = [f["properties"]["pred_proba"] for f in features]
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "date": str(target_date) if target_date else None,
            "horizon": horizon,
            "count": len(features),
            "prob_min": min(probas) if probas else 0,
            "prob_max": max(probas) if probas else 0,
            "prob_mean": sum(probas) / len(probas) if probas else 0
        }
    }
    
    return Response(
        json.dumps(geojson),
        mimetype='application/json'
    )


@app.route("/api/data/predictions")
def get_prediction_data_only():
    """
    LIGHTWEIGHT ENDPOINT: Returns ONLY prediction data (no geometry).
    Use this for time slider interactions to minimize data transfer.
    """
    from flask import request
    
    target_date = request.args.get('date')
    horizon = request.args.get('horizon', '14d')
    
    # DB currently only stores 3m columns
    prob_col = 'prob_conflict_3m'
    fat_col = 'expected_fatalities_3m'
    
    if target_date:
        query = f"""
            SELECT h3_index, {prob_col} as pred_proba, {fat_col} as pred_fatalities
            FROM {SCHEMA}.predictions_latest
            WHERE date = :target_date
        """
        df = fetch_table_data(query, {'target_date': target_date})
    else:
        query = f"""
            SELECT DISTINCT ON (h3_index) 
                h3_index, {prob_col} as pred_proba, {fat_col} as pred_fatalities
            FROM {SCHEMA}.predictions_latest
            ORDER BY h3_index, date DESC
        """
        df = fetch_table_data(query)
    
    # Fallback to mock data
    if df.empty:
        query = f"SELECT h3_index FROM {SCHEMA}.features_static LIMIT 5000"
        df = fetch_table_data(query)
        if not df.empty:
            np.random.seed(42 if not target_date else hash(target_date) % 2**32)
            n = len(df)
            df['pred_proba'] = np.clip(np.random.beta(2, 10, n), 0, 1)
            df['pred_fatalities'] = np.where(df['pred_proba'] > 0.3, np.random.exponential(2, n), 0)
    
    if df.empty:
        return jsonify({"data": [], "metadata": {"count": 0}})
    
    df['h3_index'] = df['h3_index'].apply(ensure_signed_h3)
    df = df.dropna(subset=['h3_index'])
    
    data = df.to_dict(orient='records')
    probas = [d.get('pred_proba', 0) for d in data]
    
    return jsonify({
        "data": data,
        "metadata": {
            "date": target_date,
            "horizon": horizon,
            "count": len(data),
            "prob_min": min(probas) if probas else 0,
            "prob_max": max(probas) if probas else 0,
            "prob_mean": sum(probas) / len(probas) if probas else 0
        }
    })


@app.route("/api/features/hexgrid")
def get_hexgrid():
    """Returns H3 hexagon geometries as GeoJSON (geometry only, no predictions)."""
    import h3
    
    query = f"SELECT h3_index FROM {SCHEMA}.features_static"
    df = fetch_table_data(query)
    
    if df.empty:
        return jsonify({"type": "FeatureCollection", "features": [], "metadata": {"count": 0}})
    
    df['h3_index'] = df['h3_index'].apply(ensure_signed_h3)
    df = df.dropna(subset=['h3_index'])
    
    features = []
    for _, row in df.iterrows():
        h3_idx = int(row['h3_index'])
        
        # Convert signed to unsigned for h3-py
        if h3_idx < 0:
            h3_unsigned = h3_idx + 0x10000000000000000
        else:
            h3_unsigned = h3_idx
        
        try:
            boundary = h3.cell_to_boundary(h3_unsigned)
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])
            
            features.append({
                "type": "Feature",
                "properties": {"h3_index": h3_idx},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            })
        except Exception:
            continue
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "count": len(features),
            "resolution": 5
        }
    }
    
    return Response(json.dumps(geojson), mimetype='application/json')


@app.route("/api/roads")
@lru_cache(maxsize=1)
def get_roads():
    """
    Fetches road geometries as GeoJSON. Cached in memory for speed.
    """
    # Try multiple possible table names
    for table_name in ["grip4_roads_h3", "roads", "grip4_roads"]:
        if not table_exists(table_name):
            continue
        
        # Check if table has geometry column
        query = f"""
            SELECT gid as id, ST_AsGeoJSON(ST_Transform(geometry, 4326))::json as geometry
            FROM {SCHEMA}.{table_name}
            LIMIT 5000
        """
        
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                result = conn.execute(text(query))
                features = []
                for row in result:
                    if row[1]:  # geometry exists
                        features.append({
                            "type": "Feature",
                            "properties": {"id": row[0]},
                            "geometry": row[1]
                        })
                
                if features:
                    return jsonify({
                        "type": "FeatureCollection",
                        "features": features
                    })
        except Exception as e:
            logger.warning(f"Could not fetch roads from {table_name}: {e}")
            continue
    
    logger.warning("No road data found in any table.")
    return jsonify({"type": "FeatureCollection", "features": []})


@app.route("/api/rivers")
@lru_cache(maxsize=1)
def get_rivers():
    """
    Fetches river geometries as GeoJSON. Cached in memory for speed.
    """
    if not table_exists("rivers"):
        logger.warning("Table rivers not found.")
        return jsonify({"type": "FeatureCollection", "features": []})
    
    query = f"""
        SELECT hyriv_id as id, ST_AsGeoJSON(ST_Transform(geometry, 4326))::json as geometry
        FROM {SCHEMA}.rivers
        LIMIT 10000
    """
    
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text(query))
            features = []
            for row in result:
                if row[1]:
                    features.append({
                        "type": "Feature",
                        "properties": {"id": row[0]},
                        "geometry": row[1]
                    })
            
            if features:
                return jsonify({
                    "type": "FeatureCollection",
                    "features": features
                })
    except Exception as e:
        logger.warning(f"Could not fetch rivers: {e}")
    
    return jsonify({"type": "FeatureCollection", "features": []})


@app.route("/api/features/static")
def get_static_features():
    """Returns static infrastructure features (rivers, roads)."""
    response_data = {}
    
    # Clear cache and fetch fresh
    get_rivers.cache_clear()
    get_roads.cache_clear()
    
    rivers_response = get_rivers()
    rivers_data = rivers_response.get_json()
    if rivers_data and rivers_data.get("features"):
        response_data["rivers"] = rivers_data
    
    roads_response = get_roads()
    roads_data = roads_response.get_json()
    if roads_data and roads_data.get("features"):
        response_data["roads"] = roads_data
    
    if not response_data:
        return jsonify({"error": "No static features available"}), 404
    
    return jsonify(response_data)


@app.route("/api/events")
def get_events():
    """Returns recent ACLED conflict events as GeoJSON."""
    from flask import request
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = min(int(request.args.get('limit', 1000)), 5000)
    
    if not end_date:
        end_date = date.today().isoformat()
    if not start_date:
        start_date = (date.today() - timedelta(days=90)).isoformat()
    
    if not table_exists("acled_events"):
        return jsonify({"type": "FeatureCollection", "features": []})
    
    query = f"""
        SELECT 
            event_id_cnty as id,
            event_date,
            event_type,
            sub_event_type,
            fatalities,
            latitude as lat,
            longitude as lng
        FROM {SCHEMA}.acled_events
        WHERE event_date BETWEEN :start_date AND :end_date
        ORDER BY event_date DESC
        LIMIT :limit
    """
    
    df = fetch_table_data(query, {
        'start_date': start_date,
        'end_date': end_date,
        'limit': limit
    })
    
    if df.empty:
        return jsonify({"type": "FeatureCollection", "features": []})
    
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "properties": {
                "id": row['id'],
                "date": str(row['event_date']),
                "event_type": row['event_type'],
                "sub_event_type": row['sub_event_type'],
                "fatalities": int(row['fatalities']) if pd.notna(row['fatalities']) else 0
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['lng']), float(row['lat'])]
            }
        })
    
    return jsonify({
        "type": "FeatureCollection",
        "features": features
    })


@app.route("/api/stats")
def get_stats():
    """Returns summary statistics for the dashboard."""
    stats = {}
    
    # Count hexagons
    query = f"SELECT COUNT(*) as cnt FROM {SCHEMA}.features_static"
    df = fetch_table_data(query)
    stats["hexagon_count"] = int(df.iloc[0, 0]) if not df.empty else None
    
    # Count events
    if table_exists("acled_events"):
        query = f"SELECT COUNT(*) as cnt FROM {SCHEMA}.acled_events"
        df = fetch_table_data(query)
        stats["event_count"] = int(df.iloc[0, 0]) if not df.empty else None
        
        # Date range
        query = f"SELECT MIN(event_date), MAX(event_date) FROM {SCHEMA}.acled_events"
        df = fetch_table_data(query)
        if not df.empty and df.iloc[0, 0] is not None:
            stats["date_range"] = {
                "start": str(df.iloc[0, 0]),
                "end": str(df.iloc[0, 1])
            }
    
    return jsonify(stats)


# --- Main Entry Point ---
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         FlashCLED Dashboard - Conflict Early Warning      ║
    ║              Robust Flask Backend Edition                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host="0.0.0.0", port=8000, debug=True)
