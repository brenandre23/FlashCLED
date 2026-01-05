"""
app.py - CEWP Dashboard Backend (Optimized)
=============================================
FastAPI server for the Conflict Early Warning Pipeline visualization dashboard.

OPTIMIZATIONS:
1. Direct JSON string response (avoids triple serialization)
2. Lightweight data-only endpoint for slider interactions
3. Geometry cached separately from prediction data
4. Connection pooling and query optimization

Run with: uvicorn app:app --reload --port 8000
"""

import os
import sys
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from functools import lru_cache

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import json
import orjson  # Fast JSON serialization (fallback to json if not available)

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
DASHBOARD_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CEWP-Dashboard")

# Database schema
SCHEMA = "car_cewp"

# Try to use orjson for faster serialization, fallback to json
try:
    import orjson
    def fast_json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
    logger.info("Using orjson for fast JSON serialization")
except ImportError:
    def fast_json_dumps(obj):
        return json.dumps(obj, separators=(',', ':'))
    logger.info("Using standard json (install orjson for better performance)")

# -----------------------------------------------------------------------------
# 2. DATABASE CONNECTION
# -----------------------------------------------------------------------------
def get_db_engine() -> Engine:
    """
    Creates SQLAlchemy engine from environment variables.
    Supports both Windows native and WSL environments.
    """
    load_dotenv(ENV_PATH)
    
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5433")
    db_name = os.environ.get("DB_NAME", "thesis_db")
    db_user = os.environ.get("DB_USER", "postgres")
    db_pass = os.environ.get("DB_PASS", "")
    
    # WSL detection and host resolution
    try:
        uname = os.uname()
        if "microsoft" in uname.release.lower() and db_host == "localhost":
            import subprocess
            cmd = "ip route show | awk '/default/ {print $3}'"
            gateway_ip = subprocess.check_output(cmd, shell=True).decode().strip()
            db_host = gateway_ip
            logger.info(f"WSL detected: routing to Windows host at {gateway_ip}")
    except (AttributeError, Exception):
        pass  # Not on Linux/WSL
    
    url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    logger.info(f"Connecting to database: {db_host}:{db_port}/{db_name}")
    
    return create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True
    )

# Global engine (initialized in lifespan)
engine: Optional[Engine] = None

# Geometry cache (loaded once, reused for all requests)
GEOMETRY_CACHE: Optional[str] = None  # Cached GeoJSON string
GEOMETRY_LOOKUP: Optional[Dict[int, dict]] = None  # h3_index -> geometry dict

# -----------------------------------------------------------------------------
# 3. H3 GEOMETRY UTILITIES
# -----------------------------------------------------------------------------
def h3_to_polygon(h3_index: int) -> Optional[Polygon]:
    """
    Converts H3 index (signed int64) to Shapely Polygon.
    Handles the signed-to-unsigned conversion for h3-py.
    """
    if h3_index < 0:
        h3_index = h3_index + 0x10000000000000000
    
    try:
        boundary = h3.cell_to_boundary(h3_index)
        coords = [(lng, lat) for lat, lng in boundary]
        coords.append(coords[0])
        return Polygon(coords)
    except Exception as e:
        logger.warning(f"Failed to convert H3 {h3_index}: {e}")
        return None

def h3_to_geojson_geometry(h3_index: int) -> Optional[dict]:
    """
    Converts H3 index directly to GeoJSON geometry dict.
    More efficient than going through Shapely.
    """
    if h3_index < 0:
        h3_index = h3_index + 0x10000000000000000
    
    try:
        boundary = h3.cell_to_boundary(h3_index)
        coords = [[lng, lat] for lat, lng in boundary]
        coords.append(coords[0])  # Close the polygon
        return {
            "type": "Polygon",
            "coordinates": [coords]
        }
    except Exception:
        return None

def ensure_signed_h3(h3_val) -> Optional[int]:
    """Ensures H3 index is a signed int64 (PostgreSQL BIGINT compatible)."""
    try:
        if pd.isna(h3_val):
            return None
        val = int(h3_val)
        if val > 0x7FFFFFFFFFFFFFFF:
            val = val - 0x10000000000000000
        return val
    except (ValueError, TypeError):
        return None

# -----------------------------------------------------------------------------
# 4. DATA ACCESS LAYER
# -----------------------------------------------------------------------------
def get_available_dates() -> List[date]:
    """Returns list of available prediction dates."""
    query = text(f"""
        SELECT DISTINCT date 
        FROM {SCHEMA}.predictions_latest
        ORDER BY date DESC
        LIMIT 100
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            return [row[0] for row in result]
    except Exception as e:
        logger.warning(f"Could not fetch dates from predictions_latest: {e}")
        try:
            query2 = text(f"""
                SELECT DISTINCT date 
                FROM {SCHEMA}.temporal_features
                ORDER BY date DESC
                LIMIT 100
            """)
            with engine.connect() as conn:
                result = conn.execute(query2)
                return [row[0] for row in result]
        except Exception as e2:
            logger.error(f"No date data available: {e2}")
            return []

def load_hexgrid_geometries() -> tuple[str, Dict[int, dict]]:
    """
    Loads all H3 hexagon geometries ONCE.
    Returns both GeoJSON string and lookup dictionary.
    This is called at startup and cached.
    """
    logger.info("Loading hexgrid geometries (one-time operation)...")
    
    query = text(f"""
        SELECT h3_index
        FROM {SCHEMA}.features_static
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    df["h3_index"] = df["h3_index"].apply(ensure_signed_h3)
    df = df.dropna(subset=["h3_index"])
    
    # Build features list and lookup dictionary
    features = []
    lookup = {}
    
    for _, row in df.iterrows():
        h3_idx = int(row["h3_index"])
        geom = h3_to_geojson_geometry(h3_idx)
        if geom:
            feature = {
                "type": "Feature",
                "properties": {"h3_index": h3_idx},
                "geometry": geom
            }
            features.append(feature)
            lookup[h3_idx] = geom
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "count": len(features),
            "resolution": 5
        }
    }
    
    logger.info(f"Loaded {len(features)} hexagon geometries")
    
    # Return as JSON string (pre-serialized) and lookup dict
    return fast_json_dumps(geojson), lookup

def get_prediction_data_only(target_date: date, horizon: str = "14d") -> List[dict]:
    """
    Fetches ONLY prediction data (no geometry) for a specific date and horizon.
    Returns lightweight list of dicts: [{"h3_index": ..., "pred_proba": ..., "pred_fatalities": ...}, ...]
    """
    # DB currently only stores 3m columns; serve them for any requested horizon.
    if horizon != "3m":
        logger.warning(f"Horizon '{horizon}' requested but only 3m data available; serving 3m data.")
    query = text(f"""
        SELECT 
            h3_index, 
            prob_conflict_3m AS pred_proba, 
            expected_fatalities_3m AS pred_fatalities
        FROM {SCHEMA}.predictions_latest
        WHERE date = :target_date
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "target_date": target_date
            })
    except Exception as e:
        logger.warning(f"predictions_latest query failed: {e}")
        df = pd.DataFrame()
    
    if df.empty:
        df = generate_mock_prediction_data(target_date, horizon)
    
    # Convert to list of dicts (lightweight)
    df["h3_index"] = df["h3_index"].apply(ensure_signed_h3)
    df = df.dropna(subset=["h3_index"])
    
    return df.to_dict(orient='records')

def get_predictions_with_geometry(target_date: date, horizon: str = "14d") -> str:
    """
    Returns predictions WITH geometry as pre-serialized GeoJSON string.
    Used for initial load or when geometry is needed.
    
    OPTIMIZATION: Returns raw JSON string, not parsed object.
    """
    global GEOMETRY_LOOKUP
    
    # Get prediction data
    data = get_prediction_data_only(target_date, horizon)
    
    if not data:
        return fast_json_dumps({
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"date": str(target_date), "horizon": horizon, "count": 0}
        })
    
    # If we have geometry cache, use it
    if GEOMETRY_LOOKUP:
        features = []
        for row in data:
            h3_idx = int(row["h3_index"])
            geom = GEOMETRY_LOOKUP.get(h3_idx)
            if geom:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "h3_index": h3_idx,
                        "pred_proba": row.get("pred_proba", 0),
                        "pred_fatalities": row.get("pred_fatalities", 0)
                    },
                    "geometry": geom
                })
        
        # Calculate stats
        probas = [f["properties"]["pred_proba"] for f in features]
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "date": str(target_date),
                "horizon": horizon,
                "count": len(features),
                "prob_min": min(probas) if probas else 0,
                "prob_max": max(probas) if probas else 0,
                "prob_mean": sum(probas) / len(probas) if probas else 0
            }
        }
        
        return fast_json_dumps(geojson)
    
    # Fallback: generate geometry on the fly (slower)
    features = []
    for row in data:
        h3_idx = int(row["h3_index"])
        geom = h3_to_geojson_geometry(h3_idx)
        if geom:
            features.append({
                "type": "Feature",
                "properties": {
                    "h3_index": h3_idx,
                    "pred_proba": row.get("pred_proba", 0),
                    "pred_fatalities": row.get("pred_fatalities", 0)
                },
                "geometry": geom
            })
    
    probas = [f["properties"]["pred_proba"] for f in features]
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "date": str(target_date),
            "horizon": horizon,
            "count": len(features),
            "prob_min": min(probas) if probas else 0,
            "prob_max": max(probas) if probas else 0,
            "prob_mean": sum(probas) / len(probas) if probas else 0
        }
    }
    
    return fast_json_dumps(geojson)

def generate_mock_prediction_data(target_date: date, horizon: str) -> pd.DataFrame:
    """
    Generates mock prediction DATA (no geometry) from features_static table.
    """
    query = text(f"""
        SELECT h3_index
        FROM {SCHEMA}.features_static
        LIMIT 5000
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    except Exception as e:
        logger.error(f"Could not generate mock predictions: {e}")
        return pd.DataFrame(columns=["h3_index", "pred_proba", "pred_fatalities"])
    
    if df.empty:
        return pd.DataFrame(columns=["h3_index", "pred_proba", "pred_fatalities"])
    
    # Seed based on date for consistent results
    seed = int(target_date.strftime("%Y%m%d")) if target_date else 42
    np.random.seed(seed)
    n = len(df)
    
    # Create spatially-varying mock probabilities
    base_prob = np.random.beta(2, 10, n)
    
    # Add hotspots
    hotspot_indices = np.random.choice(n, size=min(50, n//10), replace=False)
    base_prob[hotspot_indices] = np.random.beta(5, 2, len(hotspot_indices))
    
    # Adjust by horizon (longer = lower certainty, spread out)
    if horizon == "1m":
        base_prob = base_prob * 0.9 + np.random.uniform(0, 0.1, n)
    elif horizon == "3m":
        base_prob = base_prob * 0.8 + np.random.uniform(0, 0.2, n)
    
    df["pred_proba"] = np.clip(base_prob, 0, 1)
    df["pred_fatalities"] = np.where(
        df["pred_proba"] > 0.3,
        np.random.exponential(2, n),
        0
    )
    
    return df

def get_rivers() -> Optional[str]:
    """Fetches river geometries as pre-serialized GeoJSON string."""
    query = text(f"""
        SELECT 
            hyriv_id as id,
            ST_AsGeoJSON(ST_Transform(geometry, 4326))::json as geometry
        FROM {SCHEMA}.rivers
        LIMIT 10000
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            features = []
            for row in result:
                features.append({
                    "type": "Feature",
                    "properties": {"id": row[0]},
                    "geometry": row[1]
                })
            
            if not features:
                return None
            
            return fast_json_dumps({
                "type": "FeatureCollection",
                "features": features
            })
    except Exception as e:
        logger.warning(f"Could not fetch rivers: {e}")
        return None

def get_roads() -> Optional[str]:
    """Fetches road geometries as pre-serialized GeoJSON string."""
    for table_name in ["grip4_roads_h3", "roads", "grip4_roads"]:
        query = text(f"""
            SELECT 
                gid as id,
                ST_AsGeoJSON(ST_Transform(geometry, 4326))::json as geometry
            FROM {SCHEMA}.{table_name}
            LIMIT 5000
        """)
        
        try:
            with engine.connect() as conn:
                result = conn.execute(query)
                features = []
                for row in result:
                    features.append({
                        "type": "Feature",
                        "properties": {"id": row[0]},
                        "geometry": row[1]
                    })
                
                if features:
                    return fast_json_dumps({
                        "type": "FeatureCollection",
                        "features": features
                    })
        except Exception:
            continue
    
    logger.warning("No road data found")
    return None

def get_conflict_events(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 1000
) -> Optional[str]:
    """Fetches recent ACLED conflict events as pre-serialized GeoJSON string."""
    
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=90)
    
    query = text(f"""
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
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit
            })
            
            features = []
            for row in result:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "id": row[0],
                        "date": str(row[1]),
                        "event_type": row[2],
                        "sub_event_type": row[3],
                        "fatalities": row[4]
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row[6], row[5]]
                    }
                })
            
            if features:
                return fast_json_dumps({
                    "type": "FeatureCollection",
                    "features": features
                })
    except Exception as e:
        logger.warning(f"Could not fetch ACLED events: {e}")
    
    return None

# -----------------------------------------------------------------------------
# 5. FASTAPI APPLICATION
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global engine, GEOMETRY_CACHE, GEOMETRY_LOOKUP
    
    # Startup
    logger.info("Starting CEWP Dashboard API...")
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection established")
        
        # Pre-load geometry cache
        try:
            GEOMETRY_CACHE, GEOMETRY_LOOKUP = load_hexgrid_geometries()
            logger.info("Geometry cache loaded successfully")
        except Exception as e:
            logger.warning(f"Could not pre-load geometry cache: {e}")
            GEOMETRY_CACHE = None
            GEOMETRY_LOOKUP = None
            
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if engine:
        engine.dispose()

app = FastAPI(
    title="CEWP Dashboard API",
    description="Conflict Early Warning Pipeline - Visualization Backend (Optimized)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files - mount BEFORE routes
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

# -----------------------------------------------------------------------------
# 6. API ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """Serve the main dashboard page."""
    return FileResponse(DASHBOARD_DIR / "index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "geometry_cached": GEOMETRY_CACHE is not None,
            "cached_hexagons": len(GEOMETRY_LOOKUP) if GEOMETRY_LOOKUP else 0
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/api/dates")
async def get_dates():
    """Returns available prediction dates."""
    dates = get_available_dates()
    return {
        "dates": [str(d) for d in dates],
        "latest": str(dates[0]) if dates else None,
        "earliest": str(dates[-1]) if dates else None
    }

@app.get("/api/predictions")
async def api_predictions(
    date: Optional[str] = Query(None, description="Target date (YYYY-MM-DD)"),
    horizon: str = Query("14d", description="Forecast horizon: 14d, 1m, 3m")
):
    """
    Returns predictions WITH geometry as GeoJSON.
    
    OPTIMIZATION: Returns pre-serialized JSON string directly.
    Avoids FastAPI's automatic JSON serialization overhead.
    """
    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")
    else:
        available = get_available_dates()
        target_date = available[0] if available else datetime.now().date()
    
    # Validate horizon
    if horizon not in ["14d", "1m", "3m"]:
        raise HTTPException(400, "Invalid horizon. Use: 14d, 1m, 3m")
    
    try:
        # Get pre-serialized JSON string
        json_str = get_predictions_with_geometry(target_date, horizon)
        
        # Return raw JSON response (no re-serialization!)
        return Response(
            content=json_str,
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(500, f"Error fetching predictions: {str(e)}")

@app.get("/api/data/predictions")
async def api_prediction_data(
    date: Optional[str] = Query(None, description="Target date (YYYY-MM-DD)"),
    horizon: str = Query("14d", description="Forecast horizon: 14d, 1m, 3m")
):
    """
    LIGHTWEIGHT ENDPOINT: Returns ONLY prediction data (no geometry).
    
    Use this endpoint for time slider interactions to minimize data transfer.
    Frontend should join this data with cached geometry.
    
    Returns: [{"h3_index": 123, "pred_proba": 0.5, "pred_fatalities": 1.2}, ...]
    """
    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")
    else:
        available = get_available_dates()
        target_date = available[0] if available else datetime.now().date()
    
    # Validate horizon
    if horizon not in ["14d", "1m", "3m"]:
        raise HTTPException(400, "Invalid horizon. Use: 14d, 1m, 3m")
    
    try:
        data = get_prediction_data_only(target_date, horizon)
        
        # Calculate stats
        probas = [d.get("pred_proba", 0) for d in data]
        
        response = {
            "data": data,
            "metadata": {
                "date": str(target_date),
                "horizon": horizon,
                "count": len(data),
                "prob_min": min(probas) if probas else 0,
                "prob_max": max(probas) if probas else 0,
                "prob_mean": sum(probas) / len(probas) if probas else 0
            }
        }
        
        # Return pre-serialized
        return Response(
            content=fast_json_dumps(response),
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error fetching prediction data: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/features/hexgrid")
async def api_hexgrid():
    """
    Returns H3 hexagon geometries as GeoJSON.
    
    OPTIMIZATION: Returns cached pre-serialized JSON string.
    This endpoint should be called ONCE on page load.
    """
    global GEOMETRY_CACHE
    
    if GEOMETRY_CACHE:
        return Response(
            content=GEOMETRY_CACHE,
            media_type="application/json"
        )
    
    # Fallback: generate on the fly
    try:
        GEOMETRY_CACHE, GEOMETRY_LOOKUP = load_hexgrid_geometries()
        return Response(
            content=GEOMETRY_CACHE,
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Error fetching hexgrid: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/features/static")
async def api_static_features():
    """Returns static infrastructure features (rivers, roads)."""
    response_data = {}
    
    rivers = get_rivers()
    if rivers:
        response_data["rivers"] = json.loads(rivers)
    
    roads = get_roads()
    if roads:
        response_data["roads"] = json.loads(roads)
    
    if not response_data:
        return JSONResponse(
            status_code=404,
            content={"error": "No static features available"}
        )
    
    return response_data

@app.get("/api/features/rivers")
async def api_rivers():
    """Returns river geometries as GeoJSON."""
    rivers = get_rivers()
    if not rivers:
        raise HTTPException(404, "No river data found")
    return Response(content=rivers, media_type="application/json")

@app.get("/api/features/roads")
async def api_roads():
    """Returns road geometries as GeoJSON."""
    roads = get_roads()
    if not roads:
        raise HTTPException(404, "No road data found")
    return Response(content=roads, media_type="application/json")

@app.get("/api/events")
async def api_events(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(1000, le=5000)
):
    """Returns recent conflict events as GeoJSON."""
    start = None
    end = None
    
    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Invalid start_date format")
    
    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Invalid end_date format")
    
    events = get_conflict_events(start, end, limit)
    if not events:
        return Response(
            content=fast_json_dumps({"type": "FeatureCollection", "features": []}),
            media_type="application/json"
        )
    
    return Response(content=events, media_type="application/json")

@app.get("/api/stats")
async def api_stats():
    """Returns summary statistics for the dashboard."""
    stats = {}
    
    # Use cached count if available
    if GEOMETRY_LOOKUP:
        stats["hexagon_count"] = len(GEOMETRY_LOOKUP)
    else:
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.features_static"))
                stats["hexagon_count"] = result.scalar()
        except Exception:
            stats["hexagon_count"] = None
    
    # Count events
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.acled_events"))
            stats["event_count"] = result.scalar()
    except Exception:
        stats["event_count"] = None
    
    # Date range
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT MIN(event_date), MAX(event_date) 
                FROM {SCHEMA}.acled_events
            """))
            row = result.fetchone()
            if row:
                stats["date_range"] = {
                    "start": str(row[0]) if row[0] else None,
                    "end": str(row[1]) if row[1] else None
                }
    except Exception:
        stats["date_range"] = None
    
    return stats

# -----------------------------------------------------------------------------
# 7. MAIN ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         CEWP Dashboard - Conflict Early Warning           ║
    ║              Optimized Performance Edition                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
