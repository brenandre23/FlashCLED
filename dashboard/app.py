"""
app.py - CEWP Dashboard Backend
================================
FastAPI server for the Conflict Early Warning Pipeline visualization dashboard.
Connects to local PostGIS database and serves prediction/feature data as GeoJSON.

Run with: uvicorn app:app --reload --port 8000
"""

import os
import sys
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import json

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CEWP-Dashboard")

# Database schema
SCHEMA = "car_cewp"

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

# -----------------------------------------------------------------------------
# 3. H3 GEOMETRY UTILITIES
# -----------------------------------------------------------------------------
def h3_to_polygon(h3_index: int) -> Polygon:
    """
    Converts H3 index (signed int64) to Shapely Polygon.
    Handles the signed-to-unsigned conversion for h3-py.
    """
    # Convert signed int64 to unsigned for h3 library
    if h3_index < 0:
        h3_index = h3_index + 0x10000000000000000
    
    # Get boundary coordinates
    try:
        boundary = h3.cell_to_boundary(h3_index)
        # h3-py returns (lat, lng) tuples, convert to (lng, lat) for GeoJSON
        coords = [(lng, lat) for lat, lng in boundary]
        # Close the polygon
        coords.append(coords[0])
        return Polygon(coords)
    except Exception as e:
        logger.warning(f"Failed to convert H3 {h3_index}: {e}")
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
        # Fallback: try temporal_features table
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

def get_predictions(target_date: date, horizon: str = "14d") -> gpd.GeoDataFrame:
    """
    Fetches predictions for a specific date and horizon.
    Returns GeoDataFrame with H3 geometries.
    """
    # Try predictions_latest first
    query = text(f"""
        SELECT h3_index, pred_proba, pred_fatalities
        FROM {SCHEMA}.predictions_latest
        WHERE date = :target_date
          AND horizon = :horizon
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "target_date": target_date,
                "horizon": horizon
            })
    except Exception as e:
        logger.warning(f"predictions_latest query failed: {e}")
        # Fallback: Generate mock predictions from temporal_features
        df = generate_mock_predictions(target_date, horizon)
    
    if df.empty:
        df = generate_mock_predictions(target_date, horizon)
    
    # Convert H3 to geometries
    df["h3_index"] = df["h3_index"].apply(ensure_signed_h3)
    df = df.dropna(subset=["h3_index"])
    
    geometries = df["h3_index"].apply(h3_to_polygon)
    valid_mask = geometries.notna()
    
    gdf = gpd.GeoDataFrame(
        df[valid_mask].copy(),
        geometry=geometries[valid_mask].tolist(),
        crs="EPSG:4326"
    )
    
    return gdf

def generate_mock_predictions(target_date: date, horizon: str) -> pd.DataFrame:
    """
    Generates mock predictions from features_static table.
    Used when predictions_latest doesn't exist yet.
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
    
    # Generate spatially-varying mock probabilities
    np.random.seed(42)  # Reproducible
    n = len(df)
    
    # Create some spatial clustering in the mock data
    base_prob = np.random.beta(2, 10, n)  # Right-skewed distribution
    
    # Add some spatial hotspots
    hotspot_indices = np.random.choice(n, size=min(50, n//10), replace=False)
    base_prob[hotspot_indices] = np.random.beta(5, 2, len(hotspot_indices))
    
    df["pred_proba"] = np.clip(base_prob, 0, 1)
    df["pred_fatalities"] = np.where(
        df["pred_proba"] > 0.3,
        np.random.exponential(2, n),
        0
    )
    
    return df

def get_hexgrid() -> gpd.GeoDataFrame:
    """
    Fetches all H3 hexagons from features_static.
    Returns lightweight GeoDataFrame with just geometries.
    """
    query = text(f"""
        SELECT h3_index
        FROM {SCHEMA}.features_static
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    df["h3_index"] = df["h3_index"].apply(ensure_signed_h3)
    df = df.dropna(subset=["h3_index"])
    
    geometries = df["h3_index"].apply(h3_to_polygon)
    valid_mask = geometries.notna()
    
    gdf = gpd.GeoDataFrame(
        df[valid_mask][["h3_index"]].copy(),
        geometry=geometries[valid_mask].tolist(),
        crs="EPSG:4326"
    )
    
    return gdf

def get_rivers() -> Optional[gpd.GeoDataFrame]:
    """Fetches river geometries from database."""
    query = text(f"""
        SELECT 
            gid as id,
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
                
            return {
                "type": "FeatureCollection",
                "features": features
            }
    except Exception as e:
        logger.warning(f"Could not fetch rivers: {e}")
        return None

def get_roads() -> Optional[dict]:
    """Fetches road geometries from database."""
    # Try multiple possible table names
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
                    return {
                        "type": "FeatureCollection",
                        "features": features
                    }
        except Exception:
            continue
    
    logger.warning("No road data found")
    return None

def get_admin_boundaries() -> Optional[dict]:
    """Fetches administrative boundaries."""
    query = text(f"""
        SELECT 
            gid as id,
            name_1 as name,
            ST_AsGeoJSON(ST_Transform(geometry, 4326))::json as geometry
        FROM {SCHEMA}.admin_boundaries
        WHERE admin_level = 1
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            features = []
            for row in result:
                features.append({
                    "type": "Feature",
                    "properties": {"id": row[0], "name": row[1]},
                    "geometry": row[2]
                })
            
            if features:
                return {
                    "type": "FeatureCollection",
                    "features": features
                }
    except Exception as e:
        logger.warning(f"Could not fetch admin boundaries: {e}")
    
    return None

def get_conflict_events(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 1000
) -> Optional[dict]:
    """Fetches recent ACLED conflict events."""
    
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
            ST_Y(ST_Transform(geometry, 4326)) as lat,
            ST_X(ST_Transform(geometry, 4326)) as lng
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
                return {
                    "type": "FeatureCollection",
                    "features": features
                }
    except Exception as e:
        logger.warning(f"Could not fetch ACLED events: {e}")
    
    return None

# -----------------------------------------------------------------------------
# 5. FASTAPI APPLICATION
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global engine
    
    # Startup
    logger.info("Starting CEWP Dashboard API...")
    try:
        engine = get_db_engine()
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection established")
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
    description="Conflict Early Warning Pipeline - Visualization Backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for frontend
DASHBOARD_DIR = Path(__file__).parent
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
        return {"status": "healthy", "database": "connected"}
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
    Returns predictions as GeoJSON FeatureCollection.
    H3 hexagons with probability and fatality predictions.
    """
    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")
    else:
        # Use latest available date
        available = get_available_dates()
        target_date = available[0] if available else datetime.now().date()
    
    # Validate horizon
    if horizon not in ["14d", "1m", "3m"]:
        raise HTTPException(400, "Invalid horizon. Use: 14d, 1m, 3m")
    
    try:
        gdf = get_predictions(target_date, horizon)
        
        if gdf.empty:
            return {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {
                    "date": str(target_date),
                    "horizon": horizon,
                    "count": 0
                }
            }
        
        # Convert to GeoJSON
        geojson = json.loads(gdf.to_json())
        geojson["metadata"] = {
            "date": str(target_date),
            "horizon": horizon,
            "count": len(gdf),
            "prob_min": float(gdf["pred_proba"].min()),
            "prob_max": float(gdf["pred_proba"].max()),
            "prob_mean": float(gdf["pred_proba"].mean())
        }
        
        return geojson
        
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(500, f"Error fetching predictions: {str(e)}")

@app.get("/api/features/hexgrid")
async def api_hexgrid():
    """
    Returns H3 hexagon geometries as GeoJSON.
    Lightweight endpoint for base grid visualization.
    """
    try:
        gdf = get_hexgrid()
        
        if gdf.empty:
            raise HTTPException(404, "No hexgrid data found")
        
        geojson = json.loads(gdf.to_json())
        geojson["metadata"] = {
            "count": len(gdf),
            "resolution": 5
        }
        
        return geojson
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching hexgrid: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/features/static")
async def api_static_features():
    """
    Returns static infrastructure features (rivers, roads).
    """
    response = {}
    
    rivers = get_rivers()
    if rivers:
        response["rivers"] = rivers
    
    roads = get_roads()
    if roads:
        response["roads"] = roads
    
    boundaries = get_admin_boundaries()
    if boundaries:
        response["boundaries"] = boundaries
    
    if not response:
        return JSONResponse(
            status_code=404,
            content={"error": "No static features available"}
        )
    
    return response

@app.get("/api/features/rivers")
async def api_rivers():
    """Returns river geometries as GeoJSON."""
    rivers = get_rivers()
    if not rivers:
        raise HTTPException(404, "No river data found")
    return rivers

@app.get("/api/features/roads")
async def api_roads():
    """Returns road geometries as GeoJSON."""
    roads = get_roads()
    if not roads:
        raise HTTPException(404, "No road data found")
    return roads

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
        return {"type": "FeatureCollection", "features": []}
    
    return events

@app.get("/api/stats")
async def api_stats():
    """Returns summary statistics for the dashboard."""
    stats = {}
    
    # Count hexagons
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
    ║                   Starting Server...                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
