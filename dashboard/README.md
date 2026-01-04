# CEWP Dashboard

Interactive web dashboard for the Conflict Early Warning Pipeline (CEWP). Visualizes conflict risk predictions across the Central African Republic using a high-performance H3 hexagonal grid.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  MapLibre   │  │   Deck.gl   │  │       Chart.js          │ │
│  │  (Base Map) │  │ (H3 Hexes)  │  │  (Performance Metrics)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ REST API
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ /api/predict │  │ /api/features│  │ /api/events          │  │
│  │              │  │  /static     │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ SQL
┌─────────────────────────────────────────────────────────────────┐
│                    PostGIS Database                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ predictions  │  │ features_    │  │    acled_events      │  │
│  │  _latest     │  │   static     │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI | REST API server |
| Database | PostgreSQL + PostGIS | Spatial data storage |
| Base Map | MapLibre GL JS | Open-source map rendering |
| Hex Layer | Deck.gl | High-performance WebGL visualization |
| Tile Source | CartoDB Positron | Free raster tiles (no API key) |
| Charts | Chart.js | Performance metrics visualization |
| Styling | Tailwind CSS | Responsive UI framework |

## Quick Start

### 1. Install Dependencies

```bash
cd dashboard
pip install -r requirements_dashboard.txt
```

### 2. Configure Database Connection

Ensure your `.env` file (in the project root) contains:

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=thesis_db
DB_USER=postgres
DB_PASS=your_password
```

### 3. Run the Server

```bash
# From the dashboard directory
python app.py

# Or using uvicorn directly
uvicorn app:app --reload --port 8000
```

### 4. Open the Dashboard

Navigate to: **http://localhost:8000**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve dashboard HTML |
| `/api/health` | GET | Health check |
| `/api/dates` | GET | Available prediction dates |
| `/api/predictions` | GET | Prediction data (GeoJSON) |
| `/api/features/static` | GET | Rivers, roads, boundaries |
| `/api/features/hexgrid` | GET | H3 grid geometries |
| `/api/events` | GET | ACLED conflict events |
| `/api/stats` | GET | Summary statistics |

### Query Parameters

**`/api/predictions`**
- `date` (optional): Target date (YYYY-MM-DD)
- `horizon` (optional): Forecast horizon (`14d`, `1m`, `3m`)

**`/api/events`**
- `start_date` (optional): Filter start (YYYY-MM-DD)
- `end_date` (optional): Filter end (YYYY-MM-DD)
- `limit` (optional): Max results (default: 1000)

## Database Schema

The dashboard expects these tables in the `car_cewp` schema:

### Required Tables

```sql
-- Predictions (if available)
CREATE TABLE car_cewp.predictions_latest (
    h3_index BIGINT,
    date DATE,
    horizon TEXT,
    pred_proba FLOAT,
    pred_fatalities FLOAT,
    PRIMARY KEY (h3_index, date, horizon)
);

-- Static features (required)
CREATE TABLE car_cewp.features_static (
    h3_index BIGINT PRIMARY KEY,
    geometry GEOMETRY(Polygon, 4326),
    -- ... other features
);

-- Optional: Rivers
CREATE TABLE car_cewp.rivers (
    gid SERIAL PRIMARY KEY,
    geometry GEOMETRY(LineString, 4326)
);

-- Optional: ACLED Events
CREATE TABLE car_cewp.acled_events (
    event_id_cnty TEXT PRIMARY KEY,
    event_date DATE,
    event_type TEXT,
    fatalities INTEGER,
    geometry GEOMETRY(Point, 4326)
);
```

## Features

### Interactive Map
- **H3 Hexagon Layer**: Color-coded by conflict probability
- **Layer Toggles**: Show/hide rivers, roads, conflict events
- **Hover Tooltips**: View cell-level predictions
- **Zoom Controls**: Navigate the map

### Time Navigation
- **Date Picker**: Select specific forecast dates
- **Time Slider**: Navigate through available dates
- **Horizon Selector**: Toggle between 14d/1m/3m forecasts

### Performance Dashboard
- **PR-AUC Comparison**: XGBoost vs LightGBM
- **Operational Recall**: Top-10% capture rate
- **Data Source Registry**: Interactive exploration

## Development

### File Structure

```
dashboard/
├── app.py                    # FastAPI backend
├── index.html                # Main HTML page
├── script.js                 # Frontend JavaScript
├── style.css                 # Custom styles
├── requirements_dashboard.txt # Python dependencies
└── README.md                 # This file
```

### Adding New Endpoints

```python
@app.get("/api/custom")
async def custom_endpoint(param: str = Query(None)):
    # Your logic here
    return {"result": "data"}
```

### Modifying Map Layers

Edit `script.js` and add/modify layer creation functions:

```javascript
function createCustomLayer() {
    return new deck.GeoJsonLayer({
        id: 'custom-layer',
        data: STATE.customData,
        // ... layer options
    });
}
```

## Troubleshooting

### "Database connection failed"
- Verify PostgreSQL is running
- Check `.env` credentials
- For WSL: ensure Windows host IP is routable

### "No prediction data"
- Dashboard generates mock data if `predictions_latest` doesn't exist
- Run the modeling pipeline to generate actual predictions

### "Map not loading"
- Check browser console for errors
- Verify CDN scripts are accessible
- Try clearing browser cache

## License

Part of the CEWP Master's Thesis Project (2025).
