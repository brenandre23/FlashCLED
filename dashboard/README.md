# FlashCLED Dashboard

Interactive web dashboard for the Conflict Early Warning Pipeline (CEWP). Visualizes conflict risk predictions across the Central African Republic using a high-performance H3 hexagonal grid.

## Architecture (Refactored)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  MapLibre   │  │   Deck.gl        │  │   Responsive      │  │
│  │  (Base Map) │  │  MapboxOverlay   │  │   Sidebar UI      │  │
│  └─────────────┘  └──────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ REST API
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (Flask)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ /api/h3_data │  │ /api/roads   │  │ /api/events          │  │
│  │              │  │ /api/rivers  │  │                      │  │
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

## Key Improvements (v2.0)

| Issue | Fix |
|-------|-----|
| Invisible Hexagons | Robust `ensure_signed_h3()` handles hex-string vs signed-int mismatches |
| Fragile Connectivity | JSON error responses instead of crashes |
| Rigid Layout | Flexbox sidebar + map-container responsive design |
| Deck.gl Integration | MapboxOverlay (modern standard for Deck.gl + MapLibre) |
| Static Data Performance | `lru_cache` on roads/rivers endpoints |

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | **Flask** | REST API server |
| Database | PostgreSQL + PostGIS | Spatial data storage |
| Base Map | MapLibre GL JS | Open-source map rendering |
| Hex Layer | Deck.gl + H3HexagonLayer | High-performance WebGL visualization |
| Tile Source | CARTO Dark Matter | Free vector tiles (no API key) |
| Styling | Custom CSS (Flexbox) | Responsive UI |

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
```

Server runs on: **http://localhost:8000**

### 4. Open the Dashboard

Navigate to: **http://localhost:8000**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve dashboard HTML |
| `/api/health` | GET | Health check |
| `/api/h3_data` | GET | H3 hexagons with risk scores |
| `/api/predictions` | GET | Full predictions with geometry (GeoJSON) |
| `/api/data/predictions` | GET | Lightweight predictions (data only) |
| `/api/features/hexgrid` | GET | H3 grid geometries only |
| `/api/roads` | GET | Road geometries (cached) |
| `/api/rivers` | GET | River geometries (cached) |
| `/api/events` | GET | ACLED conflict events |
| `/api/stats` | GET | Summary statistics |

### Query Parameters

**`/api/predictions`**
- `date` (optional): Target date (YYYY-MM-DD)
- `horizon` (optional): Forecast horizon (`14d`, `1m`, `3m`)

**`/api/events`**
- `start_date` (optional): Filter start (YYYY-MM-DD)
- `end_date` (optional): Filter end (YYYY-MM-DD)
- `limit` (optional): Max results (default: 1000, max: 5000)

## Features

### Interactive Map
- **H3 Hexagon Layer**: Color-coded by conflict probability (YlOrRd scale)
- **Layer Toggles**: Show/hide roads, rivers, conflict events
- **Hover Tooltips**: View cell-level predictions
- **Responsive Sidebar**: Controls and legend

### Layer Controls
- ✅ Conflict Risk (H3) - Hexagonal risk visualization
- ☐ GRIP4 Roads - Transportation network
- ☐ Rivers - Hydrological features  
- ☐ Conflict Events - ACLED point events

### Status Indicators
- 🟢 Connected - API healthy
- 🟡 Connecting - Loading
- 🔴 Disconnected - API unavailable

## File Structure

```
dashboard/
├── app.py                    # Flask backend (refactored)
├── index.html                # Simplified responsive layout
├── script.js                 # MapboxOverlay + async/await
├── style.css                 # Flexbox architecture
├── requirements_dashboard.txt # Python dependencies
└── README.md                 # This file
```

## Database Schema

The dashboard expects these tables in the `car_cewp` schema:

### Required Tables

```sql
-- Static features (required for hexgrid)
CREATE TABLE car_cewp.features_static (
    h3_index BIGINT PRIMARY KEY
);

-- Predictions (optional - mock data generated if missing)
CREATE TABLE car_cewp.predictions_latest (
    h3_index BIGINT,
    date DATE,
    prob_conflict_3m FLOAT,
    expected_fatalities_3m FLOAT,
    PRIMARY KEY (h3_index, date)
);
```

### Optional Tables

```sql
-- Rivers
CREATE TABLE car_cewp.rivers (
    hyriv_id INTEGER PRIMARY KEY,
    geometry GEOMETRY(LineString, 32634)
);

-- Roads (tries multiple table names)
CREATE TABLE car_cewp.grip4_roads_h3 (
    gid SERIAL PRIMARY KEY,
    geometry GEOMETRY(LineString, 32634)
);

-- ACLED Events
CREATE TABLE car_cewp.acled_events (
    event_id_cnty TEXT PRIMARY KEY,
    event_date DATE,
    event_type TEXT,
    sub_event_type TEXT,
    fatalities INTEGER,
    latitude FLOAT,
    longitude FLOAT
);
```

## Troubleshooting

### "Invisible Hexagons"
The refactored `ensure_signed_h3()` function handles:
- Hex strings (e.g., `'852a104ffffffff'`)
- Unsigned integers from DB
- Converts to signed 64-bit int for Deck.gl

### "Database connection failed"
- Verify PostgreSQL is running
- Check `.env` credentials
- For WSL: ensure Windows host IP is routable

### "No prediction data"
- Dashboard generates mock data if `predictions_latest` doesn't exist
- Run the modeling pipeline to generate actual predictions

### "Roads/Rivers not showing"
- Check tables exist: `SELECT to_regclass('car_cewp.rivers')`
- Backend tries multiple table names for roads

## Development

### Adding New Layers

1. Add data fetching function in `script.js`:
```javascript
async function fetchCustomData() {
    const response = await fetch(`${API_URL}/custom`);
    return await response.json();
}
```

2. Add layer creation function:
```javascript
function createCustomLayer(data) {
    return new deck.GeoJsonLayer({
        id: 'custom-layer',
        data: data,
        // ... options
    });
}
```

3. Add to `renderLayers()`:
```javascript
if (layersState.custom && cachedData.custom) {
    layers.push(createCustomLayer(cachedData.custom));
}
```

4. Add toggle in `index.html` and listener in `setupToggleListeners()`

## License

Part of the CEWP Master's Thesis Project (2025).
