# FlashCLED Dashboard — Three-Map Architecture

Interactive web dashboard for the Conflict Early Warning Pipeline (CEWP). Features **strict data lineage** with three separate maps that never mix data sources.

## Architecture: Three Isolated Maps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Map 1         │  │   Map 2         │  │   Map 3         │             │
│  │   PREDICTIONS   │  │   TEMPORAL      │  │   STATIC        │             │
│  │   (Primary)     │  │   FEATURES      │  │   FEATURES      │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│  /api/predictions      /api/temporal_feature  /api/static_feature          │
│  /api/dates/predictions /api/dates/temporal                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ SQL (No Mixing!)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PostGIS Database                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   predictions   │  │ temporal_       │  │ features_       │             │
│  │                 │  │   features      │  │   static        │             │
│  │  (Model Output) │  │ (Time-Varying)  │  │ (Time-Invariant)│             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Lineage Rules (Non-Negotiable)

| Map | Data Source | Endpoints Used | Never Queries |
|-----|-------------|----------------|---------------|
| **Map 1: Predictions** | `car_cewp.predictions` | `/api/predictions`, `/api/dates/predictions` | temporal_features, features_static data |
| **Map 2: Temporal** | `car_cewp.temporal_features` | `/api/temporal_feature`, `/api/dates/temporal` | predictions |
| **Map 3: Static** | `car_cewp.features_static` | `/api/static_feature` | predictions, temporal_features |

## Slider Direction Fix

**Problem:** Moving slider right showed earlier dates (descending order).

**Solution:**
- Dates are sorted **ascending** from the server (`ORDER BY date ASC`)
- Slider `value=0` → oldest date (left)
- Slider `value=max` → newest date (right)
- Moving slider **right** always shows **later dates**

Client-side guard:
```javascript
if (dates.length > 1 && dates[0] > dates[1]) {
    console.warn('Dates were descending, reversing');
    dates = dates.reverse();
}
```

## API Endpoints

### Date Endpoints (Separate by Source)

| Endpoint | Table | Response |
|----------|-------|----------|
| `GET /api/dates/predictions` | `predictions` | `{ "dates": ["2020-01-01", "2020-01-15", ...] }` |
| `GET /api/dates/temporal` | `temporal_features` | `{ "dates": ["2020-01-01", "2020-01-15", ...] }` |

### Map 1: Predictions (Primary)

| Endpoint | Description |
|----------|-------------|
| `GET /api/predictions?date=YYYY-MM-DD` | H3 predictions for date |
| `GET /api/predictions/cube` | Full history for animation |
| `GET /api/analytics/prediction/hex/<h3>` | Time-series for single hex |

**Response format:**
```json
[
  { "hex": "-8608933471623168", "risk": 0.42, "fatalities": 1.2 }
]
```

### Map 2: Temporal Features

| Endpoint | Description |
|----------|-------------|
| `GET /api/temporal_feature?feature=<name>&date=YYYY-MM-DD` | Single feature layer |
| `GET /api/temporal_features/list` | Available features by category |
| `GET /api/analytics/temporal/hex/<h3>?feature=<name>` | Time-series for hex |

**Allowed features:**
- Environmental: `chirps_precip_anomaly`, `era5_temp_anomaly`, `ndvi_anomaly`, etc.
- Conflict: `fatalities_14d_sum`, `protest_count_lag1`, `riot_count_lag1`, etc.
- GDELT: `gdelt_event_count`, `gdelt_avg_tone`, `gdelt_goldstein_mean`
- Market: `price_maize`, `price_rice`, `price_oil`, etc.
- Macroeconomics: `gold_price_usd_lag1`, `oil_price_usd_lag1`, etc.

### Map 3: Static Features

| Endpoint | Description |
|----------|-------------|
| `GET /api/static_feature?feature=<name>` | Static feature layer |
| `GET /api/static_features/list` | Available features |

**Allowed features:**
- Distance: `dist_to_capital`, `dist_to_border`, `dist_to_road`, `dist_to_city`
- Geographic: `elevation_mean`, `slope_mean`
- Population: `population`, `pop_density`

## Color Schemes

| Map | Ramp | Visual |
|-----|------|--------|
| Predictions | Light rose → Dark red | Risk probability |
| Temporal (varies) | Feature-dependent | See code for mappings |
| Static | Light lavender → Dark purple | Consistent single-hue |

**Normalization:** All features use 5th-95th percentile clipping before color mapping.

## Quick Start

### 1. Install Dependencies

```bash
cd dashboard
pip install -r requirements_dashboard.txt
```

### 2. Configure Database

```env
# .env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=thesis_db
DB_USER=postgres
DB_PASS=your_password
```

### 3. Run Server

```bash
python app.py
# Server: http://localhost:8000
```

## Database Schema

### Required Tables

```sql
-- Model predictions (Map 1 source)
CREATE TABLE car_cewp.predictions (
    h3_index BIGINT,
    date DATE,
    prob_conflict_3m FLOAT,
    expected_fatalities_3m FLOAT,
    PRIMARY KEY (h3_index, date)
);

-- Time-varying features (Map 2 source)
CREATE TABLE car_cewp.temporal_features (
    h3_index BIGINT,
    date DATE,
    chirps_precip_anomaly FLOAT,
    era5_temp_anomaly FLOAT,
    ndvi_anomaly FLOAT,
    fatalities_14d_sum FLOAT,
    -- ... other temporal columns
    PRIMARY KEY (h3_index, date)
);

-- Static features (Map 3 source)
CREATE TABLE car_cewp.features_static (
    h3_index BIGINT PRIMARY KEY,
    dist_to_capital FLOAT,
    dist_to_border FLOAT,
    elevation_mean FLOAT,
    population FLOAT,
    -- ... other static columns
);
```

## File Structure

```
dashboard/
├── app.py                      # Flask backend (strict endpoints)
├── index.html                  # Three-tab UI
├── script.js                   # Three isolated Deck.gl instances
├── style.css                   # Modern styling
├── requirements_dashboard.txt  # Dependencies
└── README.md                   # This file
```

## What Changed (v2.0)

### Removed
- ❌ Mixed `/api/h3_data` endpoint (combined sources)
- ❌ Mixed `/api/dates` endpoint
- ❌ Mixed `/api/analytics/hex/<h3>` (combined predictions + drivers)
- ❌ Mock/fallback data generation
- ❌ Descending date sorting (caused slider bug)

### Added
- ✅ `/api/dates/predictions` — predictions dates only
- ✅ `/api/dates/temporal` — temporal dates only
- ✅ `/api/predictions` — predictions data only
- ✅ `/api/temporal_feature` — single temporal feature only
- ✅ `/api/static_feature` — single static feature only
- ✅ `/api/analytics/prediction/hex/<h3>` — prediction history only
- ✅ `/api/analytics/temporal/hex/<h3>` — temporal history only
- ✅ Feature allowlists with strict validation
- ✅ Ascending date sorting (slider fix)
- ✅ Three isolated map instances with separate state
- ✅ Feature-dependent color ramps

## Troubleshooting

### "Feature not in allowlist"
- Only allowlisted features can be queried (security)
- Check `/api/temporal_features/list` or `/api/static_features/list`

### "Slider still shows wrong direction"
- Clear browser cache
- Check console for "Dates were descending" warning
- Verify `/api/dates/*` returns ascending order

### "Map shows no data"
- Check `/api/health` for table status
- Ensure tables have data for the selected date
- No fallback data is generated — empty = truly empty

## License

Part of the CEWP Master's Thesis Project (2025).
