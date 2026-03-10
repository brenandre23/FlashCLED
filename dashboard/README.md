# FlashCLED Dashboard

Production-ready web UI for the Conflict Early Warning Pipeline (CEWP). Three completely isolated map views: predictions, temporal features, and static features. No cross-source mixing; each map hits its own endpoints and tables.

## Quick Start
- Prereqs: Python 3.10+, PostGIS with CEWP tables populated.
- Install: `cd dashboard && pip install -r requirements_dashboard.txt`
- Configure `.env` (copy from repo root if present):
  - `DB_HOST`, `DB_PORT` (default 5433), `DB_NAME`, `DB_USER`, `DB_PASS`
- Run: `python app.py` (default http://localhost:8000)

## Data Contracts
- Predictions map: `car_cewp.predictions`; endpoints `/api/predictions`, `/api/dates/predictions`, `/api/analytics/prediction/hex/<h3>`
- Temporal map: `car_cewp.temporal_features`; endpoints `/api/temporal_feature`, `/api/dates/temporal`, `/api/analytics/temporal/hex/<h3>`
- Static map: `car_cewp.features_static`; endpoints `/api/static_feature`, `/api/static_features/list`
- Strict allowlists enforced per endpoint; no blended queries.

## Behavior & UX Notes
- Temporal map date slider: ascending dates from API; left = oldest, right = newest. Client auto-reverses only if API misorders.
- Predictions map currently has no date slider in the UI and defaults to latest available date for the selected horizon/learner.
- Predictions map now supports operational presets:
  - Show top: `1% | 5% | 10% | 15% | 25% | 30% | 50%`
  - Ranked by: `Worst-Case (BCCP Upper)` (default), `Expected Fatalities`, `Conflict Probability`, `Uncertainty Width`
- Default display is `Top 5%` ranked by `Worst-Case (BCCP Upper)`.
- Color ramps: feature-dependent; values clipped to 5th-95th percentiles for stable visuals.
- No fallback/mock data. Empty map = truly empty table/date.

## Predictions Map Function Contract

### End-to-end flow
1. Frontend initializes state and maps.
2. `loadPredictionsDates()` calls `/api/dates/predictions`.
3. `loadPredictionsData()` calls `/api/predictions` for selected `horizon`, `learner`, `level`, and date.
4. `renderPredictionsLayer()` draws H3 polygons or optional event points.
5. On cell click, `handlePredictionClick()` fetches:
   - `/api/analytics/prediction/hex/<h3>` for risk/fatality history
   - `/api/predictions/shap` for explanations
   - `/api/analytics/temporal/hex/<h3>?feature=fatalities_14d_sum` for actual fatality overlay
6. `renderPredictionInspector()` and `renderPredictionTrend()` update right-side inspector and trend chart.

### Backend functions (`app.py`)
- `get_prediction_dates()` (`/api/dates/predictions`)
  - Input: optional `horizon`, `learner`
  - Output: ascending list of dates from `car_cewp.predictions`
- `get_predictions()` (`/api/predictions`)
  - Input: `date`, `horizon`, optional `learner`, `level`
  - Output fields used by map:
    - `hex`
    - `risk` = `conflict_prob`
    - `fatalities` = `predicted_fatalities`
    - `expected_fatalities` = `conflict_prob * predicted_fatalities`
    - `is_priority` (if available)
  - Admin levels aggregate by admin unit, then broadcast values to member H3 cells.
- `get_prediction_analytics()` (`/api/analytics/prediction/hex/<h3>`)
  - Input: `h3`, `horizon`, optional `learner`
  - Output: time series of `risk`, `fatalities`, `expected_fatalities`
- `get_predictions_shap()` (`/api/predictions/shap`)
  - Input: `hex`, `horizon`, `learner`, optional date/mode
  - Output: per-cell top contributing features
- `get_temporal_analytics()` (`/api/analytics/temporal/hex/<h3>`)
  - Used by predictions inspector to fetch `fatalities_14d_sum` as observed/actual overlay series.
- `get_events()` (`/api/events`)
  - Optional ACLED point layer for map context only.

### Frontend functions (`script.js`)
- `buildPredictionQuery(basePath, params)`
  - Builds URL query with `horizon`, `learner`, `level`, optional `date`.
- `loadPredictionsDates()`
  - Pulls date list and stores it in `state.predictions.dates`.
- `loadPredictionsData(date = null)`
  - Pulls prediction rows and stores in `state.predictions.data`.
- `renderPredictionsLayer()`
  - Applies top-percent filtering and ranking before rendering.
  - Creates `deck.H3HexagonLayer`:
    - `getHexagon`: `d.hex`
    - `getFillColor`: from risk via `getPredictionColor(d.risk)`
    - `getElevation`: `(d.fatalities || 0) * 50000` (predicted fatalities extrusion)
    - Highlights top-ranked cells (strongest styling on top 20).
  - Optional `deck.GeoJsonLayer` for ACLED event points.
- `updateTooltip('predictions', info)`
  - Tooltip fields:
    - Risk (%)
    - Fatalities (predicted fatalities)
    - Expected (`expected_fatalities`, fallback `risk * fatalities`)
- `handlePredictionClick({ object })`
  - Fetches prediction history, SHAP explanations, and actual fatality overlay.
- `fetchActualFatalHistory(hex, dateLabels)`
  - Fetches temporal `fatalities_14d_sum` and aligns it to prediction-history dates.
- `renderPredictionInspector(data, object, explainData)`
  - Renders selected cell metadata and current stats.
- `renderPredictionTrend(history)`
  - Trend lines:
    - Predicted Risk
    - Predicted Fatalities
    - Expected Fatalities
    - Actual Fatalities (from temporal endpoint overlay)

### Fatality semantics on the main predictions map
- `fatalities` on map cards/tooltips/extrusion is model output (`predicted_fatalities`), not ACLED observed counts.
- `expected_fatalities` is computed as `conflict_prob * predicted_fatalities` in backend query.
- "Actual Fatalities" appear only in the inspector trend chart via temporal feature `fatalities_14d_sum`.

## Architecture
- Frontend: `index.html`, `script.js`, `style.css` (three Deck.gl instances, isolated state)
- Backend: `app.py` (Flask), exposes only single-source endpoints above
- Dependencies: see `requirements_dashboard.txt`
- Docker: `dashboard/Dockerfile` available for containerized runs

## File Layout
- `app.py`                Flask backend
- `index.html`            UI entry point (three tabs)
- `script.js`             Map logic and API calls
- `style.css`             Styling
- `requirements_dashboard.txt`  Dependencies
- `Dockerfile`            Container build
- `README.md`             This file

## Troubleshooting
- Feature rejected: check `/api/temporal_features/list` or `/api/static_features/list` and use an allowlisted name.
- No data: confirm DB has rows for requested date/table; check `/api/health`.
- Temporal slider direction: clear cache; ensure `/api/dates/temporal` returns ascending order.
- Predictions horizons:
  - Supported in UI/API: `14d`, `1m`, `3m`.
  - Forecast-date helper mapping uses `14d`, `28d`, and `84d` offsets respectively.

## License
Part of the CEWP Master's Thesis Project (2025).
