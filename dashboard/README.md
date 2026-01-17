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
- Date slider: ascending dates from API; left = oldest, right = newest. Client auto-reverses only if API misorders.
- Color ramps: feature-dependent; values clipped to 5th-95th percentiles for stable visuals.
- No fallback/mock data. Empty map = truly empty table/date.

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
- Slider direction: clear cache; ensure `/api/dates/*` returns ascending order.

## License
Part of the CEWP Master's Thesis Project (2025).
