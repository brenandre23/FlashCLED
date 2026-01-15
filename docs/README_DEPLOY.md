# FlashCLED Dashboard: Container Deployment

This covers production-style containerization of the Flask/MapLibre dashboard with Gunicorn and PostGIS.

## Prerequisites
- Docker Engine + Docker Compose v2 (`docker compose` CLI).
- Copy `.env.example` to `.env` and set `DB_*` credentials (defaults are provided in `docker-compose.yml`).

## Build & Run
- Build images: `docker compose build`
- Start stack (web + PostGIS): `docker compose up -d`
- App listens on `http://localhost:8000`. PostGIS listens on `localhost:5432`.

## Logs & Health
- Follow web logs: `docker compose logs -f web`
- Follow db logs: `docker compose logs -f db`
- Restart a service: `docker compose restart web` (or `db`).

## Importing SQL Dumps into PostGIS
Use your `.sql` dump (schema or data). Example workflow:
1) Copy dump into the running db container:
   - `docker cp /path/to/dump.sql $(docker compose ps -q db):/tmp/dump.sql`
2) Execute import with psql (uses env defaults or your `.env`):
   - `docker compose exec db psql -U ${DB_USER:-cewp_user} -d ${DB_NAME:-car_cewp} -f /tmp/dump.sql`
3) Optional: open an interactive psql shell:
   - `docker compose exec -it db psql -U ${DB_USER:-cewp_user} -d ${DB_NAME:-car_cewp}`

## Shutdown & Cleanup
- Stop containers: `docker compose down`
- Remove containers + PostGIS volume (destructive): `docker compose down -v`
