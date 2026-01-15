#!/usr/bin/env bash
# Prepare WSL DB connection using the Windows gateway IP and drop the mines table.
set -euo pipefail

# Path to the project .env on Windows; do not use .env_windows
ENV_FILE="/mnt/c/Users/Brenan/Desktop/Thesis/Scratch/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing .env at $ENV_FILE" >&2
  exit 1
fi

# Load credentials from .env (strip CRLF if present)
set -a
source <(tr -d '\r' < "$ENV_FILE")
set +a

# Hardcode DB host/port for WSL: gateway IP + port 5433
export DB_HOST="$(ip route show | awk '/default/ {print $3}')"
export DB_PORT="5433"

echo "Using DB_HOST=$DB_HOST DB_PORT=$DB_PORT"

# Drop the mines table to let the pipeline recreate schema cleanly
PGPASSWORD="${DB_PASS:?}" psql -h "$DB_HOST" -p "$DB_PORT" -U "${DB_USER:?}" -d "${DB_NAME:?}" \
  -c "DROP TABLE IF EXISTS car_cewp.mines_h3 CASCADE;"

echo "Done."
