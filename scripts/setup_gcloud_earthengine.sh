#!/usr/bin/env bash
# Install Google Cloud SDK (gcloud) into $HOME (no sudo needed) and ensure Earth Engine CLI is present.
# Idempotent: reuses existing installs.

set -euo pipefail

GCLOUD_DIR="${HOME}/google-cloud-sdk"
GCLOUD_TAR="google-cloud-cli-466.0.0-linux-x86_64.tar.gz"
GCLOUD_URL="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${GCLOUD_TAR}"

echo "=== Setup: gcloud + Earth Engine CLI (user-local) ==="

# Install gcloud locally if missing
if ! command -v gcloud >/dev/null 2>&1; then
  echo "[1/3] Installing Google Cloud SDK to ${GCLOUD_DIR}..."
  mkdir -p "${HOME}/downloads"
  cd "${HOME}/downloads"
  if [ -d "${GCLOUD_DIR}" ]; then
    echo "  Existing partial install found; removing ${GCLOUD_DIR}..."
    rm -rf "${GCLOUD_DIR}"
  fi
  if [ ! -f "${GCLOUD_TAR}" ]; then
    echo "  Downloading ${GCLOUD_TAR}..."
    curl -fLO "${GCLOUD_URL}"
  else
    echo "  Tarball already present."
  fi

  echo "  Extracting..."
  tar -xzf "${GCLOUD_TAR}" -C "${HOME}"

  echo "  Running installer..."
  # Ensure sane terminal dimensions so textwrap does not error
  if [ -z "${COLUMNS:-}" ] || [ "${COLUMNS:-0}" -lt 10 ]; then export COLUMNS=80; fi
  if [ -z "${LINES:-}" ] || [ "${LINES:-0}" -lt 10 ]; then export LINES=24; fi
  "${GCLOUD_DIR}/install.sh" --quiet --usage-reporting=false --path-update=true --rc-path="${HOME}/.bashrc"
else
  echo "[1/3] gcloud already installed: $(gcloud --version | head -n 1)"
fi
# Ensure current shell can see gcloud
if [ -x "${GCLOUD_DIR}/bin/gcloud" ]; then
  export PATH="${GCLOUD_DIR}/bin:${PATH}"
fi

# Ensure earthengine CLI is available (install into dedicated venv)
if ! command -v earthengine >/dev/null 2>&1; then
  echo "[2/3] Installing Earth Engine CLI in user venv..."
  PY_BIN="$(command -v python3 || true)"
  if [ -z "${PY_BIN}" ]; then PY_BIN="$(command -v python || true)"; fi
  if [ -z "${PY_BIN}" ]; then
    echo "  ERROR: No python or python3 found; install Python first."
    exit 1
  fi
  VENV_PATH="${HOME}/.local/venvs/ee-cli"
  if [ ! -d "${VENV_PATH}" ]; then
    if ! "${PY_BIN}" -m venv "${VENV_PATH}"; then
      echo "  venv creation failed; attempting bundled gcloud Python for install..."
      BUNDLED_PY="${GCLOUD_DIR}/platform/bundledpythonunix/bin/python3"
      if [ -x "${BUNDLED_PY}" ]; then
        "${BUNDLED_PY}" -m pip install --upgrade --user earthengine-api
        export PATH="${HOME}/.local/bin:${PATH}"
        echo "  Installed earthengine-api with bundled Python."
        else
          echo "  ERROR: bundled Python not found. Install python3-venv (apt install python3-venv) and rerun."
          exit 1
        fi
      fi
  fi
  if [ -d "${VENV_PATH}" ]; then
    if ! "${VENV_PATH}/bin/python" -m pip --version >/dev/null 2>&1; then
      echo "  Bootstrapping pip inside venv..."
      TMP_GETPIP="$(mktemp /tmp/get-pip.XXXXXX.py)"
      curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${TMP_GETPIP}"
      "${VENV_PATH}/bin/python" "${TMP_GETPIP}"
      rm -f "${TMP_GETPIP}"
    fi
    "${VENV_PATH}/bin/python" -m pip install --upgrade pip
    "${VENV_PATH}/bin/python" -m pip install --upgrade earthengine-api
    export PATH="${VENV_PATH}/bin:${PATH}"
  fi
else
  echo "[2/3] Earth Engine CLI already installed: $(earthengine --version 2>/dev/null || true)"
fi

echo "[3/3] Verification:"
if command -v gcloud >/dev/null 2>&1; then
  gcloud --version | head -n 3
fi
earthengine --version 2>/dev/null || true

cat <<'EOF'

Next steps (manual, one-time):
  1) Run: source ~/.bashrc   # ensure PATH includes ~/google-cloud-sdk/bin
  2) Run: gcloud auth login
  3) Run: earthengine authenticate

After authentication, re-run your pipeline.
EOF
