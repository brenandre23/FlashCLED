# dashboard/Dockerfile

# ==========================================
# STAGE 1: Builder (Compiles C extensions)
# ==========================================
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies (Required for GeoPandas/Shapely/GDAL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_dashboard.txt .
# Build wheels into /wheels folder to reuse in next stage
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements_dashboard.txt

# ==========================================
# STAGE 2: Runner (Lightweight Runtime)
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime libraries only (GDAL/PROJ)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (Production Best Practice)
RUN useradd --create-home --uid 1000 appuser

# Copy built wheels from builder stage
COPY --from=builder /wheels /wheels
COPY requirements_dashboard.txt .

# Install dependencies from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements_dashboard.txt && \
    rm -rf /wheels

# Copy application source code
COPY . .

# Exposure and Startup
EXPOSE 8000
USER appuser

# Run Gunicorn: 4 Workers (Parallelism), 2 Threads (Concurrency for MVT)
CMD ["gunicorn", "--workers", "4", "--threads", "2", "--worker-class", "gthread", "--bind", "0.0.0.0:8000", "app:app"]