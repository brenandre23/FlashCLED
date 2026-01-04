# Database Setup and Schema Documentation

## Overview
The CEWP pipeline uses PostgreSQL with PostGIS and H3 extensions to store and process geospatial conflict data. This document covers database architecture, setup, and maintenance.

## Architecture

### Schema: `car_cewp`
All pipeline tables are created in a dedicated schema to avoid conflicts with other applications.

### Table Categories

#### 1. Spatial Foundation
- **features_static** - Master H3 grid with static features
  - Primary key: `h3_index` (BIGINT)
  - ~9,000 hexagons at resolution 5 (10km)
  - Columns: terrain, distances to infrastructure, EPR features

#### 2. Temporal Data
- **temporal_features** - Time-series features aggregated to 14-day windows
  - Primary key: `(h3_index, date)`
  - Contains: environmental, conflict, economic, social indicators

#### 3. Raw Ingestion Tables
- **acled_events** - Conflict events from ACLED
- **environmental_features** - Satellite data (CHIRPS, ERA5, MODIS, VIIRS)
- **food_security** - WFP market prices
- **iom_displacement** - IOM displacement tracking
- **economic_drivers** - Macro indicators (gold, oil, FX rates)

#### 4. Processing Tables
- **population_h3** - WorldPop data aggregated to H3
- **grip4_roads_h3** - Road network density
- **ipc_h3** - Food security classifications
- **geoepr_polygons** - Ethnic power relations spatial data

## Extension Requirements

### PostGIS (3.0+)
Provides spatial data types and operations.

**Key functions used:**
- `ST_Intersects()` - Spatial joins
- `ST_Distance()` - Distance calculations
- `ST_Transform()` - CRS conversions
- `ST_Buffer()` - Proximity analysis

### H3 (4.0+)
Hexagonal hierarchical spatial indexing.

**Key functions used:**
- `h3_cell_to_boundary()` - Convert H3 to polygon
- `h3_is_valid_cell()` - Validate H3 indices
- `h3_get_resolution()` - Get resolution of cell

**Critical:** All `h3_index` columns MUST be `BIGINT` (signed 64-bit integer), not `VARCHAR` or `TEXT`.

## Setup Script

The `init_db.py` script handles:
1. Extension installation
2. Schema creation
3. Type validation and migration (VARCHAR → BIGINT for h3_index)

```bash
python init_db.py
```

## Connection Parameters

### Local Development
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=car_cewp
DB_USER=cewp_user
DB_PASS=your_password
```

### WSL2 + Windows PostgreSQL
```bash
# Use Windows host IP, not localhost
DB_HOST=172.20.240.1  # Example, get via: ip route show | awk '/default/ {print $3}'
DB_PORT=5432
```

### Production (Docker)
```bash
DB_HOST=postgres  # Service name in docker-compose.yml
DB_PORT=5432
```

## Performance Optimization

### Indexes
The pipeline automatically creates spatial indexes:

```sql
-- Example from features_static
CREATE INDEX idx_features_static_geom ON car_cewp.features_static USING GIST (geometry);
CREATE INDEX idx_temporal_features_date ON car_cewp.temporal_features (date);
CREATE INDEX idx_temporal_features_h3 ON car_cewp.temporal_features (h3_index);
```

### Recommended Settings (postgresql.conf)

```conf
# Memory
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 256MB
maintenance_work_mem = 1GB

# Parallelism
max_parallel_workers_per_gather = 4
max_parallel_workers = 8

# PostGIS-specific
random_page_cost = 1.1  # For SSD
effective_io_concurrency = 200
```

## Backup and Restore

### Backup
```bash
# Full database
pg_dump -Fc -d car_cewp > car_cewp_backup_$(date +%Y%m%d).dump

# Schema only
pg_dump --schema-only -d car_cewp > schema.sql
```

### Restore
```bash
pg_restore -d car_cewp car_cewp_backup_20260104.dump
```

## Common SQL Queries

### Check table sizes
```sql
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'car_cewp'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Verify H3 types
```sql
SELECT 
    table_name, 
    column_name, 
    data_type
FROM information_schema.columns
WHERE table_schema = 'car_cewp'
  AND column_name = 'h3_index';
```

### Count records by table
```sql
SELECT 
    schemaname,
    tablename,
    n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE schemaname = 'car_cewp'
ORDER BY n_live_tup DESC;
```

## Troubleshooting

### Extension not found
```bash
# Check available extensions
psql -c "SELECT * FROM pg_available_extensions WHERE name LIKE '%h3%';"

# Install from source if needed
git clone https://github.com/zachasme/h3-pg.git
cd h3-pg && make && sudo make install
```

### Type mismatch errors
```bash
# Run type migration
python init_db.py

# Manually fix a specific table
psql -d car_cewp -c "ALTER TABLE car_cewp.your_table ALTER COLUMN h3_index TYPE BIGINT USING h3_index::bigint;"
```

### Connection refused
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check port
sudo netstat -plnt | grep 5432

# Allow remote connections (postgresql.conf)
listen_addresses = '*'

# Update pg_hba.conf
host    car_cewp    cewp_user    0.0.0.0/0    md5
```
