# CEWP Data Source Audit v5.0

**Spatio-Temporal Resolutions & Imputation Methods**

| Field | Value |
|-------|-------|
| Pipeline Version | January 2026 |
| Target Region | Central African Republic (CAR) |
| Analysis Grid | H3 Resolution 5 (~10km hexagonal cells) |
| Grid Coverage | ~3,407 cells |
| Temporal Spine | 14-day intervals aligned to 2000-01-01 |
| Total Features | 138 unique columns (pre-pruning) |
| Generated | 2026-01-25 |

---

## Executive Summary

This audit documents all data sources integrated into the Conflict Early Warning Pipeline (CEWP), detailing their native resolutions, pipeline transformations, and imputation strategies for missing values. The pipeline harmonizes 21+ distinct data sources across seven thematic categories into a unified analytical framework suitable for conflict prediction modeling.

| Category | Sources | Native Resolution | Target Resolution |
|----------|---------|-------------------|-------------------|
| Environmental | 8 | 100m – 11km, hourly/daily | H3-5, 14-day |
| Conflict & Events | 3 | Point, daily | H3-5, 14-day |
| Socio-Political | 3 | Admin-1/2, annual/quarterly | H3-5, 14-day |
| Economic | 6 | National/Market, daily/monthly | National/H3-5, 14-day |
| Infrastructure | 5 | Point/polygon, static | H3-5, static |
| Demographics | 1 | 100m, annual | H3-5, annual |
| NLP/Semantic | 3 | Event-level text / Monthly reports | H3-5, 14-day / Monthly |

---

## Temporal Lag Handling

The pipeline distinguishes between two independent lag mechanisms:

### Publication Lags (Data Availability)

Publication lags account for the delay between data collection and public availability. These are applied at ingestion or immediately after merge so timestamps reflect when data would actually be available in a real-time deployment.

| Source | Publication Lag | Applied At |
|--------|-----------------|------------|
| GEE Environmental (ERA5, CHIRPS, MODIS, VIIRS, JRC, Dynamic World) | 14 days (1 step) | Storage (GEE window shift) |
| ACLED Hybrid NLP Drivers | 14 days (1 step) | Storage (event_date shift) |
| ACLED Conflict Counts | 14 days (1 step) | Feature engineering (post-merge) |
| Food Prices (WFP/FEWS NET) | 56 days (4 steps) | Storage (date shift) |
| CrisisWatch Reports | 14 days (1 step) | Storage (report_date shift) |

---

## Incremental Ingestion Architecture (Jan 2026 Update)

- **Central helper:** `get_incremental_window` (utils.py) checks `MAX(date)` in Postgres vs. the requested end date (from CLI or `data.global_date_window`). If current data covers the end date, the fetch is skipped; otherwise only the missing window is requested.
- **Force refresh:** CLI flag `--no-incremental` (main.py) propagates to dynamic ingests to override freshness and refetch the full window.
- **Safety overlaps:** Sources with latency/reprocessing re-fetch a small buffer:
  - GEE environmental: 14-day overlap on start window.
  - Dynamic World: 14-day overlap on start window.
  - IOM DTM: 14-day overlap on start window.
- **Per-source adoption:**
  - ACLED events: incremental window + API/CSV fallback; PK normalization for `event_id_cnty`.
  - Economy (Yahoo Finance): incremental window; no drop/recreate; upsert on `date`.
  - Food Security (FEWS NET): incremental window; publication lag handled downstream.
  - GDELT dual (events + GKG): incremental window.
  - IODA outages: incremental window.
  - IOM displacement: incremental window + overlap; filtered to start/end window.
  - Dynamic World: incremental window + overlap; bounded by helper end date.
  - GEE environmental: incremental window + 14-day overlap; bounded by helper end date.
- **Upserts:** All ingests use `upload_to_postgis` with primary keys set per table (e.g., `date`, `h3_index` combos or natural IDs) to avoid duplicates while allowing reruns.

### Analytical Lags (Leakage Control)

Analytical lags ensure features only use prior-period values, preventing temporal leakage into the model. These are applied downstream in feature engineering via lag_1_step and decay transforms.

| Mechanism | Purpose | Implementation |
|-----------|---------|----------------|
| LAG() / shift(1) | Features use prior time step values | Applied to all model features |
| LEAD() | Targets reference future time steps | Applied to prediction targets |
| Rolling windows | Strictly historical aggregation | No overlap with target period |

**Key distinction:** Publication lag moves the timestamp to when data is available; analytical lag shifts what the model sees to a prior window. A feature can have both—e.g., GEE data has a 14-day publication lag at ingestion AND an analytical lag when used as a model feature.

---

## 1. Environmental Data

Environmental variables are aggregated from Google Earth Engine collections to H3 resolution 5 with 14-day temporal windows. Server-side processing via a Map-Reduce architecture reduces data transfer and enables efficient historical analysis.

**Publication Lag:** All GEE-sourced data uses window [T, T+13] but is stored/labeled at date T+14 to prevent temporal leakage.

### 1.1 CHIRPS Precipitation

| Field | Value |
|-------|-------|
| Source | UCSB-CHG/CHIRPS/DAILY (via GEE) |
| Native Spatial | 0.05° (~5.5km) |
| Native Temporal | Daily |
| Aggregation | H3-5 zonal mean, 14-day sum |
| Imputation | Forward-fill (limit=4 steps / 56 days) |
| Coverage | 1981–present |
| Publication Lag | 14 days (applied at storage) |

**Output Columns:**
- `precip_mean_depth_mm` → `chirps_precip_anomaly`

### 1.2 ERA5-Land Climate

| Field | Value |
|-------|-------|
| Source | ECMWF/ERA5_LAND/HOURLY (via GEE) |
| Native Spatial | 0.1° (~11km) |
| Native Temporal | Hourly |
| Aggregation | H3-5 zonal mean, 14-day mean |
| Imputation | Forward-fill (limit=4 steps) |
| Coverage | 1950–present (~5-day lag) |
| Publication Lag | 14 days (applied at storage) |

**Variables:**
- Temperature (2m)
- Dewpoint (2m)
- Soil Moisture (Layer 1)

**Output Columns:**
- `era5_temp_anomaly`
- `era5_soil_moisture_anomaly`

### 1.3 MODIS Vegetation (NDVI)

| Field | Value |
|-------|-------|
| Source | MODIS/061/MCD43A4 (via GEE) |
| Native Spatial | 500m |
| Native Temporal | Daily (16-day composite) |
| Aggregation | H3-5 zonal max, 14-day max |
| Imputation | Forward-fill (limit=4 steps) |
| Notes | NDVI = (NIR - Red) / (NIR + Red); range [-1, 1] |
| Publication Lag | 14 days (applied at storage) |

**Output Columns:**
- `ndvi_max` → `ndvi_anomaly`

### 1.4 VIIRS Nighttime Lights (Integrated Strategy)

| Field | Value |
|-------|-------|
| Source | NASA/VIIRS/002/VNP46A2 (via GEE) |
| Native Spatial | 500m |
| Native Temporal | Daily |
| Aggregation | H3-5 zonal mean/max/median, 14-day |
| Imputation | **Pass-through** (NaNs are preserved; not zero-filled) |
| Coverage | 2012–present |
| Structural Break | Pre-2012 data = NULL (sensor launch date) |
| Publication Lag | 14 days (applied at storage) |

**Methodology:**
The pipeline employs a "Reliability-Weighted" and "Kinetic-Aware" strategy to distinguish infrastructure from conflict events.

| Signal Type | Variable | GEE Band | Aggregation | Description |
|-------------|----------|----------|-------------|-------------|
| **Stability** | `ntl_mean` | `Gap_Filled_DNB_BRDF_Corrected_NTL` | Mean | Proxy for static infrastructure and electrification. |
| **Kinetic** | `ntl_peak` | `DNB_BRDF_Corrected_NTL` | Max | Raw peak radiance capturing fires, explosions, or temporary camps. |
| **Uncertainty** | `ntl_stale_days` | `Latest_High_Quality_Retrieval` | Median | Days since last high-quality observation. Informs model of data age/reliability. |
| **Derived** | `ntl_kinetic_delta` | (Calculated) | - | `peak - mean`. Isolates transient high-radiance events (fires) from background light. |

**Availability Guard:** `viirs_data_available` = 1 if `date >= 2012-01-28` AND `ntl_mean` is not null.

### 1.5 JRC Global Surface Water

| Field | Value |
|-------|-------|
| Source | JRC/GSW1_4/MonthlyHistory (pre-2022) or Landsat 8/9 (2022+) |
| Native Spatial | 30m |
| Native Temporal | Monthly (JRC) / Daily composite (Landsat) |
| Aggregation | H3-5 zonal mean + max, 14-day |
| Imputation | Forward-fill (limit=4 steps) |
| Notes | Binary water detection (MNDWI > 0.1 for Landsat) |
| Publication Lag | 14 days (applied at storage) |

**Output Columns:**
- `water_coverage_lag1`
- `water_presence_lag1`

### 1.6 Dynamic World Landcover

| Field | Value |
|-------|-------|
| Source | GOOGLE/DYNAMICWORLD/V1 (via GEE) |
| Native Spatial | 10m |
| Native Temporal | Near real-time (Sentinel-2 derived) |
| Aggregation | H3-5 zonal mean, 14-day mean probability |
| Availability Flag | `landcover_avail`: 1 if date ≥ 2015-06-27, else 0 |
| Imputation | Forward-fill (limit=4 steps) |
| Coverage | 2015-06-27–present |
| Publication Lag | 14 days (applied at storage) |

**Output Columns (all as mean fraction 0-1):**
- `landcover_grass`
- `landcover_crops`
- `landcover_trees`
- `landcover_bare`
- `landcover_built`

---

## 2. Conflict & Event Data

Event data is aggregated from point locations to H3 cells with 14-day temporal windows. Missing values are zero-filled under the assumption that no recorded event equals no event occurrence.

### 2.1 ACLED (Armed Conflict Location & Event Data)

| Field | Value |
|-------|-------|
| Source | Local CSV (data/raw/acled.csv) |
| Native Spatial | Point (lat/lon, geo_precision 1-3) |
| Native Temporal | Daily (event_date) |
| Aggregation | H3-5 point-in-polygon, 14-day sum |
| Imputation | Zero-fill for missing (no events = 0) |
| Coverage | 1997–present (CAR: 2000+) |
| Volume | ~7,500+ events in CAR |
| Publication Lag | 14 days (applied to features, not targets) |

**Output Columns:**
- `fatalities_14d_sum`
- `fatalities_1m_lag`
- `conflict_density_10km`
- `protest_count_lag1`
- `riot_count_lag1`
- `regional_risk_score_lag1`

**Note:** ACLED features and targets are handled separately. Features receive 14-day publication lag; targets remain unlagged to preserve prediction validity.

### 2.2 ACLED Hybrid NLP Features

Semi-supervised semantic projection approach using ensemble scoring and residual clustering.

| Field | Value |
|-------|-------|
| Source | ACLED notes field processed by `process_acled_hybrid.py` |
| Native Spatial | Event-level (inherits H3 from parent event) |
| Native Temporal | Event date |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional) |
| Method | Hybrid: 8 semantic themes (residual clustering) + 5 regex/anchor drivers |
| Scoring | Ensemble: regex pattern matching + cosine similarity to anchor phrases |
| Output Type | Continuous risk scores (0.0-1.0) |
| Imputation | Zero-fill for missing |
| Coverage | Full ACLED history |
| Total Features | 13 (8 themes + 5 drivers) |
| Publication Lag | 14 days (shifted at storage) |

**Theme Columns (8):**
- `theme_context_0`
- `theme_context_1`
- `theme_context_2`
- `theme_context_3`
- `theme_context_4`
- `theme_context_5`
- `theme_context_6`
- `theme_context_7`

#### 2.2.1 Explicit Drivers (5 total)

| Driver | Description |
|--------|-------------|
| `driver_resource_cattle` | Cattle/livestock-related conflict (raiding, grazing disputes) |
| `driver_resource_mining` | Mining-related violence (artisanal mining, resource extraction) |
| `driver_econ_taxation` | Economic extortion (illegal taxation, checkpoint fees) |
| `driver_political_coup` | Political instability indicators (coup attempts, power struggles) |
| `driver_civilian_abuse` | Human rights violations (civilian targeting, abuse) |

### 2.3 GDELT (Global Database of Events, Language, and Tone)

| Field | Value |
|-------|-------|
| Source | GDELT v2 Master File List (CSV/ZIP via HTTP) |
| Native Spatial | Point (Actor1Geo_Lat/Lon) |
| Native Temporal | Daily (SQLDATE) |
| Aggregation | H3-5 point-in-polygon, 14-day sum/mean |
| Imputation | Zero-fill for missing |
| Coverage | 2015–present (v2) |
| Notes | Uses FIPS code "CT" for CAR |

**Output Columns:**
- `events_3m_lag`
- `gdelt_decay_30d`
- `gdelt_avg_tone_decay_30d`

---

## 3. NLP & Narrative Signals (v2.0)

This section documents the "Predatory Ecosystem" NLP framework, which provides targeted detection of conflict mechanisms rooted in the political economy of the Central African Republic.

### 3.1 CrisisWatch Regime Pillars

| Field | Value |
|-------|-------|
| Source | International Crisis Group CrisisWatch monthly reports |
| Native Spatial | National (CAR-level) |
| Native Temporal | Monthly |
| Embedding Model | `snowood1/ConfliBERT-scr-uncased` (specialized for conflict text) |
| Method | Semantic scoring against theoretical regime pillars |
| Output Type | Continuous scores (0.0-1.0) |
| Imputation | Forward-fill (limit=2 months) |
| Coverage | 2003–present |
| Publication Lag | 14 days (applied at storage) |

#### Regime Pillar Features (4 total)

| Feature Name | Description |
|--------------|-------------|
| `regime_parallel_governance` | Semantic score for state substitution activities (taxation, illicit licensing). |
| `regime_transnational_predation` | Semantic score for foreign resource extraction and sovereign nexus (Wagner, Midas). |
| `regime_guerrilla_fragmentation` | Semantic score for rebel splintering, fluid alliances, and hit-and-run tactics. |
| `regime_ethno_pastoral_rupture` | Semantic score for the breakdown of customary mediation and militarized transhumance. |

### 3.2 Narrative Dynamics

| Feature Name | Source | Method | Description | Frequency |
|--------------|--------|--------|-------------|-----------|
| `narrative_velocity_lag1` | CrisisWatch | Cosine Distance | The rate of semantic change in reporting between month *t* and *t-1*. Measures "narrative drift." | Monthly |

### 3.3 ACLED Hybrid Mechanisms

| Field | Value |
|-------|-------|
| Source | ACLED notes field processed by `process_acled_hybrid.py` |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Method | Semi-supervised: MiniLM embeddings + regex pattern matching |
| Output Type | Binary flags or continuous scores |
| Aggregation | H3-5, weekly aggregation to 14-day windows |

#### Mechanism Features (4 total)

| Feature Name | Description |
|--------------|-------------|
| `mech_gold_pivot` | Detection of violence shifting from diamond zones to gold mining/dredging sites. |
| `mech_predatory_tax` | Detection of economic violence: illegal checkpoints, customs extortion, and road levies. |
| `mech_factional_infighting` | Detection of intra-rebel clashes (e.g., FPRC vs UPC) distinct from anti-state violence. |
| `mech_collective_punishment` | Detection of punitive expeditions and scorched-earth tactics against civilians. |

### 3.4 Fusion Signals

| Feature Name | Source | Method | Description | Frequency |
|--------------|--------|--------|-------------|-----------|
| `gdelt_shock_signal` | GDELT + CW | Composite Index | A calculated "Shock" metric: (AvgTone × -1) - (RegimeRisk). High values indicate media panic exceeding structural risk. | Daily |
| `gdelt_border_buffer_flag` | GDELT | Spatial Buffer | Binary flag (1) indicating if an event originated in the 50km "spillover zone" of Chad or Sudan. | Daily |

#### Interaction Features (New)

| Feature Name | Source | Description |
|--------------|--------|-------------|
| `cw_onset_amplifier` | CW x ACLED | Fragmentation × Wagner Risk. |
| `cw_mass_casualty_risk` | CW Interaction | Pastoral Rupture × Fragmentation. |
| `cw_extraction_violence` | CW x ACLED | Parallel Gov × Wagner Risk. |
| `cw_pastoral_predation` | CW Interaction | Parallel Gov × Pastoral Rupture. |
| `fusion_gold_signal` | CW x ACLED | Wagner Risk × Gold Pivot (Lag 1). |
| `fusion_fragmentation_confirmed` | CW x ACLED | Fragmentation × Factional Infighting (Lag 1). |
| `fusion_escalation_momentum` | CW x ACLED | Max(Delta, 0) × Mechanism Intensity. |

---

## 4. Socio-Political Data

### 4.1 EPR (Ethnic Power Relations)

| Field | Value |
|-------|-------|
| Source | ETH Zurich EPR-2021 (CSV + GeoJSON) |
| Native Spatial | Ethnic group polygon (GeoEPR) |
| Native Temporal | Annual (from-to year ranges) |
| Aggregation | Polygon → H3-5 via `polygon_to_cells()`, annual |
| Imputation | Zero-fill for cells outside any ethnic polygon |
| Coverage | 1946–2021 (expanded year-by-year) |
| Post-2021 | Uses 2021 data (acceptable for slow-changing features) |

**Output Columns:**
- `ethnic_group_count`
- `epr_excluded_groups_count`
- `epr_discriminated_groups_count`
- `epr_status_mean`

### 4.2 IOM DTM (Displacement Tracking Matrix)

| Field | Value |
|-------|-------|
| Source | IOM DTM API v3 |
| Native Spatial | Admin-2 (sub-prefecture) or Admin-1 |
| Native Temporal | Survey rounds (irregular, ~quarterly) |
| Aggregation | Admin-2 → H3-5 via area-weighted density, 14-day |
| Imputation | Zero-fill between survey rounds |
| Coverage | 2015-01-31–present |
| Notes | Fallback to Admin-1 if Admin-2 unavailable |

**Output Columns:**
- `iom_displacement_count_lag1`
- `iom_data_available` (structural break flag)

### 4.3 IODA Internet Outage Detection

| Field | Value |
|-------|-------|
| Source | Georgia Tech IODA API (outage events endpoint only) |
| Native Spatial | National (country-level; regional data not available for CAR) |
| Native Temporal | Event-based (near real-time outage detection) |
| Aggregation | National → broadcast to all H3 cells, 14-day weighted sum |
| Imputation | Zero-fill (no outage = 0) |
| Coverage | 2022-02-01–present |
| Notes | Outage events only; no connectivity index feed. Higher score = more/worse outages. |

**Output Columns:**
- `ioda_outage_score`
- `ioda_data_available`

---

## 5. Economic Data

### 5.1 Macro-Economic Indicators

| Field | Value |
|-------|-------|
| Source | Yahoo Finance (yfinance package) |
| Tickers | GC=F (Gold), CL=F (Oil), ^GSPC (S&P 500), EURUSD=X |
| Native Spatial | Global (non-spatial, broadcast to all cells) |
| Native Temporal | Daily (trading days) |
| Aggregation | National-level, 14-day mean |
| Imputation | Forward-fill for non-trading days |
| Coverage | 2003-12-01–present (code clamps start date) |
| Structural Break | `econ_data_available` = 0 before 2003-12-01, else 1 |

**Output Columns:**
- `gold_price_usd_lag1`
- `oil_price_usd_lag1`
- `sp500_index_lag1`
- `eur_usd_rate_lag1`

### 5.2 WFP/FEWS NET Food Prices

| Field | Value |
|-------|-------|
| Source | FEWS NET Data Warehouse API |
| Native Spatial | Market locations (10+ markets in CAR) |
| Native Temporal | Monthly |
| Aggregation | Market → nearest H3-5 cell, 14-day |
| Imputation | Forward-fill |
| Coverage | 2015–present |
| Publication Lag | 56 days (4 steps) shifted at storage |
| Notes | Food price index averages only available, non-zero commodities |

**Output Columns:**
- `price_maize`
- `price_rice`
- `price_oil`
- `price_sorghum`
- `food_price_index`

**Shock Features:**
- `price_maize_shock`
- `price_rice_shock`
- `price_oil_shock`
- `price_sorghum_shock`

### 5.3 Supplemental Data Tables

| Table | Primary Key | Columns | Notes |
|-------|-------------|---------|-------|
| `market_locations` | `market_id` | `market_name`, `latitude`, `longitude` | Static market coordinates; contains no admin names. |
| `environmental_features` (Sync) | `(h3_index, date)` | See Section 1 | `fetch_gee_server_side.py` automatically syncs existing CSVs from `data/raw/` to the DB before starting new queries. |

---

## 6. Infrastructure & Geography (Static)

| Data Source | Native Resolution | Output Features |
|-------------|-------------------|-----------------|
| Copernicus DEM | 90m raster | `elevation_mean`, `slope_mean`, `terrain_ruggedness_index` |
| GRIP4 Roads | Polyline network | `dist_to_road` (km) |
| HydroRIVERS | Polyline network | `dist_to_river` (km) |
| IPIS Mining Sites | Point locations | `dist_to_diamond_mine`, `dist_to_gold_mine`, `dist_to_large_mine`, `dist_to_controlled_mine`, `dist_to_large_gold_mine` |
| OSM Settlements | Point locations | `dist_to_city`, `dist_to_capital` (km) |
| CAR Boundary | Polygon | `dist_to_border` (km) |

---

## 7. Demographics

### 7.1 WorldPop Population

| Field | Value |
|-------|-------|
| Source | WorldPop 100m constrained (R2025A) |
| Native Spatial | 100m raster |
| Native Temporal | Annual |
| Aggregation | H3-5 zonal sum, annual |
| Imputation | Forward-fill within each hex; pre-coverage gaps set to 0 (no backward extrapolation) |
| Transform | `log1p(pop_count)` → `pop_log` |
| Coverage | 2000–2030 |
| Structural Break | 2015: V1 (census-adjusted) → V2 (constrained model), flagged via `is_worldpop_v1` |

**Output Columns:**
- `pop_count`
- `pop_log`
- `is_worldpop_v1`

---

## 8. Temporal Context Features

| Feature | Description |
|---------|-------------|
| `month_sin` | Sine of month (cyclical seasonality) |
| `month_cos` | Cosine of month (cyclical seasonality) |
| `is_dry_season` | Binary: 1 if Nov-Mar |

---

## 9. Structural Break Handling

| Break Flag | Threshold Date | Purpose |
|------------|----------------|---------|
| `is_worldpop_v1` | Pre-2015 | Census-adjusted vs constrained population |
| `iom_data_available` | Pre-2015-01-31 | IOM DTM data start |
| `econ_data_available` | Pre-2003-12-01 | Yahoo Finance coverage start |
| `ioda_data_available` | Pre-2022-02-01 | IODA monitoring start |
| `landcover_avail` | Pre-2015-06-27 | Dynamic World start |

---

## 10. Feature Count Summary

**Total Features: 141 unique columns (pre-pruning)**

| Category | Count |
|----------|-------|
| Environmental | 29 |
| Conflict | 20 |
| ACLED Hybrid NLP | 13 |
| NLP & Narrative (v2.0) | 27 |
| Economic | 20 |
| Socio-Political | 13 |
| Infrastructure | 12 |
| Demographics | 4 |
| Temporal Context | 3 |
| **TOTAL** | **141** |

---

## Document Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-11 | Initial release |
| 2.0 | 2026-01-07 | Added macro-economic, ACLED Hybrid NLP, seasonal features |
| 3.0 | 2026-01-15 | Corrected ACLED drivers to 5, added IODA initial |
| 4.0 | 2026-01-17 | IOM date → 2015-01-31, Econ date → 2003-12-01, IODA finalized (2022-02-01), Dynamic World added, variables listed vertically |
| 5.0 | 2026-01-20 | **"Predatory Ecosystem" NLP Refactor (v2.0):** Added CrisisWatch Regime Pillars (4 features), Narrative Velocity, ACLED Hybrid Mechanisms (4 features), Fusion Signals (2 features). Updated NLP/Semantic source count to 3. Total features increased from 111 to 124. |
| 5.1 | 2026-01-24 | **CrisisWatch Interactions & Refinement:** Added 7 interaction/fusion features (CW x ACLED). Updated ACLED Contrastive Anchoring logic and CrisisWatch normalization pipeline. Total features increased from 124 to 131. |
| 5.2 | 2026-01-25 | **Final Feature Registry Update:** Integrated interaction features into main config, removed deprecated ACLED themes and legacy lags. Finalized registry at 138 unique columns before pruning. |
| 5.3 | 2026-01-26 | **VIIRS Integrated Strategy:** Replaced single NTL intensity with Stability, Kinetic, and Uncertainty signals. Added derived `ntl_kinetic_delta`. Feature count increased to 141. |

---

**Generated:** 2026-01-26 | **Pipeline Version:** January 2026
