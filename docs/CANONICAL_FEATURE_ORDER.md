# Canonical Feature Processing Order

This document defines the canonical execution order for feature availability flags,
recency values, missing-value handling, publication lags, and derived transforms
(including anomalies, shocks, and NTL kinetic delta).

It is based on the current ETL implementation in:
- `pipeline/processing/process_spine_and_infrastructure.py`
- `pipeline/processing/process_conflict_features.py`
- `pipeline/processing/process_acled_hybrid.py`
- `pipeline/modeling/build_feature_matrix.py`

## Scope

- This is the **processing-time order** (how features are created).
- This is not the same as the registry listing order in `configs/features.yaml`.
- Different source families run in different phases, then are merged onto the same spine.

## Global Stage Order

1. Build master H3 x Date spine.
2. Apply source-specific temporal alignment/publication lag.
3. Merge source data onto spine.
4. Apply source/domain missing-value policy:
   - `flow`: zero-fill (`0`)
   - `stock`: forward-fill to domain limit, then preserve `NaN`
5. Build recency features for stock-like series (where defined).
6. Build source-local derived features (lags/anomalies/shocks/decays/interactions/spatial lags).
7. Compute/overwrite availability flags (post-imputation, data-aware).
8. Run numeric sanitization and NaN diagnostics.
9. Upload/upsert into `temporal_features`.
10. Assemble model matrix from `temporal_features` + `features_static` + `features_acled_hybrid`.

## Family-Specific Order

### Economics (global market series)

1. Merge-asof onto spine (`14d` tolerance).
2. Impute as `economic` domain (`stock` rules).
3. Compute `lag_1_step` outputs.
4. Compute `econ_data_available`.

### Food security (local prices)

1. Spatial broadcast + merge.
2. Impute price columns as `economic` domain (`stock` rules).
3. Compute recency (`price_*_recency_days`).
4. Compute shocks (`price_*_shock` as ratio to rolling 12m baseline).
5. Compute `food_price_index`.
6. Compute `food_price_index_recency_days`.
7. Compute `food_data_available`.

### Social (IOM displacement)

1. Apply publication lag (`IOM_DTM` from `features.temporal.publication_lags`).
2. Merge lagged records onto spine.
3. Impute as `social` domain (`stock` rules).
4. Compute recency (`iom_displacement_sum_recency_days`).
5. Compute lag output (`iom_displacement_count_lag1`).
6. Compute `iom_data_available`.

### Demographic (WorldPop)

1. Merge annual population.
2. Forward-fill within hex (stock-like behavior).
3. Compute transformed output (`pop_log`).
4. Compute structural break flag (`is_worldpop_v1`).

### Environmental (GEE core)

1. Merge raw environmental columns.
2. Impute non-NTL environmental columns as `environmental` domain (`stock` rules).
3. Compute recency for `ntl_mean`.
4. Compute transformed outputs (including anomaly-registered outputs).
5. Compute `viirs_data_available` (then see VIIRS override below).

### Dynamic World

1. Merge landcover fractions.
2. Impute as `environmental` domain (`stock` rules).
3. Compute `landcover_data_available`.

### VIIRS override block (post environment + dynamic world)

1. Recompute strict `viirs_data_available` with dual guard:
   - `date >= 2012-01-28`
   - `ntl_mean.notna()`
2. Compute `ntl_kinetic_delta = clip(ntl_peak - ntl_mean, lower=0)`.
3. Preserve `NaN` propagation for missing inputs.

### Conflict/NLP overlay phase

Conflict features are layered onto the existing context spine.

#### ACLED core

1. Bin events to spine dates.
2. Merge sparse aggregates to full spine.
3. Zero-fill conflict flow columns.
4. Compute lag features on full spine (for example `fatalities_lag1`).

#### ACLED hybrid mechanisms

1. Apply publication lag (from data config + step days).
2. Snap to spine grid.
3. Aggregate/upload mechanism features.
4. Merge into temporal layer (via conflict processing path).

#### GDELT

1. Bin to spine dates.
2. Pivot local variables.
3. Build spatial lags.
4. Merge local + national aggregates.
5. Zero-fill GDELT flow columns.
6. Compute decays and derived lagged aggregates.
7. Compute `gdelt_data_available`.

#### CrisisWatch

1. Apply publication lag (`NLP_CrisisWatch`).
2. Bin to spine dates and pivot.
3. Apply bounded persistence (`ffill(limit=2)`), then zero-fill.
4. Build composites, deltas, decays, narrative derivatives, interactions, spatial diffusion.

#### IODA

1. Bin and aggregate outage signals.
2. Zero-fill outage flow output.
3. Compute `ioda_data_available` (date-gated).

#### Fusion features

1. Compute fusion outputs such as `gdelt_shock_signal`.

## Placement of Requested Feature Types

- Publication lags: before merge/bucketing for lagged sources.
- Missing-value policy: immediately after merge, before recency and most derived transforms.
- Recency values: after imputation for stock-like series.
- Availability flags: after imputation/derivation within each family; VIIRS is explicitly re-guarded later.
- Anomalies:
  - Climate anomaly outputs are currently produced in the environmental transform block.
  - Anomaly transforms compute standardized Z-scores over a rolling window baseline to measure acute deviations.
- Shocks:
  - Food shocks are generated after recency from imputed price series.
  - `gdelt_shock_signal` is generated in fusion phase of conflict processing.
- Kinetic delta:
  - Computed in dedicated VIIRS derived step after environmental + dynamic-world phases.

## Precedence Rules

When duplicated semantics exist, precedence is:

1. Dedicated processing-script computation (for example strict VIIRS flag, kinetic delta, conflict fusion).
2. Source-family transform block.
3. Registry declaration in `features.yaml` (descriptive/config intent, not guaranteed execution engine).

## Practical Interpretation

- There is no single universal row-by-row transform order for all features.
- There is a canonical **phase order** with source-specific inner order.
- Model-facing consistency comes from:
  - writing all outputs to `temporal_features`,
  - then deterministic assembly in `build_feature_matrix.py`.
