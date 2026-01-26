# Sidecar Analysis Tools

This repository contains specialized analysis scripts ("sidecars") that exist outside the automated orchestration of `main.py`. These tools are designed for manual, periodic execution to validate core assumptions, audit data stability, or provide deep-dive insights into model behavior.

**CRITICAL NOTE:** These scripts are NOT required for the core pipeline to run. They should only be executed by a researcher or maintainer when specific validation or insight is required.

---

## 1. `agrimatrix.py` (Sub-Ensemble Correlation Deep Dive)

**Purpose:** 
A rapid, standalone diagnostic tool for analyzing the predictive signal of individual sub-ensembles (e.g., "Conflict History", "Economics", "Demographics"). It functions as a "sanity check" for feature engineering quality before committing to full model training.

**When to Run:**
*   After major changes to `feature_engineering.py` or feature configuration.
*   When a specific sub-ensemble performs poorly in the main model (to debug why).
*   **Goal:** To be run for every sub-ensemble defined in `configs/models.yaml` to visualize its isolated correlation with future fatalities.

**Key Outputs:**
*   Correlation heatmaps (`agrimatrix_outputs/<submodel>_correlation.png`).
*   Non-linear feature importance rankings (Random Forest-based).
*   Console logs highlighting top predictors and potential data quality issues (NaNs).

**Usage:**
```bash
python agrimatrix.py
```
*Requires `data/processed/feature_matrix.parquet` to exist.*

---

## 2. `acled_actor_correlations.py` (Actor Stability Audit)

**Purpose:**
A temporal stability auditor for the ACLED "Actor1" field. It determines whether the set of "most lethal actors" is stable over time or if it churns rapidly. This directly informs the `ACTOR_RISK_WEIGHTS` used in `process_acled_hybrid.py`.

**When to Run:**
*   **Monthly:** As new ACLED data arrives.
*   If the "Actor Risk" feature in the main model loses predictive power.
*   If the stability score drops below 0.5 (indicating high churn), the static weights in the hybrid pipeline must be updated.

**Key Outputs:**
*   **Stability Score (0.0 - 1.0):** Jaccard index of the top 20 actors month-over-month.
*   **Trend Analysis:** Classifies actors as "Rising", "Falling", or "Flat" based on fatality trends over the last 6 months.

**Usage:**
```bash
python acled_actor_correlations.py
```
*Requires a populated `acled_events` table in the database.*

---

## Maintenance Policy

*   **Do not delete** these files during routine cleanup.
*   **Do not import** these scripts into the main automation pipeline (`main.py`). They are designed for human-in-the-loop validation.
*   If `agrimatrix.py` reveals critical collinearity or leakage, fix the issue in `configs/features.yaml` or the respective processing script, not in the sidecar itself.
