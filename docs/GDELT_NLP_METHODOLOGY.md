# GDELT NLP Methodology: The "Dual-Sensor" Approach

## Overview
This document outlines the refactored methodology for ingesting and processing GDELT data. We utilize a **"Dual-Sensor"** architecture that treats GDELT not as a single stream, but as two distinct signals:
1.  **Kinetic Sensor (Events):** Precise, localized reports of physical actions (fights, coercion, assaults).
2.  **Context Sensor (GKG):** Broader thematic narratives (predation, displacement) extracted from the Global Knowledge Graph.

The goal is to filter out the high-volume "noise" inherent in GDELT and capture high-fidelity signals of **Predatory Violence** and **Border Instability**.

---

## 1. The Kinetic Sensor (Events)
**Source:** `gdelt-bq.gdeltv2.events`

Instead of ingesting all political events (which includes handshake agreements and diplomatic visits), we apply a **Predatory Action Filter** based on CAMEO codes. We only care about events that signal "Predatory" behavior.

### 1.1 Predatory CAMEO Codes
We filter `EventCode` for the following prefixes:
*   **10 (Demand):** Demands for leadership change, rights, or resources.
*   **11 (Disapprove/Threaten):** Verbal threats, halting negotiations, sanctions.
*   **17 (Coerce):** Seizure of property, destruction, imposition of curfews.
*   **18 (Assault):** Use of unconventional violence, sexual violence, torture.
*   **19 (Fight):** Use of conventional military force, artillery, small arms.

**Output Feature:** `gdelt_predatory_action_count` (Sum of event weights).

---

## 2. The Context Sensor (GKG Themes)
**Source:** `gdelt-bq.gdeltv2.gkg` (Global Knowledge Graph)

We extract thematic context to understand *why* violence might occur. Instead of using generic "Conflict" themes, we scan for specific **Theme Clusters** relevant to the Conflict Ecosystem.

### 2.1 Theme Clusters
We parse the `V2Themes` column for the following keywords:
*   **Resource Predation:** `NATURAL_RESOURCES`, `MINING`, `SMUGGLING`, `TAX_FNCACT`, `EXTORTION`, `BLACK_MARKET`.
*   **Displacement:** `REFUGEES`, `DISPLACEMENT`, `FAMINE`, `FOOD_SECURITY`.
*   **Governance Breakdown:** `CORRUPTION`, `UNGOVERNED`, `FAILED_STATE`, `REBELLION`, `COUP`.

**Output Features:** `gdelt_theme_resource_predation_count`, `gdelt_theme_displacement_count`, etc.

### 2.2 Strict Spatial Filtering
GKG data is notoriously noisy with location tagging (the "Capital Blob" effect). To combat this, we apply a strict spatial filter:
1.  **Parse `V2Locations`:** Extract specific Lat/Lon coordinates from the article.
2.  **Reject Generics:** If an article mentions "Sudan" but provides no specific coordinates, it is discarded.
3.  **Buffer Check:** A theme is only counted if its coordinates fall inside **CAR** or the **50km Buffer Zone** (see below).

---

## 3. Spatial Logic: The "Ghost Cells"
Violence in CAR is often driven by cross-border dynamics. To capture this, we allow our grid to "sense" activity just outside the national borders.

*   **Grid:** H3 Resolution 5 (approx. 8.5km edge).
*   **Buffer:** 50km buffer around the CAR border into Chad (CD/TD) and Sudan (SU/SD).
*   **Logic:**
    *   Events/Themes inside CAR -> Mapped to their H3 cell.
    *   Events/Themes in Chad/Sudan -> **REJECTED** unless they fall within the 50km buffer.
    *   **Ghost Cells:** H3 cells in the buffer zone accumulate risk scores (e.g., "Troop buildup 10km inside Chad").
*   **Diffusion:** Using `feature_engineering.py`, we calculate **Spatial Lags** (k=1 neighbors). This allows a cell inside CAR to "see" the risk accumulating in a neighboring Ghost Cell across the border.

---

## 4. Signal Fusion (Feature Engineering)
Once the raw counts are ingested, we process them into three distinct signal types for the XGBoost model.

### 4.1 The Local Signal (R5)
*   **What:** The raw count of events/themes in a specific H3 cell.
*   **Use:** Pinpoints exact locations of unrest.
*   **Columns:** `gdelt_predatory_action_count`, `gdelt_theme_resource_predation_count`.

### 4.2 The "Ghost" Signal (Spatial Lag)
*   **What:** The sum of counts in all neighboring cells (including cross-border Ghost Cells).
*   **Use:** Captures spillover risk and proximity to danger.
*   **Columns:** `gdelt_predatory_action_count_spatial_lag`.

### 4.3 The "National Heat" Signal (Context)
*   **What:** The daily sum of *all* counts across the entire study area.
*   **Use:** Provides a baseline "temperature" for the country. If National Heat is high, the baseline probability of violence rises everywhere.
*   **Columns:** `national_predatory_action_sum`, `national_theme_resource_predation_sum`.

### 4.4 Temporal Decay (Stress)
*   **What:** A 30-day half-life decay applied to all signals.
*   **Use:** Ensures that a major event (e.g., a massacre) continues to influence risk scores for weeks after it happens, rather than disappearing the next day.
*   **Columns:** `*_decay_30d`.

---

## Summary of Data Flow
1.  **Ingest:** `fetch_gdelt_dual.py` queries BigQuery with strict filters.
2.  **Map:** Points are mapped to H3 R5 cells.
3.  **Filter:** Points outside CAR and the 50km buffer are dropped.
4.  **Aggregate:** National totals are calculated.
5.  **Diffuse:** Spatial lags are calculated (Ghost Cell fusion).
6.  **Decay:** Temporal half-life is applied.
7.  **Model:** The final feature matrix contains Local, Neighbor, and National signals for every cell/day.
