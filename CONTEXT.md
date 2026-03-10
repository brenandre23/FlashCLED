## 2026-02-20 — L2 Sonnet — Final Sprint: Config Complete, Training IN PROGRESS

**STATUS: CONFIGS DONE. TRAINING NOT YET COMPLETE. START HERE ON NEXT SESSION.**

### What changed this session
1. **Meta-learner** → switched to `logistic` (C=0.1, class_weight=null) — already confirmed winner +12% PR-AUC
2. **Sub-models disabled:** `nlp_crisiswatch` (PR-AUC 1.6×), `nlp_gdelt` (ROC-AUC 0.487 < random)
3. **`configs/pruned_features.yaml`** — feature_count: 109 (was 125). Removed:
   - `month_sin`, `month_cos` (seasonal artifact, OOS ROC-AUC=0.49)
   - `price_maize`, `price_maize_recency_days`, `price_oil`, `price_oil_recency_days`, `price_rice`, `price_rice_recency_days` (zero RF importance, raw levels)
   - `food_price_index_recency_days` (zero RF importance)
   - `sp500_index_lag1` (not in any named sub-model)
4. **Economics sub-model** effective features (9): `price_maize_shock`, `price_rice_shock`, `price_oil_shock`, `gold_price_usd_lag1`, `oil_price_usd_lag1`, `eur_usd_rate_lag1`, `food_price_index`, `food_data_available`, `econ_data_available`
5. **Temporal context** effective features (1): `is_dry_season` only

### 8-Phase Implementation Plan (NEXT STEPS)

**Phase 2 — Training (START HERE)**
```bash
conda run -n geo_env python main.py --skip-static --skip-dynamic --skip-features
```
Runtime: ~60–90 min. Previous runs were killed by OOM (suspended zombie processes holding 5 GB RAM + 95% swap). WSL was shut down to clear memory before this run.

**Phase 3 — Archive**
```bash
conda run -n geo_env python scripts/archive_run.py --label logistic_pruned
```

**Phase 4 — Diagnostics (sequential)**
```bash
conda run -n geo_env python pipeline/analysis/analyze_predictions.py
conda run -n geo_env python research_questions_diagnostic.py
conda run -n geo_env python pipeline/analysis/generate_fast_shap.py
```

**Phase 6 — LaTeX Updates**
- `04-methodology-II.tex`: sub-model count 11→9 (2 disabled), meta-learner → logistic
- `05-results.tex`:  fresh diagnostics must be updated once model is done training
- `08-appendix.tex`: remove temporal_context/nlp_crisiswatch/nlp_gdelt rows from sub-model table; update meta-learner params
---

## 2026-02-19 — L2 Sonnet — Polish Sprint (COMPLETE)
**4 parallel workers executed simultaneously:**

**Worker A — Appendix layout:**
- `\small` → `\fontsize{9}{11}\selectfont` in all 10 appendix table files
- `harmonization-table.tex`: cols 4–5 narrowed 4.6→3.8 cm (total 19.6 cm, no overflow)
- `imputation-registry.tex`: 135 feature rows → 16 domain groups + footnote
- `hyperparameter-table.tex`: 17 rows → 13 (removed non-tuned/impl-detail rows)
- `reproducibility-table.tex`: no stale entries found


Python scripts:
- OUTPUT_DIR in both generate_thesis_figures.py and research_questions_diagnostic.py → `Overleaf/Newest Figures/`
- Removed fig_2_3 and fig_3_3 from FIGURE_REGISTRY and deleted their generator functions
- fig_5_7 now saves extra copy as `spatial_residuals.png` to satisfy the Spatial Diagnostics LaTeX ref

LaTeX:
- `main.tex` graphicspath: added `{Newest Figures/}`
- `04-methodology-II.tex`: feature count corrected → 85 raw + 53 PCA = 138 total (was 118/125/136)
- `05-results.tex`: fig_5_7_recall_top10.png → fig_5_7.png; fig_5_8 and fig_5_9 inserted with captions before FloatBarrier
- Double spaces: confirmed none in prose text (LaTeX indentation was intentional)

---

## 2026-02-19 — L3 GSD Sprint — Model-Independent Thesis Fixes (COMPLETE)

**8 tasks executed via parallel worker pipeline (GSD methodology).**

**Files changed:**
- `Overleaf/sections/02-Literature-review.tex` — Fig 2.2/2.3 caption attribution strengthened (honest authorship, birch_2007 added)
- `Overleaf/sections/03-methodology-I.tex` — Bangui-Bimbo corrected: 3 zones (not 4), per-zone splits shown, source flagged for verification
- `Overleaf/sections/04-methodology-II.tex` — "(hereafter referred to as adaptive thresholds)" added at first use
- `Overleaf/sections/07-conclusion.tex` — Two new Future Work subsections: "Causal Inference and Interpretability" (SHAP/DoWhy) + "NLP Sub-Theme Architecture Refinement" (4-stream proposal)
- `Overleaf/declarations.tex` — SDG 16 logo block inserted before \newpage using existing Icons/ asset
- `Overleaf/bibliography.bib` — Added sharma_kiciman_2020 (DoWhy, arXiv:2011.04216)
- `Overleaf/Figures/fig3.3(b).png` — "FIG 3.3 - SPATIO-TEMPORAL INDEXING" baked-in text removed via PIL; bottom whitespace cropped

**Remaining for next session (see TASK_QUEUE.md §5):**
- Bangui-Bimbo data source verification (ICASEES vs WorldPop)
- Fig 2.4 image decision (fig_2_4.png vs acled_fatality_distribution_verified.png)
- Delete stale 2.4 placeholder PNG

**Still blocked on model:**
- Step 1 predictions (14d LightGBM + 30d/90d horizons) still running
- Step 2 diagnostics, Figure 5.8/5.9, Chapter 5 placeholder fill — all gated on Step 1

---

## 2026-02-16 20:15 — L2 Sonnet — Step 4: Figure Generation (Partial Complete)

**Context:** Executing Steps 1-4 after user approved thesis integration of tiered operational system.

**Step 4 Progress:**
✅ Added two new thesis figures to `generate_thesis_figures.py`:
- **Figure 5.8**: Pareto efficiency curve (Recall @ Tier)
  - Shows fatality capture rates at Critical (5%), High (15%), Elevated (30%) thresholds
  - Includes Pareto ratios and comparison to random baseline
  - Visualizes efficiency gains from model-guided monitoring

- **Figure 5.9**: Risk tier spatial distribution map
  - Geographic visualization of tier assignments with thesis color palette
  - Shows Critical (red), High (orange), Elevated (yellow) cells
  - Includes summary statistics and date context

**Files Modified:**
- `generate_thesis_figures.py`:
  - Lines 423-438: Added 2 new FigureSpec registrations (5.8, 5.9)
  - Lines 2568-2740: Added 2 new generator functions with robust error handling

**Remaining Step 4 Work:**
- Figure 5.Y (NLP Tactical Shift): Generated by `scripts/nlp_tactical_impact.py` (already created in Step 3)
- Figure 5.2 PR-AUC update: Will auto-update once new `comparison_metrics.csv` generated by diagnostics

**Background Status:**
- Step 1 (predictions): 14d XGBoost DONE (1.79M rows with risk_tier), 14d LightGBM in progress
- Next: Wait for Step 1 completion, then run Step 2 (diagnostics)

---

Project Snapshot
- Goal: Conflict Early Warning & Prediction System for CAR; thesis-ready figures and analysis.
- Stack: Python 3.10/3.11, PostgreSQL + PostGIS + H3, Google Earth Engine, LightGBM/XGBoost hurdle ensemble.
- Spatial/temporal: H3 resolution 5 (~10 km), 14-day spine, horizons 14d/1m/3m.
- Key entrypoints: `main.py` (orchestration), `pipeline/` (phases), `generate_thesis_figures.py` (figures), `analysis/` outputs in `data/processed/analysis/`.
- Data paths: raw `data/raw/`, processed `data/processed/`, models `models/`, configs `configs/`.

Current Milestone
- Fix/verify figures and tables for thesis (see `Overleaf/figure_fixes.txt`).
- Address empty fatality scatter and related analysis plots.

Where things land
- Predictions & SHAP: `data/processed/predictions_*.parquet`, `data/processed/shap_explanations_*.parquet`.
- Analysis plots/tables: `data/processed/analysis/` (e.g., `comparison_metrics.csv`, `fatality_scatter.png`).
- Figures: `Figures/` (final exports), intermediate source images from `data/processed/analysis/`.

## 2026-02-16 — L2 Sonnet — ACLED Precision Filter Fix (Critical)

**Problem Found:**
During verification, discovered inconsistent precision filtering across ACLED pipelines:
- `process_conflict_features.py`: Correctly filtered to geo/time precision 1-2
- `process_acled_hybrid.py`: Incorrectly included ALL precision levels (1-3) despite spatial H3 aggregation

This meant 206 events with geo-precision=3 (imprecise coordinates like "somewhere in prefecture") were being assigned to specific 10km H3 hexagons for NLP features (mechanism detection, actor risk), creating spatial noise.

**Fix Applied:**
1. **Code** (`pipeline/processing/process_acled_hybrid.py`, lines 624-637):
   - Added precision filter to SQL query: `AND geo_precision IN (1, 2) AND time_precision IN (1, 2)`
   - Now consistent with conflict feature processing
   - Will exclude 206 geo-level-3 + 149 time-level-3 events from NLP features

2. **Thesis** (`Overleaf/sections/03-methodology-I.tex`, lines 158-162):
   - Added explicit "Precision Filtering" subsection documenting the filter
   - Clarified that filter applies to ALL ACLED-derived features (events, fatalities, AND text/NLP)
   - Emphasized rationale: prevent spurious spatial aggregation artifacts
   - Documented minimal data loss (96.6% of events already at levels 1-2)

**Next Step (User Action Required):**
User will re-run NLP processing to regenerate `features_acled_hybrid` table with corrected precision filter.

**Impact:**
- More spatially reliable NLP features
- Thesis now accurately documents data processing
- Consistent precision standards across all ACLED-derived features

## 2026-02-16 — L2 Sonnet — Per-Column Availability Date Validation

**Enhancement Implemented:**
Added per-column availability date support to data quality validation system to eliminate false warnings for multi-source tables like `environmental_features`.

**Changes Made:**

1. **configs/features.yaml** - Added `available_from` field to 50+ features:
   - VIIRS NTL features: "2012-01-28"
   - Dynamic World landcover: "2017-04-01"
   - GDELT events: "2015-02-18"
   - IODA outages: "2022-01-01"
   - IOM displacement: "2018-01-31"
   - Food security prices: "2018-01-31"
   - CrisisWatch NLP: "2003-08-01"
   - Economy features: "2003-12-01"
   - ACLED Hybrid NLP: "2000-01-15"

2. **pipeline/processing/utils_processing.py** - Three function updates:
   - `get_available_from_date()`: New helper to look up per-column dates from features.yaml
   - `compute_expected_nan_percentage()`: Added `available_from` parameter, prefers per-column date over table-level MIN(date)
   - `sanitize_numeric_columns()`: Updated call site to look up and pass available_from dates

**Benefits:**
- Eliminates false "unexpected NaN" warnings for VIIRS (was showing 47% NaN as unexpected when it's correct pre-2012 behavior)
- Catches real data quality issues (e.g., VIIRS missing after 2012 would now be flagged)
- Documents data availability explicitly in feature configs
- Backward compatible (features without available_from use table-level dates)

**Example Impact:**
- Before: "ntl_peak: 47.165% observed vs 0.000% expected (Available from 2000-01-01)" ❌ FALSE ALARM
- After: "ntl_peak: 47.165% observed vs 47.xxx% expected (Available from 2012-01-28)" ✅ CORRECT

**Next Pipeline Run:**
Feature matrix validation will now use per-column dates and eliminate noisy warnings.
