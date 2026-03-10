# PRD: Enhanced Hard-Onset Diagnostic Script (v2)

**Feature:** `scripts/diagnostics/onset_diagnostic.py` — full rebuild
**Status:** DRAFT — awaiting approval
**Author:** L3 Architect
**Date:** 2026-03-10

---

## 1. Problem Statement

The v1 script (`onset_diagnostic.py`) was written and validated in the same session as initial
analysis. Five structural limitations were immediately identified in the output that make v1 unsuitable
for thesis citation or rigorous evaluation:

| # | Issue | Impact |
|---|-------|--------|
| 1 | **Lead-time saturation** — LOOKBACK_STEPS=6 is the ceiling; 226/266 onset events hit t-6, giving a flat, uninformative distribution | Understates true early-warning capability |
| 2 | **t-0 signal dip unflagged** — mean_prob drops at exact onset window; v1 shows the numbers but offers no structural explanation or thesis framing | Key result goes un-contextualized |
| 3 | **Never-flagged blind spots uncharacterized** — 43 events (13.9%) are silently missed; no geographic, temporal, or data-quality analysis | Cannot explain where model fails |
| 4 | **PR-AUC baseline wrong** — reports comparison implicitly against global conflict rate; onset-only random baseline (~0.0025) and explicit lift ratio not shown | Misleading metric framing |
| 5 | **No feature-level explanation for onset** — script treats the model as a black box; no link between structural absence of `fatalities_lag1` and model behaviour for early-warning cases | Thesis RQ2 / RQ3 gap |

---

## 2. Goals

1. Fix all 5 structural issues in a single, authoritative v2 script.
2. Produce output that can be directly cited in thesis Chapter 5 (Results).
3. Keep the script self-contained (reads from existing parquet files, no DB dependency).
4. Optional `--plot` mode generates publication-quality figures to `Overleaf/Newest Figures/`.

---

## 3. Non-Goals

- Not changing the model or retraining.
- Not modifying the linear meta-learner (separate effort).
- Not adding new data sources.

---

## 4. Detailed Requirements

### 4.1 Lead-Time Extension (Fix #1)

**Problem:** `LOOKBACK_STEPS=6` saturates. True lead-time distribution is unknown.

**Solution:**
- Extend `LOOKBACK_STEPS` to **16** (= 224 days ≈ 7.5 months).
- Since `predictions_14d_xgboost.parquet` already covers 2000–2025 (all rows), in-sample
  predictions are already available — no re-generation needed.
- Add a sentinel note in output: "X events exceed 16-step window (truncated)" so the ceiling
  is always visible.
- Report both: (a) distribution histogram by step, (b) cumulative detection curve
  (% of onset events detected vs. lead time in days).

**Acceptance criteria:**
- Distribution no longer saturates at final bin.
- Cumulative curve reaches final plateau before the ceiling.

---

### 4.2 Structural Signal Dip — Thesis Framing Section (Fix #2)

**Problem:** The t-0 probability dip is a clean thesis result but v1 buries it in a raw number table.

**Solution:**
- Add a dedicated `── STRUCTURAL SIGNAL DIP ANALYSIS ──` section.
- Break the signal-buildup table by `fatalities_lag1` bins: `{0, 1–4, 5–19, 20+}` to show
  that the dip is *exclusively* in the zero-lag bin (hard onset) and absent in escalation.
- Print an explicit interpretive note:
  > "The probability dip at t-0 for onset events is structural: `fatalities_lag1 == 0`
  > by definition removes the model's strongest predictor. Early-warning signal (t-1, t-2)
  > originates from NLP / environmental / economic features, not conflict history."
- If `--plot` is set, produce a bar chart: mean conflict_prob at t-2, t-1, t-0 for each
  lag bin — this is directly suitable for a thesis figure.

**Acceptance criteria:**
- Interpretive note appears in stdout regardless of `--plot`.
- Plot (when requested) clearly separates onset from escalation signal trajectories.

---

### 4.3 Never-Flagged Blind Spot Analysis (Fix #3)

**Problem:** 43 missed onset events are not characterized.

**Solution:**
- For each never-flagged onset event, extract from the feature matrix:
  - `dist_to_border`, `dist_to_capital`, `dist_to_road` (geographic periphery proxy)
  - `viirs_data_available`, `ntl_stale_days` (data quality)
  - `iom_displacement_sum_recency_days`, `food_price_index_recency_days` (staleness)
  - `fatalities_lag1` (confirm == 0), `target_fatalities_1_step` (severity of missed event)
- Compare mean values of each variable: **never-flagged** vs **detected onset** vs **all OOS negatives**.
- Output a formatted comparison table.
- Temporal clustering: list months / quarters with highest concentration of never-flagged events.
- If `--plot`, produce a scatter plot of never-flagged events by `dist_to_border` vs `ntl_stale_days`,
  coloured by `target_fatalities_1_step` (severity).

**Acceptance criteria:**
- Output clearly identifies whether never-flagged events are peripheral, data-poor, or random.
- Comparison table shows statistically distinguishable patterns (or explicitly notes absence of pattern).

---

### 4.4 Corrected PR-AUC Baseline (Fix #4)

**Problem:** Lift ratio is computed against the wrong baseline.

**Solution:**
- For each partition (all positives, onset, escalation) compute:
  - `random_baseline = n_pos / (n_pos + n_neg)`  ← partition-specific
  - `lift = PR_AUC / random_baseline`
- Display as an additional column in the Partition Metrics table.
- Add a footnote: "Lift = PR-AUC ÷ random baseline (positive rate for that partition)."

**Acceptance criteria:**
- Onset lift displays ~15× (not implicitly ~11× vs global baseline).
- Table is self-explanatory without additional context.

---

### 4.5 Feature-Level Signal Source for Onset (Fix #5)

**Problem:** v1 is a black box — no link between feature absence (`fatalities_lag1 == 0`) and
what IS driving early-warning for onset cells.

**Solution:**
- Load the 14d XGBoost model pickle (`data/models/two_stage_ensemble_14d_xgboost.pkl`).
- For onset events detected ≥2 steps in advance, run SHAP on the predictions at t-1 and t-2
  (i.e., the flagging period, before onset fires).
- Aggregate mean |SHAP| by feature theme group (using the theme_map from `configs/models.yaml`):
  `conflict_history`, `environment`, `nlp`, `market`, `displacement`, `geography`, etc.
- Compare theme contributions: onset t-1 vs escalation t-1.
- This directly answers: "What IS driving early warning for onset cells, if not conflict history?"

**Acceptance criteria:**
- SHAP theme table shows `conflict_history` contribution is lower for onset-at-t-1 than escalation-at-t-1.
- At least one non-conflict-history theme shows higher contribution for onset.
- Falls back gracefully if model pickle is missing (warn + skip section).

---

## 5. Output Specification

### Stdout sections (always produced)
```
1. PARTITION SUMMARY          — counts + positive rates
2. PARTITION METRICS          — PR-AUC, ROC-AUC, Prec, Rec, F1, MCC + lift ratio
3. TIER RECALL TABLE          — Critical/High/Elevated × partition
4. LEAD-TIME DISTRIBUTION     — histogram + cumulative curve (text)
5. STRUCTURAL SIGNAL DIP      — mean prob by lag bin + interpretive note
6. NEVER-FLAGGED ANALYSIS     — comparison table + temporal clustering
7. SIGNAL BUILDUP             — mean prob at t-2, t-1, t-0 by partition
8. FEATURE THEME ATTRIBUTION  — SHAP theme table (onset vs escalation)
```

### Files produced (with `--plot`)
| File | Description |
|------|-------------|
| `onset_diagnostic_14d_xgboost.png` | Multi-panel figure (PR curves, distributions, lead-time, tier recall) |
| `onset_signal_dip_14d_xgboost.png` | Bar chart: mean prob by lag bin across t-2/t-1/t-0 |
| `onset_blindspots_14d_xgboost.png` | Scatter: never-flagged events by geography + data quality |

---

## 6. Implementation Constraints

- Must run under `conda run -n geo_env python`.
- RAM budget: feature matrix is 1.79M rows × 146 cols. Load only required columns.
- SHAP section: sample at most 500 onset events + 500 escalation events to keep runtime < 5 min.
- All hard-coded paths must use `PATHS` from `utils.py`.
- Must not modify `two_stage_ensemble.py` or any training pipeline file.

---

## 7. Files Modified / Created

| File | Action |
|------|--------|
| `scripts/diagnostics/onset_diagnostic.py` | Full rewrite (v2) |

---

## 8. Acceptance Test

```bash
# Full run with plots — must complete without error
conda run -n geo_env python scripts/diagnostics/onset_diagnostic.py --plot

# Verify all 8 sections appear in stdout
conda run -n geo_env python scripts/diagnostics/onset_diagnostic.py 2>/dev/null | grep "──"
```

Expected section headers in output:
```
── 1. PARTITION SUMMARY
── 2. PARTITION METRICS
── 3. TIER RECALL TABLE
── 4. LEAD-TIME DISTRIBUTION
── 5. STRUCTURAL SIGNAL DIP
── 6. NEVER-FLAGGED ANALYSIS
── 7. SIGNAL BUILDUP
── 8. FEATURE THEME ATTRIBUTION
```
