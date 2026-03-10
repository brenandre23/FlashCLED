# FlashCLED — ML Specification Reference (for Slides)

---

## 1. Train / Validation / Test Splits

**Strategy:** Expanding Window Cross-Validation (time-blocked to prevent leakage)

| Component | Period | Notes |
|-----------|--------|-------|
| **Reference baselines** | 2000–2020 | Used for climate anomaly z-scores (CHIRPS, MODIS NDVI) |
| **Training data** | 2018–2023 (ACLED/conflict data from 2015) | Expanding window; ROC-AUC stable >0.99 on training folds (2021–2023) |
| **Out-of-time test set** | 2024–2025 | Strict temporal "Wall of Separation" — no data leakage |

- **Cross-validation:** 5 temporal folds, held out by year
- **No SMOTE/oversampling** — synthetic H3-cell interpolations violate spatiotemporal autocorrelation; cost-sensitive learning (`scale_pos_weight`) used instead
- **Class imbalance:** Positive rate ≈ 0.12% (>99% of cell-timesteps record zero fatalities)

---

## 2. Two-Stage Hurdle Ensemble Architecture

The model factorises conflict risk into two sequential stages:

### Stage 1 — Binary Classification (Onset)
- **Task:** Predicts P(conflict event in cell-period) — crossing the "hurdle" from peace (0) to conflict (≥1)
- **Objective:** Binary logistic (XGBoost)
- **Meta-learner:** Stacked XGBoost ensemble
  - `n_estimators = 50`
  - `max_depth = 3` (shallow trees to prevent overfitting on 11 stacked inputs)
  - `scale_pos_weight = 8` (moderate re-weighting; sub-models handle primary imbalance)

### Stage 2 — Severity Regression (Intensity)
- **Task:** Estimates fatality counts conditional on onset (zero-truncated count prediction)
- **Base learners:** Squared-error regression (XGBoost)
- **Meta-learner:** Poisson GLM with log-link
  - Respects non-negative, overdispersed nature of fatality data
  - Poisson assumption applied only at meta-learner level
- **Intensity stability cap:** Predictions clipped at 500 fatalities (99.9th percentile of observed ACLED data) to prevent numerical overflow

### Forecast Horizons
| Horizon | Operational Use | PR-AUC |
|---------|----------------|--------|
| **14-day** | Tactical — immediate force protection | 0.0568 |
| **30-day** | Logistics — pre-positioning | 0.0547 |
| **90-day** | Strategic planning | 0.0454 |

---

## 3. Sub-Model Hyperparameters (XGBoost Base Learners)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_depth` | 8 | Captures complex Conflict × NLP × Env interactions |
| `learning_rate` | 0.03 | Conservative; prevents over-adaptation to rare events |
| `n_estimators` | 350 | Prevents under-fitting on high-dimensional feature sets |
| `subsample` | 0.8 | Stochastic subsampling for robustness to outlier events |
| `colsample_bytree` | 0.85 | Forces learning from weaker NLP features |
| `scale_pos_weight` | 18.0 | Handles class imbalance in theme sub-models (reduced from 35.0 to prevent probability floor inflation) |
| Onset threshold | 0.7 | Actionable risk trigger for binary onset prediction |

---

## 4. Eleven Thematic Sub-Models

Each sub-model is trained independently on an exclusive feature set and contributes an out-of-fold (OOF) probability estimate to the meta-learner. Features are exclusively assigned — no overlap.

| # | Thematic Category | Sub-Model Key | Signal Description | Status |
|---|-------------------|---------------|--------------------|--------|
| 1 | Conflict History | `conflict_history` | Autoregressive ACLED event counts, fatality lags, spatial lags, regional risk indicators | Active |
| 2 | Economics | `economics` | Commodity price shocks (gold, oil), food market prices, exchange rate volatility, internet outage score | Active |
| 3 | Environmental | `environmental` | Climate anomalies (NDVI, CHIRPS, ERA5-Land), VIIRS nighttime light decomposition (stability, kinetic, staleness) | Active |
| 4 | Ethnic Power Relations | `epr` | GeoEPR excluded group counts, settlement proximity, ethnic boundary overlaps | Active |
| 5 | Demographic | `demographic` | WorldPop log-population, availability flags | Active |
| 6 | Terrain | `terrain` | Copernicus DEM elevation/slope, distance to roads (GRIP4), rivers (HydroRIVERS), mining sites (IPIS), dynamic landcover | Active |
| 7 | Temporal Context | `temporal_context` | Dry season indicator, IOM displacement counts and lags | Active |
| 8 | NLP — ACLED | `nlp_acled` | Mechanism detection: gold pivot, predatory taxation, factional infighting, collective punishment (MiniLM contrastive anchoring) | Active |
| 9 | NLP — CrisisWatch | `nlp_crisiswatch` | Regime pillar scores (parallel governance, transnational interference), narrative velocity | Active |
| 10 | NLP — Interactions | `nlp_interactions` | Cross-source fusion signals | Active |
| 11 | Broad PCA | `broad_pca` | Principal-component summary across features | Active |
| ~~12~~ | ~~NLP — GDELT~~ | ~~`nlp_gdelt`~~ | ~~Global news thematic densities~~ | **Excluded** (ROC-AUC: 0.487, below random) |

---

## 5. Bin-Conditional Conformal Prediction (BCCP)

### The Problem
Standard (Marginal) Conformal Prediction achieves target coverage *on average* across the full dataset. With >99% zero-fatality observations, it can achieve 90% marginal coverage simply by correctly predicting "peace" — while failing on rare, high-fatality events. Statistically valid but **operationally useless** for humanitarian planning.

### The Solution
BCCP (following Randahl et al.) partitions the calibration set into **risk bins** and enforces calibrated coverage **within each bin**, ensuring prediction intervals are equally reliable for high-fatality events as for peace periods.

### BCCP Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Alpha (α)** | 0.1 | 90% prediction interval target coverage |
| **Bin Edges** | [0, 1, 3, 8, 21, 55, 149, ∞] | Exponentially-spaced bins for fat-tailed fatalities (following ViEWS/Randahl et al.) |

### How It Works
1. Fit the Two-Stage Hurdle model
2. On a calibration set, compute residuals (predicted − actual fatalities)
3. **Partition** residuals into risk bins using the exponential bin edges above
4. Within each bin, compute the conformal quantile — the (1−α) quantile of absolute residuals
5. At inference, classify the prediction into a risk bin, then apply that bin's quantile to produce an uncertainty interval

### Key Limitation
BCCP bounds were computed on training indices rather than held-out data — they are **indicative rather than formally guaranteed**.

---

## 6. Operational Tiering (Quantile-Based Risk Tiers)

Dynamic quantile thresholds translate probabilities into actionable priorities:

| Tier | Coverage | User Accuracy (Precision) | Producer Accuracy (Recall) | Pareto Ratio | MCC |
|------|----------|--------------------------|---------------------------|--------------|-----|
| **Critical** (Top 5%) | 5.0% | 2.2% | 92.0% | 1:18 | 0.137 |
| **High** (Top 15%) | 15.0% | 0.8% | 95.5% | 1:6 | 0.079 |
| **Elevated** (Top 30%) | 30.0% | 0.4% | 98.0% | 1:3 | — |

Interpretation: Monitoring only the **Top 5%** of cells anticipates **92% of all violence** (Pareto efficiency 1:18).

---

## 7. Key Performance Results (Out-of-Time: 2024–2025)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.913 | Excellent discrimination (interpret cautiously under imbalance) |
| **PR-AUC** | 0.070 | 58× lift over random baseline (0.0012) |
| **MCC** | 0.220 | Robust imbalanced correlation |
| **Balanced Accuracy** | 0.747 | Reliable per-class accuracy |
| **Recall@Top-10%** | 95.8% | Captures 2,010 of 2,099 conflict cells |
| **Within-admin variance** | 22.13 | H3 captures 22× more spatial detail than admin boundaries |
| **F1 fusion gain (NLP)** | +1.7 pp | NLP adds unique complementary signal |
| **Best NLP sub-model** | `nlp_acled` (PR-AUC: 0.061) | Isolated tactical mechanism detection |

---

## 8. Ablation Study Design

Sequential feature addition evaluated on held-out test data:

1. **Baseline:** Structural features only (Infrastructure, Demographics, Environmental anomalies, lagged conflict counts)
2. **+ Economic:** Add price shocks and commodity indicators
3. **+ ACLED NLP:** Add mechanism detection features
4. **+ CrisisWatch NLP:** Add regime pillar scores and narrative velocity
5. **+ Fusion:** Add cross-source interaction terms

**Result:** Full fused model achieves 8.3% relative improvement in F1-score over structural-only baseline.

---

## 9. Interpretability

- **Method:** SHAP (TreeExplainer) — polynomial-time exact Shapley values for tree ensembles
- **Background sample:** N = 2,000 observations from test set (computational constraint on 1.8M rows)
- **SHAP Interaction Values:** Decompose predictions into main effects + pairwise interactions for RQ3 (multi-modal integration)
