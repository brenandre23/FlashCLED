# CEWP Codebase Audit Report

**Generated:** 2025-01-07  
**Scope:** Comprehensive audit of Conflict Early Warning Pipeline (CEWP) codebase  
**Root Directory:** `C:\Users\Brenan\Desktop\Thesis\Scratch`

---

## Executive Summary

The CEWP codebase is a production-grade geospatial ML pipeline for predicting armed conflict risk in the Central African Republic. The audit identified **59 Python scripts** across 6 pipeline modules with generally solid architecture but several critical issues requiring attention.

### Critical Issues (P0)
| Issue | Location | Impact | Fix Complexity |
|-------|----------|--------|----------------|
| SHAP PCA reconstruction may cause feature mismatch | `analyze_feature_importance.py` | Model explainability invalid | Medium |
| Missing validation for ACLED hybrid features | `build_feature_matrix.py` | Silent model failures | Low |
| Forward-fill without temporal guards | `feature_engineering.py` | Potential data leakage | Medium |

### High Priority Issues (P1)
| Issue | Location | Impact | Fix Complexity |
|-------|----------|--------|----------------|
| No VIF/collinearity checks in pipeline | Global | Feature redundancy undetected | Medium |
| PCA loadings not persisted for SHAP reconstruction | `train_models.py` | Explainability reproducibility | Low |
| Hardcoded step_days assumption in multiple scripts | Various | Configuration drift risk | Low |

### Moderate Issues (P2)
| Issue | Location | Impact | Fix Complexity |
|-------|----------|--------|----------------|
| Duplicate code patterns across ingestion scripts | `pipeline/ingestion/` | Maintenance overhead | Medium |
| Missing unit tests | Global | Regression risk | High |
| Inconsistent error handling | Various | Debug difficulty | Medium |

---

## 1. Project Structure Analysis

### 1.1 Directory Tree
```
Scratch/
├── main.py                    # Master orchestrator (CEWPPipeline class)
├── utils.py                   # Centralized utilities (load_configs, DB, H3, etc.)
├── init_db.py                 # Database initialization
├── setup.py                   # Package setup
├── clean_cache.py             # Cache cleanup
├── diagnose_*.py              # Diagnostic scripts (6 files)
├── configs/
│   ├── data.yaml              # Data source configuration
│   ├── features.yaml          # Feature registry (96+ features)
│   └── models.yaml            # Model architecture (8 submodels, 3 horizons)
├── pipeline/
│   ├── ingestion/             # 18 scripts
│   ├── processing/            # 9 scripts
│   ├── modeling/              # 4 scripts
│   ├── analysis/              # 10 scripts
│   └── validation/            # 1 script
├── data/
│   ├── raw/
│   └── processed/
├── models/                    # Trained ensemble pkl files
└── docs/                      # Documentation
```

### 1.2 Script Inventory Summary
| Module | Count | Key Scripts |
|--------|-------|-------------|
| ingestion | 18 | fetch_acled, fetch_gee_server_side, ingest_economy |
| processing | 9 | feature_engineering, spatial_disaggregation, process_acled_hybrid |
| modeling | 4 | build_feature_matrix, train_models, two_stage_ensemble |
| analysis | 10 | analyze_feature_importance, analyze_subtheme_shap |
| validation | 1 | run_assertions |
| root | 11 | main.py, utils.py, diagnose_* |

---

## 2. Configuration Audit

### 2.1 `configs/data.yaml`
**Status:** ✅ Well-structured

Key parameters:
- Global date window: 2000-01-01 to 2025-12-31
- Train/test split: test_year=2023, train_cutoff=2020-12-31
- Spatial: CAR bbox [14.0, 2.0, 28.0, 11.5], H3 resolution 5

**Issues Found:**
- ⚠️ `train_cutoff` (2020-12-31) precedes `test_year` (2023) by 2+ years - potential gap in validation set

### 2.2 `configs/features.yaml`
**Status:** ⚠️ Complex, needs documentation

Registry contains 96+ features across 7 categories:
1. Environmental (CHIRPS, ERA5, MODIS, VIIRS)
2. Conflict (ACLED, GDELT)
3. Economic (gold, oil, SP500, EUR/USD)
4. Food Security (maize, rice, sorghum prices)
5. Social/Demographic (IPC, IOM, WorldPop)
6. Static Geography (distances, terrain)
7. ACLED Hybrid (8 semantic contexts + 6 explicit drivers)

**Issues Found:**
- ⚠️ Some features have `enabled: true` but are not in `models.yaml` submodels
- ⚠️ `transformation_params` varies in structure (some have `lookback_months`, others `half_life_days`)
- ❌ No schema validation for registry entries

### 2.3 `configs/models.yaml`
**Status:** ✅ Well-structured

Architecture:
- 3 horizons: 14d (1 step), 1m (2 steps), 3m (6 steps)
- 8 submodels: conflict_history, economics, environmental, epr, demographic, terrain, temporal_context, acled_hybrid
- 2 learners: XGBoost (primary), LightGBM (secondary)
- PCA: 90% variance retention on `broad_pca`

**Issues Found:**
- ⚠️ `broad_pca.features` is dynamically populated at training time - not persisted in config

---

## 3. Core Pipeline Audit

### 3.1 `main.py` - Master Orchestrator
**Status:** ✅ Good architecture

The `CEWPPipeline` class implements 5 phases:
1. Static Ingestion
2. Dynamic Ingestion
3. Feature Engineering
4. Modeling
5. Analysis

**Strengths:**
- Comprehensive `validate_pipeline_prerequisites()` function
- Modular phase execution with skip flags
- Logging throughout

**Issues Found:**
- ⚠️ Phase dependencies are implicit (not enforced programmatically)
- ⚠️ No checkpointing between phases (full restart on failure)

### 3.2 `utils.py` - Centralized Utilities
**Status:** ✅ Production-grade

Key functions:
| Function | Purpose | Status |
|----------|---------|--------|
| `load_configs()` | YAML configuration loader | ✅ |
| `get_db_engine()` | PostgreSQL connection pooling | ✅ |
| `ensure_h3_int64()` | H3 hex→BIGINT conversion | ✅ Critical |
| `upload_to_postgis()` | Upsert via temp table + COPY | ✅ |
| `apply_forward_fill()` | Config-driven imputation | ⚠️ Needs temporal guard |
| `validate_h3_types()` | Pre-flight H3 type check | ✅ |

**Issues Found:**
- ❌ `apply_forward_fill()` lacks temporal boundary enforcement (could fill across temporal boundaries)
- ⚠️ `get_db_engine()` hardcodes some connection parameters

### 3.3 `pipeline/processing/feature_engineering.py`
**Status:** ⚠️ Complex, needs refactoring

This 900+ line script handles all feature transformations:

**Strengths:**
- Config-driven transformations via `parse_registry()`
- `safe_merge()` with deterministic sort
- Seasonal features (month_sin, month_cos, is_dry_season)

**Issues Found:**
| Issue | Line(s) | Severity | Description |
|-------|---------|----------|-------------|
| Forward-fill without temporal guard | Various | P0 | `apply_forward_fill()` could leak future info |
| Hardcoded `step_days` assumptions | ~280 | P1 | Uses `step_days` from config but some calculations assume 14 |
| Price shock window calculation | ~340 | P2 | Complex nested logic, hard to verify |
| Missing null checks after merge_asof | ~195 | P2 | Could propagate NaN silently |

**Recommendation:** Split into sub-modules:
- `fe_economics.py`
- `fe_environment.py`
- `fe_conflict.py`
- `fe_social.py`

### 3.4 `pipeline/modeling/train_models.py`
**Status:** ✅ Solid implementation

Key methodology:
- Two-Stage Hurdle Ensemble (XGBoost base learners)
- Dynamic class weighting (`scale_pos_weight`)
- PCA with 90% variance retention
- TimeSeriesSplit for OOF predictions

**Strengths:**
- `validate_training_data()` with comprehensive checks
- Safety cap on dynamic weight (max 10000)
- PCA bundle persistence (pca, scaler, loadings)

**Issues Found:**
| Issue | Severity | Description |
|-------|----------|-------------|
| PCA loadings transposed | P1 | `pca.components_` rows are components, columns are features - verify SHAP uses this correctly |
| No hyperparameter validation | P2 | Accepts any params from YAML without bounds checking |

### 3.5 `pipeline/modeling/two_stage_ensemble.py`
**Status:** ✅ Clean implementation

Architecture:
```
Level 1: Theme-specific base learners (XGBoost/LightGBM)
         ├── binary_model (conflict occurrence)
         └── regress_model (fatality count | occurrence)

Level 2: Meta-learners
         ├── LogisticRegression (aggregates binary predictions)
         └── PoissonRegressor (aggregates fatality predictions)
```

**Strengths:**
- TimeSeriesSplit prevents leakage in OOF generation
- `log1p` transformation on regressor inputs (prevents double-exponential)
- Non-negativity constraints enforced

**Issues Found:**
- ⚠️ `clone(base_clf)` may not deep-copy all parameters for custom XGBoost configs
- ⚠️ No handling for edge case where all validation fold samples are non-conflict

### 3.6 `pipeline/modeling/build_feature_matrix.py`
**Status:** ⚠️ Needs hardening

**Strengths:**
- Dynamic column classification (temporal vs static vs hybrid)
- Pre-flight column validation
- Streaming with chunked reads

**Issues Found:**
| Issue | Severity | Description |
|-------|----------|-------------|
| ACLED hybrid merge outside SQL | P1 | Second merge after main query - potential date alignment issues |
| Missing features silently filled | P1 | `hybrid_cols` filled with 0 without logging which were missing |
| `dist_` sentinel value arbitrary | P2 | Uses 500km as max distance - should be configurable |

---

## 4. SHAP Analysis Audit

### 4.1 `pipeline/analysis/analyze_feature_importance.py`
**Status:** ❌ Critical issues

**Architecture:**
- MACRO level: Meta-learner coefficients
- MICRO level: SHAP TreeExplainer per theme

**Issues Found:**
| Issue | Severity | Description |
|-------|----------|-------------|
| PCA reconstruction assumes column order | P0 | `apply_pca_if_needed()` must match training column order exactly |
| `check_additivity=False` hides issues | P1 | Disabling additivity check may mask SHAP calculation errors |
| Sample size too small for stable SHAP | P2 | 1000 samples may not capture feature interactions |

**Critical Bug Risk:**
```python
# In apply_pca_if_needed():
X_scaled = scaler.transform(X[input_features].fillna(0))
# RISK: If input_features order differs from training, PCA reconstruction is WRONG
```

**Recommendation:** Add explicit column ordering assertion:
```python
assert list(X[input_features].columns) == list(pca_input_features), \
    "Column order mismatch in PCA reconstruction"
```

### 4.2 `pipeline/analysis/analyze_subtheme_shap.py`
**Status:** ⚠️ Moderate issues

**Strengths:**
- Dedicated per-theme analysis
- 2000 sample size (better than main script)

**Issues Found:**
- ⚠️ No handling for categorical features (SHAP may misinterpret)
- ⚠️ Beeswarm plots may be unreadable with many features

---

## 5. Collinearity Analysis Audit

### 5.1 Current State
**Status:** ❌ No VIF/collinearity detection implemented

The codebase lacks:
1. Variance Inflation Factor (VIF) calculation
2. Correlation matrix thresholding
3. Feature clustering for redundancy detection

### 5.2 Impact Assessment

Without collinearity checks:
- PCA broad_pca may retain redundant dimensions
- SHAP values may be unstable for correlated features
- Model coefficients may be unreliable

### 5.3 Recommended Implementation

```python
# Add to pipeline/processing/collinearity_check.py
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Calculate VIF for all numeric features.
    VIF > 5 indicates moderate collinearity
    VIF > 10 indicates severe collinearity
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].dropna()
    
    vif_data = pd.DataFrame({
        'feature': numeric_cols,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(numeric_cols))]
    })
    
    high_vif = vif_data[vif_data['VIF'] > threshold]
    if not high_vif.empty:
        logger.warning(f"High VIF features (>{threshold}):\n{high_vif}")
    
    return vif_data
```

---

## 6. Code Quality Issues

### 6.1 Duplicate Code Patterns

| Pattern | Locations | Recommendation |
|---------|-----------|----------------|
| DB connection setup | All ingestion scripts | Use `@contextmanager` decorator |
| H3 type conversion | 12+ files | Already centralized in `utils.py` - enforce usage |
| Date range validation | Multiple | Create `validate_date_range()` utility |
| Logging setup | Each script | Move to `utils.py` singleton |

### 6.2 Naming Inconsistencies

| Current | Recommended | Locations |
|---------|-------------|-----------|
| `fat_col` / `fatalities` | `target_fatalities` | train_models.py |
| `bin_col` / `binary` | `target_binary` | train_models.py |
| `acled_raw` / `df_acled` | `acled_events` | feature_engineering.py |

### 6.3 Magic Numbers

| Value | Location | Recommended |
|-------|----------|-------------|
| `9999` (time since event sentinel) | feature_engineering.py:545 | `MAX_DAYS_SINCE_EVENT` constant |
| `500.0` (max distance km) | build_feature_matrix.py:280 | Config parameter |
| `10000.0` (weight cap) | train_models.py:82 | Config parameter |
| `4` (forward-fill limit) | utils.py | Config parameter |

---

## 7. Testing Gaps

### 7.1 Missing Test Coverage

| Component | Test File | Status |
|-----------|-----------|--------|
| H3 type conversion | - | ❌ None |
| PCA reconstruction | - | ❌ None |
| SHAP calculations | - | ❌ None |
| Feature engineering | - | ❌ None |
| Model training | - | ❌ None |

### 7.2 Recommended Test Suite

```
tests/
├── unit/
│   ├── test_h3_utils.py           # H3 type conversion, validation
│   ├── test_feature_engineering.py # Transform functions
│   ├── test_pca_reconstruction.py  # Column ordering, inverse transform
│   └── test_shap_consistency.py    # Additivity, feature alignment
├── integration/
│   ├── test_pipeline_phases.py     # Phase execution order
│   └── test_db_operations.py       # Upsert, schema validation
└── fixtures/
    ├── sample_features.parquet
    └── sample_model.pkl
```

---

## 8. Recommendations Summary

### P0 - Critical (Fix Immediately)

1. **SHAP PCA Column Ordering**
   - Add explicit assertion in `apply_pca_if_needed()`
   - Persist column order with PCA bundle

2. **Forward-Fill Temporal Guards**
   - Add date boundary checking in `apply_forward_fill()`
   - Prevent filling across temporal discontinuities

3. **ACLED Hybrid Validation**
   - Add explicit logging for missing hybrid columns
   - Validate date alignment after merge

### P1 - High Priority (Fix This Sprint)

1. **Implement VIF Checks**
   - Add `collinearity_check.py` module
   - Integrate into feature engineering phase

2. **Configuration Validation**
   - Add JSON schema for YAML configs
   - Validate on load

3. **PCA Loadings Persistence**
   - Ensure loadings are correctly oriented
   - Add reconstruction test

### P2 - Moderate (Fix This Quarter)

1. **Refactor feature_engineering.py**
   - Split into domain-specific modules
   - Add docstrings

2. **Add Unit Tests**
   - Start with H3 utilities
   - Add SHAP consistency tests

3. **Standardize Constants**
   - Move magic numbers to config
   - Create constants module

---

## 9. Technical Debt Inventory

| Item | Location | Effort | Priority |
|------|----------|--------|----------|
| 900-line feature_engineering.py | processing/ | 2 days | P2 |
| No test suite | global | 5 days | P1 |
| Hardcoded DB schema | utils.py | 1 day | P2 |
| Duplicate logging setup | all scripts | 0.5 day | P3 |
| Missing docstrings | multiple | 2 days | P3 |

---

## 10. Appendix

### A. Scripts Not Fully Audited
These scripts were identified but not deeply reviewed:
- `diagnose_*.py` (6 files) - diagnostic utilities
- `clean_cache.py` - cache cleanup
- `setup.py` - package setup

### B. Configuration Files
Full YAML configurations available in `configs/` directory.

### C. Audit Methodology
1. Directory structure mapping
2. Configuration review
3. Core script analysis (main.py, utils.py)
4. Pipeline module deep-dive
5. SHAP/collinearity focus audit
6. Code quality review

---

*End of Audit Report*
