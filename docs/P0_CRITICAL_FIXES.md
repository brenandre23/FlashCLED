# CEWP P0 Critical Fixes

This document contains the code patches for P0 (Critical) issues identified in the audit.

---

## Fix 1: SHAP PCA Column Ordering Assertion

**File:** `pipeline/analysis/analyze_feature_importance.py`

**Problem:** `apply_pca_if_needed()` assumes column order matches training without verification.

**Patch:**

Find the `apply_pca_if_needed` function and replace with:

```python
def apply_pca_if_needed(X: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """
    Reconstructs PCA features from raw inputs if the model was trained with PCA.
    
    CRITICAL: Enforces column ordering to match training exactly.
    """
    if "pca" not in model_bundle:
        return X
    
    pca = model_bundle["pca"]
    scaler = model_bundle["pca_scaler"]
    pca_input_features = model_bundle["pca_input_features"]
    pca_cols = model_bundle["pca_component_names"]
    
    # CRITICAL FIX: Verify all required input features exist
    missing_inputs = [f for f in pca_input_features if f not in X.columns]
    if missing_inputs:
        raise ValueError(
            f"PCA reconstruction failed: Missing input features: {missing_inputs[:10]}..."
            f" (total {len(missing_inputs)} missing)"
        )
    
    # CRITICAL FIX: Enforce exact column ordering from training
    X_ordered = X[pca_input_features].copy()
    
    # Verify ordering matches
    if list(X_ordered.columns) != list(pca_input_features):
        raise ValueError(
            "PCA column ordering mismatch. Training order:\n"
            f"  {pca_input_features[:5]}...\n"
            f"Inference order:\n"
            f"  {list(X_ordered.columns)[:5]}..."
        )
    
    # Apply transformation
    X_scaled = scaler.transform(X_ordered.fillna(0))
    X_pca = pca.transform(X_scaled)
    
    # Create PCA feature DataFrame
    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
    
    # Concatenate with original features (excluding PCA inputs to avoid duplication)
    non_pca_cols = [c for c in X.columns if c not in pca_input_features]
    result = pd.concat([X[non_pca_cols], pca_df], axis=1)
    
    logger.info(f"✓ PCA reconstruction: {len(pca_input_features)} inputs → {len(pca_cols)} components")
    
    return result
```

---

## Fix 2: Forward-Fill Temporal Guards

**File:** `utils.py`

**Problem:** `apply_forward_fill()` can fill across temporal boundaries, potentially leaking information.

**Patch:**

Replace the existing `apply_forward_fill` function:

```python
def apply_forward_fill(
    df: pd.DataFrame, 
    col: str, 
    groupby_col: str = 'h3_index',
    config: dict = None,
    domain: str = "default",
    max_gap_days: int = None
) -> pd.Series:
    """
    Apply forward-fill imputation with temporal boundary enforcement.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column to fill
    col : str
        Column name to forward-fill
    groupby_col : str
        Column to group by (default: 'h3_index')
    config : dict
        Configuration dictionary with imputation settings
    domain : str
        Domain for looking up imputation limits (e.g., 'environmental', 'conflict')
    max_gap_days : int, optional
        Maximum number of days to forward-fill across. Overrides config.
        
    Returns
    -------
    pd.Series
        Forward-filled column with temporal guards applied
        
    Notes
    -----
    CRITICAL: This function enforces temporal boundaries to prevent data leakage.
    - Respects year boundaries by default
    - Limits fill distance based on domain-specific configuration
    - Logs when filling is capped due to temporal guards
    """
    if col not in df.columns:
        logger.warning(f"Column '{col}' not found in DataFrame. Returning NaN series.")
        return pd.Series(index=df.index, dtype=float)
    
    # Get domain-specific fill limit from config
    default_limits = {
        "environmental": 4,  # 4 steps = ~56 days
        "conflict": 2,       # 2 steps = ~28 days
        "economic": 6,       # 6 steps = ~84 days
        "default": 4
    }
    
    fill_limit = default_limits.get(domain, 4)
    
    if config and isinstance(config, dict):
        imputation_cfg = config.get('imputation', {})
        domain_cfg = imputation_cfg.get(domain, {})
        fill_limit = domain_cfg.get('forward_fill_limit', fill_limit)
    
    if max_gap_days is not None:
        # Convert days to steps (assuming 14-day steps)
        step_days = 14
        if config and 'temporal' in config:
            step_days = config['temporal'].get('step_days', 14)
        fill_limit = max(1, max_gap_days // step_days)
    
    # CRITICAL: Create temporal boundary markers
    result = df[col].copy()
    
    if 'date' in df.columns and groupby_col in df.columns:
        # Sort to ensure correct fill direction
        sort_cols = [groupby_col, 'date']
        df_sorted = df.sort_values(sort_cols)
        result = result.loc[df_sorted.index]
        
        # Create year column for boundary detection
        years = pd.to_datetime(df_sorted['date']).dt.year
        
        # Apply grouped forward-fill with limit
        def fill_within_boundaries(group):
            """Fill within group, respecting year boundaries."""
            filled = group.ffill(limit=fill_limit)
            
            # Reset fill at year boundaries
            year_changes = years.loc[group.index].diff().fillna(0) != 0
            if year_changes.any():
                # Re-apply fill only within same-year segments
                segments = year_changes.cumsum()
                filled = group.groupby(segments).ffill(limit=fill_limit)
            
            return filled
        
        result = df_sorted.groupby(groupby_col)[col].transform(fill_within_boundaries)
        
        # Restore original index order
        result = result.loc[df.index]
        
        # Log filling statistics
        original_nulls = df[col].isna().sum()
        remaining_nulls = result.isna().sum()
        filled_count = original_nulls - remaining_nulls
        
        if filled_count > 0:
            logger.debug(
                f"Forward-fill '{col}' ({domain}): "
                f"{filled_count:,} values filled (limit={fill_limit} steps), "
                f"{remaining_nulls:,} nulls remain"
            )
    else:
        # Fallback: simple forward-fill without temporal guards
        result = result.ffill(limit=fill_limit)
        logger.warning(
            f"Forward-fill '{col}': No date/groupby columns found. "
            "Applied simple ffill without temporal guards."
        )
    
    return result
```

---

## Fix 3: ACLED Hybrid Validation

**File:** `pipeline/modeling/build_feature_matrix.py`

**Problem:** Missing ACLED hybrid columns are silently filled with 0 without proper logging or validation.

**Patch:**

Replace the ACLED hybrid loading section (around line 260):

```python
    # --- ACLED HYBRID FEATURES WITH VALIDATION ---
    logger.info("Loading Optimized ACLED Hybrid Features...")
    
    # Define expected hybrid columns
    EXPECTED_HYBRID_COLS = [
        "theme_context_0", "theme_context_1", "theme_context_2", "theme_context_3",
        "theme_context_4", "theme_context_5", "theme_context_6", "theme_context_7",
        "driver_resource_cattle", "driver_resource_mining", "driver_econ_taxation",
        "driver_civilian_abduct", "driver_civilian_loot", "driver_political_coup",
    ]
    
    query_acled = """
        SELECT event_date as date, h3_index,
               theme_context_0, theme_context_1, theme_context_2,
               theme_context_3, theme_context_4, theme_context_5,
               theme_context_6, theme_context_7,
               driver_resource_cattle, driver_resource_mining,
               driver_econ_taxation, driver_civilian_abduct,
               driver_civilian_loot, driver_political_coup
        FROM car_cewp.features_acled_hybrid
    """
    
    try:
        df_acled = pd.read_sql(query_acled, engine)
    except Exception as e:
        logger.error(f"Failed to load ACLED hybrid features: {e}")
        df_acled = pd.DataFrame()
    
    if df_acled.empty:
        logger.warning(
            "⚠️ ACLED hybrid table is empty or missing. "
            f"Filling {len(EXPECTED_HYBRID_COLS)} hybrid features with 0."
        )
        for col in EXPECTED_HYBRID_COLS:
            master_df[col] = 0
    else:
        # Validate columns exist
        actual_cols = [c for c in EXPECTED_HYBRID_COLS if c in df_acled.columns]
        missing_cols = [c for c in EXPECTED_HYBRID_COLS if c not in df_acled.columns]
        
        if missing_cols:
            logger.warning(
                f"⚠️ Missing ACLED hybrid columns (will fill with 0): {missing_cols}"
            )
        
        logger.info(
            f"ACLED hybrid: {len(df_acled):,} rows, "
            f"{len(actual_cols)}/{len(EXPECTED_HYBRID_COLS)} columns present"
        )
        
        # Prepare for merge
        df_acled["date"] = pd.to_datetime(df_acled["date"])
        df_acled["h3_index"] = df_acled["h3_index"].astype("int64")
        
        # Validate date alignment
        acled_date_range = (df_acled["date"].min(), df_acled["date"].max())
        master_date_range = (master_df["date"].min(), master_df["date"].max())
        
        if acled_date_range[0] > master_date_range[0]:
            logger.warning(
                f"⚠️ ACLED hybrid starts at {acled_date_range[0].date()}, "
                f"after feature matrix start {master_date_range[0].date()}. "
                "Early dates will have 0 values."
            )
        
        # Perform merge
        pre_merge_rows = len(master_df)
        master_df = master_df.merge(
            df_acled, 
            on=["date", "h3_index"], 
            how="left",
            validate="1:1"  # Enforce no duplicates
        )
        post_merge_rows = len(master_df)
        
        if pre_merge_rows != post_merge_rows:
            raise ValueError(
                f"ACLED hybrid merge changed row count: {pre_merge_rows} → {post_merge_rows}. "
                "Possible duplicate (date, h3_index) combinations in hybrid table."
            )
        
        # Fill missing values and add missing columns
        for col in EXPECTED_HYBRID_COLS:
            if col not in master_df.columns:
                master_df[col] = 0
            else:
                null_count = master_df[col].isna().sum()
                if null_count > 0:
                    logger.debug(f"  {col}: filling {null_count:,} nulls with 0")
                master_df[col] = master_df[col].fillna(0)
        
        # Log coverage statistics
        coverage_stats = {}
        for col in EXPECTED_HYBRID_COLS:
            non_zero = (master_df[col] != 0).sum()
            coverage_stats[col] = non_zero / len(master_df) * 100
        
        avg_coverage = sum(coverage_stats.values()) / len(coverage_stats)
        logger.info(
            f"✓ ACLED hybrid merge complete. "
            f"Average non-zero coverage: {avg_coverage:.1f}%"
        )
```

---

## Verification Steps

After applying these fixes, verify with:

### 1. PCA Column Ordering Test
```python
# Add to tests/test_pca_reconstruction.py
def test_pca_column_ordering():
    """Verify PCA reconstruction maintains column order."""
    import joblib
    from pathlib import Path
    
    model_path = Path("models/two_stage_ensemble_14d_xgboost.pkl")
    bundle = joblib.load(model_path)
    
    if "pca" in bundle:
        input_features = bundle["pca_input_features"]
        loadings = bundle["pca_loadings"]
        
        # Verify loadings columns match input features
        assert list(loadings.columns) == list(input_features), \
            "PCA loadings columns don't match input features"
        
        print(f"✓ PCA column ordering verified: {len(input_features)} features")
```

### 2. Forward-Fill Temporal Guard Test
```python
# Add to tests/test_forward_fill.py
def test_forward_fill_year_boundary():
    """Verify forward-fill respects year boundaries."""
    import pandas as pd
    from utils import apply_forward_fill
    
    # Create test data with year boundary
    df = pd.DataFrame({
        'h3_index': [1, 1, 1, 1],
        'date': pd.to_datetime(['2020-12-15', '2020-12-29', '2021-01-12', '2021-01-26']),
        'value': [10.0, None, None, 20.0]
    })
    
    result = apply_forward_fill(df, 'value', config=None, domain='environmental')
    
    # Value at 2020-12-29 should be filled from 2020-12-15
    assert result.iloc[1] == 10.0, "Should fill within same year"
    
    # Value at 2021-01-12 should NOT be filled (year boundary)
    assert pd.isna(result.iloc[2]), "Should not fill across year boundary"
    
    print("✓ Forward-fill temporal guard working")
```

### 3. ACLED Hybrid Validation Test
```python
# Add to tests/test_acled_hybrid.py
def test_acled_hybrid_merge_validation():
    """Verify ACLED hybrid merge doesn't change row count."""
    import pandas as pd
    from sqlalchemy import create_engine
    
    engine = create_engine("postgresql://...")
    
    # Get row counts
    master_count = pd.read_sql(
        "SELECT COUNT(*) FROM car_cewp.temporal_features", engine
    ).iloc[0, 0]
    
    hybrid_count = pd.read_sql(
        "SELECT COUNT(DISTINCT (event_date, h3_index)) FROM car_cewp.features_acled_hybrid", 
        engine
    ).iloc[0, 0]
    
    # Check for duplicates
    dup_count = pd.read_sql("""
        SELECT COUNT(*) FROM (
            SELECT event_date, h3_index, COUNT(*) 
            FROM car_cewp.features_acled_hybrid 
            GROUP BY event_date, h3_index 
            HAVING COUNT(*) > 1
        ) dups
    """, engine).iloc[0, 0]
    
    assert dup_count == 0, f"Found {dup_count} duplicate (date, h3_index) in hybrid table"
    print(f"✓ ACLED hybrid: {hybrid_count} unique (date, h3_index) pairs, 0 duplicates")
```

---

## Deployment Checklist

- [ ] Apply Fix 1 to `pipeline/analysis/analyze_feature_importance.py`
- [ ] Apply Fix 2 to `utils.py`
- [ ] Apply Fix 3 to `pipeline/modeling/build_feature_matrix.py`
- [ ] Run verification tests
- [ ] Re-run full pipeline to verify no regressions
- [ ] Update `docs/CHANGELOG.md` with fixes

---

*End of P0 Fixes Document*
