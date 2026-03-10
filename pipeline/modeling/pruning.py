"""
pipeline/modeling/pruning.py
============================
Stability-based feature pruning using permutation importance with time-blocked CV.

ARCHITECTURE:
- Class A (STATIC_CONTEXT): Immutable spatiotemporal anchors - never pruned
- Class B (RESCUE_BUNDLES): Asymmetric dependencies - parent survival rescues children
- Class C (BIDIRECTIONAL_BUNDLES): Atomic pairs - any survival rescues all
- Class D (FAMILY_GUARDS): Group rescue - any family member rescues the flag

ALGORITHM:
1. Time-blocked CV with shadow feature (noise baseline)
2. Permutation importance on held-out validation sets (replaces SHAP due to version bugs)
3. Features beating shadow threshold are "stable"
4. Union across multiple objectives (onset + intensity)
5. Apply bundling logic to preserve theory-driven relationships

UPDATES (2026-01-28):
- Replaced SHAP with permutation importance (avoids SHAP serialization bug)
- Added XGBoost native importance as fallback
- Faster execution, more robust
"""

import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, mean_squared_error

# Internal project imports
from utils import logger

try:
    from pipeline.modeling import load_data_utils
except ImportError:
    logger.warning("Could not import load_data_utils. Pruning will fail if data loading is attempted.")
    load_data_utils = None


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_and_report_dtypes(df: pd.DataFrame, candidates: List[str], target_col: str) -> Dict[str, Any]:
    """Validate dtypes and report issues before training."""
    report = {
        'valid': True,
        'non_numeric_candidates': [],
        'constant_candidates': [],
        'high_null_candidates': [],
        'target_issues': None
    }
    
    for col in candidates:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            report['non_numeric_candidates'].append((col, str(df[col].dtype)))
            report['valid'] = False
            continue
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() <= 1:
            report['constant_candidates'].append(col)
        null_pct = df[col].isna().mean()
        if null_pct > 0.9:
            report['high_null_candidates'].append((col, f"{null_pct:.1%}"))
    
    if target_col in df.columns:
        if not np.issubdtype(df[target_col].dtype, np.number):
            report['target_issues'] = f"Non-numeric dtype: {df[target_col].dtype}"
            report['valid'] = False
    else:
        report['target_issues'] = "Column not found"
        report['valid'] = False
    
    return report


def sanitize_and_filter_columns(
    df: pd.DataFrame,
    candidate_cols: List[str],
    target_cols: List[str],
    coercion_threshold: float = 0.01
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Coerce columns to numeric, strip brackets, drop high-coercion columns."""
    df = df.copy()
    cols_to_process = [c for c in candidate_cols + target_cols if c in df.columns and c != "date"]
    dropped = []
    drop_reasons = {}

    for col in cols_to_process:
        s = df[col]
        non_null = s.notna().sum()
        
        # Force through string cleaning for robustness
        cleaned = s.astype(str).str.strip()
        cleaned = cleaned.str.replace(r'^\[', '', regex=True)
        cleaned = cleaned.str.replace(r'\]$', '', regex=True)
        cleaned = cleaned.replace(['', 'nan', 'NaN', 'None', 'null', 'NULL', 'NA', 'N/A'], pd.NA)
        coerced = pd.to_numeric(cleaned, errors="coerce")

        original_non_na = (~s.isna()).sum()
        coerced_na = coerced.isna().sum()
        new_na = coerced_na - s.isna().sum()
        frac = new_na / original_non_na if original_non_na > 0 else 0
        
        if col in target_cols:
            if frac > 0:
                logger.warning(f"Pruning: target '{col}' had {new_na} coerced values ({frac:.2%}); filling with 0.")
            df[col] = coerced.fillna(0).astype(np.float64)
            continue

        if non_null > 0 and frac > coercion_threshold:
            logger.warning(f"Pruning: dropping '{col}' - coercion failures: {new_na}/{non_null} = {frac:.2%}")
            dropped.append(col)
            drop_reasons[col] = f"coercion_failures_{frac:.2%}"
            continue

        df[col] = coerced.fillna(0).astype(np.float64)

    return df, dropped, drop_reasons


# =============================================================================
# STRUCTURAL DEFINITIONS
# =============================================================================

STATIC_CONTEXT = [
    'month_sin', 'month_cos', 'is_dry_season',
    'dist_to_capital', 'dist_to_border', 'dist_to_road', 
    'dist_to_river', 'dist_to_city', 'dist_to_market_km',
    'terrain_ruggedness_index', 'elevation_mean', 'slope_mean',
    'epr_excluded_groups_count', 'epr_status_mean',
    'epr_discriminated_groups_count', 'ethnic_group_count',
    # --- THESIS ANCHORS (Mandatory for ablation/theory) ---
    'narrative_velocity_lag1', 'mech_gold_pivot_lag1', 
    'acled_combined_risk_score', 'gdelt_shock_signal',
    # --- PROTECTED NLP / GDELT (Soft Signals) ---
    'gdelt_event_count', 'gdelt_avg_tone_decay_30d', 'gdelt_goldstein_mean',
    'gdelt_mentions_total', 'cw_score_local', 'cw_score_local_delta',
    'regime_parallel_governance', 'regime_transnational_predation',
    'regime_guerrilla_fragmentation', 'regime_ethno_pastoral_rupture'
]

RESCUE_BUNDLES = {
    'ntl_mean':           ['ntl_stale_days', 'ntl_trust_frac'],
    'ntl_peak':           ['ntl_stale_days', 'ntl_trust_frac'],
    'ntl_kinetic_delta':  ['ntl_stale_days', 'ntl_peak'],
    'price_maize':        ['price_maize_recency_days', 'price_maize_shock'],
    'price_rice':         ['price_rice_recency_days', 'price_rice_shock'],
    'price_oil':          ['price_oil_recency_days', 'price_oil_shock'],
    'price_sorghum':      ['price_sorghum_recency_days', 'price_sorghum_shock'],
    'price_cassava':      ['price_cassava_recency_days', 'price_cassava_shock'],
    'price_groundnuts':   ['price_groundnuts_recency_days', 'price_groundnuts_shock'],
    'food_price_index':   ['food_price_index_recency_days'],
    'price_maize_shock':       ['price_maize'],
    'price_rice_shock':        ['price_rice'],
    'price_oil_shock':         ['price_oil'],
    'price_sorghum_shock':     ['price_sorghum'],
    'price_cassava_shock':     ['price_cassava'],
    'price_groundnuts_shock':  ['price_groundnuts'],
    'cw_score_local_spatial_lag':    ['cw_score_local'],
    'gdelt_event_count_spatial_lag': ['gdelt_event_count'],
    'acled_fatalities_spatial_lag':  ['fatalities_14d_sum']
}

BIDIRECTIONAL_BUNDLES = {
    'mech_gold_pivot':            ['mech_gold_pivot_uncertainty'],
    'mech_predatory_tax':         ['mech_predatory_tax_uncertainty'],
    'mech_factional_infighting':  ['mech_factional_infighting_uncertainty'],
    'mech_collective_punishment': ['mech_collective_punishment_uncertainty']
}

FAMILY_GUARDS = {
    'food_data_available':      [f'price_{c}' for c in ['maize', 'rice', 'oil', 'sorghum', 'cassava', 'groundnuts']] + ['food_price_index'],
    'iom_data_available':       ['iom_displacement_count_lag1'],
    'viirs_data_available':     ['ntl_mean', 'ntl_peak', 'ntl_kinetic_delta'],
    'gdelt_data_available':     ['gdelt_event_count', 'gdelt_predatory_action_decay_30d', 'gdelt_shock_signal'],
    'econ_data_available':      ['gold_price_usd_lag1', 'oil_price_usd_lag1', 'eur_usd_rate_lag1', 'sp500_index_lag1'],
    'landcover_data_available': ['landcover_grass', 'landcover_crops', 'landcover_trees', 'landcover_bare', 'landcover_built']
}


# =============================================================================
# STABILITY ENGINE (Using Permutation Importance - Avoids SHAP Bug)
# =============================================================================

def get_time_blocked_folds(df: pd.DataFrame, n_splits: int = 5):
    """Generates indices for strictly time-ordered blocking."""
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    if n_dates < n_splits + 1:
        raise ValueError(f"Insufficient time steps ({n_dates}) for {n_splits} splits.")
    
    fold_size = n_dates // (n_splits + 1)
    
    for i in range(n_splits):
        split_idx = (i + 1) * fold_size
        train_dates = dates[:split_idx]
        val_start_idx = split_idx
        val_end_idx = split_idx + fold_size
        val_dates = dates[val_start_idx:val_end_idx]
        
        train_mask = df['date'].isin(train_dates)
        val_mask = df['date'].isin(val_dates)
        
        yield train_mask, val_mask


def run_importance_stability_selection(
    df: pd.DataFrame,
    candidates: List[str],
    target_series: pd.Series,
    n_splits: int = 5,
    threshold_ratio: float = 1.0,
    mode: str = "binary",
    use_permutation: bool = True,
    n_repeats: int = 5,
) -> Dict[str, float]:
    """
    Computes stability using feature importance with shadow feature comparison.
    
    Uses permutation importance (more accurate) or native XGBoost importance (faster).
    Shadow feature provides noise baseline - features must beat it to be considered.
    
    Args:
        df: DataFrame with features and date column
        candidates: List of candidate feature names
        target_series: Target variable
        n_splits: Number of CV folds
        threshold_ratio: Multiplier for shadow threshold (1.0 = strict)
        mode: "binary" or "count"
        use_permutation: Use permutation importance (slower but more accurate)
        n_repeats: Repeats for permutation importance
        
    Returns:
        Dictionary of {feature: stability_score} where score is [0.0, 1.0]
    """
    df_proc = df.copy()
    
    # Filter to valid numeric candidates
    valid_candidates = []
    for c in candidates:
        if c not in df_proc.columns:
            continue
        if not np.issubdtype(df_proc[c].dtype, np.number):
            continue
        if df_proc[c].nunique() <= 1:
            continue
        valid_candidates.append(c)
    
    if not valid_candidates:
        logger.warning("No valid numeric candidates after filtering.")
        return {f: 0.0 for f in candidates}
    
    # Inject Shadow Feature (noise baseline)
    df_proc['shadow_random'] = np.random.uniform(0, 1, size=len(df_proc))
    df_proc['_target'] = target_series.values
    
    all_feats = valid_candidates + ['shadow_random']
    selection_counts = {f: 0 for f in valid_candidates}
    valid_folds = 0
    
    try:
        folds = list(get_time_blocked_folds(df_proc, n_splits))
    except ValueError as e:
        logger.error(f"Time blocking failed: {e}")
        return {f: 0.0 for f in candidates}
    
    for fold_idx, (train_mask, val_mask) in enumerate(folds):
        # Prepare data as numpy arrays
        X_train = df_proc.loc[train_mask, all_feats].values.astype(np.float64)
        y_train = df_proc.loc[train_mask, '_target'].values.astype(np.float64)
        X_val = df_proc.loc[val_mask, all_feats].values.astype(np.float64)
        y_val = df_proc.loc[val_mask, '_target'].values.astype(np.float64)
        
        # Fill any NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0)
        
        # Class scarcity check
        if mode == "binary":
            train_pos = y_train.sum()
            val_pos = y_val.sum()
            if train_pos < 20 or val_pos < 5:
                logger.debug(f"Fold {fold_idx} skipped: class scarcity (train_pos={train_pos}, val_pos={val_pos})")
                continue
        
        valid_folds += 1
        
        # Configure model
        if mode == "count":
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                objective="reg:squarederror",
                random_state=42 + fold_idx
            )
            scoring = 'neg_mean_squared_error'
        else:
            pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
            logger.debug(f"Fold {fold_idx}: pos_weight={pos_weight:.2f}")
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                scale_pos_weight=pos_weight,
                eval_metric='logloss',
                early_stopping_rounds=10,
                random_state=42 + fold_idx
            )
            scoring = 'roc_auc'
        
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        except Exception as e:
            logger.warning(f"Fold {fold_idx} training failed: {e}")
            continue
        
        # Get feature importance
        try:
            if use_permutation:
                # Permutation importance on validation set (more reliable)
                perm_result = permutation_importance(
                    model, X_val, y_val,
                    n_repeats=n_repeats,
                    random_state=42 + fold_idx,
                    scoring=scoring,
                    n_jobs=-1
                )
                importance_values = perm_result.importances_mean
            else:
                # Native XGBoost importance (faster but less reliable)
                importance_values = model.feature_importances_
            
            # Map to feature names
            importance_dict = dict(zip(all_feats, importance_values))
            
            # Get shadow feature importance as threshold
            noise_baseline = importance_dict.get('shadow_random', 0)
            threshold = max(noise_baseline * threshold_ratio, 1e-10)  # Minimum threshold
            
            # Count features that beat the shadow
            for f in valid_candidates:
                if f in importance_dict and importance_dict[f] > threshold:
                    selection_counts[f] += 1
                    
        except Exception as e:
            logger.warning(f"Fold {fold_idx} importance calculation failed: {e}")
            logger.debug(traceback.format_exc())
            continue

    if valid_folds == 0:
        logger.warning("No valid folds completed for stability selection.")
        return {f: 0.0 for f in candidates}
    
    # Build final scores
    result = {}
    for f in candidates:
        if f in selection_counts:
            result[f] = selection_counts[f] / valid_folds
        else:
            result[f] = 0.0
            
    return result


# =============================================================================
# BUNDLING LOGIC
# =============================================================================

def apply_bundling_logic(stable_set: Set[str], available_cols: List[str]) -> List[str]:
    """Applies Rescue, Bidirectional, and Family logic to the stable feature set."""
    final_features = set(stable_set)
    available_set = set(available_cols)
    
    # Bidirectional Bundles
    for key, siblings in BIDIRECTIONAL_BUNDLES.items():
        family = set([key] + siblings)
        if family.intersection(final_features):
            final_features.update([s for s in family if s in available_set])

    # Rescue Bundles (iterate for chains)
    changing = True
    while changing:
        start_len = len(final_features)
        for feat in list(final_features):
            if feat in RESCUE_BUNDLES:
                children = RESCUE_BUNDLES[feat]
                final_features.update([c for c in children if c in available_set])
        changing = len(final_features) != start_len

    # Family Guards
    for flag, family in FAMILY_GUARDS.items():
        if flag in available_set and not final_features.isdisjoint(family):
            final_features.add(flag)

    return sorted(list(final_features))


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_from_config(engine, configs: Dict[str, Any]):
    """Master entry point."""
    pruning_cfg = configs.get("models", {}).get("pruning", {})
    if not pruning_cfg.get("enabled", False):
        logger.info("Pruning disabled in config. Skipping.")
        return

    start_date = pruning_cfg.get("training_window", {}).get("start_date", "2000-01-01")
    end_date = pruning_cfg.get("training_window", {}).get("end_date", "2020-12-31")
    objectives = pruning_cfg.get("objectives", [])
    output_path = pruning_cfg.get("output_path", "configs/pruned_features.yaml")

    if not objectives:
        logger.warning("No pruning objectives defined. Skipping.")
        return

    logger.info("=" * 60)
    logger.info("STABILITY-BASED FEATURE PRUNING")
    logger.info("=" * 60)
    logger.info(f"Window: {start_date} to {end_date}")
    logger.info(f"Objectives: {len(objectives)}")
    logger.info("Using: Permutation Importance (SHAP disabled due to version bug)")

    needed_targets = list(set(obj['target'] for obj in objectives))
    
    logger.info("Loading training data panel...")
    try:
        df_panel = load_data_utils.load_training_data(
            engine,
            start_date=start_date,
            end_date=end_date,
            targets=needed_targets
        )
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        traceback.print_exc()
        return

    if df_panel.empty:
        logger.error("Training data panel is empty. Cannot prune.")
        return

    df_panel.sort_values(by='date', inplace=True)
    
    logger.info(f"Panel loaded: {len(df_panel):,} rows, {len(df_panel.columns)} columns")
    logger.info(f"Date range: {df_panel['date'].min().date()} to {df_panel['date'].max().date()}")
    
    # Multi-objective stability loop
    union_stable_features = set()
    
    valid_static = [c for c in STATIC_CONTEXT if c in df_panel.columns]
    union_stable_features.update(valid_static)
    logger.info(f"Static context features (protected): {len(valid_static)}")

    for obj_idx, obj in enumerate(objectives):
        target_col = obj['target']
        mode = obj.get('type', 'binary')
        threshold = obj.get('threshold', 0.6)
        
        logger.info("-" * 40)
        logger.info(f"Objective {obj_idx + 1}: {target_col} ({mode}) >= {threshold}")

        if target_col not in df_panel.columns:
            logger.warning(f"Target {target_col} not found. Skipping.")
            continue

        y_series = df_panel[target_col].fillna(0)
        if mode == 'binary':
            y_series = (y_series > 0).astype(int)
        
        if mode == 'binary':
            pos_rate = y_series.mean() * 100
            logger.info(f"  Target distribution: {pos_rate:.2f}% positive ({int(y_series.sum())} events)")
        else:
            logger.info(f"  Target stats: mean={y_series.mean():.4f}, std={y_series.std():.4f}")
        
        candidates = [
            c for c in df_panel.columns 
            if c not in ['h3_index', 'date'] + needed_targets + STATIC_CONTEXT
            and not c.startswith('target_')  # Prevent leakage from other target variants
        ]

        dtype_report = validate_and_report_dtypes(df_panel, candidates, target_col)
        if dtype_report['constant_candidates']:
            logger.info(f"  Constant candidates (skipped): {len(dtype_report['constant_candidates'])}")

        # Sanitize
        df_panel, dropped_cols, _ = sanitize_and_filter_columns(
            df_panel, candidates, [target_col], coercion_threshold=0.01
        )
        
        if dropped_cols:
            candidates = [c for c in candidates if c not in dropped_cols]
            logger.warning(f"  Dropped {len(dropped_cols)} columns")

        candidates = [c for c in candidates if c in df_panel.columns and np.issubdtype(df_panel[c].dtype, np.number)]
        logger.info(f"  Final candidate count: {len(candidates)}")

        # Run stability engine
        scores = run_importance_stability_selection(
            df_panel, 
            candidates=candidates, 
            target_series=y_series, 
            n_splits=5,
            threshold_ratio=1.0,
            mode=mode,
            use_permutation=True,
            n_repeats=5
        )
        
        passed = {f for f, s in scores.items() if s >= threshold}
        logger.info(f"  Features passing threshold: {len(passed)}")
        
        top_features = sorted(scores.items(), key=lambda x: -x[1])[:10]
        logger.info(f"  Top 10 by stability: {[(f, f'{s:.2f}') for f, s in top_features]}")
        
        union_stable_features.update(passed)

    # Apply bundling
    logger.info("-" * 40)
    logger.info("Applying bundling logic...")
    
    pre_bundle = len(union_stable_features)
    final_features = apply_bundling_logic(union_stable_features, df_panel.columns.tolist())
    post_bundle = len(final_features)
    
    logger.info(f"Features before bundling: {pre_bundle}")
    logger.info(f"Features after bundling: {post_bundle} (rescued {post_bundle - pre_bundle})")
    
    # Save
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    registry_data = {
        "generated_at": str(pd.Timestamp.now()),
        "window": f"{start_date}_{end_date}",
        "feature_count": len(final_features),
        "objectives": [obj['target'] for obj in objectives],
        "active_features": final_features
    }
    
    with open(out_file, "w") as f:
        yaml.dump(registry_data, f, sort_keys=False)
    
    logger.info("=" * 60)
    logger.info(f"✅ PRUNING COMPLETE")
    logger.info(f"   Saved {len(final_features)} features to {output_path}")
    logger.info("=" * 60)


def main():
    """Run pruning."""
    try:
        from utils import load_configs, get_db_engine
    except Exception as e:
        logger.critical(f"Failed to import utils: {e}")
        return 1

    configs = load_configs()
    
    try:
        engine = get_db_engine()
    except Exception as e:
        logger.warning(f"Could not init DB engine ({e}); proceeding without engine.")
        engine = None

    if load_data_utils is None:
        logger.error("load_data_utils unavailable.")
        return 1

    run_from_config(engine, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
