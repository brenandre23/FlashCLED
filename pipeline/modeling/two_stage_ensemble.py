"""
two_stage_ensemble.py
=====================
Two-stage stacked ensemble for conflict prediction with calibrated probabilities
(sigmoid default, isotonic optional).

Thesis Methodology Enforced:
1. Stage 2 Regressor is strictly PoissonRegressor (Count Model).
2. Non-negativity constraints AND upper clipping applied to all inputs.
3. Validation uses time-series splits to prevent leakage.
4. Calibration layer applied post-hoc for probability reliability.
5. PREDICT returns unconditional expectation (Risk = Prob * Severity).

FIXES (2026-01-25):
- Added defensive validation for empty feature sets in theme models
- Graceful handling of missing columns in X[feature_cols]
- Clear error messages for debugging feature availability issues
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss

try:
    from pipeline.modeling.temporal_splits import date_block_folds, temporal_tail_split
except ImportError:  # executed as a script from pipeline/modeling
    from temporal_splits import date_block_folds, temporal_tail_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageEnsemble:
    """
    Two-stage stacked ensemble for conflict prediction.
    
    Architecture:
    - Level 1: Theme-specific base learners (classifiers + regressors)
    - Level 2: Meta-learners (XGBoost for binary onset, PoissonRegressor for fatality intensity)
    - Level 3: Calibration layer (optional, defaults to none in v2.0+)
    
    CRITICAL FEATURES:
    - XGBoost Meta-Learner: Captures non-linear interactions between thematic themes.
    - Intensity Clipping: Base regressor outputs are capped at 500 (99.9th percentile) 
      to prevent numerical explosion in the Poisson log-link function.
    - Unconditional Prediction: predict() returns P(Conflict) * E[Fatalities | Conflict].
    """

    def __init__(
        self,
        theme_models: List[Dict],
        n_folds: int = 5,
        random_state: int = 42,
        calibration_config: Optional[Dict[str, Any]] = None,
        meta_config: Optional[Dict[str, Any]] = None,
        # Safety cap for base regressor outputs before they hit the Poisson meta-learner
        regressor_input_cap: float = 500.0,
        embargo_steps: int = 0,
        step_days: int = 14,
        date_col: str = "date",
    ) -> None:
        self.theme_models = theme_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.regressor_input_cap = regressor_input_cap
        # Leak-safe splitting (see temporal_splits.py): the forecast horizon in
        # steps is the embargo between any training block and what it is scored on.
        self.embargo_steps = int(embargo_steps)
        self.step_days = int(step_days)
        self.date_col = date_col
        # Meta-learner input scales (see _binary_meta_inputs / _regress_meta_inputs).
        # Stored on the instance so models pickled before this change still predict
        # with the scale they were fitted on (attribute absent -> raw inputs).
        self.meta_logit_inputs = True
        self.regress_meta_log_inputs = True
        # Out-of-fold predictions of the stacked model, exposed for conformal
        # calibration when no separate calibration set is held out. Cleared by
        # the caller before pickling.
        self.oof_idx_: Optional[np.ndarray] = None
        self.oof_prob_: Optional[np.ndarray] = None
        self.oof_expected_fatalities_: Optional[np.ndarray] = None
        self.meta_config = meta_config or {"type": "xgboost", "params": {"n_estimators": 50, "max_depth": 3, "scale_pos_weight": 8}}
        
        # Parse calibration config (SINGLE SOURCE OF TRUTH)
        cal_cfg = calibration_config or {}
        self.calibration_method = cal_cfg.get("method", "sigmoid")
        self.calibration_split_method = cal_cfg.get("split_method", "temporal_tail")
        self.calibration_fraction = cal_cfg.get("fraction", 0.2)
        self.min_positive_events = cal_cfg.get("min_positive_events", 10)
        self.fallback_fraction = cal_cfg.get("fallback_fraction", 0.3)
        self.calibration_random_seed = cal_cfg.get("random_seed", 42)

        self.meta_binary: Optional[Any] = None
        self.meta_regress: Optional[PoissonRegressor] = None
        self.calibrator: Optional[Any] = None
        self.is_fitted: bool = False
        
        # EXPOSED INDICES: Single source of truth for train/calibration splits
        self.train_idx_: Optional[np.ndarray] = None
        self.cal_idx_: Optional[np.ndarray] = None
        
        self.calibration_diagnostics: Dict[str, Any] = {}
        
        # Track validated theme models (set during fit)
        self._validated_theme_models: Optional[List[Dict]] = None

    def _validate_theme_models(self, X: pd.DataFrame) -> List[Dict]:
        """
        Validate that all theme models have features available in X.
        
        Returns:
            List of validated theme models (only those with available features)
        
        Raises:
            ValueError: If no theme models have valid features
        """
        available_cols = set(X.columns)
        validated_models = []
        skipped_models = []
        
        for theme in self.theme_models:
            name = theme.get("name", "unnamed")
            feature_cols = theme.get("features", [])
            
            if not feature_cols:
                logger.warning(f"Theme '{name}' has no features configured. Skipping.")
                skipped_models.append((name, "no features configured"))
                continue
            
            # Check which features exist
            missing_cols = [f for f in feature_cols if f not in available_cols]
            valid_cols = [f for f in feature_cols if f in available_cols]
            
            if missing_cols:
                logger.warning(
                    f"Theme '{name}': {len(missing_cols)}/{len(feature_cols)} features missing: "
                    f"{missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}"
                )
            
            if not valid_cols:
                logger.warning(f"Theme '{name}' has NO valid features. Skipping entirely.")
                skipped_models.append((name, f"{len(missing_cols)} features all missing"))
                continue
            
            # Create a copy with only valid features
            validated_theme = theme.copy()
            validated_theme["features"] = valid_cols
            validated_theme["_original_features"] = feature_cols
            validated_theme["_missing_features"] = missing_cols
            validated_models.append(validated_theme)
            
            logger.info(f"Theme '{name}': {len(valid_cols)}/{len(feature_cols)} features valid")
        
        if not validated_models:
            raise ValueError(
                f"No theme models have valid features in the data!\n"
                f"Available columns: {len(available_cols)}\n"
                f"Skipped models: {skipped_models}\n\n"
                "Check that feature_matrix contains the expected columns."
            )
        
        logger.info(f"Validated {len(validated_models)}/{len(self.theme_models)} theme models")
        return validated_models

    def _safe_get_features(self, X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Safely extract features from DataFrame, handling missing columns gracefully.
        
        Args:
            X: Input DataFrame
            feature_cols: List of column names to extract
            
        Returns:
            DataFrame with available columns only
            
        Raises:
            ValueError: If resulting DataFrame would be empty
        """
        available = [c for c in feature_cols if c in X.columns]
        
        if not available:
            raise ValueError(
                f"None of the requested features exist in DataFrame!\n"
                f"Requested: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}\n"
                f"Available: {list(X.columns)[:10]}{'...' if len(X.columns) > 10 else ''}"
            )
        
        return X[available]

    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate out-of-fold predictions for stacking."""
        n_samples = len(X)
        n_themes = len(self._validated_theme_models)

        if n_themes == 0:
            raise ValueError("No validated theme models available for OOF prediction!")

        oof_binary = np.zeros((n_samples, n_themes), dtype=float)
        oof_regress = np.zeros((n_samples, n_themes), dtype=float)
        is_valid_oof = np.zeros(n_samples, dtype=bool)

        folds = self._make_folds(X)

        for theme_idx, theme in enumerate(self._validated_theme_models):
            feature_cols = theme["features"]
            name = theme.get("name", f"theme_{theme_idx}")
            
            # Defensive check: ensure we have features
            if not feature_cols:
                logger.error(f"Theme '{name}' has empty feature list during OOF! Skipping.")
                continue
                
            X_theme = self._safe_get_features(X, feature_cols)
            
            # Double-check we got columns
            if X_theme.shape[1] == 0:
                logger.error(f"Theme '{name}' resulted in 0-column DataFrame! Skipping.")
                continue

            base_clf = theme["binary_model"]
            base_reg = theme["regress_model"]

            for train_idx, val_idx in folds:
                X_train, X_val = X_theme.iloc[train_idx], X_theme.iloc[val_idx]
                y_train_bin = y_binary.iloc[train_idx]
                y_train_fat = y_fatalities.iloc[train_idx]

                # 1. Fit & Predict Classifier
                clf_fold = clone(base_clf)
                clf_fold.fit(X_train, y_train_bin)
                
                try:
                    preds = clf_fold.predict_proba(X_val)[:, 1]
                except AttributeError:
                    preds = clf_fold.predict(X_val)
                oof_binary[val_idx, theme_idx] = preds

                # 2. Fit & Predict Regressor (on conflict cases only)
                reg_fold = clone(base_reg)
                conflict_mask = (y_train_bin == 1)
                
                if conflict_mask.sum() > 0:
                    reg_fold.fit(X_train[conflict_mask], y_train_fat[conflict_mask])
                    oof_regress[val_idx, theme_idx] = reg_fold.predict(X_val)
                
                # Mark these indices as valid for meta-learner training
                is_valid_oof[val_idx] = True

        return oof_binary, oof_regress, is_valid_oof

    def _binary_meta_inputs(self, probs: np.ndarray) -> np.ndarray:
        """
        Base-learner probabilities -> meta-learner inputs, on the logit scale.

        A logistic meta-learner is linear in its inputs on the log-odds scale, so
        feeding it probabilities mis-specifies it: the base learners are trained
        with scale_pos_weight and squeeze almost all rows near 0, where equal
        probability differences mean very different evidence. On logits the
        stack is a weighted sum of log-odds plus an intercept, i.e. a joint
        Platt-style recalibration to the true base rate. Tree meta-learners are
        unaffected (monotone transform).
        """
        if not getattr(self, "meta_logit_inputs", False):
            return probs
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _regress_meta_inputs(self, preds: np.ndarray) -> np.ndarray:
        """
        Base-learner fatality predictions -> Poisson meta-learner inputs.

        The Poisson GLM has a log link: mu = exp(b0 + sum b_k x_k). With raw
        counts as x_k the prediction grows exponentially in the base prediction,
        which is what produced the overflows the 500 cap was added to contain.
        With x_k = log1p(prediction) the stack is a weighted geometric mean of
        the base predictions (b_k = 1 reproduces a base model exactly), the
        natural scale for a log-link model. The cap stays as a safety bound.
        """
        a = np.minimum(np.maximum(preds, 0), self.regressor_input_cap)
        return np.log1p(a) if getattr(self, "regress_meta_log_inputs", False) else a

    def _make_folds(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Stacking folds. With a date column: expanding window over whole dates
        with an embargo of `embargo_steps` (no look-ahead, no date shared
        between train and validation). Without one: positional TimeSeriesSplit,
        which is only temporal if the caller sorted the rows by date.
        """
        if self.date_col in X.columns:
            folds = date_block_folds(X[self.date_col], self.n_folds, self.embargo_steps, self.step_days)
            logger.info(
                f"Stacking folds: {len(folds)} date blocks, embargo={self.embargo_steps} step(s) "
                f"x {self.step_days} days"
            )
            return folds
        logger.warning(
            f"No '{self.date_col}' column passed to TwoStageEnsemble.fit: falling back to positional "
            "TimeSeriesSplit. This is leak-safe ONLY if rows are sorted by date."
        )
        return list(TimeSeriesSplit(n_splits=self.n_folds).split(X))

    def _split_once(self, n: int, fraction: float, dates: Optional[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """One train/calibration split for the configured method."""
        all_indices = np.arange(n)
        if self.calibration_split_method == "temporal_tail":
            if dates is not None:
                # Whole dates only, with an embargo of one forecast horizon
                # between the last training date and the calibration tail.
                return temporal_tail_split(dates, fraction, self.embargo_steps, self.step_days)
            logger.warning("temporal_tail calibration split without dates: positional split, no embargo.")
            split_idx = int(n * (1 - fraction))
            return all_indices[:split_idx], all_indices[split_idx:]
        if self.calibration_split_method == "random":
            logger.warning(
                "calibration split_method='random' mixes periods between training and calibration; "
                "it is not leak-safe for forecasting. Prefer 'temporal_tail'."
            )
            rng = np.random.default_rng(self.calibration_random_seed)
            shuffled = rng.permutation(all_indices)
            split_idx = int(n * (1 - fraction))
            return np.sort(shuffled[:split_idx]), np.sort(shuffled[split_idx:])
        if self.calibration_split_method == "rolling":
            raise NotImplementedError("Rolling calibration not yet implemented")
        raise ValueError(f"Unknown split_method: {self.calibration_split_method}")

    def _compute_calibration_indices(
        self,
        y_binary: pd.Series,
        dates: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute train/calibration indices based on calibration policy from config.

        SINGLE SOURCE OF TRUTH: This is the ONLY place calibration splits are computed.
        All downstream code must use self.train_idx_ and self.cal_idx_.

        temporal_tail holds out the last `fraction` of distinct dates; training
        rows stop one forecast horizon earlier (rows in the gap are unused).
        """
        n = len(y_binary)
        train_idx, cal_idx = self._split_once(n, self.calibration_fraction, dates)

        n_pos_cal = y_binary.iloc[cal_idx].sum()
        if n_pos_cal < self.min_positive_events:
            logger.warning(
                f"Only {n_pos_cal} positive cases in calibration set "
                f"(min={self.min_positive_events}). Applying fallback_fraction={self.fallback_fraction}"
            )
            train_idx, cal_idx = self._split_once(n, self.fallback_fraction, dates)
            n_pos_cal = y_binary.iloc[cal_idx].sum()
            logger.info(f"After fallback: {n_pos_cal} positive cases in calibration set")

        logger.info(
            f"Calibration split ({self.calibration_split_method}): "
            f"train={len(train_idx):,}, cal={len(cal_idx):,} "
            f"(cal positives: {n_pos_cal:,})"
        )
        return train_idx, cal_idx

    def _split_for_calibration(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split data for calibration using pre-computed indices.
        
        IMPORTANT: Indices are computed once in _compute_calibration_indices() and
        stored in self.train_idx_ and self.cal_idx_. This method uses those indices.
        """
        # Compute and persist indices (SINGLE SOURCE OF TRUTH)
        dates = X[self.date_col] if self.date_col in X.columns else None
        self.train_idx_, self.cal_idx_ = self._compute_calibration_indices(y_binary, dates)
        
        X_train = X.iloc[self.train_idx_]
        X_cal = X.iloc[self.cal_idx_]
        y_bin_train = y_binary.iloc[self.train_idx_]
        y_bin_cal = y_binary.iloc[self.cal_idx_]
        y_fat_train = y_fatalities.iloc[self.train_idx_]
        y_fat_cal = y_fatalities.iloc[self.cal_idx_]
        
        return X_train, X_cal, y_bin_train, y_bin_cal, y_fat_train, y_fat_cal

    def _fit_calibrator(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        """Fit calibration on held-out calibration set."""
        if self.calibration_method == "none":
            self.calibrator = None
            return
            
        n_themes = len(self._validated_theme_models)
        n_cal = len(X_cal)
        
        l1_binary = np.zeros((n_cal, n_themes))
        
        for i, theme in enumerate(self._validated_theme_models):
            X_theme = self._safe_get_features(X_cal, theme["features"])
            try:
                l1_binary[:, i] = theme["binary_model"].predict_proba(X_theme)[:, 1]
            except AttributeError:
                l1_binary[:, i] = theme["binary_model"].predict(X_theme)
        
        # Get raw meta-learner probabilities
        raw_probs = self.meta_binary.predict_proba(self._binary_meta_inputs(l1_binary))[:, 1]
        
        if self.calibration_method == "sigmoid":
            from sklearn.linear_model import LogisticRegression as LR_Calib
            self.calibrator = LR_Calib(solver="lbfgs")
            self.calibrator.fit(raw_probs.reshape(-1, 1), y_cal.values)
            cal_probs = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        elif self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
            self.calibrator.fit(raw_probs, y_cal.values)
            cal_probs = self.calibrator.predict(raw_probs)
        else:
            raise ValueError(f"Unsupported calibration method: {self.calibration_method}")
        
        self.calibration_diagnostics = {
            "n_calibration_samples": n_cal,
            "raw_brier": float(brier_score_loss(y_cal, raw_probs)),
            "calibrated_brier": float(brier_score_loss(y_cal, cal_probs)),
            "calibrated_std": float(np.std(cal_probs))
        }
        
        logger.info(f"Brier improvement: {self.calibration_diagnostics['raw_brier']:.4f} -> "
                    f"{self.calibration_diagnostics['calibrated_brier']:.4f}")
        
        if self.calibration_diagnostics["calibrated_std"] < 1e-5:
            logger.warning("🚨 CALIBRATOR COLLAPSE DETECTED: Output variance is near zero. "
                           "Consider increasing calibration_fraction or using isotonic method.")

    def fit(self, X: pd.DataFrame, y_binary: pd.Series, y_fatalities: pd.Series) -> None:
        """Fit the complete two-stage ensemble with calibration."""
        logger.info("Fitting TwoStageEnsemble (Methodology: Poisson + Hurdle + Sigmoid Calibration)...")

        y_binary = pd.Series(y_binary).astype(int).reset_index(drop=True)
        y_fatalities = pd.Series(y_fatalities).astype(float).reset_index(drop=True)
        X = X.reset_index(drop=True)

        # CRITICAL: Validate theme models against available features
        self._validated_theme_models = self._validate_theme_models(X)
        
        if not self._validated_theme_models:
            raise ValueError("No theme models have valid features. Cannot fit ensemble.")

        # 1. Split for calibration (temporal)
        if self.calibration_method != "none" and self.calibration_fraction > 0:
            X_train, X_cal, y_bin_train, y_bin_cal, y_fat_train, y_fat_cal = \
                self._split_for_calibration(X, y_binary, y_fatalities)
        else:
            # Fallback: If no calibration split, indices point to full dataset
            self.train_idx_ = np.arange(len(X))
            self.cal_idx_ = self.train_idx_
            X_train, X_cal = X, X
            y_bin_train, y_bin_cal = y_binary, y_binary
            y_fat_train, y_fat_cal = y_fatalities, y_fatalities

        # 2. Generate OOF Predictions (Level 1) on training data
        # Positional alignment: after a calibration split the pandas index is no
        # longer 0..n-1, while every OOF array below is positional.
        X_train = X_train.reset_index(drop=True)
        y_bin_train = y_bin_train.reset_index(drop=True)
        y_fat_train = y_fat_train.reset_index(drop=True)

        oof_binary, oof_regress, is_valid_oof = self._generate_oof_predictions(X_train, y_bin_train, y_fat_train)

        # 3. Train Meta-Classifier (using only rows with valid OOF predictions)
        X_meta_bin = self._binary_meta_inputs(oof_binary[is_valid_oof])
        y_meta_bin = y_bin_train[is_valid_oof]
        
        meta_type = self.meta_config.get("type", "logistic")
        meta_params = self.meta_config.get("params", {})
        
        if meta_type == "xgboost":
            from xgboost import XGBClassifier
            # Ensure basic defaults if not provided
            params = {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "random_state": self.random_state,
                "tree_method": "hist"
            }
            params.update(meta_params)
            self.meta_binary = XGBClassifier(**params)
        else:
            # Logistic Default
            params = {
                "solver": "lbfgs",
                "max_iter": 1000,
                "n_jobs": -1,
                "random_state": self.random_state,
                "class_weight": "balanced"
            }
            params.update(meta_params)
            self.meta_binary = LogisticRegression(**params)
            
        self.meta_binary.fit(X_meta_bin, y_meta_bin)

        # 4. Train Meta-Regressor (Poisson) with Safety Clipping
        # Also using only rows with valid OOF predictions AND positive onset
        conflict_mask = (y_bin_train == 1) & is_valid_oof
        
        if conflict_mask.sum() > 0:
            self.meta_regress = PoissonRegressor(alpha=1.0, max_iter=1000)
            y_fat_conflict = np.maximum(y_fat_train[conflict_mask], 0)
            
            # CLIP INPUTS: prevent base model outliers from exploding the Poisson exponent
            X_regress = self._regress_meta_inputs(oof_regress[conflict_mask.values, :])

            self.meta_regress.fit(X_regress, y_fat_conflict)
        else:
            self.meta_regress = None

        # 5. Final Retrain of Base Models on Full Training Data
        # Out-of-fold predictions of the full stack. Base-learner inputs are
        # strictly out-of-fold (date blocks + embargo); only the low-capacity
        # meta-learners have seen these rows. Used for conformal calibration
        # when no separate calibration set exists.
        oof_prob = self.meta_binary.predict_proba(X_meta_bin)[:, 1]
        if self.meta_regress is not None:
            oof_inputs = self._regress_meta_inputs(oof_regress[is_valid_oof])
            oof_mu = np.clip(self.meta_regress.predict(oof_inputs), 0.0, None)
        else:
            oof_mu = np.zeros_like(oof_prob)
        self.oof_idx_ = np.asarray(self.train_idx_)[np.flatnonzero(is_valid_oof)]
        self.oof_prob_ = oof_prob
        self.oof_expected_fatalities_ = oof_prob * oof_mu

        logger.info("Retraining base models on full training dataset...")
        for theme in self._validated_theme_models:
            X_theme = self._safe_get_features(X_train, theme["features"])
            theme["binary_model"].fit(X_theme, y_bin_train)
            if conflict_mask.sum() > 0:
                theme["regress_model"].fit(X_theme[conflict_mask], y_fat_train[conflict_mask])

        # 6. Fit Calibrator on held-out calibration set
        if X_cal is not None and self.calibration_method != "none":
            self._fit_calibrator(X_cal, y_bin_cal)
        else:
            self.calibrator = None
            
        self.is_fitted = True
        logger.info("TwoStageEnsemble fitting complete.")

    def predict(self, X: pd.DataFrame, return_components: bool = False) -> Any:
        """
        Generate predictions.
        
        Returns:
            prob: Calibrated probability of conflict
            expected_fatalities: Unconditional expectation (prob * conditional_mu)
            
        If return_components=True:
            (prob, expected_fatalities, conditional_mu)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if self._validated_theme_models is None:
            raise RuntimeError("Model fitted but no validated theme models found (internal error)")

        # Ensure X is a DataFrame for consistent feature name handling
        if not isinstance(X, pd.DataFrame):
            # This is a fallback; ideally X is always a DataFrame from upstream
            logger.warning("predict() received non-DataFrame input. Attempting to convert, but feature alignment risk is high.")
            X = pd.DataFrame(X)

        n_samples = len(X)
        n_themes = len(self._validated_theme_models)
        
        l1_binary = np.zeros((n_samples, n_themes))
        l1_regress = np.zeros((n_samples, n_themes))

        # Level 1 Predictions
        for i, theme in enumerate(self._validated_theme_models):
            feature_cols = theme["features"]
            X_theme = self._safe_get_features(X, feature_cols)
            
            # Ensure we pass a DataFrame with names to base models
            try:
                l1_binary[:, i] = theme["binary_model"].predict_proba(X_theme)[:, 1]
            except AttributeError:
                l1_binary[:, i] = theme["binary_model"].predict(X_theme)
            l1_regress[:, i] = theme["regress_model"].predict(X_theme)

        # Level 2: Meta-learner probabilities
        raw_prob = self.meta_binary.predict_proba(self._binary_meta_inputs(l1_binary))[:, 1]

        # Level 3: Calibration
        if self.calibrator is not None:
            if self.calibration_method == "sigmoid":
                prob = self.calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
            else:
                prob = self.calibrator.predict(raw_prob)
        else:
            prob = raw_prob

        # Fatality predictions (Conditional)
        if self.meta_regress:
            # SAFETY CLIP on inputs before feeding to Poisson meta-learner
            regress_inputs = self._regress_meta_inputs(l1_regress)
            
            conditional_mu = self.meta_regress.predict(regress_inputs)
            conditional_mu = np.clip(conditional_mu, 0.0, None)
        else:
            conditional_mu = np.zeros_like(prob)

        # CALCULATION: Unconditional Expected Fatalities
        # E[Y] = P(Conflict) * E[Y|Conflict]
        expected_fatalities = prob * conditional_mu

        if return_components:
            return prob, expected_fatalities, conditional_mu
        
        return prob, expected_fatalities
    
    def get_feature_diagnostics(self) -> Dict[str, Any]:
        """
        Return diagnostics about feature availability after fitting.
        
        Returns:
            Dict with per-theme feature validation info
        """
        if self._validated_theme_models is None:
            return {"error": "Model not yet fitted"}
        
        diagnostics = {}
        for theme in self._validated_theme_models:
            name = theme.get("name", "unnamed")
            diagnostics[name] = {
                "valid_features": theme["features"],
                "original_features": theme.get("_original_features", theme["features"]),
                "missing_features": theme.get("_missing_features", []),
                "valid_count": len(theme["features"]),
                "missing_count": len(theme.get("_missing_features", [])),
            }
        return diagnostics
