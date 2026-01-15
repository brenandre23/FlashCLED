"""
two_stage_ensemble.py
=====================
Two-stage stacked ensemble for conflict prediction with isotonic calibration.

Thesis Methodology Enforced:
1. Stage 2 Regressor is strictly PoissonRegressor (Count Model).
2. Non-negativity constraints applied to all inputs and outputs.
3. Validation uses time-series splits to prevent leakage.
4. Isotonic calibration applied post-hoc for probability reliability.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageEnsemble:
    """
    Two-stage stacked ensemble for conflict prediction.
    
    Architecture:
    - Level 1: Theme-specific base learners (classifiers + regressors)
    - Level 2: Meta-learners (LogisticRegression for binary, PoissonRegressor for counts)
    - Level 3: Isotonic calibration for reliable probability estimates
    
    The isotonic calibration layer addresses the probability compression issue
    common with extreme class imbalance. It maps the meta-learner's raw 
    probabilities to empirically calibrated values using a monotonic function.
    """

    def __init__(
        self,
        theme_models: List[Dict],
        n_folds: int = 5,
        random_state: int = 42,
        calibration_method: str = "isotonic",  # "isotonic" | "none"
        calibration_fraction: float = 0.2,     # Fraction of training data for calibration
    ) -> None:
        self.theme_models = theme_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction

        self.meta_binary: Optional[LogisticRegression] = None
        self.meta_regress: Optional[PoissonRegressor] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.is_fitted: bool = False
        
        # Calibration diagnostics
        self.calibration_diagnostics: Dict[str, Any] = {}

    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate out-of-fold predictions for stacking."""
        n_samples = len(X)
        n_themes = len(self.theme_models)

        oof_binary = np.zeros((n_samples, n_themes), dtype=float)
        oof_regress = np.zeros((n_samples, n_themes), dtype=float)

        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        for theme_idx, theme in enumerate(self.theme_models):
            feature_cols = theme["features"]
            X_theme = X[feature_cols]

            base_clf = theme["binary_model"]
            base_reg = theme["regress_model"]

            for train_idx, val_idx in tscv.split(X_theme):
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

                # 2. Fit & Predict Regressor (on conflict cases)
                reg_fold = clone(base_reg)
                conflict_mask = (y_train_bin == 1)
                
                if conflict_mask.sum() > 0:
                    reg_fold.fit(X_train[conflict_mask], y_train_fat[conflict_mask])
                    oof_regress[val_idx, theme_idx] = reg_fold.predict(X_val)

        return oof_binary, oof_regress

    def _split_for_calibration(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Split data temporally for calibration.
        Uses the last `calibration_fraction` of data for calibration fitting.
        """
        n = len(X)
        split_idx = int(n * (1 - self.calibration_fraction))
        
        # Ensure at least some positive cases in calibration set
        y_cal_check = y_binary.iloc[split_idx:]
        n_pos_cal = y_cal_check.sum()
        
        if n_pos_cal < 10:
            # Fallback: use more data for calibration
            logger.warning(f"Only {n_pos_cal} positive cases in calibration set. Expanding.")
            split_idx = int(n * 0.7)  # Use 30% for calibration
        
        X_train = X.iloc[:split_idx]
        X_cal = X.iloc[split_idx:]
        y_bin_train = y_binary.iloc[:split_idx]
        y_bin_cal = y_binary.iloc[split_idx:]
        y_fat_train = y_fatalities.iloc[:split_idx]
        y_fat_cal = y_fatalities.iloc[split_idx:]
        
        logger.info(f"Calibration split: train={len(X_train):,}, cal={len(X_cal):,} "
                    f"(cal positives: {y_bin_cal.sum():,})")
        
        return X_train, X_cal, y_bin_train, y_bin_cal, y_fat_train, y_fat_cal

    def _fit_calibrator(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> None:
        """
        Fit isotonic calibration on held-out calibration set.
        
        Uses Level-1 predictions -> Meta-learner -> Raw probabilities,
        then fits IsotonicRegression to map raw probs to calibrated probs.
        """
        if self.calibration_method == "none":
            self.calibrator = None
            return
            
        # Generate Level-1 predictions for calibration set
        n_themes = len(self.theme_models)
        n_cal = len(X_cal)
        
        l1_binary = np.zeros((n_cal, n_themes))
        
        for i, theme in enumerate(self.theme_models):
            X_theme = X_cal[theme["features"]]
            try:
                l1_binary[:, i] = theme["binary_model"].predict_proba(X_theme)[:, 1]
            except AttributeError:
                l1_binary[:, i] = theme["binary_model"].predict(X_theme)
        
        # Get raw meta-learner probabilities
        raw_probs = self.meta_binary.predict_proba(l1_binary)[:, 1]
        
        # Fit isotonic calibration
        # y_min/y_max bounds ensure output stays in [0, 1]
        self.calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
            increasing=True,
        )
        self.calibrator.fit(raw_probs, y_cal.values)
        
        # Compute diagnostics
        cal_probs = self.calibrator.predict(raw_probs)
        
        self.calibration_diagnostics = {
            "n_calibration_samples": n_cal,
            "n_positive_samples": int(y_cal.sum()),
            "raw_prob_range": (float(raw_probs.min()), float(raw_probs.max())),
            "calibrated_prob_range": (float(cal_probs.min()), float(cal_probs.max())),
            "raw_brier": float(brier_score_loss(y_cal, raw_probs)),
            "calibrated_brier": float(brier_score_loss(y_cal, cal_probs)),
            "raw_logloss": float(log_loss(y_cal, raw_probs)),
            "calibrated_logloss": float(log_loss(y_cal, cal_probs)),
        }
        
        logger.info(f"Calibration fitted: "
                    f"raw_range={self.calibration_diagnostics['raw_prob_range']}, "
                    f"cal_range={self.calibration_diagnostics['calibrated_prob_range']}")
        logger.info(f"Brier improvement: "
                    f"{self.calibration_diagnostics['raw_brier']:.4f} -> "
                    f"{self.calibration_diagnostics['calibrated_brier']:.4f}")

    def fit(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> None:
        """
        Fit the complete two-stage ensemble with calibration.
        
        Steps:
        1. Split data: training portion vs calibration portion (temporal)
        2. Generate OOF predictions on training portion
        3. Fit meta-learners on training portion
        4. Retrain base models on full training portion
        5. Fit calibrator on calibration portion
        """
        logger.info("Fitting TwoStageEnsemble (Methodology: Poisson + Hurdle + Isotonic)...")

        y_binary = pd.Series(y_binary).astype(int).reset_index(drop=True)
        y_fatalities = pd.Series(y_fatalities).astype(float).reset_index(drop=True)
        X = X.reset_index(drop=True)

        # Split for calibration (temporal split)
        if self.calibration_method != "none" and self.calibration_fraction > 0:
            X_train, X_cal, y_bin_train, y_bin_cal, y_fat_train, y_fat_cal = \
                self._split_for_calibration(X, y_binary, y_fatalities)
        else:
            X_train, X_cal = X, None
            y_bin_train, y_bin_cal = y_binary, None
            y_fat_train, y_fat_cal = y_fatalities, None

        # 1. Generate OOF Predictions (Level 1) on training data
        oof_binary, oof_regress = self._generate_oof_predictions(
            X_train, y_bin_train, y_fat_train
        )

        # 2. Train Meta-Classifier (Logistic Regression)
        self.meta_binary = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.meta_binary.fit(oof_binary, y_bin_train)

        # 3. Train Meta-Regressor (Poisson Regression)
        conflict_mask = y_bin_train == 1
        if conflict_mask.sum() > 0:
            self.meta_regress = PoissonRegressor(alpha=1.0, max_iter=1000)
            y_fat_conflict = np.maximum(y_fat_train[conflict_mask], 0)
            X_regress = np.log1p(np.maximum(oof_regress[conflict_mask.values, :], 0))
            self.meta_regress.fit(X_regress, y_fat_conflict)
        else:
            self.meta_regress = None

        # 4. Final Retrain of Base Models on Full Training Data
        logger.info("Retraining base models on full training dataset...")
        for theme in self.theme_models:
            X_theme = X_train[theme["features"]]
            theme["binary_model"].fit(X_theme, y_bin_train)
            if conflict_mask.sum() > 0:
                theme["regress_model"].fit(
                    X_theme[conflict_mask],
                    y_fat_train[conflict_mask]
                )

        # 5. Fit Calibrator on held-out calibration set
        if X_cal is not None and self.calibration_method != "none":
            self._fit_calibrator(X_cal, y_bin_cal)
        else:
            self.calibrator = None
            
        self.is_fitted = True
        logger.info("TwoStageEnsemble fitting complete.")

    def predict(
        self,
        X: pd.DataFrame,
        return_uncalibrated: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            return_uncalibrated: If True, returns (calibrated, uncalibrated, fatalities)
                                 Otherwise returns (calibrated, fatalities)
        
        Returns:
            prob: Calibrated probability of conflict (or uncalibrated if no calibrator)
            mu_fatal: Expected fatalities conditional on conflict
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        n_samples = len(X)
        n_themes = len(self.theme_models)
        
        l1_binary = np.zeros((n_samples, n_themes))
        l1_regress = np.zeros((n_samples, n_themes))

        # Level 1 Predictions
        for i, theme in enumerate(self.theme_models):
            X_theme = X[theme["features"]]
            
            try:
                l1_binary[:, i] = theme["binary_model"].predict_proba(X_theme)[:, 1]
            except AttributeError:
                l1_binary[:, i] = theme["binary_model"].predict(X_theme)
                
            l1_regress[:, i] = theme["regress_model"].predict(X_theme)

        # Level 2: Meta-learner probabilities (raw/uncalibrated)
        raw_prob = self.meta_binary.predict_proba(l1_binary)[:, 1]

        # Level 3: Calibration
        if self.calibrator is not None:
            prob = self.calibrator.predict(raw_prob)
        else:
            prob = raw_prob

        # Fatality predictions
        if self.meta_regress:
            regress_inputs = np.log1p(np.maximum(l1_regress, 0))
            mu_fatal = self.meta_regress.predict(regress_inputs)
            mu_fatal = np.clip(mu_fatal, 0.0, None)
        else:
            mu_fatal = np.zeros_like(prob)

        if return_uncalibrated:
            return prob, raw_prob, mu_fatal
        
        return prob, mu_fatal

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw (uncalibrated) probabilities from meta-learner."""
        prob, raw_prob, _ = self.predict(X, return_uncalibrated=True)
        return raw_prob

    def get_calibration_diagnostics(self) -> Dict[str, Any]:
        """Return calibration diagnostics from fitting."""
        return self.calibration_diagnostics.copy()
