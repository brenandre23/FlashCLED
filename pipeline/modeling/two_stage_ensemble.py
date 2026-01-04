import logging
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwoStageEnsemble:
    """
    Two-stage stacked ensemble for conflict prediction.
    
    Thesis Methodology Enforced:
    1. Stage 2 Regressor is strictly PoissonRegressor (Count Model).
    2. Non-negativity constraints applied to all inputs and outputs.
    3. Validation uses time-series splits to prevent leakage.
    """

    def __init__(
        self,
        theme_models: List[Dict],
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.theme_models = theme_models
        self.n_folds = n_folds
        self.random_state = random_state

        self.meta_binary: Optional[LogisticRegression] = None
        self.meta_regress: Optional[PoissonRegressor] = None
        self.is_fitted: bool = False

    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        n_themes = len(self.theme_models)

        oof_binary = np.zeros((n_samples, n_themes), dtype=float)
        oof_regress = np.zeros((n_samples, n_themes), dtype=float)

        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        for theme_idx, theme in enumerate(self.theme_models):
            feature_cols = theme["features"]
            X_theme = X[feature_cols]

            # Clone base models to avoid mutating original instances
            base_clf = theme["binary_model"]
            base_reg = theme["regress_model"]

            for train_idx, val_idx in tscv.split(X_theme):
                # Split data
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

                # 2. Fit & Predict Regressor (Conflict cases only)
                reg_fold = clone(base_reg)
                conflict_mask = (y_train_bin == 1)
                
                if conflict_mask.sum() > 0:
                    reg_fold.fit(X_train[conflict_mask], y_train_fat[conflict_mask])
                    
                    # Predict only where model is valid (technically anywhere, but meaningfulness varies)
                    # We store predictions for all validation points to feed meta-learner
                    oof_regress[val_idx, theme_idx] = reg_fold.predict(X_val)

        return oof_binary, oof_regress

    def fit(self, X: pd.DataFrame, y_binary: pd.Series, y_fatalities: pd.Series) -> None:
        logger.info("Fitting TwoStageEnsemble (Methodology: Poisson + Hurdle)...")

        y_binary = pd.Series(y_binary).astype(int)
        y_fatalities = pd.Series(y_fatalities).astype(float)

        # 1. Generate OOF Predictions (Level 1)
        oof_binary, oof_regress = self._generate_oof_predictions(X, y_binary, y_fatalities)

        # 2. Train Meta-Classifier (Logistic Regression)
        self.meta_binary = LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=-1)
        self.meta_binary.fit(oof_binary, y_binary)

        # 3. Train Meta-Regressor (Poisson Regression)
        # Strictly enforces Thesis requirement: Count-based loss
        conflict_mask = y_binary == 1
        if conflict_mask.sum() > 0:
            self.meta_regress = PoissonRegressor(alpha=1.0, max_iter=1000)
            
            # Constraint: Poisson requires non-negative target
            y_fat_conflict = np.maximum(y_fatalities[conflict_mask], 0)
            
            # Constraint: Enforce non-negativity on inputs from base models
            X_regress = np.maximum(oof_regress[conflict_mask.values, :], 0)
            
            self.meta_regress.fit(X_regress, y_fat_conflict)
        else:
            self.meta_regress = None

        # 4. Final Retrain of Base Models on Full Data
        logger.info("Retraining base models on full dataset...")
        for theme in self.theme_models:
            X_theme = X[theme["features"]]
            theme["binary_model"].fit(X_theme, y_binary)
            if conflict_mask.sum() > 0:
                theme["regress_model"].fit(X_theme[conflict_mask], y_fatalities[conflict_mask])

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        n_samples = len(X)
        n_themes = len(self.theme_models)
        
        l1_binary = np.zeros((n_samples, n_themes))
        l1_regress = np.zeros((n_samples, n_themes))

        # Level 1 Predictions
        for i, theme in enumerate(self.theme_models):
            X_theme = X[theme["features"]]
            
            # Classifier
            try:
                l1_binary[:, i] = theme["binary_model"].predict_proba(X_theme)[:, 1]
            except AttributeError:
                l1_binary[:, i] = theme["binary_model"].predict(X_theme)
                
            # Regressor
            l1_regress[:, i] = theme["regress_model"].predict(X_theme)

        # Level 2 Predictions
        prob = self.meta_binary.predict_proba(l1_binary)[:, 1]

        if self.meta_regress:
            # Constraint: Enforce non-negativity on inputs
            regress_inputs = np.maximum(l1_regress, 0)
            mu_fatal = self.meta_regress.predict(regress_inputs)
            # Constraint: Enforce non-negativity on outputs
            mu_fatal = np.clip(mu_fatal, 0.0, None)
        else:
            mu_fatal = np.zeros_like(prob)

        return prob, mu_fatal