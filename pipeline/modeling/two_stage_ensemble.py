import logging
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
)

# Optional imports for type hinting convenience
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageEnsemble:
    """
    Two-stage stacked ensemble for conflict prediction.
    Model-Agnostic: Supports XGBoost, LightGBM, or any Scikit-Learn compatible estimator.

    Stage 1:
        - A set of theme-specific base models (each with a classifier and regressor)
        - Each theme has its own feature subset.

    Stage 2 (meta-learners):
        - LogisticRegression on stacked classifier outputs (probabilities)
        - Ridge regression on stacked regressor outputs (expected fatalities)

    The key constraint:
        - Out-of-fold (OOF) predictions are used to train the meta-learners.
          This avoids leakage and yields honest validation for the ensemble.

    Time-series validation:
        - Uses TimeSeriesSplit with expanding window to respect temporal order
        - Training data always comes before validation data
    """

    def __init__(
        self,
        theme_models: List[Dict],
        n_folds: int = 5,
        random_state: int = 42,
        classifier_downsample_ratio: Optional[float] = None,
    ) -> None:
        """
        Args:
            theme_models:
                List of dicts with the following keys:
                    - 'name': str, theme name
                    - 'features': list[str], feature column names used by this theme
                    - 'binary_model': an unfitted classifier (XGBClassifier/LGBMClassifier)
                    - 'regress_model': an unfitted regressor (XGBRegressor/LGBMRegressor)

            n_folds:
                Number of TimeSeriesSplit splits used to generate OOF predictions.

            random_state:
                Random seed for downsampling.

            classifier_downsample_ratio:
                If provided (e.g. 3.0), in each CV training fold we keep all
                conflict cases (y_binary == 1) and randomly sample at most
                `ratio * n_positive` non-conflict cases. Validation folds are
                *never* downsampled.
        """
        self.theme_models = theme_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.classifier_downsample_ratio = classifier_downsample_ratio

        self.meta_binary: Optional[LogisticRegression] = None
        self.meta_regress: Optional[Ridge] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate out-of-fold predictions for each theme model using time-series splits.

        Returns:
            oof_binary:  (n_samples, n_themes) array of classifier probabilities
            oof_regress: (n_samples, n_themes) array of regressor predictions
                         (zero where model is not applied in validation fold).
        """
        n_samples = len(X)
        n_themes = len(self.theme_models)

        oof_binary = np.zeros((n_samples, n_themes), dtype=float)
        oof_regress = np.zeros((n_samples, n_themes), dtype=float)

        # Use TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(
            n_splits=self.n_folds,
        )

        for theme_idx, theme in enumerate(self.theme_models):
            theme_name = theme.get("name", f"theme_{theme_idx}")
            feature_cols = theme["features"]

            logger.info(f"Generating OOF predictions for theme {theme_idx+1}/{n_themes}: '{theme_name}'")

            missing_feats = [c for c in feature_cols if c not in X.columns]
            if missing_feats:
                raise ValueError(
                    f"Missing features for theme '{theme_name}': {missing_feats}"
                )

            X_theme = X[feature_cols]

            for fold_idx, (train_idx, val_idx) in enumerate(
                tscv.split(X_theme)
            ):
                logger.info(
                    f"  Fold {fold_idx+1}/{self.n_folds} for theme '{theme_name}' - "
                    f"Train: {len(train_idx)}, Val: {len(val_idx)}"
                )

                # Ensure contiguous memory for LightGBM stability
                X_train_fold = X_theme.iloc[train_idx]
                X_val_fold = X_theme.iloc[val_idx]
                y_train_binary = y_binary.iloc[train_idx]
                y_val_binary = y_binary.iloc[val_idx]
                y_train_fatalities = y_fatalities.iloc[train_idx]

                # ---------------------------------------------------------
                # Optional downsampling for TRAINING fold (classifier only)
                # ---------------------------------------------------------
                X_train_for_clf = X_train_fold
                y_train_for_clf = y_train_binary

                ratio = self.classifier_downsample_ratio
                if ratio is not None and ratio > 0:
                    y_arr = y_train_binary.values
                    pos_mask = y_arr == 1
                    neg_mask = ~pos_mask

                    pos_indices = np.where(pos_mask)[0]
                    neg_indices = np.where(neg_mask)[0]

                    if len(pos_indices) > 0 and len(neg_indices) > 0:
                        max_neg = int(ratio * len(pos_indices))
                        n_neg_keep = min(len(neg_indices), max_neg)

                        rng = np.random.RandomState(
                            self.random_state + theme_idx * 1000 + fold_idx
                        )
                        neg_keep = rng.choice(
                            neg_indices,
                            size=n_neg_keep,
                            replace=False,
                        )

                        keep_idx = np.concatenate([pos_indices, neg_keep])
                        keep_idx = np.unique(keep_idx)

                        X_train_for_clf = X_train_fold.iloc[keep_idx]
                        y_train_for_clf = y_train_binary.iloc[keep_idx]
                    else:
                        logger.warning(
                            "  Skipping downsampling in this fold due to single-class data."
                        )
                # ---------------------------------------------------------

                # 1) Fit classifier on (possibly) downsampled training data
                binary_model = theme["binary_model"]
                # Use clone to ensure a fresh fit for each fold
                binary_fold_model = clone(binary_model)
                binary_fold_model.fit(X_train_for_clf, y_train_for_clf)

                # Predict probabilities on **full** validation fold
                try:
                    probs = binary_fold_model.predict_proba(X_val_fold)[:, 1]
                except AttributeError:
                    probs = binary_fold_model.predict(X_val_fold)
                
                oof_binary[val_idx, theme_idx] = probs

                # 2) Fit regressor ONLY on conflict cases in the full training fold
                regress_model = theme["regress_model"]
                regress_fold_model = clone(regress_model)
                
                conflict_mask_train = (y_train_binary == 1)
                
                if conflict_mask_train.sum() > 0:
                    X_train_conflict = X_train_fold[conflict_mask_train]
                    y_train_fatalities_conflict = y_train_fatalities[conflict_mask_train]

                    regress_fold_model.fit(X_train_conflict, y_train_fatalities_conflict)

                    # Predict on validation conflicts
                    conflict_mask_val = (y_val_binary == 1)
                    if conflict_mask_val.sum() > 0:
                        X_val_conflict = X_val_fold[conflict_mask_val]
                        preds = regress_fold_model.predict(X_val_conflict)

                        # Place predictions only on conflict positions for this fold
                        oof_regress[val_idx[conflict_mask_val], theme_idx] = preds

        return oof_binary, oof_regress

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> None:
        """
        Fit the two-stage ensemble.

        Args:
            X:
                DataFrame of all candidate features. Each theme model
                will select its own subset.

            y_binary:
                Binary indicator (0/1) of whether any conflict occurs.

            y_fatalities:
                Fatality counts for the same horizon. Often 0 where
                y_binary == 0; positive for conflict cases.
        """
        logger.info("=" * 80)
        logger.info("FITTING TwoStageEnsemble")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        y_binary = pd.Series(y_binary).astype(int)
        y_fatalities = pd.Series(y_fatalities).astype(float)

        # Generate OOF predictions for each theme
        oof_binary, oof_regress = self._generate_oof_predictions(
            X, y_binary, y_fatalities
        )

        # ------------------------------------------------------------------
        # Stage 2: Meta-learners
        # ------------------------------------------------------------------
        logger.info("Training meta-learners on OOF predictions...")

        # 1) Meta-classifier on all samples (using OOF binary probs)
        self.meta_binary = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1,
        )
        self.meta_binary.fit(oof_binary, y_binary)

        # 2) Meta-regressor: only train on conflict cases
        conflict_mask = y_binary == 1
        if conflict_mask.sum() > 0:
            self.meta_regress = Ridge(alpha=1.0, random_state=self.random_state)
            self.meta_regress.fit(
                oof_regress[conflict_mask.values, :],
                y_fatalities[conflict_mask],
            )
        else:
            logger.warning(
                "No positive (conflict) cases in training data; "
                "meta_regress will remain None."
            )
            self.meta_regress = None

        # Final Retrain of Base Models on FULL Dataset
        logger.info("Retraining base models on full dataset...")
        for theme in self.theme_models:
            X_theme = X[theme["features"]]
            
            # Classifier
            theme["binary_model"].fit(X_theme, y_binary)
            
            # Regressor
            if conflict_mask.sum() > 0:
                theme["regress_model"].fit(X_theme[conflict_mask], y_fatalities[conflict_mask])

        self.is_fitted = True
        logger.info("TwoStageEnsemble fitted successfully.")

    def predict(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict conflict probability and expected fatalities.

        Args:
            X: DataFrame with the same feature columns used during training.

        Returns:
            prob:     (n_samples,) array of conflict probabilities
            mu_fatal: (n_samples,) array of expected fatalities
        """
        if not self.is_fitted:
            raise RuntimeError("TwoStageEnsemble must be fitted before calling predict().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_samples = len(X)
        n_themes = len(self.theme_models)

        # 1) Stage 1 predictions
        binary_preds = np.zeros((n_samples, n_themes), dtype=float)
        regress_preds = np.zeros((n_samples, n_themes), dtype=float)

        for theme_idx, theme in enumerate(self.theme_models):
            theme_name = theme.get("name", f"theme_{theme_idx}")
            feature_cols = theme["features"]

            missing_feats = [c for c in feature_cols if c not in X.columns]
            if missing_feats:
                raise ValueError(
                    f"Missing features for prediction in theme '{theme_name}': {missing_feats}"
                )

            X_theme = X[feature_cols]

            binary_model = theme["binary_model"]
            regress_model = theme["regress_model"]

            # Predict classifier probability
            try:
                binary_preds[:, theme_idx] = binary_model.predict_proba(X_theme)[:, 1]
            except AttributeError:
                binary_preds[:, theme_idx] = binary_model.predict(X_theme)

            # Predict fatalities
            regress_preds[:, theme_idx] = regress_model.predict(X_theme)

        # 2) Stage 2 meta-predictions
        prob = self.meta_binary.predict_proba(binary_preds)[:, 1]

        if self.meta_regress is not None:
            mu_fatal = self.meta_regress.predict(regress_preds)
            mu_fatal = np.clip(mu_fatal, 0.0, None)
        else:
            mu_fatal = np.zeros_like(prob)

        return prob, mu_fatal

    def evaluate(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_fatalities: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble on a labeled dataset using several metrics.

        Metrics:
            - PR AUC (average precision)
            - Brier score (calibration)
            - Top 10% recall (how many true positives are found in the highest-risk decile)
            - RMSE of expected fatalities on conflict cases
        """
        logger.info("Evaluating TwoStageEnsemble...")

        y_binary = pd.Series(y_binary).astype(int)
        y_fatalities = pd.Series(y_fatalities).astype(float)

        prob, mu_fatal = self.predict(X)

        # PR AUC
        if y_binary.sum() > 0:
            pr_auc = average_precision_score(y_binary, prob)
        else:
            logger.warning("No positive cases in y_binary; PR AUC set to NaN.")
            pr_auc = np.nan

        # Brier score (requires probabilities)
        try:
            brier = brier_score_loss(y_binary, prob)
        except Exception:
            brier = np.nan

        # Top 10% recall
        n = len(prob)
        if n > 0 and y_binary.sum() > 0:
            k = max(1, int(0.10 * n))
            top_k_indices = np.argsort(-prob)[:k]  # descending sort
            top_k_true_positives = y_binary.iloc[top_k_indices].sum()
            total_positives = y_binary.sum()
            top_10_recall = top_k_true_positives / total_positives
        else:
            top_10_recall = np.nan

        # RMSE on conflict cases
        conflict_mask = y_binary == 1
        if conflict_mask.sum() > 0:
            rmse = np.sqrt(
                mean_squared_error(
                    y_fatalities[conflict_mask],
                    mu_fatal[conflict_mask.values],
                )
            )
        else:
            rmse = np.nan

        metrics = {
            "pr_auc": float(pr_auc) if pr_auc is not np.nan else np.nan,
            "brier_score": float(brier) if brier is not np.nan else np.nan,
            "top_10_recall": float(top_10_recall) if top_10_recall is not np.nan else np.nan,
            "rmse": float(rmse) if rmse is not np.nan else np.nan,
        }

        logger.info("\nEvaluation Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if v == v else f"  {k}: NaN")

        return metrics