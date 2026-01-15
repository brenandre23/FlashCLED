"""
conformal_prediction.py
=======================
Bin-Conditional Conformal Prediction (BCCP) for fatality forecasting.

Implements the methodology from:
    Randahl, Williams & Hegre (2026). "Bin-Conditional Conformal Prediction 
    of Fatalities from Armed Conflict." Political Analysis.

Key Features:
- Provides prediction intervals with guaranteed coverage across fatality bins
- Addresses the coverage failure of standard conformal prediction on zero-inflated data
- Model-agnostic: wraps any point predictor (including TwoStageEnsemble)

Usage:
    bccp = BinConditionalConformalPredictor(
        bins=[0, 1, 3, 8, 21, 55, 149, np.inf],  # ViEWS-style exponential bins
        alpha=0.1  # 90% coverage
    )
    bccp.fit(y_cal, y_pred_cal)  # Fit on calibration set
    intervals = bccp.predict_intervals(y_pred_test)  # Get prediction intervals
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionInterval:
    """Container for prediction interval results."""
    lower: np.ndarray
    upper: np.ndarray
    point_prediction: np.ndarray
    bin_assignments: np.ndarray  # Which bin each prediction falls into
    coverage_target: float
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "point_prediction": self.point_prediction,
            "lower": self.lower,
            "upper": self.upper,
            "interval_width": self.upper - self.lower,
            "bin": self.bin_assignments,
        })


@dataclass 
class BCCPDiagnostics:
    """Diagnostics from BCCP fitting."""
    n_calibration_samples: int
    bin_edges: List[float]
    bin_counts: Dict[int, int]
    bin_quantiles: Dict[int, float]  # Non-conformity score quantiles per bin
    alpha: float
    
    def summary(self) -> str:
        lines = [
            f"BCCP Diagnostics (α={self.alpha}, coverage={1-self.alpha:.0%})",
            f"  Calibration samples: {self.n_calibration_samples:,}",
            f"  Bins: {len(self.bin_edges) - 1}",
            "  Bin distribution:",
        ]
        for bin_idx, count in sorted(self.bin_counts.items()):
            lower = self.bin_edges[bin_idx]
            upper = self.bin_edges[bin_idx + 1]
            q = self.bin_quantiles.get(bin_idx, np.nan)
            lines.append(f"    Bin {bin_idx} [{lower:.0f}, {upper:.0f}): n={count:,}, q={q:.4f}")
        return "\n".join(lines)


class BinConditionalConformalPredictor:
    """
    Bin-Conditional Conformal Prediction (BCCP) for fatality forecasting.
    
    This method extends standard conformal prediction to ensure coverage
    guarantees hold within user-specified bins of the outcome variable,
    not just marginally.
    
    For zero-inflated fatality data, this prevents the common failure mode
    where standard CP over-covers zeros (98%) and under-covers conflicts (52%).
    
    Parameters
    ----------
    bins : list of float
        Bin edges for partitioning the outcome space.
        Example: [0, 1, 3, 8, 21, 55, 149, np.inf] creates 7 bins:
            Bin 0: [0, 1) = exactly 0 fatalities
            Bin 1: [1, 3) = 1-2 fatalities
            Bin 2: [3, 8) = 3-7 fatalities
            ...
    alpha : float
        Significance level. Coverage target is (1 - alpha).
        Default 0.1 gives 90% coverage.
    nonconformity : str
        Non-conformity measure. Options:
        - "absolute": |y - y_hat| (default)
        - "signed": y - y_hat (asymmetric intervals)
    contiguous : bool
        If True, merge discontiguous intervals into one contiguous interval.
        Slightly over-covers but more interpretable.
    log_scale : bool
        If True, predictions and targets are on log1p scale.
        Intervals are computed on log scale then transformed back.
    """
    
    def __init__(
        self,
        bins: Optional[List[float]] = None,
        alpha: float = 0.1,
        nonconformity: str = "absolute",
        contiguous: bool = True,
        log_scale: bool = True,
    ):
        # Default bins following ViEWS/paper convention
        if bins is None:
            bins = [0, 1, 3, 8, 21, 55, 149, np.inf]
        
        self.bin_edges = np.array(bins, dtype=float)
        self.alpha = alpha
        self.nonconformity = nonconformity
        self.contiguous = contiguous
        self.log_scale = log_scale
        
        # Fitted state
        self.is_fitted = False
        self.bin_quantiles: Dict[int, float] = {}
        self.diagnostics: Optional[BCCPDiagnostics] = None
        
    def _to_original_scale(self, y: np.ndarray) -> np.ndarray:
        """Transform from log1p scale to original fatality counts."""
        if self.log_scale:
            return np.expm1(np.clip(y, 0, 20))  # Clip to prevent overflow
        return y
    
    def _to_log_scale(self, y: np.ndarray) -> np.ndarray:
        """Transform from original scale to log1p scale."""
        if self.log_scale:
            return np.log1p(np.maximum(y, 0))
        return y
    
    def _assign_bins(self, y: np.ndarray) -> np.ndarray:
        """Assign observations to bins based on their values."""
        # Convert to original scale for binning
        y_orig = self._to_original_scale(y)
        # np.digitize returns bin index; subtract 1 for 0-indexing
        bins = np.digitize(y_orig, self.bin_edges) - 1
        # Clip to valid range
        bins = np.clip(bins, 0, len(self.bin_edges) - 2)
        return bins
    
    def _compute_nonconformity(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute non-conformity scores."""
        if self.nonconformity == "absolute":
            return np.abs(y_true - y_pred)
        elif self.nonconformity == "signed":
            return y_true - y_pred
        else:
            raise ValueError(f"Unknown nonconformity measure: {self.nonconformity}")
    
    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> "BinConditionalConformalPredictor":
        """
        Fit the BCCP model on calibration data.
        
        Computes the (1 - alpha) quantile of non-conformity scores
        within each bin.
        
        Parameters
        ----------
        y_true : array-like
            True values (on log1p scale if log_scale=True)
        y_pred : array-like
            Predicted values (on same scale as y_true)
            
        Returns
        -------
        self
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        # Compute non-conformity scores
        scores = self._compute_nonconformity(y_true, y_pred)
        
        # Assign to bins based on TRUE values (not predictions)
        # This is critical: we bin by actual outcomes, not predicted
        bins = self._assign_bins(y_true)
        
        # Compute quantile for each bin
        self.bin_quantiles = {}
        bin_counts = {}
        
        for bin_idx in range(len(self.bin_edges) - 1):
            mask = bins == bin_idx
            bin_scores = scores[mask]
            bin_counts[bin_idx] = len(bin_scores)
            
            if len(bin_scores) > 0:
                # Conformal quantile: (1 - alpha) * (n + 1) / n
                # This ensures finite-sample coverage guarantee
                n = len(bin_scores)
                q_level = min((1 - self.alpha) * (n + 1) / n, 1.0)
                self.bin_quantiles[bin_idx] = float(np.quantile(bin_scores, q_level))
            else:
                # No data in this bin - use global quantile as fallback
                logger.warning(f"Bin {bin_idx} has no calibration data. Using global quantile.")
                n = len(scores)
                q_level = min((1 - self.alpha) * (n + 1) / n, 1.0)
                self.bin_quantiles[bin_idx] = float(np.quantile(scores, q_level))
        
        # Store diagnostics
        self.diagnostics = BCCPDiagnostics(
            n_calibration_samples=len(y_true),
            bin_edges=self.bin_edges.tolist(),
            bin_counts=bin_counts,
            bin_quantiles=self.bin_quantiles.copy(),
            alpha=self.alpha,
        )
        
        logger.info(f"BCCP fitted on {len(y_true):,} calibration samples")
        for bin_idx, count in bin_counts.items():
            if count > 0:
                logger.info(f"  Bin {bin_idx}: n={count:,}, quantile={self.bin_quantiles[bin_idx]:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict_intervals(
        self,
        y_pred: np.ndarray,
        return_all_bins: bool = False,
    ) -> Union[PredictionInterval, Dict[int, PredictionInterval]]:
        """
        Generate prediction intervals for new predictions.
        
        The BCCP algorithm computes intervals for each bin separately,
        then combines them. For a given prediction, we compute what
        the interval would be if the true value fell in each bin.
        
        Parameters
        ----------
        y_pred : array-like
            Point predictions (on log1p scale if log_scale=True)
        return_all_bins : bool
            If True, return intervals for each bin separately
            
        Returns
        -------
        PredictionInterval or dict of PredictionInterval
        """
        if not self.is_fitted:
            raise RuntimeError("BCCP not fitted. Call fit() first.")
        
        y_pred = np.asarray(y_pred).ravel()
        n = len(y_pred)
        n_bins = len(self.bin_edges) - 1
        
        # Compute intervals for each bin
        bin_intervals = {}
        for bin_idx in range(n_bins):
            q = self.bin_quantiles[bin_idx]
            
            if self.nonconformity == "absolute":
                lower = y_pred - q
                upper = y_pred + q
            else:  # signed
                # For signed, we'd need separate upper/lower quantiles
                # Simplify to symmetric for now
                lower = y_pred - q
                upper = y_pred + q
            
            bin_intervals[bin_idx] = (lower, upper)
        
        if return_all_bins:
            result = {}
            for bin_idx, (lower, upper) in bin_intervals.items():
                result[bin_idx] = PredictionInterval(
                    lower=lower,
                    upper=upper,
                    point_prediction=y_pred,
                    bin_assignments=np.full(n, bin_idx),
                    coverage_target=1 - self.alpha,
                )
            return result
        
        # Combine intervals across bins
        # Strategy: For each prediction, determine which bin it likely falls into
        # based on the point prediction, then use that bin's quantile
        pred_bins = self._assign_bins(y_pred)
        
        lower = np.zeros(n)
        upper = np.zeros(n)
        
        for i in range(n):
            bin_idx = pred_bins[i]
            lower[i], upper[i] = bin_intervals[bin_idx][0][i], bin_intervals[bin_idx][1][i]
        
        if self.contiguous:
            # If discontiguous intervals exist, take the union (widest bounds)
            # This is the "contiguized" approach from the paper
            all_lowers = np.array([bin_intervals[b][0] for b in range(n_bins)])
            all_uppers = np.array([bin_intervals[b][1] for b in range(n_bins)])
            lower = all_lowers.min(axis=0)
            upper = all_uppers.max(axis=0)
        
        # Ensure non-negativity on original scale
        if self.log_scale:
            lower = np.maximum(lower, 0)  # log1p(0) = 0
        
        return PredictionInterval(
            lower=lower,
            upper=upper,
            point_prediction=y_pred,
            bin_assignments=pred_bins,
            coverage_target=1 - self.alpha,
        )
    
    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate coverage of prediction intervals on test data.
        
        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Point predictions
            
        Returns
        -------
        dict with coverage metrics
        """
        if not self.is_fitted:
            raise RuntimeError("BCCP not fitted. Call fit() first.")
        
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        intervals = self.predict_intervals(y_pred)
        
        # Check if true values fall within intervals
        covered = (y_true >= intervals.lower) & (y_true <= intervals.upper)
        
        # Overall coverage
        overall_coverage = covered.mean()
        
        # Coverage by bin (based on TRUE values)
        true_bins = self._assign_bins(y_true)
        bin_coverage = {}
        bin_counts = {}
        
        for bin_idx in range(len(self.bin_edges) - 1):
            mask = true_bins == bin_idx
            if mask.sum() > 0:
                bin_coverage[bin_idx] = covered[mask].mean()
                bin_counts[bin_idx] = int(mask.sum())
        
        # Interval width statistics
        widths = intervals.upper - intervals.lower
        
        return {
            "overall_coverage": float(overall_coverage),
            "target_coverage": 1 - self.alpha,
            "coverage_by_bin": bin_coverage,
            "counts_by_bin": bin_counts,
            "mean_interval_width": float(widths.mean()),
            "median_interval_width": float(np.median(widths)),
            "interval_width_by_bin": {
                bin_idx: float(widths[true_bins == bin_idx].mean())
                for bin_idx in range(len(self.bin_edges) - 1)
                if (true_bins == bin_idx).sum() > 0
            },
        }


class StandardConformalPredictor:
    """
    Standard (marginal) Conformal Prediction for comparison.
    
    This is the baseline that BCCP improves upon. SCP provides
    only marginal coverage, which fails on zero-inflated data.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        nonconformity: str = "absolute",
        log_scale: bool = True,
    ):
        self.alpha = alpha
        self.nonconformity = nonconformity
        self.log_scale = log_scale
        self.quantile: Optional[float] = None
        self.is_fitted = False
        
    def _compute_nonconformity(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.nonconformity == "absolute":
            return np.abs(y_true - y_pred)
        return y_true - y_pred
    
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "StandardConformalPredictor":
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        scores = self._compute_nonconformity(y_true, y_pred)
        n = len(scores)
        q_level = min((1 - self.alpha) * (n + 1) / n, 1.0)
        self.quantile = float(np.quantile(scores, q_level))
        
        logger.info(f"SCP fitted: quantile={self.quantile:.4f}")
        self.is_fitted = True
        return self
    
    def predict_intervals(self, y_pred: np.ndarray) -> PredictionInterval:
        if not self.is_fitted:
            raise RuntimeError("SCP not fitted")
        
        y_pred = np.asarray(y_pred).ravel()
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        
        if self.log_scale:
            lower = np.maximum(lower, 0)
        
        return PredictionInterval(
            lower=lower,
            upper=upper,
            point_prediction=y_pred,
            bin_assignments=np.zeros(len(y_pred), dtype=int),
            coverage_target=1 - self.alpha,
        )
    
    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        y_true = np.asarray(y_true).ravel()
        intervals = self.predict_intervals(y_pred)
        covered = (y_true >= intervals.lower) & (y_true <= intervals.upper)
        widths = intervals.upper - intervals.lower
        
        return {
            "overall_coverage": float(covered.mean()),
            "target_coverage": 1 - self.alpha,
            "mean_interval_width": float(widths.mean()),
        }


def create_default_fatality_bins() -> List[float]:
    """
    Create default fatality bins following the ViEWS/paper convention.
    
    Bins are exponentially increasing to handle the heavy right tail:
        Bin 0: 0 fatalities (zeros)
        Bin 1: 1-2 fatalities
        Bin 2: 3-7 fatalities
        Bin 3: 8-20 fatalities
        Bin 4: 21-54 fatalities
        Bin 5: 55-148 fatalities
        Bin 6: 149+ fatalities
    """
    return [0, 1, 3, 8, 21, 55, 149, np.inf]


def create_simple_fatality_bins() -> List[float]:
    """
    Create simple 2-bin partition: zeros vs non-zeros.
    
    This is the minimal BCCP configuration that still provides
    major improvement over SCP for zero-inflated data.
    """
    return [0, 1, np.inf]


def create_quartile_bins(y: np.ndarray, log_scale: bool = True) -> List[float]:
    """
    Create bins based on quartiles of the observed data.
    
    Parameters
    ----------
    y : array-like
        Observed values (on log1p scale if log_scale=True)
    log_scale : bool
        Whether y is on log1p scale
        
    Returns
    -------
    list of bin edges
    """
    if log_scale:
        y_orig = np.expm1(y)
    else:
        y_orig = y
    
    # Handle zero-inflation: always keep zero as its own bin
    y_nonzero = y_orig[y_orig > 0]
    
    if len(y_nonzero) < 4:
        return [0, 1, np.inf]
    
    q25, q50, q75 = np.percentile(y_nonzero, [25, 50, 75])
    
    return [0, 1, q25, q50, q75, np.inf]
