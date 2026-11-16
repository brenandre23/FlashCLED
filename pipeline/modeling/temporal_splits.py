"""
temporal_splits.py
==================
Leak-safe temporal splitting for h-step-ahead forecasting on the H3 x date panel.

A row dated ``d`` for horizon ``h`` carries the label of the window ending
``d + h * step_days``. Two rules follow, and every split in the modelling code
goes through this module so they cannot be forgotten:

1. EMBARGO. A model evaluated from date ``b`` onward may only be trained on
   rows whose *label* was already observable before ``b``:
   ``d + h * step_days < b``. Splitting on the row date alone lets the last
   ``h`` steps before the boundary carry labels from inside the evaluation
   period; neighbouring rows on the other side have near-identical features
   and autocorrelated labels, so a flexible learner can simply memorise them.

2. WHOLE DATES. A boundary never cuts through a date. All cells of one date
   are on the same side, so no fold trains on some cells of a period and is
   scored on their neighbours in the same period (spatial leakage through
   spatially correlated labels).

Row order is irrelevant: everything is derived from the date values.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd


def _as_dates(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).normalize()


def embargo_delta(horizon_steps: int, step_days: int) -> pd.Timedelta:
    return pd.Timedelta(days=int(horizon_steps) * int(step_days))


def embargoed_train_mask(dates, boundary, horizon_steps: int, step_days: int) -> np.ndarray:
    """True for rows whose label window ends strictly before ``boundary``."""
    d = _as_dates(dates)
    return np.asarray(d + embargo_delta(horizon_steps, step_days) < pd.Timestamp(boundary).normalize())


def date_block_folds(
    dates, n_folds: int, horizon_steps: int, step_days: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window folds over whole dates with an embargo of ``horizon_steps``.

    The unique dates are cut into ``n_folds + 1`` consecutive blocks; blocks
    1..n_folds are validation blocks (the first block is train-only, as in
    sklearn's TimeSeriesSplit). For each validation block starting at ``b`` the
    training rows are those with ``date + horizon < b``.

    Returns positional (train_idx, val_idx) pairs; folds whose training side
    would be empty are skipped.
    """
    d = _as_dates(dates)
    unique = d.unique().sort_values()
    if len(unique) < n_folds + 1:
        raise ValueError(f"Need at least {n_folds + 1} distinct dates for {n_folds} folds, got {len(unique)}.")
    block = len(unique) // (n_folds + 1)
    delta = embargo_delta(horizon_steps, step_days)
    values = d.values
    folds = []
    for i in range(n_folds):
        start = len(unique) - (n_folds - i) * block
        stop = start + block
        val_start, val_last = unique[start], unique[stop - 1]
        train_idx = np.flatnonzero(values < (val_start - delta).to_datetime64())
        val_idx = np.flatnonzero((values >= val_start.to_datetime64()) & (values <= val_last.to_datetime64()))
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    if not folds:
        raise ValueError("No usable folds: the embargo leaves no training rows before any validation block.")
    return folds


def temporal_tail_split(
    dates, fraction: float, horizon_steps: int, step_days: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hold out the last ``fraction`` of distinct dates; train rows end one embargo earlier.
    Rows inside the embargo gap belong to neither side.
    """
    d = _as_dates(dates)
    unique = d.unique().sort_values()
    n_tail = max(1, int(round(len(unique) * fraction)))
    if n_tail >= len(unique):
        raise ValueError("Calibration fraction leaves no training dates.")
    tail_start = unique[len(unique) - n_tail]
    values = d.values
    cal_idx = np.flatnonzero(values >= tail_start.to_datetime64())
    train_idx = np.flatnonzero(values < (tail_start - embargo_delta(horizon_steps, step_days)).to_datetime64())
    if len(train_idx) == 0:
        raise ValueError("Embargo leaves no training rows before the calibration tail.")
    return train_idx, cal_idx
