"""
Leakage guards for the forecasting pipeline.

Two kinds of test:
  * split tests   - no training label may come from the period a model is scored on,
                    and no date may sit on both sides of a split;
  * truncation    - a feature value for period t must be identical whether or not
                    data after t exists. If deleting the future changes the past,
                    the feature leaks.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.modeling.temporal_splits import (
    date_block_folds, embargoed_train_mask, temporal_tail_split,
)

STEP = 14


def make_panel(n_cells=12, n_dates=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-04", periods=n_dates, freq=f"{STEP}D")
    df = pd.MultiIndex.from_product([np.arange(n_cells) + 1000, dates], names=["h3_index", "date"]).to_frame(index=False)
    df["x"] = rng.standard_normal(len(df))
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # deliberately unsorted


# --------------------------------------------------------------------------- splits

@pytest.mark.parametrize("horizon", [1, 2, 6])
def test_folds_respect_embargo_and_never_share_a_date(horizon):
    df = make_panel()
    folds = date_block_folds(df["date"], n_folds=5, horizon_steps=horizon, step_days=STEP)
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        train_dates, val_dates = df["date"].iloc[train_idx], df["date"].iloc[val_idx]
        assert set(train_dates).isdisjoint(set(val_dates))
        # every training LABEL (date + horizon) is observed before validation starts
        assert (train_dates + pd.Timedelta(days=horizon * STEP)).max() < val_dates.min()
        # whole dates: each validation date carries all cells
        assert df.iloc[val_idx].groupby("date")["h3_index"].nunique().eq(df["h3_index"].nunique()).all()


def test_folds_do_not_depend_on_row_order():
    df = make_panel()
    ordered = df.sort_values(["h3_index", "date"]).reset_index(drop=True)  # cell-major order
    for (tr_a, va_a), (tr_b, va_b) in zip(
        date_block_folds(df["date"], 5, 2, STEP), date_block_folds(ordered["date"], 5, 2, STEP)
    ):
        key = lambda frame, idx: set(map(tuple, frame.iloc[idx][["h3_index", "date"]].to_numpy()))
        assert key(df, tr_a) == key(ordered, tr_b)
        assert key(df, va_a) == key(ordered, va_b)


@pytest.mark.parametrize("horizon", [1, 6])
def test_train_cutoff_embargo(horizon):
    df = make_panel()
    cutoff = pd.Timestamp("2018-06-01")
    mask = embargoed_train_mask(df["date"], cutoff, horizon, STEP)
    label_dates = df.loc[mask, "date"] + pd.Timedelta(days=horizon * STEP)
    assert label_dates.max() < cutoff
    # nothing usable was thrown away: the next step's label would cross the cutoff
    assert label_dates.max() + pd.Timedelta(days=STEP) >= cutoff


def test_calibration_tail_is_embargoed_and_whole_dates():
    df = make_panel()
    train_idx, cal_idx = temporal_tail_split(df["date"], 0.2, horizon_steps=6, step_days=STEP)
    train_dates, cal_dates = df["date"].iloc[train_idx], df["date"].iloc[cal_idx]
    assert (train_dates + pd.Timedelta(days=6 * STEP)).max() < cal_dates.min()
    assert cal_dates.nunique() == round(df["date"].nunique() * 0.2)


# --------------------------------------------------------------------------- targets

def test_targets_match_label_date_and_mask_unobserved_periods():
    from pipeline.modeling.build_feature_matrix import compute_targets

    df = make_panel(n_cells=3, n_dates=30)[["h3_index", "date"]]
    df["fatalities_14d_sum"] = np.arange(len(df), dtype="float32")
    horizons = [{"name": "14d", "steps": 1}, {"name": "3m", "steps": 6}]
    acled_end = df["date"].max() - pd.Timedelta(days=3 * STEP)

    out = compute_targets(df, horizons, STEP, label_known_until=acled_end)
    truth = df.set_index(["h3_index", "date"])["fatalities_14d_sum"]
    for steps in (1, 6):
        col = f"target_fatalities_{steps}_step"
        label_date = out["date"] + pd.Timedelta(days=steps * STEP)
        known = label_date <= acled_end
        expected = truth.reindex(pd.MultiIndex.from_arrays([out["h3_index"], label_date])).to_numpy()
        assert np.array_equal(out.loc[known, col].to_numpy(), expected[known.to_numpy()])
        # beyond ACLED coverage the outcome is UNKNOWN, never a zero
        assert out.loc[~known, col].isna().all()
        assert out.loc[~known, f"target_binary_{steps}_step"].isna().all()


def test_targets_survive_a_gap_in_the_spine():
    from pipeline.modeling.build_feature_matrix import compute_targets

    df = make_panel(n_cells=2, n_dates=20)[["h3_index", "date"]]
    df["fatalities_14d_sum"] = 1.0
    gap = sorted(df["date"].unique())[10]
    df = df[df["date"] != gap]
    out = compute_targets(df, [{"name": "14d", "steps": 1}], STEP)
    # the row just before the gap has no observed next window: unknown, not the window after
    before_gap = out["date"] == gap - pd.Timedelta(days=STEP)
    assert out.loc[before_gap, "target_fatalities_1_step"].isna().all()


# --------------------------------------------------------------------------- features: truncation invariance

def test_lag_features_do_not_use_future_values():
    from pipeline.processing.process_spine_and_infrastructure import lag1_stationary_start

    df = make_panel(n_cells=4, n_dates=60).sort_values(["h3_index", "date"]).reset_index(drop=True)
    # sparse stock series: source starts late and has long gaps (like IOM displacement)
    df.loc[df["date"] < "2015-09-01", "x"] = np.nan
    df.loc[(df["date"] > "2016-03-01") & (df["date"] < "2016-09-01"), "x"] = np.nan

    full = lag1_stationary_start(df, "x")
    cut = pd.Timestamp("2016-05-01")
    past = df[df["date"] <= cut]
    truncated = lag1_stationary_start(past, "x")
    pd.testing.assert_series_equal(full.loc[past.index], truncated)
    # periods before the source starts stay unknown instead of receiving a later value
    assert full[df["date"] < "2015-09-01"].isna().all()


def test_actor_risk_weights_are_point_in_time():
    from pipeline.processing.process_acled_hybrid import (
        point_in_time_actor_weights, score_actor_risk_point_in_time,
    )

    rng = np.random.default_rng(1)
    n = 600
    events = pd.DataFrame({
        "event_date": pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D"),
        "actor1": rng.choice(["seleka", "anti-balaka", "upc", "3r", "fprc"], n),
        "actor2": rng.choice(["civilians", "minusca", None], n),
        "fatalities": rng.poisson(1.5, n),
    })
    # a group that only becomes lethal late must not colour earlier periods
    late = events["event_date"] >= "2020-01-01"
    events.loc[late, "actor1"] = "wagner group"
    events.loc[late, "fatalities"] = 50
    events["actor"] = events["actor1"].str.lower().str.strip()

    weights = point_in_time_actor_weights(events, lookback_years=1)
    scores_full = score_actor_risk_point_in_time(events, weights)

    cut = pd.Timestamp("2019-06-30")
    past = events[events["event_date"] <= cut]
    scores_past = score_actor_risk_point_in_time(past, point_in_time_actor_weights(past, lookback_years=1))
    pd.testing.assert_series_equal(scores_full.loc[past.index], scores_past)

    for month, mapping in weights.items():
        if month < pd.Timestamp("2020-01-01"):
            assert "wagner group" not in mapping


# --------------------------------------------------------------------------- ensemble end to end

def test_ensemble_fits_with_date_folds_and_exposes_out_of_fold_predictions():
    xgboost = pytest.importorskip("xgboost")
    from pipeline.modeling.two_stage_ensemble import TwoStageEnsemble

    df = make_panel(n_cells=40, n_dates=90, seed=3)
    rng = np.random.default_rng(3)
    df["z"] = rng.standard_normal(len(df))
    y_fat = pd.Series(np.where(df["x"] + rng.standard_normal(len(df)) > 1.2, rng.poisson(3, len(df)) + 1, 0)).astype(float)
    y_bin = (y_fat > 0).astype(int)
    themes = [{
        "name": "t", "features": ["x", "z"],
        "binary_model": xgboost.XGBClassifier(n_estimators=20, max_depth=3, n_jobs=1),
        "regress_model": xgboost.XGBRegressor(n_estimators=20, max_depth=3, objective="count:poisson", n_jobs=1),
    }]
    ens = TwoStageEnsemble(
        themes, calibration_config={"method": "none", "fraction": 0.0},
        meta_config={"type": "logistic", "params": {"max_iter": 500}},
        embargo_steps=2, step_days=STEP,
    )
    ens.fit(df, y_bin, y_fat)

    # OOF rows exclude the first (train-only) block and line up with the caller's rows
    first_val = df["date"].iloc[ens.oof_idx_].min()
    assert first_val > df["date"].min()
    assert len(ens.oof_idx_) == len(ens.oof_expected_fatalities_) == len(ens.oof_prob_)
    assert np.isfinite(ens.oof_expected_fatalities_).all()
    prob, expected = ens.predict(df)
    assert prob.shape == (len(df),) and (expected >= 0).all()
