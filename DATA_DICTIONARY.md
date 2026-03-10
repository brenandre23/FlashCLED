Core processed datasets (data/processed/)
- feature_matrix.parquet — full ABT used for modeling; columns: features + `target_{steps}_step`.
- matrix_meta.json — schema/meta for feature_matrix.
- features_crisiswatch.parquet — CrisisWatch-derived feature subset.
- predictions_{14d,1m,3m}_{lightgbm,xgboost}.parquet — forecast outputs; key cols: `predicted_conflict_prob` (if present), `predicted_fatalities`, `fatalities_lower/upper`.
- shap_explanations_14d_*.parquet — SHAP values for 14d models.
- copernicus_dem_90m.tif, slope_car.tif, tri_car.tif — terrain rasters used in feature engineering.

Analysis outputs (data/processed/analysis/)
- comparison_metrics.csv — ROC/PR/RMSE/Poisson dev metrics by horizon/learner.
- temporal_auc_by_year.csv — yearly AUC (if generated).
- thesis_intensity.png — hexbin comparison plot (14d baseline vs weighted).
- fatality_scatter.png — multi-horizon actual vs predicted fatalities scatter.
- model_selection_curves.png / model_selection_bars.png — ROC/PR curves and bars.

Column conventions
- Targets: `target_{steps}_step` where steps = 1 (14d), 2 (1m), 6 (3m); fallbacks include `target_fatalities_{steps}_step`, `target_binary_{steps}_step`, or `fatalities_14d_sum`.
- Fatality predictions: may appear as `predicted_fatalities`; plotters also look for columns containing both learner key and “fatal”.
- Learner keys: `xgboost`, `lightgbm`, plus variants with `baseline`/`weighted` in names for plots.
