# FlashCLED implementation plan

Written 2026-11-16. Branch: `claude/new-session-7y01je` on `brenandre23/FlashCLED` (head `5bb63d4`).

This plan covers everything still open from the efficiency audit, the leakage review and the thesis comparison. It is organised so one Opus 5.5 session manages several Sonnet 5.5 workers running in parallel.

## 1. Where things stand

Done and pushed (13 tests passing on synthetic data; never run against the real database):

| Area | Change |
|---|---|
| Splits | Train cutoff from `configs/data.yaml`; one-horizon embargo; date-block stacking folds; calibration tail on whole dates |
| Conformal | Calibrated on out-of-fold predictions when no calibration set is held out |
| Features | Point-in-time actor risk weights; unlimited backfill removed from lagged economy and displacement features; price shock uses the median of the previous 26 steps |
| Stacking | Logistic meta-learner on log-odds; Poisson meta-learner on `log1p` inputs |
| Targets | Matched on label date; unknown beyond ACLED coverage |
| Forecasting | Newest periods scored from `feature_matrix_forecast.parquet`; tiers ranked on expected fatalities |
| ACLED | Current OAuth API with cursor pagination; pipeline stops if coverage is more than 45 days stale |
| Operations | Model fingerprint (stale models moved to `models/superseded/`); base learners use all cores |
| Config | Window to 2026-11-16; cutoff 2026-01-01 |

Not done: every item in sections 4 to 6 below.

## 2. Roles

**Manager: Opus 5.5 (`claude-opus-5-5`).** One session, working in a local clone of the branch.

- Writes each worker's brief: the task, the exact files it owns, the acceptance check, and the invariants in section 3.
- Launches workers in parallel, each in its own git worktree (Agent tool with `model: "sonnet"`, `isolation: "worktree"`).
- Reviews every diff, runs `python -m pytest tests -q` and `ruff check --select F,E9` on the merged result, and merges one worker at a time.
- Owns all methodology decisions and all edits to shared files (`utils.py`, `configs/*.yaml`, `two_stage_ensemble.py`, `temporal_splits.py`). Workers do not touch these unless the table says so.
- Stops and asks the user at the gates in section 7.

**Workers: Sonnet 5.5 (`claude-sonnet-5-5`).** One task each, disjoint files, no methodology choices. A worker that finds it needs a file it does not own reports back instead of editing it.

## 3. Invariants every worker must keep

1. No feature for period t may use data from after t. Any new feature code needs a truncation test in the style of `tests/test_no_leakage.py`.
2. Splits are never positional. Use `pipeline/modeling/temporal_splits.py`.
3. The out-of-sample boundary comes only from `utils.get_train_cutoff()`.
4. Stock variables keep NaN after the forward-fill limit; flow variables are zero-filled. Order of operations follows `docs/CANONICAL_FEATURE_ORDER.md`.
5. `h3_index` stays int64. Re-runs stay idempotent.
6. Track 1 efficiency changes must give identical output. State how that was checked.
7. No edits to `configs/models.yaml` or `configs/data.yaml`.

## 4. Wave 1: parallel, no dependencies

Seven workers at once. File ownership is exclusive.

| ID | Task | Files owned | Acceptance check |
|---|---|---|---|
| W1 | Dashboard: cache the engine (remove per-query dispose), vectorise `/api/predictions/cube`, add gzip to responses | `dashboard/app.py`, `dashboard/requirements_dashboard.txt` | Cube response byte-identical before and after on a fixture, including the `forecast_date` branch and a NULL `priority` |
| W2 | Index `predictions (horizon, date, learner)` after upload; correct the index advice in the docs | `pipeline/modeling/generate_predictions.py`, `docs/DATABASE_SETUP.md` | `CREATE INDEX IF NOT EXISTS` runs after the upsert; no change to rows written |
| W3 | Vectorise `add_spatial_diffusion_features` (audit 3.6, with the all-NaT guard); fix the `anomalies` NameError; delete the `ensure_output_table_schema` stub | `pipeline/processing/utils_processing.py`, new `tests/test_spatial_lag.py` | New function equals the old one with `np.array_equal` on dense, sparse, duplicate-key, NaT and all-NaT inputs |
| W4 | Faster upload path via `pyarrow.csv` for frames that are entirely numeric, boolean or datetime; text frames keep the old path | `utils.py` (function `upload_to_postgis` only), new `tests/test_upload_csv.py` | Parsed values and NULL positions identical to the pandas path; a frame with a `str` or `object` column takes the old path |
| W5 | Dead code: broken `--run-diagnostics-only` import, unreachable collinearity branch, duplicate `gc` import, schema-only read at `main.py:697`, missing `load_configs` import in `process_crisiswatch.py`, `setup.py` entry point, dead README links, `ruff --select F401,F811 --fix` | `main.py`, `pipeline/processing/process_crisiswatch.py` (import line only), `setup.py`, `README.md`, unused imports repo-wide except files owned by other workers | `python main.py --help` works; ruff F clean; no behaviour change in working modes |
| W6 | Out-of-sample filter for the residual and precision-recall maps; column-pruned read in `analyze_predictions.py`; refit PCA inside the training window in the sensitivity analysis | `pipeline/analysis/analyze_spatial_residuals.py`, `generate_precision_recall_maps.py`, `analyze_predictions.py`, `analyze_sensitivity.py` | Maps use only rows dated on or after the cutoff; same columns loaded as before |
| W7 | Methodology change note for the thesis: every place the code now differs from the text, with the reason (section 8 is the starting list) | new `docs/METHODOLOGY_CHANGES.md` | Each entry cites the thesis section and the code location |

## 5. Wave 2: parallel, after Wave 1 is merged

| ID | Task | Files owned | Depends on | Acceptance check |
|---|---|---|---|---|
| W8 | Freshness guards for GDELT, IODA and CrisisWatch, modelled on `assert_acled_fresh`; index `temporal_features (date)` after the conflict upsert | new `pipeline/common/freshness.py`, `pipeline/processing/process_conflict_features.py` | W3 | Pipeline stops with a clear message when a source ends before the window; override by environment variable |
| W9 | Rolling-origin backtest: retrain at yearly cutoffs (2022 to 2026), score only each origin's following year, pool the metrics, add block-bootstrap intervals by date; report recall-at-tier for three rankings (expected fatalities, onset probability, conformal upper bound) | new `scripts/rolling_origin_backtest.py`; manager adds a cutoff and models-directory override to `utils.py` | W4 | Each origin's models live in their own directory; no origin scores rows before its own cutoff; production `models/` untouched |
| W10 | Cache ACLED note embeddings by model name and text hash | `pipeline/processing/process_acled_hybrid.py` (embedding step only) | none | Second run encodes only new notes; scores equal within 1e-6 |
| W11 | Extract the price-shock and anomaly calculations into functions and add truncation tests | `pipeline/processing/process_spine_and_infrastructure.py`, new tests | none | Output unchanged; tests fail if the current or a future step enters the baseline |

W9 is the one that matters most for reported results: a single ten-month hold-out is too short to support the thesis claims.

## 6. Wave 3: manager-led, needs a decision first

**Parquet as the feature store (audit 4.2).** Opus designs it; two Sonnet workers implement it in sequence.

1. Conflict features go to their own table and are joined in the matrix query. This removes the full-table upsert.
2. Stages hand off through Parquet; Postgres is loaded once at the end for the dashboard.

If this is approved, drop the `temporal_features` half of W4's benefit from expectations; W4 still helps `predictions`.

Readers and writers of `temporal_features` inside the repo (checked 2026-11-16), which the design must keep working:

- Writers: `process_spine_and_infrastructure.py` (drop and reload), `calculate_epr_features.py` (upsert), `process_conflict_features.py` (upsert).
- Readers mid-run: `process_conflict_features.py` (spine keys), `calculate_epr_features.py`, `validation/run_assertions.py`, `build_feature_matrix.py`.
- Readers after the run: `dashboard/app.py`, `scripts/audit_source_dates.sql`.

Nothing outside the repo is assumed to read the table. The table stays in Postgres, loaded once at the end, so the dashboard and the audit SQL are unaffected.

## 7. Gates that need the user

| Gate | When | Question or action |
|---|---|---|
| G1 | Before anything | Say whether the USB backup exists and what is on it (section 10) |
| G2 | Before the first real run | Provide credentials and the manual input files listed in section 10 |
| G3 | After Wave 2 | First real run. Compare new metrics with the thesis figures and decide what to report |
| G4 | After W9 | Choose the tier ranking from the backtest comparison (below) |

**Tier ranking.** The code now ranks on expected fatalities; before it ranked on the conformal upper bound; the thesis text says onset probability. W9 must report recall-at-tier for all three so the choice is made on evidence. Until then expected fatalities is the default.

Workers never run the real pipeline, drop tables or push. The manager pushes only after tests pass.

## 8. Thesis text that no longer matches the code

Starting list for W7:

- Section 4.5: onset meta-learner is logistic regression on log-odds, not a shallow XGBoost.
- Section 4.5.2: intensity base learners use a Poisson objective, not squared error; the Poisson meta-learner takes `log1p` inputs.
- Section 4.6: tiers are ranked on expected fatalities.
- Section 4.7: conformal calibration uses out-of-fold predictions.
- Section 5.1: hold-out is now 2026 onward with a one-horizon embargo; the 2024 to 2025 figures were produced before the leakage fixes.
- Chapter 6: "~900 / 2,700 / 5,400 cells" does not match a 3,407-cell grid.
- Section 5.1.1: 92.0% and 94% are quoted for the same quantity.
- Actor risk weights are point-in-time (check the appendix entry that mentions `lookback_months=12`).

## 10. Rebuilding on a new machine

The original machine was wiped. Assume the database, `data/`, `models/`, `.env` and `configs/pruned_features.yaml` are gone unless the USB backup has them. None of these were ever in git.

**If the USB has a database dump or the `data/` folder:** restore it, then run incrementally with `python main.py --skip-static`. Check what the dump's ACLED coverage ends at before trusting it.

**If nothing was saved, rebuild in this order:**

1. Python 3.10 or 3.11 and `pip install -r requirements.txt`.
2. PostgreSQL 13+ with the `postgis`, `h3` and `h3_postgis` extensions; create the database (`docs/DATABASE_SETUP.md`, `docs/INSTALL.md`).
3. `.env` from `.env.example`. The code reads: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`, `ACLED_EMAIL`, `ACLED_PASSWORD`, `FEWS_NET_EMAIL`, `FEWS_NET_PASSWORD`, `IOM_PRIMARY_KEY`, `SH_CLIENT_ID`, `SH_CLIENT_SECRET`, `GOOGLE_APPLICATION_CREDENTIALS` (Earth Engine and BigQuery).
4. Manual files in `data/raw/`: a current CrisisWatch text file, `regions.geojson`, `prefectures.json`, `subprefectures.json`; optionally `acled.csv` as the API fallback. Check `docs/INSTALL.md` for anything else that is downloaded by hand.
5. `python init_db.py`, then the full run: `python main.py --no-incremental --run-pruning`.

Things to expect on a rebuild:

- The Earth Engine exports for 2000 to 2026 are the slowest step; their yearly CSV cache was lost with the machine.
- `--run-pruning` regenerates `configs/pruned_features.yaml`. Without it every candidate feature is used, which is not what the thesis models used. Its training window ends 2022-12-31, well before the cutoff, so it does not leak.
- A rebuilt database will not be identical to the thesis one: ACLED revises past events and other sources are re-published.
- The dashboard container in `docker-compose.yml` has a hardcoded host IP for the old machine.

The manager should start Wave 1 while the rebuild runs: no Wave 1 or Wave 2 task needs the data.

## 9. Final verification (manager)

1. `python -m pytest tests -q` and ruff clean on the merged branch.
2. On the real feature matrix: confirm the training log reports the last label window ending before the cutoff.
3. One horizon and learner trained with 1 thread and with all threads; predictions compared with `np.array_equal`.
4. `EXPLAIN ANALYZE` on the dashboard queries shows index scans.
5. Rolling-origin metrics reported with intervals, next to the single hold-out figures.
