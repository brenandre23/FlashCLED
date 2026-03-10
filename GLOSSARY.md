Glossary
- Horizon keys: 14d = 2-Week, 1m = 4-Week, 3m = 12-Week.
- Learner keys: `xgboost`, `lightgbm`; may include suffixes like `baseline`, `weighted`.
- Targets: `target_{steps}_step` (steps 1/2/6); `target_fatalities_{steps}_step` fallback.
- Fatality preds: `predicted_fatalities`, `fatalities_lower/upper`; some plots expect learner-keyed columns containing “fatal”.
- PATHS (from utils): `PATHS['data_proc'] = data/processed`, `PATHS['analysis'] = analysis/`, `PATHS['models'] = models/`.
- CEWP: Conflict Early Warning & Prediction (project name).
