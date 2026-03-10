Figure/Table Sources
- Figure 1.3 (Research question alignment matrix): `generate_thesis_figures.py` (alignment chart logic).
- Figure 2.3 (Neighbor definitions): `generate_thesis_figures.py` neighbor section.
- Figure 2.7 (Two-Stage Hurdle Ensemble): architecture diagram in `Figures/` built from `generate_thesis_figures.py`.
- Figure 3.1 (Geospatial pipeline DAG): `generate_thesis_figures.py` DAG drawing.
- Figure 3.5 (Dasymetric disaggregation): generated via `generate_thesis_figures.py`; source data in `data/processed/`.
- Table 3.1 (NLP semantic anchors): `generate_thesis_figures.py` table builder.
- Table 4.1 (RQ-to-metric alignment): `generate_thesis_figures.py` formatting section.
- Figure 5.2 panels:
  - 5.2a ROC/PR: `data/processed/analysis/model_selection_curves.png`
  - 5.2b/5.2c bars: `data/processed/analysis/model_selection_bars.png`
  - 5.2d Prediction Scatter: `data/processed/analysis/fatality_scatter.png` (created by `pipeline/analysis/analyze_predictions.py`)
- Other analysis plots: `data/processed/analysis/` (e.g., `thesis_intensity.png`, `comparison_metrics.csv`).

Figure generation workflow
1) Run analysis to refresh inputs: `python pipeline/analysis/analyze_predictions.py`
2) Regenerate all thesis figures: `python generate_thesis_figures.py`
