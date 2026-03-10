# TASK_QUEUE.md - Project Tracking

---

## 🚨 Active Sprint: Final Model Run & Thesis Submission
*2026-02-20 | Approved plan. See `.claude/plans/swift-juggling-hedgehog.md` for full detail.*

### Execution Phases

| Phase | Task | Status |
|:---|:---|:---|
| 1 | Edit `configs/models.yaml` — logistic meta + disable 3 noise sub-models | 🔄 IN PROGRESS |
| 2 | `python main.py --skip-static --skip-dynamic --skip-features` | ⏳ Pending |
| 3 | `archive_run.py --label logistic_pruned` | ⏳ Pending |
| 4 | `analyze_predictions.py` → `research_questions_diagnostic.py` → `generate_fast_shap.py` | ⏳ Pending |
| 5 | `generate_thesis_figures.py --all` | ⏳ Pending |
| 6 | LaTeX updates: 04-methodology-II.tex, 05-results.tex, 08-appendix.tex | ⏳ Pending |
| 7 | Thesis polish: table formatting, prose (Opus), word count ≤15k | ⏳ Pending |
| 8 | Integrity checks: acronyms, citations, figure refs | ⏳ Pending |

**Sub-models retained (8):** terrain, demographic, environmental, epr, conflict_history, nlp_acled, nlp_interactions, broad_pca
**Sub-models disabled (3):** temporal_context (0.5× baseline), economics (1.1×), nlp_crisiswatch (1.6×)
**Meta-learner:** logistic, C=0.1, class_weight=null

---

## ✅ Overnight Pipeline Sprint — COMPLETE
*2026-02-20 | All 8 tasks completed. See CONTEXT.md for full summary.*

### Execution Order (Strict Sequential)
```
[User: archive_run.py --label xgboost_meta]
   ↓
Task 1: Comparison Agent     → data/runs/comparison_report.md
   ↓
Task 2: Activation Agent     → models/ + data/processed/ updated
   ↓
Task 3: Diagnostics Agent    → summary_metrics.csv + RQ figures
   ↓
Task 4: Figure Agent         → Overleaf/Newest Figures/ (all figs)
   ↓
Task 5: Results Writer       → 05-results.tex (all [X.XXX] filled)
   ↓
Task 6: Table Reformatter    → all landscape tables → sidewaystable/9pt
   ↓
Task 7: Doc Polish (Opus)    → prose quality, structure, no repetition
   ↓
Task 8: Word Count Audit     → enforce 15,000-word body limit
```

### Task 1 — Model Comparison Agent
- [ ] Read `xgboost_meta/metrics/` vs `logistic_meta/metrics/` on PR-AUC, MCC, Recall@Top-10%, RMSE
- [ ] Run `python generate_thesis_figures.py --compare-runs`
- [ ] Write `data/runs/comparison_report.md` — winner + metric table + 3-5 sentence rationale

### Task 2 — Activation Agent
- [ ] Copy winning run models → `models/`, predictions → `data/processed/`, analysis → `data/processed/analysis/`
- [ ] Update `CONTEXT.md` with winner decision
- [ ] Mark Tasks 4 & 5 in this queue as active

### Task 3 — Diagnostics Agent
- [ ] `conda run -n geo_env python pipeline/analysis/analyze_predictions.py`
- [ ] `conda run -n geo_env python research_questions_diagnostic.py`
- [ ] `conda run -n geo_env python stats/recall_top10_decomposition.py`
- [ ] Outputs: `summary_metrics.csv`, `rq1/rq2/rq3_*.png`, `recall_top10_decomposition.png`
- [ ] On crash: log error to `CONTEXT.md`, continue pipeline

### Task 4 — Figure Generation Agent
- [ ] `conda run -n geo_env python generate_thesis_figures.py --all`
- [ ] Verify minimum: `fig_5_1.png` through `fig_5_9.png` + subfigures in `Overleaf/Newest Figures/`

### Task 5 — Results Writer Agent (Sonnet)
- [ ] Read all metrics CSVs + `summary_metrics.csv` + `comparison_report.md`
- [ ] Fill all 70+ `[X.XXX]` placeholders in `Overleaf/sections/05-results.tex`
- [ ] Write thesis-quality commentary per section (Overall, RQ1, RQ2, RQ3)
- [ ] Flag BCCP calibration limitation inline

### Task 6 — Table Reformatting Agent
- [ ] Audit ALL `.tex` files: `Overleaf/sections/` + any appendix table files
- [ ] Replace `\begin{landscape}...\end{landscape}` with `\begin{sidewaystable}...\end{sidewaystable}`
- [ ] Apply `\fontsize{9}{11}\selectfont` to all wide tables (minimum 9pt — no smaller)
- [ ] Tables exceeding `\textwidth` even rotated → split into Table A1/A2
- [ ] Do NOT change landscape page orientation — rotation only

### Task 7 — Documentation Polish Agent (Opus 4.6)
- [ ] Scan all 8 section files for run-on sentences → split or restructure
- [ ] Fix spurious `?` from broken `\ref{}` / `\cite{}` (log all instances)
- [ ] Flatten headings nested beyond 3 levels (e.g., `4.3.2.2` → merge with parent or promote)
- [ ] Eliminate repeated explanations — each concept stated once, revisited only if essential
- [ ] Cut wordy paragraphs — prefer concise over comprehensive
- [ ] Flag/remove insubstantial figures and text that don't advance the argument

### Task 8 — Word Count & Conciseness Agent
- [ ] Count words in Chapters 1–7 (body only; appendix excluded from limit)
- [ ] Report word count per chapter (before edits)
- [ ] Identify top 3 chapters most over-budget
- [ ] Execute targeted cuts: redundant sentences, repeated background, decorative prose
- [ ] Target: ≤ 15,000 words body total
- [ ] Report word count per chapter (after edits) in `CONTEXT.md`

---

## 🚨 Active Sprint: Dual Meta-Learner Comparison & Selection
*Determining the optimal stacking architecture (XGBoost vs. Logistic) for final thesis results.*

### 1. Model & Parameter Alignment (DONE)
- [x] **BCCP Synchronization:** Wire `models.yaml` to `train_single_model.py` and `generate_predictions.py` (enabled, contiguous, log_scale).
- [x] **Methodology Audit:** Update LaTeX files (`hyperparameter-table.tex`, `04-methodology-II.tex`, etc.) to reflect meta-learner options and corrected `scale_pos_weight` (18/8).
- [x] **Placeholder Scrub:** Replace all hardcoded results in Chapter 5 and Conclusion with `[XX.X]` placeholders.
- [x] **FastSHAP Update:** Synchronize `generate_fast_shap.py` to support XGBoost meta-learner importance.
- [x] **Archive Script Fix:** Fixed `archive_run.py` to correctly locate `conformal_diagnostics.csv`.

### 2. XGBoost Meta-Learner Run (ERROR - INCOMPLETE)
- [x] **Training & Prediction:** Completed XGBoost meta-learner run.
- [x] **Archive:** Snapshotted to `data/runs/xgboost_meta`. Note: `research_questions_diagnostic.py` was NOT run on this specific archive set yet.
- [ ] **Error:** It was overwritten by the logistric regression learner. Currently the XGboost is now underway and will be saved to xgboost_meta when complete.

### 3. Logistic Meta-Learner Run (DONE)
- [x] **Config Swap:** Updated `models.yaml` to `type: logistic` with `C=0.1`.
- [x] **Execution:** `python main.py --skip-static --skip-dynamic --skip-features` was run
- [x] **Archive:** Immediately run `python scripts/archive_run.py --label logistic_meta` once complete (skip diagnostics for now). 

### 4. Head-to-Head Comparison (→ Overnight Pipeline Task 1)
- [ ] **Generate Figure:** Run `python generate_thesis_figures.py --compare-runs` to produce `fig_meta_comparison.png`.
- [ ] **Architectural Selection:** Analyze PR curves and probability floor to select the final "Winner" (the primary model for the thesis).

### 5. Final Winning Run & Full Diagnostics (→ Overnight Pipeline Tasks 2–5)
- [ ] **Restore Winner:** Copy models and predictions of the selected winner back to active directories (`models/`, `data/processed/`).
- [ ] **Full Diagnostics:** Run `python research_questions_diagnostic.py` ONLY on the winning model to generate final MCC, Balanced Acc, and Producer/User Accuracy tables.
- [ ] **Placeholder Population:** Fill Chapter 5 LaTeX placeholders with the winning metrics.
- [ ] **Analysis & Visuals:** Run `python pipeline/analysis/analyze_predictions.py` and `python generate_thesis_figures.py` to refresh all thesis panels.

### 6. Final Thesis Polishing (DONE - 2026-02-19)
- [x] **SDG Logo Integration:** SDG 16 logo added to `declarations.tex`.
- [x] **Figure Captions:** Citations strengthened for Fig 2.2, 2.3, 2.4.
- [x] **Admin Logic:** Bangui-Bimbo hybrid-region correction logic verified and updated.
- [x] **Image Cleanup:** `fig3.3(b).png` labels removed via PIL.
- [x] **Future Work:** Added Causal Inference and NLP Architecture refinement subsections.

### 7. Remaining Tasks (Next Session)
- [x] **Remove Bangui-Bimbo Verification Note:** Note was not found in LaTeX; confirmed clean.
- [x] **Fix Feature Count Contradiction:** Reconciled in `04-methodology-II.tex`. Corrected to 120 raw features + 53 PCA = 173 total variables.
- [ ] **Fig 2.4 Decision:** User has fixed `fig_2_4.png` — verify reference in `02-Literature-review.tex` remains unchanged.
- [ ] **Delete stale placeholder:** Remove old simulated fatality distribution image from Overleaf.

---

## 🚨 Pipeline Bug: Calibration Warning (NEEDS INVESTIGATION)
*Seen during logistic meta-learner run (2026-02-19 17:18:50):*
`INFO - ℹ️ Calibration disabled; metrics/BCCP using training indices.`

- [ ] **Triage:** Determine if calibration is intentionally disabled or a config/flag bug. Check `models.yaml`, `train_single_model.py`, and `generate_predictions.py` for the BCCP calibration flag. If calibration should be active, fix and re-run.
- [ ] **Verify:** Confirm BCCP metrics are computed on held-out (test) indices, NOT training indices.

---

## 📐 Appendix Consolidation & Layout Fixes (~40 pages → target: ~15 pages)
*Priority: High. Current appendix section is far too long for a thesis.*

### A. Layout & Formatting Fixes
- [ ] **Margin overflow audit:** Scan all appendix `.tex` files for tables that exceed `\textwidth`. Apply `\resizebox{\textwidth}{!}{...}` or reduce font to `\footnotesize` / `\fontsize{9pt}{11pt}\selectfont` as needed. **Minimum font size: 9pt** — do not go smaller.
- [ ] **Word overlap / column spacing:** Fix tables where cell content collides. Use `tabularx`, `p{Xcm}` columns, or line-break long strings.

### B. Content Consolidation
- [ ] **Imputation appendix:** Replace per-variable rows with category-level summary rows (e.g., "Conflict event counts (8 features) — Forward-fill, max gap 14d"). Target: one row per feature group, not one row per feature.
- [ ] **Feature list appendix:** Group by data source/theme (ACLED, GDELT, VIIRS, IOM, etc.) with counts, not exhaustive listings. Drop low-information columns (e.g., raw units if obvious).
- [ ] **Hyperparameter table:** Consolidate sub-model rows; remove parameters left at default; keep only tuned hyperparameters.
- [ ] **Reproducibility/Temporal appendix:** Review for redundancy with body text; trim to essential entries only.
- [ ] **General rule:** Any table > 1 page must be justified or split into a summary table (in body) + detail in appendix. Any appendix table > 2 pages should be summarized.

---

## 🎨 Figure Color Palette Audit & Logical Consistency
*All figures in `generate_thesis_figures.py` and `research_questions_diagnostic.py` must use the approved palette and apply colors with semantic logic.*

### Approved Palette
| Hex | RGB | Role |
|:----|:----|:-----|
| `#C96B5D` | (201, 107, 93) | Warm red — conflict/risk/alert |
| `#5C967C` | (92, 150, 124) | Green/teal — humanitarian/positive |
| `#965D79` | (150, 93, 121) | Muted purple — NLP/semantic |
| `#EAE6E3` | (234, 230, 227) | Light neutral — background/grid |
| `#C9AC6A` | (201, 172, 106) | Mustard/gold — economic/food |
| `#426A88` | (66, 106, 136) | Blue — spatial/environmental |
| `#414141` | (65, 65, 65) | Dark gray — text/axes/neutral |

### Color Rules (Non-Negotiable)
1. **No color reuse across different concepts.** Each distinct idea, process, or data source must have a unique color. A color must never represent two different things across any two figures.
2. **Palette-first, extend as needed.** Use the 7 approved colors above for primary assignments. If a figure needs more colors than the palette provides, add new harmonious colors (do NOT double-up palette colors on different concepts to stay within 7).
3. **Extended colors** should be visually distinct from all 7 palette entries and from each other. Suggest picking from a perceptually uniform space (e.g., HCL-balanced). Document any extended colors added.

### Tasks
- [ ] **Build a semantic color map:** Before touching any script, produce a master mapping of `concept/process → hex color` (e.g., NLP/semantic → `#965D79`, conflict/ACLED → `#C96B5D`, environmental/spatial → `#426A88`, economic → `#C9AC6A`, humanitarian/IOM → `#5C967C`). This map governs all figures.
- [ ] **Audit `generate_thesis_figures.py`:** For every figure generator function, verify colors match the semantic map. Flag any matplotlib defaults (`tab:blue`, `C0`, `steelblue`, etc.) or cases where the same color is used for two different concepts.
- [ ] **Audit `research_questions_diagnostic.py`:** Same check — verify all figure colors against the semantic map.
- [ ] **Extend palette where needed:** For any figure requiring more than 7 distinct categories, assign new unique colors and add them to the semantic map. Do not reuse an existing palette color for a new concept.
- [ ] **Risk tier colors:** Verify Critical/High/Elevated tiers use palette colors consistently across all figures (likely `#C96B5D` → `#C9AC6A` → `#5C967C`). Same tier must be same color everywhere.
- [ ] **Legend & caption alignment:** Ensure every figure legend label matches the color logic (no "Series 1" defaults).

---

## 📝 Acronym & Citation Integrity Audit

### Acronym Verification
- [ ] **Full scan:** Search all `.tex` files in `Overleaf/sections/` for acronyms (all-caps tokens ≥ 3 chars). For each, verify:
  1. Defined in full at first appearance (e.g., "Conflict Early Warning and Prediction (CEWP)").
  2. Used as acronym consistently thereafter (not redefined or spelled out again mid-chapter).
- [ ] **Priority acronyms to check:** CEWP, ACLED, GDELT, IOM, BCCP, SHAP, UPC, CAR, NLP, PCA, H3, VIIRS, PR-AUC, MCC, SDG, GEE.

### Bibliography / Citation Verification
- [ ] **Cite-vs-bib cross-check:** Extract all `\cite{}` / `\citep{}` / `\citet{}` keys from `.tex` files and verify each key exists in `Overleaf/bibliography.bib`. Flag any missing entries.
- [ ] **Bib-vs-cite cross-check (optional cleanup):** Identify bib entries never cited in text and flag for removal or retention review.
- [ ] **URL/DOI validity spot-check:** For the 10 most recently added bib entries, verify DOIs resolve.

---

---

## 📐 Thesis Finalization Sprint (New — 2026-02-20)
*Runs after overnight pipeline Tasks 1–5. Goal: submission-ready LaTeX.*

### Table Reformatting (Task 6)
- [ ] Landscape → `sidewaystable` conversion across all section + appendix `.tex` files
- [ ] 9pt font floor for wide tables
- [ ] A1/A2 splits for tables that are too wide even rotated

### Documentation Polish (Task 7 — Opus 4.6)
- [ ] Run-on sentences fixed across all 8 section files
- [ ] Spurious `?` from broken refs/cites resolved
- [ ] Heading depth ≤ 3 levels (no `X.X.X.X` headings)
- [ ] Each concept stated once; repetition eliminated
- [ ] Concise prose — no filler, no decorative padding
- [ ] Insubstantial figures/text flagged and removed

### Word Count Enforcement (Task 8)
- [ ] Body chapters (1–7) audited — target ≤ 15,000 words
- [ ] Per-chapter word count reported in `CONTEXT.md`
- [ ] Budget cuts executed — shortest path to under-limit

### Calibration Bug (Deferred — Not Blocking)
- [ ] **BCCP calibration warning** from logistic run: `Calibration disabled; metrics/BCCP using training indices`
  - Writer (Task 5) inserts inline limitation note
  - Full fix (re-run with calibration) deferred to post-submission or if examiner flags it

---

## ✅ Tracked & Completed (From Refactoring Plan)
- [x] **PCA Integration:** PCA re-integrated as a stable sub-model.
- [x] **NLP Split:** Modeling pipeline treats Semantic signals as independent sub-models.
- [x] **Numeric Citations:** Switched to `natbib` numbers style.
- [x] **UPC Definition:** Added at first kinetic mention in Chapter 1.
- [x] **Appendix Tables:** Reproducibility and Temporal tables created.
- [x] **Tiered System Integration:** Implementation of Critical/High/Elevated risk stratification.
- [x] **Figure Registry Cleanup:** Removed/Commented out redundant registrations before Fig 2.6.
- [x] **Chapter 1 & 2 Polish:** Acronyms, citations, and static figure captions finalized.
