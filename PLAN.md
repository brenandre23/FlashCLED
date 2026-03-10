# PLAN.md — Sprint: Final Audit, Cleanup & Git Commit
**GSD Methodology | 2026-03-10**

---

## Context

Round 1 deprecation audit complete. Round 2 goal:
1. Second pass audit — verify everything remaining is either active or documented
2. Audit `scripts/` directory (14 files, not properly audited in round 1)
3. Update `.gitignore` for local-only files
4. Stage all changes and create properly named git commits
5. Push to remote

---

## Wave 1 — Three Parallel Workers

### W-A · scripts/ Directory Audit + Delete
**Scope:** Every file in `scripts/` except `scripts/diagnostics/onset_diagnostic.py`

Read each file and classify as KEEP or DELETE using these rules:
- **KEEP:** Called by `main.py`, referenced in `MANIFEST.md`/`CONTEXT.md`, or active thesis utility with no equivalent
- **DELETE:** One-time fix scripts, dev utilities superseded by pipeline, duplicate functionality

**Files to audit:**
- `scripts/RootList.py` — already in .gitignore; check if it does anything useful
- `scripts/__init__.py` — package init; check if imported anywhere
- `scripts/archive_run.py` — used to snapshot model runs; check if still needed
- `scripts/audit_data_availability.py` — data audit; superseded?
- `scripts/audit_docs_repo.py` — doc auditing; one-time?
- `scripts/audit_source_dates.sql` — SQL file; one-time setup?
- `scripts/build_graph.py` — builds dependency graph; check if final_audit.ini supersedes
- `scripts/clean_cache.py` — already in .gitignore; check usefulness
- `scripts/collinearity_check.py` — called by `main.py --run-collinearity-only`; KEEP
- `scripts/fix_fig_5_4d.py` — one-time figure fix; DELETE if already applied
- `scripts/fix_fig_5_7.py` — one-time figure fix; DELETE if already applied
- `scripts/fix_figures_5_4bc_5_5c.py` — one-time figure fix; DELETE if already applied
- `scripts/generate_appendix_data.py` — check if appendix is final
- `scripts/github_commit.py` — git automation; superseded by standard git?
- `scripts/regenerate_appendix_tables.py` — check if appendix is final
- `scripts/rq2_tempo_analysis.py` — RQ2 temporal analysis; check if superseded by onset_diagnostic.py
- `scripts/run_wsl_db_setup.sh` — one-time DB setup shell script
- `scripts/setup_gcloud_earthengine.sh` — one-time GEE setup; DELETE

For each DELETE: actually delete the file.
For each KEEP: note which doc references it (MANIFEST.md, CONTEXT.md, RUNBOOK.md, final_audit.ini).

Write audit results to `.claude/drafts/WA_scripts_audit.md`

---

### W-B · Documentation Cross-Reference Audit
**Scope:** Verify every kept file is referenced somewhere in the project documentation.

**Check 1 — All remaining pipeline/analysis/ files:**
For each file in `pipeline/analysis/` (7 analysis + 5 sidecar = 12 files), verify it appears in at least one of:
- `final_audit.ini`
- `MANIFEST.md`
- `CONTEXT.md`
- `RUNBOOK.md`
- `FIGURES.md`

**Check 2 — Root-level docs that should be tracked:**
Verify these files exist on disk and are coherent:
- `MANIFEST.md`, `CONTEXT.md`, `TASK_QUEUE.md`, `DECISIONS.md`
- `DATA_DICTIONARY.md`, `FIGURES.md`, `GLOSSARY.md`, `RUNBOOK.md`
- `final_audit.ini`, `DEPRECATION_MANIFEST.md`

**Check 3 — configs/ completeness:**
Verify `configs/models.yaml`, `configs/features.yaml`, `configs/data.yaml` are present and non-empty.

**Check 4 — pipeline/modeling/config/pruning_config.py:**
Verify it's imported by `pipeline/modeling/pruning.py` and document what it does.

Write findings to `.claude/drafts/WB_doc_crossref.md`. Flag any kept file with NO documentation reference.

---

### W-C · .gitignore Update
**Scope:** Identify all files/dirs that exist locally but should NOT go to GitHub. Update `.gitignore`.

**Known gaps from W-04 audit:**
```
Figures/                    # generated figure outputs
Overleaf/                   # LaTeX thesis (too large, separate repo)
data/runs/                  # model run archives
Thesis Defense Lessons/     # personal notes
conflict_onset.txt          # diagnostic output artifact
DEPRECATION_MANIFEST.md     # internal planning doc
.claude/settings.local.json # machine-specific settings
ImputationDQ/               # local data quality work
```

**Also check:**
- `agrimatrix_outputs/` — already deleted but add to gitignore to prevent recurrence
- `.gemini/` — already deleted; add to gitignore
- `__pycache__/` in scripts/ — covered by global rule?
- `data/runs/` — model run snapshots, too large for git

**Action:** Edit `.gitignore` to add all missing entries with comments.
Write a summary of changes to `.claude/drafts/WC_gitignore.md`.

---

## Wave 2 — Manager (me) after workers complete

### W-D · Git Staging & Commit
**Scope:** Stage all changes and create atomic, well-named commits.

Commit strategy — split into logical units:

**Commit 1: `cleanup: remove deprecated pipeline stubs and one-time scripts`**
- `pipeline/analysis/population_comparison.py` (deleted)
- `pipeline/analysis/road_network_comparison.py` (deleted)
- `pipeline/database/` (deleted)
- `pipeline/feature_engineering/` (deleted)
- `pipeline/modeling/config/__init__.py` (deleted)
- `scripts/diagnostics/` deleted files
- Any scripts/ deletes from W-A

**Commit 2: `cleanup: remove deprecated docs and scratch files`**
- `docs/CEWP_Data_Source_Audit.md` (deleted)
- `AI_ML_promptdiagnostics.txt` (deleted)
- `research_questions_diagnostic.py` (deleted)
- All `agrimatrix*` deleted files
- All `docs/CEWP_*` deleted files
- `nlp_changes.txt`, `CRITICAL_ISSUES.md` deleted

**Commit 3: `chore: update .gitignore for local-only artifacts`**
- `.gitignore` changes from W-C

**Commit 4: `docs: add project documentation and memory anchors`**
- All new untracked `.md` files: `MANIFEST.md`, `CONTEXT.md`, `DECISIONS.md`, `DATA_DICTIONARY.md`, `FIGURES.md`, `GLOSSARY.md`, `RUNBOOK.md`, `TASK_QUEUE.md`, `PLAN.md`, `DEPRECATION_MANIFEST.md`, `final_audit.ini`
- `MCP/` directory
- `docs/CANONICAL_FEATURE_ORDER.md`, `docs/CEWP_Data_Source_Audit.md` (new)

**Commit 5: `feat: add nlp-refactor pipeline modules`**
- All new untracked Python files in `pipeline/analysis/`, `pipeline/modeling/`, `scripts/`
- `configs/pruned_features.yaml` (if regenerated)
- `pipeline/modeling/config/pruning_config.py`
- `pipeline/modeling/load_data_utils.py`
- `pipeline/modeling/pruning.py`

**Commit 6: `chore: stage deletion of scripts/ ghost entries`**
- `git add -u scripts/` to clear ghost tracked files

After all commits: `git push origin nlp-refactor`

---

## Done Criteria

- [ ] W-A: `scripts/` audited, deprecated files deleted, results documented
- [ ] W-B: All kept files cross-referenced in docs, gaps flagged
- [ ] W-C: `.gitignore` updated with local-only entries
- [ ] W-D: 6 atomic commits created and pushed
- [ ] Zero unintended files on GitHub (data, models, Overleaf)
