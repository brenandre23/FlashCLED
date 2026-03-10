# Architectural & Modeling Decisions

## Core System Architecture
- **H3 Grid (Res 5):** Chosen to balance local granularity (~10km) with computational efficiency across the Central African Republic. Signed BIGINT used for DB compatibility.
- **Two-Stage Hurdle Ensemble:** Uses a classification stage (conflict occurrence) and a regression stage (fatality intensity) to handle zero-inflation in conflict data.
- **Memory Isolation:** Modeling phases (matrix building, training, prediction) are run as separate subprocesses to prevent Python OOM (Out-of-Memory) errors during large feature joins.
- **Hub-and-Spoke Imports:** All pipeline modules must import `utils.py` for configurations and DB engines. Cross-module imports (e.g., ingestion importing processing) are strictly forbidden to prevent circular dependencies.

## Data & Features
- **14-Day Temporal Spine:** Standard temporal unit for all features and targets.
- **Stocks vs. Flows Imputation:**
    - *Stocks* (e.g., population, distance to road) use Forward Fill (limit: ∞).
    - *Flows* (e.g., prices, rainfall) use limited Forward Fill or Zero-fill depending on domain volatility (see `process_spine_and_infrastructure.py`).
- **NLP Sentiment/Thematics:** ACLED hybrid scoring uses `all-MiniLM-L6-v2` for semantic similarity to anchor event descriptions to theoretical risk mechanisms.

## Operational Principles
- **Verify then Execute (MCP):** Agents must read the `MANIFEST.md` and relevant `DATA_DICTIONARY.md` before proposing code changes.
- **Propose-Draft-Stop:** L1 Workers write code to `.claude/drafts/` first; L2 Managers review and apply to the main codebase.
