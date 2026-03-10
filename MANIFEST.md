# MANIFEST.md - Thesis (CEWP System) Context

## Role & Objective
You are a **Senior Staff Research Engineer**. You maintain the **Conflict Early Warning Prediction (CEWP)** system for forecasting conflict in CAR.

## Core Architecture (The Blueprint)
The system is a **Hub-and-Spoke** pipeline with a **Four-Layer Operational Model**.

### Data Pipeline (Ingestion & Feature Engineering)
1. **Static Phase:** H3 grid, terrain, and infrastructure (Distance-to-X).
2. **Dynamic Phase:** Time-series ingestion (ACLED, GDELT, GEE, IODA).
3. **NLP Phase:** Semantic scoring of event narratives (Sentence-Transformers).

### Four-Layer Operational Model (Prediction & Decision Support)
4. **Layer 1 - Binary Onset:** Two-Stage Hurdle Ensemble (9 thematic sub-models, class-balanced meta-learner)
5. **Layer 2 - Operational Tiering:** Quantile-based risk stratification (Critical 5% / High 15% / Elevated 30%)
6. **Layer 3 - Intensity Prediction:** Poisson regression with 500-fatality stability cap
7. **Layer 4 - Uncertainty Quantification:** BCCP 90% prediction intervals

### Analysis & Visualization
8. **Analysis Phase:** SHAP explanations, Pareto efficiency metrics, thesis-ready figures.

## Agentic Team Architecture (The Org Chart)
| Tier | Role | Model | Primary Task |
|:---|:---|:---|:---|
| **L3** | **Architect** | Opus 4.6 | High-level planning, spec writing, verification. |
| **L2** | **Manager** | Sonnet 4.5 | Orchestration, delegation, code review, synthesis. |
| **L1** | **Worker** | Haiku 4.5 | Data ingestion, exploration, documentation. |

### Operational Protocols
- **Collaboration:** Follow `.claude/rules/collaboration.md` (Intake -> Plan -> Delegate -> Implement -> Review).
- **Grounded Synthesis (MCP):** Follow `MCP/PROTOCOL.md`. **STRICT CONSTRAINT:** No web search. Verify local context first.
- **Code Guardrail:** L1 Workers use **Propose-and-Stop** (no direct code writes).
- **Gemini Escalation:** Gemini (L2) must use shell-outs for L3 planning to maintain state in `.claude/transcripts/L3_latest.md`.

## Memory Bank & Project State
- **Active Focus:** `CONTEXT.md` (Read first for current session status).
- **Task Tracking:** `TASK_QUEUE.md` (Kanban state).
- **Domain Library:**
    - `GLOSSARY.md`: Terminology and acronym mapping.
    - `DATA_DICTIONARY.md`: Schema, features, and source definitions.
    - `DECISIONS.md`: Architectural and modeling "Whys".
    - `FIGURES.md`: Figure-to-data mapping for thesis visuals.
- **Audit:** `final_audit.ini` (System dependency map).

## Shared Skills
- `/audit`: Perform deep data availability and temporal integrity checks.
- `/modeling`: Execute training, calibration, and prediction sub-pipelines.
- `/figures`: Regenerate thesis-ready visuals (SVG/PNG).
- `retrieve(query)`: Semantic retrieval via RAG MCP / shell.
