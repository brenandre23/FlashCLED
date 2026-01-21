# Documentation Update Instructions

**Date:** January 20, 2026
**Context:** "Predatory Ecosystem" NLP Refactor (v2.0)

This document outlines the specific text and tables that need to be added to the source files for `CEWP_Data_Source_Audit_v4.pdf` and `CEWP_Thesis_Overview.pdf` to reflect the latest codebase changes.

---

## 1. Updates for `CEWP_Data_Source_Audit_v4`

### Section: Data Dictionary / Feature Registry

**Action:** Add a new subsection titled **"NLP & Narrative Signals (v2.0)"** and insert the following table rows.

| Feature Name | Source | Method | Description | Frequency |
| :--- | :--- | :--- | :--- | :--- |
| **CrisisWatch Regime Pillars** | | | | |
| `regime_parallel_governance` | CrisisWatch | ConfliBERT Embedding | Semantic score for state substitution activities (taxation, illicit licensing). | Monthly |
| `regime_transnational_predation` | CrisisWatch | ConfliBERT Embedding | Semantic score for foreign resource extraction and sovereign nexus (Wagner, Midas). | Monthly |
| `regime_guerrilla_fragmentation` | CrisisWatch | ConfliBERT Embedding | Semantic score for rebel splintering, fluid alliances, and hit-and-run tactics. | Monthly |
| `regime_ethno_pastoral_rupture` | CrisisWatch | ConfliBERT Embedding | Semantic score for the breakdown of customary mediation and militarized transhumance. | Monthly |
| **Narrative Dynamics** | | | | |
| `narrative_velocity_lag1` | CrisisWatch | Cosine Distance | The rate of semantic change in reporting between month $t$ and $t-1$. Measures "narrative drift." | Monthly |
| **ACLED Hybrid Mechanisms** | | | | |
| `mech_gold_pivot` | ACLED | MiniLM + Regex | Detection of violence shifting from diamond zones to gold mining/dredging sites. | Weekly |
| `mech_predatory_tax` | ACLED | MiniLM + Regex | Detection of economic violence: illegal checkpoints, customs extortion, and road levies. | Weekly |
| `mech_factional_infighting` | ACLED | MiniLM + Regex | Detection of intra-rebel clashes (e.g., FPRC vs UPC) distinct from anti-state violence. | Weekly |
| `mech_collective_punishment` | ACLED | MiniLM + Regex | Detection of punitive expeditions and scorched-earth tactics against civilians. | Weekly |
| **Fusion Signals** | | | | |
| `gdelt_shock_signal` | GDELT + CW | Composite Index | A calculated "Shock" metric: $(AvgTone \times -1) - (RegimeRisk)$. High values indicate media panic exceeding structural risk. | Daily |
| `gdelt_border_buffer_flag` | GDELT | Spatial Buffer | Binary flag (1) indicating if an event originated in the 50km "spillover zone" of Chad or Sudan. | Daily |

---

## 2. Updates for `CEWP_Thesis_Overview`

### Section: Methodology / NLP Framework

**Action:** Replace or update the "Topic Modeling" section with the following **"Predatory Ecosystem Framework"** description.

#### The "Predatory Ecosystem" Framework (v2.0)
The NLP module has been refactored from generic topic modeling to a targeted detection system rooted in the political economy of conflict.

1.  **Regime Health Detection (CrisisWatch):**
    *   **Model:** `snowood1/ConfliBERT-scr-uncased` (specialized for conflict text).
    *   **Logic:** Reports are scored against four specific theoretical pillars: *Parallel Governance*, *Transnational Predation*, *Guerrilla Fragmentation*, and *Ethno-Pastoral Rupture*.
    *   **Innovation:** We calculate **Narrative Velocity**, a vector-based metric measuring how fast the conflict narrative is mutating, serving as a proxy for uncertainty.

2.  **Mechanism Detection (ACLED):**
    *   **Model:** `sentence-transformers/all-MiniLM-L6-v2`.
    *   **Logic:** A semi-supervised approach that projects event notes into a semantic space defined by specific tactical anchors (e.g., "artisanal gold mining" vs. "diamond mining"). This distinguishes between *types* of violence (economic vs. political) rather than just counting fatalities.

3.  **Border Porosity & Shock Signals:**
    *   **Spatial Buffering:** Ingestion now includes a 50km buffer zone into Chad and Sudan to capture cross-border spillover events (refugee flows, rebel safe havens) that are strictly outside CAR borders but operationally relevant.
    *   **The Shock Signal:** A fusion feature that mathematically isolates "panic." It subtracts the slow-moving "Regime Risk" (structural) from high-frequency "Media Tone" (tactical). A high positive residual indicates a sudden shock event that has not yet been absorbed by the structural narrative.

### Section: System Architecture

**Action:** Update the architecture diagram/description to include:
*   **Ingestion:** GDELT "Border Buffer" logic (CAR + 50km Neighbor Zone).
*   **Processing:** Separation of `process_crisiswatch.py` (Long-text embedding) and `process_acled_hybrid.py` (Short-text semantic search).
*   **Fusion:** The "Ghost Lookup" pattern in Feature Engineering, ensuring border cells calculate spatial lags using full transnational data before filtering for the model.
