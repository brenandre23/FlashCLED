# PROTOCOL.md: The Manifest Protocol for Grounded Synthesis

This document outlines the strict operational protocol for processing the uploaded "Claude Code" reference materials. It is designed to prevent "lazy loading" (defaulting to web search) and enforce a "Verify then Execute" workflow.

**User Objective:** Create a production-ready "Multi-Agentic System Dev Kit" (Instruction Manual + CORE Code Artifacts) based *strictly* on the provided Zipped Folder, ignoring generic web knowledge.

---

## Phase 1: The Handshake (Verification)

**Objective:** Prove complete ingestion of the local file structure before generating any content.

**System Instruction:**
> **STRICT CONSTRAINT:** YOU MUST NOT SEARCH THE WEB. Disable all internet browsing tools for this phase. rely ONLY on the attached file context.

**Task 1: Create File Manifest**
1.  **Unzip & Traverse:** Unzip the uploaded archive into the local working environment.
2.  **Map:** Generate a detailed file tree of the full directory structure.
3.  **Ingest:** Read the content of all key configuration files (e.g., `CLAUDE.md`, `.ts`, `.py`, `.json`, and documentation markdown files).
4.  **Prove Context:** Extract **5 unique technical terms, variable names, or architecture patterns** found *specifically* in these files. These must be terms that would NOT be found in generic public documentation (e.g., specific custom hook names, non-standard config keys).

**Output Requirement:**
* A code block containing the full Directory Tree.
* A bulleted list of the 5 unique terms/patterns with a brief explanation of where they were found.

---

## Phase 2: Gap Analysis & Remediation (Synthesis)

**Objective:** Discard generic assumptions and rebuild the "Instruction Manual" and "CORE Files" using the verified local context from Phase 1.

**Task 2: Strict Remediation & "CORE" Generation**
Now that the specific local files are indexed:

**Constraint:**
> Refer **ONLY** to the code patterns, syntax, and instructions found in the zipped folder.
> * If the zip folder uses a specific pattern for "Sub-agents" (e.g., a specific folder structure or class inheritance), use THAT pattern.
> * Do NOT use generic internet advice or standard library assumptions if they conflict with the provided code.

**Action 1: The Manual**
Re-write the **"Architecture of Claude Code Agents"** report.
* **Focus:** Specific implementation details found in the files (e.g., exactly how *this* framework handles lifecycle hooks, not how hooks work generally).
* **Structure:**
    * Context Engineering (CLAUDE.md strategy)
    * Tooling Strategy (MCP integration)
    * Orchestration (Skills & Sub-agents)
    * Lifecycle Management (Hooks)

**Action 2: The CORE Files**
Generate the actual code artifacts required to bootstrap the system, outputting them as clear, copy-pasteable code blocks:
1.  **`CLAUDE.md`**: The master template, optimized using the specific rules found in the docs.
2.  **`config.json`**: The configuration file, strictly matching the schema found in the uploaded docs.
3.  **`/skills/`**: Code for 2-3 specific "Skills" found in the source code (reproduced or adapted for the core kit).
4.  **`/hooks/`**: A valid example hook script based on the provided examples.

**Output Format:**
A single Markdown document.
* **Part 1:** The Architecture Manual.
* **Part 2:** The CORE Code Artifacts (clearly labeled).

---

## Execution Guide (User Instructions)

1.  **Upload** the Zipped Folder containing the reference materials.
2.  **Copy & Paste** the "Phase 1" prompt below into the chat to initiate the handshake.
3.  **Verify** the output (File Tree + Unique Terms).
4.  **Copy & Paste** the "Phase 2" prompt below to generate the final kit.

### Prompt for Phase 1 (Copy/Paste this first):

```text
System Instruction:
YOU MUST NOT SEARCH THE WEB. Disable all internet browsing tools.

Task 1: Create File Manifest
I have uploaded a zipped folder. Your first task is to unzip this archive into your working environment and strictly read the local files.

1. List the full directory structure of the zipped folder so I can verify you see the files.
2. Ingest: Read the key configuration files (e.g., CLAUDE.md, specific .ts or .py files, and any documentation files).
3. Prove Context: Extract 5 unique technical terms, variable names, or architecture patterns found specifically in these files that would not be found in generic public documentation.

Output:
A detailed file tree and the list of 5 unique extracted terms.