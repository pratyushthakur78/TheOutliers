# Synthetic Data Foundry - Help Me

This guide explains what is built in the app, how each architecture block works, and where data flows internally.

## 1) Architecture Overview

The UI is organized into 3 clear parts:

- **Gateway (Input)** - source data and input mode control
- **Synthetic Data Foundry (Core Engine)** - Lens, Architect, Critic, Validation execution
- **Artifact (Output)** - final exports and reports

## 2) Gateway (Input) - What It Does

Gateway is the first control plane for all runs.

- **Type of Input**
  - `Seed Data` -> enables file upload, disables Natural Language prompt
  - `Natural Language` -> enables prompt, disables file upload
  - `Both` -> enables both
- **File Upload**
  - Supports CSV, XLSX/XLS, JSON
  - Uploaded files are parsed and stored in `st.session_state.data_registry`
  - Upload automatically routes user to Lens
- **Testing Demo Files**
  - `Save Demo Files` creates deterministic CSVs in `demo_files/`
  - `Load Demo Files` injects those files directly into Gateway registry
- **Generate via AI Astra**
  - Routes directly to Architect in natural language path

## 3) Core Engine Tabs

### Lens

- Profiles selected dataset (schema, types, statistics, distributions)
- Detects potentially sensitive columns
- Offers data and insights export

### Architect

Architect execution is driven by Gateway `Type of Input`.

- **Seed Data path**
  - Uses Gateway datasets only (no re-upload requirement)
  - Applies privacy masking/noise controls
  - Runs generation methods (CTGAN/TVAE/bootstrap/perturb/etc.)
  - Preserves binary indicators like `default_flag_12m` as strict 0/1
- **Natural Language path**
  - Reads prompt from Gateway prompt box
  - Generates tabular output via Azure/LLM flow

### Critic

- Compares seed vs synthetic quality
- Computes JS divergence and utility metrics
- Runs guardrails checks and correlation comparisons
- Exports critic report JSON

### Model Validation Sandbox

- Trains baseline on real split and compares synthetic-trained behavior on real holdout
- Includes sparse-class safeguards and empty-subset protections
- Exports ML alignment details into artifact report

## 4) Artifact (Output)

Artifact consolidates final outputs from current session:

- **Export final dataset (CSV)**
  - Uses latest available generated output
- **Export final report (JSON)**
  - Includes Critic outputs + Validation Sandbox outputs
- **Help Me**
  - This full guide is shown in-app and downloadable

## 5) Key Session State Objects

- `data_registry` -> uploaded/demo datasets
- `synthetic_df` -> Architect seed-based synthetic output
- `bot_generated_df` -> natural language generated output
- `critic_js_df`, `critic_utility_df`, `critic_guardrails_report` -> critic outputs
- `mv_bundle` -> validation sandbox result payload
- `gateway_input_mode`, `gateway_nl_prompt` -> gateway input controls

## 6) Demo Flow (Recommended for Presentations)

1. In Gateway, choose `Seed Data` or `Both`
2. Click `Load Demo Files`
3. Lens opens automatically and shows profile
4. Run Architect generation
5. Run Critic
6. Run Validation Sandbox
7. Export outputs from Artifact panel

## 7) Implementation Notes

- Main entry: `app.py`
- Shared modules:
  - `modules/data_processing.py`
  - `modules/synthetic_engine.py`
- Fixed demo files path: `demo_files/`
- Help file path: `HELP_ME.md`

