# Synthetic Data Foundry - Module Guide

This project now follows a clearer module-first structure for the final product flow:

- `Lens`
- `Architect`
- `Critic`
- `Model Validation Sandbox`

## Why this structure

The app supports an end-to-end synthetic data lifecycle:

1. Profile and inspect source data
2. Generate synthetic output via Architect
3. Auto-check quality and guardrails in Critic
4. Validate downstream ML behavior in Sandbox

To keep this maintainable, common logic is separated from UI orchestration.

## Modules

### `app.py`

Main Streamlit entrypoint and screen composition.

- Sets page layout, routing, tab order
- Renders sidebar controls and top-level heading/tagline
- Coordinates state and tab-level workflow

### `modules/data_processing.py`

Reusable dataset and schema utilities.

- File parsing for CSV/XLSX/XLS/JSON
- Column statistics generation
- PII detection helpers
- Excel export helpers
- Seed upload parsing helpers

### `modules/synthetic_engine.py`

Synthetic generation and evaluation primitives.

- Privacy transforms and perturbation
- Bootstrap and SDV generation
- Binary-indicator preservation (keeps 0/1 flags strict)
- Seed-to-synthetic alignment and skew injection
- Fidelity metrics
- Guardrails checks and visualization helpers

## Architect input modes

Architect now exposes 3 input modes inside one unified tab:

1. `Seed Data`
2. `Natural Language`
3. `AI Astra`

## Relevant final UI options

The top navigation is intentionally limited to the final structure:

- `Lens`
- `Architect`
- `Critic`
- `Model Validation Sandbox`

This keeps the user journey focused on profiling, generation, validation, and export.

