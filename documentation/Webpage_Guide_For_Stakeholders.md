# Synthetic Data Foundry - Stakeholder Guide

## What this webpage does

Synthetic Data Foundry helps teams:

1. Bring in real input data
2. Understand the data quality and structure
3. Generate synthetic (safe-to-share) data
4. Validate if the synthetic data is trustworthy
5. Export final outputs for business and model use

This is designed as an end-to-end workflow in one place.

---

## 3-part layout (easy view)

The page is organized into 3 clear blocks:

- **Gateway (Input)** - where users choose input type and provide data/prompt
- **Synthetic Data Foundry (Core Engine)** - where analysis and generation happen
- **Artifact (Output)** - where users export final dataset and final reports

---

## Gateway (Input)

Gateway controls how users want to create synthetic data:

- **Type of Input**
  - `Seed Data` - use uploaded files
  - `Natural Language` - use text prompt
  - `Both` - keep both options available

- **File Upload**
  - Supports CSV, Excel, and JSON
  - Used when Seed Data is selected

- **Natural Language Prompt**
  - Used when Natural Language is selected
  - Example: "Generate credit-risk style customer and loan behavior data"

- **Testing Demo Files**
  - Saves and loads fixed demo datasets for quick demonstrations
  - Useful for repeatable stakeholder walkthroughs

---

## Core Engine tabs

### 1) Lens
Purpose: **Understand the data quickly**

- Shows schema, column types, key statistics, and previews
- Highlights potentially sensitive fields
- Helps users verify that input looks correct before generation

### 2) Architect
Purpose: **Generate synthetic data**

- Uses Gateway input mode to decide generation path:
  - Seed Data path
  - Natural Language path
- Supports privacy/masking controls in seed-based path
- Produces synthetic output dataset

### 3) Critic
Purpose: **Quality-check synthetic output**

- Compares synthetic data against seed behavior
- Runs consistency/guardrail checks
- Highlights issues and confidence signals

### 4) Model Validation Sandbox
Purpose: **Business/model confidence check**

- Compares model behavior on real vs synthetic scenarios
- Helps determine whether synthetic data is suitable for downstream model testing

---

## Artifact (Output)

Artifact is the final handoff section.

- **Export final dataset (CSV)**
  - Download latest generated synthetic dataset

- **Export final report (JSON)**
  - Combined summary from Critic + Validation Sandbox

- **Help Me section**
  - In-app reference and downloadable support content

---

## Typical business demo flow (recommended)

1. Select input mode in Gateway
2. Upload or load demo files
3. Review Lens summary
4. Generate from Architect
5. Review quality in Critic
6. Validate confidence in Sandbox
7. Download final dataset and final report from Artifact

---

## Business value

- Faster synthetic data creation for pilots and testing
- Better governance through visible checks
- Easier stakeholder sign-off via consolidated outputs
- Reduced dependency on multiple disconnected tools

