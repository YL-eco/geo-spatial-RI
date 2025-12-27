# Spatial Randomization Inference for 2SLS (SE / p-values)

This repository provides a **three-step randomization inference (RI) workflow** to obtain **sampling uncertainty for 2SLS coefficients** under a **spatially correlated outcome model**. It is built to be **large-N friendly** and to match common two-way fixed-effect IV specifications by using fast absorption (iterative demeaning) rather than heavyweight regression frameworks.

The pipeline is intentionally **modular**: each step produces a concrete output artifact that the next step consumes.

---

## Repository Architecture

step1_observed_2sls.py
step2_spatial_randomization_inference.py
step3_ri_visualization.py

results/ # RI coefficient draws + observed coefficients
plots/ # histograms / density plots + optional combined PDF


**Data flow**

Residualized dataset (from your main IV pipeline)
│
▼
Step 1: observed coefficients (one row)
│
▼
Step 2: placebo coefficient draws (many rows)
│
▼
Step 3: plots + RI p-values


---

## What This Repo Does (and Doesn’t)

### ✅ Does
- Compute **observed 2SLS coefficients** for multiple partner-block specifications
- Simulate **spatially correlated placebo outcomes** using a **group-level spatial kernel**
- Recompute 2SLS across simulations to form an **RI coefficient distribution**
- Produce **RI p-values** and publication-ready **distribution plots**

### ❌ Does not
- Estimate standard errors via asymptotic formulas (this is RI, not sandwich SE)
- Choose instruments or controls (inputs are assumed fixed)
- Rebuild residuals from scratch unless needed by the RI engine (Step 2 handles internal residualization for simulated outcomes)

---

## Dependencies

Python 3.9+ recommended.

Install:
```bash
pip install numpy pandas matplotlib scipy

Notes:

Step 2 uses SciPy for distance/kernel utilities.

Step 3 optionally uses SciPy for KDE smoothing (falls back gracefully if unavailable).

Inputs

This repo expects you already have (from a separate preprocessing/selection pipeline):

A residualized dataset (CSV) containing:

residualized outcome and weight variables

residualized treatment variable

fixed-effect identifiers

partner identifiers / flags needed for spec masks

the fixed instrument columns

Additionally, Step 2 can optionally merge from baseline + historical sources if you prefer to construct the working panel internally (it is written to support that mode).

Step 1 — Observed 2SLS (No SE)

Purpose: Compute the observed 2SLS coefficients on the residualized dataset (single run).

What it does

Loads the residualized dataset

Builds partner-block masks (e.g., Russia / China / USA / EU block)

Absorbs two-way FE using fast iterative demeaning

Computes 2SLS coefficients with fixed IVs (no standard errors)

Writes one-row CSV of observed coefficients

Output

observed_coefficients.csv (one row; one column per specification)

Run

python step1_observed_2sls.py

Step 2 — Spatial Randomization Inference Engine

Purpose: Generate the placebo distribution of coefficients under a spatial dependence model.

Core idea
The placebo outcomes are simulated from a group-level spatial process:

outcome = mean + group spatial component + idiosyncratic noise

spatial component is correlated across groups via an exponential kernel on group centroids

variance is split into between-group (τ²) and within-group (σ²), estimated from the data

What it does

Builds / loads the working dataset (merge + FE ids + partner flags)

Residualizes placebo outcomes against selected controls (internal residualizer)

Absorbs two-way FE for each spec mask

Computes 2SLS coefficients using cached IV objects for speed

Writes:

a coefficient draw matrix (S rows × #specs columns)

a “current placebo batch” file (overwritten each batch) for debugging/inspection

Outputs

coeff_ri.csv (appended/recreated per run; RI draws)

placebo_current.csv (overwritten each batch; quick sanity checks)

Run

python step2_spatial_randomization_inference.py

Step 3 — Visualization + RI p-values

Purpose: Convert RI draws into interpretable inference visuals.

What it does

Loads:

observed coefficients (Step 1)

placebo coefficients (Step 2)

For each specification:

plots placebo distribution

overlays observed coefficient line

computes two-sided RI p-value

optionally overlays KDE / Normal fit

Saves one PNG per spec + optional combined PDF

Outputs

plots/<spec>_ri.png

optional combined PDF (all panels)

Run

python step3_ri_visualization.py

How to Use (Minimal)

Set paths in each script’s CONFIG block (input + output directories).

Run the three steps in order:

python step1_observed_2sls.py
python step2_spatial_randomization_inference.py
python step3_ri_visualization.py

Notes for Extension

Common modifications are intentionally localized:

Change the set of specifications: edit the spec builder function (mask logic)

Change FE structure: update the two FE keys in config

Change spatial dependence: replace kernel or distance scale logic in the group engine

Increase simulation power: increase S and adjust batch_size for memory/throughput

Outputs You Should Commit vs Ignore

Recommended to commit:

observed_coefficients.csv

plots or PDF outputs you cite in paper drafts

Recommended to ignore (gitignore):

placebo_current.csv (debug artifact; overwritten)

large coeff_ri.csv if S is large (consider compressing or storing externally)

License

MIT

