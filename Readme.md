# Spatial Randomization Inference for 2SLS (SE / p-values)

This repo is a **3-step pipeline** that produces **randomization-inference (RI) uncertainty** for fixed-effect 2SLS coefficients under a **spatially correlated placebo outcome model**.  
It is engineered to be **directional and restartable**: each step writes a concrete artifact that the next step consumes.

---

## 0) What You Get

After running all three scripts, you will have:

- **Observed 2SLS coefficients** (one row)
- **RI placebo coefficient draws** (many rows)
- **Plots + RI p-values** (one PNG per spec, optional combined PDF)

This repo does **not** run your final paper regressions; it produces an RI distribution you can use for inference.

---

## 1) Repo Layout (Architecture)

Place these scripts in the repo root:

- `step1_observed_2sls.py`  
  Computes observed 2SLS coefficients on the residualized dataset (no SE).

- `step2_spatial_randomization_inference.py`  
  Generates spatially correlated placebo outcomes and recomputes 2SLS across simulations.

- `step3_ri_visualization.py`  
  Compares observed vs placebo draws, computes RI p-values, and saves plots.

Recommended folders:

- `results/`  (all CSV outputs)
- `plots/`    (PNGs and optional PDF)

Data flow:

(residualized dataset) ──▶ Step 1 ──▶ observed_coefficients.csv
(residualized dataset) ──▶ Step 2 ──▶ coeff_ri.csv
observed_coefficients.csv + coeff_ri.csv ──▶ Step 3 ──▶ plots/


---

## 2) Dependencies

Python 3.9+.

Install:

pip install numpy pandas matplotlib scipy

Notes:
- SciPy is used for distance/kernel calculations (Step 2).
- KDE smoothing in Step 3 is optional; if SciPy KDE is unavailable it will still run.

---

## 3) Inputs You Must Provide

You need either:

A) A **residualized dataset CSV** (recommended), produced by your main pipeline  
OR  
B) Baseline + historical datasets + a selected-controls CSV (Step 2 supports merging + residualizing internally)

Minimum practical requirement for Step 1 and Step 3:
- Step 1 reads a residualized dataset and writes observed coefficients.
- Step 3 reads observed coefficients + placebo coefficients and makes plots.

Step 2 can run in a “full build” mode (merge + residualize) based on its CONFIG.

---

## 4) Step-by-Step

### Step 1 — Observed 2SLS (No SE)
Goal: compute the **observed** coefficients for the predefined partner-block specs.

What it does:
- Loads a residualized dataset
- Builds partner-block masks (e.g., Russia/China/USA/EU-block)
- Absorbs two-way FE using fast iterative demeaning
- Computes 2SLS coefficients with fixed IVs (no standard errors)
- Writes **one row** to CSV

Output:
- results/observed_coefficients.csv

Run:
python step1_observed_2sls.py

---

### Step 2 — Spatial Randomization Inference Engine
Goal: generate the **placebo distribution** of coefficients under spatial dependence.

Core model (implemented at group level):
- outcome = mean + spatial group shock + idiosyncratic noise
- group shocks are correlated via an exponential kernel on group centroids
- variance split (between-group vs within-group) is estimated from data
- for each simulation draw:
  - generate placebo outcomes
  - residualize placebo outcomes on controls
  - absorb two-way FE per spec
  - compute 2SLS coefficient (fast cached IV algebra)

Outputs:
- results/coeff_ri.csv          (S rows × specs columns; recreated per run)
- results/placebo_current.csv   (overwritten each batch; debugging artifact)

Run:
python step2_spatial_randomization_inference.py

---

### Step 3 — Visualization + RI p-values
Goal: produce plots and p-values comparing observed vs placebo coefficients.

What it does:
- Loads observed coefficients (Step 1)
- Loads placebo coefficients (Step 2)
- For each spec:
  - histogram / density of placebo distribution
  - vertical line for observed coefficient
  - two-sided RI p-value (share of |placebo| ≥ |observed|)
- Saves one PNG per spec, and optionally a combined PDF

Outputs:
- plots/<spec>_ri.png
- optional combined PDF (if enabled in CONFIG)

Run:
python step3_ri_visualization.py

---

## 5) How To Run (Minimal Recipe)

1) Open each script and edit the CONFIG block:
- input paths (datasets)
- output folder paths (results_dir / plots_dir)
- optional simulation size S

2) Run in order:

python step1_observed_2sls.py
python step2_spatial_randomization_inference.py
python step3_ri_visualization.py

---

## 6) Outputs and Restart Logic

- Step 1 overwrites/rewrites the observed CSV.
- Step 2 recreates coeff_ri.csv each run; placebo_current.csv is overwritten per batch.
- Step 3 overwrites plots each run.

If something fails, you can restart from the first missing artifact:
- no observed CSV → rerun Step 1
- no coeff_ri.csv → rerun Step 2
- no plots → rerun Step 3

---

## 7) What To Customize (Where)

- Change which coefficients/specs are produced:
  - edit the spec builder in Step 1 and Step 2

- Change spatial correlation model:
  - edit the group spatial engine in Step 2 (kernel, distance scale, variance split)

- Change FE structure:
  - edit fe keys in CONFIG (Step 1 and Step 2)

- Change simulation scale:
  - edit S and batch_size in Step 2 CONFIG

---

## License

MIT
