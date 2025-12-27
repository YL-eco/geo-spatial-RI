#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 1 — Observed 2SLS on residualized dataset (no re-residualizing)
--------------------------------------------------------------------
Inputs:
  • residualized_dataset.csv  (must contain: y_resid, w_resid, d_resid,
                               ym, HS_Code_Group_2, origin_country_name,
                               and IV columns: dev_prcp_1931_02, dev_prcp_1932_01, dev_prcp_1933_07)

What it does:
  • Creates main_eu_partner flag (per your Stata rules)
  • For 8 specs (y/w × RU, CN, USA, main EU partners):
      - Absorb two-way FE: (ym, HS_Code_Group_2) by fast demeaning
      - Run 2SLS with fixed IVs (no SE)
  • Saves ONE ROW of observed coefficients to CSV.

Output:
  • observed_coefficients.csv   (columns: y_RUS, y_CHN, y_USA, y_EU, w_RUS, w_CHN, w_USA, w_EU)
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ CONFIG                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
CONFIG = {
    "residualized_path": r"",
    "results_dir":       r"",
    "observed_csv":      "observed_coefficients.csv",

    # Fixed effects (match ivreghdfe absorb(ym hs_code_group_2))
    "fe1": "ym",
    "fe2": "HS_Code_Group_2",

    # Columns (already residualized)
    "y_resid": "y_resid",
    "w_resid": "w_resid",
    "d_resid": "d_resid",

    # Fixed IVs (must be present in residualized CSV)
    "IVS": ["dev_prcp_1931_02", "dev_prcp_1932_01", "dev_prcp_1933_07"],
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ IMPORTS                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import os
import numpy as np
import pandas as pd

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ HELPERS                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
EU_MAIN_PARTNERS = {
    "Poland","Germany","Italy","Netherlands","France","Spain","Romania",
    "Hungary","Czech Republic","Lithuania"
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def add_main_eu_partner_flag(df):
    df["main_eu_partner"] = df["origin_country_name"].isin(EU_MAIN_PARTNERS).astype(np.int8)
    return df

# two-way FE absorption by iterative demeaning
def fe_codes_and_sizes(cat_series):
    codes = pd.Categorical(cat_series).codes
    sizes = np.bincount(codes)
    return codes, sizes

def demean_by_codes(vec, codes, sizes):
    sums = np.bincount(codes, weights=vec, minlength=sizes.size)
    means = sums[codes] / sizes[codes]
    return vec - means

def absorb_twofe_fast(vec, g1, g1_sizes, g2, g2_sizes, tol=1e-8, max_iter=50):
    y = vec.astype(np.float64, copy=True)
    last = np.full_like(y, np.inf)
    for _ in range(max_iter):
        y = demean_by_codes(y, g1, g1_sizes)
        y = demean_by_codes(y, g2, g2_sizes)
        delta = np.max(np.abs(y - last))
        if delta < tol:
            break
        last[:] = y
    return y.astype(np.float64)

# 2SLS with pre-absorbed variables (no intercept, no SE)
def cache_iv_eq(tilde_d, tilde_Z):
    Z = tilde_Z
    d = tilde_d.reshape(-1,1)
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    Ztd = Z.T @ d
    denom = float((Ztd.T @ ZtZ_inv @ Ztd)[0,0])  # d' Pz d
    return {"Z": Z, "ZtZ_inv": ZtZ_inv, "Ztd": Ztd, "denom": denom}

def iv_beta_from_cache(tilde_y, cache):
    Z, ZtZ_inv, Ztd, denom = cache["Z"], cache["ZtZ_inv"], cache["Ztd"], cache["denom"]
    Zy = Z.T @ tilde_y.reshape(-1,1)
    num = float((Ztd.T @ ZtZ_inv @ Zy)[0,0])      # d' Pz y
    return num / denom

def build_specs():
    return [
        ("y_RUS", "y", lambda df: (df["origin_country_name"] == "Russian Federation").to_numpy()),
        ("y_CHN", "y", lambda df: (df["origin_country_name"] == "China").to_numpy()),
        ("y_USA", "y", lambda df: (df["origin_country_name"] == "United States of America").to_numpy()),
        ("y_EU",  "y", lambda df: (df["main_eu_partner"] == 1).to_numpy()),
        ("w_RUS", "w", lambda df: (df["origin_country_name"] == "Russian Federation").to_numpy()),
        ("w_CHN", "w", lambda df: (df["origin_country_name"] == "China").to_numpy()),
        ("w_USA", "w", lambda df: (df["origin_country_name"] == "United States of America").to_numpy()),
        ("w_EU",  "w", lambda df: (df["main_eu_partner"] == 1).to_numpy()),
    ]

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ MAIN                                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main(cfg=CONFIG):
    ensure_dir(cfg["results_dir"])

    # 1) load residualized dataset
    path = cfg["residualized_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Residualized dataset not found: {path}")
    df = pd.read_csv(path)

    # minimal column checks
    needed = [cfg["y_resid"], cfg["w_resid"], cfg["d_resid"],
              cfg["fe1"], cfg["fe2"], "origin_country_name"] + cfg["IVS"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in residualized dataset: {missing}")

    # 2) add main EU partner flag
    df = add_main_eu_partner_flag(df)

    # 3) prefetch residualized columns
    y_resid = df[cfg["y_resid"]].to_numpy(dtype=np.float64)
    w_resid = df[cfg["w_resid"]].to_numpy(dtype=np.float64)
    d_resid = df[cfg["d_resid"]].to_numpy(dtype=np.float64)
    Z_all   = df[cfg["IVS"]].to_numpy(dtype=np.float64)

    # 4) specs
    specs = build_specs()
    observed = {}

    for spec_name, out_key, mask_fn in specs:
        mask = mask_fn(df)
        if not mask.any():
            observed[spec_name] = np.nan
            continue

        # FE absorption (within mask)
        g1_codes, g1_sizes = fe_codes_and_sizes(df.loc[mask, cfg["fe1"]])
        g2_codes, g2_sizes = fe_codes_and_sizes(df.loc[mask, cfg["fe2"]])

        # absorb d and Z
        tilde_d = absorb_twofe_fast(d_resid[mask], g1_codes, g1_sizes, g2_codes, g2_sizes)
        Zm = Z_all[mask, :]
        tilde_Z = np.column_stack([
            absorb_twofe_fast(Zm[:, j], g1_codes, g1_sizes, g2_codes, g2_sizes)
            for j in range(Zm.shape[1])
        ])

        # absorb outcome
        y_in = y_resid if out_key == "y" else w_resid
        tilde_y = absorb_twofe_fast(y_in[mask], g1_codes, g1_sizes, g2_codes, g2_sizes)

        # 2SLS beta (no SE)
        cache = cache_iv_eq(tilde_d, tilde_Z)
        beta = iv_beta_from_cache(tilde_y, cache)
        observed[spec_name] = beta

    # 5) write one-row CSV
    out_file = os.path.join(cfg["results_dir"], cfg["observed_csv"])
    col_order = [s for s,_,_ in specs]
    pd.DataFrame([observed])[col_order].to_csv(out_file, index=False)

    print(f"[Step 1] Wrote observed coefficients → {out_file}")
    print(pd.DataFrame([observed])[col_order].to_string(index=False))

if __name__ == "__main__":
    main()

