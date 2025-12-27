#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomization Inference (group-level spatial kernel):
  • Merge baseline + historical (on 'oblast'); construct ym as Year_Month
  • Residualize on selected controls via adaptive Cholesky (ridge) + eigen fallback
  • Two-way FE absorption (ym, HS_Code_Group_2) via iterative demeaning
  • IVs fixed: dev_prcp_1931_02, dev_prcp_1932_01, dev_prcp_1933_07
  • EU partner flag
  • Spatial simulation (large-n friendly):
        y = μ + u_g + ε
        u_g ~ N(0, τ² K_G) with exp kernel on oblast centroids (Cholesky at group level)
        ε  ~ N(0, σ²), where τ² = between-oblast variance, σ² = pooled within-oblast variance (data-driven)
  • Outputs under CONFIG['results_dir']:
        - placebo_current.csv (overwritten per batch; 5 id cols + 2*batch_size placebo cols)
        - coeff_ri.csv        (recreated each run; S rows × 8 spec columns)
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ CONFIG                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
CONFIG = {
    # Input paths
    "baseline_path": r"C:/Research material/all trade data/import/baseline_import.csv",
    "hist_path":     r"C:/Research material/Historical data/oblast_weighted_averages.csv",
    "selected_controls_path": r"C:/Research material/all trade data/import/results/selected_controls.csv",

    # Output paths (RI output folder you set)
    "results_dir": r"C:/Research material/all trade data/import RI",
    "placebo_csv": "placebo_current.csv",
    "coef_csv":    "coeff_ri.csv",

    # Variables
    "y_var": "Import_USD",
    "w_var": "Net_Weight_kg",
    "d_var": "loss_per1000_33_34",

    # Fixed effects
    "fe1": "ym",                     # built from Year/Month if missing
    "fe2": "HS_Code_Group_2",

    # Instruments (fixed set)
    "IVS": ["dev_prcp_1931_02", "dev_prcp_1932_01", "dev_prcp_1933_07"],

    # Coordinates — preferred names; auto-detects if these differ
    "x_coord": "xcoord_y",
    "y_coord": "ycoord_y",

    # RI settings
    "S": 100,
    "batch_size": 5,
    "random_seed": 2025,

    # Numerics
    "use_float32": True,

    # Two-way FE absorption
    "absorb_tol": 1e-8,
    "absorb_max_iter": 50,
    "absorb_verbose": False,

    # d_resid rounding/grouping
    "round_decimals": 3,
    "minor_count_cut": 100,
    "major_count_cut": 1000,

    # Cholesky + eigen fallback for controls projection
    "ridge_start": 1e-8,
    "ridge_max": 1e+2,
    "ridge_grow": 10.0,
    "eig_tol": 1e-10,

    # Kernel jitter
    "jitter": 1e-6,

    # Logging
    "print_every": 10,
}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ IMPORTS                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import os
import numpy as np
import pandas as pd
from numpy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ UTILITIES                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def f32(a, on=True): return a.astype(np.float32) if (on and getattr(a, "dtype", None) != np.float32) else a

def downcast_df(df):
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif df[c].dtype == "int64":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

EU_MAIN_PARTNERS = {
    "Poland","Germany","Italy","Netherlands","France","Spain","Romania",
    "Hungary","Czech Republic","Lithuania"
}
def add_geo_flags(df):
    df["main_eu_partner"] = df["origin_country_name"].isin(EU_MAIN_PARTNERS).astype(np.int8)
    return df

def autodetect_coords(base, hist, want_x, want_y):
    cand_pairs = [
        (want_x, want_y),
        ("xcoord_y","ycoord_y"),
        ("X","Y"), ("x","y"),
        ("lon","lat"), ("longitude","latitude"),
        ("LONGITUDE","LATITUDE")
    ]
    def find_pair(src):
        for xc, yc in cand_pairs:
            if xc in src.columns and yc in src.columns:
                return xc, yc
        return None, None
    xb, yb = find_pair(base)
    xh, yh = find_pair(hist)
    if xb and yb: return xb, yb, "base"
    if xh and yh: return xh, yh, "hist"
    return None, None, None

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ MERGE                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def merge_panel(baseline_path, hist_path, y_var, w_var, d_var, selected_x, ivs, fe1, fe2, x_coord, y_coord):
    base = pd.read_csv(baseline_path)
    hist = pd.read_csv(hist_path)

    # Build ym like your Step-2
    if fe1 == "ym" and "ym" not in base.columns:
        if ("Year" not in base.columns) or ("Month" not in base.columns):
            raise KeyError("To construct 'ym', baseline must contain 'Year' and 'Month'.")
        base["ym"] = base["Year"].astype(str) + "_" + base["Month"].astype(str)

    # Coordinates (auto-detect)
    det_x, det_y, src = autodetect_coords(base, hist, x_coord, y_coord)

    base_need = [
        "oblast", "Year", "Month",
        fe1, fe2,
        "origin_country_name",
        "HS_Code_Position_4","HS_Code_Subposition_6","HS_Code_Category_8",
        y_var, w_var
    ]
    if src == "base":
        base_need += [det_x, det_y]

    hist_need = ["oblast", d_var] + selected_x + ivs
    if src == "hist":
        hist_need += [det_x, det_y]

    for c in base_need:
        if c not in base.columns: raise KeyError(f"Missing in baseline: {c}")
    for c in hist_need:
        if c not in hist.columns: raise KeyError(f"Missing in historical: {c}")

    df = pd.merge(base[base_need], hist[hist_need], on="oblast", how="left")
    df = add_geo_flags(df)

    # Standardize coordinate names in merged df to x_coord / y_coord if detected
    if src in ("base","hist"):
        if det_x != x_coord:
            df[x_coord] = pd.to_numeric(df[det_x], errors="coerce")
        if det_y != y_coord:
            df[y_coord] = pd.to_numeric(df[det_y], errors="coerce")
    if x_coord in df.columns and y_coord in df.columns:
        df[x_coord] = pd.to_numeric(df[x_coord], errors="coerce")
        df[y_coord] = pd.to_numeric(df[y_coord], errors="coerce")

    df = downcast_df(df)
    for c in [fe1, fe2, "origin_country_name"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ CONTROLS MATRIX                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def build_controls_matrix(df, selected_x, use32=True, const_tol=1e-12):
    Xraw = df[selected_x].to_numpy()
    col_var = Xraw.var(axis=0)
    keep = col_var > const_tol
    if not np.all(keep):
        dropped = [c for c, k in zip(selected_x, keep) if not k]
        print(f"[controls] Dropped {len(dropped)} constant cols: {dropped[:10]}{'...' if len(dropped)>10 else ''}")
    X1 = Xraw[:, keep]
    names1 = [c for c, k in zip(selected_x, keep) if k]

    # drop exact duplicate columns
    if X1.shape[1] > 1:
        rounded = np.round(X1, 12)
        seen, keep_idx = {}, []
        for j in range(X1.shape[1]):
            h = hash(rounded[:, j].tobytes())
            if h not in seen:
                seen[h] = j; keep_idx.append(j)
        if len(keep_idx) < X1.shape[1]:
            dropped_dups = [names1[j] for j in range(X1.shape[1]) if j not in keep_idx]
            print(f"[controls] Dropped {len(dropped_dups)} duplicate cols: {dropped_dups[:10]}{'...' if len(dropped_dups)>10 else ''}")
        X2 = X1[:, keep_idx]; names2 = [names1[j] for j in keep_idx]
    else:
        X2, names2 = X1, names1

    X2 = X2.astype(np.float32) if use32 else X2.astype(np.float64)
    return X2, names2

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SOLVERS (for control residualization)                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def xtx_cholesky_adaptive(X, ridge_start=1e-8, ridge_max=1e+2, grow=10.0):
    XtX = X.T @ X
    lam = ridge_start
    I = np.eye(XtX.shape[0], dtype=XtX.dtype)
    while lam <= ridge_max:
        try:
            L = np.linalg.cholesky(XtX + lam * I)
            if lam > ridge_start:
                print(f"[cholesky] Succeeded with ridge lambda={lam:.1e}")
            return L, lam
        except np.linalg.LinAlgError:
            lam *= grow
    print("[cholesky] Failed even with large ridge; will fall back to eigen solver.")
    return None, None

def gram_eigen_solver(X, eig_tol=1e-10):
    G = X.T @ X
    w, V = eigh(G)
    wmax = w.max()
    keep = w > (eig_tol * wmax)
    if keep.sum() == 0:
        raise np.linalg.LinAlgError("All eigenvalues ~0; controls degenerate.")
    w_kept, V_kept = w[keep], V[:, keep]
    inv_w = 1.0 / w_kept
    def solve_beta(XtY):
        alpha = V_kept.T @ XtY
        return V_kept @ ((inv_w[:, None]) * alpha)
    return solve_beta

def make_residualizer(X, ridge_start=1e-8, ridge_max=1e+2, grow=10.0, eig_tol=1e-10):
    L, _ = xtx_cholesky_adaptive(X, ridge_start, ridge_max, grow)
    if L is not None:
        def residualize_block(Yblock):
            XtY = X.T @ Yblock
            z = np.linalg.solve(L, XtY)
            B = np.linalg.solve(L.T, z)
            return Yblock - X @ B
        return residualize_block
    solve_beta = gram_eigen_solver(X, eig_tol=eig_tol)
    def residualize_block(Yblock):
        XtY = X.T @ Yblock
        B = solve_beta(XtY)
        return Yblock - X @ B
    return residualize_block

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ D RESID ROUND/GROUP                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def round_group_d(d_resid, decimals, minor_cut, major_cut):
    rounded = np.round(d_resid, decimals=decimals)
    vc = pd.Series(rounded).value_counts()
    minor = vc[vc < minor_cut].index
    major = vc[vc >= major_cut].index
    if len(major) == 0:
        return rounded.astype(np.float32)
    mapping = {val: major[np.argmin(np.abs(major - val))] for val in minor}
    return pd.Series(rounded).replace(mapping).to_numpy(dtype=np.float32)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ TWO-WAY FE ABSORPTION                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def demean_by_codes(y, codes, sizes):
    sums = np.bincount(codes, weights=y, minlength=sizes.size)
    means = sums[codes] / sizes[codes]
    return y - means

def absorb_twofe_fast(y, g1_codes, g1_sizes, g2_codes, g2_sizes, tol=1e-8, max_iter=50, verbose=False):
    y = y.astype(np.float64, copy=True)
    last = np.full_like(y, np.inf)
    for it in range(max_iter):
        y = demean_by_codes(y, g1_codes, g1_sizes)
        y = demean_by_codes(y, g2_codes, g2_sizes)
        delta = np.max(np.abs(y - last))
        if verbose:
            print(f"  absorb iter {it+1}: max Δ={delta:.3e}")
        if delta < tol: break
        last[:] = y
    return y.astype(np.float32)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ 2SLS CORE                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def cache_iv_eq(tilde_d, tilde_Z):
    Z = tilde_Z; d = tilde_d.reshape(-1,1)
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    Ztd = Z.T @ d
    denom = float((Ztd.T @ ZtZ_inv @ Ztd)[0,0])  # d' Pz d
    return {"Z": Z, "ZtZ_inv": ZtZ_inv, "Ztd": Ztd, "denom": denom}

def iv_beta_from_cache(tilde_y, cache):
    Z, ZtZ_inv, Ztd, denom = cache["Z"], cache["ZtZ_inv"], cache["Ztd"], cache["denom"]
    Zy = Z.T @ tilde_y.reshape(-1,1)
    num = float((Ztd.T @ ZtZ_inv @ Zy)[0,0])      # d' Pz y
    return num / denom

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SPECS                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
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
# ║ GROUP-LEVEL SPATIAL ENGINE (DATA-DRIVEN τ², σ²)                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def build_group_spatial_engine(y, df, x_coord, y_coord, jitter=1e-6, use32=True):
    """
    Build group-level spatial simulator:
        y_i = mu + u_{g(i)} + eps_i
        Var(u_g)   = τ² K_G  (exponential kernel on oblast centroids)
        Var(eps_i) = σ²      (pooled within-oblast variance)
    τ² = between-oblast variance; σ² = pooled within-oblast variance (both data-driven).
    """
    y = y.astype(np.float64)
    cat = pd.Categorical(df["oblast"])
    if (cat.codes < 0).any():
        raise ValueError("Missing 'oblast' values detected; cannot build group spatial engine.")
    g = cat.codes.astype(np.int64)
    G = int(g.max() + 1)
    n = y.shape[0]

    # Overall mean and group means
    mu = float(y.mean())
    counts = np.bincount(g, minlength=G).astype(np.float64)
    sum_y  = np.bincount(g, weights=y, minlength=G)
    mean_g = sum_y / counts  # shape (G,)

    # Data-driven variance split (per-observation scaling)
    tau2 = float(np.sum(counts * (mean_g - mu)**2) / n)  # between-oblast variance
    resid_within = y - mean_g[g]
    sigma2 = float(np.dot(resid_within, resid_within) / n)  # pooled within-oblast variance

    # Oblast centroids (align with category order)
    if (x_coord in df.columns) and (y_coord in df.columns):
        cent = df.groupby("oblast")[[x_coord, y_coord]].mean(numeric_only=True)
        cent = cent.reindex(cat.categories)  # align to category order
        cent = cent.to_numpy(dtype=np.float64)
        bad = ~np.isfinite(cent).all(axis=1)
        if bad.any():
            idx = np.arange(G, dtype=np.float64).reshape(-1,1)
            cent[bad, :] = np.hstack([idx[bad], np.zeros((bad.sum(), 1))])
    else:
        cent = np.column_stack([np.arange(G, dtype=np.float64), np.zeros(G, dtype=np.float64)])

    # Exponential kernel on group centroids
    if G > 1:
        Dg = squareform(pdist(cent))
        dvec = Dg[Dg > 0]
        theta = np.median(dvec) if dvec.size > 0 else 1.0
        Kg = np.exp(-Dg / theta)
        Sg = tau2 * Kg + jitter * np.eye(G)
    else:
        Sg = np.array([[max(tau2, 0.0)]], dtype=np.float64)

    # Cholesky at group level
    Lg = np.linalg.cholesky(Sg)

    return {
        "mode": "group_spatial",
        "mu": np.float32(mu) if use32 else mu,
        "group_codes": g.astype(np.int32),
        "L_group": f32(Lg, use32),
        "sigma": np.float32(np.sqrt(max(sigma2, 0.0))),
        "G": G,
        "tau2": tau2,
        "sigma2": sigma2
    }

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ MAIN                                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main(cfg=CONFIG):
    ensure_dir(cfg["results_dir"])

    # 1) Controls list (first column of selected_controls.csv)
    selected_x = pd.read_csv(cfg["selected_controls_path"]).iloc[:,0].tolist()

    # 2) Merge panel
    df = merge_panel(
        cfg["baseline_path"], cfg["hist_path"],
        cfg["y_var"], cfg["w_var"], cfg["d_var"],
        selected_x, cfg["IVS"],
        cfg["fe1"], cfg["fe2"],
        cfg["x_coord"], cfg["y_coord"]
    )
    n = df.shape[0]
    print(f"[merge] n={n:,} rows")

    # 3) Controls matrix & residualizer
    X, kept_names = build_controls_matrix(df, selected_x, use32=cfg["use_float32"])
    print(f"[controls] kept {X.shape[1]} controls")
    resid_block = make_residualizer(
        X,
        ridge_start=cfg["ridge_start"],
        ridge_max=cfg["ridge_max"],
        grow=cfg["ridge_grow"],
        eig_tol=cfg["eig_tol"]
    )

    # 4) Residualize treatment once + round/group
    d_vec = df[cfg["d_var"]].to_numpy().astype(np.float64)
    d_resid0 = resid_block(d_vec.reshape(-1,1)).ravel()
    d_resid  = round_group_d(d_resid0, cfg["round_decimals"], cfg["minor_count_cut"], cfg["major_count_cut"])

    # 5) IV matrix + GLOBAL residualization on controls (fix for shape mismatch)
    for iv in cfg["IVS"]:
        if iv not in df.columns: raise KeyError(f"IV not found: {iv}")
    Z = f32(df[cfg["IVS"]].to_numpy(), cfg["use_float32"])   # (n x k)
    Z_resid = resid_block(Z)                                  # (n x k) <-- residualize ONCE globally

    # 6) Specs & per-spec caches (slice Z_resid by mask; then FE absorb)
    specs = build_specs()
    per_spec = {}
    for spec_name, out_key, mask_fn in specs:
        mask = mask_fn(df)
        if not mask.any(): continue

        g1 = pd.Categorical(df.loc[mask, cfg["fe1"]]).codes
        g2 = pd.Categorical(df.loc[mask, cfg["fe2"]]).codes
        g1_sizes = np.bincount(g1); g2_sizes = np.bincount(g2)

        d_m = d_resid[mask]
        tilde_d = absorb_twofe_fast(d_m, g1, g1_sizes, g2, g2_sizes,
                                    tol=cfg["absorb_tol"], max_iter=cfg["absorb_max_iter"],
                                    verbose=cfg["absorb_verbose"])

        Zm_resid = Z_resid[mask, :]  # slice AFTER global residualization
        tilde_Z = np.column_stack([
            absorb_twofe_fast(Zm_resid[:, j], g1, g1_sizes, g2, g2_sizes,
                              tol=cfg["absorb_tol"], max_iter=cfg["absorb_max_iter"])
            for j in range(Zm_resid.shape[1])
        ])

        cache = cache_iv_eq(tilde_d, tilde_Z)
        per_spec[spec_name] = {"mask": mask, "g1": g1, "g2": g2,
                               "g1_sizes": g1_sizes, "g2_sizes": g2_sizes,
                               "cache": cache, "out_key": out_key}

    # 7) Build GROUP-LEVEL spatial engines (data-driven τ², σ²)
    y_engine = build_group_spatial_engine(df[cfg["y_var"]].to_numpy(), df, cfg["x_coord"], cfg["y_coord"],
                                          jitter=cfg["jitter"], use32=cfg["use_float32"])
    w_engine = build_group_spatial_engine(df[cfg["w_var"]].to_numpy(), df, cfg["x_coord"], cfg["y_coord"],
                                          jitter=cfg["jitter"], use32=cfg["use_float32"])
    print(f"[engine {cfg['y_var']}] mode={y_engine['mode']}, G={y_engine['G']}, tau2≈{y_engine['tau2']:.3g}, sigma2≈{y_engine['sigma2']:.3g}")
    print(f"[engine {cfg['w_var']}] mode={w_engine['mode']}, G={w_engine['G']}, tau2≈{w_engine['tau2']:.3g}, sigma2≈{w_engine['sigma2']:.3g}")

    # 8) Prepare coeff CSV (restart each run)
    coef_path = os.path.join(cfg["results_dir"], cfg["coef_csv"])
    coef_cols = [s for s,_,_ in specs]
    try:
        if os.path.exists(coef_path): os.remove(coef_path)
    except Exception as e:
        print(f"[warn] Could not remove existing {coef_path}: {e}")
    coef_df = pd.DataFrame(columns=coef_cols)
    coef_df.to_csv(coef_path, index=False)
    sims_done = 0

    # 9) RI loop
    S, B = int(cfg["S"]), int(cfg["batch_size"])
    rng_master = np.random.SeedSequence(cfg["random_seed"])
    child_seeds = rng_master.spawn(int(np.ceil(S / B)) + 3)

    placebo_path = os.path.join(cfg["results_dir"], cfg["placebo_csv"])
    print(f"Starting RI at sim={sims_done+1} to S={S} in batches of {B}...")

    while sims_done < S:
        b = min(B, S - sims_done)
        rng = np.random.default_rng(child_seeds[sims_done // B])

        def simulate_block(engine):
            # group-level Cholesky (GxG) + broadcast + nugget
            G = engine["G"]
            Zg = rng.standard_normal((G, b)).astype(np.float32)
            Sg = engine["L_group"] @ Zg                               # (G x b)
            group_part = Sg[engine["group_codes"], :]                 # broadcast (n x b)
            eps = rng.standard_normal((n, b)).astype(np.float32) * engine["sigma"]
            mu_vec = np.full((n, 1), engine["mu"], dtype=np.float32)
            return mu_vec + group_part + eps

        Y_block = simulate_block(y_engine)   # Import_USD placebos (n x b)
        W_block = simulate_block(w_engine)   # Net_Weight_kg placebos (n x b)

        # Overwrite placebo CSV with identifiers + current batch placebos
        placebo_df = pd.DataFrame({
            "oblast": df["oblast"].astype(str),
            cfg["fe1"]: df[cfg["fe1"]].astype(str),
            cfg["fe2"]: df[cfg["fe2"]].astype(str),
            "origin_country_name": df["origin_country_name"].astype(str),
            "main_eu_partner": df["main_eu_partner"].astype(np.int8),
        })
        for j in range(b):
            placebo_df[f"Y_p{j+1}"] = Y_block[:, j]
            placebo_df[f"W_p{j+1}"] = W_block[:, j]
        placebo_df.to_csv(placebo_path, index=False)

        # Residualize outcomes on controls
        Y_resid_block = resid_block(Y_block)  # (n x b)
        W_resid_block = resid_block(W_block)  # (n x b)

        # Compute 2SLS β per draw/spec
        new_rows = []
        for j in range(b):
            row = {}
            for spec_name, out_key, _ in specs:
                if spec_name not in per_spec:
                    row[spec_name] = np.nan
                    continue
                sc = per_spec[spec_name]
                mask = sc["mask"]; g1 = sc["g1"]; g2 = sc["g2"]
                g1_sizes = sc["g1_sizes"]; g2_sizes = sc["g2_sizes"]; cache = sc["cache"]

                yj = (Y_resid_block[:, j] if out_key == "y" else W_resid_block[:, j])[mask]
                tilde_y = absorb_twofe_fast(yj, g1, g1_sizes, g2, g2_sizes,
                                            tol=cfg["absorb_tol"], max_iter=cfg["absorb_max_iter"],
                                            verbose=cfg["absorb_verbose"])
                row[spec_name] = iv_beta_from_cache(tilde_y, cache)
            new_rows.append(row)

        coef_df = pd.concat([coef_df, pd.DataFrame(new_rows)], ignore_index=True)
        coef_df.to_csv(coef_path, index=False)

        sims_done += b
        if (sims_done % cfg["print_every"] == 0) or (sims_done == S):
            print(f"Progress: {sims_done}/{S} sims -> coeffs in {coef_path}; placebo overwritten at {placebo_path}")

    print("Done.")
    print(f"Final placebo file (overwritten per batch): {placebo_path}")
    print(f"Aggregated coefficients file: {coef_path} ({coef_df.shape[0]} rows × {coef_df.shape[1]} cols)")

if __name__ == "__main__":
    main()
