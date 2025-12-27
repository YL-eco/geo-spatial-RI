#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomization Inference Visualization (labeled)
-----------------------------------------------
Reads:
  1) observed CSV (Step 1) — one row of observed coefficients
  2) coeff_ri.csv (Step 2) — placebo coefficients (many rows)

Outputs:
  • PNG plots for each spec in `plots_dir`
  • Optional combined PDF with all panels
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional KDE (uses SciPy if available)
try:
    from scipy.stats import gaussian_kde
    _HAVE_KDE = True
except Exception:
    _HAVE_KDE = False

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ CONFIG                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
CONFIG = {
    # Adjust if your files live elsewhere
    "observed_csv": r"C:/Research material/all trade data/import RI/observed_coefficients.csv",
    "placebo_csv":  r"C:/Research material/all trade data/import RI/coeff_ri.csv",
    "plots_dir":    r"C:/Research material/all trade data/import RI/ri_plots",

    # Plot settings
    "bins": 50,
    "figsize": (7.5, 4.5),
    "dpi": 150,
    "show_plot": False,
    "draw_kde": True,           # requires SciPy (falls back if missing)
    "overlay_normal": False,    # overlay Normal(μ,σ) of placebos
    "alpha_hist": 0.70,

    # Combined PDF
    "make_pdf": True,
    "pdf_path": r"C:/Research material/all trade data/import RI/ri_plots/ri_all_specs.pdf",
}

# Pretty names for countries/blocks
COUNTRY_LABEL = {
    "RUS": "Russia",
    "CHN": "China",
    "USA": "United States",
    "EU" : "European Union",
}

def parse_spec_label(colname: str):
    """
    Map y_RUS -> ('USD', 'Russia', 'Coefficient (USD)', 'USD with Russia (y_RUS)')
        w_EU  -> ('Weight (kg)', 'European Union', 'Coefficient (kg)', 'Weight (kg) with European Union (w_EU)')
    """
    try:
        prefix, code = colname.split("_", 1)
    except ValueError:
        # Fallback if unexpected name
        return ("Value", colname, "Coefficient", f"{colname}")

    if prefix.lower() == "y":
        var_label = "USD"
        xlab = "Coefficient (USD)"
    elif prefix.lower() == "w":
        var_label = "Weight (kg)"
        xlab = "Coefficient (kg)"
    else:
        var_label = "Value"
        xlab = "Coefficient"

    partner = COUNTRY_LABEL.get(code.upper(), code.upper())
    title = f"{var_label} with {partner} ({colname})"
    return (var_label, partner, xlab, title)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ HELPERS                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def ri_p_value(placebos: np.ndarray, observed: float) -> float:
    """Two-sided RI p-value."""
    abs_obs = abs(observed)
    return float(np.mean(np.abs(placebos) >= abs_obs))

def _overlay_kde(ax, x, nbins=200):
    if not _HAVE_KDE:
        return
    try:
        x = x[~np.isnan(x)]
        if x.size < 5:
            return
        kde = gaussian_kde(x)
        xs = np.linspace(np.min(x), np.max(x), nbins)
        ys = kde(xs)
        ax.plot(xs, ys, linewidth=2, label="KDE")
    except Exception:
        pass

def _overlay_normal(ax, x, nbins=200):
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return
    xs = np.linspace(np.nanmin(x), np.nanmax(x), nbins)
    from math import sqrt, pi, exp
    ys = (1.0/(sd*sqrt(2*pi))) * np.exp(-0.5*((xs-mu)/sd)**2)
    ax.plot(xs, ys, linestyle="--", linewidth=1.5, label="Normal (μ,σ)")

def _numeric_common_columns(obs_df: pd.DataFrame, plc_df: pd.DataFrame):
    # keep columns that appear in both and are numeric in both
    common = [c for c in obs_df.columns if c in plc_df.columns]
    keep = []
    for c in common:
        if pd.api.types.is_numeric_dtype(obs_df[c]) and pd.api.types.is_numeric_dtype(plc_df[c]):
            keep.append(c)
    return keep

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ MAIN                                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main(cfg=CONFIG):
    os.makedirs(cfg["plots_dir"], exist_ok=True)

    # Load data
    obs_df = pd.read_csv(cfg["observed_csv"])
    plc_df = pd.read_csv(cfg["placebo_csv"])

    # Ensure columns match and are numeric
    cols = _numeric_common_columns(obs_df, plc_df)
    if not cols:
        raise ValueError("No matching numeric coefficient names between observed and placebo CSVs.")

    # Prepare PDF if requested
    pdf = None
    if cfg["make_pdf"]:
        from matplotlib.backends.backend_pdf import PdfPages
        os.makedirs(os.path.dirname(cfg["pdf_path"]), exist_ok=True)
        pdf = PdfPages(cfg["pdf_path"])

    print(f"Matched specs ({len(cols)}): {', '.join(cols)}")

    for col in cols:
        # Pull observed and placebos
        try:
            observed_val = float(obs_df[col].iloc[0])
        except Exception:
            raise ValueError(f"Observed CSV must have one row; couldn't read value for '{col}'.")
        placebos = pd.to_numeric(plc_df[col], errors="coerce").dropna().to_numpy()

        if placebos.size == 0:
            print(f"[warn] No placebo values for {col}; skipping.")
            continue

        # Labels
        _, _, xlab, title = parse_spec_label(col)

        # Stats
        pval = ri_p_value(placebos, observed_val)
        qlo, qhi = np.quantile(placebos, [0.025, 0.975])
        mean_pl, sd_pl = np.mean(placebos), np.std(placebos, ddof=1)

        # Plot
        fig, ax = plt.subplots(figsize=cfg["figsize"])
        ax.hist(placebos, bins=cfg["bins"], density=True, alpha=cfg["alpha_hist"],
                edgecolor="k", label="Placebo density")

        # Optional smooth overlays
        if cfg["draw_kde"]:
            _overlay_kde(ax, placebos)
        if cfg["overlay_normal"]:
            _overlay_normal(ax, placebos)

        # Observed & RI band
        ax.axvline(observed_val, color="red", linestyle="--", linewidth=2,
                   label=f"Observed = {observed_val:.3g}")
        ax.axvline(qlo, color="gray", linestyle=":", linewidth=1.5, label="2.5% / 97.5% RI band")
        ax.axvline(qhi, color="gray", linestyle=":", linewidth=1.5)

        # Labels
        ax.set_title(f"{title} — RI p = {pval:.3f}")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Density")
        ax.legend()

        # Save
        out_png = os.path.join(cfg["plots_dir"], f"{col}_ri.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=cfg["dpi"])
        if cfg["show_plot"]:
            plt.show()
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

        print(f"Saved: {out_png}  |  p = {pval:.3f}  |  mean={mean_pl:.3g}, sd={sd_pl:.3g}, "
              f"RI95%=[{qlo:.3g}, {qhi:.3g}]")

    if pdf is not None:
        pdf.close()
        print(f"Combined PDF saved: {cfg['pdf_path']}")

    print("All RI plots generated.")

if __name__ == "__main__":
    main()
