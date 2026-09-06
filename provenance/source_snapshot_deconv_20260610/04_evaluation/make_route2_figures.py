#!/usr/bin/env python
"""Unified Route 2 comparison figures.
Fig1: HCA 5-fold  Route2 vs BayesPrism vs pure-prior, across the 5 unified-Pearson axes.
Fig2: GSE no-regression  Delta_Z(recorded) vs Route2, per axis, per dataset.
"""
import os, sys, glob, argparse, numpy as np, pandas as pd
import matplotlib;
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_ap = argparse.ArgumentParser(description="HCA 5-fold + GSE no-regression Route2 figures")
_ap.add_argument("--base", default="/disk1/maijl/deconv/deconv_20260605", help="dir holding result CSVs")
_ap.add_argument("--outdir", default=None, help="figure output dir (default <base>/figures_route2_20260606)")
_args, _ = _ap.parse_known_args()
BASE = _args.base
HCA = f"{BASE}/route2_otherfolds_20260606"      # result_fold_*.csv + bp_perfold.csv
GSE = BASE                                       # result_gse141115.csv / result_gse159585.csv (top level)
OUT = _args.outdir or f"{BASE}/figures_route2_20260606"; os.makedirs(OUT, exist_ok=True)
AXES = ["prof_all", "samp_all", "gene_all", "prof_mk", "gene_mk"]
AXLAB = ["profile\n(all)", "sample\n(all)", "gene-wise\n(all)", "profile\n(marker)", "gene-wise\n(marker)"]
C = {"pure-prior": "#9aa0a6", "Route2": "#c0392b", "BayesPrism": "#2c6fbb", "DeltaZ(rec)": "#7f8c8d"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.9, "figure.dpi": 140})


def load_concat(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)


# ---------------- Figure 1: HCA 5-fold ----------------
hca = load_concat(f"{HCA}/result_fold_*.csv")
bp = pd.read_csv(f"{HCA}/bp_perfold.csv")
hca = pd.concat([hca, bp], ignore_index=True)
methods1 = ["pure-prior", "Route2", "BayesPrism"]
fig, ax = plt.subplots(figsize=(8.4, 4.3))
x = np.arange(len(AXES)); wbar = 0.26
for k, mth in enumerate(methods1):
    sub = hca[hca.variant == mth]
    means = [sub[a].mean() for a in AXES]; sds = [sub[a].std() for a in AXES]
    pos = x + (k - 1) * wbar
    ax.bar(pos, means, wbar, yerr=sds, capsize=2.5, color=C[mth], label=mth,
           edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.8))
    for a_i, a in enumerate(AXES):  # per-fold dots
        vals = sub[a].values
        ax.scatter(np.full_like(vals, pos[a_i], dtype=float), vals, s=7, color="black", alpha=0.45, zorder=3, linewidths=0)
ax.set_xticks(x); ax.set_xticklabels(AXLAB); ax.set_ylim(0, 1.05)
ax.set_ylabel("Pearson correlation (median over cells/genes)")
ax.set_title("HCA (5-fold CV): Route 2 vs BayesPrism", fontweight="bold", loc="left")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
ax.axhline(0, color="black", lw=0.6)
for spine in ["left"]:
    ax.spines[spine].set_position(("outward", 3))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig1_hca_5fold_route2_vs_bp.{ext}", bbox_inches="tight")
plt.close(fig)

# win annotation table (Route2 vs BP per axis)
r2m = hca[hca.variant == "Route2"]; bpm = hca[hca.variant == "BayesPrism"]
wins = {a: int((r2m.sort_values('dataset')[a].values > bpm.sort_values('dataset')[a].values).sum()) for a in AXES}
print("HCA Route2 vs BP wins / 5 folds:", wins)


# ---------------- Figure 2: GSE no-regression ----------------
g1 = pd.read_csv(f"{GSE}/result_gse141115.csv")
g2 = pd.read_csv(f"{GSE}/result_gse159585.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, df, ttl, axes_use, axlab_use in [
    (axs[0], g1, "GSE141115 (per-sample×cell-type)", AXES, AXLAB),
    (axs[1], g2, "GSE159585 (cell-type level)", ["prof_all", "gene_all", "prof_mk", "gene_mk"],
     ["profile\n(all)", "gene-wise\n(all)", "profile\n(marker)", "gene-wise\n(marker)"])]:
    base = df[df.variant == "DeltaZ(rec)"].iloc[0]; r2 = df[df.variant == "Route2"].iloc[0]
    x = np.arange(len(axes_use)); wbar = 0.36
    ax.bar(x - wbar/2, [base[a] for a in axes_use], wbar, color=C["DeltaZ(rec)"], label="Delta_Z (recorded)", edgecolor="white", lw=0.5)
    ax.bar(x + wbar/2, [r2[a] for a in axes_use], wbar, color=C["Route2"], label="Route 2 (unified)", edgecolor="white", lw=0.5)
    for a_i, a in enumerate(axes_use):
        d = r2[a] - base[a]
        ax.annotate(f"{d:+.3f}", (a_i, max(base[a], r2[a]) + 0.015), ha="center", fontsize=8.5,
                    color=("#1a7a3a" if d >= 0 else "#b03030"), fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(axlab_use); ax.set_ylim(0, 1.05)
    ax.set_title(ttl, fontweight="bold", loc="left", fontsize=11)
    ax.axhline(0, color="black", lw=0.6)
axs[0].set_ylabel("Pearson correlation")
axs[1].legend(frameon=False, loc="lower right")
fig.suptitle("GSE no-regression: unified Route 2 generation ≥ recorded Delta_Z baseline", fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig2_gse_no_regression.{ext}", bbox_inches="tight")
plt.close(fig)
print("saved figures to", OUT)
