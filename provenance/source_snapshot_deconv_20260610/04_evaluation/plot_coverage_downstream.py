#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coverage -> downstream impact figure (deconv_20260605 vs deconv_20260610).

Visualises the single mechanism that links the proportion-inference strategy to
every cell-type-grouped downstream analysis: COVERAGE (which cell types receive
>=1 generated cell). Expression-prediction PCC is proportion-invariant, so the
only knob the solver turns downstream is coverage.

Panel A: per-sample cell-type coverage fraction (= per-sample covered types /
         total types), 4 solvers x 3 datasets. Numbers are VERBATIM from
         deconv_20260605/03_result_tables/unified/UNIFIED_PROPORTIONS_RESULTS.md
         section 7.5 (per-sample / union counts).

Panel B: downstream impact matrix, 7 analyses x 3 production regimes:
           - 605 softmax            (published tree, near-full coverage, no floor)
           - 610 prophead (no floor)   (run_scheme.sh: MIN="")
           - 610 nnls (floor=20)       (run_scheme.sh: --min_cells_per_type 20)
         Colour = green: full & faithful | amber: present but abundance floored |
                  red: cell types dropped / edges broken.

Self-contained (plain matplotlib, English labels to stay font-safe). Run:
    $PY 04_evaluation/plot_coverage_downstream.py
Outputs: deconv_20260610/coverage_downstream_impact.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---- Panel A data: VERBATIM per-sample / total cell types (UNIFIED §7.5) -------
DATASETS = ["HCA (23)", "GSE141115 (20)", "GSE159585 (62)"]
TOTAL    = np.array([23, 20, 62], dtype=float)
SOLVERS  = ["softmax", "dirichlet_map", "prophead", "nnls_core"]
# per-sample covered cell types, rows = solver, cols = dataset
PERSAMPLE = {
    "softmax":       np.array([23, 20, 62], dtype=float),
    "dirichlet_map": np.array([21, 18, 61], dtype=float),
    "prophead":      np.array([13, 14, 40], dtype=float),
    "nnls_core":     np.array([10,  3,  2], dtype=float),
}
FRAC = {s: PERSAMPLE[s] / TOTAL for s in SOLVERS}
SOLVER_COLORS = {
    "softmax":       "#4C78A8",
    "dirichlet_map": "#72B7B2",
    "prophead":      "#54A24B",
    "nnls_core":     "#E45756",
}

# ---- Panel B data: downstream impact matrix -----------------------------------
ANALYSES = ["Expression PCC*", "Ablation\n(attention/network)", "Generated UMAP",
            "Marker heatmap", "ssGSEA", "CellChat", "Monocle2"]
REGIMES  = ["605\nsoftmax", "610 prophead\n(no floor)", "610 nnls\n(floor=20)"]
G, Y, R = 0, 1, 2  # green / amber / red
STATUS = np.array([
    # 605 softmax | 610 prophead no-floor | 610 nnls floor=20
    [G, G, G],  # Expression PCC  (value proportion-invariant everywhere)
    [G, G, G],  # Ablation        (architecture-level, full data)
    [G, R, Y],  # Generated UMAP  (prophead drops clusters; nnls floor sizes distorted)
    [G, R, G],  # Marker heatmap  (per-type mean; abundance-independent)
    [G, R, G],  # ssGSEA          (per-(sample,celltype) mean; abundance-independent)
    [G, R, G],  # CellChat        (min.cells=2 drops missing types under prophead)
    [G, R, G],  # Monocle2        (missing progenitor -> wrong root under prophead)
], dtype=int)
STATUS_COLORS = {G: "#54A24B", Y: "#F0C000", R: "#E45756"}

# ------------------------------------------------------------------ figure ------
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "savefig.bbox": "tight"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                               gridspec_kw={"width_ratios": [1.15, 1.0]})

# ----- Panel A: grouped bars ----
x = np.arange(len(DATASETS))
nb = len(SOLVERS)
w = 0.80 / nb
for i, s in enumerate(SOLVERS):
    xs = x - 0.40 + w * (i + 0.5)
    bars = axA.bar(xs, FRAC[s], width=w, color=SOLVER_COLORS[s], label=s,
                   edgecolor="white", linewidth=0.5)
    for xi, di in zip(xs, range(len(DATASETS))):
        frac = FRAC[s][di]
        axA.text(xi, frac + 0.015,
                 f"{frac:.2f}\n{int(PERSAMPLE[s][di])}/{int(TOTAL[di])}",
                 ha="center", va="bottom", fontsize=6.6, color="#222")
axA.set_xticks(x); axA.set_xticklabels(DATASETS)
axA.set_ylim(0, 1.18); axA.set_yticks(np.arange(0, 1.01, 0.25))
axA.set_ylabel("per-sample cell-type coverage  (covered / total)")
axA.set_title("A  Coverage by proportion solver", loc="left", fontweight="bold")
axA.axhline(1.0, color="#999", lw=0.8, ls="--", zorder=0)
legA = axA.legend(title="solver", fontsize=8.5, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.10), frameon=False,
                  columnspacing=1.4, handlelength=1.3)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)

# ----- Panel B: status matrix ----
nr, nc = STATUS.shape
for r in range(nr):
    for c in range(nc):
        axB.add_patch(plt.Rectangle((c, nr - 1 - r), 1, 1,
                      facecolor=STATUS_COLORS[STATUS[r, c]],
                      edgecolor="white", linewidth=2))
axB.set_xlim(0, nc); axB.set_ylim(0, nr)
axB.set_xticks(np.arange(nc) + 0.5); axB.set_xticklabels(REGIMES, fontsize=9)
axB.set_yticks(np.arange(nr) + 0.5)
axB.set_yticklabels(ANALYSES[::-1], fontsize=9)
axB.xaxis.tick_top(); axB.xaxis.set_label_position("top")
axB.tick_params(length=0)
for sp in axB.spines.values():
    sp.set_visible(False)
axB.set_title("B  Zero-count impact on downstream", loc="left",
              fontweight="bold", pad=28)

legend_handles = [
    Patch(facecolor="#54A24B", label="full & faithful (unaffected)"),
    Patch(facecolor="#F0C000", label="present, abundance floored"),
    Patch(facecolor="#E45756", label="cell types dropped / edges broken"),
]
axB.legend(handles=legend_handles, frameon=False, fontsize=8,
           loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1,
           handlelength=1.2, labelspacing=0.3)

fig.suptitle("Proportion strategy → coverage → downstream  "
             "(605 softmax vs 610 prophead/nnls)", fontsize=13, fontweight="bold")
fig.text(0.012, -0.02,
         "*Expression PCC is proportion-invariant (per-(sample,celltype) rows are normalised, count cancels); "
         "under prophead-no-floor the PCC values are unchanged but fewer cell types are measurable.\n"
         "Coverage numbers verbatim from UNIFIED_PROPORTIONS_RESULTS.md §7.5. "
         "nnls's --min_cells_per_type 20 floor restores per-type downstream coverage but the floored abundances no longer reflect the (collapsed) proportions.",
         fontsize=6.8, color="#444", va="top")

fig.tight_layout(rect=[0, 0.02, 1, 0.95])
png = os.path.join(OUTDIR, "coverage_downstream_impact.png")
pdf = os.path.join(OUTDIR, "coverage_downstream_impact.pdf")
fig.savefig(png, dpi=300)
fig.savefig(pdf)
print(f"[saved] {png}")
print(f"[saved] {pdf}")
