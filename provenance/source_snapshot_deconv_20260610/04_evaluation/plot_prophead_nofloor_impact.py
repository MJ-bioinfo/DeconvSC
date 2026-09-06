#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimental impact of prophead WITHOUT --min_cells_per_type on downstream.
All numbers are MEASURED (see measure_prophead_nofloor_impact.py / measure_cellchat_edges.R /
measure_ssgsea_result_loss.py), hardcoded here for plotting.
Output: deconv_20260610/prophead_nofloor_downstream_impact.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCH = ["softmax\n(605, no floor)", "prophead\n(610, NO floor)", "nnls\n(610, floor=20)"]
COL = ["#4C78A8", "#54A24B", "#E45756"]
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight"})

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

def bars(a, vals, title, ylab, ref=None, fmt="{:.0f}", ann=None, ymax=None):
    x = np.arange(len(vals))
    b = a.bar(x, vals, color=COL, edgecolor="white", width=0.66)
    a.set_xticks(x); a.set_xticklabels(SCH, fontsize=9)
    a.set_title(title, fontweight="bold", loc="left")
    a.set_ylabel(ylab)
    if ref is not None:
        a.axhline(ref, ls="--", lw=0.9, color="#888", zorder=0)
    top = ymax if ymax else max(vals) * 1.22
    a.set_ylim(0, top)
    for xi, v in zip(x, vals):
        a.text(xi, v + top * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=9.5)
    if ann:
        for xi, txt in ann.items():
            a.text(xi, top * 0.5, txt, ha="center", va="center", fontsize=9,
                   color="#B00", fontweight="bold")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

# ---- A: COVID coverage (62 types) — per-sample mean ----
covA = ax[0, 0]
ps = [62.0, 46.1, 62.0]      # per-sample mean covered types
union = [62, 52, 62]          # union (covid∪normal)
x = np.arange(3); w = 0.38
covA.bar(x - w/2, ps, w, color=COL, edgecolor="white", label="per-sample mean")
covA.bar(x + w/2, union, w, color=COL, alpha=0.45, edgecolor="white", hatch="//", label="union (covid∪normal)")
covA.axhline(62, ls="--", lw=0.9, color="#888", zorder=0)
covA.set_xticks(x); covA.set_xticklabels(SCH, fontsize=9)
covA.set_ylim(0, 74); covA.set_ylabel("cell types covered  (of 62)")
covA.set_title("A  COVID coverage (cause)", fontweight="bold", loc="left")
for xi, v in zip(x - w/2, ps):
    covA.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
for xi, v in zip(x + w/2, union):
    covA.text(xi, v + 1, f"{v}", ha="center", va="bottom", fontsize=8.5)
covA.text(0.5, 30, "prophead drops 10–11 types;\n44% of kept groups ≤5 cells (min 1)",
          ha="center", fontsize=8.6, color="#B00", fontweight="bold")
covA.legend(frameon=False, fontsize=8, loc="lower left")
for sp in ("top", "right"):
    covA.spines[sp].set_visible(False)

# ---- B: ssGSEA testable cell types + result loss ----
bars(ax[0, 1], [62, 52, 62], "B  ssGSEA: testable cell types", "uniq cell types (GSVA rows)",
     ref=62, ymax=78,
     ann={1: "−10 types\n−20.5% of the 1242\npublished sig. assoc.\n(median 8 cells, min 1)"})

# ---- C: CellChat COVID, normalized to softmax=100% ----
cc = ax[1, 0]
metrics = ["nodes", "edges (>0)", "total\ninteractions"]
soft = np.array([62, 3844, 635852.0])
prop = np.array([51, 2209, 281381.0])
nn = np.array([62, 3844, 732802.0])
pct = np.vstack([soft/soft, prop/soft, nn/soft]) * 100
x = np.arange(len(metrics)); w = 0.26
for i in range(3):
    cc.bar(x + (i-1)*w, pct[i], w, color=COL[i], edgecolor="white")
cc.axhline(100, ls="--", lw=0.9, color="#888", zorder=0)
cc.set_xticks(x); cc.set_xticklabels(metrics, fontsize=9)
cc.set_ylim(0, 130); cc.set_ylabel("% of softmax (=100%)")
cc.set_title("C  CellChat COVID network", fontweight="bold", loc="left")
for i in range(3):
    for xi, v in zip(x + (i-1)*w, pct[i]):
        cc.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
cc.text(1.0, 118, "prophead: −43% edges, −56% interactions;  migDC & pDC (≈27k int each) GONE",
        ha="center", fontsize=8.0, color="#B00", fontweight="bold")
for sp in ("top", "right"):
    cc.spines[sp].set_visible(False)

# ---- D: Monocle2 / iMGL (5 stages) ----
md = ax[1, 1]
union5 = [5, 5, 5]; persmin = [5, 4, 5]
x = np.arange(3); w = 0.38
md.bar(x - w/2, union5, w, color=COL, edgecolor="white", label="union types")
md.bar(x + w/2, persmin, w, color=COL, alpha=0.45, edgecolor="white", hatch="//", label="per-sample min")
md.axhline(5, ls="--", lw=0.9, color="#888", zorder=0)
md.set_xticks(x); md.set_xticklabels(SCH, fontsize=9)
md.set_ylim(0, 6.4); md.set_ylabel("iMGL stages (of 5)")
md.set_title("D  Monocle2 / iMGL trajectory", fontweight="bold", loc="left")
for xi, v in zip(x - w/2, union5):
    md.text(xi, v + 0.08, f"{v}", ha="center", va="bottom", fontsize=8.5)
for xi, v in zip(x + w/2, persmin):
    md.text(xi, v + 0.08, f"{v}", ha="center", va="bottom", fontsize=8.5)
md.text(1.0, 2.4, "union 5/5 for ALL schemes →\ntrajectory intact (progenitor survives);\nonly per-sample sparsity",
        ha="center", fontsize=8.6, color="#0a0", fontweight="bold")
md.legend(frameon=False, fontsize=8, loc="lower right")
for sp in ("top", "right"):
    md.spines[sp].set_visible(False)

fig.suptitle("prophead WITHOUT --min_cells_per_type: measured downstream impact "
             "(COVID 62 types; iMGL 5 stages)", fontsize=13, fontweight="bold")
fig.text(0.012, -0.01,
         "Measured from the on-disk runs (softmax=deconv_20260605, prophead/nnls=deconv_20260610). "
         "ssGSEA loss = share of the published 1242 significant cell-type×pathway associations carried by the 10 dropped types. "
         "CellChat from covid_cellchat.rds (@net$count). Verdict: ssGSEA & CellChat materially hurt; Monocle2 (coarse 5-stage) robust.",
         fontsize=7, color="#444", va="top")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
png = os.path.join(OUTDIR, "prophead_nofloor_downstream_impact.png")
pdf = os.path.join(OUTDIR, "prophead_nofloor_downstream_impact.pdf")
fig.savefig(png, dpi=300); fig.savefig(pdf)
print(f"[saved] {png}"); print(f"[saved] {pdf}")
