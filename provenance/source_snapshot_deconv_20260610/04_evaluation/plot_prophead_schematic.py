#!/usr/bin/env python
"""Schematic of the prophead cell-type proportion estimator (paper methods figure).
Training (top): reference -> pi_ref Dirichlet -> simulate labelled pseudo-bulks -> PropHead MLP
-> softmax, optimised by CE(p||q) with input-noise augmentation. Inference (bottom): query bulk
-> same core feature -> frozen head -> proportions f_s -> optional simplex refinement vs the
anchored basis. Arial + EMF via lib/plot_style. Output -> /disk1/maijl/deconv/manuscript/."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import plot_style  # Arial on import + plot_style.save (png/pdf/svg/emf)

OUT = os.environ.get("SCHEMATIC_OUT", "/disk1/maijl/deconv/manuscript")
os.makedirs(OUT, exist_ok=True)

# palette (consistent with the DeconvSC figures)
C_DATA = ("#DCE7EE", "#5E86A0")   # reference / simulate (blue)
C_PRIOR = ("#EAE3F3", "#8A6FB0")  # pi_ref (purple)
C_HEAD = ("#DEBBB1", "#B5651D")   # PropHead (DeconvSC pink)
C_LOSS = ("#F3F0EA", "#9A9A9A")   # training objective
C_INF = ("#E6EFE6", "#5E8A5E")    # inference (green)
C_OUT = ("#F6E9D8", "#C08A3E")    # outputs


def box(ax, x0, y0, w, h, text, fc, ec, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=("bold" if bold else "normal"), zorder=5)


def arrow(ax, xy0, xy1, label=None, ls="-", color="#333333", rad=0.0, lfs=8.5, lpad=0.12):
    ax.annotate("", xy=xy1, xytext=xy0,
                arrowprops=dict(arrowstyle="-|>", lw=1.7, color=color, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2), zorder=4)
    if label:
        mx, my = (xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2
        ax.text(mx, my, label, fontsize=lfs, ha="center", va="center", color=color, zorder=6,
                bbox=dict(boxstyle=f"round,pad={lpad}", fc="white", ec="none", alpha=0.9))


fig, ax = plt.subplots(figsize=(13.5, 7.4))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.4); ax.axis("off")

# ---- TRAINING (top, vertical left column) ----
ax.text(2.1, 7.2, "Training (simulation-supervised)", ha="center", fontsize=11, fontweight="bold", color=C_DATA[1])
box(ax, 0.4, 6.0, 3.4, 0.95, "Reference scRNA-seq\n(donors $\\times$ cell types)", *C_DATA)
box(ax, 0.4, 4.45, 3.4, 0.95, "$\\pi_{\\mathrm{ref}}$: Dirichlet prior\nmoment-match   $a_t = s\\,m_t$", *C_PRIOR)
box(ax, 0.4, 2.5, 3.4, 1.3,
    "Simulate $K{=}200$ pseudo-bulks / donor\n$b=\\sum_t p_t\\,\\bar{x}_{d,t}$,   $p\\sim\\mathrm{Dir}(a)$\n"
    "CP10K $\\to$ log1p $\\to$ core $= x$", *C_DATA, fs=9.8)
arrow(ax, (2.1, 6.0), (2.1, 5.42), "per-type fractions $m_t,v_t$")
arrow(ax, (2.1, 4.45), (2.1, 3.82), "sample  $p\\sim\\mathrm{Dir}(a)$")

# ---- PropHead (centre) ----
box(ax, 5.3, 3.55, 3.7, 1.35,
    "PropHead (MLP)\n$x \\to 512 \\to 256 \\to T$\nsoftmax $\\to q \\in \\Delta^{T}$", *C_HEAD, fs=10.5, bold=False)
arrow(ax, (3.8, 3.15), (5.3, 4.0), "labelled  $(x,\\,p)$", rad=-0.12)

# ---- training objective (top right) ----
box(ax, 5.3, 5.55, 3.7, 1.15,
    "Objective:  $\\min\\ \\mathrm{CE}(p\\,\\|\\,q)$\n$=-\\sum_t p_t\\log q_t$\n"
    "+ input noise  $\\sigma{=}0.2$  (Scaden/DISSECT)", *C_LOSS, fs=9.6)
arrow(ax, (6.6, 4.9), (6.6, 5.55), "$q$")
arrow(ax, (7.7, 5.55), (7.7, 4.9), "update", ls=(0, (4, 3)), color="#8A6FB0")

# ---- INFERENCE (bottom, horizontal) ----
ax.text(2.1, 1.95, "Inference (head frozen)", ha="center", fontsize=11, fontweight="bold", color=C_INF[1])
box(ax, 0.4, 0.55, 1.95, 0.95, "Query bulk\n$b_s$", *C_INF)
box(ax, 2.75, 0.55, 2.75, 0.95, "core feature  $x_s$\nCP10K$\\to$log1p$\\to$core", *C_INF, fs=9.6)
box(ax, 9.5, 0.55, 1.9, 0.95, "proportions\n$f_s$", *C_OUT)
box(ax, 11.65, 0.4, 1.7, 1.25,
    "optional\nsimplex refine\n$\\min\\,\\|\\sum_t f_t\\hat{e}_t-b_s\\|$\n(20-step EG)", *C_OUT, fs=8.4)
arrow(ax, (2.35, 1.02), (2.75, 1.02))
arrow(ax, (5.5, 1.1), (6.7, 3.55), "frozen head", rad=0.12)
arrow(ax, (8.0, 3.55), (9.9, 1.5), "$f_s=\\mathrm{softmax}(q)$", rad=0.12)
arrow(ax, (11.4, 1.02), (11.65, 1.02))

ax.text(6.75, 7.15, "prophead: simulation-trained cell-type proportion head",
        ha="center", fontsize=12.5, fontweight="bold")
plt.tight_layout()
plot_style.save(fig, os.path.join(OUT, "prophead_schematic"), dpi=600)
plt.close()
print(f"[saved] {OUT}/prophead_schematic.{{png,pdf,svg,emf}}")
