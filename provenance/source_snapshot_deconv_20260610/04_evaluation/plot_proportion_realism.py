#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Proportion-realism comparison figure (numbers from proportion_realism_scorecard.py).
Output: deconv_20260610/proportion_realism.{png,pdf}"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
M = ["softmax", "prophead\n(±floor: same p)", "nnls_core"]
C = ["#4C78A8", "#54A24B", "#E45756"]
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight"})
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

def lab(a, xs, vals, fmt="{:.2f}", dy=0.01, fs=9.5):
    for x, v in zip(xs, vals):
        a.text(x, v + dy, fmt.format(v), ha="center", va="bottom", fontsize=fs)

# A: held-out GT accuracy (only quantitative truth)
a = ax[0, 0]; x = np.arange(3)
pcc = [0.724, 0.774, 0.617]
a.bar(x, pcc, color=C, width=0.6, edgecolor="white")
a.set_xticks(x); a.set_xticklabels(M, fontsize=9); a.set_ylim(0, 0.9)
a.set_ylabel("held-out pseudobulk PCC (vs known GT)")
a.set_title("A  Quantitative accuracy on held-out GT", fontweight="bold", loc="left")
lab(a, x, pcc)
for xi, ep in zip(x, [31.6, 13.5, 4.2]):
    a.text(xi, 0.05, f"eff_pred\n{ep}", ha="center", fontsize=8, color="#333")
a.text(1, 0.84, "GT eff#types = 9.6  →  prophead closest", ha="center", fontsize=8.5,
       color="#0a0", fontweight="bold")
for s in ("top", "right"): a.spines[s].set_visible(False)

# B: real-bulk composition diversity (eff, inverse-Simpson) + identity
a = ax[0, 1]; x = np.arange(3)
eff = [20.3, 9.6, 1.6]
a.bar(x, eff, color=C, width=0.6, edgecolor="white")
a.axhline(9.6, ls="--", lw=1.0, color="#0a0"); a.text(2.35, 10.2, "realistic\n(GT 9.6)", color="#0a0", fontsize=8)
a.set_xticks(x); a.set_xticklabels(M, fontsize=9); a.set_ylim(0, 24)
a.set_ylabel("real-bulk eff#types (inverse-Simpson)")
a.set_title("B  Composition diversity + sample identity", fontweight="bold", loc="left")
lab(a, x, eff, fmt="{:.1f}", dy=0.3)
for xi, idv in zip(x, [36, 64, 21]):
    a.text(xi, eff[["softmax","prophead\n(±floor: same p)","nnls_core"].index(M[int(xi)])] if False else eff[int(xi)]/2,
           f"identity\n{idv}%", ha="center", fontsize=8.2, color="white", fontweight="bold")
a.text(1, 22, "softmax over-uniform · nnls collapse · prophead ≈ GT", ha="center", fontsize=8, color="#333")
for s in ("top", "right"): a.spines[s].set_visible(False)

# C: dominant-type plausibility (overlap with known lung-dominant set, of top-8)
a = ax[1, 0]; x = np.arange(3)
ov = [3, 7, 3]
a.bar(x, ov, color=C, width=0.6, edgecolor="white"); a.set_ylim(0, 8.6)
a.set_xticks(x); a.set_xticklabels(M, fontsize=9)
a.set_ylabel("top-8 ∩ known lung-dominant set (/8)")
a.set_title("C  Biological plausibility of dominant types", fontweight="bold", loc="left")
lab(a, x, ov, fmt="{:.0f}", dy=0.05)
tops = ["#1 AT2 20.7%\n(buries capillary 1.5%)", "#1 cap_1 23.7%\nAT2/AT1/fibro/macro", "#1 monocyte_IL1B\n78.6% (collapse)"]
for xi, t in zip(x, tops):
    a.text(xi, 0.4, t, ha="center", fontsize=7.6, color="white", fontweight="bold")
for s in ("top", "right"): a.spines[s].set_visible(False)

# D: rare-but-real types (pDC/migDC/neuroendocrine) predicted %
a = ax[1, 1]
rare = ["pDC", "migDC", "neuroendocrine"]
data = {"softmax": [1.223, 1.157, 1.071], "prophead": [0.0, 0.001, 0.0], "nnls_core": [0.0, 0.0, 0.0]}
x = np.arange(len(rare)); w = 0.26
for i, (k, col) in enumerate(zip(["softmax", "prophead", "nnls_core"], C)):
    a.bar(x + (i-1)*w, data[k], w, color=col, edgecolor="white",
          label=k if k != "prophead" else "prophead (±floor)")
a.set_xticks(x); a.set_xticklabels(rare, fontsize=9); a.set_ylim(0, 1.5)
a.set_ylabel("predicted proportion (%)")
a.set_title("D  Rare-but-real types: inflate vs zero", fontweight="bold", loc="left")
a.legend(frameon=False, fontsize=8, loc="upper right")
a.text(1, 1.32, "softmax inflates (~1% each, uniform) · prophead zeros\n(lost downstream unless floored)",
       ha="center", fontsize=8, color="#B00", fontweight="bold")
for s in ("top", "right"): a.spines[s].set_visible(False)

fig.suptitle("Which solver predicts the most realistic cell proportions?  "
             "(real GSE159585 COVID/normal bulk)", fontsize=13, fontweight="bold")
fig.text(0.012, -0.01,
         "Verdict: prophead most realistic on GT accuracy, composition diversity, identity & dominant-type plausibility. "
         "floor does NOT change p (prophead±floor identical here). Caveat: a crude COVID−normal direction heuristic "
         "favored softmax 3/3 vs prophead 1/3 (confounded; see PROPORTION_REALISM_ANALYSIS.md). Rare tail: neither is perfect.",
         fontsize=7, color="#444", va="top")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(f"{OUT}/proportion_realism.png", dpi=300); fig.savefig(f"{OUT}/proportion_realism.pdf")
print(f"[saved] {OUT}/proportion_realism.png")
