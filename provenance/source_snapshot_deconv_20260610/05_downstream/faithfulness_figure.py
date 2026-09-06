#!/usr/bin/env python
"""Honest summary figure: with the CORRECT data (OLD=application/, truth=snRNAseq), OLD(Delta_z) is
NOT collapsed and is slightly MORE faithful to the real COVID-Normal program than NEW(Route2) at the
per-cell-type expression level. So Figure-5 / ssGSEA is NOT where Route2's advantage lies."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
from scipy.stats import pearsonr
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
bydf=pd.read_csv(f"{OUT}/faithfulness_vs_truth_by_celltype.csv")
cons=pd.read_csv(f"{OUT}/faithfulness_consensus_vectors.csv", index_col=0)
summ=pd.read_csv(f"{OUT}/beyond_centroid_summary.csv")
CN="#C0392B"; CO="#2C3E50"
fig,axs=plt.subplots(1,3, figsize=(15,4.6))

# (A) sample-specificity magnitude (corrected): both ~equal, NOT collapsed
ax=axs[0]
sd=dict(zip(summ["model"], summ["resid_to_centroid"]))
oldk=[k for k in sd if "Delta" in k][0]; newk=[k for k in sd if "Route2" in k][0]
ax.bar(["OLD\nDelta_z","NEW\nRoute2"], [sd[oldk]*100, sd[newk]*100], color=[CO,CN], edgecolor="black")
for i,v in enumerate([sd[oldk]*100, sd[newk]*100]): ax.text(i,v+0.3,f"{v:.1f}%",ha="center",fontweight="bold")
ax.set_ylabel("within-cell-type sample variation\n(residual / centroid, %)"); ax.set_ylim(0,24)
ax.set_title("(A) Sample-specificity magnitude\nCORRECTED: OLD ≈ NEW, NOT collapsed", fontsize=10, fontweight="bold")

# (B) faithfulness to truth: per-celltype corr distribution
ax=axs[1]
ax.boxplot([bydf["OLD_vs_real"], bydf["NEW_vs_real"]], tick_labels=["OLD\nDelta_z","NEW\nRoute2"],
           patch_artist=True, showmeans=True,
           boxprops=dict(facecolor="#bbb"), medianprops=dict(color="black"))
for i,(m,c) in enumerate([("OLD_vs_real",CO),("NEW_vs_real",CN)]):
    y=bydf[m].values; x=np.random.RandomState(0).normal(i+1,0.05,len(y))
    ax.scatter(x,y,s=10,color=c,alpha=0.6,zorder=3)
ax.set_ylabel("per-cell-type corr( model , REAL )\nCOVID−Normal shift")
ax.set_title(f"(B) Faithfulness to real biology\nmean: OLD {bydf['OLD_vs_real'].mean():.2f} > NEW {bydf['NEW_vs_real'].mean():.2f}", fontsize=10, fontweight="bold")
ax.set_ylim(0,1)

# (C) consensus scatter: model COVID-Normal vs truth
ax=axs[2]
rN=pearsonr(cons["NEW"],cons["real"])[0]; rO=pearsonr(cons["OLD"],cons["real"])[0]
ax.scatter(cons["real"],cons["OLD"], s=5, alpha=0.3, color=CO, label=f"OLD r={rO:.2f}")
ax.scatter(cons["real"],cons["NEW"], s=5, alpha=0.3, color=CN, label=f"NEW r={rN:.2f}")
lim=np.percentile(np.abs(cons["real"]),99.5)
ax.plot([-lim,lim],[-lim,lim],"k--",lw=0.8); ax.set_xlim(-lim,lim); ax.set_ylim(-lim*1.5,lim*1.5)
ax.set_xlabel("REAL consensus COVID−Normal logFC"); ax.set_ylabel("model consensus COVID−Normal")
ax.set_title("(C) Consensus disease program vs truth\nboth track truth; OLD slightly closer", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
fig.suptitle("Faithfulness of within-cell-type COVID−Normal shift to REAL data (truth = GSE159585 snRNAseq; OLD = application/)",
             fontsize=12, fontweight="bold", y=1.03)
plt.tight_layout(); plt.savefig(f"{OUT}/faithfulness_figure.png", dpi=160, bbox_inches="tight")
print(f"[saved] {OUT}/faithfulness_figure.png  (NEW {bydf['NEW_vs_real'].mean():.3f} vs OLD {bydf['OLD_vs_real'].mean():.3f})")
