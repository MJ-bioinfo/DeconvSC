#!/usr/bin/env python
"""Sample-level ssGSEA heatmap WITHOUT z-score: (left) raw GSVA real units -> row-baseline dominates;
(right) centroid-removed (row-mean subtracted) real units -> COVID/Normal block + per-sample gradient
return. Same rows as Fig5b (top differential celltype@@pathway). This shows the z-score in Fig5b was
hiding magnitude, and that removing the baseline (not scaling) is the honest fix."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
from matplotlib.gridspec import GridSpec
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
raw=pd.read_csv(f"{OUT}/gsva_sample_raw_nozscore.csv", index_col=0)
cen=pd.read_csv(f"{OUT}/gsva_sample_centroidremoved_nozscore.csv", index_col=0)
meta=pd.read_csv(f"{OUT}/gsva_sample_meta.csv")
logfc=pd.read_csv(f"{OUT}/gsva_sample_rows_logfc.csv").set_index("CP")["logFC"]
# order rows by logFC (Normal-high top -> COVID-high bottom)
order=logfc.reindex(raw.index).sort_values().index
raw=raw.loc[order]; cen=cen.loc[order]
cond=meta["condition"].values; nN=int((cond=="Normal").sum())
rl=[r.replace("@@"," · ")[:46] for r in raw.index]

fig=plt.figure(figsize=(16,11)); gs=GridSpec(1,2, wspace=0.62)
def panel(ax, M, title, cmap, vlo, vhi, center0=False):
    im=ax.imshow(M.values, aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi)
    ax.axvline(nN-0.5, color="black", lw=1.6)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(rl))); ax.set_yticklabels(rl, fontsize=6)
    for i,c in enumerate(cond):
        ax.add_patch(plt.Rectangle((i-0.5,-1.6),1,1.0, color="#C0392B" if c=="COVID19" else "#2C3E50", clip_on=False))
    ax.text(nN/2-0.5,-2.2,"Normal",ha="center",fontsize=8,color="#2C3E50",fontweight="bold")
    ax.text(nN+(len(cond)-nN)/2-0.5,-2.2,"COVID",ha="center",fontsize=8,color="#C0392B",fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

# raw: GSVA scores roughly in [-1,1]; center colormap at the matrix median to show baseline domination
ax1=fig.add_subplot(gs[0])
vlo,vhi=np.percentile(raw.values,2),np.percentile(raw.values,98)
panel(ax1, raw, "(1) RAW ssGSEA, NO z-score (real units)\nCOVID/Normal block IS visible — signal not lost",
      "viridis", vlo, vhi)
# centroid-removed: symmetric around 0
ax2=fig.add_subplot(gs[1])
v=np.percentile(np.abs(cen.values),98)
panel(ax2, cen, "(2) centroid-removed (row-mean subtracted), NO z-score\nbaseline centered at 0 → block + sample gradient sharper (same COVID−Normal)",
      "RdBu_r", -v, v)
# separation metric per panel
def sep(M):
    d=M.values[:, nN:].mean(1)-M.values[:, :nN].mean(1)
    return np.abs(d).mean()
fig.suptitle(f"Sample-level ssGSEA WITHOUT z-score (45 top-differential celltype·pathway rows)   |   "
             f"COVID−Normal separation = {sep(raw):.2f} GSVA units (identical in both; centroid removal only re-centers rows). Discrimination is REAL without z-score; Fig5b's per-row z-score is what flattened it to a mirror.",
             fontsize=11.5, fontweight="bold", y=1.01)
plt.savefig(f"{OUT}/gsva_sample_nozscore_heatmap.png", dpi=150, bbox_inches="tight")
print(f"[saved] {OUT}/gsva_sample_nozscore_heatmap.png")
print(f"  raw GSVA value range: [{raw.values.min():.2f}, {raw.values.max():.2f}]; centroid-removed |COVID-Normal| mean={sep(cen):.3f}")
