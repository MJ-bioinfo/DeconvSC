#!/usr/bin/env python
"""Figures for the beyond-centroid demonstration (OLD Delta_z vs NEW Route2)."""
import numpy as np, pandas as pd, os, json
import matplotlib; matplotlib.use("Agg")
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
import matplotlib.pyplot as plt
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
S=json.load(open(f"{OUT}/beyond_centroid_summary.json"))
NEW=[k for k in S if "Route2" in k][0]; OLD=[k for k in S if "Delta" in k][0]
Dn=pd.read_csv(f"{OUT}/within_celltype_COVIDminusNormal_NEW.csv", index_col=0)  # celltype x gene
Do=pd.read_csv(f"{OUT}/within_celltype_COVIDminusNormal_OLD.csv", index_col=0)
ISG=["ISG15","IFI6","IFI27","IFI44","IFI44L","IFIT1","IFIT2","IFIT3","IFITM1","IFITM3","MX1","MX2",
 "OAS1","OAS2","OAS3","OASL","RSAD2","STAT1","STAT2","IRF7","XAF1","BST2","LY6E","GBP1","GBP5",
 "CXCL10","EIF2AK2","USP18","HERC5","DDX58","IFIH1","SIGLEC1","SERPING1","IFI35","SAMD9","SAMD9L","PARP9","DDX60"]
isg=[g for g in ISG if g in Dn.columns and g in Do.columns]
CN="#C0392B"; CO="#2C3E50"

fig=plt.figure(figsize=(15,9))
gs=fig.add_gridspec(2,3, height_ratios=[1,1.25], hspace=0.42, wspace=0.32)

# (A) sample-specificity magnitude
ax=fig.add_subplot(gs[0,0])
vals=[S[OLD]["between_sample_sd"], S[NEW]["between_sample_sd"]]
ax.bar(["OLD\nDelta_z","NEW\nRoute2"], vals, color=[CO,CN], edgecolor="black")
for i,v in enumerate(vals): ax.text(i, v+0.1, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("within-cell-type\nbetween-sample SD (log1p)")
ax.set_title("(A) Sample-specificity magnitude\nOLD collapses to one centroid/cell type", fontsize=10, fontweight="bold")
ax.text(0.5,0.78,f"residual/centroid:\nOLD {S[OLD]['resid_to_centroid']*100:.1f}%  vs  NEW {S[NEW]['resid_to_centroid']*100:.1f}%",
        transform=ax.transAxes, ha="center", fontsize=8, bbox=dict(boxstyle="round",fc="wheat",alpha=0.6))

# (B) coherence: pairwise corr distribution of D_t across cell types
ax=fig.add_subplot(gs[0,1])
Cn=np.corrcoef(Dn.values); Co=np.corrcoef(Do.values)
iun=np.triu_indices_from(Cn,1)
ax.hist(Co[iun], bins=40, alpha=0.55, color=CO, label=f"OLD (mean {np.nanmean(Co[iun]):.2f})", density=True)
ax.hist(Cn[iun], bins=40, alpha=0.55, color=CN, label=f"NEW (mean {np.nanmean(Cn[iun]):.2f})", density=True)
ax.axvline(np.nanmean(Co[iun]),color=CO,ls="--"); ax.axvline(np.nanmean(Cn[iun]),color=CN,ls="--")
ax.set_xlabel("pairwise corr of COVID−Normal shift\nbetween cell types"); ax.set_ylabel("density")
ax.set_title("(B) Coherence of the disease program\nshared across cell types = real biology", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)

# (C) ISG up-regulation in COVID: mean across cell types, NEW vs OLD
ax=fig.add_subplot(gs[0,2])
mn=Dn[isg].mean(0).sort_values(); 
ax.barh(range(len(isg)), Do[isg].mean(0).reindex(mn.index).values, color=CO, alpha=0.7, label="OLD")
ax.barh(range(len(isg)), mn.values, left=0, color="none", edgecolor=CN, lw=1.2)
ax.scatter(mn.values, range(len(isg)), color=CN, s=14, label="NEW", zorder=5)
ax.set_yticks(range(len(isg))); ax.set_yticklabels(mn.index, fontsize=5.5)
ax.axvline(0,color="black",lw=0.5); ax.set_xlabel("mean COVID−Normal logFC (across cell types)")
ax.set_title("(C) Interferon-stimulated genes ↑ in COVID\nNEW recovers ISG program, OLD ~flat", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")

# (D,E) ISG x immune-celltype heatmaps NEW vs OLD
immune=["monocyte_classical","monocyte_non_classical","monocyte_IL1B","macro_alveolar","macro_alveolar_SPP1",
 "macrophage_interstitial","cDC_type_1","cDC_type_2","pDC","CD8_EM","CD8_RM","CD4_naive","NK","B_cell","plasma"]
immune=[c for c in immune if c in Dn.index and c in Do.index]
isg_h=[g for g in isg if g in Dn.columns]
vmax=max(abs(Dn.loc[immune,isg_h].values).max(), abs(Do.loc[immune,isg_h].values).max())*0.8
for col,(tag,D) in zip([gs[1,0],gs[1,1]], [("NEW (Route2)",Dn),("OLD (Delta_z)",Do)]):
    ax=fig.add_subplot(col)
    im=ax.imshow(D.loc[immune,isg_h].values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(isg_h))); ax.set_xticklabels(isg_h, rotation=90, fontsize=5)
    ax.set_yticks(range(len(immune))); ax.set_yticklabels(immune, fontsize=6.5)
    ax.set_title(f"({'D' if tag.startswith('NEW') else 'E'}) {tag}: ISG COVID−Normal logFC\nby immune cell type", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (F) consensus top COVID-up program (NEW), bar
ax=fig.add_subplot(gs[1,2])
cons=pd.read_csv(f"{OUT}/consensus_COVIDminusNormal_NEW.csv", index_col=0).iloc[:,0].sort_values(ascending=False).head(20)
cols=["#C0392B" if g in ISG else "#E67E22" for g in cons.index]
ax.barh(range(len(cons)), cons.values[::-1], color=cols[::-1], edgecolor="black", lw=0.3)
ax.set_yticks(range(len(cons))); ax.set_yticklabels(cons.index[::-1], fontsize=6.5)
ax.set_xlabel("mean COVID−Normal logFC"); 
ax.set_title("(F) NEW consensus COVID-up program\n(orange=remodeling/stress, red=ISG)", fontsize=10, fontweight="bold")

fig.suptitle("Does the model capture sample-specific transcriptome BEYOND the cell-type centroid?  (GSE159585, 7 COVID + 7 Normal, 62 cell types)",
             fontsize=12, fontweight="bold", y=0.97)
plt.savefig(f"{OUT}/beyond_centroid_demonstration.png", dpi=155, bbox_inches="tight")
print(f"[saved] {OUT}/beyond_centroid_demonstration.png")
