#!/usr/bin/env python
"""WITHIN-CONDITION sample resolution: can the 7 COVID samples be told apart from each other
(and 7 Normal from each other), at GENE level vs ssGSEA level? Remove centroid (cell-type mean)
AND condition mean -> only within-condition sample variation remains. Key question: does ssGSEA
keep it or compress it? Data = NEW (Route2), the Figure-5 pipeline."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
SS="/disk1/maijl/deconv/deconv_20260605/downstream/ssGSEA"
# ---- gene level: genes x (sample,celltype) ----
expr=pd.read_csv(f"{SS}/ssgsea_avg_expr.csv", index_col=0)
meta=pd.read_csv(f"{SS}/ssgsea_avg_meta.csv")
m=meta.set_index("combined_id"); samp=m["Sample"].astype(str); ct=m["CellType"].astype(str); st=m["disease_status"].astype(str)
# ---- ssGSEA level: celltype@@pathway x 14 samples ----
G=pd.read_csv(f"{OUT}/combined_gsva_full.csv", index_col=0)
gcond=np.array([c.split("_")[0] for c in G.columns]); gsamp=np.array([c.split("_",1)[1] for c in G.columns])
G_ct=np.array([r.split("@@")[0] for r in G.index])

def within_fraction(level):
    """avg over features of: within-condition SS / total within-celltype SS."""
    fr=[]
    if level=="gene":
        cts=ct.unique()
        for c in cts:
            cols=ct[ct==c].index
            if len(cols)<6: continue
            X=expr[cols].values  # genes x samples_of_this_ct
            cst=st[cols].values
            for cond in ["Normal","COVID19"]:
                pass
            # within-celltype total = var across all samples; condition SS removed -> within-condition
            tot=X.var(1)*X.shape[1]
            wcond=np.zeros(X.shape[0])
            for cond in np.unique(cst):
                Xi=X[:,cst==cond]
                if Xi.shape[1]<2: continue
                wcond += ((Xi-Xi.mean(1,keepdims=True))**2).sum(1)
            good=tot>1e-9
            fr.append((wcond[good]/tot[good]))
        return np.concatenate(fr)
    else:
        for c in np.unique(G_ct):
            rows=G.index[G_ct==c]
            if len(rows)<3: continue
            X=G.loc[rows].values  # pathways x 14 samples
            tot=X.var(1)*X.shape[1]
            wcond=np.zeros(X.shape[0])
            for cond in np.unique(gcond):
                Xi=X[:,gcond==cond]
                wcond += ((Xi-Xi.mean(1,keepdims=True))**2).sum(1)
            good=tot>1e-9
            fr.append(wcond[good]/tot[good])
        return np.concatenate(fr)

fg=within_fraction("gene"); fs=within_fraction("ssgsea")
print(f"within-condition variance fraction (of within-celltype):  GENE {np.nanmean(fg):.3f}   ssGSEA {np.nanmean(fs):.3f}")

# ---- sample x sample resolution within COVID, gene vs ssGSEA ----
def cond_sig(level, cond):
    if level=="gene":
        common=[c for c in ct.unique() if sum((ct==c)&(st==cond))>=1]
        samples=sorted(samp[(st==cond)].unique())
        common=[c for c in common if all(((samp==s)&(ct==c)&(st==cond)).any() for s in samples)]
        sig={}
        for s in samples:
            vecs=[]
            for c in common:
                cid=m.index[(samp==s)&(ct==c)&(st==cond)]
                if len(cid): vecs.append(expr[cid[0]].values)
            sig[s]=np.concatenate(vecs)
        S=pd.DataFrame(sig)  # feat x samples
        return S.sub(S.mean(1),axis=0)  # remove condition mean
    else:
        cols=G.columns[gcond==cond]; samples=[c.split("_",1)[1] for c in cols]
        S=G[cols].copy(); S.columns=samples
        return S.sub(S.mean(1),axis=0)

for cond in ["COVID19"]:
    rg=cond_sig("gene",cond); rs=cond_sig("ssgsea",cond)
    # align sample order
    common_s=[s for s in rg.columns if s in set(rs.columns)]
    Cg=np.corrcoef(rg[common_s].T.values); Cs=np.corrcoef(rs[common_s].T.values)
    iu=np.triu_indices_from(Cg,1)
    agree=np.corrcoef(Cg[iu],Cs[iu])[0,1]
    print(f"[{cond}] {len(common_s)} samples; gene vs ssGSEA sample-structure agreement r={agree:.3f}")

# ---- figure ----
fig,axs=plt.subplots(1,3, figsize=(14,4.4))
ax=axs[0]
ax.bar(["GENE\nlevel","ssGSEA\nlevel"],[np.nanmean(fg)*100,np.nanmean(fs)*100],color=["#27AE60","#8E44AD"],edgecolor="black")
for i,v in enumerate([np.nanmean(fg)*100,np.nanmean(fs)*100]): ax.text(i,v+0.8,f"{v:.0f}%",ha="center",fontweight="bold")
ax.set_ylabel("within-condition variance\n(% of within-cell-type variance)")
ax.set_title("(A) How much within-cell-type variation is\nbetween same-condition samples", fontsize=10, fontweight="bold")
rg=cond_sig("gene","COVID19"); rs=cond_sig("ssgsea","COVID19")
common_s=[s for s in rg.columns if s in set(rs.columns)]
Cg=np.corrcoef(rg[common_s].T.values); Cs=np.corrcoef(rs[common_s].T.values)
for ax,C,tt in [(axs[1],Cg,"(B) COVID sample×sample (GENE)"),(axs[2],Cs,"(C) COVID sample×sample (ssGSEA)")]:
    im=ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(common_s))); ax.set_xticklabels([s[-3:] for s in common_s],fontsize=7)
    ax.set_yticks(range(len(common_s))); ax.set_yticklabels([s[-3:] for s in common_s],fontsize=7)
    ax.set_title(tt+"\n(condition mean removed)", fontsize=10, fontweight="bold")
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
iu=np.triu_indices_from(Cg,1); agree=np.corrcoef(Cg[iu],Cs[iu])[0,1]
fig.suptitle(f"Within-condition sample resolution: GENE keeps {np.nanmean(fg)*100:.0f}% vs ssGSEA {np.nanmean(fs)*100:.0f}%; "
             f"COVID sample-structure agreement gene↔ssGSEA r={agree:.2f}", fontsize=11, fontweight="bold", y=1.04)
plt.tight_layout(); plt.savefig(f"{OUT}/within_condition_resolution.png", dpi=160, bbox_inches="tight")
print(f"[saved] {OUT}/within_condition_resolution.png")
