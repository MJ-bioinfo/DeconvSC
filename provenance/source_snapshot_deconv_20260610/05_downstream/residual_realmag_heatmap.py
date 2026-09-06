#!/usr/bin/env python
"""Centroid-removed residual heatmap in REAL magnitude (log1p units, NOT z-scored), per sample,
so you see (i) COVID/Normal block, (ii) sample-to-sample gradient, (iii) true amplitude.
3 blocks: REAL truth (COVID-Normal logFC) | NEW(Route2) per-sample residual | OLD(Delta_z) per-sample residual.
Genes chosen data-drivenly from the real consensus (top COVID-up + Normal-up)."""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
from matplotlib.gridspec import GridSpec
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
FILES={"NEW":{"COVID":"/disk1/maijl/deconv/deconv_20260605/downstream/covid/generated_data_covid.h5ad",
        "Normal":"/disk1/maijl/deconv/deconv_20260605/downstream/normal_18_24/generated_data_normal_18_24.h5ad"},
       "OLD":{"COVID":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_covid.h5ad",
        "Normal":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_normal.h5ad"}}
MIN=10
ISGSET=set(["ISG15","IFI6","IFI27","IFI44","IFI44L","IFIT1","IFIT2","IFIT3","IFITM1","IFITM3","MX1","MX2","OAS1","OAS2","OAS3","OASL","RSAD2","STAT1","STAT2","IRF7","XAF1","BST2","LY6E"])
COLSET=set(["COL1A1","COL1A2","COL3A1","COL5A1","COL6A1","COL6A2","FN1","TIMP1","VCAN","LUM","DCN","SPARC","BGN","POSTN"])

def per_ct_sample(path):
    a=ad.read_h5ad(path); X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    g=a.var_names.astype(str).to_numpy(); s=a.obs["Sample"].astype(str).to_numpy(); c=a.obs["Cell_type"].astype(str).to_numpy()
    rows={}
    for u in np.unique(s):
        for cc in np.unique(c[s==u]):
            m=(s==u)&(c==cc)
            if m.sum()>=MIN: rows[(u,cc)]=X[m].mean(0)
    del a,X; return pd.DataFrame(rows, index=g)

def model_resid_persample(model):
    cov=per_ct_sample(FILES[model]["COVID"]); nor=per_ct_sample(FILES[model]["Normal"])
    g=cov.index.intersection(nor.index); E=pd.concat([nor.loc[g],cov.loc[g]],axis=1)
    samples=sorted({k[0] for k in E.columns}); cts=sorted({k[1] for k in E.columns})
    cond={k[0]:("COVID" if k[0] in {c[0] for c in cov.columns} else "Normal") for k in E.columns}
    centroid={c:E[[k for k in E.columns if k[1]==c]].mean(1) for c in cts}
    # per sample: residual averaged over all cell types present
    R={}
    for s in samples:
        cols=[(s,c) for c in cts if (s,c) in E.columns]
        R[s]=pd.concat([E[(s,c)]-centroid[c] for (s,c) in cols], axis=1).mean(1)
    Rd=pd.DataFrame(R)  # gene x sample
    order=sorted(samples, key=lambda x:(cond[x]!="Normal", x))  # Normal first
    return Rd[order], [cond[s] for s in order], g

def main():
    real=pd.read_csv(f"{OUT}/faithfulness_consensus_vectors.csv", index_col=0)  # real/NEW/OLD consensus per gene
    Rn,condn,gn=model_resid_persample("NEW"); Ro,condo,go=model_resid_persample("OLD")
    g=real.index.intersection(Rn.index).intersection(Ro.index)
    rc=real.loc[g,"real"].sort_values()
    up=[x for x in rc.index[::-1] if x in set(gn)][:20]      # COVID-up by real
    dn=[x for x in rc.index if x in set(gn)][:10]            # Normal-up by real
    genes=dn[::-1]+up                                         # Normal-up (top) ... COVID-up (bottom)
    lab=[f"{x}{' [ISG]' if x in ISGSET else (' [COL]' if x in COLSET else '')}" for x in genes]
    vmax=0.6
    fig=plt.figure(figsize=(13,9)); gs=GridSpec(1,3, width_ratios=[0.5,1,1], wspace=0.05)
    # REAL strip
    ax0=fig.add_subplot(gs[0])
    im=ax0.imshow(real.loc[genes,["real"]].values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax0.set_xticks([0]); ax0.set_xticklabels(["REAL\nCOVID−Normal\nlogFC"], fontsize=8)
    ax0.set_yticks(range(len(genes))); ax0.set_yticklabels(lab, fontsize=6.5)
    ax0.set_title("truth", fontsize=10, fontweight="bold")
    for r,(R,cond,tag) in enumerate([(Rn,condn,"NEW (Route2)"),(Ro,condo,"OLD (Delta_z)")]):
        ax=fig.add_subplot(gs[r+1])
        M=R.loc[genes].values
        im=ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        nN=sum(c=="Normal" for c in cond)
        ax.axvline(nN-0.5, color="black", lw=1.5)
        ax.set_yticks([]); ax.set_xticks(range(len(cond)))
        ax.set_xticklabels([s for s in R.columns], rotation=90, fontsize=5)
        # condition bar
        for i,c in enumerate(cond):
            ax.add_patch(plt.Rectangle((i-0.5,-1.4),1,0.9, color="#C0392B" if c=="COVID" else "#2C3E50", clip_on=False))
        ax.text(nN/2-0.5,-1.9,"Normal",ha="center",fontsize=7,color="#2C3E50",fontweight="bold")
        ax.text(nN+(len(cond)-nN)/2-0.5,-1.9,"COVID",ha="center",fontsize=7,color="#C0392B",fontweight="bold")
        # COVID-Normal (model) vs truth corr on these genes (NOT mean-over-samples, which cancels)
        mC=[i for i,c in enumerate(cond) if c=="COVID"]; mN=[i for i,c in enumerate(cond) if c=="Normal"]
        dmodel=R.loc[genes].iloc[:,mC].mean(1).values - R.loc[genes].iloc[:,mN].mean(1).values
        cc=np.corrcoef(dmodel, real.loc[genes,"real"].values)[0,1]
        ax.set_title(f"{tag}  (per-sample residual)\nmodel COVID−Normal vs truth r={cc:.2f}", fontsize=10, fontweight="bold")
    cb=fig.colorbar(im, ax=fig.axes, fraction=0.025, pad=0.02); cb.set_label("centroid-removed residual / logFC (log1p units)", fontsize=8)
    fig.suptitle("Centroid-removed residual in REAL magnitude (not z-scored): COVID/Normal block + per-sample gradient + true amplitude\n"
                 "Both models show the disease block & sample gradient and track the real program (OLD slightly closer here)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.savefig(f"{OUT}/residual_realmag_heatmap.png", dpi=155, bbox_inches="tight")
    print(f"[saved] {OUT}/residual_realmag_heatmap.png")

if __name__=="__main__": main()
