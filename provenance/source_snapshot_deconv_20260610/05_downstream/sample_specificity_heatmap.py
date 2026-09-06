#!/usr/bin/env python
"""Build heatmaps that DO reveal sample-specificity (vs Fig5a logFC which collapses 7+7 samples
into 2 group means, and Fig5b which is row-z-scored). Two views, NEW(Route2) vs OLD(Delta_z):
 (A) sample x sample correlation of the CENTROID-REMOVED residual signature (pooled over common
     cell types). NEW -> COVID/Normal blocks (coherent sample/disease specificity); OLD -> noise.
 (B) disease-program x sample residual heatmap (centroid removed, real units, NOT z-scored),
     averaged over immune cell types. NEW -> COVID-up programs with per-sample gradient; OLD -> flat.
"""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"; os.makedirs(OUT, exist_ok=True)
FILES={"NEW (Route2 anchored)":{"COVID":"/disk1/maijl/deconv/deconv_20260605/downstream/covid/generated_data_covid.h5ad",
        "Normal":"/disk1/maijl/deconv/deconv_20260605/downstream/normal_18_24/generated_data_normal_18_24.h5ad"},
       "OLD (Delta_z)":{"COVID":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_covid.h5ad",
        "Normal":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_normal.h5ad"}}
MIN=10
PROGRAMS={
 "Interferon (ISG)":["ISG15","IFI6","IFI27","IFI44","IFI44L","IFIT1","IFIT2","IFIT3","IFITM1","IFITM3","MX1","MX2","OAS1","OAS2","OAS3","OASL","RSAD2","STAT1","STAT2","IRF7","XAF1","BST2","LY6E"],
 "Collagen/ECM/fibrosis":["COL1A1","COL1A2","COL3A1","COL5A1","COL6A1","COL6A2","FN1","TIMP1","VCAN","LUM","DCN","SPARC","FBN1","BGN","POSTN","MMP2"],
 "Complement":["C1QA","C1QB","C1QC","C1R","C1S","C3","CFB","C2","SERPING1"],
 "Coagulation":["F3","F5","F13A1","SERPINE1","PLAT","THBD","VWF","FGA","FGB","FGG","PROS1"],
 "MHC-II / Ag-presentation":["HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DPB1","HLA-DQA1","CD74","HLA-DMA","HLA-DMB"],
 "Stress / UPR":["HSPA1A","HSPA1B","HSPA5","DDIT3","XBP1","ATF4","HSPB1","DNAJB1"]}
IMMUNE=["monocyte_classical","monocyte_non_classical","monocyte_IL1B","macro_alveolar","macro_alveolar_SPP1",
 "macrophage_interstitial","macro_HS3ST2","cDC_type_1","cDC_type_2","pDC","migDC","CD8_EM","CD8_RM","CD8_EMRA",
 "CD4_naive","CD4_EM","NK","NK_KLRC1","B_cell","plasma","granulocyte","mast_cell"]

def per_ct_sample(path, status):
    a=ad.read_h5ad(path); X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    genes=a.var_names.astype(str).to_numpy(); samp=a.obs["Sample"].astype(str).to_numpy(); ct=a.obs["Cell_type"].astype(str).to_numpy()
    rows={}
    for s in np.unique(samp):
        for c in np.unique(ct[samp==s]):
            m=(samp==s)&(ct==c)
            if m.sum()>=MIN: rows[(f"{status}:{s}",c)]=X[m].mean(0)
    del a,X; return pd.DataFrame(rows, index=genes)

def model_mats(model):
    cov=per_ct_sample(FILES[model]["COVID"],"COVID"); nor=per_ct_sample(FILES[model]["Normal"],"Normal")
    g=cov.index.intersection(nor.index); E=pd.concat([nor.loc[g],cov.loc[g]],axis=1)  # genes x (sample,ct)
    samples=sorted({k[0] for k in E.columns}, key=lambda x:(x.split(":")[0]!="Normal", x))  # Normal first
    cts=sorted({k[1] for k in E.columns})
    # centroid per cell type (mean over samples present), residual r = e - centroid
    resid={}; centroid={}
    for c in cts:
        cols=[k for k in E.columns if k[1]==c]; centroid[c]=E[cols].mean(1).values
    # (A) pooled residual signature per sample over cell types present in ALL samples
    common=[c for c in cts if all((f"{'COVID' if s.startswith('COVID') else s.split(':')[0]}", )  or True for s in samples)
            and all(((sm,c) in E.columns) for sm in samples)]
    sig={}
    for s in samples:
        vecs=[E[(s,c)].values-centroid[c] for c in common]
        sig[s]=np.concatenate(vecs) if vecs else np.array([])
    sigdf=pd.DataFrame(sig)  # (gene*ct) x samples
    corr=np.corrcoef(sigdf.T.values)  # samples x samples
    # (B) program residual per sample, averaged over immune cell types present
    gset=set(g); progmat=pd.DataFrame(index=list(PROGRAMS), columns=samples, dtype=float)
    imm=[c for c in IMMUNE if c in cts]
    for p,gg in PROGRAMS.items():
        gi=[x for x in gg if x in gset]
        for s in samples:
            vals=[]
            for c in imm:
                if (s,c) in E.columns:
                    vals.append(np.mean(E[(s,c)].loc[gi].values - centroid[c][[E.index.get_loc(x) for x in gi]]))
            progmat.loc[p,s]=np.nanmean(vals) if vals else np.nan
    return samples, corr, progmat.astype(float)

def main():
    R={}
    for m in FILES: R[m]=model_mats(m); print(f"[{m}] {len(R[m][0])} samples")
    NEW=[m for m in FILES if "Route2" in m][0]; OLD=[m for m in FILES if "Delta" in m][0]
    fig=plt.figure(figsize=(14,8)); gs=fig.add_gridspec(2,2, width_ratios=[1,1.35], hspace=0.33, wspace=0.28)
    for r,m in enumerate([NEW,OLD]):
        samples,corr,prog=R[m]
        lab=[s.split(":")[0][0]+s.split(":")[1][-3:] for s in samples]  # short
        cond=[s.split(":")[0] for s in samples]
        # (A) sample-sample residual correlation
        ax=fig.add_subplot(gs[r,0])
        im=ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        n=len(samples); nN=sum(c=="Normal" for c in cond)
        ax.axhline(nN-0.5,color="k",lw=1); ax.axvline(nN-0.5,color="k",lw=1)
        ax.set_xticks(range(n)); ax.set_xticklabels([c[0] for c in cond], fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels([s.split(':')[1] for s in samples], fontsize=5)
        ax.set_title(f"{m}\n(A) sample×sample corr of centroid-removed residual", fontsize=9, fontweight="bold")
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        # (B) program residual x sample
        ax=fig.add_subplot(gs[r,1])
        vmax=np.nanmax(np.abs(R[NEW][2].values))*0.9
        im=ax.imshow(prog.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_xticklabels([s.split(':')[1] for s in samples], rotation=90, fontsize=5)
        ax.axvline(nN-0.5,color="k",lw=1.2)
        ax.set_yticks(range(len(prog.index))); ax.set_yticklabels(prog.index, fontsize=8)
        ax.set_title(f"(B) disease-program residual per sample (immune avg, centroid-removed)\nNormal | COVID", fontsize=9, fontweight="bold")
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.suptitle("Sample-specificity heatmaps (centroid removed): BOTH NEW(Route2) and OLD(Delta_z, application/) resolve samples & the COVID block",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.savefig(f"{OUT}/sample_specificity_heatmap.png", dpi=155, bbox_inches="tight")
    print(f"[saved] {OUT}/sample_specificity_heatmap.png")

if __name__=="__main__": main()
