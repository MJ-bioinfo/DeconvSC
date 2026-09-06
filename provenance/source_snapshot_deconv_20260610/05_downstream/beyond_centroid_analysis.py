#!/usr/bin/env python
"""Demonstrate that the NEW anchored (Route2) model encodes sample-specific transcriptome
WITHIN a cell type (beyond the shared centroid), faithfully, while the OLD Delta_z model does
not. Same 7 COVID + 7 Normal samples, same 62 cell types, same genes for both models.

Key idea: per (sample, cell type) mean log1p profile e_{s,t}. The centroid_t = mean over all
samples is SHARED; any COVID-vs-Normal difference within a cell type, D_t = mean_covid e - mean_normal e,
is BEYOND the centroid by construction. We test whether D_t is (1) non-trivial, (2) biologically
COHERENT across cell types (a shared interferon program => real biology; idiosyncratic => noise),
(3) made of real interferon-stimulated genes (ISGs)."""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, os
from scipy.stats import pearsonr
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"; os.makedirs(OUT, exist_ok=True)
FILES={
 "NEW (Route2 anchored)":{
   "COVID":"/disk1/maijl/deconv/deconv_20260605/downstream/covid/generated_data_covid.h5ad",
   "Normal":"/disk1/maijl/deconv/deconv_20260605/downstream/normal_18_24/generated_data_normal_18_24.h5ad"},
 "OLD (Delta_z)":{
   "COVID":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_covid.h5ad",
   "Normal":"/disk1/maijl/deconv/cVAE/GSE159585/application/generated_data_normal.h5ad"}}
MIN_CELLS=10                      # per (sample,celltype) to trust the mean
ISG=["ISG15","IFI6","IFI27","IFI44","IFI44L","IFIT1","IFIT2","IFIT3","IFITM1","IFITM3","MX1","MX2",
 "OAS1","OAS2","OAS3","OASL","RSAD2","STAT1","STAT2","IRF7","XAF1","BST2","LY6E","GBP1","GBP5",
 "CXCL10","EIF2AK2","USP18","HERC5","DDX58","IFIH1","SIGLEC1","SERPING1","IFI35","SAMD9","SAMD9L","PARP9","DDX60"]

def per_ct_sample_mean(path, status):
    a=ad.read_h5ad(path)
    X=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    genes=a.var_names.astype(str).to_numpy()
    samp=a.obs["Sample"].astype(str).to_numpy(); ct=a.obs["Cell_type"].astype(str).to_numpy()
    rows={}
    for s in np.unique(samp):
        for c in np.unique(ct[samp==s]):
            m=(samp==s)&(ct==c)
            if m.sum()>=MIN_CELLS: rows[(s,c,status)]=X[m].mean(0)
    df=pd.DataFrame(rows, index=genes)  # genes x (s,c,status)
    del a, X; return df

def build(model):
    cov=per_ct_sample_mean(FILES[model]["COVID"],"COVID")
    nor=per_ct_sample_mean(FILES[model]["Normal"],"Normal")
    g=cov.index.intersection(nor.index)
    return cov.loc[g], nor.loc[g], g

def analyze(model):
    cov,nor,genes=build(model)
    # cell types present in >=4 covid and >=4 normal samples
    cov_ct=pd.Series([c for (s,c,st) in cov.columns]).value_counts()
    nor_ct=pd.Series([c for (s,c,st) in nor.columns]).value_counts()
    cts=sorted([c for c in cov_ct.index if cov_ct[c]>=4 and nor_ct.get(c,0)>=4])
    # within-cell-type COVID-Normal shift D_t (gene vector), + centroid + residual magnitudes
    D={}; cent_norm=[]; resid_sd=[]
    for c in cts:
        cc=cov.loc[:, [k for k in cov.columns if k[1]==c]].values  # genes x n_cov
        nn=nor.loc[:, [k for k in nor.columns if k[1]==c]].values
        allp=np.concatenate([cc,nn],1)
        centroid=allp.mean(1)
        D[c]=cc.mean(1)-nn.mean(1)
        cent_norm.append(np.linalg.norm(centroid))
        resid_sd.append(np.linalg.norm(allp-centroid[:,None])/np.sqrt(allp.shape[1]))  # between-sample spread
    Dmat=pd.DataFrame(D, index=genes).T  # celltype x gene
    # (2) coherence: mean pairwise corr of D_t across cell types + PC1 variance fraction
    C=np.corrcoef(Dmat.values)
    iu=np.triu_indices_from(C,1); mean_pair=np.nanmean(C[iu])
    # PC1 fraction (SVD on centered-by-gene? use raw D, capture shared program)
    U,S,Vt=np.linalg.svd(Dmat.values - Dmat.values.mean(0, keepdims=True), full_matrices=False)
    pc1=float((S[0]**2)/np.sum(S**2))
    # (3) consensus COVID-up program = mean D across cell types; ISG enrichment in top-50
    consensus=Dmat.mean(0).sort_values(ascending=False)
    top50=list(consensus.index[:50]); isg_in_top=[g for g in top50 if g in ISG]
    isg_present=[g for g in ISG if g in genes]
    isg_meanD=float(Dmat[isg_present].mean().mean()) if isg_present else np.nan  # mean COVID-Normal of ISGs across cts
    # ISG consensus rank percentile (lower=more up)
    ranks={g:int(np.where(consensus.index==g)[0][0]) for g in isg_present}
    isg_med_pct=float(np.median([ranks[g]/len(consensus) for g in isg_present]))
    summary=dict(model=model, n_celltypes=len(cts), n_genes=len(genes),
        centroid_norm=float(np.mean(cent_norm)), between_sample_sd=float(np.mean(resid_sd)),
        resid_to_centroid=float(np.mean(resid_sd)/np.mean(cent_norm)),
        coherence_mean_pairwise_corr=float(mean_pair), coherence_PC1_varfrac=pc1,
        ISG_mean_COVIDminusNormal=isg_meanD, ISG_in_top50=len(isg_in_top),
        ISG_median_rank_pctile=isg_med_pct, top10_consensus=list(consensus.index[:10]))
    return summary, Dmat, consensus, cts, isg_present

def main():
    res={}; Dmats={}; cons={}; isgp=None
    for model in FILES:
        s,D,c,cts,isgp=analyze(model); res[model]=s; Dmats[model]=D; cons[model]=c
        print(f"\n### {model}: {s['n_celltypes']} cell types, {s['n_genes']} genes")
        for k,v in s.items():
            if k not in ("model","top10_consensus"): print(f"   {k}: {v}")
        print("   top10 COVID-up consensus genes:", s["top10_consensus"])
    pd.DataFrame([{k:v for k,v in res[m].items() if k!='top10_consensus'} for m in res]).to_csv(
        f"{OUT}/beyond_centroid_summary.csv", index=False)
    # save D matrices + consensus
    for m in FILES:
        tag="NEW" if "Route2" in m else "OLD"
        Dmats[m].round(4).to_csv(f"{OUT}/within_celltype_COVIDminusNormal_{tag}.csv")
        cons[m].round(5).to_csv(f"{OUT}/consensus_COVIDminusNormal_{tag}.csv", header=["mean_logFC_COVID_minus_Normal"])
    import json
    json.dump({m:res[m] for m in res}, open(f"{OUT}/beyond_centroid_summary.json","w"), indent=2)
    print(f"\n[saved] {OUT}/beyond_centroid_summary.csv + within/consensus CSVs + json")

if __name__=="__main__":
    main()
