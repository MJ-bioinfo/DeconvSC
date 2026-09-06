#!/usr/bin/env python
"""The PROPER differentiator: both OLD(Delta_z, application/) and NEW(Route2) produce comparable
within-cell-type sample variation, so 'amount of sample-specificity' does NOT separate them.
The real question is FAITHFULNESS: does the model's within-cell-type COVID-Normal shift match the
REAL data? Ground truth = GSE159585_combined.h5ad (real COVID + Normal cells, 62 cell types).
For each cell type t: corr( D_t^model , D_t^real ) where D = mean_COVID - mean_Normal (log1p-CP10K)."""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, os
from scipy.stats import pearsonr
OUT="/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
REAL="/disk1/maijl/deconv/data/GSE159585/GSE159585_snRNAseq_raw_counts.h5ad"  # raw counts; condition COVID-19/IPF/control
NORMAL={"control","Normal","normal"}

def real_shift():
    """REAL within-cell-type COVID-19 vs control shift (IPF excluded), log1p-CP10K, memory-safe (sparse)."""
    a=ad.read_h5ad(REAL)
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X); mx=float(X.max())
    if mx>=30:  # raw counts -> CP10K + log1p (sparse-preserving)
        s=np.asarray(X.sum(1)).ravel(); s[s==0]=1
        X=sp.diags(1e4/s)@X; X.data=np.log1p(X.data)
        print(f"[real] raw counts (max={mx:.1f}) -> log1p-CP10K")
    else:
        X.data=X.data; print(f"[real] already log-scaled (max={mx:.1f})")
    genes=a.var_names.astype(str).to_numpy()
    cond=a.obs["condition"].astype(str).to_numpy(); ct=a.obs["cell type"].astype(str).to_numpy()
    D={}
    for c in np.unique(ct):
        mc=(ct==c)&(cond=="COVID-19"); mn=(ct==c)&np.isin(cond,list(NORMAL))
        if mc.sum()>=20 and mn.sum()>=20:
            D[c]=np.asarray(X[mc].mean(0)).ravel()-np.asarray(X[mn].mean(0)).ravel()
    print(f"[real] {len(D)} cell types with >=20 COVID & >=20 control cells (IPF excluded)")
    return pd.DataFrame(D, index=genes).T  # celltype x gene

def main():
    Dreal=real_shift()
    Dn=pd.read_csv(f"{OUT}/within_celltype_COVIDminusNormal_NEW.csv", index_col=0)
    Do=pd.read_csv(f"{OUT}/within_celltype_COVIDminusNormal_OLD.csv", index_col=0)
    genes=Dn.columns.intersection(Do.columns).intersection(Dreal.columns)
    cts=sorted(set(Dn.index)&set(Do.index)&set(Dreal.index))
    print(f"[align] {len(cts)} cell types, {len(genes)} genes")
    rows=[]
    for c in cts:
        r=Dreal.loc[c,genes].values.astype(float)
        n=Dn.loc[c,genes].values.astype(float); o=Do.loc[c,genes].values.astype(float)
        if np.std(r)==0: continue
        rows.append(dict(cell_type=c,
            NEW_vs_real=pearsonr(n,r)[0], OLD_vs_real=pearsonr(o,r)[0],
            n_covid_genes=int((np.abs(r)>0.5).sum())))
    df=pd.DataFrame(rows)
    df["NEW_better"]=df["NEW_vs_real"]>df["OLD_vs_real"]
    df=df.sort_values("NEW_vs_real", ascending=False)
    df.round(4).to_csv(f"{OUT}/faithfulness_vs_truth_by_celltype.csv", index=False)
    # consensus (mean over cell types) alignment
    gc=[g for g in genes]
    consensus_real=Dreal.loc[cts,gc].mean(0).values
    consensus_new=Dn.loc[cts,gc].mean(0).values; consensus_old=Do.loc[cts,gc].mean(0).values
    cN=pearsonr(consensus_new,consensus_real)[0]; cO=pearsonr(consensus_old,consensus_real)[0]
    print("\n=== Faithfulness to REAL COVID-Normal shift (per cell type, mean over cell types) ===")
    print(f"  mean per-celltype corr:  NEW {df['NEW_vs_real'].mean():.4f}   OLD {df['OLD_vs_real'].mean():.4f}")
    print(f"  median per-celltype corr: NEW {df['NEW_vs_real'].median():.4f}   OLD {df['OLD_vs_real'].median():.4f}")
    print(f"  NEW beats OLD in {df['NEW_better'].sum()}/{len(df)} cell types")
    print(f"  consensus program corr:  NEW {cN:.4f}   OLD {cO:.4f}")
    summ=dict(metric=["mean_percelltype_corr","median_percelltype_corr","consensus_corr","NEW_wins_celltypes","n_celltypes"],
              NEW=[df['NEW_vs_real'].mean(),df['NEW_vs_real'].median(),cN,int(df['NEW_better'].sum()),len(df)],
              OLD=[df['OLD_vs_real'].mean(),df['OLD_vs_real'].median(),cO,int((~df['NEW_better']).sum()),len(df)])
    pd.DataFrame(summ).to_csv(f"{OUT}/faithfulness_vs_truth_summary.csv", index=False)
    # save consensus vectors for plotting
    pd.DataFrame({"real":consensus_real,"NEW":consensus_new,"OLD":consensus_old}, index=gc).to_csv(
        f"{OUT}/faithfulness_consensus_vectors.csv")
    print(f"\n[saved] faithfulness_vs_truth_by_celltype.csv / _summary.csv / _consensus_vectors.csv")

if __name__=="__main__": main()
