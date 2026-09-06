#!/usr/bin/env python
"""Preprocess for ssGSEA: average log1p-CP10K expression per (Sample, cell type) for the
Route2 COVID + Normal_18_24 generated data, exactly replicating ssGSEA.R::calc_avg
(mean over cells in log space; cell-type '_' -> '-'; combined_id = "<Sample>_<celltype>").
Writes a genes x (sample_celltype) matrix + a per-column meta table that the R script reads.
"""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, os

DOWN = os.environ.get("DOWN_ROOT", "/disk1/maijl/deconv/deconv_20260605/downstream")
OUT = f"{DOWN}/ssGSEA"
os.makedirs(OUT, exist_ok=True)
FILES = {
    "Normal":  f"{DOWN}/normal_18_24/generated_data_normal_18_24.h5ad",
    "COVID19": f"{DOWN}/covid/generated_data_covid.h5ad",
}


def avg_per_group(a, status):
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    genes = a.var_names.astype(str).to_numpy()
    samp = a.obs["Sample"].astype(str).to_numpy()
    ct = pd.Series(a.obs["cell type"].astype(str).to_numpy()).str.replace("_", "-", regex=False).to_numpy()
    cid = np.array([f"{s}_{c}" for s, c in zip(samp, ct)])
    cols, meta = {}, []
    for u in pd.unique(cid):
        m = cid == u
        cols[u] = X[m].mean(axis=0)
        s = samp[m][0]; c = ct[m][0]
        meta.append((u, s, c, status, int(m.sum())))
    expr = pd.DataFrame(cols, index=genes)  # genes x columns
    md = pd.DataFrame(meta, columns=["combined_id", "Sample", "CellType", "disease_status", "n_cells"])
    return expr, md


exprs, metas = [], []
for status, fp in FILES.items():
    a = ad.read_h5ad(fp)
    e, m = avg_per_group(a, status)
    print(f"[{status}] {a.n_obs} cells -> {e.shape[1]} (sample,celltype) columns, {e.shape[0]} genes")
    exprs.append(e); metas.append(m)

# common genes (identical var here) then cbind Normal then COVID (matches script order)
common = exprs[0].index.intersection(exprs[1].index)
expr = pd.concat([exprs[0].loc[common], exprs[1].loc[common]], axis=1)
meta = pd.concat(metas, ignore_index=True)
assert list(expr.columns) == meta["combined_id"].tolist(), "column/meta order mismatch"
expr.to_csv(f"{OUT}/ssgsea_avg_expr.csv")
meta.to_csv(f"{OUT}/ssgsea_avg_meta.csv", index=False)
print(f"[saved] expr {expr.shape} -> {OUT}/ssgsea_avg_expr.csv")
print(f"[saved] meta {meta.shape} -> {OUT}/ssgsea_avg_meta.csv")
print("  columns/condition:", meta.groupby("disease_status")["combined_id"].nunique().to_dict())
print("  celltypes:", meta["CellType"].nunique(), "| samples:", meta["Sample"].nunique())
