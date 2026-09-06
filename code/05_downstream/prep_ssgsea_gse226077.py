#!/usr/bin/env python
"""Preprocess for GSE226077 ssGSEA: average log1p expression per (Sample, Cell_type) of the
Route2 generated_data.h5ad (replicating ssGSEA_GSE226077.R::calc_avg). Writes a genes x column
matrix (safe integer column keys to avoid commas in cell-type names) + a meta table that the R
script maps back to Sample / Cell_type / Status(day = word 4 of Sample)."""
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, os
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
GDIR = os.environ.get("IMGL_ROOT", str(RELEASE_ROOT / "work/downstream/GSE226077"))
OUT = f"{GDIR}/ssGSEA"; os.makedirs(OUT, exist_ok=True)
generated = os.environ.get("IMGL_H5AD", str(RELEASE_ROOT / "model_outputs/GSE226077/application_generated.h5ad"))
a = ad.read_h5ad(generated)
X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
genes = a.var_names.astype(str).to_numpy()
samp = a.obs["Sample"].astype(str).to_numpy()
ct = a.obs["Cell_type"].astype(str).to_numpy()
day = a.obs["development_stage"].astype(str).to_numpy() if "development_stage" in a.obs else \
    np.array([s.split("_")[3] for s in samp])
cid = np.array([f"{s}||{c}" for s, c in zip(samp, ct)])

MIN_CELLS = int(os.environ.get("SSGSEA_MIN_CELLS", "20"))  # drop (Sample,Cell_type) below N cells:
# a mean over too few cells is not a representative cell-type profile (and would also distort GSVA's
# cross-column normalisation). Default 20 = the chosen iMGL threshold (excludes the near-depleted
# Myeloid-progenitor & Homeostatic-non-proliferative states from the D4-vs-D0 contrast); set 0 to disable.
cols, meta = {}, []
n_drop = 0
for i, u in enumerate(pd.unique(cid)):
    m = cid == u
    if int(m.sum()) < MIN_CELLS:
        n_drop += 1; continue
    key = f"col{i}"
    cols[key] = X[m].mean(axis=0)
    meta.append((key, samp[m][0], ct[m][0], day[m][0], int(m.sum())))
if MIN_CELLS > 0:
    print(f"[filter] SSGSEA_MIN_CELLS={MIN_CELLS}: kept {len(meta)}, dropped {n_drop} (Sample,Cell_type) profiles")
expr = pd.DataFrame(cols, index=genes)
md = pd.DataFrame(meta, columns=["col_key", "Sample", "Cell_type", "Status", "n_cells"])
expr.to_csv(f"{OUT}/ssgsea_avg_expr.csv")
md.to_csv(f"{OUT}/ssgsea_avg_meta.csv", index=False)
print(f"[saved] expr {expr.shape} (genes x sample-celltype) -> {OUT}/ssgsea_avg_expr.csv")
print(f"[saved] meta {md.shape} -> {OUT}/ssgsea_avg_meta.csv")
print("  Status x Cell_type columns:\n", md.groupby(["Status", "Cell_type"]).size().unstack(fill_value=0))
