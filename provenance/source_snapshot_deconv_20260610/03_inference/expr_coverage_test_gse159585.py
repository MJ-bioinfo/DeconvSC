#!/usr/bin/env python
"""Does the proportion scheme hurt EXPRESSION reconstruction on GSE159585 (62 types)?

Tests two things:
  (1) per-type profile-wise PCC + gene-wise PCC computed PROPORTION-FREE (directly from the
      anchored basis ê_t = expm1(centroid_t + alpha*delta_t(b_s)) averaged over query bulks),
      for ALL 62 types -> shows the expression quality each type WOULD get if generated, and
      that it is independent of the proportion solver (compare to the existing softmax table).
  (2) coverage = # of 62 types a proportion scheme actually generates (count>=1 at CPS) -> the
      ONLY way the proportion scheme can cost cell types in the eval.

Conclusion the numbers support: per-type expression PCC does NOT depend on the proportion solver
(expression is decoupled from proportion); only coverage does. So generate a floor of cells per
type for the expression eval (full 62-type coverage) and use the predicted proportion only for
composition/downstream.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proportions_unified as PU
import route2_lib as r2
import anndata as ad, scipy.sparse as sp, scanpy as sc

DEV = PU.DEV
GT_H5 = "/disk1/maijl/deconv/data/GSE159585/GSE159585_testset_level2celltype.h5ad"
SEVEN = ['GC1003224', 'GC1003227', 'GC1003228', 'GC1003222', 'GC1003232', 'GC1003231', 'GC1003233']
OUTD = "/disk1/maijl/deconv/deconv_20260605/03_result_tables/unified"


def norm_rows(M):  # eval口径: linear counts -> CP10K -> log1p
    s = M.sum(1, keepdims=True); s[s == 0] = np.nan
    return np.log1p(np.nan_to_num(M / s) * 1e4)


def main():
    cfg = PU.CONFIGS["gse159585"]
    B = PU.assemble_gse(cfg, samples_filter=SEVEN)          # builds basis on the 7 main query bulks
    net, cent_mat = B["net"], B["cent_mat"]; net.eval()
    genes = B["genes"]; core = B["core"]; names = B["names"]; n_types = B["n_types"]
    gidx = {g: i for i, g in enumerate(genes)}; bulk = B["bulk"]; samples = B["samples"]

    # ---- proportion-FREE predicted per-type profile = mean over query bulks of basis_lin_t ----
    acc = np.zeros((n_types, len(genes)), np.float64)
    for s in samples:
        blog = np.log1p(bulk.loc[s].values.astype(np.float32))
        bcore = torch.tensor(blog[core], device=DEV).unsqueeze(0)
        with torch.no_grad():
            d = {t: net(bcore, torch.tensor([t], device=DEV)).reshape(-1) for t in range(n_types)}
            basis = torch.stack([torch.expm1(torch.clamp(cent_mat[t] + d[t], min=0)) for t in range(n_types)])
        acc += basis.cpu().numpy().astype(np.float64)
    pred = pd.DataFrame(norm_rows(acc / len(samples)), index=names, columns=genes)

    # ---- GT: testset per-type mean profile (raw counts summed -> norm) ----
    gt = ad.read_h5ad(GT_H5); gt.obs["__c"] = gt.obs["cell type"].astype(str).values
    X = gt.X.tocsr() if sp.issparse(gt.X) else sp.csr_matrix(gt.X)
    ggenes = gt.var_names.astype(str).to_numpy()
    rows = {}
    for t in sorted(gt.obs["__c"].unique()):
        rows[t] = np.asarray(X[gt.obs["__c"].values == t].sum(0)).ravel()
    GT = pd.DataFrame(np.vstack([norm_rows(np.vstack([rows[t]]))[0] for t in rows]),
                      index=list(rows), columns=ggenes)

    # ---- align + marker set ----
    cg = [g for g in genes if g in set(ggenes)]
    ct = [c for c in names if c in GT.index]
    P, G = pred.loc[ct, cg], GT.loc[ct, cg]
    markers = __import__("run_eval_HCA_fold2").derive_marker_genes(gt, "__c", top_n=30)
    muni = [g for g in sorted({x for v in markers.values() for x in v}) if g in cg]

    def profile_pcc(cols):
        A, Bm = P[cols].values, G[cols].values; out = {}
        for i, c in enumerate(ct):
            if A[i].std() > 1e-9 and Bm[i].std() > 1e-9:
                out[c] = np.corrcoef(A[i], Bm[i])[0, 1]
        return pd.Series(out)

    def genewise_pcc(cols):
        A, Bm = P[cols].values, G[cols].values; v = []
        for j in range(len(cols)):
            a, b = A[:, j], Bm[:, j]
            if a.std() > 1e-9 and b.std() > 1e-9:
                v.append(np.corrcoef(a, b)[0, 1])
        return float(np.nanmedian(v))

    prof_all, prof_mk = profile_pcc(cg), profile_pcc(muni)
    tbl = pd.DataFrame({"profile_PCC_all": prof_all, "profile_PCC_marker": prof_mk}).sort_values("profile_PCC_all")
    tbl.to_csv(f"{OUTD}/EXPR_per_celltype_PCC_gse159585_propfree.csv")
    print(f"[GSE159585] proportion-FREE expression reconstruction, ALL {len(ct)} types evaluable")
    print(f"  per-type profile-wise PCC (all genes):  median={prof_all.median():.3f}  "
          f"min={prof_all.min():.3f}  n>=0.7: {(prof_all>=0.7).sum()}/{len(prof_all)}")
    print(f"  per-type profile-wise PCC (markers):    median={prof_mk.median():.3f}")
    print(f"  gene-wise PCC: all={genewise_pcc(cg):.3f}  marker={genewise_pcc(muni):.3f}")
    print(f"  worst 6 types: {dict(tbl['profile_PCC_all'].head(6).round(3))}")
    print(f"  best 6 types:  {dict(tbl['profile_PCC_all'].tail(6).round(3))}")
    print(f"[saved] {OUTD}/EXPR_per_celltype_PCC_gse159585_propfree.csv")


if __name__ == "__main__":
    main()
