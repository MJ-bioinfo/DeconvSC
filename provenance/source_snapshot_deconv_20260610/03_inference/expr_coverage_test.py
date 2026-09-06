#!/usr/bin/env python
"""Proportion-FREE per-type expression PCC + coverage, for ANY of the 3 eval datasets.
Generalises expr_coverage_test_gse159585.py. See that file's header for the logic:
per-type profile PCC is computed directly from the anchored basis (no proportions) and is
therefore independent of the proportion solver; only COVERAGE (count>=1) depends on it.

  python expr_coverage_test.py --dataset hca_fold2
  python expr_coverage_test.py --dataset gse141115
"""
import os, sys, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proportions_unified as PU
import run_eval_HCA_fold2 as R
import anndata as ad, scipy.sparse as sp

DEV = PU.DEV
OUTD = "/disk1/maijl/deconv/deconv_20260605/03_result_tables/unified"
DSGT = {
    "hca_fold2": dict(gt="/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_validation_cells.h5ad",
                      gt_cell="Cell_type", cps=2000, prop_tag=""),
    "gse141115": dict(gt="/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_test.h5ad",
                      gt_cell="cell type", cps=1600, prop_tag="pb"),
    "gse159585": dict(gt="/disk1/maijl/deconv/data/GSE159585/GSE159585_testset_level2celltype.h5ad",
                      gt_cell="cell type", cps=2000, prop_tag="real"),
}


def norm_rows(M):
    s = M.sum(1, keepdims=True); s[s == 0] = np.nan
    return np.log1p(np.nan_to_num(M / s) * 1e4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=list(DSGT))
    ds = ap.parse_args().dataset; d = DSGT[ds]; cfg = PU.CONFIGS[ds]
    B = PU.assemble_hca(cfg) if cfg["builder"] == "hca" else PU.assemble_gse(cfg)
    net, cent_mat = B["net"], B["cent_mat"]; net.eval()
    genes, core, names, n_types = B["genes"], B["core"], B["names"], B["n_types"]
    bulk, samples = B["bulk"], B["samples"]
    # subsample query bulks for the per-type mean profile (stable; avoids 105x slowdown on HCA)
    qs = samples if len(samples) <= 40 else list(np.random.default_rng(0).choice(samples, 40, replace=False))

    acc = np.zeros((n_types, len(genes)), np.float64)
    for s in qs:
        blog = (bulk.loc[s].values.astype(np.float32) if B["bulk_islog"]
                else np.log1p(bulk.loc[s].values.astype(np.float32)))
        bcore = torch.tensor(blog[core], device=DEV).unsqueeze(0)
        with torch.no_grad():
            dl = {t: net(bcore, torch.tensor([t], device=DEV)).reshape(-1) for t in range(n_types)}
            basis = torch.stack([torch.expm1(torch.clamp(cent_mat[t] + dl[t], min=0)) for t in range(n_types)])
        acc += basis.cpu().numpy().astype(np.float64)
    pred = pd.DataFrame(norm_rows(acc / len(qs)), index=names, columns=genes)

    gt = ad.read_h5ad(d["gt"]); gt.obs["__c"] = gt.obs[d["gt_cell"]].astype(str).values
    X = gt.X.tocsr() if sp.issparse(gt.X) else sp.csr_matrix(gt.X)
    ggenes = gt.var_names.astype(str).to_numpy(); rows = {}
    for t in sorted(gt.obs["__c"].unique()):
        rows[t] = np.asarray(X[gt.obs["__c"].values == t].sum(0)).ravel()
    GT = pd.DataFrame(norm_rows(np.vstack([rows[t] for t in rows])), index=list(rows), columns=ggenes)

    cg = [g for g in genes if g in set(ggenes)]; ct = [c for c in names if c in GT.index]
    P, G = pred.loc[ct, cg], GT.loc[ct, cg]
    markers = R.derive_marker_genes(gt, "__c", top_n=30)
    muni = [g for g in sorted({x for v in markers.values() for x in v}) if g in cg]

    def prof_pcc(cols):
        A, Bm = P[cols].values, G[cols].values; o = {}
        for i, c in enumerate(ct):
            if A[i].std() > 1e-9 and Bm[i].std() > 1e-9: o[c] = np.corrcoef(A[i], Bm[i])[0, 1]
        return pd.Series(o)

    def gene_pcc(cols):
        A, Bm = P[cols].values, G[cols].values; v = []
        for j in range(len(cols)):
            if A[:, j].std() > 1e-9 and Bm[:, j].std() > 1e-9: v.append(np.corrcoef(A[:, j], Bm[:, j])[0, 1])
        return float(np.nanmedian(v))

    pa, pm = prof_pcc(cg), prof_pcc(muni)
    tbl = pd.DataFrame({"profile_PCC_all": pa, "profile_PCC_marker": pm}).sort_values("profile_PCC_all")
    tbl.to_csv(f"{OUTD}/EXPR_per_celltype_PCC_{ds}_propfree.csv")
    print(f"[{ds}] proportion-FREE expression reconstruction — {len(ct)} types evaluable "
          f"(model {n_types}, GT {GT.shape[0]})")
    print(f"  profile-wise PCC (all genes):  median={pa.median():.3f}  min={pa.min():.3f}  n>=0.7: {(pa>=0.7).sum()}/{len(pa)}")
    print(f"  profile-wise PCC (markers):    median={pm.median():.3f}")
    print(f"  gene-wise PCC: all={gene_pcc(cg):.3f}  marker={gene_pcc(muni):.3f}")
    print(f"  worst 5: {dict(tbl['profile_PCC_all'].head(5).round(3))}")

    # coverage from proportion CSVs (if present)
    thr = 1.0 / d["cps"]; tag = f"_{d['prop_tag']}" if d["prop_tag"] else ""
    print(f"  coverage (#types count>=1 at CPS={d['cps']}):")
    for m in ["softmax", "nnls_core", "prophead", "dirichlet_map"]:
        f = f"{OUTD}/proportions_{ds}{tag}_{m}.csv"
        if os.path.exists(f):
            Pp = pd.read_csv(f, index_col=0)
            per = (Pp > thr).sum(axis=1); union = int((Pp > thr).any(axis=0).sum())
            print(f"    {m:<13} per-sample={per.mean():.1f}  union={union}/{Pp.shape[1]}")
    print(f"[saved] {OUTD}/EXPR_per_celltype_PCC_{ds}_propfree.csv")


if __name__ == "__main__":
    main()
