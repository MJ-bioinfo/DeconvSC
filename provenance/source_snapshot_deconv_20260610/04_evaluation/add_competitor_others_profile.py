#!/usr/bin/env python
"""Recompute the PROFILE-axis per-cell-type PCC for the competitors (BayesPrism/DISSECT/TAPE) from
their own per-(sample,celltype) predicted expression (benchmark_fold2/<m>/results/pred_celltype_expr.h5ad),
using run_eval_v3_route2's exact functions. This recovers the 'Others' (=HLCA 'None') group that the
reused competitor profile CSVs were missing. Verifies the 20 shared types match the published CSV;
with --apply, writes the 'Others' row into each competitor's pearson_profile_{all,marker}_genes.csv.
"""
import os, sys, argparse, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import run_eval_v3_route2 as RE   # norm, sct_pseudobulk, gt_pseudobulk, markers_union, axes_vectors

WORK = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260610/prophead")
TB = f"{WORK}/03_result_tables/HCA_fold2"
BF = "/disk1/maijl/deconv/benchmark_fold2"
GT_H5AD = "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_validation_cells.h5ad"
COMP = {"BayesPrism": f"{BF}/BayesPrism/results/pred_celltype_expr.h5ad",
        "DISSECT":    f"{BF}/DISSECT/results/pred_celltype_expr.h5ad",
        "TAPE":       f"{BF}/TAPE/results/pred_celltype_expr.h5ad"}


def detect_is_log(h5):
    a = ad.read_h5ad(h5, backed="r"); X = a.X[:200] if a.n_obs > 200 else a.X[:]
    mx = float(np.asarray(X.todense()).max() if sp.issparse(X) else np.asarray(X).max())
    return mx < 30.0   # log1p-CP10K stays < ~12; linear CPM/counts >> 30


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    GT, gt_ad = RE.gt_pseudobulk(GT_H5AD, "pseudo_bulk_id", "Cell_type", is_raw=False)
    markers = RE.markers_union(gt_ad, "__c", top=30)
    genesets = {"all_genes": list(GT.columns), "marker_genes": markers}
    for m, h5 in COMP.items():
        is_log = detect_is_log(h5)
        P = RE.sct_pseudobulk(h5, "pseudo_bulk_id", "Cell_type", is_log=is_log)
        print(f"\n===== {m}  (is_log={is_log}, P cells={P.shape[0]}) =====")
        for gs, genes in genesets.items():
            prof = RE.axes_vectors(P, GT, genes, sample_level=True)["profile"]   # per-cell-type, incl 'Others'
            csv = f"{TB}/{m}/pearson_profile_{gs}.csv"
            pub = pd.read_csv(csv); pub.columns = ["cell_type", "profile"]
            pub_map = dict(zip(pub.cell_type.astype(str), pub.profile.astype(float)))
            shared = [c for c in prof.index if c in pub_map and c != "Others"]
            re_v = np.array([prof[c] for c in shared]); pb_v = np.array([pub_map[c] for c in shared])
            corr = np.corrcoef(re_v, pb_v)[0, 1] if len(shared) > 2 else np.nan
            mad = float(np.abs(re_v - pb_v).mean())
            others = float(prof["Others"]) if "Others" in prof.index else np.nan
            print(f"  {gs:12s}: recompute vs published on {len(shared)} shared types -> corr={corr:.4f} MAD={mad:.4f} | NEW Others={others:.4f} (published has Others={'Others' in pub_map})")
            if a.apply and np.isfinite(others) and "Others" not in pub_map:
                out = pub[pub.cell_type != "Others"].copy()
                out = pd.concat([out, pd.DataFrame([{"cell_type": "Others", "profile": round(others, 6)}])], ignore_index=True)
                out.to_csv(csv, index=False)
                print(f"     [applied] appended Others -> {csv}")
    if not a.apply:
        print("\n(report only; re-run with --apply to write the Others rows)")


if __name__ == "__main__":
    main()
