#!/usr/bin/env python
"""Prepare unified benchmark inputs for fold_2 (server-side, all tools).

Fairness contract:
  - Same reference cells (97489), same gene order (25366 ENSG), same 105 bulks,
    same cell-type set (23, incl 'None'), same evaluation formulas.
  - Each tool is fed the input scale it natively expects:
      * reference EXPRESSION  -> raw counts (.raw.X)  [TAPE/DISSECT/BayesPrism all
        simulate/aggregate from counts internally]
      * query bulk            -> linear CPM (sample x gene)
  - log1p(CPT,1e4) is used ONLY at evaluation time (gt cells + unified
    pred_celltype_expr normalization), independent of tool inputs.

Outputs under BENCH_ROOT:
  inputs_manifest.json
  common/bulk_cpm_linear.tsv        (105 x 25366, sample x gene)   [copy]
  common/bulk_log1p_cpm.tsv         (105 x 25366)                  [copy, for eval]
  common/ref_counts.h5ad            (97489 x 25366, X=raw counts,
                                      obs: Celltype / cell_type / labels / donor_id)
  TAPE/ref.h5ad      -> symlink to common/ref_counts.h5ad  (obs['Celltype'])
  DISSECT/ref.h5ad   -> symlink to common/ref_counts.h5ad  (obs['cell_type'])
  BayesPrism/ ...    (written lazily by its own runner from ref_counts.h5ad)
"""
import os, json, shutil, hashlib, sys
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd

FOLD_DIR = "/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v2/fold_2"
BENCH_ROOT = "/disk1/maijl/deconv/benchmark_fold2"
LABEL_COL = "labels"   # 23 classes incl 'None'


def log(*a):
    print(*a, flush=True)


def md5_of_list(xs):
    h = hashlib.md5()
    for x in xs:
        h.update(str(x).encode())
        h.update(b"\n")
    return h.hexdigest()


def main():
    os.makedirs(os.path.join(BENCH_ROOT, "common"), exist_ok=True)

    # --- reference: build counts h5ad from .raw ---
    log("[prep] loading reference_train.h5ad (full) ...")
    ref = ad.read_h5ad(os.path.join(FOLD_DIR, "reference_train.h5ad"))
    assert ref.raw is not None, "reference has no .raw counts"
    genes = ref.var_names.tolist()
    n_genes = len(genes)
    assert n_genes == 25366, n_genes

    raw = ref.raw.X
    if not sp.issparse(raw):
        raw = sp.csr_matrix(raw)
    raw = raw.tocsr().astype(np.float32)
    # raw.var_names should align with ref.var_names; assert order matches
    raw_genes = ref.raw.var_names.tolist()
    assert raw_genes == genes, "raw var order != X var order"

    labels = ref.obs[LABEL_COL].astype(str).values
    donors = ref.obs["donor_id"].astype(str).values
    celltype_set = sorted(pd.unique(labels).tolist())
    log(f"[prep] {ref.n_obs} cells, {n_genes} genes, {len(celltype_set)} cell types")
    log(f"[prep] None cells: {(labels=='None').sum()}")

    obs = pd.DataFrame(
        {
            "Celltype": labels,      # TAPE expects obs['Celltype']
            "cell_type": labels,     # DISSECT expects obs['cell_type']
            "labels": labels,
            "donor_id": donors,
        },
        index=ref.obs_names,
    )
    ref_counts = ad.AnnData(
        X=raw,
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    out_ref = os.path.join(BENCH_ROOT, "common", "ref_counts.h5ad")
    log(f"[prep] writing {out_ref} ...")
    ref_counts.write_h5ad(out_ref)

    # raw-counts sanity
    samp = raw[:50].toarray()
    log(f"[prep] raw counts sample min/max/mean/int = "
        f"{samp.min()}/{samp.max()}/{samp.mean():.2f}/"
        f"{np.allclose(samp, np.round(samp))}")

    # --- bulks: copy linear CPM + log1p CPM ---
    for src, dst in [
        ("synthetic_bulks_cpm_linear.tsv", "bulk_cpm_linear.tsv"),
        ("synthetic_bulks_log1p_cpm.tsv", "bulk_log1p_cpm.tsv"),
    ]:
        s = os.path.join(FOLD_DIR, src)
        d = os.path.join(BENCH_ROOT, "common", dst)
        shutil.copyfile(s, d)
        log(f"[prep] copied {dst}")

    bulk_ids = pd.read_csv(
        os.path.join(BENCH_ROOT, "common", "bulk_cpm_linear.tsv"),
        sep="\t", index_col=0, usecols=[0],
    ).index.tolist()
    assert len(bulk_ids) == 105, len(bulk_ids)

    # --- per-tool dirs + ref symlinks ---
    for tool in ["TAPE", "DISSECT", "BayesPrism", "CIBERSORTx"]:
        d = os.path.join(BENCH_ROOT, tool)
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
    for tool in ["TAPE", "DISSECT"]:
        link = os.path.join(BENCH_ROOT, tool, "ref.h5ad")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.relpath(out_ref, os.path.join(BENCH_ROOT, tool)), link)

    # --- manifest ---
    manifest = {
        "fold": 2,
        "n_genes": n_genes,
        "gene_list_md5": md5_of_list(genes),
        "celltype_set": celltype_set,
        "n_celltypes": len(celltype_set),
        "none_cells_kept": True,
        "bulk_n_samples": len(bulk_ids),
        "bulk_scale": "linear_CPM",
        "ref_expr_scale_for_tools": "raw_counts",
        "eval_scale": "log1p_CPT_1e4",
        "label_col_source": LABEL_COL,
        "fold_dir": FOLD_DIR,
        "bench_root": BENCH_ROOT,
    }
    with open(os.path.join(BENCH_ROOT, "inputs_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("[prep] wrote inputs_manifest.json")
    log("[prep] DONE")


if __name__ == "__main__":
    main()
