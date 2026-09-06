#!/usr/bin/env python
"""Build held-out-cell PSEUDOBULKS with KNOWN ground-truth proportions for the 62-type GSE
datasets (the real bulk has no GT). The GSE159585 test set is a single held-out donor (LUNG09,
61 types) — ideal for a PROPORTION benchmark because expression profiles are fixed and only the
composition varies. Three splits separate "fits its own prior" from genuine generalisation:

  real : the held-out donor's true composition (1 sample, realistic in-vivo).
  rreal: compositions drawn from Dir(kappa * f_donor) — realistic-SHAPED, varied (near in-vivo).
  rbroad: compositions drawn from Dir(alpha) — broad/shifted, unlike the donor -> distribution shift.
All cells are sampled from the held-out donor (no train leakage). GT = realised cell fractions.

Outputs (genes x samples TSV matching the real-bulk format + samples x types GT):
  <OUTD>/pb_<ds>_bulk.tsv , pb_<ds>_gt.csv , pb_<ds>_split.csv

Usage:
  python make_pseudobulk_gt.py --dataset gse159585
"""
import os, argparse, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

WORKROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605")  # deconv_20260610: outputs root (default = established working tree)
OUTD = f"{WORKROOT}/03_result_tables/unified/pseudobulk"
DS = {
    "gse159585": dict(test="/disk1/maijl/deconv/data/GSE159585/GSE159585_testset_level2celltype.h5ad",
                      celltype="cell type", donor="Sample ID"),
    "gse141115": dict(test="/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_test.h5ad",
                      celltype="cell type", donor="sample"),
    "hra000917": dict(test="/disk1/maijl/deconv/cVAE/HRA000917/ref/HRA000917_test.h5ad",
                      celltype="cell_type", donor="sample"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DS))
    ap.add_argument("--n_rand", type=int, default=30, help="per rand split (rreal + rbroad)")
    ap.add_argument("--cells_per", type=int, default=1200)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dir conc for rbroad (shifted)")
    ap.add_argument("--kappa", type=float, default=60.0, help="Dir conc multiplier for rreal (realistic-shaped)")
    ap.add_argument("--min_cells_type", type=int, default=20)
    a = ap.parse_args()
    c = DS[a.dataset]; os.makedirs(OUTD, exist_ok=True)
    rng = np.random.default_rng(0)

    adata = ad.read_h5ad(c["test"])
    genes = adata.var_names.astype(str).to_numpy()
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    xmax = float(X[:min(2000, X.shape[0])].max())
    print(f"[{a.dataset}] test cells={X.shape[0]} genes={len(genes)} X.max(sample)={xmax:.2f} "
          f"({'raw-counts' if xmax > 30 else 'CAUTION: looks normalised/log'})")
    ct = adata.obs[c["celltype"]].astype(str).values
    types = sorted(np.unique(ct))
    by_type = {t: np.where(ct == t)[0] for t in types}
    by_type = {t: ix for t, ix in by_type.items() if len(ix) >= a.min_cells_type}
    types = list(by_type)
    f_donor = np.array([len(by_type[t]) for t in types], np.float64); f_donor /= f_donor.sum()
    print(f"[{a.dataset}] usable types={len(types)} (>= {a.min_cells_type} cells); "
          f"donor realistic composition entropy={-(f_donor*np.log(f_donor+1e-12)).sum():.2f}")

    bulks, gts, splits, snames = [], [], [], []

    def pseudobulk(idx):
        return np.asarray(X[idx].sum(axis=0)).ravel()

    def sample_from(counts):
        idx = np.concatenate([rng.choice(by_type[types[j]], size=counts[j], replace=True)
                              for j in range(len(types)) if counts[j] > 0])
        return pseudobulk(idx), counts / counts.sum()

    # ---- real: held-out donor's true composition (1 sample) ----
    counts0 = np.maximum(np.round(f_donor * a.cells_per).astype(int), 0)
    b, g = sample_from(counts0); bulks.append(b); gts.append(g); splits.append("real"); snames.append("real_donor")

    # ---- rreal: realistic-shaped compositions ~ Dir(kappa * f_donor) ----
    for i in range(a.n_rand):
        p = rng.dirichlet(a.kappa * f_donor + 1e-3)
        counts = np.maximum(np.round(p * a.cells_per).astype(int), 0)
        if counts.sum() == 0:
            continue
        b, g = sample_from(counts); bulks.append(b); gts.append(g)
        splits.append("rreal"); snames.append(f"rreal_{i:03d}")

    # ---- rbroad: shifted/broad compositions ~ Dir(alpha) ----
    for i in range(a.n_rand):
        p = rng.dirichlet([a.alpha] * len(types))
        counts = np.maximum(np.round(p * a.cells_per).astype(int), 0)
        if counts.sum() == 0:
            continue
        b, g = sample_from(counts); bulks.append(b); gts.append(g)
        splits.append("rbroad"); snames.append(f"rbroad_{i:03d}")

    bulk_df = pd.DataFrame(np.vstack(bulks).T, index=genes, columns=snames)   # genes x samples (raw)
    bulk_df.index.name = "gene"
    gt_df = pd.DataFrame(np.vstack(gts), index=snames, columns=types); gt_df.index.name = "sample"
    split_df = pd.Series(splits, index=snames, name="split")
    bulk_df.to_csv(f"{OUTD}/pb_{a.dataset}_bulk.tsv", sep="\t")
    gt_df.to_csv(f"{OUTD}/pb_{a.dataset}_gt.csv")
    split_df.to_csv(f"{OUTD}/pb_{a.dataset}_split.csv")
    print(f"[saved] {len(snames)} pseudobulks "
          f"(real={splits.count('real')} rreal={splits.count('rreal')} rbroad={splits.count('rbroad')}), "
          f"{len(types)} types -> {OUTD}/pb_{a.dataset}_*")


if __name__ == "__main__":
    main()
