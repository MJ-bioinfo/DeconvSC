#!/usr/bin/env python
"""Sample-identity + composition health on a REAL bulk (no GT), uniformly across all methods.

Identity = does each sample's RECONSTRUCTED expression (props . anchored basis) best match its
OWN real bulk among all samples? (diagonal-argmax of the cross-sample correlation matrix, the
same notion as downstream/solver_search_covid.py score(), here on the cached core genes so every
method is scored identically). This is the metric the reproduction manual reports for COVID
(softmax ~14%, NNLS ~43%); higher = more sample-specific composition.

Usage:
  python identity_realbulk.py --cache <cache_gse159585_real.npz> [--split covid,normal]
"""
import os, sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_lib as r2

OUTD = "/disk1/maijl/deconv/deconv_20260605/03_result_tables/unified"
METHODS = ["softmax", "nnls_core", "nnls_ridge", "dirichlet_map", "prophead", "prophead_ref"]
COVID = ['GC1003242', 'GC1003243', 'GC1003244', 'GC1003245', 'GC1003246', 'GC1003247', 'GC1003248']
NORMAL = ['GC1003234', 'GC1003233', 'GC1003230', 'GC1003229', 'GC1003226', 'GC1003225', 'GC1003221']


def identity(recon, bulk):
    """recon, bulk: (n_samples, n_genes) log1p. Center per gene across samples, diagonal-argmax %."""
    R = recon - recon.mean(0, keepdims=True); Bm = bulk - bulk.mean(0, keepdims=True)
    n = recon.shape[0]
    C = np.array([[np.corrcoef(R[i], Bm[j])[0, 1] for j in range(n)] for i in range(n)])
    return 100.0 * np.mean([C[i, i] == np.nanmax(C[i]) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="real")
    a = ap.parse_args()
    cache = {k: v for k, v in np.load(a.cache, allow_pickle=True).items()}
    names = list(cache["names"]); samples = list(cache["samples"])
    bc, yc = cache["basis_core"], cache["y_core"]               # (S,T,nc),(S,nc)
    bulk_log = np.log1p(yc)
    groups = {"all": samples,
              "covid": [s for s in samples if s in COVID],
              "normal": [s for s in samples if s in NORMAL]}
    idx = {s: i for i, s in enumerate(samples)}

    print(f"{'method':<14}{'eff#types':>10}{'n_act@2%':>10}"
          f"{'id_all':>9}{'id_covid':>10}{'id_normal':>11}")
    rows = {}
    for m in METHODS:
        f = f"{OUTD}/proportions_gse159585_{a.tag}_{m}.csv"
        if not os.path.exists(f):
            continue
        P = pd.read_csv(f, index_col=0).reindex(columns=names).fillna(0.0)
        P = P.reindex([str(s) for s in samples]).values                   # (S,T)
        recon = np.log1p(np.einsum("st,stg->sg", P, bc))                  # (S,nc) reconstructed
        cm = r2.composition_metrics(P)
        ids = {}
        for g, ss in groups.items():
            ii = [idx[s] for s in ss]
            ids[g] = identity(recon[ii], bulk_log[ii]) if len(ii) > 1 else float("nan")
        rows[m] = dict(eff=cm["eff_ntypes"], nact=cm["n_active"], **ids)
        print(f"{m:<14}{cm['eff_ntypes']:>10.1f}{cm['n_active']:>10.1f}"
              f"{ids['all']:>9.0f}{ids['covid']:>10.0f}{ids['normal']:>11.0f}")
    pd.DataFrame(rows).T.to_csv(f"{OUTD}/UNIFIED_realbulk_identity_gse159585.csv")
    print(f"\n[saved] {OUTD}/UNIFIED_realbulk_identity_gse159585.csv")


if __name__ == "__main__":
    main()
