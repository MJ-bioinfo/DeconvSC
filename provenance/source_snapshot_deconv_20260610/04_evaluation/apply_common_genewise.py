#!/usr/bin/env python
"""Rewrite the all-gene gene-wise rows (gene_set=all_genes, axis=gene) of summary_all_methods.csv
to COMMON-gene-set values — the intersection of every method's reported all-gene genes (per dataset)
— so that column becomes apples-to-apples across methods. Marker gene-wise is left untouched.

Run AFTER run_eval_v3_route2.py and BEFORE the table builders, per scheme:
    WORK_ROOT=<scheme> python 04_evaluation/apply_common_genewise.py

Rationale: run_eval scores all_genes on each method's OWN gene set (n differs 10x+), which makes
CIBERSORTx (HiRes, ~1675 genes on HCA) look spuriously high. See lib/common_gene_set.py.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import common_gene_set as cg

ROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605") + "/03_result_tables"
METHODS = ["cVAE", "BayesPrism", "CIBERSORTx", "DISSECT", "TAPE"]   # storage read-keys (dirs <ds>/cVAE/)
DISPLAY = {"cVAE": "DeconvSC"}                                      # method label in the summary rows
DATASETS = ["HCA_fold2", "GSE141115", "GSE159585"]
SUMMARY = f"{ROOT}/summary_all_methods.csv"
STATCOLS = ["n", "mean", "median", "q25", "q75", "min", "max"]


def main():
    S = pd.read_csv(SUMMARY)
    for ds in DATASETS:
        common = cg.common_index(ROOT, ds, METHODS)
        if not common:
            print(f"  {ds}: no common genes — skip"); continue
        nup = 0
        for m in METHODS:
            st = cg.stats_on_common(ROOT, ds, m, common)
            if st is None:
                continue
            label = DISPLAY.get(m, m)
            mask = (S.dataset == ds) & (S.method == label) & (S.gene_set == "all_genes") & (S.axis == "gene")
            if mask.sum() == 0:                       # legacy: label still 'cVAE'
                mask = (S.dataset == ds) & (S.method == m) & (S.gene_set == "all_genes") & (S.axis == "gene")
            if mask.sum() == 0:
                continue
            for k in STATCOLS:
                S.loc[mask, k] = st[k]
            nup += 1
        print(f"  {ds}: common={len(common)} genes; all-gene gene-wise updated for {nup} methods")
    S.to_csv(SUMMARY, index=False)
    print(f"[done] -> {SUMMARY}  (all-gene gene-wise = common gene set; marker untouched)")


if __name__ == "__main__":
    main()
