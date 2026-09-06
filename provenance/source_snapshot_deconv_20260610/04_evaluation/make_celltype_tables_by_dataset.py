#!/usr/bin/env python
"""Per-dataset (Method, Cell type) tables, built from the per-metric CSVs under
   03_result_tables/<dataset>/<method>/pearson_*.csv   (no recompute).

Layout per dataset (3 datasets -> 3 CSVs + 3 sheets in one xlsx):

  [ left block: per (Method, Cell type) ]                       (blank cols)  [ right block: per Method ]
  Method | Cell type | profile-wise all/marker | pseudo-bulk all/marker        Method | sample-wise all/marker | gene-wise all/marker

Why the split: profile-wise and pseudo-bulk are per cell type, so they get one row per
(Method, Cell type). sample-wise (per sample; sums over cell types) and gene-wise (per gene)
have NO per-cell-type value, so each is a single per-method number placed in the side block.

Values (verbatim / simple mean of existing PCC numbers — no PCC is recomputed):
  profile-wise (per cell type) = pearson_profile_<gset>.csv. If that file is per (sample,celltype)
       (3-col; GSE141115 competitors), it is averaged over samples per cell type (the canonical
       per-cell-type aggregation). cVAE & HCA/GSE159585 files are already per cell type (2-col).
  pseudo-bulk  (per cell type) = pearson_pseudobulk_<gset>.csv (2-col), verbatim.
  sample-wise  (per method)    = mean AND median over pearson_sample_<gset>.csv (per-sample rows).
  gene-wise    (per method)    = mean AND median over pearson_gene_<gset>.csv (per-gene rows).

The per-method side block now reports both the mean and the median (suffixed " [mean]" /
" [median]"), so the boxplot figures — whose central line is the median while the annotated
number is the mean — reconcile against the table either way. The left (per cell type) block
stays a single point value per (method, cell type): there is no distribution to take a median
over at that granularity (the across-cell-type mean/median lives in summary_table_<ds>.csv).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import common_gene_set as cg   # all-gene gene-wise is scored on the cross-method COMMON gene set

ROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605") + "/03_result_tables"
METHODS = ["cVAE", "BayesPrism", "CIBERSORTx", "DISSECT", "TAPE"]   # storage read-key (dirs <ds>/cVAE/)
# Output label rename: the per-method storage directory stays `cVAE` (the route2/run_eval key), but
# the human-facing method label in every table is `DeconvSC` (matches the figures). Applied to the
# Method column at output only — never to the read paths.
DISPLAY = {"cVAE": "DeconvSC"}
DATASETS = ["HCA_fold2", "GSE141115", "GSE159585"]
ND = 4

H_PROF = ("All gene profile-wise PCC", "Marker gene profile-wise PCC")   # left block: per (Method, Cell type) point value
H_PB   = ("All gene pseudo-bulk PCC",  "Marker gene pseudo-bulk PCC")    # left block: per (Method, Cell type) point value
H_SAMP = ("All gene sample-wise PCC",  "Marker gene sample-wise PCC")    # right block: per Method, gets [mean]/[median]
H_GENE = ("All gene gene-wise PCC (common set)", "Marker gene gene-wise PCC")  # all-gene = COMMON cross-method gene set


def _pair(base):
    """('<base> [mean]', '<base> [median]') header pair."""
    return (f"{base} [mean]", f"{base} [median]")


def rcsv(f):
    return pd.read_csv(f) if os.path.exists(f) else None


def _rename_none(ct):
    """The HLCA placeholder 'None'/NA group -> a readable 'Others' label (kept, not dropped)."""
    return "Others" if str(ct).strip().lower() in ("none", "na", "nan") else str(ct)


def profile_per_ct(ds, m, gset):
    """{cell_type: profile-wise PCC} per cell type."""
    df = rcsv(f"{ROOT}/{ds}/{m}/pearson_profile_{gset}.csv")
    if df is None:
        return {}
    if df.shape[1] >= 3:                       # (sample, cell_type, r) -> mean over samples
        d = df.iloc[:, [1, -1]].copy()
    else:                                      # (cell_type, r)
        d = df.iloc[:, [0, -1]].copy()
    d.columns = ["cell_type", "v"]
    d["cell_type"] = d["cell_type"].astype(str).map(_rename_none)
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    d = d[d["v"].notna() & d["cell_type"].str.len().gt(0)]
    return {ct: round(float(v), ND) for ct, v in d.groupby("cell_type")["v"].mean().items()}


def pb_per_ct(ds, m, gset):
    df = rcsv(f"{ROOT}/{ds}/{m}/pearson_pseudobulk_{gset}.csv")
    if df is None:
        return {}
    d = df.iloc[:, [0, -1]].copy()
    d.columns = ["cell_type", "v"]
    d["cell_type"] = d["cell_type"].astype(str).map(_rename_none)
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    d = d[d["v"].notna() & d["cell_type"].str.len().gt(0)]
    return {ct: round(float(v), ND) for ct, v in zip(d["cell_type"], d["v"])}


def axis_method_stat(ds, m, gset, axis):
    """Per-method (mean, median) over a per-row metric file (sample / gene)."""
    df = rcsv(f"{ROOT}/{ds}/{m}/pearson_{axis}_{gset}.csv")
    if df is None:
        return np.nan, np.nan
    v = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
    if not v.size:
        return np.nan, np.nan
    return round(float(v.mean()), ND), round(float(v.median()), ND)


def main():
    out_tables = {}
    for ds in DATASETS:
        # ---- left block: per (Method, Cell type) ----
        rows = []
        for m in METHODS:
            pa, pm = profile_per_ct(ds, m, "all_genes"), profile_per_ct(ds, m, "marker_genes")
            ba, bm = pb_per_ct(ds, m, "all_genes"), pb_per_ct(ds, m, "marker_genes")
            cts = sorted(set(pa) | set(pm) | set(ba) | set(bm))
            for ct in cts:
                rows.append([m, ct, pa.get(ct, np.nan), pm.get(ct, np.nan),
                             ba.get(ct, np.nan), bm.get(ct, np.nan)])
        left_df = pd.DataFrame(rows, columns=["Method", "Cell type",
                                              H_PROF[0], H_PROF[1], H_PB[0], H_PB[1]])
        left_df["Method"] = pd.Categorical(left_df["Method"], categories=METHODS, ordered=True)
        left_df = left_df.sort_values(["Method", "Cell type"]).reset_index(drop=True)
        left_df["Method"] = left_df["Method"].astype(str).replace(DISPLAY)

        # ---- right block: per Method (sample-wise + gene-wise), each with [mean] and [median] ----
        right_cols = (["Method"] + list(_pair(H_SAMP[0])) + list(_pair(H_SAMP[1]))
                      + list(_pair(H_GENE[0])) + list(_pair(H_GENE[1])))
        right_rows = []
        common = cg.common_index(ROOT, ds, METHODS)   # all-gene gene-wise: cross-method common gene set
        for m in METHODS:
            sa = axis_method_stat(ds, m, "all_genes", "sample")
            sm = axis_method_stat(ds, m, "marker_genes", "sample")
            ga_st = cg.stats_on_common(ROOT, ds, m, common)                   # all-gene = COMMON set
            ga = (round(ga_st["mean"], ND), round(ga_st["median"], ND)) if ga_st else (np.nan, np.nan)
            gm = axis_method_stat(ds, m, "marker_genes", "gene")             # marker = unchanged (own set)
            right_rows.append([m, *sa, *sm, *ga, *gm])
        right_df = pd.DataFrame(right_rows, columns=right_cols)
        right_df["Method"] = right_df["Method"].replace(DISPLAY)

        # ---- side-by-side (2 blank spacer cols); right block fills the first 5 rows ----
        n = len(left_df)
        blank = pd.DataFrame({" ": [""] * n, "  ": [""] * n})
        right_pad = right_df.reindex(range(n)).reset_index(drop=True)
        out = pd.concat([left_df, blank, right_pad], axis=1)
        out_tables[ds] = out
        out.to_csv(f"{ROOT}/celltype_table_{ds}.csv", index=False)
        print(f"  {ds:12s} -> celltype_table_{ds}.csv  (left {n} rows; right {len(right_df)} methods)")

    with pd.ExcelWriter(f"{ROOT}/celltype_tables_by_dataset.xlsx", engine="openpyxl") as xw:
        from openpyxl.utils import get_column_letter
        for ds in DATASETS:
            out_tables[ds].to_excel(xw, sheet_name=ds[:31], index=False)
            wsv = xw.sheets[ds[:31]]
            for ci, col in enumerate(out_tables[ds].columns, start=1):
                wsv.column_dimensions[get_column_letter(ci)].width = 26 if ci == 2 else 13
                if str(col).strip() == "":
                    wsv.cell(row=1, column=ci).value = None
    print(f"[done] -> {ROOT}/celltype_tables_by_dataset.xlsx (+ 3 celltype_table_<ds>.csv)")


if __name__ == "__main__":
    main()
