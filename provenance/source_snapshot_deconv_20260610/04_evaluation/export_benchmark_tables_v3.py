#!/usr/bin/env python
"""Export per-dataset benchmark PCC tables (v3 Route2 tree) — MERGE-ONLY, no recompute.

Pulls verbatim values from the existing CSVs under
  /disk1/maijl/deconv/deconv_20260605/03_result_tables/<dataset>/<method>/pearson_*.csv
and reshapes them into, per dataset, one table indexed by (Method, Sample, Cell type):
  - All gene profile-wise PCC / Markers profile-wise PCC   (per (sample,celltype) row Pearson over genes)
  - All gene pseudo-bulk PCC  / Markers pseudo-bulk PCC    (per celltype row Pearson, samples re-summed)
plus each method's gene-wise PCC (all + markers), the per-gene column Pearson averaged over genes.

Granularity (per the PCC_formulas_and_aggregation.md spec):
  profile-wise = canonical 'sample_profile' (the finest row axis; one r per (sample,celltype))
  pseudo-bulk  = canonical 'pseudobulk'     (one r per celltype, joined onto each sample row)
  gene-wise    = canonical 'gene'           (one r per gene; reported as per-method mean/median)

Label handling (no PCC is recomputed; only existing numbers are used):
  * Competitors (BayesPrism/CIBERSORTx/DISSECT/TAPE) keep their labeled 3-col profile-wise files
    (HCA: pearson_sample_profile_*; GSE141115: pearson_profile_*).
  * cVAE's per-(sample,celltype) file is label-less. Where its row order matches a competitor's
    labeled file (GSE141115 -> BayesPrism, checksum-validated against cVAE's own per-celltype
    profile file), we borrow those labels and attach cVAE's saved values verbatim.
  * cVAE on HCA matches no competitor's order, so it falls back to its labeled per-celltype
    profile file with Sample='all'.
  * GSE159585 has no sample dimension at all: every method is per-celltype, Sample='-'.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import common_gene_set as cg   # all-gene gene-wise is scored on the cross-method COMMON gene set

ROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605") + "/03_result_tables"
METHODS = ["cVAE", "BayesPrism", "CIBERSORTx", "DISSECT", "TAPE"]   # storage read-key (dirs <ds>/cVAE/)
DISPLAY = {"cVAE": "DeconvSC"}   # output-label rename only (read paths keep `cVAE`); matches the figures
DATASETS = ["HCA_fold2", "GSE141115", "GSE159585"]
SAMPLE_LEVEL = {"HCA_fold2": True, "GSE141115": True, "GSE159585": False}
DONORS = ["BayesPrism", "DISSECT", "TAPE", "CIBERSORTx"]  # candidate label donors for cVAE

COL_PROF = {"all_genes": "All gene profile-wise PCC", "marker_genes": "Markers profile-wise PCC"}
COL_PB   = {"all_genes": "All gene pseudo-bulk PCC",  "marker_genes": "Markers pseudo-bulk PCC"}
COL_GW   = {"all_genes": "All gene gene-wise PCC (common set, mean)",
            "marker_genes": "Markers gene-wise PCC (per-method mean)"}
FINAL_COLS = (["Method", "Sample", "Cell type",
               COL_PROF["all_genes"], COL_PROF["marker_genes"],
               COL_PB["all_genes"], COL_PB["marker_genes"],
               COL_GW["all_genes"], COL_GW["marker_genes"]])


def rcsv(f):
    if not os.path.exists(f):
        return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None


def _clean3(df):
    d = df.iloc[:, [0, 1, -1]].copy()
    d.columns = ["sample", "cell_type", "value"]
    d["sample"] = d["sample"].astype(str)
    d["cell_type"] = d["cell_type"].astype(str)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d[d["value"].notna() & d["cell_type"].str.len().gt(0)].reset_index(drop=True)


def labeled3(ds, method, gset):
    """Competitor labeled per-(sample,celltype) df, from whichever file is 3-col."""
    for ax in ("sample_profile", "profile"):
        df = rcsv(f"{ROOT}/{ds}/{method}/pearson_{ax}_{gset}.csv")
        if df is not None and df.shape[1] >= 3:
            return _clean3(df)
    return None


def profile2col(ds, method, gset):
    """Per-celltype 2-col profile file -> [cell_type, value]."""
    df = rcsv(f"{ROOT}/{ds}/{method}/pearson_profile_{gset}.csv")
    if df is None or df.shape[1] != 2:
        return None
    d = df.iloc[:, [0, -1]].copy()
    d.columns = ["cell_type", "value"]
    d["cell_type"] = d["cell_type"].astype(str)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d[d["value"].notna() & d["cell_type"].str.len().gt(0)].reset_index(drop=True)


def borrow_labels(ds, gset):
    """cVAE: attach a donor's (sample,cell_type) labels to cVAE's label-less sample_profile
    VALUES, accepting only if groupby-celltype-mean reproduces cVAE's own profile file
    (exact checksum). Returns [sample, cell_type, value] or None."""
    sp = rcsv(f"{ROOT}/{ds}/cVAE/pearson_sample_profile_{gset}.csv")
    if sp is None or sp.shape[0] == 0:
        return None
    vals = pd.to_numeric(sp.iloc[:, -1], errors="coerce").values
    prof = profile2col(ds, "cVAE", gset)
    if prof is None:
        return None
    pmap = dict(zip(prof["cell_type"], prof["value"]))
    for donor in DONORS:
        dl = labeled3(ds, donor, gset)
        if dl is None or len(dl) != len(vals):
            continue
        chk = pd.DataFrame({"cell_type": dl["cell_type"].values, "v": vals}).groupby("cell_type")["v"].mean()
        ok = all(abs(mv - pmap[c]) < 1e-6 for c, mv in chk.items() if c in pmap and not np.isnan(pmap[c]))
        if ok:
            return pd.DataFrame({"sample": dl["sample"].values,
                                 "cell_type": dl["cell_type"].values, "value": vals})
    return None


def profilewise(ds, method, gset):
    """Return [sample, cell_type, value]; None if unavailable for this method/dataset."""
    nosamp_mark = "-" if not SAMPLE_LEVEL[ds] else "all"
    # 1) labeled 3-col (competitors with a sample dimension)
    d = labeled3(ds, method, gset)
    if d is not None:
        return d
    # 2) cVAE label-less sample_profile -> borrow validated donor labels
    if method == "cVAE" and SAMPLE_LEVEL[ds]:
        d = borrow_labels(ds, gset)
        if d is not None:
            return d
    # 3) fall back to per-celltype profile (cVAE-on-HCA; everyone on GSE159585)
    d = profile2col(ds, method, gset)
    if d is not None:
        d.insert(0, "sample", nosamp_mark)
        return d
    return None


def pseudobulk(ds, method, gset):
    df = rcsv(f"{ROOT}/{ds}/{method}/pearson_pseudobulk_{gset}.csv")
    if df is None:
        return None
    d = df.iloc[:, [0, -1]].copy()
    d.columns = ["cell_type", "value"]
    d["cell_type"] = d["cell_type"].astype(str)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d[d["value"].notna() & d["cell_type"].str.len().gt(0)].reset_index(drop=True)


def genewise(ds, method, gset):
    df = rcsv(f"{ROOT}/{ds}/{method}/pearson_gene_{gset}.csv")
    if df is None:
        return (np.nan, np.nan, 0)
    v = pd.to_numeric(df.iloc[:, -1], errors="coerce").dropna()
    if v.size == 0:
        return (np.nan, np.nan, 0)
    return (float(v.mean()), float(v.median()), int(v.size))


def build_dataset(ds):
    gw_rows, blocks, prov = [], [], []
    common = cg.common_index(ROOT, ds, METHODS)   # all-gene gene-wise: cross-method common gene set
    for m in METHODS:
        st = cg.stats_on_common(ROOT, ds, m, common)                       # all-gene = COMMON set
        gw = {"all_genes": (st["mean"], st["median"], st["n"]) if st else (np.nan, np.nan, 0),
              "marker_genes": genewise(ds, m, "marker_genes")}             # marker = own set (unchanged)
        gw_rows.append({"Method": m,
                        COL_GW["all_genes"]: gw["all_genes"][0],
                        "All gene gene-wise PCC (common set, median)": gw["all_genes"][1],
                        "n genes (all)": gw["all_genes"][2],
                        COL_GW["marker_genes"]: gw["marker_genes"][0],
                        "Markers gene-wise PCC (median)": gw["marker_genes"][1],
                        "n genes (markers)": gw["marker_genes"][2]})

        pa = profilewise(ds, m, "all_genes")
        pm = profilewise(ds, m, "marker_genes")
        if pa is None and pm is None:
            prov.append(f"    {m:11s}: ABSENT (no profile-wise file)")
            continue
        pa = pa if pa is not None else pd.DataFrame(columns=["sample", "cell_type", "value"])
        pm = pm if pm is not None else pd.DataFrame(columns=["sample", "cell_type", "value"])
        merged = pd.merge(pa.rename(columns={"value": COL_PROF["all_genes"]}),
                          pm.rename(columns={"value": COL_PROF["marker_genes"]}),
                          on=["sample", "cell_type"], how="outer")
        for g in ("all_genes", "marker_genes"):
            pb = pseudobulk(ds, m, g)
            merged = (merged.merge(pb.rename(columns={"value": COL_PB[g]}), on="cell_type", how="left")
                      if pb is not None else merged.assign(**{COL_PB[g]: np.nan}))
        merged.insert(0, "Method", m)
        merged[COL_GW["all_genes"]] = gw["all_genes"][0]
        merged[COL_GW["marker_genes"]] = gw["marker_genes"][0]
        merged = merged.rename(columns={"sample": "Sample", "cell_type": "Cell type"})
        blocks.append(merged)
        smark = sorted(merged["Sample"].unique())
        prov.append(f"    {m:11s}: rows={len(merged):5d}  Sample={'per-sample' if smark not in (['all'],['-']) else smark[0]}")

    table = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=FINAL_COLS)
    for c in FINAL_COLS:
        if c not in table.columns:
            table[c] = np.nan
    table = table[FINAL_COLS]
    table["Method"] = pd.Categorical(table["Method"], categories=METHODS, ordered=True)
    table = table.sort_values(["Method", "Cell type", "Sample"]).reset_index(drop=True)
    table["Method"] = table["Method"].astype(str).replace(DISPLAY)

    gw_df = pd.DataFrame(gw_rows)
    gw_df["Method"] = pd.Categorical(gw_df["Method"], categories=METHODS, ordered=True)
    gw_df = gw_df.sort_values("Method").reset_index(drop=True)
    gw_df["Method"] = gw_df["Method"].astype(str).replace(DISPLAY)
    return table, gw_df, prov


def main():
    tables, gws = {}, {}
    for ds in DATASETS:
        print(f"=== {ds} ===")
        t, g, prov = build_dataset(ds)
        print("\n".join(prov))
        tables[ds], gws[ds] = t, g
        t.to_csv(f"{ROOT}/TABLE_{ds}.csv", index=False)
        g.to_csv(f"{ROOT}/TABLE_{ds}_genewise_by_method.csv", index=False)
        print(f"  -> TABLE_{ds}.csv  (rows={len(t)}, methods={t['Method'].nunique()})\n")

    with pd.ExcelWriter(f"{ROOT}/benchmark_tables.xlsx", engine="openpyxl") as xw:
        for ds in DATASETS:
            tables[ds].round(4).to_excel(xw, sheet_name=ds[:31], index=False)
        allgw = []
        for ds in DATASETS:
            gg = gws[ds].copy(); gg.insert(0, "Dataset", ds); allgw.append(gg)
        pd.concat(allgw, ignore_index=True).round(4).to_excel(xw, sheet_name="genewise_by_method", index=False)
    print(f"[done] -> {ROOT}/  (TABLE_<ds>.csv, TABLE_<ds>_genewise_by_method.csv, benchmark_tables.xlsx)")


if __name__ == "__main__":
    main()
