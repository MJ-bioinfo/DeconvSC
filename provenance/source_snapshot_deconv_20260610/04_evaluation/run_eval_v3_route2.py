#!/usr/bin/env python
"""v3 evaluation update: re-evaluate with cVAE = Route 2 generated cells, reusing the
exact unified-Pearson axis definitions of run_eval_HCA_fold2.py / extended_multi_method_
evaluation.py. The 4 competing methods (BayesPrism/CIBERSORTx/DISSECT/TAPE) are reused
verbatim from 03_results/<dataset>/<method>/pearson_*.csv; only cVAE is recomputed on the
Route2 cells (cells -> pseudobulk -> axes, the pipeline path). Outputs to a NEW v3 tree:
  03_results_v3_route2/<dataset>/{cVAE,<others>}/pearson_*.csv  + summary_all_methods.csv
  03_result_figures/figures/*.{pdf,png,svg}
"""
import os, sys, glob, shutil, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import anndata as ad, scipy.sparse as sp, scanpy as sc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local run_eval_HCA_fold2
import run_eval_HCA_fold2 as R  # reuse derive_marker_genes (has a small-group fallback)
import matplotlib; matplotlib.use("Agg")
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)
import matplotlib.pyplot as plt
try: import seaborn as sns; HAVE_SNS = True
except Exception: HAVE_SNS = False

# (legacy 20260528 eval tree no longer referenced; competitors come from COMPETITOR_ROOT below)
WORKROOT = os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605")  # deconv_20260610: where Route2 generated cells were written (must match generation stage)
# competitors (BayesPrism/CIBERSORTx/DISSECT/TAPE) reused verbatim from the deconv_20260605 table tree (NOT the 20260528 tree)
COMPETITOR_ROOT = os.environ.get("COMPETITOR_ROOT", "/disk1/maijl/deconv/deconv_20260605/03_result_tables")
SRC_RESULTS = COMPETITOR_ROOT
OUT_RESULTS = f"{WORKROOT}/03_result_tables"
OUT_FIG = f"{WORKROOT}/03_result_figures/figures"
os.makedirs(OUT_FIG, exist_ok=True)
PALETTE = {"cVAE": "#D7301F", "BayesPrism": "#1B9E77", "CIBERSORTx": "#7570B3", "DISSECT": "#E7298A", "TAPE": "#E6AB02"}
OTHERS = ["BayesPrism", "CIBERSORTx", "DISSECT", "TAPE"]
AXES4 = ["sample", "profile", "gene", "pseudobulk"]


def norm(pb):
    s = pb.sum(axis=1).replace(0, np.nan)
    return np.log1p(pb.div(s, axis=0).fillna(0.0) * 1e4)


def rename_none(arr):
    """Rename the HLCA placeholder cell type 'None'/NA -> 'Others' (a readable label kept in all
    outputs). Vectorised over a str array; applied to both generated and GT cell labels so the
    per-(sample,cell_type) pearson_*.csv carry 'Others' instead of 'None' for every method."""
    s = pd.Series(arr, dtype=str).str.strip()
    return s.mask(s.str.lower().isin(["none", "na", "nan"]), "Others").values


def sct_pseudobulk(h5ad, sample_col, cell_col, is_log=True):
    """generated cells -> (sample,cell_type) x gene counts-like (expm1 of log, summed)."""
    a = ad.read_h5ad(h5ad)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X); X = X.copy()
    if is_log: X.data = np.expm1(np.clip(X.data, 0, None))
    s = a.obs[sample_col].astype(str).values; c = rename_none(a.obs[cell_col].astype(str).values)
    grp = np.array([f"{si}\t{ci}" for si, ci in zip(s, c)]); u, inv = np.unique(grp, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(grp), np.float32), (inv, np.arange(len(grp)))), shape=(len(u), len(grp)))
    df = pd.DataFrame((G @ X).toarray(), index=u, columns=a.var_names.astype(str))
    pp = df.index.str.split("\t", expand=True)
    df.index = pd.MultiIndex.from_arrays([pp.get_level_values(0), pp.get_level_values(1)], names=["sample", "cell_type"])
    return df


def gt_pseudobulk(h5ad, sample_col, cell_col, is_raw):
    a = ad.read_h5ad(h5ad); a.obs["__c"] = rename_none(a.obs[cell_col].astype(str).values)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, np.float32)); X = X.copy()
    if not is_raw: X.data = np.expm1(np.clip(X.data, 0, None))
    genes = a.var_names.astype(str)
    if sample_col and sample_col in a.obs.columns:
        s = a.obs[sample_col].astype(str).values
        grp = np.array([f"{si}\t{ci}" for si, ci in zip(s, a.obs["__c"].values)])
        u, inv = np.unique(grp, return_inverse=True)
        G = sp.csr_matrix((np.ones(len(grp), np.float32), (inv, np.arange(len(grp)))), shape=(len(u), len(grp)))
        df = pd.DataFrame((G @ X).toarray(), index=u, columns=genes)
        pp = df.index.str.split("\t", expand=True)
        df.index = pd.MultiIndex.from_arrays([pp.get_level_values(0), pp.get_level_values(1)], names=["sample", "cell_type"])
        return df, a
    c = a.obs["__c"].values; u, inv = np.unique(c, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(c), np.float32), (inv, np.arange(len(c)))), shape=(len(u), len(c)))
    return pd.DataFrame((G @ X).toarray(), index=pd.Index(u, name="cell_type"), columns=genes), a


def markers_union(a, cell_col, top=30):
    mk = R.derive_marker_genes(a, cell_col, top_n=top)  # robust: wilcoxon w/ mean-based fallback
    return sorted({g for v in mk.values() for g in v})


def axes_vectors(P, GT, gset, sample_level):
    """Return dict axis-> pd.Series (the per-row/per-gene pearson vectors)."""
    cg = [g for g in gset if g in P.columns and g in GT.columns]
    Pc, Gc = P[cg], GT[cg]
    out = {}
    if sample_level:
        Pl, Gl = norm(Pc), norm(Gc)
        cidx = Pl.index.intersection(Gl.index)
        # sample_profile (per sample,celltype) + profile (per celltype mean)
        spv = {}
        for i in cidx:
            a, b = Pl.loc[i].values, Gl.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9: spv[i] = np.corrcoef(a, b)[0, 1]
        sp_ser = pd.Series(spv)
        prof = sp_ser.groupby([k[1] for k in sp_ser.index]).mean()
        prof.index.name = "cell_type"; out["profile"] = prof.rename("profile")
        out["sample_profile"] = pd.Series(list(sp_ser.values), name="sample_profile")
        # sample
        Ps, Gs = norm(Pc.groupby(level="sample").sum()), norm(Gc.groupby(level="sample").sum())
        sc_idx = Ps.index.intersection(Gs.index); sv = {}
        for i in sc_idx:
            a, b = Ps.loc[i].values, Gs.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9: sv[i] = np.corrcoef(a, b)[0, 1]
        out["sample"] = pd.Series(sv, name="sample"); out["sample"].index.name = "sample"
        # pseudobulk (per celltype, summed over samples)
        Pct, Gct = norm(Pc.groupby(level="cell_type").sum()), norm(Gc.groupby(level="cell_type").sum())
        ci = Pct.index.intersection(Gct.index); pv = {}
        for i in ci:
            a, b = Pct.loc[i].values, Gct.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9: pv[i] = np.corrcoef(a, b)[0, 1]
        out["pseudobulk"] = pd.Series(pv, name="pseudobulk"); out["pseudobulk"].index.name = "cell_type"
        gP, gG = Pl, Gl
    else:
        Pct, Gct = norm(Pc), norm(Gc); ci = Pct.index.intersection(Gct.index)
        pv = {}
        for i in ci:
            a, b = Pct.loc[i].values, Gct.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9: pv[i] = np.corrcoef(a, b)[0, 1]
        out["profile"] = pd.Series(pv, name="profile"); out["profile"].index.name = "cell_type"
        out["pseudobulk"] = out["profile"].rename("pseudobulk")
        out["sample"] = pd.Series([], name="sample", dtype=float)
        out["sample_profile"] = pd.Series([], name="sample_profile", dtype=float)
        gP, gG = Pct.loc[ci], Gct.loc[ci]
    # gene axis (per-gene over the granular matrix; align rows to common (sample,celltype))
    rr = gP.index.intersection(gG.index); cc = gP.columns.intersection(gG.columns)
    A, B = gP.loc[rr, cc].values, gG.loc[rr, cc].values; gv = {}
    for j, g in enumerate(cc):
        a, b = A[:, j], B[:, j]
        if a.std() > 1e-9 and b.std() > 1e-9: gv[g] = np.corrcoef(a, b)[0, 1]
    out["gene"] = pd.Series(gv, name="gene"); out["gene"].index.name = ""
    return out


DATASETS = {
    "HCA_fold2": dict(
        cells=f"{WORKROOT}/HCA/generated_data.h5ad",
        cells_sample="pseudo_bulk_id", cells_cell="Cell_type",
        gt="/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_validation_cells.h5ad",
        gt_sample="pseudo_bulk_id", gt_cell="Cell_type", gt_raw=False, sample_level=True),
    "GSE141115": dict(
        cells=f"{WORKROOT}/GSE141115/generated_data.h5ad",
        cells_sample="Sample", cells_cell="Cell_type",
        gt="/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_test.h5ad",
        gt_sample="sample", gt_cell="cell type", gt_raw=True, sample_level=True),
    "GSE159585": dict(
        cells=f"{WORKROOT}/GSE159585/generated_data.h5ad",
        cells_sample="Sample", cells_cell="Cell_type",
        gt="/disk1/maijl/deconv/data/GSE159585/GSE159585_testset_level2celltype.h5ad",
        gt_sample=None, gt_cell="cell type", gt_raw=True, sample_level=False),
}


def summarize(s):
    s = s.dropna()
    return dict(n=int(s.size), mean=float(s.mean()) if s.size else np.nan,
                median=float(s.median()) if s.size else np.nan,
                q25=float(s.quantile(.25)) if s.size else np.nan, q75=float(s.quantile(.75)) if s.size else np.nan,
                min=float(s.min()) if s.size else np.nan, max=float(s.max()) if s.size else np.nan)


def main():
    rows = []
    box_data = {}  # dataset -> {method -> {geneset -> {axis: Series}}}
    for ds, c in DATASETS.items():
        print(f"\n=== {ds} ===")
        outd = f"{OUT_RESULTS}/{ds}"; os.makedirs(f"{outd}/cVAE", exist_ok=True)
        box_data[ds] = {"cVAE": {}}
        axes_all = AXES4 + ["sample_profile"]
        cached = all(os.path.exists(f"{outd}/cVAE/pearson_{ax}_{gn}.csv")
                     for gn in ("all_genes", "marker_genes") for ax in ("profile", "gene"))
        if cached and not os.environ.get("AG_FORCE"):
            print("  [cache] loading cVAE(Route2) vectors from existing CSVs")
            for gname in ("all_genes", "marker_genes"):
                box_data[ds]["cVAE"][gname] = {}
                for axis in axes_all:
                    f = f"{outd}/cVAE/pearson_{axis}_{gname}.csv"
                    if not os.path.exists(f): continue
                    ser = pd.read_csv(f).iloc[:, -1].dropna()
                    box_data[ds]["cVAE"][gname][axis] = ser
                    rows.append(dict(dataset=ds, method="DeconvSC", gene_set=gname, axis=axis, **summarize(ser)))  # storage dir stays <ds>/cVAE/; only the summary label is DeconvSC
        else:
            GT, gt_ad = gt_pseudobulk(c["gt"], c["gt_sample"], c["gt_cell"], c["gt_raw"])
            muni = markers_union(gt_ad, c["gt_cell"])
            P = sct_pseudobulk(c["cells"], c["cells_sample"], c["cells_cell"], is_log=True)
            if not c["sample_level"]:
                P = P.groupby(level="cell_type").sum()  # collapse 7 bulks -> per cell type
            genes_common = list(P.columns.intersection(GT.columns))
            GT = GT.reindex(columns=P.columns).fillna(0.0)
            gsets = {"all_genes": genes_common, "marker_genes": [g for g in muni if g in genes_common]}
            for gname, gset in gsets.items():
                vec = axes_vectors(P, GT, gset, c["sample_level"])
                box_data[ds]["cVAE"][gname] = vec
                for axis, ser in vec.items():
                    ser.to_csv(f"{outd}/cVAE/pearson_{axis}_{gname}.csv")
                    rows.append(dict(dataset=ds, method="DeconvSC", gene_set=gname, axis=axis, **summarize(ser)))  # storage dir stays <ds>/cVAE/; only the summary label is DeconvSC
                print(f"  cVAE[{gname}] prof={vec['profile'].median():.3f} "
                      f"samp={vec['sample'].median() if vec['sample'].size else float('nan'):.3f} "
                      f"gene={vec['gene'].median():.3f}")
        # ---- reuse other methods' vectors from the existing eval ----
        for mth in OTHERS:
            box_data[ds][mth] = {}
            md = f"{SRC_RESULTS}/{ds}/{mth}"
            if not os.path.isdir(md):
                continue
            os.makedirs(f"{outd}/{mth}", exist_ok=True)
            for gname in ("all_genes", "marker_genes"):
                box_data[ds][mth][gname] = {}
                for axis in AXES4 + ["sample_profile"]:
                    f = f"{md}/pearson_{axis}_{gname}.csv"
                    if not os.path.exists(f): continue
                    shutil.copy(f, f"{outd}/{mth}/pearson_{axis}_{gname}.csv")
                    df = pd.read_csv(f)
                    ser = df.iloc[:, -1].dropna()
                    box_data[ds][mth][gname][axis] = ser
                    rows.append(dict(dataset=ds, method=mth, gene_set=gname, axis=axis, **summarize(ser)))
        pd.DataFrame([r for r in rows if r["dataset"] == ds]).to_csv(f"{outd}/summary_all_methods.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{OUT_RESULTS}/summary_all_methods.csv", index=False)
    print(f"\n[summary] -> {OUT_RESULTS}/summary_all_methods.csv")
    make_figures(summary, box_data)


def make_figures(summary, box_data):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
    AXP = {"sample": "Sample", "profile": "Profile", "gene": "Gene", "pseudobulk": "Pseudo-bulk"}
    # NOTE (2026-06-10): all DeconvSC paper figures — the <ds>_methods_box_* boxplots, the
    # per-cell-type / per-dataset PCC bars, and the per-dataset gene-wise boxes — are now produced
    # by 04_plot_pcc_per_dataset.R from the landed pearson_*.csv, using ONE shared pastel palette so
    # every method's colour is identical across all figures. run_eval only writes the tables here.
    # (Removed: the Dark2-coloured methods_box plotting and the GRAND median heatmaps.)
    print(f"[figures] handled by 04_plot_pcc_per_dataset.R; tables -> {OUT_RESULTS}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="v3 Route2 evaluation + figures")
    ap.add_argument("--outdir", default=None,
                    help="base dir for outputs; writes <outdir>/03_results_v3_route2 and "
                         "<outdir>/03_result_figures/figures. Default = the cVAE eval tree.")
    a = ap.parse_args()
    if a.outdir:
        OUT_RESULTS = f"{a.outdir}/03_result_tables"
        OUT_FIG = f"{a.outdir}/03_result_figures/figures"
        os.makedirs(OUT_FIG, exist_ok=True)
        print(f"[outdir] results -> {OUT_RESULTS}\n[outdir] figures -> {OUT_FIG}")
    main()
