#!/usr/bin/env python
"""Real-vs-Generated marker-gene heatmap (panel e) for Fig2/Fig3/SuppFig3.

Merges deconv_20260605/script/export_marker_zscore.py (top-5 t-test markers/celltype ->
per-celltype mean -> per-gene z-score) with the two-panel renderer from
script/GSE159585-geneembed.py::plot_marker_expression_comparison (RdBu_r, vmin -1.5 / vmax 2.5).

Reads  $ROOT/<dataset>/generated_data.h5ad  and the (scheme-invariant) real reference;
writes $ROOT/<dataset>/{real,gen}_marker_expression_zscore.csv + marker_expression_pearson.csv
       (real-vs-gen marker-matrix Pearson r) + comparison_marker_heatmap.{png,svg,pdf}
Usage:  python plot_marker_heatmap.py [--dataset HCA|GSE141115|GSE159585] [--root $WORK_ROOT]
"""
import os, sys, argparse, warnings, numpy as np, pandas as pd, anndata as ad, scanpy as sc
from sklearn.preprocessing import scale
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import plot_style                       # Arial font (applied on import) + EMF export
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0
TOP_N = 5
# diverging colormap for the marker heatmap (low -> mid -> high)
HEATMAP_CMAP = LinearSegmentedColormap.from_list("marker_div", ["#6E94CA", "#F7F7F7", "#D96A5D"])

DATASETS = {
    "HCA": dict(real="/disk1/maijl/deconv/cVAE/Simulation_HCA/result/HCA/v3_pureprior/fold_2/gt_validation_cells.h5ad",
                real_celltype="Cell_type", real_is_raw=False),
    "GSE141115": dict(real="/disk1/maijl/deconv/data/GSE141115/GSE141115_sc_raw_counts_test.h5ad",
                      real_celltype="cell type", real_is_raw=True),
    "GSE159585": dict(real="/disk1/maijl/deconv/data/GSE159585/GSE159585_testset_level2celltype.h5ad",
                      real_celltype="cell type", real_is_raw=True),
}


def to_dense_mean(X):
    return np.asarray(X.mean(axis=0)).ravel()


def safe_zscore(df):
    if df.empty or df.shape[1] == 0 or df.shape[0] == 0:
        return pd.DataFrame(0, index=df.index, columns=df.columns)
    z = scale(df.values, axis=0)  # per-gene (per-column) z-score
    return pd.DataFrame(z, index=df.index, columns=df.columns).fillna(0)


def run(ds, cfg, root):
    outdir = os.path.join(root, ds)
    gen_path = os.path.join(outdir, "generated_data.h5ad")
    if not os.path.exists(gen_path):
        print(f"[{ds}] SKIP: {gen_path} not found"); return
    def _others(series):   # HLCA placeholder 'None'/NA -> 'Others' (consistent with run_eval/the other figures)
        s = series.astype(str).str.strip()
        return s.mask(s.str.lower().isin(["none", "na", "nan"]), "Others").values
    real = ad.read_h5ad(cfg["real"]); real.obs["labels"] = _others(real.obs[cfg["real_celltype"]])
    if cfg["real_is_raw"]:
        sc.pp.normalize_total(real, target_sum=1e4); sc.pp.log1p(real)
    real.var_names_make_unique()
    gen = ad.read_h5ad(gen_path); gen.var_names_make_unique()
    gen.obs["Cell_type"] = _others(gen.obs["Cell_type"])

    groups = sorted(str(g) for g in real.obs["labels"].unique())
    vc = real.obs["labels"].astype(str).value_counts()
    rank_groups = [g for g in groups if vc.get(g, 0) >= 2]
    sc.tl.rank_genes_groups(real, groupby="labels", method="t-test_overestim_var", groups=rank_groups)
    gene_list = []
    for g in groups:
        if g in rank_groups:
            try: gene_list.extend(sc.get.rank_genes_groups_df(real, group=g).head(TOP_N)["names"].tolist())
            except Exception: pass
    gene_list = [g for g in gene_list if g in set(real.var_names)]

    real_mean = pd.DataFrame(index=groups, columns=gene_list, dtype=float)
    for g in groups:
        real_mean.loc[g] = to_dense_mean(real[real.obs["labels"] == g, gene_list].X)
    valid = [g for g in gene_list if g in set(gen.var_names)]
    gen_mean = pd.DataFrame(index=groups, columns=valid, dtype=float)
    gen_labels = set(gen.obs["Cell_type"].astype(str).values)
    for g in groups:
        if g not in gen_labels: gen_mean.loc[g] = 0.0; continue
        gen_mean.loc[g] = to_dense_mean(gen[gen.obs["Cell_type"].astype(str) == g, valid].X)

    real_norm = safe_zscore(real_mean.astype(float)); gen_norm = safe_zscore(gen_mean.astype(float))
    real_norm.to_csv(os.path.join(outdir, "real_marker_expression_zscore.csv"))
    gen_norm.to_csv(os.path.join(outdir, "gen_marker_expression_zscore.csv"))

    # Drop cell types with NO generated cells: in the gen panel they render as flat/empty rows
    # (z-scored zeros), so remove them from BOTH panels to keep the two heatmaps row-aligned.
    drop = [g for g in groups if g not in gen_labels]                       # absent in generated data
    drop += list(gen_norm.index[(gen_norm == 0).all(axis=1)])               # any residual all-zero row
    drop = [g for g in dict.fromkeys(drop) if g in real_norm.index]
    if drop:
        print(f"[{ds}] dropping {len(drop)} cell types with 0 generated cells: {drop}")
    real_norm = real_norm.drop(drop); gen_norm = gen_norm.drop(drop)
    real_norm = real_norm[~real_norm.index.isna()]; gen_norm = gen_norm[~gen_norm.index.isna()]

    # --- Pearson correlation of the marker-expression matrices (real vs generated) ---
    # Recorded alongside the heatmap: per-cell-type r on the raw per-type mean expression
    # (pearson_r_expr) and on the plotted per-gene z-scores (pearson_r_zscore), + two summary
    # rows (mean over cell types, and the whole-matrix flattened r). One file per dataset.
    def _pcc(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return float(np.corrcoef(a, b)[0, 1]) if (a.std() > 1e-9 and b.std() > 1e-9) else np.nan
    shared_ct = list(real_norm.index)                       # cell types shown in the heatmap (row-aligned)
    shared_mk = [g for g in valid if g in real_mean.columns]  # markers present in BOTH matrices
    Re, Ge = real_mean.loc[shared_ct, shared_mk].astype(float), gen_mean.loc[shared_ct, shared_mk].astype(float)
    Rz, Gz = real_norm.loc[shared_ct, shared_mk], gen_norm.loc[shared_ct, shared_mk]
    crows = [{"cell_type": ct, "pearson_r_expr": _pcc(Re.loc[ct], Ge.loc[ct]),
              "pearson_r_zscore": _pcc(Rz.loc[ct], Gz.loc[ct]), "n_markers": len(shared_mk)}
             for ct in shared_ct]
    cdf = pd.DataFrame(crows)
    cdf = pd.concat([cdf, pd.DataFrame([
        {"cell_type": "__mean_over_celltypes__", "pearson_r_expr": cdf.pearson_r_expr.mean(),
         "pearson_r_zscore": cdf.pearson_r_zscore.mean(), "n_markers": len(shared_ct)},
        {"cell_type": "__overall_flattened__", "pearson_r_expr": _pcc(Re.values.ravel(), Ge.values.ravel()),
         "pearson_r_zscore": _pcc(Rz.values.ravel(), Gz.values.ravel()), "n_markers": int(Re.size)},
    ])], ignore_index=True)
    cdf.to_csv(os.path.join(outdir, "marker_expression_pearson.csv"), index=False)
    print(f"[{ds}] marker-expression Pearson (real vs gen): per-celltype mean expr="
          f"{cdf.pearson_r_expr.iloc[:-2].mean():.3f}, overall expr={cdf.pearson_r_expr.iloc[-1]:.3f}"
          f" -> {outdir}/marker_expression_pearson.csv")

    # Diverging z-score scale, asymmetric but CENTRED on 0 via TwoSlopeNorm: white falls exactly on 0
    # while the blue arm spans only [-1.5, 0] and the red arm [0, 2.5]. This matches the right-skewed
    # per-gene marker z-score range (median ~ -0.3, only ~1-2% of cells < -1, strong markers up to +2.5)
    # instead of a symmetric +/-2.5 scale whose unused -2.5..-1.5 blue arm washed the panel out.
    # seaborn's center= forces a SYMMETRIC range, so we render with imshow to honour the TwoSlopeNorm.
    # Fig2e / Fig3e / SuppFig3d all render from here, so all three stay locked together.
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=2.5)
    # Two EQUAL-width heatmaps + a thin DEDICATED colorbar column so both heatmaps stay identical in width.
    fig, axes = plt.subplots(1, 3, figsize=(13, 7),
                             gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.06})
    im_kw = dict(cmap=HEATMAP_CMAP, norm=norm, aspect="auto", interpolation="nearest")
    axes[0].imshow(real_norm.values, **im_kw)
    axes[0].set_xlabel("Real data marker", fontsize=14)        # panel label BELOW the heatmap (top title removed)
    axes[0].set_ylabel("Cell Types", fontsize=13)
    axes[0].set_yticks(range(len(real_norm.index))); axes[0].set_yticklabels(real_norm.index, fontsize=11)
    axes[0].set_xticks([])
    im = axes[1].imshow(gen_norm.values, **im_kw)
    axes[1].set_xlabel("Generated data marker", fontsize=14)   # panel label BELOW the heatmap (top title removed)
    axes[1].set_yticks([]); axes[1].set_xticks([])
    for _ax in (axes[0], axes[1]):                             # remove the black panel frame (outer border)
        for _s in _ax.spines.values():
            _s.set_visible(False)
    plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.10)   # top tightened: panel titles removed
    # Colourbar: shorten to ~60% of the panel height (vertically centred; was full height) and enlarge its
    # title + tick labels. Repositioned AFTER subplots_adjust so the adjust does not override the new box.
    _p = axes[2].get_position(); _nh = _p.height * 0.6
    axes[2].set_position([_p.x0, _p.y0 + (_p.height - _nh) / 2.0, _p.width, _nh])
    cb = fig.colorbar(im, cax=axes[2])
    cb.set_label("Expression z-score (per gene)", fontsize=14)   # enlarged colourbar title
    cb.ax.tick_params(labelsize=12)                             # enlarged colourbar tick labels
    base = os.path.join(outdir, "comparison_marker_heatmap")
    plot_style.save(fig, base, formats=("png", "pdf", "svg"), dpi=600, emf=False, bbox_inches=None)  # crisp png/pdf/svg
    # EMF: the heatmap is a raster that inkscape stores UNCOMPRESSED, so a 600-dpi embed balloons the .emf
    # to ~80 MB. Re-render the EMF's source SVG at 150 dpi (heatmap stays a clean raster, TEXT stays vector
    # Arial) before converting -> ~3-5 MB instead of 80 MB. (Same trick as plot_umap_panels.py.)
    _emf_svg = base + ".__emf__.svg"
    fig.savefig(_emf_svg, dpi=150, bbox_inches=None)
    plot_style.relabel_svg(_emf_svg)
    plot_style.svg_to_emf(_emf_svg)
    if os.path.exists(base + ".__emf__.emf"):
        os.replace(base + ".__emf__.emf", base + ".emf")
    if os.path.exists(_emf_svg):
        os.remove(_emf_svg)
    plt.close()
    print(f"[{ds}] heatmap saved ({real_norm.shape[0]} cell types x {real_norm.shape[1]} markers) -> {outdir}/comparison_marker_heatmap.*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=list(DATASETS), help="default: all three")
    ap.add_argument("--root", default=os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605"))
    a = ap.parse_args()
    for ds in ([a.dataset] if a.dataset else list(DATASETS)):
        run(ds, DATASETS[ds], a.root)
