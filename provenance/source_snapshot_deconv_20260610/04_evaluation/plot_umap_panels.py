#!/usr/bin/env python
"""UMAP panels (a,b) for the manuscript Fig2/Fig3/SuppFig3 — generalized from
deconv_20260605/script/plot_hca_umap.py to ANY core dataset and ANY scheme root.

Reads $ROOT/<dataset>/generated_data.h5ad and writes, next to it:
  umap_celltype.{pdf,svg,png}        (panel a: coloured by cell type)
  umap_sample_mixing.{pdf,svg,png}   (panel b: coloured by sample; shows inter-sample mixing)
  umap_embedding.h5ad                (cached embedding)

Subsamples CELLS per sample (not samples) so the embedding is dense + balanced.
Usage:  python plot_umap_panels.py --dataset {HCA,GSE141115,GSE159585} [--root $WORK_ROOT]
"""
import os, argparse
import scanpy as sc, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib")); import plot_style  # noqa: F401  (Arial font applied on import + plot_style.save/emf_dir for EMF)

# per-dataset obs columns (sample id / cell-type) — verified against run_eval_v3_route2.DATASETS
DS = {
    "HCA":       dict(sample="pseudo_bulk_id", ct="Cell_type", stitle="Pseudo-bulk sample"),
    "GSE141115": dict(sample="Sample",         ct="Cell_type", stitle="Sample"),
    "GSE159585": dict(sample="Sample",         ct="Cell_type", stitle="Sample"),
}


def build_embedding(src, emb, sample_col, ct_col, per_sample):
    a = sc.read_h5ad(src)
    rng = np.random.default_rng(0)
    keep = []
    for s, idx in a.obs.groupby(sample_col, observed=True).indices.items():
        idx = np.asarray(idx)
        keep.append(idx if len(idx) <= per_sample else rng.choice(idx, per_sample, replace=False))
    keep = np.sort(np.concatenate(keep))
    a = a[keep].copy()
    print(f"[subsample] {a.n_obs} cells from {a.obs[sample_col].nunique()} samples (<= {per_sample}/sample)")
    sc.pp.pca(a, n_comps=30); sc.pp.neighbors(a, n_neighbors=15); sc.tl.umap(a)
    a.obs[ct_col] = a.obs[ct_col].astype("category")
    a.obs[sample_col] = a.obs[sample_col].astype("category")
    a.write_h5ad(emb)
    return a


def _style(figsize):
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = 16; plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.labelweight'] = 'bold'   # UMAP1/UMAP2 axis labels, enlarged
    plt.rcParams['axes.unicode_minus'] = False


def auto_legend_ncol(n_cats, one_col_max=25, per_col=15, cap=3):
    """Adaptive legend columns: keep ONE column while the list is short enough to sit beside the
    ~8-inch UMAP without an over-long legend (n_cats <= one_col_max, e.g. GSE141115's 13, HCA's ~23);
    spill into extra columns for long lists (GSE159585's ~34 -> 3), capped at `cap`."""
    if n_cats <= one_col_max:
        return 1
    return min(cap, max(2, -(-n_cats // per_col)))   # ceil(n/per_col), >=2, <=cap


def plot_umap(adata_path, color, legend_title, plot_path, legend_col="auto", figsize=(7, 7),
              legend_fontsize=12, legend_markerscale=1.0):
    adata = sc.read_h5ad(adata_path)
    _style(figsize)
    ax = sc.pl.umap(adata, color=color, title=None, legend_fontsize=14,
                    legend_fontweight="bold", size=30, show=False)
    ax.set_title("")   # drop scanpy's default "<color-column>" title (the redundant "Cell_type"
                       # underscore / sample-id); identity is already carried by the legend title. Keeps
                       # the Fig2/Fig3/SuppFig3 UMAP panels on one clean, consistent template.
    # Rasterize ONLY the scatter points (clarity + small vector files); text/axes/legend stay vector.
    for collection in ax.collections:
        collection.set_rasterized(True)
    n_cats = adata.obs[color].astype("category").cat.categories.size
    ncol = auto_legend_ncol(n_cats) if legend_col == "auto" else int(legend_col)
    print(f"[legend] {color}: {n_cats} categories -> {ncol} column(s)")
    # Legend in the right margin; columns auto-scale to the list length (1 col unless it would be
    # too tall, e.g. 50 cell types -> 3). NOTE: do NOT call plt.tight_layout() — it rescales the
    # main axes to fit the external legend and visibly compresses the UMAP. The main axes keeps its
    # figsize; bbox_inches='tight' at save time simply widens the canvas to include the legend.
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, fontsize=legend_fontsize,
               ncol=ncol, title=legend_title, title_fontsize=14, markerscale=legend_markerscale,
               columnspacing=0.8, handletextpad=0.25, labelspacing=0.3, borderaxespad=0.0)
    # dpi=600 so the rasterized scatter is crisp in the vector (pdf/svg) outputs too
    plot_style.save(ax.figure, plot_path, formats=("png", "pdf", "svg"), dpi=600)  # +emf, Arial
    # The rasterized scatter is stored UNCOMPRESSED inside EMF, so a 600dpi UMAP balloons the .emf to
    # ~40MB. Regenerate ONLY the .emf from a 150dpi render (scatter ~4MB; text/axes stay vector Arial);
    # the png/pdf/svg keep the full 600dpi.
    _emf_svg = f"{plot_path}.__emf__.svg"
    ax.figure.savefig(_emf_svg, dpi=150, bbox_inches="tight")
    plot_style.relabel_svg(_emf_svg)
    plot_style.svg_to_emf(_emf_svg)
    if os.path.exists(f"{plot_path}.__emf__.emf"):
        os.replace(f"{plot_path}.__emf__.emf", f"{plot_path}.emf")
    if os.path.exists(_emf_svg):
        os.remove(_emf_svg)
    plt.close()


def build_sample_subset(emb_path, out_path, sample_col, limit=28, seed=0):
    """Subset to `limit` samples (<= default_28 palette) so each gets a unique colour + a real
    legend; keeps all their cells on the SAME UMAP coordinates."""
    a = sc.read_h5ad(emb_path)
    all_samples = list(a.obs[sample_col].astype("category").cat.categories)
    rng = np.random.default_rng(seed)
    keep_s = set(rng.choice(all_samples, min(limit, len(all_samples)), replace=False))
    a = a[a.obs[sample_col].isin(keep_s)].copy()
    # Shorten over-long sample IDs in the legend: strip the prefix shared by ALL kept samples so only
    # the distinguishing tail shows (HCA "homosapiens_None_..._483747THD0007__rep0000" -> "THD0007 rep0000").
    # Guarded so short IDs that merely share a long prefix are NOT over-stripped (an earlier unconditional
    # strip turned GSE141115 "GC1003224"->"24"): only strip when the common prefix is long AND every
    # remaining tail stays informative (>=4 chars).
    labels = a.obs[sample_col].astype(str)
    uniq = sorted(labels.unique())
    pref = os.path.commonprefix(uniq)
    if len(uniq) > 1 and len(pref) >= 12 and min(len(u) - len(pref) for u in uniq) >= 4:
        short = {u: u[len(pref):].lstrip("_").replace("__", " ") for u in uniq}
        labels = labels.map(short)
        print(f"[legend] stripped {len(pref)}-char common sample-id prefix -> e.g. {short[uniq[0]]!r}")
    a.obs[sample_col] = labels.astype("category")
    a.write_h5ad(out_path)
    print(f"[subset] {a.n_obs} cells from {a.obs[sample_col].nunique()} samples (<= {limit})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DS))
    ap.add_argument("--root", default=os.environ.get("WORK_ROOT", "/disk1/maijl/deconv/deconv_20260605"),
                    help="scheme root holding <dataset>/generated_data.h5ad (default $WORK_ROOT)")
    ap.add_argument("--per_sample", type=int, default=int(os.environ.get("PER_SAMPLE", "90")))
    ap.add_argument("--sample_limit", type=int, default=int(os.environ.get("SAMPLE_LIMIT", "28")))
    ap.add_argument("--ct_legend_col", default="auto",
                    help="cell-type legend columns: 'auto' (1 col unless many types -> up to 3) or an int")
    ap.add_argument("--replot", action="store_true",
                    help="reuse the cached umap_embedding.h5ad (skip the UMAP recompute) — just redo plots")
    a = ap.parse_args()
    cfg = DS[a.dataset]; d = f"{a.root}/{a.dataset}"
    src = f"{d}/generated_data.h5ad"; emb = f"{d}/umap_embedding.h5ad"
    sample_col, ct_col = cfg["sample"], cfg["ct"]
    if a.replot and os.path.exists(emb):
        print(f"[replot] reusing cached embedding {emb} (no UMAP recompute)")
    else:
        build_embedding(src, emb, sample_col, ct_col, a.per_sample)
    # panel a (cell type): 3-column legend + bigger square canvas so the 50-type legend no longer
    # compresses the UMAP; smaller legend font keeps the long list readable beside the main plot.
    plot_umap(emb, ct_col, "Cell type", f"{d}/umap_celltype",
              legend_col=a.ct_legend_col, figsize=(8, 8), legend_fontsize=9, legend_markerscale=0.6)
    print(f"[saved] {d}/umap_celltype.*")
    sub = build_sample_subset(emb, f"{d}/umap_embedding_sample_subset.h5ad", sample_col, limit=a.sample_limit)
    plot_umap(sub, sample_col, cfg["stitle"], f"{d}/umap_sample_mixing", legend_col=1, figsize=(7, 7))
    print(f"[saved] {d}/umap_sample_mixing.*")
