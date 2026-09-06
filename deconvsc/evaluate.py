import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.preprocessing import scale
from scipy.stats import pearsonr, ttest_rel
from sklearn.feature_selection import mutual_info_regression
from scipy.cluster.hierarchy import linkage, fcluster

from .utils import set_seed
from . import plot_style   # Arial font (applied on import) + EMF export


# ============================================================================
#  DeconvSC generated-data evaluation  --  deconv_20260610 (canonical) method
#  ------------------------------------------------------------------------
#  Faithful re-implementation (inlined, no external imports) of the canonical
#  evaluation in deconv_20260610/04_evaluation/:
#    * run_eval_v3_route2.py   -> profile-wise + gene-wise PCC via
#                                 per-(sample, cell_type) pseudobulks
#    * plot_marker_heatmap.py  -> real-vs-generated marker-gene z-score heatmap
#    * plot_umap_panels.py     -> generated-data UMAP (cell type + sample mixing)
#  The real reference is the held-out single-cell test h5ad (NOT a per-type mean
#  tensor): both generated and real cells are collapsed to (sample, cell_type)
#  pseudobulks, CP10k-log1p normalised, then correlated axis-by-axis -- the
#  rigorous "unified Pearson" definition used throughout the manuscript.
# ============================================================================

# diverging colormap for the marker heatmap (low -> mid -> high)
_HEATMAP_CMAP = LinearSegmentedColormap.from_list("marker_div", ["#6E94CA", "#F7F7F7", "#D96A5D"])


# ----------------------------------------------------------------------------
#  pseudobulk + axis helpers (ported from run_eval_v3_route2.py)
# ----------------------------------------------------------------------------
def _rename_none(arr):
    """Placeholder label 'None'/'NA'/'nan' -> 'Others' (a readable, consistent name)."""
    s = pd.Series(arr, dtype=str).str.strip()
    return s.mask(s.str.lower().isin(["none", "na", "nan"]), "Others").values


def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _norm_pb(pb):
    """(sample,cell_type) x gene counts -> per-row CP10k -> log1p (the unified-Pearson axis space)."""
    s = pb.sum(axis=1).replace(0, np.nan)
    return np.log1p(pb.div(s, axis=0).fillna(0.0) * 1e4)


def derive_marker_genes(adata, cell_col, top_n=30):
    """Per-cell-type marker genes (wilcoxon top-n, with a mean-expression fallback). Mirrors
    deconv_20260610/lib/run_eval_HCA_fold2.derive_marker_genes."""
    a = adata.copy()
    xmax = a.X.max()
    if xmax > 50:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    a.obs["__g"] = a.obs[cell_col].astype(str).values
    try:
        sc.tl.rank_genes_groups(a, "__g", method="wilcoxon", n_genes=top_n)
        names = a.uns["rank_genes_groups"]["names"]
        return {ct: list(names[ct][:top_n]) for ct in names.dtype.names}
    except Exception:
        out = {}
        Xd = _to_dense(a.X)
        cells = a.obs["__g"].values
        gns = list(a.var_names)
        for ct in np.unique(cells):
            mu = np.asarray(Xd[cells == ct].mean(0)).ravel()
            out[ct] = [gns[i] for i in np.argsort(-mu)[:top_n]]
        return out


def _markers_union(adata, cell_col, top=30):
    mk = derive_marker_genes(adata, cell_col, top_n=top)
    return sorted({g for v in mk.values() for g in v})


def _gen_pseudobulk(adata_gen, sample_col, cell_col, is_log=True):
    """generated cells -> (sample,cell_type) x gene counts-like (expm1 of log, then summed)."""
    a = adata_gen
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.copy()
    if is_log:
        X.data = np.expm1(np.clip(X.data, 0, None))
    s = a.obs[sample_col].astype(str).values
    c = _rename_none(a.obs[cell_col].astype(str).values)
    grp = np.array([f"{si}\t{ci}" for si, ci in zip(s, c)])
    u, inv = np.unique(grp, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(grp), np.float32), (inv, np.arange(len(grp)))), shape=(len(u), len(grp)))
    df = pd.DataFrame((G @ X).toarray(), index=u, columns=a.var_names.astype(str))
    pp = df.index.str.split("\t", expand=True)
    df.index = pd.MultiIndex.from_arrays([pp.get_level_values(0), pp.get_level_values(1)],
                                         names=["sample", "cell_type"])
    return df


def _gt_pseudobulk(adata_real, sample_col, cell_col, is_raw):
    """real cells -> (sample,cell_type) x gene counts-like. is_raw=False -> de-log (expm1) first.
    Falls back to a per-cell_type index when no usable sample column is given (sample_level=False)."""
    a = adata_real
    c = _rename_none(a.obs[cell_col].astype(str).values)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, np.float32))
    X = X.copy()
    if not is_raw:
        X.data = np.expm1(np.clip(X.data, 0, None))
    genes = a.var_names.astype(str)
    if sample_col and sample_col in a.obs.columns:
        s = a.obs[sample_col].astype(str).values
        grp = np.array([f"{si}\t{ci}" for si, ci in zip(s, c)])
        u, inv = np.unique(grp, return_inverse=True)
        G = sp.csr_matrix((np.ones(len(grp), np.float32), (inv, np.arange(len(grp)))), shape=(len(u), len(grp)))
        df = pd.DataFrame((G @ X).toarray(), index=u, columns=genes)
        pp = df.index.str.split("\t", expand=True)
        df.index = pd.MultiIndex.from_arrays([pp.get_level_values(0), pp.get_level_values(1)],
                                             names=["sample", "cell_type"])
        return df
    u, inv = np.unique(c, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(c), np.float32), (inv, np.arange(len(c)))), shape=(len(u), len(c)))
    return pd.DataFrame((G @ X).toarray(), index=pd.Index(u, name="cell_type"), columns=genes)


def _axes_vectors(P, GT, gset, sample_level):
    """Return {axis -> pd.Series} of per-row/per-gene Pearson vectors, exactly as
    run_eval_v3_route2.axes_vectors: profile / sample_profile / sample / pseudobulk / gene."""
    cg = [g for g in gset if g in P.columns and g in GT.columns]
    Pc, Gc = P[cg], GT[cg]
    out = {}
    if sample_level:
        Pl, Gl = _norm_pb(Pc), _norm_pb(Gc)
        cidx = Pl.index.intersection(Gl.index)
        spv = {}
        for i in cidx:
            a, b = Pl.loc[i].values, Gl.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9:
                spv[i] = np.corrcoef(a, b)[0, 1]
        sp_ser = pd.Series(spv)
        # profile = per-cell-type mean of the per-(sample,cell_type) PCC
        prof = sp_ser.groupby([k[1] for k in sp_ser.index]).mean()
        prof.index.name = "cell_type"
        out["profile"] = prof.rename("profile")
        out["sample_profile"] = pd.Series(list(sp_ser.values), name="sample_profile")
        # sample = per-sample pseudobulk PCC (summed over cell types)
        Ps, Gs = _norm_pb(Pc.groupby(level="sample").sum()), _norm_pb(Gc.groupby(level="sample").sum())
        sc_idx = Ps.index.intersection(Gs.index)
        sv = {}
        for i in sc_idx:
            a, b = Ps.loc[i].values, Gs.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9:
                sv[i] = np.corrcoef(a, b)[0, 1]
        out["sample"] = pd.Series(sv, name="sample")
        out["sample"].index.name = "sample"
        # pseudobulk = per-cell-type pseudobulk PCC (summed over samples)
        Pct, Gct = _norm_pb(Pc.groupby(level="cell_type").sum()), _norm_pb(Gc.groupby(level="cell_type").sum())
        ci = Pct.index.intersection(Gct.index)
        pv = {}
        for i in ci:
            a, b = Pct.loc[i].values, Gct.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9:
                pv[i] = np.corrcoef(a, b)[0, 1]
        out["pseudobulk"] = pd.Series(pv, name="pseudobulk")
        out["pseudobulk"].index.name = "cell_type"
        gP, gG = Pl, Gl
    else:
        Pct, Gct = _norm_pb(Pc), _norm_pb(Gc)
        ci = Pct.index.intersection(Gct.index)
        pv = {}
        for i in ci:
            a, b = Pct.loc[i].values, Gct.loc[i].values
            if a.std() > 1e-9 and b.std() > 1e-9:
                pv[i] = np.corrcoef(a, b)[0, 1]
        out["profile"] = pd.Series(pv, name="profile")
        out["profile"].index.name = "cell_type"
        out["pseudobulk"] = out["profile"].rename("pseudobulk")
        out["sample"] = pd.Series([], name="sample", dtype=float)
        out["sample_profile"] = pd.Series([], name="sample_profile", dtype=float)
        gP, gG = Pct.loc[ci], Gct.loc[ci]
    # gene axis: per-gene Pearson over the granular (sample,cell_type) rows
    rr = gP.index.intersection(gG.index)
    cc = gP.columns.intersection(gG.columns)
    A, B = gP.loc[rr, cc].values, gG.loc[rr, cc].values
    gv = {}
    for j, g in enumerate(cc):
        a, b = A[:, j], B[:, j]
        if a.std() > 1e-9 and b.std() > 1e-9:
            gv[g] = np.corrcoef(a, b)[0, 1]
    out["gene"] = pd.Series(gv, name="gene")
    out["gene"].index.name = ""
    return out


def _auto_sample_align(gt_samples, gen_samples, explicit=None):
    """Return a {gt_sample -> gen_sample} remap so the (sample,cell_type) pseudobulk indices overlap.
    Uses ``explicit`` if given; otherwise, when the two sets are disjoint, aligns by trailing integer
    (e.g. GT 'LD01' -> generated 'LDK1' because both end in 1). No-op when already overlapping."""
    gt_set, gen_set = set(map(str, gt_samples)), set(map(str, gen_samples))
    if explicit:
        return {str(k): str(v) for k, v in dict(explicit).items() if str(k) in gt_set}
    if gt_set & gen_set:
        return {}

    def _tail_int(s):
        m = re.search(r"(\d+)\s*$", str(s))
        return int(m.group(1)) if m else None

    gen_by_num = {}
    for g in gen_set:
        k = _tail_int(g)
        if k is not None:
            gen_by_num.setdefault(k, g)
    mp = {}
    for s in gt_set:
        k = _tail_int(s)
        if k is not None and k in gen_by_num:
            mp[s] = gen_by_num[k]
    return mp


# ----------------------------------------------------------------------------
#  marker heatmap (ported from plot_marker_heatmap.py)
# ----------------------------------------------------------------------------
def _safe_zscore(df):
    if df.empty or df.shape[1] == 0 or df.shape[0] == 0:
        return pd.DataFrame(0, index=df.index, columns=df.columns)
    z = scale(df.values, axis=0)  # per-gene (per-column) z-score
    return pd.DataFrame(z, index=df.index, columns=df.columns).fillna(0)


def _plot_marker_heatmap(adata_real, adata_gen, real_cell_col, gen_cell_col, real_is_raw,
                         output_dir, top_n=5, emf=True):
    """Real-vs-generated marker heatmap: per-cell-type top-`top_n` t-test markers -> per-type mean
    -> per-gene z-score -> two equal panels (real | generated) on a TwoSlopeNorm diverging scale.
    Also writes real/gen z-score CSVs and a marker_expression_pearson.csv."""
    real = adata_real.copy()
    real.obs["labels"] = _rename_none(real.obs[real_cell_col].astype(str).values)
    if real_is_raw:
        sc.pp.normalize_total(real, target_sum=1e4)
        sc.pp.log1p(real)
    real.var_names_make_unique()
    gen = adata_gen.copy()
    gen.var_names_make_unique()
    gen.obs[gen_cell_col] = _rename_none(gen.obs[gen_cell_col].astype(str).values)

    groups = sorted(str(g) for g in pd.unique(real.obs["labels"]))
    vc = pd.Series(real.obs["labels"].astype(str)).value_counts()
    rank_groups = [g for g in groups if vc.get(g, 0) >= 2]
    sc.tl.rank_genes_groups(real, groupby="labels", method="t-test_overestim_var", groups=rank_groups)
    gene_list = []
    for g in groups:
        if g in rank_groups:
            try:
                gene_list.extend(sc.get.rank_genes_groups_df(real, group=g).head(top_n)["names"].tolist())
            except Exception:
                pass
    gene_list = [g for g in gene_list if g in set(real.var_names)]

    real_mean = pd.DataFrame(index=groups, columns=gene_list, dtype=float)
    for g in groups:
        real_mean.loc[g] = np.asarray(_to_dense(real[real.obs["labels"] == g, gene_list].X).mean(axis=0)).ravel()
    valid = [g for g in gene_list if g in set(gen.var_names)]
    gen_mean = pd.DataFrame(index=groups, columns=valid, dtype=float)
    gen_labels = set(gen.obs[gen_cell_col].astype(str).values)
    for g in groups:
        if g not in gen_labels:
            gen_mean.loc[g] = 0.0
            continue
        gen_mean.loc[g] = np.asarray(_to_dense(gen[gen.obs[gen_cell_col].astype(str) == g, valid].X).mean(axis=0)).ravel()

    real_norm = _safe_zscore(real_mean.astype(float))
    gen_norm = _safe_zscore(gen_mean.astype(float))
    real_norm.to_csv(os.path.join(output_dir, "real_marker_expression_zscore.csv"))
    gen_norm.to_csv(os.path.join(output_dir, "gen_marker_expression_zscore.csv"))

    # Drop cell types with NO generated cells (flat/empty rows) from BOTH panels to keep them aligned.
    drop = [g for g in groups if g not in gen_labels]
    drop += list(gen_norm.index[(gen_norm == 0).all(axis=1)])
    drop = [g for g in dict.fromkeys(drop) if g in real_norm.index]
    if drop:
        print(f"[heatmap] dropping {len(drop)} cell types with 0 generated cells: {drop}")
    real_norm = real_norm.drop(drop)
    gen_norm = gen_norm.drop(drop)
    real_norm = real_norm[~pd.isna(real_norm.index)]
    gen_norm = gen_norm[~pd.isna(gen_norm.index)]

    # --- real-vs-generated marker-matrix Pearson (per cell type + two summary rows) ---
    def _pcc(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return float(np.corrcoef(a, b)[0, 1]) if (a.std() > 1e-9 and b.std() > 1e-9) else np.nan

    shared_ct = list(real_norm.index)
    shared_mk = [g for g in valid if g in real_mean.columns]
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
    cdf.to_csv(os.path.join(output_dir, "marker_expression_pearson.csv"), index=False)
    print(f"[heatmap] marker-expression Pearson (real vs gen): per-celltype mean expr="
          f"{cdf.pearson_r_expr.iloc[:-2].mean():.3f}, overall expr={cdf.pearson_r_expr.iloc[-1]:.3f}")

    # Diverging z-score scale CENTRED on 0 via TwoSlopeNorm (white at 0, blue arm [-1.5,0],
    # red arm [0,2.5]); imshow honours the asymmetric norm (seaborn's center= would force symmetry).
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=2.5)
    fig, axes = plt.subplots(1, 3, figsize=(13, 7),
                             gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.06})
    im_kw = dict(cmap=_HEATMAP_CMAP, norm=norm, aspect="auto", interpolation="nearest")
    axes[0].imshow(real_norm.values, **im_kw)
    axes[0].set_xlabel("Real data marker", fontsize=14)
    axes[0].set_ylabel("Cell Types", fontsize=13)
    axes[0].set_yticks(range(len(real_norm.index)))
    axes[0].set_yticklabels(real_norm.index, fontsize=11)
    axes[0].set_xticks([])
    im = axes[1].imshow(gen_norm.values, **im_kw)
    axes[1].set_xlabel("Generated data marker", fontsize=14)
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    for _ax in (axes[0], axes[1]):
        for _s in _ax.spines.values():
            _s.set_visible(False)
    plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.10)
    _p = axes[2].get_position()
    _nh = _p.height * 0.6
    axes[2].set_position([_p.x0, _p.y0 + (_p.height - _nh) / 2.0, _p.width, _nh])
    cb = fig.colorbar(im, cax=axes[2])
    cb.set_label("Expression z-score (per gene)", fontsize=14)
    cb.ax.tick_params(labelsize=12)
    base = os.path.join(output_dir, "comparison_marker_heatmap")
    plot_style.save(fig, base, formats=("png", "pdf", "svg"), dpi=600, emf=False, bbox_inches=None)
    # EMF: re-render at 150 dpi (heatmap raster stays clean, text stays vector Arial) so the .emf
    # does not balloon (inkscape stores the raster uncompressed).
    if emf:
        _emf_svg = base + ".__emf__.svg"
        fig.savefig(_emf_svg, dpi=150, bbox_inches=None)
        plot_style.relabel_svg(_emf_svg)
        plot_style.svg_to_emf(_emf_svg)
        if os.path.exists(base + ".__emf__.emf"):
            os.replace(base + ".__emf__.emf", base + ".emf")
        if os.path.exists(_emf_svg):
            os.remove(_emf_svg)
    plt.close()
    print(f"[heatmap] saved ({real_norm.shape[0]} cell types x {real_norm.shape[1]} markers) "
          f"-> {output_dir}/comparison_marker_heatmap.*")


# ----------------------------------------------------------------------------
#  UMAP panels (ported from plot_umap_panels.py)
# ----------------------------------------------------------------------------
def _umap_style(figsize):
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.unicode_minus'] = False


def _auto_legend_ncol(n_cats, one_col_max=25, per_col=15, cap=3):
    if n_cats <= one_col_max:
        return 1
    return min(cap, max(2, -(-n_cats // per_col)))


def _plot_one_umap(adata, color, legend_title, plot_path, legend_col="auto", figsize=(7, 7),
                   legend_fontsize=12, legend_markerscale=1.0, emf=True):
    _umap_style(figsize)
    ax = sc.pl.umap(adata, color=color, title=None, legend_fontsize=14,
                    legend_fontweight="bold", size=30, show=False)
    ax.set_title("")   # drop scanpy's default "<color-column>" title; the legend carries identity
    for collection in ax.collections:   # rasterize ONLY the scatter (clarity + small vector files)
        collection.set_rasterized(True)
    n_cats = adata.obs[color].astype("category").cat.categories.size
    ncol = _auto_legend_ncol(n_cats) if legend_col == "auto" else int(legend_col)
    print(f"[umap] {color}: {n_cats} categories -> {ncol} legend column(s)")
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, fontsize=legend_fontsize,
               ncol=ncol, title=legend_title, title_fontsize=14, markerscale=legend_markerscale,
               columnspacing=0.8, handletextpad=0.25, labelspacing=0.3, borderaxespad=0.0)
    plot_style.save(ax.figure, plot_path, formats=("png", "pdf", "svg"), dpi=600, emf=False)
    if emf:
        _emf_svg = f"{plot_path}.__emf__.svg"
        ax.figure.savefig(_emf_svg, dpi=150, bbox_inches="tight")
        plot_style.relabel_svg(_emf_svg)
        plot_style.svg_to_emf(_emf_svg)
        if os.path.exists(f"{plot_path}.__emf__.emf"):
            os.replace(f"{plot_path}.__emf__.emf", f"{plot_path}.emf")
        if os.path.exists(_emf_svg):
            os.remove(_emf_svg)
    plt.close()


def _plot_umap_panels(adata_gen, sample_col, ct_col, output_dir, per_sample=90, sample_limit=28,
                      ct_legend_col="auto", emf=True):
    """Generated-data UMAP: panel a coloured by cell type, panel b by sample (inter-sample mixing).
    Subsamples CELLS per sample so the embedding is dense + balanced. Caches umap_embedding.h5ad."""
    a = adata_gen.copy()
    rng = np.random.default_rng(0)
    keep = []
    for s, idx in a.obs.groupby(sample_col, observed=True).indices.items():
        idx = np.asarray(idx)
        keep.append(idx if len(idx) <= per_sample else rng.choice(idx, per_sample, replace=False))
    keep = np.sort(np.concatenate(keep))
    a = a[keep].copy()
    print(f"[umap] {a.n_obs} cells from {a.obs[sample_col].nunique()} samples (<= {per_sample}/sample)")
    n_comps = int(min(30, max(2, a.n_obs - 1), a.shape[1] - 1))
    sc.pp.pca(a, n_comps=n_comps)
    sc.pp.neighbors(a, n_neighbors=min(15, max(2, a.n_obs - 1)))
    sc.tl.umap(a)
    a.obs[ct_col] = a.obs[ct_col].astype("category")
    a.obs[sample_col] = a.obs[sample_col].astype("category")
    a.write_h5ad(os.path.join(output_dir, "umap_embedding.h5ad"))
    # panel a: cell type
    _plot_one_umap(a, ct_col, "Cell type", os.path.join(output_dir, "umap_celltype"),
                   legend_col=ct_legend_col, figsize=(8, 8), legend_fontsize=9,
                   legend_markerscale=0.6, emf=emf)
    print(f"[umap] saved -> {output_dir}/umap_celltype.*")
    # panel b: sample mixing (subset to <= sample_limit samples for a readable legend)
    all_samples = list(a.obs[sample_col].astype("category").cat.categories)
    rng2 = np.random.default_rng(0)
    keep_s = set(rng2.choice(all_samples, min(sample_limit, len(all_samples)), replace=False))
    sub = a[a.obs[sample_col].isin(keep_s)].copy()
    sub.obs[sample_col] = sub.obs[sample_col].astype(str).astype("category")
    _plot_one_umap(sub, sample_col, "Sample", os.path.join(output_dir, "umap_sample_mixing"),
                   legend_col=1, figsize=(7, 7), emf=emf)
    print(f"[umap] saved -> {output_dir}/umap_sample_mixing.*")


# ----------------------------------------------------------------------------
#  main entry
# ----------------------------------------------------------------------------
def evaluate_generated_data(adata_gen,
                            adata_real,
                            output_dir,
                            gen_sample_col="Sample",
                            gen_cell_col="Cell_type",
                            real_sample_col="sample",
                            real_cell_col="cell type",
                            real_is_raw=True,
                            sample_level=True,
                            sample_map=None,
                            marker_top_n=30,
                            heatmap_top_n=5,
                            umap_per_sample=90,
                            umap_sample_limit=28,
                            make_umap=True,
                            make_heatmap=True,
                            emf=True,
                            seed=18):
    """Evaluate DeconvSC generated cells with the deconv_20260610 methodology.

    Real reference = the held-out single-cell test AnnData ``adata_real`` (raw counts when
    ``real_is_raw``). Produces, in ``output_dir``:
      * pearson_profile_{all_genes,marker_genes}.csv  -- profile-wise PCC (per cell type)
      * pearson_gene_{all_genes,marker_genes}.csv     -- gene-wise PCC (per gene)
      * pearson_{sample,pseudobulk,sample_profile}_*.csv -- the remaining unified-Pearson axes
      * summary_pcc.csv                                -- n/mean/median per (gene_set, axis)
      * comparison_marker_heatmap.{png,pdf,svg,emf} + real/gen z-score CSVs + marker pearson CSV
      * umap_celltype.* + umap_sample_mixing.* + umap_embedding.h5ad

    ``sample_map`` (e.g. {"LD01": "LDK1", ...}) aligns GT sample IDs to the generated ones; leave
    None to auto-align by trailing integer when the two sample sets are disjoint.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    print("\n=== DeconvSC evaluation (deconv_20260610 methodology) ===")
    print(f"  generated: {adata_gen.shape} | real GT: {adata_real.shape} | sample_level={sample_level}")

    # --- align GT sample IDs to the generated ones (e.g. 'LD01' -> 'LDK1') ---
    if sample_level and real_sample_col and real_sample_col in adata_real.obs.columns:
        gen_samples = pd.unique(adata_gen.obs[gen_sample_col].astype(str))
        gt_samples = pd.unique(adata_real.obs[real_sample_col].astype(str))
        mp = _auto_sample_align(gt_samples, gen_samples, explicit=sample_map)
        if mp:
            print(f"[align] remapping {len(mp)} GT sample IDs to match generated: {mp}")
            adata_real = adata_real.copy()
            adata_real.obs[real_sample_col] = adata_real.obs[real_sample_col].astype(str).replace(mp).values

    # --- 1/2. profile- and gene-wise PCC via (sample,cell_type) pseudobulks ---
    GT = _gt_pseudobulk(adata_real, real_sample_col, real_cell_col, real_is_raw)
    muni = _markers_union(adata_real, real_cell_col, top=marker_top_n)
    P = _gen_pseudobulk(adata_gen, gen_sample_col, gen_cell_col, is_log=True)
    if not sample_level:
        P = P.groupby(level="cell_type").sum()    # collapse samples -> per cell type
    genes_common = list(P.columns.intersection(GT.columns))
    GT = GT.reindex(columns=P.columns).fillna(0.0)
    gsets = {"all_genes": genes_common, "marker_genes": [g for g in muni if g in genes_common]}
    print(f"  common genes={len(genes_common)} | marker genes={len(gsets['marker_genes'])}")

    summary_rows = []
    for gname, gset in gsets.items():
        vec = _axes_vectors(P, GT, gset, sample_level)
        for axis, ser in vec.items():
            ser.to_csv(os.path.join(output_dir, f"pearson_{axis}_{gname}.csv"))
            s = ser.dropna()
            summary_rows.append(dict(gene_set=gname, axis=axis, n=int(s.size),
                                     mean=float(s.mean()) if s.size else np.nan,
                                     median=float(s.median()) if s.size else np.nan))
        print(f"  [{gname}] profile median={vec['profile'].dropna().median():.3f}  "
              f"gene median={vec['gene'].dropna().median():.3f}")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "summary_pcc.csv"), index=False)

    # --- 3. comparison marker heatmap ---
    if make_heatmap:
        try:
            _plot_marker_heatmap(adata_real, adata_gen, real_cell_col, gen_cell_col, real_is_raw,
                                 output_dir, top_n=heatmap_top_n, emf=emf)
        except Exception as e:
            print(f"[heatmap] SKIPPED due to error: {e}")

    # --- 4. generated-data UMAP ---
    if make_umap:
        try:
            _plot_umap_panels(adata_gen, gen_sample_col, gen_cell_col, output_dir,
                              per_sample=umap_per_sample, sample_limit=umap_sample_limit, emf=emf)
        except Exception as e:
            print(f"[umap] SKIPPED due to error: {e}")

    print(f"=== Evaluation complete -> {output_dir} ===")
    return summary


def advanced_attention_evaluation(adata_real, adata_ours, adata_other, other_label, output_dir):
    """
    Evaluation focused on attention mechanism advantages: modularity, hub gene preservation, non-linear mutual information.
    """
    print("\n=== Phase 3: Starting Advanced Attention Evaluation ===")
    
    # 0. data preparation
    common_genes = list(set(adata_real.var_names) & set(adata_ours.var_names) & set(adata_other.var_names))
    temp = adata_real[:, common_genes].copy()
    
    if hasattr(temp.X, 'toarray'):
        temp.X.data = np.nan_to_num(temp.X.data, posinf=0, neginf=0, nan=0)
    else:
        temp.X = np.nan_to_num(temp.X, posinf=0, neginf=0, nan=0)
        
    sc.pp.highly_variable_genes(temp, n_top_genes=1000)
    hvg = temp.var[temp.var['highly_variable']].index.tolist()
    print(f"Analyzing {len(hvg)} HVGs for network structure...")
    
    def get_clean_matrix_scientific(adata, genes, name):
        X = adata[:, genes].X
        if hasattr(X, 'toarray'): X = X.toarray()
        vars_ = np.var(X, axis=0)
        valid_indices = np.where(vars_ > 1e-9)[0]
        n_dropped = X.shape[1] - len(valid_indices)
        if n_dropped > 0:
            print(f"  [Info] {name}: Dropped {n_dropped} genes due to Mode Collapse (Variance=0).")
        return X[:, valid_indices], valid_indices

    _, valid_real = get_clean_matrix_scientific(adata_real, hvg, "Real")
    _, valid_ours = get_clean_matrix_scientific(adata_ours, hvg, "Ours")
    _, valid_other = get_clean_matrix_scientific(adata_other, hvg, other_label)
    
    final_indices_set = set(valid_real) & set(valid_ours) & set(valid_other)
    final_indices = sorted(list(final_indices_set)) 
    final_genes = [hvg[i] for i in final_indices]
    n_final = len(final_genes)
    print(f"Final evaluation on {len(final_genes)} genes (intersection of non-collapsed genes).")
    
    X_real = adata_real[:, final_genes].X.toarray() if hasattr(adata_real.X, 'toarray') else adata_real[:, final_genes].X
    X_ours = adata_ours[:, final_genes].X.toarray() if hasattr(adata_ours.X, 'toarray') else adata_ours[:, final_genes].X
    X_other = adata_other[:, final_genes].X.toarray() if hasattr(adata_other.X, 'toarray') else adata_other[:, final_genes].X

    def safe_corrcoef(X):
        corr = np.corrcoef(X, rowvar=False)
        if np.isnan(corr).any():
            corr = np.nan_to_num(corr, nan=0.0)
        return corr

    print("Computing correlation matrices...")
    corr_real = safe_corrcoef(X_real)
    corr_ours = safe_corrcoef(X_ours)
    corr_other = safe_corrcoef(X_other)
    
    # --- Experiment A: Hub Gene Neighbor Preservation ---
    print("\n[Exp A] Analyzing Hub Gene Connectivity...")
    degree_real = np.sum(np.abs(corr_real) - np.eye(n_final), axis=1)
    top_hub_indices = np.argsort(degree_real)[-50:]
    
    jaccard_ours, jaccard_other = [], []
    k_neighbors = 20 
    
    for idx in top_hub_indices:
        real_neighbors = set(np.argsort(np.abs(corr_real[idx]))[-k_neighbors:])
        ours_neighbors = set(np.argsort(np.abs(corr_ours[idx]))[-k_neighbors:])
        other_neighbors = set(np.argsort(np.abs(corr_other[idx]))[-k_neighbors:])
        
        jaccard_ours.append(len(real_neighbors & ours_neighbors) / len(real_neighbors | ours_neighbors))
        jaccard_other.append(len(real_neighbors & other_neighbors) / len(real_neighbors | other_neighbors))
        
    data_box = pd.DataFrame({
        'Model': ['Ours (Attention)']*50 + [other_label]*50,
        'Jaccard Similarity': jaccard_ours + jaccard_other
    })
    data_box.to_csv(f"{output_dir}/Exp_A_Hub_Preservation_Jaccard.csv", index=False)
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=data_box, x='Model', y='Jaccard Similarity', palette=['#d62728', '#1f77b4'])
    plt.title(f"Hub Gene Neighbor Preservation\n(Top 50 Hubs, K={k_neighbors})")
    plt.ylabel("Jaccard Index (Higher is Better)")
    _, p_val = ttest_rel(jaccard_ours, jaccard_other)
    plt.xlabel(f"Paired T-test p-value: {p_val:.2e}")
    plt.savefig(f"{output_dir}/Exp_A_Hub_Preservation.png")
    plt.close()
    
    print(f"  -> Mean Jaccard: Ours={np.mean(jaccard_ours):.4f}, {other_label}={np.mean(jaccard_other):.4f}")
    
    # --- Experiment B: Mutual Information ---
    print("\n[Exp B] Analyzing Non-linear Mutual Information...")
    np.random.seed(42)
    n_pairs = 100
    idx_pairs = np.random.choice(n_final, (n_pairs, 2), replace=False)
    
    mi_real, mi_ours, mi_other = [], [], []
    for i in range(n_pairs):
        idx1, idx2 = idx_pairs[i]
        mi_real.append(mutual_info_regression(X_real[:, [idx1]], X_real[:, idx2], discrete_features=False)[0])
        mi_ours.append(mutual_info_regression(X_ours[:, [idx1]], X_ours[:, idx2], discrete_features=False)[0])
        mi_other.append(mutual_info_regression(X_other[:, [idx1]], X_other[:, idx2], discrete_features=False)[0])
        
    corr_mi_ours = pearsonr(mi_real, mi_ours)[0]
    corr_mi_other = pearsonr(mi_real, mi_other)[0]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(mi_real, mi_ours, alpha=0.6, color='#d62728')
    m, b = np.polyfit(mi_real, mi_ours, 1)
    ax[0].plot(mi_real, m*np.array(mi_real)+b, 'k--')
    ax[0].set_title(f"Ours: MI Correlation\nR = {corr_mi_ours:.4f}")
    ax[0].set_xlabel("Real Mutual Information")
    ax[0].set_ylabel("Generated MI")
    
    ax[1].scatter(mi_real, mi_other, alpha=0.6, color='#1f77b4')
    m, b = np.polyfit(mi_real, mi_other, 1)
    ax[1].plot(mi_real, m*np.array(mi_real)+b, 'k--')
    ax[1].set_title(f"{other_label}: MI Correlation\nR = {corr_mi_other:.4f}")
    ax[1].set_xlabel("Real Mutual Information")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Exp_B_Mutual_Info.png")
    plt.close()
    print(f"  -> MI Correlation: Ours={corr_mi_ours:.4f}, {other_label}={corr_mi_other:.4f}")

    # --- Experiment C: Functional Module Cohesion ---
    print("\n[Exp C] Analyzing Functional Module Cohesion...")
    Z = linkage(corr_real, method='ward')
    labels = fcluster(Z, t=10, criterion='maxclust') 
    
    counts = np.bincount(labels)
    target_label = np.argmax(counts[1:]) + 1
    module_indices = np.where(labels == target_label)[0]
    print(f"  -> Identified a vivid module with {len(module_indices)} genes.")
    
    mod_corr_real = corr_real[np.ix_(module_indices, module_indices)]
    mod_corr_ours = corr_ours[np.ix_(module_indices, module_indices)]
    mod_corr_other = corr_other[np.ix_(module_indices, module_indices)]
    
    l2_mod_ours = np.linalg.norm(mod_corr_real - mod_corr_ours)
    l2_mod_other = np.linalg.norm(mod_corr_real - mod_corr_other)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.heatmap(mod_corr_real, ax=axes[0], cmap='viridis', cbar=False)
    axes[0].set_title("Real Module Structure")
    axes[0].axis('off')
    
    sns.heatmap(mod_corr_ours, ax=axes[1], cmap='viridis', cbar=False)
    axes[1].set_title(f"Ours (L2 Dist={l2_mod_ours:.2f})")
    axes[1].axis('off')
    
    sns.heatmap(mod_corr_other, ax=axes[2], cmap='viridis', cbar=True)
    axes[2].set_title(f"{other_label} (L2 Dist={l2_mod_other:.2f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Exp_C_Module_Structure.png")
    plt.close()
    print(f"  -> Module L2 Distance: Ours={l2_mod_ours:.2f}, {other_label}={l2_mod_other:.2f}")
    
    with open(os.path.join(output_dir, 'attention_evaluation.txt'), 'w') as f:
        f.write(f"Mean Jaccard: Ours={np.mean(jaccard_ours):.4f}, {other_label}={np.mean(jaccard_other):.4f}\n")
        f.write(f"MI Correlation: Ours={corr_mi_ours:.4f}, {other_label}={corr_mi_other:.4f}\n")
        f.write(f"Identified module size: {len(module_indices)} genes.\n")
        f.write(f"Module L2 Distance: Ours={l2_mod_ours:.2f}, {other_label}={l2_mod_other:.2f}\n")
        
    print("=== Advanced Evaluation Finished ===")
    
def compare_gene_coexpression_networks(
    adata_real, 
    adata_gen_ours, 
    adata_other, 
    output_dir, 
    n_top_genes=500, 
    other_label=None,
    specific_genes=None
    ):
    """
    对比 Real, Ours, Ablation 三种数据的基因共表达网络。
    
    参数:
        adata_real: 真实单细胞数据 (AnnData)
        adata_gen_ours: Attention模型生成的数据
        adata_gen_ablation: MLP模型(无Attention)生成的数据
        output_dir: 图片保存路径
        n_top_genes: 如果没有指定specific_genes，选取多少个HVG进行计算
        specific_genes: (可选) 基因列表，如果提供则只计算这些基因的网络
    """
    print("--- Starting Gene Co-expression Network Comparison ---")
    
    # 1. 确定要分析的基因集
    # 确保基因在三个数据集中都存在
    common_vars = list(set(adata_real.var_names) & set(adata_gen_ours.var_names) & set(adata_other.var_names))
    
    if specific_genes is not None:
        target_genes = [g for g in specific_genes if g in common_vars]
        print(f"Using {len(target_genes)} specific genes provided by user.")
    else:
        # 如果没给特定基因，就用 Real 数据的 HVG
        print(f"Selecting top {n_top_genes} HVGs from Real data...")
        temp_adata = adata_real[:, common_vars].copy()
        sc.pp.highly_variable_genes(temp_adata, n_top_genes=n_top_genes)
        target_genes = temp_adata.var[temp_adata.var['highly_variable']].index.tolist()
        print(f"Selected {len(target_genes)} HVGs.")

    # 2. 提取表达矩阵 (Cells x Genes)
    # 注意：计算共表达通常需要 Log1p 后的数据
    def get_matrix(adata, genes):
        sub = adata[:, genes]
        if isinstance(sub.X, np.ndarray):
            return sub.X
        else:
            return sub.X.toarray()

    X_real = get_matrix(adata_real, target_genes)
    X_ours = get_matrix(adata_gen_ours, target_genes)
    X_other = get_matrix(adata_other, target_genes)

    # 3. 计算基因-基因相关性矩阵 (Genes x Genes)
    # rowvar=False 表示每一列是一个变量(基因)
    # 返回矩阵形状: [n_genes, n_genes]
    print("Computing correlation matrices...")
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_ours = np.corrcoef(X_ours, rowvar=False)
    corr_other = np.corrcoef(X_other, rowvar=False)
    
    # 处理可能的 NaN (如果某个基因表达全为0)
    corr_real = np.nan_to_num(corr_real)
    corr_ours = np.nan_to_num(corr_ours)
    corr_other = np.nan_to_num(corr_other)

    # 4. 定量评估：矩阵相似度
    # 方法 A: 矩阵展平后的 Pearson 相关性 (Mantel Test 近似)
    # 衡量“真实数据里 A和B 正相关，生成数据里 A和B 是否也正相关？”
    score_ours_pcc = pearsonr(corr_real.flatten(), corr_ours.flatten())[0]
    score_other_pcc = pearsonr(corr_real.flatten(), corr_other.flatten())[0]
    
    # 方法 B: 矩阵距离 (Frobenius Norm)
    # 衡量绝对误差，越小越好
    dist_ours = np.linalg.norm(corr_real - corr_ours)
    dist_other = np.linalg.norm(corr_real - corr_other)

    print(f"\n[Quantitative Results]")
    print(f"Matrix Similarity (PCC) [Higher is Better]:")
    print(f"  Ours (Attention): {score_ours_pcc:.4f}")
    print(f"  {other_label}  : {score_other_pcc:.4f}")
    print(f"Matrix Distance (L2 Norm) [Lower is Better]:")
    print(f"  Ours (Attention): {dist_ours:.4f}")
    print(f"  {other_label}  : {dist_other:.4f}")

    # 5. 可视化绘制
    # 为了图好看，我们对基因聚类一下，让红色的块聚在一起
    # 使用 seaborn 的 clustermap 获取聚类后的索引
    print("Plotting heatmaps...")
    g = sns.clustermap(pd.DataFrame(corr_real), cmap='vlag', center=0)
    reordered_idx = g.dendrogram_row.reordered_ind
    plt.close() # 不显示这个中间图

    # 按 Real 数据的聚类顺序重新排列三个矩阵，以便直观对比
    def reorder_mat(mat, idx):
        return mat[idx, :][:, idx]

    corr_real_sorted = reorder_mat(corr_real, reordered_idx)
    corr_ours_sorted = reorder_mat(corr_ours, reordered_idx)
    corr_other_sorted = reorder_mat(corr_other, reordered_idx)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 通用绘图参数
    heatmap_kwargs = {'cmap': 'RdBu_r', 'center': 0, 'vmin': -0.8, 'vmax': 0.8, 'cbar': True, 'xticklabels': False, 'yticklabels': False}

    sns.heatmap(corr_real_sorted, ax=axes[0], **heatmap_kwargs)
    axes[0].set_title("Ground Truth (Real)\nGene Co-expression")

    sns.heatmap(corr_ours_sorted, ax=axes[1], **heatmap_kwargs)
    axes[1].set_title(f"Ours (Attention)\nPCC={score_ours_pcc:.3f}, Dist={dist_ours:.2f}")

    sns.heatmap(corr_other_sorted, ax=axes[2], **heatmap_kwargs)
    axes[2].set_title(f"{other_label}\nPCC={score_other_pcc:.3f}, Dist={dist_other:.2f}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Gene_Network_Comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

    return {
        'ours_pcc': score_ours_pcc, 
        f'{other_label}_pcc': score_other_pcc,
        'ours_dist': dist_ours,
        f'{other_label}_dist': dist_other
    }
   