#!/usr/bin/env python3
"""Build a corrected, reproducible Supplementary Figure 5 for GSE141115.

The analysis deliberately evaluates the representation produced by DeconvSC:
continuous log-normalized expression rather than raw UMI counts. Real raw
counts and generated log1p decoder output are both transformed to log1p CP10K
on the same 16,801-gene panel. Cells are matched within sample x cell type for
LDK1-LDK3 across ten deterministic subsampling seeds.

Panel a shows the per-gene mean-variance relationship separately for real and
generated data on a shared, positive-moment gene panel.
Panel b compares Fisher-averaged gene-correlation matrices on a reference-
defined 500-HVG panel, while explicitly reporting the number of HVGs that are
nonconstant in both sources across every matched draw.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter, NullLocator
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


RELEASE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RELEASE_ROOT / "code/03_model_evaluation/support"))
import plot_style  # noqa: E402


DEFAULT_GENERATED = RELEASE_ROOT / "model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad"
DEFAULT_REAL = RELEASE_ROOT / "processed_data/real_data/mouse_kidney/test_data.h5ad"
DEFAULT_OUTDIR = RELEASE_ROOT / "work/supplementary_figure5"
DONORS = ["LDK1", "LDK2", "LDK3"]
SEEDS = [18, 29, 41, 53, 67, 79, 97, 113, 131, 149]
N_HVG = 500
TARGET_SUM = 10_000.0
EPS = 1e-12

REAL_COLOR = "#7187A7"
GEN_COLOR = "#D8B2AE"
CORR_CMAP = LinearSegmentedColormap.from_list(
    "deconv_diverging", ["#2166AC", "#F7F7F7", "#B2182B"]
)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def sparse_or_dense_moments(x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return column mean, population variance, and nonzero fraction."""
    if sparse.issparse(x):
        x = x.tocsr()
        mean = np.asarray(x.mean(axis=0)).ravel().astype(np.float64)
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel().astype(np.float64)
        variance = np.maximum(mean_sq - mean**2, 0.0)
        detection = np.asarray(x.getnnz(axis=0), dtype=np.float64) / x.shape[0]
    else:
        arr = np.asarray(x, dtype=np.float64)
        mean = arr.mean(axis=0)
        variance = arr.var(axis=0)
        detection = np.count_nonzero(arr, axis=0) / arr.shape[0]
    return mean, variance, detection


def gene_correlation(x: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    if not np.isfinite(corr).all():
        raise RuntimeError("Non-finite values encountered in a gene-correlation matrix")
    return corr


def fisher_average(matrices: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(matrices, axis=0)
    z = np.arctanh(np.clip(stack, -0.999999, 0.999999))
    averaged = np.tanh(z.mean(axis=0))
    np.fill_diagonal(averaged, 1.0)
    return averaged


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(pearsonr(np.asarray(x), np.asarray(y)).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(np.asarray(x), np.asarray(y)).statistic)


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in metrics.columns:
        if column == "seed" or not pd.api.types.is_numeric_dtype(metrics[column]):
            continue
        values = metrics[column].to_numpy(dtype=float)
        rows.append({
            "metric": column,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n_seeds": len(values),
        })
    return pd.DataFrame(rows)


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.15, 1.10, letter, transform=ax.transAxes, fontsize=17,
            fontweight="bold", ha="left", va="top")


def panel_a_correlation_table(per_gene: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute mean-variance association on one common positive-moment panel.

    All displayed genes must have positive mean and variance in both sources.
    This avoids the artificial rank inflation that would result from the large
    tied block of genes whose mean and variance are simultaneously zero.
    """
    common_positive = (
        (per_gene["real_mean"].to_numpy(float) > 0)
        & (per_gene["real_variance"].to_numpy(float) > 0)
        & (per_gene["generated_mean"].to_numpy(float) > 0)
        & (per_gene["generated_variance"].to_numpy(float) > 0)
    )
    if int(common_positive.sum()) < 3:
        raise RuntimeError("Fewer than three genes have positive moments in both sources")

    rows: list[dict] = []
    for source, label in (("real", "Real scRNA-seq"), ("generated", "DeconvSC")):
        mean = per_gene.loc[common_positive, f"{source}_mean"].to_numpy(float)
        variance = per_gene.loc[common_positive, f"{source}_variance"].to_numpy(float)
        rows.append({
            "source": source,
            "display_label": label,
            "n_common_positive_genes": int(common_positive.sum()),
            "spearman_mean_variance": safe_spearman(mean, variance),
            "pearson_raw_mean_variance": safe_pearson(mean, variance),
            "pearson_log10_mean_variance": safe_pearson(
                np.log10(mean), np.log10(variance)
            ),
        })
    return pd.DataFrame(rows), common_positive


def draw_panel_a(
    ax_real: plt.Axes,
    ax_generated: plt.Axes,
    per_gene: pd.DataFrame,
    common_positive: np.ndarray,
    include_letter: bool,
) -> None:
    specifications = (
        (ax_real, "real", "Real scRNA-seq", REAL_COLOR),
        (ax_generated, "generated", "DeconvSC", GEN_COLOR),
    )
    all_means = np.concatenate([
        per_gene.loc[common_positive, "real_mean"].to_numpy(float),
        per_gene.loc[common_positive, "generated_mean"].to_numpy(float),
    ])
    all_variances = np.concatenate([
        per_gene.loc[common_positive, "real_variance"].to_numpy(float),
        per_gene.loc[common_positive, "generated_variance"].to_numpy(float),
    ])
    x_lower, x_upper = float(all_means.min()), float(all_means.max())
    y_lower, y_upper = float(all_variances.min()), float(all_variances.max())
    x_margin = 10 ** 0.08
    y_margin = 10 ** 0.08

    for ax, source, title, color in specifications:
        mean = per_gene.loc[common_positive, f"{source}_mean"].to_numpy(float)
        variance = per_gene.loc[common_positive, f"{source}_variance"].to_numpy(float)
        ax.scatter(mean, variance, s=5.0, alpha=0.25, color=color,
                   edgecolors="none", rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_lower / x_margin, x_upper * x_margin)
        ax.set_ylim(y_lower / y_margin, y_upper * y_margin)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10))
            axis.set_major_formatter(LogFormatterMathtext(base=10))
            axis.set_minor_locator(NullLocator())
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Per-gene mean")
        ax.set_ylabel("Per-gene variance")
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.grid(False)
    if include_letter:
        add_panel_letter(ax_real, "a")


def draw_panel_b(
    ax_real: plt.Axes,
    ax_generated: plt.Axes,
    corr_real: np.ndarray,
    corr_generated: np.ndarray,
    order: np.ndarray,
    include_letter: bool,
) -> matplotlib.image.AxesImage:
    real_ordered = corr_real[np.ix_(order, order)]
    generated_ordered = corr_generated[np.ix_(order, order)]
    image = ax_real.imshow(real_ordered, cmap=CORR_CMAP, vmin=-1.0, vmax=1.0,
                           interpolation="nearest", aspect="equal", rasterized=True)
    ax_generated.imshow(generated_ordered, cmap=CORR_CMAP, vmin=-1.0, vmax=1.0,
                        interpolation="nearest", aspect="equal", rasterized=True)
    # Quantitative results are reported in the caption and manuscript, not in-panel.
    ax_real.set_title("Real scRNA-seq", fontweight="bold")
    ax_generated.set_title("DeconvSC", fontweight="bold")
    for ax in (ax_real, ax_generated):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if include_letter:
        add_panel_letter(ax_real, "b")
    return image


def save_figures(
    outdir: Path,
    per_gene: pd.DataFrame,
    common_positive: np.ndarray,
    corr_real: np.ndarray,
    corr_generated: np.ndarray,
    order: np.ndarray,
) -> None:
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig_a, axes_a = plt.subplots(1, 2, figsize=(10.6, 5.1), dpi=600)
    draw_panel_a(
        axes_a[0], axes_a[1], per_gene, common_positive,
        include_letter=True,
    )
    fig_a.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.91, wspace=0.28)
    plot_style.save(fig_a, str(figures / "Supplementary_Figure5a_normalized_expression"), dpi=600)
    plt.close(fig_a)

    fig_b, axes_b = plt.subplots(1, 2, figsize=(10.6, 5.0), dpi=600)
    image = draw_panel_b(
        axes_b[0], axes_b[1], corr_real, corr_generated, order,
        include_letter=True,
    )
    fig_b.subplots_adjust(left=0.055, right=0.89, bottom=0.08, top=0.88, wspace=0.08)
    colorbar_ax = fig_b.add_axes([0.915, 0.18, 0.018, 0.62])
    colorbar = fig_b.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.ax.set_title("Gene–gene\nPearson r", fontsize=9, pad=8)
    colorbar.ax.tick_params(labelsize=8)
    plot_style.save(fig_b, str(figures / "Supplementary_Figure5b_coexpression"), dpi=600)
    plt.close(fig_b)

    fig = plt.figure(figsize=(11.2, 10.2), dpi=600)
    grid = fig.add_gridspec(
        2, 2, left=0.085, right=0.90, bottom=0.06, top=0.965,
        hspace=0.27, wspace=0.22, height_ratios=[1.0, 1.06],
    )
    ax_a1 = fig.add_subplot(grid[0, 0])
    ax_a2 = fig.add_subplot(grid[0, 1])
    ax_b1 = fig.add_subplot(grid[1, 0])
    ax_b2 = fig.add_subplot(grid[1, 1])
    draw_panel_a(
        ax_a1, ax_a2, per_gene, common_positive,
        include_letter=True,
    )
    image = draw_panel_b(
        ax_b1, ax_b2, corr_real, corr_generated, order,
        include_letter=True,
    )
    colorbar_ax = fig.add_axes([0.925, 0.105, 0.016, 0.35])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    colorbar.ax.set_title("Gene–gene\nPearson r", fontsize=9, pad=8)
    colorbar.ax.tick_params(labelsize=8)
    plot_style.save(fig, str(figures / "Supplementary_Figure5_corrected_AB"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    generated_path = args.generated.resolve()
    real_path = args.real.resolve()
    outdir = args.outdir.resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct subsampling seeds are required")
    for path in (generated_path, real_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for subdir in ("data", "figures", "logs"):
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    generated_full = sc.read_h5ad(generated_path)
    real_full = sc.read_h5ad(real_path)
    generated_original_shape = generated_full.shape
    real_original_shape = real_full.shape
    if not generated_full.var_names.is_unique or not real_full.var_names.is_unique:
        raise RuntimeError("Both inputs must have unique gene identifiers")
    if "Sample" not in generated_full.obs or "Cell_type" not in generated_full.obs:
        raise RuntimeError("Generated data require obs columns Sample and Cell_type")
    if "sample" not in real_full.obs or "cell type" not in real_full.obs:
        raise RuntimeError("Real data require obs columns sample and cell type")

    # Preserve real-data gene order; do not use a Python set as an ordering device.
    generated_gene_set = set(generated_full.var_names.astype(str))
    common_genes = [gene for gene in real_full.var_names.astype(str) if gene in generated_gene_set]
    shared_types = sorted(
        set(real_full.obs["cell type"].astype(str))
        & set(generated_full.obs["Cell_type"].astype(str))
    )
    real_mask = (
        real_full.obs["sample"].astype(str).isin(DONORS)
        & real_full.obs["cell type"].astype(str).isin(shared_types)
    )
    generated_mask = (
        generated_full.obs["Sample"].astype(str).isin(DONORS)
        & generated_full.obs["Cell_type"].astype(str).isin(shared_types)
    )
    real_scope = real_full[real_mask, common_genes].copy()
    generated_scope = generated_full[generated_mask, common_genes].copy()
    del real_full, generated_full
    gc.collect()

    real_obs = pd.DataFrame({
        "cell_id": real_scope.obs_names.astype(str),
        "sample": real_scope.obs["sample"].astype(str).to_numpy(),
        "cell_type": real_scope.obs["cell type"].astype(str).to_numpy(),
    })
    generated_obs = pd.DataFrame({
        "cell_id": generated_scope.obs_names.astype(str),
        "sample": generated_scope.obs["Sample"].astype(str).to_numpy(),
        "cell_type": generated_scope.obs["Cell_type"].astype(str).to_numpy(),
    })

    print("Transforming both sources to log1p CP10K on the common-gene panel...")
    real_norm = real_scope.copy()
    sc.pp.normalize_total(real_norm, target_sum=TARGET_SUM)
    sc.pp.log1p(real_norm)

    generated_norm = np.asarray(generated_scope.X, dtype=np.float32).copy()
    if not np.isfinite(generated_norm).all() or np.min(generated_norm) < 0:
        raise RuntimeError("Generated X must be finite, nonnegative log1p expression")
    np.expm1(generated_norm, out=generated_norm)
    generated_libraries = generated_norm.sum(axis=1, dtype=np.float64)
    if np.any(generated_libraries <= 0):
        raise RuntimeError("Generated data contain nonpositive common-panel libraries")
    generated_norm *= (TARGET_SUM / generated_libraries).astype(np.float32)[:, None]
    np.log1p(generated_norm, out=generated_norm)

    strata_index = pd.MultiIndex.from_product(
        [DONORS, shared_types], names=["sample", "cell_type"]
    )
    real_counts = real_obs.groupby(["sample", "cell_type"], observed=True).size().reindex(
        strata_index, fill_value=0
    )
    generated_counts = generated_obs.groupby(["sample", "cell_type"], observed=True).size().reindex(
        strata_index, fill_value=0
    )
    stratum_counts = pd.DataFrame({
        "real_n": real_counts.astype(int),
        "generated_n": generated_counts.astype(int),
    })
    stratum_counts["matched_n"] = stratum_counts[["real_n", "generated_n"]].min(axis=1)
    stratum_counts["eligible"] = stratum_counts["matched_n"] > 0
    stratum_counts.reset_index().to_csv(
        outdir / "data/sample_celltype_counts.csv", index=False
    )
    eligible_strata = [
        (row.sample, row.cell_type)
        for row in stratum_counts.reset_index().itertuples(index=False)
        if row.eligible
    ]

    real_lookup = {
        key: np.flatnonzero(
            (real_obs["sample"].to_numpy() == key[0])
            & (real_obs["cell_type"].to_numpy() == key[1])
        )
        for key in eligible_strata
    }
    generated_lookup = {
        key: np.flatnonzero(
            (generated_obs["sample"].to_numpy() == key[0])
            & (generated_obs["cell_type"].to_numpy() == key[1])
        )
        for key in eligible_strata
    }

    print("Creating deterministic sample x cell-type matched draws...")
    selections: dict[int, dict] = {}
    matched_rows: list[dict] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        real_selected: list[int] = []
        generated_selected: list[int] = []
        by_stratum = {}
        for key in eligible_strata:
            matched_n = int(stratum_counts.loc[key, "matched_n"])
            real_indices = np.sort(rng.choice(real_lookup[key], size=matched_n, replace=False))
            generated_indices = np.sort(
                rng.choice(generated_lookup[key], size=matched_n, replace=False)
            )
            real_selected.extend(real_indices.tolist())
            generated_selected.extend(generated_indices.tolist())
            by_stratum[key] = (real_indices, generated_indices)
            for source, indices, obs in (
                ("real", real_indices, real_obs),
                ("generated", generated_indices, generated_obs),
            ):
                for draw_order, position in enumerate(indices):
                    matched_rows.append({
                        "seed": seed,
                        "source": source,
                        "sample": key[0],
                        "cell_type": key[1],
                        "draw_order_within_stratum": draw_order,
                        "cell_id": obs.iloc[position]["cell_id"],
                    })
        selections[seed] = {
            "real": np.asarray(real_selected, dtype=int),
            "generated": np.asarray(generated_selected, dtype=int),
            "by_stratum": by_stratum,
        }
    matched_cells = pd.DataFrame(matched_rows)
    matched_cells.to_csv(outdir / "data/matched_cell_ids.csv", index=False)

    print(f"Selecting {N_HVG} reference-defined HVGs from real log1p CP10K data...")
    sc.pp.highly_variable_genes(real_norm, n_top_genes=N_HVG, flavor="seurat", inplace=True)
    hvg_genes = real_norm.var_names[real_norm.var["highly_variable"]].astype(str).to_numpy()
    if len(hvg_genes) != N_HVG:
        raise RuntimeError(f"Expected {N_HVG} HVGs, obtained {len(hvg_genes)}")
    common_index = pd.Index(common_genes)
    hvg_indices = common_index.get_indexer(hvg_genes)
    if np.any(hvg_indices < 0):
        raise RuntimeError("An HVG is absent from the common-gene panel")

    # Freeze one evaluable panel: nonconstant in both sources in every matched draw.
    evaluable = np.ones(N_HVG, dtype=bool)
    for seed in seeds:
        real_hvg = dense(real_norm.X[selections[seed]["real"]][:, hvg_indices])
        generated_hvg = generated_norm[selections[seed]["generated"]][:, hvg_indices]
        evaluable &= (np.std(real_hvg, axis=0) > EPS) & (np.std(generated_hvg, axis=0) > EPS)
    evaluable_genes = hvg_genes[evaluable]
    evaluable_indices = hvg_indices[evaluable]
    if len(evaluable_genes) < 2:
        raise RuntimeError("Fewer than two HVGs are jointly evaluable")

    real_hvg_full = dense(real_norm.X[:, hvg_indices])
    generated_hvg_full = generated_norm[:, hvg_indices]
    real_full_mean, real_full_var, real_full_detection = sparse_or_dense_moments(real_hvg_full)
    gen_full_mean, gen_full_var, gen_full_detection = sparse_or_dense_moments(generated_hvg_full)
    hvg_panel = pd.DataFrame({
        "gene": hvg_genes,
        "reference_hvg": True,
        "evaluable_all_seeds": evaluable,
        "real_scope_mean": real_full_mean,
        "generated_scope_mean": gen_full_mean,
        "real_scope_variance": real_full_var,
        "generated_scope_variance": gen_full_var,
        "real_scope_detection_fraction": real_full_detection,
        "generated_scope_detection_fraction": gen_full_detection,
    })
    hvg_panel["exclusion_reason"] = np.where(
        hvg_panel["evaluable_all_seeds"],
        "included",
        np.where(
            (hvg_panel["generated_scope_mean"] == 0)
            & (hvg_panel["generated_scope_variance"] == 0),
            "generated_all_zero",
            "constant_in_one_or_more_matched_draws",
        ),
    )
    hvg_panel.to_csv(outdir / "data/hvg_panel.csv", index=False)
    del real_hvg_full, generated_hvg_full
    gc.collect()

    print("Computing seed-level moments, diversity diagnostics, and network fidelity...")
    metric_rows: list[dict] = []
    stratum_metric_rows: list[dict] = []
    real_means: list[np.ndarray] = []
    generated_means: list[np.ndarray] = []
    real_variances: list[np.ndarray] = []
    generated_variances: list[np.ndarray] = []
    real_detections: list[np.ndarray] = []
    generated_detections: list[np.ndarray] = []
    real_correlations: list[np.ndarray] = []
    generated_correlations: list[np.ndarray] = []

    for seed in seeds:
        real_indices = selections[seed]["real"]
        generated_indices = selections[seed]["generated"]
        real_matrix = real_norm.X[real_indices]
        generated_matrix = generated_norm[generated_indices]
        real_mean, real_variance, real_detection = sparse_or_dense_moments(real_matrix)
        gen_mean, gen_variance, gen_detection = sparse_or_dense_moments(generated_matrix)
        real_means.append(real_mean)
        generated_means.append(gen_mean)
        real_variances.append(real_variance)
        generated_variances.append(gen_variance)
        real_detections.append(real_detection)
        generated_detections.append(gen_detection)

        real_eval = dense(real_matrix[:, evaluable_indices])
        gen_eval = generated_matrix[:, evaluable_indices]
        corr_real = gene_correlation(real_eval)
        corr_generated = gene_correlation(gen_eval)
        real_correlations.append(corr_real)
        generated_correlations.append(corr_generated)
        triangle = np.triu_indices(len(evaluable_genes), k=1)
        real_edges = corr_real[triangle]
        generated_edges = corr_generated[triangle]

        positive_mean = (real_mean > 0) | (gen_mean > 0)
        positive_variance = (real_variance > 0) | (gen_variance > 0)
        metric_rows.append({
            "seed": seed,
            "matched_cells_per_source": len(real_indices),
            "eligible_strata": len(eligible_strata),
            "reference_hvg_n": N_HVG,
            "evaluable_hvg_n": len(evaluable_genes),
            "hvg_coverage": len(evaluable_genes) / N_HVG,
            "mean_log1p_pearson": safe_pearson(
                np.log1p(real_mean[positive_mean]), np.log1p(gen_mean[positive_mean])
            ),
            "mean_spearman": safe_spearman(
                real_mean[positive_mean], gen_mean[positive_mean]
            ),
            "mean_log1p_mae": float(np.median(np.abs(
                np.log1p(real_mean[positive_mean]) - np.log1p(gen_mean[positive_mean])
            ))),
            "variance_log1p_pearson": safe_pearson(
                np.log1p(real_variance[positive_variance]),
                np.log1p(gen_variance[positive_variance]),
            ),
            "variance_spearman": safe_spearman(
                real_variance[positive_variance], gen_variance[positive_variance]
            ),
            "variance_log1p_mae": float(np.median(np.abs(
                np.log1p(real_variance[positive_variance])
                - np.log1p(gen_variance[positive_variance])
            ))),
            "detection_pearson": safe_pearson(real_detection, gen_detection),
            "detection_spearman": safe_spearman(real_detection, gen_detection),
            "detection_mae": float(np.mean(np.abs(real_detection - gen_detection))),
            "network_edge_pearson": safe_pearson(real_edges, generated_edges),
            "network_edge_spearman": safe_spearman(real_edges, generated_edges),
            "network_edge_mae": float(np.mean(np.abs(real_edges - generated_edges))),
            "network_edge_rmse": float(np.sqrt(np.mean((real_edges - generated_edges) ** 2))),
        })

        for key, (real_stratum_indices, generated_stratum_indices) in selections[seed]["by_stratum"].items():
            matched_n = len(real_stratum_indices)
            if matched_n < 10:
                continue
            real_stratum = dense(real_norm.X[real_stratum_indices][:, evaluable_indices])
            gen_stratum = generated_norm[generated_stratum_indices][:, evaluable_indices]
            real_var = real_stratum.var(axis=0)
            gen_var = gen_stratum.var(axis=0)
            valid = real_var > EPS
            if not np.any(valid):
                continue
            ratios = gen_var[valid] / real_var[valid]
            stratum_metric_rows.append({
                "seed": seed,
                "sample": key[0],
                "cell_type": key[1],
                "matched_n": matched_n,
                "evaluable_genes_with_real_variance": int(valid.sum()),
                "median_variance_ratio_generated_over_real": float(np.median(ratios)),
                "median_log2_variance_ratio_generated_over_real": float(
                    np.median(np.log2(np.maximum(ratios, EPS)))
                ),
                "fraction_genes_generated_variance_at_least_half_real": float(
                    np.mean(ratios >= 0.5)
                ),
            })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(outdir / "data/metrics_by_seed.csv", index=False)
    summary = metric_summary(metrics)
    summary.to_csv(outdir / "data/metric_summary.csv", index=False)
    stratum_metrics = pd.DataFrame(stratum_metric_rows)
    stratum_metrics.to_csv(outdir / "data/within_stratum_variance_by_seed.csv", index=False)
    within_seed_summary = stratum_metrics.groupby("seed", as_index=False).agg(
        median_variance_ratio_generated_over_real=(
            "median_variance_ratio_generated_over_real", "median"
        ),
        median_fraction_genes_generated_variance_at_least_half_real=(
            "fraction_genes_generated_variance_at_least_half_real", "median"
        ),
        evaluated_strata=("cell_type", "size"),
    )
    within_seed_summary.to_csv(
        outdir / "data/within_stratum_variance_summary_by_seed.csv", index=False
    )

    def aggregate(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        stack = np.stack(values, axis=0)
        return stack.mean(axis=0), stack.std(axis=0, ddof=1)

    real_mean_avg, real_mean_sd = aggregate(real_means)
    gen_mean_avg, gen_mean_sd = aggregate(generated_means)
    real_var_avg, real_var_sd = aggregate(real_variances)
    gen_var_avg, gen_var_sd = aggregate(generated_variances)
    real_det_avg, real_det_sd = aggregate(real_detections)
    gen_det_avg, gen_det_sd = aggregate(generated_detections)
    per_gene = pd.DataFrame({
        "gene": common_genes,
        "real_mean": real_mean_avg,
        "real_mean_seed_sd": real_mean_sd,
        "generated_mean": gen_mean_avg,
        "generated_mean_seed_sd": gen_mean_sd,
        "real_variance": real_var_avg,
        "real_variance_seed_sd": real_var_sd,
        "generated_variance": gen_var_avg,
        "generated_variance_seed_sd": gen_var_sd,
        "real_detection_fraction": real_det_avg,
        "real_detection_seed_sd": real_det_sd,
        "generated_detection_fraction": gen_det_avg,
        "generated_detection_seed_sd": gen_det_sd,
    })
    per_gene.to_csv(outdir / "data/per_gene_normalized_moments.csv", index=False)
    panel_a_correlations, panel_a_common_positive = panel_a_correlation_table(per_gene)
    panel_a_correlations.to_csv(
        outdir / "data/panel_a_mean_variance_correlations.csv", index=False
    )

    corr_real_average = fisher_average(real_correlations)
    corr_generated_average = fisher_average(generated_correlations)
    distance = np.clip(1.0 - corr_real_average, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    clustering = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
    order = leaves_list(clustering)
    np.savez_compressed(
        outdir / "data/correlation_matrices.npz",
        genes=evaluable_genes,
        real_fisher_average=corr_real_average,
        generated_fisher_average=corr_generated_average,
        real_cluster_order=order,
        seeds=np.asarray(seeds, dtype=int),
    )

    input_manifest = pd.DataFrame([
        {
            "name": "deconvsc_figure3_generated",
            "path": str(generated_path),
            "sha256": sha256_file(generated_path),
            "n_obs_original": generated_original_shape[0],
            "n_vars_original": generated_original_shape[1],
            "declared_scale": "continuous log1p decoder output",
            "role": "read-only generated input",
        },
        {
            "name": "gse141115_real_test",
            "path": str(real_path),
            "sha256": sha256_file(real_path),
            "n_obs_original": real_original_shape[0],
            "n_vars_original": real_original_shape[1],
            "declared_scale": "raw UMI counts",
            "role": "read-only ground truth",
        },
    ])
    input_manifest.to_csv(outdir / "input_manifest.csv", index=False)

    analysis_config = {
        "analysis_id": "supplementary_figure5_corrected_v3",
        "generated_input": str(generated_path),
        "real_input": str(real_path),
        "donors": DONORS,
        "shared_cell_types": shared_types,
        "common_gene_count": len(common_genes),
        "normalization": "expm1 generated only -> common-panel CP10K -> log1p for both sources",
        "matching": "without-replacement within sample x cell_type at min(real_n, generated_n)",
        "subsampling_seeds": seeds,
        "matched_cells_per_source_per_seed": int(metrics["matched_cells_per_source"].iloc[0]),
        "eligible_strata": len(eligible_strata),
        "reference_hvg_definition": "Scanpy seurat flavor; n_top_genes=500 from real scoped log1p CP10K",
        "evaluable_hvg_definition": "nonconstant in both sources for every matched seed",
        "evaluable_hvg_count": len(evaluable_genes),
        "median_within_stratum_variance_ratio_generated_over_real": float(
            within_seed_summary["median_variance_ratio_generated_over_real"].median()
        ),
        "panel_a_layout": "mean versus variance; real left and DeconvSC right",
        "panel_a_gene_panel": (
            "same genes in both panels; positive mean and variance in both sources"
        ),
        "panel_a_gene_count": int(panel_a_common_positive.sum()),
        "panel_a_displayed_correlation": "none",
        "panel_a_caption_correlation": (
            "cross-source Spearman correlation: real versus generated per-gene moments; "
            "median and range across matched-subsampling seeds"
        ),
        "panel_a_cross_source_mean_spearman_median": float(
            metrics["mean_spearman"].median()
        ),
        "panel_a_cross_source_mean_spearman_min": float(metrics["mean_spearman"].min()),
        "panel_a_cross_source_mean_spearman_max": float(metrics["mean_spearman"].max()),
        "panel_a_cross_source_variance_spearman_median": float(
            metrics["variance_spearman"].median()
        ),
        "panel_a_cross_source_variance_spearman_min": float(
            metrics["variance_spearman"].min()
        ),
        "panel_a_cross_source_variance_spearman_max": float(
            metrics["variance_spearman"].max()
        ),
        "panel_a_correlation_sensitivity_file": (
            "data/panel_a_mean_variance_correlations.csv"
        ),
        "network_edge_metric": "upper triangle only; diagonal and symmetric duplicate excluded",
        "correlation_matrix_aggregation": "element-wise Fisher z mean across seeds",
        "clustering": "average linkage on 1-real Fisher-averaged correlation; same order applied to both",
        "heatmap_entry": (
            "within-source gene-gene Pearson correlation across matched cells"
        ),
        "heatmap_color_scale": [-1.0, 1.0],
        "heatmap_color_direction": "negative blue; zero white; positive red",
        "heatmap_diagonal": "exactly 1 by self-correlation",
        "panel_b_displayed_metric_results": (
            "none; HVG count and off-diagonal edge PCC are reported in the caption and manuscript"
        ),
        "hypothesis_tests_displayed": False,
        "raw_count_or_zero_fraction_claim": False,
    }
    if not args.skip_figures:
        print("Rendering publication figures...")
        save_figures(
            outdir, per_gene, panel_a_common_positive, corr_real_average,
            corr_generated_average, order,
        )
    (outdir / "analysis_config.json").write_text(
        json.dumps(analysis_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    within_ratio = within_seed_summary["median_variance_ratio_generated_over_real"]
    mean_cross_source_rho = metrics["mean_spearman"]
    variance_cross_source_rho = metrics["variance_spearman"]
    edge_cross_source_pcc = metrics["network_edge_pearson"]
    caption = f"""# Corrected Supplementary Figure 5 caption

**Supplementary Figure 5. Fidelity of matched normalized expression and global co-expression structure in DeconvSC-generated cells.**

**a**, Per-gene mean-variance relationships in real scRNA-seq (left) and DeconvSC output (right) after both sources were placed on the same log1p CP10K scale. The same {int(panel_a_common_positive.sum()):,} genes with positive mean and variance in both sources are displayed in both panels; zero-valued moments cannot be shown on logarithmic axes. Values are averages across {len(seeds)} deterministic matched-subsampling seeds. For each seed, {int(metrics['matched_cells_per_source'].iloc[0]):,} cells per source were matched without replacement within {len(eligible_strata)} nonempty sample-by-cell-type strata from LDK1-LDK3. Across the ten matched draws, the Spearman correlation between real and DeconvSC per-gene normalized means had a median of {mean_cross_source_rho.median():.3f} (range, {mean_cross_source_rho.min():.3f}–{mean_cross_source_rho.max():.3f}), whereas the corresponding correlation for per-gene variances had a median of {variance_cross_source_rho.median():.3f} (range, {variance_cross_source_rho.min():.3f}–{variance_cross_source_rho.max():.3f}). Because moments were computed after pooling all composition-matched strata, variance contains both within-stratum variability and between-stratum mean differences. These are moments of normalized log expression and therefore quantify representation-level location and variability, not raw-count mean, count dispersion, or sequencing noise. **b**, Fisher-averaged gene-gene Pearson-correlation matrices. Of the {N_HVG} reference-defined highly variable genes, {len(evaluable_genes)} that were nonconstant in both sources in every matched draw were retained. Every heatmap entry is a within-source Pearson correlation between two gene-expression vectors across matched cells; the diagonal is exactly 1 because each gene is correlated with itself. The color scale is fixed from -1 to 1, with negative correlations in blue and positive correlations in red. Genes were ordered once by average-linkage clustering of the real-data correlation matrix, and the identical order was applied to DeconvSC. Off-diagonal edge PCC was defined as the Pearson correlation between corresponding unique off-diagonal upper-triangle entries of the two gene-correlation matrices, excluding the diagonal and symmetric duplicates. Across the ten matched draws, its median was {edge_cross_source_pcc.median():.3f} (range, {edge_cross_source_pcc.min():.3f}–{edge_cross_source_pcc.max():.3f}). It is not a direct correlation between the two raw expression matrices. No hypothesis tests are displayed. These analyses evaluate normalized-expression and co-expression fidelity, not raw-count dispersion, zero-fraction fidelity, absence of mode collapse, causal regulation, or gene-regulatory-network reconstruction.
"""
    (outdir / "figures/Supplementary_Figure5_corrected_caption.md").write_text(
        caption, encoding="utf-8"
    )

    run_metadata = {
        "status": "completed",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scanpy": sc.__version__,
        "anndata": ad.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seeds": seeds,
        "skip_figures": bool(args.skip_figures),
    }
    (outdir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    checks: list[dict] = []

    def add_check(name: str, passed: bool, observed, expected) -> None:
        checks.append({
            "check": name,
            "passed": bool(passed),
            "observed": str(observed),
            "expected": str(expected),
        })

    add_check("common_genes", len(common_genes) == 16801, len(common_genes), 16801)
    add_check("shared_cell_types", len(shared_types) == 16, len(shared_types), 16)
    add_check("eligible_strata", len(eligible_strata) == 46, len(eligible_strata), 46)
    add_check(
        "matched_cells_equal",
        bool((metrics["matched_cells_per_source"] == metrics["matched_cells_per_source"].iloc[0]).all()),
        metrics["matched_cells_per_source"].tolist(),
        "identical across seeds and sources",
    )
    add_check("reference_hvg_n", len(hvg_genes) == N_HVG, len(hvg_genes), N_HVG)
    add_check("evaluable_hvg_nonempty", len(evaluable_genes) > 1, len(evaluable_genes), ">1")
    add_check("metrics_finite", bool(np.isfinite(metrics.select_dtypes("number")).all().all()),
              bool(np.isfinite(metrics.select_dtypes("number")).all().all()), True)
    add_check(
        "panel_a_common_positive_genes",
        int(panel_a_common_positive.sum()) == 4575,
        int(panel_a_common_positive.sum()),
        4575,
    )
    add_check(
        "panel_a_correlations_finite",
        bool(np.isfinite(panel_a_correlations.select_dtypes("number")).all().all()),
        bool(np.isfinite(panel_a_correlations.select_dtypes("number")).all().all()),
        True,
    )
    add_check("correlations_finite", bool(np.isfinite(corr_real_average).all() and np.isfinite(corr_generated_average).all()),
              bool(np.isfinite(corr_real_average).all() and np.isfinite(corr_generated_average).all()), True)
    diagonal_is_one = bool(
        np.allclose(np.diag(corr_real_average), 1.0, atol=0.0, rtol=0.0)
        and np.allclose(np.diag(corr_generated_average), 1.0, atol=0.0, rtol=0.0)
    )
    add_check("correlation_diagonal_exactly_one", diagonal_is_one,
              diagonal_is_one, True)
    add_check("upper_triangle_definition", len(np.triu_indices(len(evaluable_genes), 1)[0]) == len(evaluable_genes) * (len(evaluable_genes) - 1) // 2,
              len(np.triu_indices(len(evaluable_genes), 1)[0]), len(evaluable_genes) * (len(evaluable_genes) - 1) // 2)
    add_check("no_hypothesis_tests", not analysis_config["hypothesis_tests_displayed"],
              analysis_config["hypothesis_tests_displayed"], False)
    if not args.skip_figures:
        expected_bases = [
            "Supplementary_Figure5a_normalized_expression",
            "Supplementary_Figure5b_coexpression",
            "Supplementary_Figure5_corrected_AB",
        ]
        missing_figures = [
            f"{base}.{extension}"
            for base in expected_bases
            for extension in ("png", "pdf", "svg", "emf")
            if not (outdir / "figures" / f"{base}.{extension}").is_file()
        ]
        add_check("all_figure_formats", not missing_figures,
                  missing_figures if missing_figures else "none", "none missing")
        combined_svg = (outdir / "figures/Supplementary_Figure5_corrected_AB.svg").read_text(
            encoding="utf-8"
        )
        add_check("editable_arial_svg", "font-family: 'Arial'" in combined_svg,
                  "Arial" if "font-family: 'Arial'" in combined_svg else "missing", "Arial")
        panel_a_svg = (outdir / "figures/Supplementary_Figure5a_normalized_expression.svg").read_text(
            encoding="utf-8"
        )
        add_check("panel_a_no_correlation_annotation", "Spearman" not in panel_a_svg,
                  "absent" if "Spearman" not in panel_a_svg else "present", "absent")
        panel_b_svg = (outdir / "figures/Supplementary_Figure5b_coexpression.svg").read_text(
            encoding="utf-8"
        )
        metric_annotation_tokens = (
            "Off-diagonal edge PCC", "Reference-clustered", f"{len(evaluable_genes)}/{N_HVG} HVGs"
        )
        panel_b_has_metric_annotation = any(
            token in panel_b_svg for token in metric_annotation_tokens
        )
        combined_has_metric_annotation = any(
            token in combined_svg for token in metric_annotation_tokens
        )
        add_check(
            "panel_b_no_metric_result_annotation",
            not panel_b_has_metric_annotation,
            "absent" if not panel_b_has_metric_annotation else "present",
            "absent",
        )
        add_check(
            "combined_panel_b_no_metric_result_annotation",
            not combined_has_metric_annotation,
            "absent" if not combined_has_metric_annotation else "present",
            "absent",
        )
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(outdir / "validation_checks.csv", index=False)
    if not bool(checks_df["passed"].all()):
        print(checks_df.to_string(index=False))
        raise SystemExit("Validation failed")

    print(metrics.to_string(index=False))
    print(f"Reference HVGs evaluable in every seed: {len(evaluable_genes)}/{N_HVG}")
    print(
        "Median within-stratum generated/real variance ratio: "
        f"{within_ratio.median():.4f} across seeds "
        f"[{within_ratio.min():.4f}, {within_ratio.max():.4f}]"
    )
    print(f"Validation passed: {len(checks_df)}/{len(checks_df)} checks")


if __name__ == "__main__":
    main()
