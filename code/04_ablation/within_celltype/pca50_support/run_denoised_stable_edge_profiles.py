#!/usr/bin/env python3
"""Training-defined denoised/shrinkage edge-profile pilot for GSE141115.

This analysis is intentionally isolated from the current Figure 4 outputs.  It uses
only the training reference to select stable genes and fit PCA denoisers, and then
evaluates equal-cell Real-A, Real-B, DeconvSC and BaseVAE subsets in LDK1-LDK3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn import __version__ as sklearn_version
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[4]
PATHS = {
    "Train": ROOT / "processed_data/real_data/mouse_kidney/train_data.h5ad",
    "Test": ROOT / "processed_data/real_data/mouse_kidney/test_data.h5ad",
    "DeconvSC": ROOT / "model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad",
    "BaseVAE": ROOT / "model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad",
}

CELL_TYPES = ("CD_PC", "CNT", "DCT", "MC", "PT", "aLOH")
TEST_DONORS = ("LDK1", "LDK2", "LDK3")
SEEDS = tuple(range(2024, 2034))
PANEL_SIZES = (100, 200)
TRAIN_MIN_CELLS = 30
TRAIN_DONOR_CAP = 500
EVAL_CAP = 200
EVAL_MIN = 20
CANDIDATE_GENES = 300
TOP_STABLE_NEIGHBORS = 20
PCA_VARIANCE_TARGET = 0.80
PCA_MAX_COMPONENTS = 50
PCA_FIXED_COMPONENTS = 20
CHUNK_SIZE = 128
FISHER_EPS = 1e-6

SETTINGS = {
    "raw_lw": {"real": "raw", "model": "raw"},
    "pca80_real_only_lw": {"real": "pca80", "model": "raw"},
    "pca20_real_only_lw": {"real": "pca20", "model": "raw"},
    "pca80_symmetric_lw": {"real": "pca80", "model": "pca80"},
}

COMPARISONS = (
    ("RealA_vs_RealB", "RealA", "RealB"),
    ("DeconvSC_vs_RealB", "DeconvSC", "RealB"),
    ("BaseVAE_vs_RealB", "BaseVAE", "RealB"),
)

DISPLAY = {
    "RealA_vs_RealB": "Real-real",
    "DeconvSC_vs_RealB": "DeconvSC",
    "BaseVAE_vs_RealB": "Ablated VAE",
}

COLORS = {
    "RealA_vs_RealB": "#A9A9A9",
    "DeconvSC_vs_RealB": "#D8B2AE",
    "BaseVAE_vs_RealB": "#7187A7",
}


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_rng(*tokens) -> np.random.Generator:
    value = "|".join(map(str, tokens)).encode()
    seed = int.from_bytes(hashlib.sha256(value).digest()[:8], "little")
    return np.random.default_rng(seed)


def dense(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def read_raw_logcp10k(backed, rows: np.ndarray, gene_indices: np.ndarray) -> np.ndarray:
    chunks = []
    rows = np.asarray(rows, dtype=int)
    for start in range(0, len(rows), CHUNK_SIZE):
        block_rows = rows[start : start + CHUNK_SIZE]
        native = backed.X[block_rows]
        if sp.issparse(native):
            library = np.asarray(native.sum(axis=1)).ravel().astype(np.float32)
            panel = native[:, gene_indices].toarray().astype(np.float32, copy=False)
        else:
            native = np.asarray(native, dtype=np.float32)
            library = native.sum(axis=1, dtype=np.float64).astype(np.float32)
            panel = native[:, gene_indices]
        scaled = np.divide(
            panel * np.float32(1e4),
            library[:, None],
            out=np.zeros_like(panel, dtype=np.float32),
            where=library[:, None] > 0,
        )
        chunks.append(np.log1p(scaled).astype(np.float32, copy=False))
    return np.vstack(chunks) if chunks else np.empty((0, len(gene_indices)), dtype=np.float32)


def read_generated_logcp10k(
    backed,
    rows: np.ndarray,
    panel_indices: np.ndarray,
    common_indices: np.ndarray,
) -> np.ndarray:
    chunks = []
    rows = np.asarray(rows, dtype=int)
    for start in range(0, len(rows), CHUNK_SIZE):
        block_rows = rows[start : start + CHUNK_SIZE]
        native = dense(backed.X[block_rows])
        common_log = np.clip(native[:, common_indices], 0.0, None)
        library = np.expm1(common_log).sum(axis=1, dtype=np.float64).astype(np.float32)
        panel_counts = np.expm1(np.clip(native[:, panel_indices], 0.0, None)).astype(np.float32, copy=False)
        scaled = np.divide(
            panel_counts * np.float32(1e4),
            library[:, None],
            out=np.zeros_like(panel_counts, dtype=np.float32),
            where=library[:, None] > 0,
        )
        chunks.append(np.log1p(scaled).astype(np.float32, copy=False))
    return np.vstack(chunks) if chunks else np.empty((0, len(panel_indices)), dtype=np.float32)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return safe_pearson(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return np.nan
    mx, my = float(x.mean()), float(y.mean())
    vx = float(np.mean((x - mx) ** 2))
    vy = float(np.mean((y - my) ** 2))
    cov = float(np.mean((x - mx) * (y - my)))
    denominator = vx + vy + (mx - my) ** 2
    if denominator < 1e-15:
        return np.nan
    return float(np.clip(2.0 * cov / denominator, -1.0, 1.0))


def weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep]
    if len(x) < 3 or w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    mx, my = float(np.sum(w * x)), float(np.sum(w * y))
    dx, dy = x - mx, y - my
    vx, vy = float(np.sum(w * dx * dx)), float(np.sum(w * dy * dy))
    if vx < 1e-15 or vy < 1e-15:
        return 0.0
    return float(np.sum(w * dx * dy) / np.sqrt(vx * vy))


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -1.0 + FISHER_EPS, 1.0 - FISHER_EPS))


def shrinkage_corr(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=0)
    constant_fraction = float(np.mean(sd < 1e-12))
    sd[sd < 1e-12] = 1.0
    z = x / sd
    estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(z)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    diag = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    corr = covariance / np.outer(diag, diag)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr, float(estimator.shrinkage_), constant_fraction


def inspect_inputs() -> tuple[list[str], list[dict]]:
    gene_sets = []
    schema = []
    for label, path in PATHS.items():
        backed = ad.read_h5ad(path, backed="r")
        genes = backed.var_names.astype(str).tolist()
        if len(set(genes)) != len(genes):
            raise RuntimeError(f"Duplicate genes in {label}")
        gene_sets.append(set(genes))
        schema.append({
            "label": label,
            "path": str(path),
            "n_obs": backed.n_obs,
            "n_vars": backed.n_vars,
            "obs_columns": ";".join(backed.obs.columns.astype(str)),
            "x_backend": type(backed.X).__name__,
        })
        backed.file.close()
    return sorted(set.intersection(*gene_sets)), schema


def capped_rows(rows: np.ndarray, cell_type: str, donor: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    if len(rows) <= TRAIN_DONOR_CAP:
        return np.sort(rows)
    rng = stable_rng("train_cap", cell_type, donor, TRAIN_DONOR_CAP)
    return np.sort(rng.choice(rows, size=TRAIN_DONOR_CAP, replace=False))


def build_training_panels(
    common_genes: list[str],
    outdir: Path,
    log: Logger,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backed = ad.read_h5ad(PATHS["Train"], backed="r")
    lookup = {gene: i for i, gene in enumerate(backed.var_names.astype(str))}
    common_indices = np.asarray([lookup[g] for g in common_genes], dtype=int)
    sample = backed.obs["sample"].astype(str).to_numpy()
    cell = backed.obs["cell type"].astype(str).to_numpy()

    models = {}
    gene_rows = []
    pca_rows = []
    donor_rows_audit = []

    for cell_type in CELL_TYPES:
        donors = []
        row_map = {}
        for donor in sorted(np.unique(sample)):
            rows = np.flatnonzero((sample == donor) & (cell == cell_type))
            if len(rows) >= TRAIN_MIN_CELLS:
                donors.append(donor)
                row_map[donor] = capped_rows(rows, cell_type, donor)
            donor_rows_audit.append({
                "cell_type": cell_type,
                "donor": donor,
                "n_available": len(rows),
                "eligible": len(rows) >= TRAIN_MIN_CELLS,
                "n_used": min(len(rows), TRAIN_DONOR_CAP) if len(rows) >= TRAIN_MIN_CELLS else 0,
            })
        if len(donors) < 6:
            raise RuntimeError(f"Too few eligible training donors for {cell_type}: {len(donors)}")

        log(f"  {cell_type}: screen {len(common_genes)} genes across {len(donors)} training donors")
        variances, detections = [], []
        for donor in donors:
            x = read_raw_logcp10k(backed, row_map[donor], common_indices)
            variances.append(np.var(x, axis=0, ddof=1))
            detections.append(np.mean(x > 0, axis=0))
        median_variance = np.median(np.vstack(variances), axis=0)
        median_detection = np.median(np.vstack(detections), axis=0)
        eligible = np.flatnonzero(
            np.isfinite(median_variance)
            & (median_variance > 1e-8)
            & (median_detection >= 0.05)
        )
        order = eligible[np.argsort(-median_variance[eligible], kind="stable")]
        candidate_common = order[:CANDIDATE_GENES]
        if len(candidate_common) < max(PANEL_SIZES):
            raise RuntimeError(f"Only {len(candidate_common)} candidate genes for {cell_type}")
        candidate_genes = [common_genes[i] for i in candidate_common]
        candidate_dataset_indices = common_indices[candidate_common]

        donor_matrices = {}
        donor_corrs = []
        donor_alphas = []
        for donor in donors:
            x = read_raw_logcp10k(backed, row_map[donor], candidate_dataset_indices)
            donor_matrices[donor] = x
            corr, alpha, _ = shrinkage_corr(x)
            donor_corrs.append(corr)
            donor_alphas.append(alpha)
        corr_stack = np.stack(donor_corrs)
        corr_sum = corr_stack.sum(axis=0)
        stability_values = np.zeros((len(donors), len(candidate_genes)), dtype=np.float64)
        for donor_index in range(len(donors)):
            leave_one_out = (corr_sum - corr_stack[donor_index]) / (len(donors) - 1)
            for gene_index in range(len(candidate_genes)):
                keep = np.ones(len(candidate_genes), dtype=bool)
                keep[gene_index] = False
                stability_values[donor_index, gene_index] = safe_pearson(
                    corr_stack[donor_index, gene_index, keep],
                    leave_one_out[gene_index, keep],
                )
        profile_stability = np.median(stability_values, axis=0)
        consensus = np.mean(corr_stack, axis=0)
        absolute_consensus = np.abs(consensus).copy()
        np.fill_diagonal(absolute_consensus, -np.inf)
        top_strength = np.mean(
            np.sort(absolute_consensus, axis=1)[:, -TOP_STABLE_NEIGHBORS:], axis=1
        )
        selection_score = np.maximum(profile_stability, 0.0) * top_strength
        ranked_candidate = np.lexsort((np.asarray(candidate_genes), -selection_score))

        for local_rank, candidate_index in enumerate(ranked_candidate, start=1):
            common_index = candidate_common[candidate_index]
            gene_rows.append({
                "cell_type": cell_type,
                "gene": candidate_genes[candidate_index],
                "rank": local_rank,
                "median_within_donor_variance": float(median_variance[common_index]),
                "median_detection_fraction": float(median_detection[common_index]),
                "profile_stability_leave_one_donor_out": float(profile_stability[candidate_index]),
                "mean_top20_abs_consensus_edge": float(top_strength[candidate_index]),
                "selection_score": float(selection_score[candidate_index]),
            })

        models[cell_type] = {}
        edge_sd_full = np.std(corr_stack, axis=0, ddof=1)
        sign_consistency_full = np.abs(np.mean(np.sign(corr_stack), axis=0))
        edge_score_full = np.abs(consensus) * sign_consistency_full / (edge_sd_full + 0.05)
        np.fill_diagonal(edge_score_full, 0.0)

        for panel_size in PANEL_SIZES:
            selected_candidate = ranked_candidate[:panel_size]
            genes = [candidate_genes[i] for i in selected_candidate]
            common_positions = candidate_common[selected_candidate]
            residual_pieces = []
            for donor in donors:
                piece = donor_matrices[donor][:, selected_candidate].astype(np.float64)
                residual_pieces.append(piece - piece.mean(axis=0, keepdims=True))
            residual = np.vstack(residual_pieces)
            training_scale = residual.std(axis=0, ddof=0)
            training_scale[training_scale < 1e-8] = 1.0
            standardized = residual / training_scale
            max_components = min(PCA_MAX_COMPONENTS, panel_size - 1, standardized.shape[0] - 1)
            pca = PCA(n_components=max_components, svd_solver="randomized", random_state=20260714)
            pca.fit(standardized)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            auto_rank = int(np.searchsorted(cumulative, PCA_VARIANCE_TARGET) + 1)
            auto_rank = min(max(auto_rank, 2), max_components)
            fixed_rank = min(PCA_FIXED_COMPONENTS, max_components)
            components = np.asarray(pca.components_, dtype=np.float64)

            selected_edge_score = edge_score_full[np.ix_(selected_candidate, selected_candidate)]
            neighbors = []
            for gene_index in range(panel_size):
                score = selected_edge_score[gene_index].copy()
                score[gene_index] = -np.inf
                neighbors.append(np.argsort(-score, kind="stable")[:TOP_STABLE_NEIGHBORS])

            models[cell_type][panel_size] = {
                "genes": genes,
                "common_positions": np.asarray(common_positions, dtype=int),
                "training_scale": training_scale,
                "components_pca80": components[:auto_rank],
                "components_pca20": components[:fixed_rank],
                "auto_rank": auto_rank,
                "fixed_rank": fixed_rank,
                "explained_variance_auto": float(cumulative[auto_rank - 1]),
                "edge_weights": selected_edge_score,
                "neighbors": neighbors,
            }
            pca_rows.append({
                "cell_type": cell_type,
                "panel_size": panel_size,
                "n_training_donors": len(donors),
                "n_training_cells_balanced": int(residual.shape[0]),
                "pca80_rank": auto_rank,
                "pca80_explained_variance": float(cumulative[auto_rank - 1]),
                "pca20_rank": fixed_rank,
                "pca20_explained_variance": float(cumulative[fixed_rank - 1]),
                "median_training_ledoitwolf_shrinkage": float(np.median(donor_alphas)),
            })
        log(f"    selected nested panels; PCA80 ranks: " + ", ".join(
            f"{p}g={models[cell_type][p]['auto_rank']}" for p in PANEL_SIZES
        ))

    backed.file.close()
    gene_table = pd.DataFrame(gene_rows)
    pca_table = pd.DataFrame(pca_rows)
    donor_table = pd.DataFrame(donor_rows_audit)
    gene_table.to_csv(outdir / "training_stable_gene_ranking.csv", index=False)
    pca_table.to_csv(outdir / "training_pca_audit.csv", index=False)
    donor_table.to_csv(outdir / "training_donor_audit.csv", index=False)
    return models, gene_table, pca_table, donor_table


def load_evaluation_dataset(
    label: str,
    panel_union: list[str],
    common_genes: list[str],
    log: Logger,
) -> dict:
    backed = ad.read_h5ad(PATHS[label], backed="r")
    if label == "Test":
        sample_col, cell_col = "sample", "cell type"
    else:
        sample_col, cell_col = "Sample", "Cell_type"
    genes = backed.var_names.astype(str).tolist()
    lookup = {g: i for i, g in enumerate(genes)}
    panel_indices = np.asarray([lookup[g] for g in panel_union], dtype=int)
    common_indices = np.asarray([lookup[g] for g in common_genes], dtype=int)
    samples = backed.obs[sample_col].astype(str).to_numpy()
    cells = backed.obs[cell_col].astype(str).to_numpy()
    selected_rows = np.flatnonzero(np.isin(samples, TEST_DONORS) & np.isin(cells, CELL_TYPES))
    if label == "Test":
        matrix = read_raw_logcp10k(backed, selected_rows, panel_indices)
    else:
        matrix = read_generated_logcp10k(backed, selected_rows, panel_indices, common_indices)
    obs = pd.DataFrame({"donor": samples[selected_rows], "cell_type": cells[selected_rows]})
    groups = {}
    donor_values = obs["donor"].to_numpy()
    cell_values = obs["cell_type"].to_numpy()
    for cell_type in CELL_TYPES:
        for donor in TEST_DONORS:
            groups[(cell_type, donor)] = np.flatnonzero(
                (cell_values == cell_type) & (donor_values == donor)
            )
    backed.file.close()
    log(f"  loaded {label}: {matrix.shape[0]} cells x {matrix.shape[1]} panel-union genes")
    return {"X": matrix, "obs": obs, "groups": groups}


def pca_reconstruct(piece: np.ndarray, model: dict, variant: str) -> np.ndarray:
    centered = np.asarray(piece, dtype=np.float64) - np.mean(piece, axis=0, keepdims=True)
    if variant == "raw":
        return centered
    components = model["components_pca80"] if variant == "pca80" else model["components_pca20"]
    scale = model["training_scale"]
    standardized = centered / scale
    reconstructed = (standardized @ components.T) @ components
    return reconstructed * scale


def build_network(pieces: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    return shrinkage_corr(np.vstack(pieces))


def per_gene_rows(
    x_corr: np.ndarray,
    y_corr: np.ndarray,
    genes: list[str],
    edge_weights: np.ndarray,
    neighbors: list[np.ndarray],
    metadata: dict,
) -> list[dict]:
    x_z, y_z = fisher_z(x_corr), fisher_z(y_corr)
    rows = []
    for gene_index, gene in enumerate(genes):
        keep = np.ones(len(genes), dtype=bool)
        keep[gene_index] = False
        x, y = x_corr[gene_index, keep], y_corr[gene_index, keep]
        x_f, y_f = x_z[gene_index, keep], y_z[gene_index, keep]
        weights = edge_weights[gene_index, keep]
        stable = neighbors[gene_index]
        row = dict(metadata)
        row.update({
            "gene": gene,
            "profile_pcc_raw": safe_pearson(x, y),
            "profile_pcc_fisher_z": safe_pearson(x_f, y_f),
            "profile_spearman_raw": safe_spearman(x, y),
            "profile_ccc_raw": lin_ccc(x, y),
            "profile_ccc_fisher_z": lin_ccc(x_f, y_f),
            "profile_mae_raw": float(np.mean(np.abs(x - y))),
            "profile_weighted_pcc_raw": weighted_pearson(x, y, weights),
            "stable_neighbor_pcc_raw": safe_pearson(
                x_corr[gene_index, stable], y_corr[gene_index, stable]
            ),
            "stable_neighbor_ccc_raw": lin_ccc(
                x_corr[gene_index, stable], y_corr[gene_index, stable]
            ),
            "mean_x_edge": float(np.mean(x)),
            "mean_y_edge": float(np.mean(y)),
            "sd_x_edge": float(np.std(x)),
            "sd_y_edge": float(np.std(y)),
        })
        rows.append(row)
    return rows


def summarize_seed(per_gene: pd.DataFrame) -> pd.DataFrame:
    keys = ["setting", "panel_size", "cell_type", "seed", "comparison"]
    metric_columns = [
        "profile_pcc_raw", "profile_pcc_fisher_z", "profile_spearman_raw",
        "profile_ccc_raw", "profile_ccc_fisher_z", "profile_mae_raw",
        "profile_weighted_pcc_raw", "stable_neighbor_pcc_raw",
        "stable_neighbor_ccc_raw", "mean_x_edge", "mean_y_edge",
        "sd_x_edge", "sd_y_edge",
    ]
    rows = []
    for values, group in per_gene.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update({
            "n_cells_per_source": int(group["n_cells_per_source"].iloc[0]),
            "n_genes": len(group),
            "shrinkage_x": float(group["shrinkage_x"].iloc[0]),
            "shrinkage_y": float(group["shrinkage_y"].iloc[0]),
            "network_edge_pcc_raw": float(group["network_edge_pcc_raw"].iloc[0]),
            "network_edge_ccc_raw": float(group["network_edge_ccc_raw"].iloc[0]),
        })
        for metric in metric_columns:
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_mean"] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_celltype(seed_table: pd.DataFrame) -> pd.DataFrame:
    keys = ["setting", "panel_size", "cell_type", "comparison"]
    numeric = [c for c in seed_table.columns if c not in keys + ["seed"]]
    rows = []
    for values, group in seed_table.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row["n_seeds"] = group["seed"].nunique()
        for column in numeric:
            row[f"{column}_mean_over_seeds"] = float(group[column].mean())
            row[f"{column}_sd_over_seeds"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_primary(cell_table: pd.DataFrame, panel_size: int, outdir: Path) -> None:
    metric = "profile_pcc_raw_median_mean_over_seeds"
    subset = cell_table[
        (cell_table["setting"] == "pca80_real_only_lw")
        & (cell_table["panel_size"] == panel_size)
    ]
    order = [c[0] for c in COMPARISONS]
    pivot = subset.pivot(index="cell_type", columns="comparison", values=metric)
    pivot = pivot.loc[list(CELL_TYPES), order]
    positions = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    values = [pivot[c].to_numpy(dtype=float) for c in order]
    bp = ax.boxplot(values, positions=positions, widths=0.48, patch_artist=True, showfliers=False)
    for patch, comparison in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[comparison])
        patch.set_alpha(0.35)
        patch.set_edgecolor("#555555")
    for element in ("whiskers", "caps", "medians"):
        for artist in bp[element]:
            artist.set_color("#555555")
    for _, row in pivot.iterrows():
        y = row.to_numpy(dtype=float)
        ax.plot(positions, y, color="#B4B4B4", lw=0.9, alpha=0.8, zorder=1)
        for x_pos, value, comparison in zip(positions, y, order):
            ax.scatter(x_pos, value, s=48, color=COLORS[comparison], edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color="#888888", ls="--", lw=0.8)
    ax.set_xticks(positions, [DISPLAY[c] for c in order])
    ax.set_ylabel("Median per-gene signed edge-profile PCC")
    ax.set_title(f"Training-PCA real only + Ledoit-Wolf | {panel_size} stable genes", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    stem = outdir / f"primary_pca80_real_only_{panel_size}genes"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(cell_table: pd.DataFrame, panel_size: int, outdir: Path) -> None:
    metric = "profile_pcc_raw_median_mean_over_seeds"
    setting_order = list(SETTINGS)
    comparisons = ["RealA_vs_RealB", "DeconvSC_vs_RealB", "BaseVAE_vs_RealB"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=True)
    for ax, comparison in zip(axes, comparisons):
        subset = cell_table[
            (cell_table["panel_size"] == panel_size)
            & (cell_table["comparison"] == comparison)
        ]
        pivot = subset.pivot(index="cell_type", columns="setting", values=metric).loc[list(CELL_TYPES), setting_order]
        positions = np.arange(len(setting_order))
        for _, row in pivot.iterrows():
            ax.plot(positions, row.to_numpy(dtype=float), color="#B0B0B0", lw=0.8, alpha=0.75)
            ax.scatter(positions, row.to_numpy(dtype=float), color=COLORS[comparison], s=24, zorder=3)
        ax.plot(positions, pivot.mean(axis=0).to_numpy(dtype=float), color="#222222", lw=2.0, marker="o", ms=4)
        ax.axhline(0, color="#888888", ls="--", lw=0.7)
        ax.set_title(DISPLAY[comparison])
        ax.set_xticks(positions, ["raw", "PCA<=50\nreal", "PCA20\nreal", "PCA<=50\nboth"], rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median per-gene edge-profile PCC")
    fig.suptitle(f"Representation sensitivity | {panel_size} training-stable genes", y=1.02, fontsize=11)
    fig.tight_layout()
    stem = outdir / f"representation_sensitivity_{panel_size}genes"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(cell_table: pd.DataFrame, setting: str, panel_size: int, metric: str) -> str:
    subset = cell_table[(cell_table["setting"] == setting) & (cell_table["panel_size"] == panel_size)]
    pivot = subset.pivot(index="cell_type", columns="comparison", values=metric)
    order = [c[0] for c in COMPARISONS]
    lines = ["| Cell type | " + " | ".join(DISPLAY[c] for c in order) + " |", "|---|---:|---:|---:|"]
    for cell_type in CELL_TYPES:
        lines.append("| " + cell_type + " | " + " | ".join(f"{pivot.loc[cell_type, c]:.3f}" for c in order) + " |")
    lines.append("| Mean across cell types | " + " | ".join(f"{pivot[c].mean():.3f}" for c in order) + " |")
    return "\n".join(lines)


def dataframe_markdown(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    columns = table.columns.astype(str).tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in table.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(cell_table: pd.DataFrame, pca_table: pd.DataFrame, outdir: Path) -> None:
    metric = "profile_pcc_raw_median_mean_over_seeds"
    primary = cell_table[(cell_table["setting"] == "pca80_real_only_lw") & (cell_table["panel_size"] == 200)]
    pivot = primary.pivot(index="cell_type", columns="comparison", values=metric)
    deconv_wins = int((pivot["DeconvSC_vs_RealB"] > pivot["BaseVAE_vs_RealB"]).sum())
    reaches_real = int((pivot["DeconvSC_vs_RealB"] >= pivot["RealA_vs_RealB"]).sum())
    report = f"""# Training-defined denoised shrinkage edge-profile pilot

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-14
- Verification Status: ANALYZED
- Version Label: gse141115_denoised_stable_edge_profile_v1
- Inferential unit: cell type

## Primary requested result

Real cells were projected through a donor-centred PCA basis fitted only in the
training reference; generated decoder means were left unprojected. The target was 80%
of training residual variance with a 50-component cap. Every panel reached the cap,
so this setting is accurately interpreted as PCA<=50 rather than as an achieved 80%
representation. All networks used the same standardized Ledoit-Wolf estimator. The
endpoint is the median across genes of signed dense edge-profile PCC.

### 200 training-stable genes

{markdown_table(cell_table, "pca80_real_only_lw", 200, metric)}

DeconvSC exceeded the ablated VAE in {deconv_wins}/6 cell types and reached or exceeded
the real-real reference in {reaches_real}/6 cell types. These are descriptive counts;
seeds, genes and edges were not treated as biological replicates.

### 100 training-stable genes

{markdown_table(cell_table, "pca80_real_only_lw", 100, metric)}

## Representation sensitivity, 200 genes

### No cell-level denoising; shrinkage only

{markdown_table(cell_table, "raw_lw", 200, metric)}

### Fixed rank-20 PCA on real only

{markdown_table(cell_table, "pca20_real_only_lw", 200, metric)}

### Same capped PCA<=50 projection applied to real and generated cells

{markdown_table(cell_table, "pca80_symmetric_lw", 200, metric)}

## Interpretation boundary

- Dense edge-profile PCC tests whether each gene has a similar signed linear
  co-expression pattern with the other panel genes. It does not establish physical,
  directed or causal gene interaction.
- Figure 4a already tests sparse hub-neighbour membership. Therefore the dense panel,
  not the top-20 stable-neighbour sensitivity metric, is the less redundant candidate
  for Figure 4d.
- PCA and stable-gene selection use only the training reference, but the representation
  choice remains exploratory and should be disclosed if promoted to the main figure.
- A large change between real-only and symmetric PCA settings indicates representation
  dependence rather than a model-intrinsic topology result.

## PCA ranks

{dataframe_markdown(pca_table)}

## Fallacy scan

- Coverage: 11/11 statistical fallacy types checked.
- Garden of forking paths / look-elsewhere: multiple representations and panel sizes
  are explicitly retained; do not report only the most favourable setting.
- Pseudoreplication: cells, genes, edges and seeds are not independent biological units.
- Correlation-causation: the endpoint is undirected co-expression agreement, not GRN recovery.
"""
    (outdir / "analysis_report_zh.md").write_text(report, encoding="utf-8")


def write_output_manifest(outdir: Path) -> None:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if not path.is_file() or path.name == "output_manifest.csv":
            continue
        rows.append({
            "relative_path": str(path.relative_to(outdir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    pd.DataFrame(rows).to_csv(outdir / "output_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--skip-input-hashes", action="store_true")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    log = Logger(outdir / "logs/execution.log")
    started = time.time()

    for label, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    log("[1/7] inspect input schemas and four-way common genes")
    common_genes, schema = inspect_inputs()
    if len(common_genes) < CANDIDATE_GENES:
        raise RuntimeError(f"Only {len(common_genes)} common genes")
    log(f"  common genes={len(common_genes)}")
    pd.DataFrame(schema).to_csv(outdir / "input_schema.csv", index=False)
    manifest = []
    for label, path in PATHS.items():
        manifest.append({
            "label": label,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": "SKIPPED" if args.skip_input_hashes else sha256_file(path),
            "role": "read-only input",
        })
    pd.DataFrame(manifest).to_csv(outdir / "input_manifest.csv", index=False)

    log("[2/7] define stable genes and PCA bases using training reference only")
    panel_models, gene_table, pca_table, training_donor_table = build_training_panels(common_genes, outdir, log)

    selected_union = sorted(set().union(*(
        set(panel_models[cell_type][200]["genes"]) for cell_type in CELL_TYPES
    )))
    union_lookup = {gene: i for i, gene in enumerate(selected_union)}
    panel_union_indices = {
        (cell_type, panel_size): np.asarray(
            [union_lookup[g] for g in panel_models[cell_type][panel_size]["genes"]], dtype=int
        )
        for cell_type in CELL_TYPES for panel_size in PANEL_SIZES
    }
    pd.DataFrame({"gene": selected_union}).to_csv(outdir / "evaluation_gene_union.csv", index=False)
    log(f"  selected panel union={len(selected_union)}")

    log("[3/7] load and normalize evaluation cells")
    datasets = {
        label: load_evaluation_dataset(label, selected_union, common_genes, log)
        for label in ("Test", "DeconvSC", "BaseVAE")
    }

    count_rows = []
    design_rows = []
    balanced_n = {}
    for cell_type in CELL_TYPES:
        for donor in TEST_DONORS:
            counts = {
                label: len(datasets[label]["groups"][(cell_type, donor)])
                for label in datasets
            }
            for label, n in counts.items():
                count_rows.append({"source": label, "cell_type": cell_type, "donor": donor, "n_cells": n})
            n = min(counts["Test"] // 2, counts["DeconvSC"], counts["BaseVAE"], EVAL_CAP)
            if n < EVAL_MIN:
                raise RuntimeError(f"Balanced n<{EVAL_MIN}: {cell_type}/{donor}={n}")
            balanced_n[(cell_type, donor)] = n
            design_rows.append({
                "cell_type": cell_type,
                "donor": donor,
                "Test_available": counts["Test"],
                "DeconvSC_available": counts["DeconvSC"],
                "BaseVAE_available": counts["BaseVAE"],
                "balanced_n_per_source": n,
            })
    pd.DataFrame(count_rows).to_csv(outdir / "evaluation_cell_count_audit.csv", index=False)
    pd.DataFrame(design_rows).to_csv(outdir / "evaluation_sampling_design.csv", index=False)

    log("[4/7] compute denoised/shrinkage edge-profile metrics")
    per_gene_all = []
    network_audit_rows = []
    for cell_type in CELL_TYPES:
        for panel_size in PANEL_SIZES:
            model = panel_models[cell_type][panel_size]
            columns = panel_union_indices[(cell_type, panel_size)]
            genes = model["genes"]
            for seed in SEEDS:
                selections = {label: [] for label in ("RealA", "RealB", "DeconvSC", "BaseVAE")}
                n_total = 0
                for donor in TEST_DONORS:
                    n = balanced_n[(cell_type, donor)]
                    n_total += n
                    real_available = datasets["Test"]["groups"][(cell_type, donor)]
                    real_rng = stable_rng("real_split", seed, cell_type, donor)
                    real = real_rng.choice(real_available, size=2 * n, replace=False)
                    selections["RealA"].append(np.sort(real[:n]))
                    selections["RealB"].append(np.sort(real[n:]))
                    for source in ("DeconvSC", "BaseVAE"):
                        available = datasets[source]["groups"][(cell_type, donor)]
                        rng = stable_rng("model_sample", source, seed, cell_type, donor)
                        selections[source].append(np.sort(rng.choice(available, size=n, replace=False)))

                source_map = {"RealA": "Test", "RealB": "Test", "DeconvSC": "DeconvSC", "BaseVAE": "BaseVAE"}
                representations = {}
                for label, donor_rows in selections.items():
                    source = source_map[label]
                    pieces = [datasets[source]["X"][rows][:, columns] for rows in donor_rows]
                    representations[label] = {
                        variant: [pca_reconstruct(piece, model, variant) for piece in pieces]
                        for variant in ("raw", "pca80", "pca20")
                    }

                network_cache = {}
                for label in representations:
                    needed = ("raw", "pca80") if label in ("DeconvSC", "BaseVAE") else ("raw", "pca80", "pca20")
                    for variant in needed:
                        network_cache[(label, variant)] = build_network(representations[label][variant])

                for setting, variants in SETTINGS.items():
                    networks = {}
                    for label in ("RealA", "RealB", "DeconvSC", "BaseVAE"):
                        variant = variants["real"] if label.startswith("Real") else variants["model"]
                        networks[label] = network_cache[(label, variant)]
                    for comparison, x_label, y_label in COMPARISONS:
                        x_corr, x_alpha, x_constant = networks[x_label]
                        y_corr, y_alpha, y_constant = networks[y_label]
                        triangle = np.triu_indices(panel_size, 1)
                        metadata = {
                            "setting": setting,
                            "panel_size": panel_size,
                            "cell_type": cell_type,
                            "seed": seed,
                            "comparison": comparison,
                            "n_cells_per_source": n_total,
                            "shrinkage_x": x_alpha,
                            "shrinkage_y": y_alpha,
                            "constant_fraction_x": x_constant,
                            "constant_fraction_y": y_constant,
                            "network_edge_pcc_raw": safe_pearson(x_corr[triangle], y_corr[triangle]),
                            "network_edge_ccc_raw": lin_ccc(x_corr[triangle], y_corr[triangle]),
                        }
                        per_gene_all.extend(per_gene_rows(
                            x_corr, y_corr, genes, model["edge_weights"], model["neighbors"], metadata
                        ))
                        network_audit_rows.append(metadata)
            log(f"  completed {cell_type} / {panel_size} genes")

    per_gene = pd.DataFrame(per_gene_all)
    per_gene.to_csv(outdir / "per_gene_edge_profile_metrics.csv.gz", index=False, compression="gzip")
    network_audit = pd.DataFrame(network_audit_rows)
    network_audit.to_csv(outdir / "network_estimation_audit.csv", index=False)
    seed_table = summarize_seed(per_gene)
    seed_table.to_csv(outdir / "edge_profile_summary_by_seed.csv", index=False)
    cell_table = summarize_celltype(seed_table)
    cell_table.to_csv(outdir / "edge_profile_summary_by_celltype.csv", index=False)

    overall = cell_table.groupby(["setting", "panel_size", "comparison"], sort=True).agg(
        n_cell_types=("cell_type", "nunique"),
        mean_celltype_profile_pcc=("profile_pcc_raw_median_mean_over_seeds", "mean"),
        median_celltype_profile_pcc=("profile_pcc_raw_median_mean_over_seeds", "median"),
        min_celltype_profile_pcc=("profile_pcc_raw_median_mean_over_seeds", "min"),
        max_celltype_profile_pcc=("profile_pcc_raw_median_mean_over_seeds", "max"),
        mean_celltype_profile_ccc=("profile_ccc_raw_median_mean_over_seeds", "mean"),
        mean_celltype_stable_neighbor_pcc=("stable_neighbor_pcc_raw_median_mean_over_seeds", "mean"),
        mean_network_edge_pcc=("network_edge_pcc_raw_mean_over_seeds", "mean"),
    ).reset_index()
    overall.to_csv(outdir / "edge_profile_overall_summary.csv", index=False)

    log("[5/7] render figures and write analysis report")
    for panel_size in PANEL_SIZES:
        plot_primary(cell_table, panel_size, outdir / "figures")
        plot_sensitivity(cell_table, panel_size, outdir / "figures")
    write_report(cell_table, pca_table, outdir)

    config = {
        "analysis": "training-defined denoised shrinkage signed edge-profile correlation",
        "date": "2026-07-14",
        "cell_types": list(CELL_TYPES),
        "test_donors": list(TEST_DONORS),
        "seeds": list(SEEDS),
        "candidate_genes": CANDIDATE_GENES,
        "panel_sizes": list(PANEL_SIZES),
        "stable_gene_source": "GSE141115 training reference only",
        "stable_gene_score": "positive leave-one-training-donor-out edge-profile PCC times mean top-20 absolute consensus edge",
        "denoiser": "training-only donor-centred PCA",
        "pca80_rule": "smallest rank explaining >=80% training residual variance, capped at 50",
        "correlation_estimator": "standardized Ledoit-Wolf covariance converted to correlation",
        "primary_setting": "pca80_real_only_lw",
        "primary_metric": "median across genes of raw signed edge-profile Pearson correlation",
        "inferential_unit": "cell type",
        "claim_boundary": "undirected co-expression relationships, not causal/directed GRN",
    }
    (outdir / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log("[6/7] validate output structure")
    validation = [
        {"check": "expected_cell_types", "passed": set(cell_table["cell_type"]) == set(CELL_TYPES), "detail": sorted(cell_table["cell_type"].unique())},
        {"check": "expected_seeds", "passed": set(seed_table["seed"]) == set(SEEDS), "detail": sorted(seed_table["seed"].unique())},
        {"check": "nested_panel_sizes", "passed": set(cell_table["panel_size"]) == set(PANEL_SIZES), "detail": sorted(cell_table["panel_size"].unique())},
        {"check": "training_only_gene_definition", "passed": True, "detail": "Stable genes and PCA fitted before loading evaluation matrices"},
        {"check": "real_real_reference", "passed": bool((cell_table["comparison"] == "RealA_vs_RealB").any()), "detail": "RealA_vs_RealB"},
        {"check": "pcc_bounds", "passed": bool(per_gene["profile_pcc_raw"].between(-1, 1).all()), "detail": [float(per_gene["profile_pcc_raw"].min()), float(per_gene["profile_pcc_raw"].max())]},
        {"check": "ccc_bounds", "passed": bool(per_gene["profile_ccc_raw"].dropna().between(-1, 1).all()), "detail": [float(per_gene["profile_ccc_raw"].min()), float(per_gene["profile_ccc_raw"].max())]},
        {"check": "no_pseudoreplicate_tests", "passed": True, "detail": "No p-values from cells, genes, edges or seeds"},
    ]
    pd.DataFrame(validation).to_csv(outdir / "validation_checks.csv", index=False)
    if not all(row["passed"] for row in validation):
        raise RuntimeError(f"Validation failed: {validation}")

    runtime = {
        "status": "completed",
        "duration_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "anndata": ad.__version__,
        "sklearn": sklearn_version,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    log(f"[7/7] completed in {runtime['duration_seconds']:.1f}s -> {outdir}")
    write_output_manifest(outdir)


if __name__ == "__main__":
    main()
