#!/usr/bin/env python3
"""Execute the pre-specified GSE141115 within-cell-type topology benchmark.

All source h5ad/CSV files are read-only.  Outputs are written only below the
requested output directory.  The inferential unit is cell type: modules and
Monte-Carlo seeds are averaged before paired statistical testing.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
from numba import njit
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, wilcoxon


warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

RELEASE_ROOT = Path(__file__).resolve().parents[4]
INTER = RELEASE_ROOT / "benchmark_tables/expression_prediction/figure4_support/GSE141115"

PATHS = {
    "gt_raw": RELEASE_ROOT / "processed_data/real_data/mouse_kidney/test_data.h5ad",
    "deconv_fig3": RELEASE_ROOT / "model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad",
    "basevae": RELEASE_ROOT / "model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad",
    "deconv_current": RELEASE_ROOT / "model_outputs/GSE141115/prophead_generated.h5ad",
    "figure3_predicted_profiles": INTER / "pb_predicted_log1p_cp10k.csv.gz",
    "legacy_within_script": RELEASE_ROOT / "code/03_model_evaluation/support/within_type_module_l2.py",
    "deconv_training_script": RELEASE_ROOT / "code/02_model_training/prophead_GSE141115_architecture_reference.py",
}

METHOD_INFO = {
    "GT": ("gt_raw", "sample", "cell type", "raw_counts"),
    "DeconvSC": ("deconv_fig3", "Sample", "Cell_type", "decoder_log_output"),
    "BaseVAE": ("basevae", "Sample", "Cell_type", "decoder_log_output"),
    "DeconvSC_current": ("deconv_current", "Sample", "Cell_type", "decoder_log_output"),
}

DONORS = ("LDK1", "LDK2", "LDK3")
SEEDS = tuple(range(2024, 2034))
TOPK = 500
MI_TOPK = 200
N_HUBS = 50
K_NEIGHBORS = 20
N_MODULES = 20
TOP_MODULES = 10
MIN_MODULE_SIZE = 6
PER_DONOR_MIN = 20
PER_DONOR_CAP = 200
PRIMARY_EXPECTED_TYPES = {"CD_PC", "CNT", "DCT", "MC", "PT", "aLOH"}
METHOD_COLORS = {"DeconvSC": "#D8B2AE", "BaseVAE": "#7187A7"}


def log(message: str) -> None:
    print(message, flush=True)


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
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def cellwise_log1p_cp10k(x) -> np.ndarray:
    """Put nonnegative decoder log-values onto a common per-cell CP10K scale."""
    source = dense(x)
    counts = np.expm1(np.clip(source, 0.0, None))
    library = counts.sum(axis=1, keepdims=True)
    library[library <= 0.0] = 1.0
    return np.asarray(np.log1p(counts / library * 1e4), dtype=np.float32)


def safe_vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def corr_matrix(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    constant_fraction = float(np.mean(np.std(x, axis=0) < 1e-12))
    r = np.corrcoef(x, rowvar=False)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.clip((r + r.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(r, 1.0)
    return r, constant_fraction


def fisher_average_corr(matrices: list[np.ndarray]) -> tuple[np.ndarray, float]:
    corrs = []
    constants = []
    for matrix in matrices:
        r, fraction = corr_matrix(matrix)
        corrs.append(r)
        constants.append(fraction)
    z = np.stack([np.arctanh(np.clip(r, -0.999999, 0.999999)) for r in corrs])
    out = np.tanh(np.mean(z, axis=0))
    np.fill_diagonal(out, 1.0)
    return out, float(np.mean(constants))


def stable_rng(seed: int, cell_type: str, donor: str, method: str, source: str) -> np.random.Generator:
    token = f"{seed}|{cell_type}|{donor}|{method}|{source}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(value)


def holm_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_median_ci(values: np.ndarray, seed: int = 20260710, n_boot: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def rank_biserial(advantage: np.ndarray) -> float:
    advantage = np.asarray(advantage, dtype=np.float64)
    advantage = advantage[np.abs(advantage) > 1e-15]
    if len(advantage) == 0:
        return 0.0
    ranks = rankdata(np.abs(advantage))
    return float((ranks[advantage > 0].sum() - ranks[advantage < 0].sum()) / ranks.sum())


def exact_signflip_p(advantage: np.ndarray) -> float:
    advantage = np.asarray(advantage, dtype=np.float64)
    observed = float(np.mean(advantage))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(advantage)):
        permuted.append(float(np.mean(advantage * np.asarray(signs))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


@njit(cache=True)
def mutual_information_all_pairs(codes: np.ndarray, nbins: int) -> np.ndarray:
    n, p = codes.shape
    out = np.empty(p * (p - 1) // 2, dtype=np.float64)
    position = 0
    for i in range(p - 1):
        for j in range(i + 1, p):
            joint = np.zeros((nbins, nbins), dtype=np.float64)
            for row in range(n):
                joint[codes[row, i], codes[row, j]] += 1.0
            px = np.zeros(nbins, dtype=np.float64)
            py = np.zeros(nbins, dtype=np.float64)
            for a in range(nbins):
                for b in range(nbins):
                    px[a] += joint[a, b]
                    py[b] += joint[a, b]
            value = 0.0
            for a in range(nbins):
                for b in range(nbins):
                    count = joint[a, b]
                    if count > 0.0 and px[a] > 0.0 and py[b] > 0.0:
                        probability = count / n
                        value += probability * math.log((count * n) / (px[a] * py[b]))
            out[position] = value
            position += 1
    return out


def equal_width_codes(x: np.ndarray, nbins: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    codes = np.zeros(x.shape, dtype=np.int16)
    for column in range(x.shape[1]):
        values = x[:, column]
        low, high = float(np.min(values)), float(np.max(values))
        if high - low > 1e-12:
            cuts = np.linspace(low, high, nbins + 1)[1:-1]
            codes[:, column] = np.clip(np.digitize(values, cuts), 0, nbins - 1)
    return codes


def rank_quantile_codes(x: np.ndarray, nbins: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    codes = np.zeros(x.shape, dtype=np.int16)
    for column in range(x.shape[1]):
        ranks = rankdata(x[:, column], method="average")
        codes[:, column] = np.minimum(nbins - 1, ((ranks - 1.0) * nbins / max(n, 1)).astype(np.int16))
    return codes


def build_modules(correlation: np.ndarray, method: str) -> list[dict]:
    distance = np.clip(1.0 - np.abs(correlation), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method=method)
    labels = fcluster(tree, t=min(N_MODULES, len(correlation)), criterion="maxclust")
    candidates = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < MIN_MODULE_SIZE:
            continue
        block = np.abs(correlation[np.ix_(indices, indices)])
        tri = block[np.triu_indices(len(indices), 1)]
        cohesion = float(np.mean(tri)) if len(tri) else 0.0
        candidates.append({"indices": indices, "cohesion": cohesion})
    candidates.sort(key=lambda row: -row["cohesion"])
    for module_id, row in enumerate(candidates[:TOP_MODULES], start=1):
        row["module_id"] = module_id
    return candidates[:TOP_MODULES]


def hub_indices(correlation: np.ndarray) -> np.ndarray:
    degree = np.sum(np.abs(correlation), axis=1) - 1.0
    return np.argsort(-degree, kind="stable")[: min(N_HUBS, len(degree))]


def neighbors(correlation: np.ndarray, gene_index: int) -> set[int]:
    scores = np.abs(correlation[gene_index]).copy()
    scores[gene_index] = -np.inf
    order = np.argsort(-scores, kind="stable")[: min(K_NEIGHBORS, len(scores) - 1)]
    return set(order.tolist())


def within_donor_variance(dataset: dict, cell_type: str) -> np.ndarray:
    x = dataset["X_cp10k"]
    obs = dataset["obs"]
    sse = np.zeros(x.shape[1], dtype=np.float64)
    degrees = 0
    for donor in DONORS:
        indices = np.flatnonzero((obs["cell_type"].values == cell_type) & (obs["donor"].values == donor))
        if len(indices) < 2:
            continue
        group = x[indices]
        if sp.issparse(group):
            sums = np.asarray(group.sum(axis=0)).ravel()
            squares = np.asarray(group.multiply(group).sum(axis=0)).ravel()
        else:
            group = np.asarray(group, dtype=np.float64)
            sums = group.sum(axis=0)
            squares = np.square(group).sum(axis=0)
        sse += np.maximum(0.0, squares - sums * sums / len(indices))
        degrees += len(indices) - 1
    if degrees <= 0:
        raise RuntimeError(f"No within-donor variance degrees of freedom for {cell_type}")
    return sse / degrees


def select_top_genes(variance: np.ndarray, common_genes: np.ndarray, topk: int = TOPK) -> np.ndarray:
    order = np.argsort(-np.nan_to_num(variance, nan=-np.inf), kind="stable")
    order = order[np.isfinite(variance[order]) & (variance[order] > 1e-12)]
    if len(order) < topk:
        raise RuntimeError(f"Only {len(order)} nonconstant genes; expected at least {topk}")
    return order[:topk]


def extract_matrix(
    dataset: dict,
    row_indices: np.ndarray,
    gene_indices: np.ndarray,
    expression_scale: str = "cellwise_log1p_cp10k",
) -> np.ndarray:
    key = "X_cp10k" if expression_scale == "cellwise_log1p_cp10k" else "X_native"
    matrix = dataset[key][row_indices][:, gene_indices]
    return dense(matrix)


def full_gt_reference(gt: dict, cell_type: str, gene_indices: np.ndarray) -> np.ndarray:
    obs = gt["obs"]
    pieces = []
    for donor in DONORS:
        rows = np.flatnonzero((obs["cell_type"].values == cell_type) & (obs["donor"].values == donor))
        if len(rows) < 2:
            continue
        matrix = extract_matrix(gt, rows, gene_indices, "cellwise_log1p_cp10k")
        matrix -= matrix.mean(axis=0, keepdims=True)
        pieces.append(matrix)
    correlation, _ = corr_matrix(np.vstack(pieces))
    return correlation


def load_inputs() -> tuple[dict[str, dict], np.ndarray, list[dict]]:
    log("[Gate 0] inspect h5ad schemas and common genes")
    gene_sets = []
    schema_rows = []
    for method, (path_key, sample_col, cell_col, scale) in METHOD_INFO.items():
        path = PATHS[path_key]
        backed = ad.read_h5ad(path, backed="r")
        genes = backed.var_names.astype(str).to_numpy()
        if len(set(genes)) != len(genes):
            raise RuntimeError(f"Duplicate var_names in {method}: {path}")
        gene_sets.append(set(genes))
        schema_rows.append({
            "method": method,
            "path": str(path),
            "n_obs_all": backed.n_obs,
            "n_vars_all": backed.n_vars,
            "sample_col": sample_col,
            "cell_type_col": cell_col,
            "declared_scale": scale,
            "obs_columns": ";".join(backed.obs.columns.astype(str)),
        })
        backed.file.close()
    common_genes = np.asarray(sorted(set.intersection(*gene_sets)), dtype=str)
    if len(common_genes) < TOPK:
        raise RuntimeError(f"Only {len(common_genes)} common genes")

    datasets: dict[str, dict] = {}
    for method, (path_key, sample_col, cell_col, scale) in METHOD_INFO.items():
        path = PATHS[path_key]
        log(f"  load {method}: {path}")
        full = ad.read_h5ad(path)
        sample_values = full.obs[sample_col].astype(str)
        selected = full[sample_values.isin(DONORS)].copy()
        if method == "GT":
            sc.pp.normalize_total(selected, target_sum=1e4)
            sc.pp.log1p(selected)
        selected = selected[:, common_genes].copy()
        if not sp.issparse(selected.X):
            selected.X = np.asarray(selected.X, dtype=np.float32)
        obs = pd.DataFrame({
            "donor": selected.obs[sample_col].astype(str).to_numpy(),
            "cell_type": selected.obs[cell_col].astype(str).to_numpy(),
        })
        sample_rows = np.linspace(0, selected.n_obs - 1, min(500, selected.n_obs), dtype=int)
        native_x = selected.X
        cp10k_x = native_x if method == "GT" else cellwise_log1p_cp10k(native_x)
        sample_matrix = dense(native_x[sample_rows])
        sample_cp10k = dense(cp10k_x[sample_rows])
        exp_library = np.expm1(np.clip(sample_matrix, 0, None)).sum(axis=1)
        cp10k_library = np.expm1(np.clip(sample_cp10k, 0, None)).sum(axis=1)
        schema = next(row for row in schema_rows if row["method"] == method)
        schema.update({
            "n_obs_ldk1_3": selected.n_obs,
            "n_common_genes": selected.n_vars,
            "x_sample_min": float(np.min(sample_matrix)),
            "x_sample_max": float(np.max(sample_matrix)),
            "x_sample_zero_fraction": float(np.mean(sample_matrix == 0)),
            "expm1_library_median_common_panel": float(np.median(exp_library)),
            "post_transform_expm1_library_median": float(np.median(cp10k_library)),
        })
        datasets[method] = {
            "X_native": native_x,
            "X_cp10k": cp10k_x,
            "obs": obs,
            "path": path,
            "native_scale": "raw-normalized log1p CP10K" if method == "GT" else "decoder log output; library not conserved",
        }
        del full, selected, sample_matrix, sample_cp10k
    return datasets, common_genes, schema_rows


def figure3_provenance_check() -> pd.DataFrame:
    expected = pd.read_csv(PATHS["figure3_predicted_profiles"], index_col=[0, 1])
    expected.index = pd.MultiIndex.from_tuples(
        [(str(a), str(b)) for a, b in expected.index], names=["sample", "cell_type"]
    )
    backed = ad.read_h5ad(PATHS["deconv_fig3"], backed="r")
    genes = backed.var_names.astype(str).tolist()
    if genes != expected.columns.astype(str).tolist():
        raise RuntimeError("Figure 3 intermediate gene order differs from source h5ad")
    samples = backed.obs["Sample"].astype(str).to_numpy()
    cell_types = backed.obs["Cell_type"].astype(str).to_numpy()
    max_difference = 0.0
    profile_differences = []
    for sample, cell_type in expected.index:
        indices = np.flatnonzero((samples == sample) & (cell_types == cell_type))
        if len(indices) == 0:
            raise RuntimeError(f"Missing Figure 3 source cells: {sample}/{cell_type}")
        values = dense(backed.X[indices])
        counts = np.expm1(np.clip(values, 0, None)).sum(axis=0)
        log_profile = np.log1p(counts / max(counts.sum(), 1.0) * 1e4)
        difference = float(np.max(np.abs(log_profile - expected.loc[(sample, cell_type)].to_numpy(dtype=float))))
        max_difference = max(max_difference, difference)
        profile_differences.append(difference)
    backed.file.close()
    return pd.DataFrame([{
        "check": "figure3_deconv_h5ad_to_verified_profile",
        "n_profiles": len(expected),
        "n_genes": expected.shape[1],
        "max_abs_difference": max_difference,
        "median_profile_max_abs_difference": float(np.median(profile_differences)),
        "tolerance": 1e-5,
        "passed": bool(max_difference < 1e-5),
        "source_h5ad": str(PATHS["deconv_fig3"]),
        "reference_profiles": str(PATHS["figure3_predicted_profiles"]),
    }])


def architecture_audit() -> pd.DataFrame:
    rows = [
        ("exact_training_code", "DeconvSC training script exists", "No exact GSE141115 BaseVAE training script/config found beside output", "UNVERIFIED", "Model comparison allowed; attention-only attribution not allowed"),
        ("train_test_split", "Not embedded in h5ad", "Not embedded in h5ad", "UNVERIFIED", "Cannot prove identical training split from output files"),
        ("losses_and_weights", "Training script includes reconstruction/KL/correlation losses", "Exact BaseVAE losses/weights unavailable", "UNVERIFIED", "Correlation loss and other differences cannot be excluded"),
        ("latent_and_capacity", "Architecture visible in DeconvSC script", "BaseVAE architecture metadata unavailable", "UNVERIFIED", "Capacity matching cannot be verified"),
        ("generation_cell_count", "9,934 cells across LDK1-6", "19,936 cells across LDK1-6", "NOT_EQUIVALENT", "Analysis must match cells by donor/type"),
        ("gene_output_dimension", "16,801 genes", "16,792 genes", "NOT_EQUIVALENT", "Use fixed four-way common gene intersection"),
        ("native_output_scale", "Decoder log output; median expm1 library is not 10,000", "Decoder log output; median expm1 library is not 10,000", "NOT_CP10K", "Primary analysis re-normalizes expm1 values per cell; native scale retained as S7"),
    ]
    return pd.DataFrame(rows, columns=["item", "deconvsc_evidence", "basevae_evidence", "status", "implication"])


def build_counts(datasets: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    group_indices: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    for method, dataset in datasets.items():
        obs = dataset["obs"]
        group_indices[method] = {}
        for donor in DONORS:
            for cell_type in sorted(obs["cell_type"].unique()):
                indices = np.flatnonzero((obs["donor"].values == donor) & (obs["cell_type"].values == cell_type))
                group_indices[method][(cell_type, donor)] = indices
                rows.append({"method": method, "donor": donor, "cell_type": cell_type, "n_cells": len(indices)})
    return pd.DataFrame(rows), group_indices


def eligible_types(
    counts: pd.DataFrame,
    deconv_method: str,
    threshold: int,
    donors: tuple[str, ...],
) -> tuple[list[str], dict[str, list[str]], list[dict]]:
    methods = ("GT", deconv_method, "BaseVAE")
    types = sorted(set(counts["cell_type"]))
    eligible = []
    valid_donors: dict[str, list[str]] = {}
    audit = []
    for cell_type in types:
        totals = {}
        for method in methods:
            value = counts[
                (counts["method"] == method)
                & (counts["cell_type"] == cell_type)
                & (counts["donor"].isin(donors))
            ]["n_cells"].sum()
            totals[method] = int(value)
        donor_ok = []
        for donor in donors:
            values = []
            for method in methods:
                q = counts[
                    (counts["method"] == method)
                    & (counts["cell_type"] == cell_type)
                    & (counts["donor"] == donor)
                ]["n_cells"]
                values.append(int(q.iloc[0]) if len(q) else 0)
            if min(values) >= PER_DONOR_MIN:
                donor_ok.append(donor)
        passed = min(totals.values()) >= threshold and len(donor_ok) >= 2
        if passed:
            eligible.append(cell_type)
            valid_donors[cell_type] = donor_ok
        audit.append({
            "cell_type": cell_type,
            "deconv_source": deconv_method,
            "threshold": threshold,
            "candidate_donors": ";".join(donors),
            "valid_donors": ";".join(donor_ok),
            "n_valid_donors": len(donor_ok),
            "GT_total": totals["GT"],
            "DeconvSC_total": totals[deconv_method],
            "BaseVAE_total": totals["BaseVAE"],
            "eligible": passed,
        })
    return eligible, valid_donors, audit


def sample_correlations(
    datasets: dict[str, dict],
    group_indices: dict,
    deconv_method: str,
    cell_type: str,
    donors: list[str],
    gene_indices: np.ndarray,
    seed: int,
    mode: str,
    expression_scale: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], int]:
    source = "figure3" if deconv_method == "DeconvSC" else "current"
    logical = {"GT": "GT", "DeconvSC": deconv_method, "BaseVAE": "BaseVAE"}
    pieces: dict[str, list[np.ndarray]] = {method: [] for method in logical}
    total = 0
    for donor in donors:
        sizes = [len(group_indices[actual][(cell_type, donor)]) for actual in logical.values()]
        n = min(*sizes, PER_DONOR_CAP)
        if n < PER_DONOR_MIN:
            raise RuntimeError(f"Unexpected ineligible donor group: {cell_type}/{donor}/{sizes}")
        total += n
        for display, actual in logical.items():
            available = group_indices[actual][(cell_type, donor)]
            rng = stable_rng(seed, cell_type, donor, display, source)
            chosen = np.sort(rng.choice(available, size=n, replace=False))
            matrix = extract_matrix(datasets[actual], chosen, gene_indices, expression_scale)
            if mode in ("pooled_centered", "fisher"):
                matrix = matrix - matrix.mean(axis=0, keepdims=True)
            pieces[display].append(matrix)

    matrices = {method: np.vstack(parts) for method, parts in pieces.items()}
    correlations: dict[str, np.ndarray] = {}
    constants: dict[str, float] = {}
    for method, parts in pieces.items():
        if mode == "fisher":
            correlations[method], constants[method] = fisher_average_corr(parts)
        else:
            correlations[method], constants[method] = corr_matrix(matrices[method])
    return correlations, matrices, constants, total


def paired_test(table: pd.DataFrame, metric: str, higher_better: bool, setting: str) -> dict:
    subset = table[table["setting"] == setting]
    pivot = subset.pivot(index="cell_type", columns="method", values=metric).dropna()
    deconv = pivot["DeconvSC"].to_numpy(dtype=float)
    base = pivot["BaseVAE"].to_numpy(dtype=float)
    advantage = deconv - base if higher_better else base - deconv
    alternative = "greater" if higher_better else "less"
    try:
        one = float(wilcoxon(deconv, base, alternative=alternative, method="exact").pvalue)
        two = float(wilcoxon(deconv, base, alternative="two-sided", method="exact").pvalue)
        method_name = "paired Wilcoxon exact"
    except ValueError:
        one = float(wilcoxon(deconv, base, alternative=alternative, method="auto").pvalue)
        two = float(wilcoxon(deconv, base, alternative="two-sided", method="auto").pvalue)
        method_name = "paired Wilcoxon auto (ties/zeros)"
    low, high = bootstrap_median_ci(advantage)
    return {
        "setting": setting,
        "metric": metric,
        "direction": "higher_better" if higher_better else "lower_better",
        "test": method_name,
        "n_cell_types": len(pivot),
        "cell_types": ";".join(pivot.index.astype(str)),
        "deconvsc_mean": float(np.mean(deconv)),
        "deconvsc_median": float(np.median(deconv)),
        "basevae_mean": float(np.mean(base)),
        "basevae_median": float(np.median(base)),
        "median_advantage_favors_deconvsc": float(np.median(advantage)),
        "mean_advantage_favors_deconvsc": float(np.mean(advantage)),
        "advantage_bootstrap_ci_low": low,
        "advantage_bootstrap_ci_high": high,
        "rank_biserial_favors_deconvsc": rank_biserial(advantage),
        "deconvsc_win_fraction": float(np.mean(advantage > 0)),
        "p_one_sided": one,
        "p_two_sided": two,
        "p_exact_signflip_mean_one_sided": exact_signflip_p(advantage),
    }


def plot_primary(cell_table: pd.DataFrame, outdir: Path) -> None:
    primary = cell_table[cell_table["setting"] == "P0_primary"].copy()
    specs = [
        ("hub_jaccard_mean", "Hub-neighbor Jaccard ↑"),
        ("mi_fidelity_error_median", "MI fidelity error ↓"),
        ("module_l2_mean", "Within-type module L2 ↓"),
        ("per_gene_coexpression_edge_mae_median", "Per-gene co-expression edge MAE ↓"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (metric, label) in zip(axes.ravel(), specs):
        pivot = primary.pivot(index="cell_type", columns="method", values=metric).dropna()
        for cell_type, row in pivot.iterrows():
            ax.plot([0, 1], [row["DeconvSC"], row["BaseVAE"]], color="#B0B0B0", lw=0.8, alpha=0.8)
            ax.scatter(0, row["DeconvSC"], color=METHOD_COLORS["DeconvSC"], s=35, zorder=3)
            ax.scatter(1, row["BaseVAE"], color=METHOD_COLORS["BaseVAE"], s=35, zorder=3)
        ax.set_xticks([0, 1], ["DeconvSC", "BaseVAE"])
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("GSE141115 within-cell-type topology (cell type is the unit)")
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"primary_topology_paired.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pivot = primary.pivot(index="cell_type", columns="method", values="module_l2_mean").dropna()
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    for cell_type, row in pivot.iterrows():
        ax.plot([0, 1], [row["DeconvSC"], row["BaseVAE"]], color="#A8A8A8", lw=1.0)
        ax.scatter([0, 1], [row["DeconvSC"], row["BaseVAE"]],
                   color=[METHOD_COLORS["DeconvSC"], METHOD_COLORS["BaseVAE"]], s=55, zorder=3)
        ax.text(1.04, row["BaseVAE"], str(cell_type), va="center", fontsize=8)
    ax.set_xticks([0, 1], ["DeconvSC", "Ablated VAE"])
    ax.set_ylabel("Within-cell-type module L2 error")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"primary_module_l2_paired.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_figure4_inputs(cell_table: pd.DataFrame, tests: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    primary = cell_table[cell_table["setting"] == "P0_primary"].copy()
    metric_map = {
        "hub_jaccard_mean": ("hub_jaccard", "higher"),
        "mi_fidelity_error_median": ("mi_fidelity_error", "lower"),
        "module_l2_mean": ("module_l2", "lower"),
        "per_gene_coexpression_edge_mae_median": ("coexpression_edge_mae", "lower"),
        "per_gene_network_fidelity_median": ("network_fidelity", "higher"),
        "network_pcc": ("network_pcc", "higher"),
    }
    long_rows = []
    for _, row in primary.iterrows():
        for column, (metric, direction) in metric_map.items():
            long_rows.append({
                "analysis": "within_celltype_P0",
                "unit": "cell_type",
                "cell_type": row["cell_type"],
                "method": "Route2" if row["method"] == "DeconvSC" else "BaseVAE",
                "display_method": row["method"],
                "metric": metric,
                "value": row[column],
                "better": direction,
                "n_seeds_averaged": len(SEEDS),
            })
    long = pd.DataFrame(long_rows)
    long.to_csv(outdir / "topology_metrics_long.csv", index=False)
    tests[tests["setting"] == "P0_primary"].to_csv(outdir / "paired_statistics.csv", index=False)

    module = primary.pivot(index="cell_type", columns="method", values="module_l2_mean").dropna().sort_index()
    module_test = tests[(tests["setting"] == "P0_primary") & (tests["metric"] == "module_l2_mean")].iloc[0]
    np.savez_compressed(
        outdir / "module_l2_withinType_dists.npz",
        Route2=module["DeconvSC"].to_numpy(dtype=np.float64),
        BaseVAE=module["BaseVAE"].to_numpy(dtype=np.float64),
        cell_types=module.index.to_numpy(dtype=str),
        p=np.float64(module_test["p_one_sided"]),
        p_two_sided=np.float64(module_test["p_two_sided"]),
        unit=np.asarray("cell_type"),
        value_definition=np.asarray("mean over modules within seed, then mean over 10 seeds"),
    )

    arrays = {"cell_types": module.index.to_numpy(dtype=str)}
    short = {
        "hubJ": "hub_jaccard_mean",
        "mierr": "mi_fidelity_error_median",
        "modL2": "module_l2_mean",
        "edgeMAE": "per_gene_coexpression_edge_mae_median",
        "netfid": "per_gene_network_fidelity_median",
        "netpcc": "network_pcc",
    }
    for prefix, column in short.items():
        pivot = primary.pivot(index="cell_type", columns="method", values=column).loc[module.index]
        arrays[f"{prefix}_Route2"] = pivot["DeconvSC"].to_numpy(dtype=np.float64)
        arrays[f"{prefix}_BaseVAE"] = pivot["BaseVAE"].to_numpy(dtype=np.float64)
    np.savez_compressed(outdir / "within_celltype_panel_data.npz", **arrays)

    readme = """# Figure 4-ready within-cell-type inputs

- `topology_metrics_long.csv`: preferred tidy input for rebuilding topology panels.
- `paired_statistics.csv`: exact cell-type-level paired tests and effect sizes.
- `within_celltype_panel_data.npz`: compact arrays for hub Jaccard, MI error, module L2, per-gene co-expression edge MAE, network fidelity and network PCC.
- `module_l2_withinType_dists.npz`: drop-in key compatibility with the existing Figure 4 panel-C loader (`Route2`, `BaseVAE`, `p`).

Important: every plotted value is one cell type after averaging modules and/or genes within each seed and then averaging 10 seeds.  Figure 4 uses hub Jaccard, MI error, module L2 and per-gene co-expression edge MAE as complementary local, nonlinear, module-level and dense gene-centric views of topology.  Do not concatenate module×seed values as independent observations.  DISSECT is intentionally absent because it has no within-cell-type single-cell output.  These data quantify expression-topology fidelity, not causal GRN recovery.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    for directory in ("scripts", "logs", "cache", "figures", "figure4_inputs"):
        (outdir / directory).mkdir(parents=True, exist_ok=True)

    start = time.time()
    for name, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input {name}: {path}")

    datasets, common_genes, schema_rows = load_inputs()
    log(f"[Gate 0] four-way common genes: {len(common_genes)}")

    manifest_rows = []
    for name, path in PATHS.items():
        manifest_rows.append({
            "name": name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": "read-only input",
        })
    pd.DataFrame(manifest_rows).to_csv(outdir / "input_manifest.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(outdir / "input_schema_audit.csv", index=False)
    pd.DataFrame([
        {
            "date": "2026-07-10",
            "gate": "Gate 0 expression-scale audit",
            "planned_assumption": "Generated matrices already satisfy per-cell log1p-CP10K and should not be normalized again",
            "observed_evidence": "Median expm1 library on the common panel was approximately 2041 for Figure3 DeconvSC and 2752 for BaseVAE, versus approximately 9995 for GT",
            "resolution": "P0 uses expm1 -> per-cell CP10K -> log1p for all methods; untransformed decoder log values are retained as S7_native_decoder_scale",
            "result_seen_before_resolution": True,
            "interpretation": "Transparent protocol amendment required by a falsified input-scale assumption; robustness must be checked in both P0 and S7",
        },
        {
            "date": "2026-07-13",
            "gate": "Figure 4d display-endpoint revision",
            "planned_assumption": "Display median per-gene connectivity-profile PCC as Figure 4d",
            "observed_evidence": "The second-order PCC was difficult to interpret on an absolute 0-to-1 scale and did not directly quantify Pearson edge-magnitude error",
            "resolution": "Display median per-gene co-expression edge MAE in Figure 4d; retain per-gene connectivity-profile PCC and whole-network PCC in the output tables as supplementary sensitivity metrics",
            "result_seen_before_resolution": True,
            "interpretation": "Post-result endpoint revision requested for interpretability; disclose the change, preserve the original PCC outputs, and apply Holm correction to the four final Figure 4 metrics",
        },
    ]).to_csv(outdir / "protocol_deviations.csv", index=False)

    log("[Gate 1] verify Figure 3 DeconvSC provenance")
    provenance = figure3_provenance_check()
    provenance.to_csv(outdir / "provenance_checks.csv", index=False)
    if not bool(provenance["passed"].all()):
        raise RuntimeError("Figure 3 provenance check failed")

    log("[Gate 2] record attention-ablation equivalence limits")
    architecture_audit().to_csv(outdir / "architecture_equivalence_audit.csv", index=False)
    eligibility_methods = pd.DataFrame([
        {"method": "GT", "native_output": "single cells", "within_celltype_eligible": True, "role": "reference"},
        {"method": "DeconvSC Figure3", "native_output": "single cells", "within_celltype_eligible": True, "role": "primary"},
        {"method": "BaseVAE", "native_output": "single cells", "within_celltype_eligible": True, "role": "model comparator"},
        {"method": "DeconvSC current", "native_output": "single cells", "within_celltype_eligible": True, "role": "sensitivity"},
        {"method": "DISSECT", "native_output": "sample x cell-type profiles", "within_celltype_eligible": False, "role": "global only"},
        {"method": "BayesPrism", "native_output": "sample x cell-type profiles", "within_celltype_eligible": False, "role": "global only"},
        {"method": "TAPE", "native_output": "sample x cell-type profiles", "within_celltype_eligible": False, "role": "global only"},
        {"method": "CIBERSORTx", "native_output": "sample x major-lineage profiles", "within_celltype_eligible": False, "role": "global major only"},
    ])
    eligibility_methods.to_csv(outdir / "eligibility_matrix.csv", index=False)

    counts, group_indices = build_counts(datasets)
    counts.to_csv(outdir / "cell_count_audit.csv", index=False)

    settings = [
        {"setting": "P0_primary", "deconv": "DeconvSC", "threshold": 100, "donors": DONORS, "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": True},
        {"setting": "S1_current", "deconv": "DeconvSC_current", "threshold": 100, "donors": DONORS, "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": True},
        {"setting": "S2a_threshold50", "deconv": "DeconvSC", "threshold": 50, "donors": DONORS, "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "S2b_threshold200", "deconv": "DeconvSC", "threshold": 200, "donors": DONORS, "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "S3_no_donor_centering", "deconv": "DeconvSC", "threshold": 100, "donors": DONORS, "corr_mode": "pooled_uncentered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "S4_shared_global_panel", "deconv": "DeconvSC", "threshold": 100, "donors": DONORS, "corr_mode": "pooled_centered", "panel": "shared", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "S5_legacy_ward", "deconv": "DeconvSC", "threshold": 100, "donors": DONORS, "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "ward", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "S6_donor_fisher", "deconv": "DeconvSC", "threshold": 100, "donors": DONORS, "corr_mode": "fisher", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "S7_native_decoder_scale", "deconv": "DeconvSC", "threshold": 100, "donors": DONORS, "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "native_decoder_log", "full_metrics": True},
        {"setting": "LODO_without_LDK1", "deconv": "DeconvSC", "threshold": 100, "donors": ("LDK2", "LDK3"), "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "LODO_without_LDK2", "deconv": "DeconvSC", "threshold": 100, "donors": ("LDK1", "LDK3"), "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
        {"setting": "LODO_without_LDK3", "deconv": "DeconvSC", "threshold": 100, "donors": ("LDK1", "LDK2"), "corr_mode": "pooled_centered", "panel": "celltype", "linkage": "average", "expression_scale": "cellwise_log1p_cp10k", "full_metrics": False},
    ]

    setting_info = {}
    eligibility_rows = []
    for setting in settings:
        types, valid, audit = eligible_types(counts, setting["deconv"], setting["threshold"], setting["donors"])
        for row in audit:
            row["setting"] = setting["setting"]
        eligibility_rows.extend(audit)
        setting_info[setting["setting"]] = {"types": types, "valid_donors": valid}
        log(f"[Gate 3] {setting['setting']}: {len(types)} cell types = {types}")
    pd.DataFrame(eligibility_rows).to_csv(outdir / "setting_eligibility.csv", index=False)
    if set(setting_info["P0_primary"]["types"]) != PRIMARY_EXPECTED_TYPES:
        raise RuntimeError(f"Primary eligibility changed: {setting_info['P0_primary']['types']}")

    log("[definitions] select GT-only cell-type-specific panels and modules")
    union_types = sorted(set().union(*(set(info["types"]) for info in setting_info.values())))
    panel_definitions: dict[str, np.ndarray] = {}
    reference_definitions: dict[str, np.ndarray] = {}
    module_definitions: dict[tuple[str, str], list[dict]] = {}
    panel_rows = []
    module_definition_rows = []
    for cell_type in union_types:
        variance = within_donor_variance(datasets["GT"], cell_type)
        indices = select_top_genes(variance, common_genes)
        panel_definitions[cell_type] = indices
        reference = full_gt_reference(datasets["GT"], cell_type, indices)
        reference_definitions[cell_type] = reference
        for rank, index in enumerate(indices, start=1):
            panel_rows.append({
                "panel": "celltype_specific",
                "cell_type": cell_type,
                "rank": rank,
                "gene": common_genes[index],
                "gt_within_donor_variance": variance[index],
            })
        for link_method in ("average", "ward"):
            modules = build_modules(reference, link_method)
            module_definitions[(cell_type, link_method)] = modules
            for module in modules:
                module_definition_rows.append({
                    "panel": "celltype_specific",
                    "cell_type": cell_type,
                    "linkage": link_method,
                    "module_id": module["module_id"],
                    "module_size": len(module["indices"]),
                    "gt_cohesion_mean_abs_r": module["cohesion"],
                    "genes": ";".join(common_genes[indices[module["indices"]]]),
                })

    primary_types = setting_info["P0_primary"]["types"]
    gt_obs = datasets["GT"]["obs"]
    global_rows = np.flatnonzero(gt_obs["cell_type"].isin(primary_types).to_numpy())
    global_adata = ad.AnnData(X=datasets["GT"]["X_cp10k"][global_rows], var=pd.DataFrame(index=common_genes))
    sc.pp.highly_variable_genes(global_adata, n_top_genes=TOPK)
    shared_indices = np.flatnonzero(global_adata.var["highly_variable"].to_numpy())
    shared_variance = np.asarray(global_adata.var.get("dispersions_norm", np.zeros(len(common_genes))), dtype=float)
    shared_indices = shared_indices[np.argsort(-np.nan_to_num(shared_variance[shared_indices]), kind="stable")][:TOPK]
    if len(shared_indices) != TOPK:
        raise RuntimeError("Shared GT HVG panel does not contain 500 genes")
    shared_corrs = [full_gt_reference(datasets["GT"], cell_type, shared_indices) for cell_type in primary_types]
    shared_reference = np.mean(np.stack(shared_corrs), axis=0)
    np.fill_diagonal(shared_reference, 1.0)
    shared_modules = build_modules(shared_reference, "average")
    for rank, index in enumerate(shared_indices, start=1):
        panel_rows.append({
            "panel": "shared_global_gt_hvg",
            "cell_type": "ALL_PRIMARY",
            "rank": rank,
            "gene": common_genes[index],
            "gt_within_donor_variance": np.nan,
        })
    for module in shared_modules:
        module_definition_rows.append({
            "panel": "shared_global_gt_hvg",
            "cell_type": "ALL_PRIMARY",
            "linkage": "average",
            "module_id": module["module_id"],
            "module_size": len(module["indices"]),
            "gt_cohesion_mean_abs_r": module["cohesion"],
            "genes": ";".join(common_genes[shared_indices[module["indices"]]]),
        })
    pd.DataFrame(panel_rows).to_csv(outdir / "within_type_gene_panel.csv", index=False)
    pd.DataFrame(module_definition_rows).to_csv(outdir / "within_type_modules.csv", index=False)

    log("[Gate 5–7] run primary and pre-specified sensitivities")
    metric_rows = []
    module_rows = []
    hub_rows = []
    for setting in settings:
        setting_id = setting["setting"]
        for cell_type in setting_info[setting_id]["types"]:
            donors = setting_info[setting_id]["valid_donors"][cell_type]
            if setting["panel"] == "shared":
                gene_indices = shared_indices
                definition_reference = shared_reference
                modules = shared_modules
                panel_id = "shared_global_gt_hvg"
            else:
                gene_indices = panel_definitions[cell_type]
                definition_reference = reference_definitions[cell_type]
                modules = module_definitions[(cell_type, setting["linkage"])]
                panel_id = "celltype_specific"
            fixed_hubs = hub_indices(definition_reference)
            panel_genes = common_genes[gene_indices]
            for seed in SEEDS:
                correlations, matrices, constants, n_matched = sample_correlations(
                    datasets, group_indices, setting["deconv"], cell_type, donors,
                    gene_indices, seed, setting["corr_mode"], setting["expression_scale"],
                )
                gt_corr = correlations["GT"]
                iu = np.triu_indices(len(gene_indices), 1)
                offdiag = ~np.eye(len(gene_indices), dtype=bool)
                mi_gt = mi_gt_rank = None
                if setting["full_metrics"]:
                    mi_gt = mutual_information_all_pairs(equal_width_codes(matrices["GT"][:, :MI_TOPK], 20), 20)
                    mi_gt_rank = mutual_information_all_pairs(rank_quantile_codes(matrices["GT"][:, :MI_TOPK], 10), 10)

                for method in ("DeconvSC", "BaseVAE"):
                    predicted = correlations[method]
                    l2_values, rmse_values, mae_values = [], [], []
                    for module in modules:
                        indices = module["indices"]
                        gt_block = gt_corr[np.ix_(indices, indices)]
                        pred_block = predicted[np.ix_(indices, indices)]
                        frobenius = float(np.linalg.norm(gt_block - pred_block))
                        size = len(indices)
                        l2 = frobenius / (size * size)
                        rmse = frobenius / size
                        triangle = np.triu_indices(size, 1)
                        edge_mae = float(np.mean(np.abs(gt_block[triangle] - pred_block[triangle])))
                        l2_values.append(l2)
                        rmse_values.append(rmse)
                        mae_values.append(edge_mae)
                        module_rows.append({
                            "setting": setting_id,
                            "cell_type": cell_type,
                            "method": method,
                            "seed": seed,
                            "panel": panel_id,
                            "linkage": setting["linkage"],
                            "module_id": module["module_id"],
                            "module_size": size,
                            "gt_cohesion_mean_abs_r": module["cohesion"],
                            "module_l2": l2,
                            "module_rmse": rmse,
                            "module_edge_mae": edge_mae,
                        })

                    row = {
                        "setting": setting_id,
                        "cell_type": cell_type,
                        "method": method,
                        "seed": seed,
                        "deconv_source": setting["deconv"],
                        "threshold": setting["threshold"],
                        "donors": ";".join(donors),
                        "n_donors": len(donors),
                        "n_cells_matched_per_method": n_matched,
                        "correlation_mode": setting["corr_mode"],
                        "expression_scale": setting["expression_scale"],
                        "panel": panel_id,
                        "linkage": setting["linkage"],
                        "n_modules": len(modules),
                        "constant_gene_fraction_gt": constants["GT"],
                        "constant_gene_fraction_method": constants[method],
                        "module_l2_mean": float(np.mean(l2_values)),
                        "module_l2_median": float(np.median(l2_values)),
                        "module_rmse_mean": float(np.mean(rmse_values)),
                        "module_edge_mae_mean": float(np.mean(mae_values)),
                        "hub_jaccard_mean": np.nan,
                        "hub_jaccard_median": np.nan,
                        "mi_fidelity_error_mean": np.nan,
                        "mi_fidelity_error_median": np.nan,
                        "mi_rank10_error_mean": np.nan,
                        "mi_rank10_error_median": np.nan,
                        "network_pcc": np.nan,
                        "per_gene_network_fidelity_mean": np.nan,
                        "per_gene_network_fidelity_median": np.nan,
                        "per_gene_coexpression_edge_mae_mean": np.nan,
                        "per_gene_coexpression_edge_mae_median": np.nan,
                    }
                    if setting["full_metrics"]:
                        jaccards = []
                        for hub in fixed_hubs:
                            real_neighbors = neighbors(gt_corr, hub)
                            predicted_neighbors = neighbors(predicted, hub)
                            union = real_neighbors | predicted_neighbors
                            value = len(real_neighbors & predicted_neighbors) / len(union) if union else 0.0
                            jaccards.append(value)
                            hub_rows.append({
                                "setting": setting_id,
                                "cell_type": cell_type,
                                "method": method,
                                "seed": seed,
                                "hub_gene": panel_genes[hub],
                                "hub_jaccard": value,
                            })
                        method_mi = mutual_information_all_pairs(equal_width_codes(matrices[method][:, :MI_TOPK], 20), 20)
                        mi_error = np.abs(method_mi - mi_gt)
                        method_mi_rank = mutual_information_all_pairs(rank_quantile_codes(matrices[method][:, :MI_TOPK], 10), 10)
                        mi_rank_error = np.abs(method_mi_rank - mi_gt_rank)
                        gene_fidelity = np.empty(len(gene_indices), dtype=np.float64)
                        for gene in range(len(gene_indices)):
                            gene_fidelity[gene] = safe_vector_corr(gt_corr[gene][offdiag[gene]], predicted[gene][offdiag[gene]])
                        absolute_edge_error = np.abs(predicted - gt_corr)
                        gene_edge_mae = (
                            np.sum(absolute_edge_error * offdiag, axis=1)
                            / max(len(gene_indices) - 1, 1)
                        )
                        row.update({
                            "hub_jaccard_mean": float(np.mean(jaccards)),
                            "hub_jaccard_median": float(np.median(jaccards)),
                            "mi_fidelity_error_mean": float(np.mean(mi_error)),
                            "mi_fidelity_error_median": float(np.median(mi_error)),
                            "mi_rank10_error_mean": float(np.mean(mi_rank_error)),
                            "mi_rank10_error_median": float(np.median(mi_rank_error)),
                            "network_pcc": safe_vector_corr(gt_corr[iu], predicted[iu]),
                            "per_gene_network_fidelity_mean": float(np.mean(gene_fidelity)),
                            "per_gene_network_fidelity_median": float(np.median(gene_fidelity)),
                            "per_gene_coexpression_edge_mae_mean": float(np.mean(gene_edge_mae)),
                            "per_gene_coexpression_edge_mae_median": float(np.median(gene_edge_mae)),
                        })
                    metric_rows.append(row)
            log(f"  {setting_id}: completed {cell_type} ({len(donors)} donors)")

    metrics_seed = pd.DataFrame(metric_rows)
    modules_seed = pd.DataFrame(module_rows)
    hubs_seed = pd.DataFrame(hub_rows)
    metrics_seed.to_csv(outdir / "within_type_metrics_by_seed.csv", index=False)
    modules_seed.to_csv(outdir / "module_l2_by_celltype_module_seed.csv", index=False)
    hubs_seed.to_csv(outdir / "hub_jaccard_by_celltype_hub_seed.csv", index=False)

    numeric_columns = [
        "n_donors", "n_cells_matched_per_method", "n_modules",
        "constant_gene_fraction_gt", "constant_gene_fraction_method",
        "module_l2_mean", "module_l2_median", "module_rmse_mean", "module_edge_mae_mean",
        "hub_jaccard_mean", "hub_jaccard_median",
        "mi_fidelity_error_mean", "mi_fidelity_error_median",
        "mi_rank10_error_mean", "mi_rank10_error_median",
        "network_pcc", "per_gene_network_fidelity_mean", "per_gene_network_fidelity_median",
        "per_gene_coexpression_edge_mae_mean", "per_gene_coexpression_edge_mae_median",
    ]
    cell_table = metrics_seed.groupby(["setting", "cell_type", "method"], sort=True)[numeric_columns].mean().reset_index()
    cell_table.to_csv(outdir / "within_type_metrics_by_celltype.csv", index=False)

    test_specs = [
        ("module_l2_mean", False, "primary"),
        ("module_rmse_mean", False, "scale_sensitivity"),
        ("module_edge_mae_mean", False, "scale_sensitivity"),
        ("hub_jaccard_mean", True, "secondary"),
        ("mi_fidelity_error_median", False, "secondary"),
        ("mi_rank10_error_median", False, "estimator_sensitivity"),
        ("per_gene_coexpression_edge_mae_median", False, "figure4_secondary"),
        ("network_pcc", True, "secondary"),
        ("per_gene_network_fidelity_median", True, "supplementary"),
    ]
    tests = []
    for metric, higher, family in test_specs:
        result = paired_test(cell_table, metric, higher, "P0_primary")
        result["family"] = family
        tests.append(result)
    figure4_metrics = {
        "hub_jaccard_mean",
        "mi_fidelity_error_median",
        "module_l2_mean",
        "per_gene_coexpression_edge_mae_median",
    }
    figure4_indices = [i for i, row in enumerate(tests) if row["metric"] in figure4_metrics]
    adjusted = holm_adjust([tests[i]["p_one_sided"] for i in figure4_indices])
    for row in tests:
        row["p_one_sided_holm"] = np.nan
        row["figure4_metric"] = row["metric"] in figure4_metrics
    for index, value in zip(figure4_indices, adjusted):
        tests[index]["p_one_sided_holm"] = value
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(outdir / "statistical_tests.csv", index=False)

    sensitivity = []
    for setting in [row["setting"] for row in settings]:
        result = paired_test(cell_table, "module_l2_mean", False, setting)
        result["analysis_role"] = "confirmatory" if setting == "P0_primary" else "sensitivity"
        sensitivity.append(result)
    sensitivity_df = pd.DataFrame(sensitivity)
    sensitivity_df.to_csv(outdir / "sensitivity_analysis.csv", index=False)

    primary = cell_table[cell_table["setting"] == "P0_primary"].copy()
    raw_arrays = {"cell_types": np.asarray(sorted(primary["cell_type"].unique()), dtype=str)}
    for metric, _, _ in test_specs:
        pivot = primary.pivot(index="cell_type", columns="method", values=metric).loc[raw_arrays["cell_types"]]
        raw_arrays[f"{metric}__DeconvSC"] = pivot["DeconvSC"].to_numpy(dtype=np.float64)
        raw_arrays[f"{metric}__BaseVAE"] = pivot["BaseVAE"].to_numpy(dtype=np.float64)
    np.savez_compressed(outdir / "raw_distributions.npz", **raw_arrays)

    plot_primary(cell_table, outdir / "figures")
    write_figure4_inputs(cell_table, tests_df, outdir / "figure4_inputs")

    config = {
        "analysis": "GSE141115 within-cell-type expression topology",
        "samples": list(DONORS),
        "seeds": list(SEEDS),
        "topology_genes": TOPK,
        "mi_genes": MI_TOPK,
        "n_hubs": N_HUBS,
        "neighbors_per_hub": K_NEIGHBORS,
        "candidate_modules": N_MODULES,
        "top_modules": TOP_MODULES,
        "minimum_module_size": MIN_MODULE_SIZE,
        "minimum_cells_per_donor_method": PER_DONOR_MIN,
        "cap_cells_per_donor_method": PER_DONOR_CAP,
        "primary_cell_types": setting_info["P0_primary"]["types"],
        "common_gene_count": len(common_genes),
        "settings": settings,
        "inferential_unit": "cell_type",
        "claim_boundary": "expression co-expression/dependency topology; not causal GRN",
        "scale_protocol_amendment": "Gate 0 showed generated native decoder log outputs do not conserve CP10K libraries; P0 therefore applies expm1 -> cellwise CP10K -> log1p, with native values retained as S7.",
    }
    (outdir / "analysis_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    runtime = {
        "status": "completed",
        "start_epoch": start,
        "end_epoch": time.time(),
        "duration_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "anndata": ad.__version__,
        "scanpy": sc.__version__,
        "numba": numba.__version__,
        "cpu_count": os.cpu_count(),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    log(f"[completed] duration={runtime['duration_seconds']:.1f}s; outputs={outdir}")


if __name__ == "__main__":
    main()
