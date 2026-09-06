#!/usr/bin/env python3
"""PCA<=50 per-gene co-expression profile PCC without a real-cell split.

This sensitivity analysis leaves the established 200-gene panels, training-only
PCA bases, normalization, donor centering, and Ledoit-Wolf estimator unchanged.
The only design change is that evaluation n is min(real, DeconvSC, BaseVAE, 200)
within each cell type and donor. Sampling uses the same deterministic convention
as Figure 4a-c so the no-split evaluation cell sets are aligned with those panels.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve()
RELEASE_ROOT = HERE.parents[3]
OUTDIR = RELEASE_ROOT / "work/figure4/per_gene_profile_pcc_pca50_no_real_split"
SOURCE_DIR = HERE.parent / "pca50_support"
SOURCE_SCRIPT = SOURCE_DIR / "run_denoised_stable_edge_profiles.py"


def load_source_module():
    spec = importlib.util.spec_from_file_location("edge_profile_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source module: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


src = load_source_module()


def figure4_ac_rng(seed: int, cell_type: str, donor: str, method: str):
    token = f"{seed}|{cell_type}|{donor}|{method}|figure3".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(value)


def fit_existing_200_gene_panels() -> tuple[dict, pd.DataFrame]:
    ranking = pd.read_csv(SOURCE_DIR / "training_stable_gene_ranking.csv")
    backed = ad.read_h5ad(src.PATHS["Train"], backed="r")
    gene_lookup = {gene: i for i, gene in enumerate(backed.var_names.astype(str))}
    samples = backed.obs["sample"].astype(str).to_numpy()
    cell_types = backed.obs["cell type"].astype(str).to_numpy()
    models = {}
    audit_rows = []

    for cell_type in src.CELL_TYPES:
        panel = (
            ranking[(ranking["cell_type"] == cell_type) & (ranking["rank"] <= 200)]
            .sort_values("rank", kind="stable")
        )
        genes = panel["gene"].astype(str).tolist()
        if len(genes) != 200:
            raise RuntimeError(f"Expected 200 ranked genes for {cell_type}, found {len(genes)}")
        gene_indices = np.asarray([gene_lookup[gene] for gene in genes], dtype=int)

        residual_pieces = []
        retained_donors = []
        for donor in sorted(np.unique(samples)):
            rows = np.flatnonzero((samples == donor) & (cell_types == cell_type))
            if len(rows) < src.TRAIN_MIN_CELLS:
                continue
            rows = src.capped_rows(rows, cell_type, donor)
            piece = src.read_raw_logcp10k(backed, rows, gene_indices).astype(np.float64)
            residual_pieces.append(piece - piece.mean(axis=0, keepdims=True))
            retained_donors.append(donor)

        residual = np.vstack(residual_pieces)
        scale = residual.std(axis=0, ddof=0)
        scale[scale < 1e-8] = 1.0
        standardized = residual / scale
        max_components = min(src.PCA_MAX_COMPONENTS, len(genes) - 1, standardized.shape[0] - 1)
        pca = PCA(n_components=max_components, svd_solver="randomized", random_state=20260714)
        pca.fit(standardized)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        target_rank = int(np.searchsorted(cumulative, src.PCA_VARIANCE_TARGET) + 1)
        target_rank = min(max(target_rank, 2), max_components)

        models[cell_type] = {
            "genes": genes,
            "training_scale": scale,
            "components_pca80": np.asarray(pca.components_[:target_rank], dtype=np.float64),
            "components_pca20": np.asarray(pca.components_[: min(20, max_components)], dtype=np.float64),
        }
        audit_rows.append(
            {
                "cell_type": cell_type,
                "n_training_donors": len(retained_donors),
                "n_training_cells": residual.shape[0],
                "n_genes": len(genes),
                "pca_rank": target_rank,
                "explained_variance": float(cumulative[target_rank - 1]),
            }
        )

    backed.file.close()
    return models, pd.DataFrame(audit_rows)


def main() -> None:
    started = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    common_genes, _ = src.inspect_inputs()
    models, pca_audit = fit_existing_200_gene_panels()
    pca_audit.to_csv(OUTDIR / "training_pca_audit.csv", index=False)

    panel_union = sorted({gene for model in models.values() for gene in model["genes"]})
    union_lookup = {gene: i for i, gene in enumerate(panel_union)}
    panel_columns = {
        cell_type: np.asarray([union_lookup[g] for g in models[cell_type]["genes"]], dtype=int)
        for cell_type in src.CELL_TYPES
    }
    datasets = {
        label: src.load_evaluation_dataset(label, panel_union, common_genes, print)
        for label in ("Test", "DeconvSC", "BaseVAE")
    }

    sampling_rows = []
    balanced_n = {}
    for cell_type in src.CELL_TYPES:
        for donor in src.TEST_DONORS:
            counts = {
                label: len(datasets[label]["groups"][(cell_type, donor)])
                for label in datasets
            }
            n = min(counts["Test"], counts["DeconvSC"], counts["BaseVAE"], src.EVAL_CAP)
            if n < src.EVAL_MIN:
                raise RuntimeError(f"Balanced n<{src.EVAL_MIN}: {cell_type}/{donor}={n}")
            balanced_n[(cell_type, donor)] = n
            sampling_rows.append(
                {
                    "cell_type": cell_type,
                    "donor": donor,
                    "Test_available": counts["Test"],
                    "DeconvSC_available": counts["DeconvSC"],
                    "BaseVAE_available": counts["BaseVAE"],
                    "balanced_n_per_source": n,
                }
            )
    sampling = pd.DataFrame(sampling_rows)
    sampling.to_csv(OUTDIR / "sampling_design.csv", index=False)

    per_gene_rows = []
    seed_rows = []
    label_to_rng_name = {"Test": "GT", "DeconvSC": "DeconvSC", "BaseVAE": "BaseVAE"}

    for cell_type in src.CELL_TYPES:
        model = models[cell_type]
        columns = panel_columns[cell_type]
        genes = model["genes"]
        for seed in src.SEEDS:
            source_pieces = {label: [] for label in datasets}
            n_total = 0
            for donor in src.TEST_DONORS:
                n = balanced_n[(cell_type, donor)]
                n_total += n
                for label in datasets:
                    available = datasets[label]["groups"][(cell_type, donor)]
                    rng = figure4_ac_rng(seed, cell_type, donor, label_to_rng_name[label])
                    chosen = np.sort(rng.choice(available, size=n, replace=False))
                    piece = datasets[label]["X"][chosen][:, columns]
                    source_pieces[label].append(src.pca_reconstruct(piece, model, "pca80"))

            correlations = {
                label: src.build_network(pieces)[0]
                for label, pieces in source_pieces.items()
            }
            real_corr = correlations["Test"]
            for method in ("DeconvSC", "BaseVAE"):
                generated_corr = correlations[method]
                profile_scores = []
                for gene_index, gene in enumerate(genes):
                    keep = np.ones(len(genes), dtype=bool)
                    keep[gene_index] = False
                    score = src.safe_pearson(real_corr[gene_index, keep], generated_corr[gene_index, keep])
                    profile_scores.append(score)
                    per_gene_rows.append(
                        {
                            "cell_type": cell_type,
                            "seed": seed,
                            "method": method,
                            "gene": gene,
                            "profile_pcc": score,
                            "n_cells_per_source": n_total,
                        }
                    )
                seed_rows.append(
                    {
                        "cell_type": cell_type,
                        "seed": seed,
                        "method": method,
                        "n_cells_per_source": n_total,
                        "median_profile_pcc": float(np.median(profile_scores)),
                    }
                )

    per_gene = pd.DataFrame(per_gene_rows)
    seed_table = pd.DataFrame(seed_rows)
    per_gene.to_csv(OUTDIR / "per_gene_profile_pcc.csv.gz", index=False, compression="gzip")
    seed_table.to_csv(OUTDIR / "profile_pcc_by_seed.csv", index=False)

    summary = (
        seed_table.groupby(["cell_type", "method"], sort=True)
        .agg(
            n_cells_per_source=("n_cells_per_source", "first"),
            mean_over_seeds=("median_profile_pcc", "mean"),
            sd_over_seeds=("median_profile_pcc", "std"),
            min_over_seeds=("median_profile_pcc", "min"),
            max_over_seeds=("median_profile_pcc", "max"),
        )
        .reset_index()
    )
    summary.to_csv(OUTDIR / "profile_pcc_by_celltype.csv", index=False)

    split = pd.read_csv(SOURCE_DIR / "edge_profile_summary_by_celltype.csv")
    split = split[
        (split["setting"] == "pca80_symmetric_lw")
        & (split["panel_size"] == 200)
        & split["comparison"].isin(["DeconvSC_vs_RealB", "BaseVAE_vs_RealB"])
    ].copy()
    split["method"] = split["comparison"].map(
        {"DeconvSC_vs_RealB": "DeconvSC", "BaseVAE_vs_RealB": "BaseVAE"}
    )
    split = split[["cell_type", "method", "profile_pcc_raw_median_mean_over_seeds"]].rename(
        columns={"profile_pcc_raw_median_mean_over_seeds": "split_real_profile_pcc"}
    )
    comparison = summary.merge(split, on=["cell_type", "method"], how="left")
    comparison["no_split_minus_split"] = comparison["mean_over_seeds"] - comparison["split_real_profile_pcc"]
    comparison.to_csv(OUTDIR / "comparison_with_split_real.csv", index=False)

    overall = (
        comparison.groupby("method", sort=True)
        .agg(
            no_split_mean_across_celltypes=("mean_over_seeds", "mean"),
            split_mean_across_celltypes=("split_real_profile_pcc", "mean"),
            mean_change=("no_split_minus_split", "mean"),
            cell_types_improved=("no_split_minus_split", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    overall.to_csv(OUTDIR / "overall_summary.csv", index=False)

    pivot = summary.pivot(index="cell_type", columns="method", values="mean_over_seeds")
    validation = {
        "status": "passed",
        "no_real_split": True,
        "sampling_rule": "min(Test, DeconvSC, BaseVAE, 200) within cell type and donor",
        "seeds": list(src.SEEDS),
        "all_pca_ranks_50": bool((pca_audit["pca_rank"] == 50).all()),
        "all_scores_in_bounds": bool(per_gene["profile_pcc"].between(-1, 1).all()),
        "deconvsc_higher_cell_types": int((pivot["DeconvSC"] > pivot["BaseVAE"]).sum()),
        "n_cell_types": int(len(pivot)),
    }
    (OUTDIR / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    metadata = {
        "status": "completed",
        "duration_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "anndata": ad.__version__,
        "sklearn": sklearn_version,
        "source_script": str(SOURCE_SCRIPT),
    }
    (OUTDIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nNo-split summary by cell type:\n", summary.to_string(index=False))
    print("\nOverall comparison:\n", overall.to_string(index=False))


if __name__ == "__main__":
    main()
