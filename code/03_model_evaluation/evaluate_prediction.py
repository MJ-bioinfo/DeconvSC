# scripts/evaluate_generation.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scanpy as sc
import torch

from deconvsc.config import load_config, save_config, get_config_value, update_nested
from deconvsc.io import load_preprocess_bundle
from deconvsc.utils import set_seed

from deconvsc.evaluate import (
    evaluate_generated_data,
    compare_gene_coexpression_networks,
    advanced_attention_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeconvSC generated single-cell data.")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation YAML config.")

    parser.add_argument("--generated", type=str, default=None, help="Override input.generated_h5ad.")
    parser.add_argument("--real-sc", type=str, default=None, help="Override input.real_sc_path.")
    parser.add_argument("--outdir", type=str, default=None, help="Override output.output_dir.")
    return parser.parse_args()


def normalize_real_sc(adata, target_sum=1e4, log1p=True):
    """
    Normalize real scRNA-seq data in the same manner as your original script.
    """
    if not isinstance(adata.X, type(None)):
        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()

    sc.pp.normalize_total(adata, target_sum=target_sum)

    if log1p:
        sc.pp.log1p(adata)

    return adata


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.generated is not None:
        update_nested(cfg, "input.generated_h5ad", args.generated)
    if args.real_sc is not None:
        update_nested(cfg, "input.real_sc_path", args.real_sc)
    if args.outdir is not None:
        update_nested(cfg, "output.output_dir", args.outdir)

    processed_dir = get_config_value(cfg, "input.processed_dir")
    metadata_file = get_config_value(cfg, "input.metadata_file", "sc_metadata.pkl")
    generated_h5ad = get_config_value(cfg, "input.generated_h5ad")
    real_sc_path = get_config_value(cfg, "input.real_sc_path")
    other_generated_h5ad = get_config_value(cfg, "input.other_generated_h5ad", None)

    output_dir = Path(get_config_value(cfg, "output.output_dir", "output/evaluate"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(get_config_value(cfg, "runtime.seed", 18))
    set_seed(seed)

    # Read the generated cells and the held-out single-cell test reference. Keep the reference RAW:
    # the deconv_20260610-method evaluation normalises internally (it builds (sample,cell_type)
    # pseudobulks from BOTH generated and real cells), so we must NOT pre-normalise it here.
    adata_gen = sc.read_h5ad(generated_h5ad)
    adata_real_raw = sc.read_h5ad(real_sc_path)

    real_celltype_key = get_config_value(cfg, "annotation.real_celltype_key", "cell type")
    real_sample_key = get_config_value(cfg, "annotation.real_sample_key", "sample")
    generated_celltype_key = get_config_value(cfg, "annotation.generated_celltype_key", "Cell_type")
    generated_sample_key = get_config_value(cfg, "annotation.generated_sample_key", "Sample")

    # 1. DeconvSC evaluation (deconv_20260610 methodology): profile-wise + gene-wise PCC (all genes
    #    and marker genes), generated-data UMAP, and the real-vs-generated comparison marker heatmap.
    evaluate_generated_data(
        adata_gen=adata_gen,
        adata_real=adata_real_raw,
        output_dir=str(output_dir),
        gen_sample_col=generated_sample_key,
        gen_cell_col=generated_celltype_key,
        real_sample_col=real_sample_key,
        real_cell_col=real_celltype_key,
        real_is_raw=bool(get_config_value(cfg, "evaluation.real_is_raw", True)),
        sample_level=bool(get_config_value(cfg, "evaluation.sample_level", True)),
        sample_map=get_config_value(cfg, "annotation.gt_sample_map", None),
        marker_top_n=int(get_config_value(cfg, "evaluation.marker_top_n", 30)),
        heatmap_top_n=int(get_config_value(cfg, "evaluation.heatmap_top_n", 5)),
        umap_per_sample=int(get_config_value(cfg, "evaluation.umap_per_sample", 90)),
        umap_sample_limit=int(get_config_value(cfg, "evaluation.umap_sample_limit", 28)),
        make_umap=bool(get_config_value(cfg, "evaluation.make_umap", True)),
        make_heatmap=bool(get_config_value(cfg, "evaluation.make_heatmap", True)),
        emf=bool(get_config_value(cfg, "evaluation.emf", True)),
        seed=seed,
    )

    # The optional legacy analyses below operate on a NORMALISED real copy (log1p CP10k).
    adata_real = adata_real_raw
    adata_real.obs["labels"] = adata_real.obs[real_celltype_key].astype(str).astype("category")
    if bool(get_config_value(cfg, "evaluation.normalize_real_sc", True)):
        adata_real = normalize_real_sc(
            adata_real,
            target_sum=float(get_config_value(cfg, "evaluation.target_sum", 1e4)),
            log1p=bool(get_config_value(cfg, "evaluation.log1p", True)),
        )

    # 2. Gene co-expression network comparison.
    n_top_genes_network = int(get_config_value(cfg, "evaluation.n_top_genes_network", 500))

    # If there is no ablation/other generated data, use real data as placeholder.
    if other_generated_h5ad is not None:
        adata_other = sc.read_h5ad(other_generated_h5ad)
    else:
        adata_other = adata_real

    try:
        compare_gene_coexpression_networks(
            adata_real=adata_real,
            adata_gen_ours=adata_gen,
            adata_gen_ablation=adata_other,
            output_dir=str(output_dir),
            n_top_genes=n_top_genes_network,
            specific_genes=None,
        )
    except TypeError:
        print(
            "[Warning] compare_gene_coexpression_networks signature does not match this script. "
            "Please make its interface: compare_gene_coexpression_networks(adata_real, adata_ours, adata_ablation, output_dir, n_top_genes, specific_genes)."
        )

    # 3. Advanced attention-related evaluation.
    try:
        advanced_attention_evaluation(
            adata_real=adata_real,
            adata_ours=adata_gen,
            adata_dissect=adata_other,
            output_dir=str(output_dir),
        )
    except TypeError:
        print(
            "[Warning] advanced_attention_evaluation signature does not match this script. "
            "Please make its interface: advanced_attention_evaluation(adata_real, adata_ours, adata_dissect, output_dir)."
        )

    save_config(cfg, output_dir / "config_used.yaml")

    print("[DeconvSC] Evaluation finished.")
    print(f"[DeconvSC] Output directory: {output_dir}")


if __name__ == "__main__":
    main()