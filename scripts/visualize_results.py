# scripts/visualize_results.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deconvsc.config import load_config, save_config, get_config_value, update_nested
from deconvsc.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DeconvSC generated AnnData.")
    parser.add_argument("--config", type=str, required=True, help="Path to visualization/evaluation YAML config.")
    parser.add_argument("--generated", type=str, default=None, help="Override input.generated_h5ad.")
    parser.add_argument("--outdir", type=str, default=None, help="Override output.output_dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.generated is not None:
        update_nested(cfg, "input.generated_h5ad", args.generated)
    if args.outdir is not None:
        update_nested(cfg, "output.output_dir", args.outdir)

    generated_h5ad = get_config_value(cfg, "input.generated_h5ad")
    output_dir = Path(get_config_value(cfg, "output.output_dir", "outputs/visualize"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(get_config_value(cfg, "runtime.seed", 18))
    set_seed(seed)

    celltype_key = get_config_value(cfg, "annotation.generated_celltype_key", "Cell_type")
    sample_key = get_config_value(cfg, "annotation.sample_key", "Sample")

    adata = sc.read_h5ad(generated_h5ad)

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    if celltype_key in adata.obs:
        sc.pl.umap(
            adata,
            color=celltype_key,
            title="Generated Data UMAP by Cell Type",
            legend_loc="on data",
            show=False,
            size=30,
        )
        plt.savefig(output_dir / "generated_umap_by_celltype.png", dpi=300, bbox_inches="tight")
        plt.close()

    if sample_key in adata.obs:
        sc.pl.umap(
            adata,
            color=sample_key,
            title="Generated Data UMAP by Sample",
            legend_loc="on data",
            show=False,
            size=30,
        )
        plt.savefig(output_dir / "generated_umap_by_sample.png", dpi=300, bbox_inches="tight")
        plt.close()

    adata.write_h5ad(output_dir / "generated_data_with_umap.h5ad")
    save_config(cfg, output_dir / "config_used.yaml")

    print("[DeconvSC] Visualization finished.")
    print(f"[DeconvSC] Output directory: {output_dir}")


if __name__ == "__main__":
    main()