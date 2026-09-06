# scripts/predict_bulk.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from deconvsc.config import load_config, save_config, get_config_value, update_nested
from deconvsc.io import load_preprocess_bundle, load_decode_params, save_generated_anndata
from deconvsc.model import AttentionVAE
from deconvsc.infer import (
    train_anchored_residual,
    generate_prior_anchored,
)
from deconvsc.utils import set_seed, get_sample_indices_from_bulk


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cell-state-resolved expression from bulk samples.")
    parser.add_argument("--config", type=str, required=True, help="Path to prediction YAML config.")

    # Optional overrides
    parser.add_argument("--processed-dir", type=str, default=None, help="Override input.processed_dir.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override input.checkpoint.")
    parser.add_argument("--prior", type=str, default=None, help="Override input.prior.")
    parser.add_argument("--bulk", type=str, default=None, help="Override input.bulk_path.")
    parser.add_argument("--outdir", type=str, default=None, help="Override output.output_dir.")
    parser.add_argument("--device", type=str, default=None, help="Override runtime.device.")
    parser.add_argument("--samples", nargs="*", default=None, help="Override samples.target_samples.")
    return parser.parse_args()


def require_reference_path(cfg) -> str:
    """Reference scRNA-seq h5ad (donor + cell-type columns) for the residual + prophead training."""
    p = get_config_value(cfg, "generation.sc_ref_path") or get_config_value(cfg, "input.sc_path")
    if p is None:
        raise KeyError("Generation needs generation.sc_ref_path (reference scRNA h5ad with donor + cell-type columns).")
    return str(p)


def core_genes_to_indices(core_genes, common_genes):
    """Map core gene names to their positions in common_genes (the model output gene order)."""
    pos = {g: i for i, g in enumerate(list(common_genes))}
    return [pos[g] for g in core_genes if g in pos]


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.processed_dir is not None:
        update_nested(cfg, "input.processed_dir", args.processed_dir)
    if args.checkpoint is not None:
        update_nested(cfg, "input.checkpoint", args.checkpoint)
    if args.prior is not None:
        update_nested(cfg, "input.prior", args.prior)
    if args.bulk is not None:
        update_nested(cfg, "input.bulk_path", args.bulk)
    if args.outdir is not None:
        update_nested(cfg, "output.output_dir", args.outdir)
    if args.device is not None:
        update_nested(cfg, "runtime.device", args.device)
    if args.samples is not None and len(args.samples) > 0:
        update_nested(cfg, "samples.target_samples", args.samples)

    processed_dir = get_config_value(cfg, "input.processed_dir")
    metadata_file = get_config_value(cfg, "input.metadata_file", "sc_metadata.pkl")
    bulk_path = get_config_value(cfg, "input.bulk_path")
    checkpoint_path = get_config_value(cfg, "input.checkpoint")
    prior_path = get_config_value(cfg, "input.prior")
    target_samples = get_config_value(cfg, "samples.target_samples", [])

    output_dir = Path(get_config_value(cfg, "output.output_dir", "output/predict"))
    generated_h5ad_name = get_config_value(cfg, "output.generated_h5ad", "generated_data.h5ad")
    generated_metadata_name = get_config_value(cfg, "output.generated_metadata", "generated_metadata.csv")
    latent_shift_name = get_config_value(cfg, "output.latent_shift", "final_z.pt")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(get_config_value(cfg, "runtime.device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    seed = int(get_config_value(cfg, "runtime.seed", 18))
    set_seed(seed)

    metadata = load_preprocess_bundle(
        processed_dir=processed_dir,
        pickle_name=metadata_file,
        device=None,
    )

    X_tensor = metadata["X_tensor"]
    labels = metadata["labels"]
    n_cell_types = int(metadata["n_cell_types"])
    real_bulk_log = metadata["real_bulk_log"]
    mapping_dict = metadata["mapping_dict"]
    common_genes = metadata["common_genes"]
    core_genes = metadata["core_genes"]
    sig_indices = metadata["sig_indices"]

    hidden_size_list = get_config_value(cfg, "model.hidden_size_list", [4096, 2048, 1024])
    mid_hidden_size = int(get_config_value(cfg, "model.mid_hidden_size", 256))
    embedding_dim = int(get_config_value(cfg, "model.embedding_dim", 32))
    nhead = int(get_config_value(cfg, "model.nhead", 4))
    num_layers = int(get_config_value(cfg, "model.num_layers", 4))
    ff_dim = int(get_config_value(cfg, "model.ff_dim", 64))

    model = AttentionVAE(
        input_size=len(core_genes),
        output_size=len(common_genes),
        hidden_size_list=hidden_size_list,
        mid_hidden_size=mid_hidden_size,
        num_cell_types=n_cell_types,
        embedding_dim=embedding_dim,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        seed=seed,
    )

    # decode-only load: prior-anchored inference uses only the decoder + priors, so the encoder
    # (and its input_size) need not match the published checkpoint's core-gene count.
    model = load_decode_params(model, checkpoint_path, device=device)

    priors = torch.load(prior_path, map_location=device)

    if not target_samples:
        # default to ALL bulk samples (genes-as-rows x samples-as-cols), mirroring deconvsc/main.py;
        # avoids hard-coding long sample-id lists (e.g. HCA's 105 pseudo_bulk_id columns).
        _sep = get_config_value(cfg, "input.sep")
        target_samples = list(pd.read_csv(bulk_path, sep=_sep if _sep else "\t", index_col=0).columns.astype(str))
        print(f"[DeconvSC] samples.target_samples empty -> using all {len(target_samples)} bulk samples.")

    sample_indices, valid_sample_names = get_sample_indices_from_bulk(
        real_bulk_path=bulk_path,
        target_samples=target_samples,
        sep=get_config_value(cfg, "input.sep"),
    )

    if not sample_indices:
        raise ValueError("No valid target samples found in bulk matrix.")
    input_bulk_log = real_bulk_log[sample_indices]

    cells_per_celltype = int(get_config_value(cfg, "inference.cells_per_celltype", 500))
    total_cells = n_cell_types * cells_per_celltype

    # ---- DeconvSC generation: prior-anchored expression + prophead proportions ----
    residual_net, centroids_by_id, core_idx, prop_head, prop_names = train_anchored_residual(
        vae=model,
        priors=priors,
        sc_ref_path=require_reference_path(cfg),
        donor_col=get_config_value(cfg, "generation.donor_col", "sample"),
        celltype_label=get_config_value(cfg, "generation.celltype_label", "cell type"),
        mapping_dict=mapping_dict,
        common_genes=common_genes,
        core_indices=core_genes_to_indices(core_genes, common_genes),
        device=device,
        n_types=n_cell_types,
        epochs=int(get_config_value(cfg, "generation.residual.epochs", 60)),
        n_top_markers=int(get_config_value(cfg, "generation.residual.n_top_markers", 30)),
        lr=float(get_config_value(cfg, "generation.residual.lr", 1e-3)),
        full_mask=bool(get_config_value(cfg, "generation.residual.full_mask", False)),
        prophead_epochs=int(get_config_value(cfg, "generation.prophead.epochs", 120)),
        prophead_K=int(get_config_value(cfg, "generation.prophead.k_pseudobulks", 200)),
        prophead_noise=float(get_config_value(cfg, "generation.prophead.noise_std", 0.2)),
        seed=seed,
    )
    adata_generated, final_z, obs_df = generate_prior_anchored(
        vae=model,
        real_bulk_tensor=input_bulk_log,
        sample_names=valid_sample_names,
        mapping_dict=mapping_dict,
        priors=priors,
        device=device,
        output_dir=str(output_dir),
        common_genes=common_genes,
        residual_net=residual_net,
        centroids_by_id=centroids_by_id,
        core_indices=core_idx,
        prop_head=prop_head,
        prop_names=prop_names,
        alpha=float(get_config_value(cfg, "generation.alpha", 1.0)),
        total_cells=total_cells,
    )

    h5ad_path = save_generated_anndata(
        adata_generated,
        output_dir=output_dir,
        filename=generated_h5ad_name,
    )

    if obs_df is not None:
        obs_df.to_csv(output_dir / generated_metadata_name)

    if final_z is not None:
        torch.save(final_z.detach().cpu() if torch.is_tensor(final_z) else final_z, output_dir / latent_shift_name)

    pd.DataFrame(
        {
            "sample_name": valid_sample_names,
            "sample_index": sample_indices,
        }
    ).to_csv(output_dir / "selected_samples.csv", index=False)

    save_config(cfg, output_dir / "config_used.yaml")

    print("[DeconvSC] Prediction finished.")
    print(f"[DeconvSC] Generated AnnData: {h5ad_path}")
    print(f"[DeconvSC] Output directory: {output_dir}")


if __name__ == "__main__":
    main()