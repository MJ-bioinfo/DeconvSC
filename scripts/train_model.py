# scripts/train_model.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from deconvsc.config import load_config, save_config, get_config_value, update_nested
from deconvsc.io import load_preprocess_bundle
from deconvsc.model import AttentionVAE
import deconvsc.trainer as trainer_mod
from deconvsc.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeconvSC AttentionVAE model.")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config.")

    # Optional overrides
    parser.add_argument("--processed-dir", type=str, default=None, help="Override input.processed_dir.")
    parser.add_argument("--outdir", type=str, default=None, help="Override output.output_dir.")
    parser.add_argument("--device", type=str, default=None, help="Override runtime.device.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override training.learning_rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.processed_dir is not None:
        update_nested(cfg, "input.processed_dir", args.processed_dir)
    if args.outdir is not None:
        update_nested(cfg, "output.output_dir", args.outdir)
    if args.device is not None:
        update_nested(cfg, "runtime.device", args.device)
    if args.epochs is not None:
        update_nested(cfg, "training.epochs", args.epochs)
    if args.batch_size is not None:
        update_nested(cfg, "training.batch_size", args.batch_size)
    if args.lr is not None:
        update_nested(cfg, "training.learning_rate", args.lr)

    processed_dir = get_config_value(cfg, "input.processed_dir")
    metadata_file = get_config_value(cfg, "input.metadata_file", "sc_metadata.pkl")
    output_dir = Path(get_config_value(cfg, "output.output_dir", "output/gse141115/train"))
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
    breed_2_list = metadata.get("breed_2_list", list(mapping_dict.keys()))
    color_map = metadata.get("color_map", {})
    core_genes = metadata["core_genes"]
    core_indices_tensor = metadata["core_indices_tensor"]

    hidden_size_list = get_config_value(cfg, "model.hidden_size_list", [4096, 2048, 1024])
    mid_hidden_size = int(get_config_value(cfg, "model.mid_hidden_size", 256))
    embedding_dim = int(get_config_value(cfg, "model.embedding_dim", 32))
    nhead = int(get_config_value(cfg, "model.nhead", 4))
    num_layers = int(get_config_value(cfg, "model.num_layers", 4))
    ff_dim = int(get_config_value(cfg, "model.ff_dim", 64))

    epochs = int(get_config_value(cfg, "training.epochs", 50))
    batch_size = int(get_config_value(cfg, "training.batch_size", 512))
    learning_rate = float(get_config_value(cfg, "training.learning_rate", 5e-5))

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

    # Compatibility patch for legacy trainer.py that still uses module-level output_dir/device.
    trainer_mod.output_dir = str(output_dir)
    trainer_mod.device = device

    save_config(cfg, output_dir / "config_used.yaml")

    result = trainer_mod.train_vae(
        vae_model=model,
        X_tensor=X_tensor,
        labels=labels,
        real_bulk_log=real_bulk_log,
        used_device=device,
        batch_size=batch_size,
        core_indices_tensor=core_indices_tensor,
        epoch_num=epochs,
        learning_rate=learning_rate,
        hidden_list=hidden_size_list,
        mid_hidden_size=mid_hidden_size,
        num_cell_types=n_cell_types,
        breed_2_list=breed_2_list,
        color_map=color_map,
        seed=seed,
        output_dir=output_dir
    )

    print("[DeconvSC] Training finished.")
    print(f"[DeconvSC] Output directory: {output_dir}")

    if isinstance(result, tuple):
        print("[DeconvSC] train_vae returned tuple. Usually: (best_model, cell_type_mu_logvar).")
    else:
        print("[DeconvSC] train_vae returned object:", type(result))


if __name__ == "__main__":
    main()