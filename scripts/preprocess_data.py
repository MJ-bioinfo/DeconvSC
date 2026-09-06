# scripts/preprocess_data.py

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import deconvsc.preprocess as preprocess_mod
from deconvsc.config import load_config, save_config, get_config_value, update_nested
from deconvsc.io import save_preprocess_bundle
from deconvsc.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq reference and bulk data for DeconvSC.")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocess YAML config.")

    # Optional command-line overrides
    parser.add_argument("--sc-ref", type=str, default=None, help="Override input.sc_path.")
    parser.add_argument("--bulk", type=str, default=None, help="Override input.bulk_path.")
    parser.add_argument("--celltype-key", type=str, default=None, help="Override input.celltype_label.")
    parser.add_argument("--outdir", type=str, default=None, help="Override output.output_dir.")
    return parser.parse_args()


def call_load_sc_data(
    sc_path: str,
    bulk_path: str,
    celltype_label: str,
    output_dir: str,
    sep: str,
):
    """
    Compatible with both old and new load_sc_data signatures.

    Old:
        load_sc_data(sc_data, real_bulk_path, celltype_label=None)

    New recommended:
        load_sc_data(sc_path, bulk_path, celltype_label="cell type",sep, output_dir=...)
    """
    # Patch module-level output_dir if legacy code still uses global output_dir.
    preprocess_mod.output_dir = output_dir

    fn = preprocess_mod.load_sc_data
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {}

    if "celltype_label" in params:
        kwargs["celltype_label"] = celltype_label

    if "output_dir" in params:
        kwargs["output_dir"] = output_dir

    if "sep" in params:
        kwargs["sep"] = sep

    return fn(sc_path, bulk_path, **kwargs)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.sc_ref is not None:
        update_nested(cfg, "input.sc_path", args.sc_ref)
    if args.bulk is not None:
        update_nested(cfg, "input.bulk_path", args.bulk)
    if args.celltype_key is not None:
        update_nested(cfg, "input.celltype_label", args.celltype_key)
    if args.outdir is not None:
        update_nested(cfg, "output.output_dir", args.outdir)

    sc_path = get_config_value(cfg, "input.sc_path")
    bulk_path = get_config_value(cfg, "input.bulk_path")
    celltype_label = get_config_value(cfg, "input.celltype_label", "cell type")
    sep = get_config_value(cfg, "input.sep", "\t")
    output_dir = get_config_value(cfg, "output.output_dir", "data/processed/default")
    seed = int(get_config_value(cfg, "runtime.seed", 18))

    if sc_path is None:
        raise ValueError("Missing config field: input.sc_path")
    if bulk_path is None:
        raise ValueError("Missing config field: input.bulk_path")
    if sep is None:
        raise ValueError("Missing config field: input.sep")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    results = call_load_sc_data(
        sc_path=str(sc_path),
        bulk_path=str(bulk_path),
        celltype_label=str(celltype_label),
        output_dir=str(output_dir),
        sep=sep
    )

    (
        dataset,
        X_tensor,
        labels,
        n_cell_types,
        mapping_dict,
        all_genes,
        real_bulk_log,
        single_cell_matrix,
        cell_number_target_num,
        signature_array,
        color_map,
        bulk_sample_names,
        final_signatures,
        sig_indices,
        core_genes,
        core_indices_tensor,
    ) = results
    breed_2_list = list(mapping_dict.keys())
    sc_metadata = {
        "input_dim": len(all_genes),
        "X_tensor": X_tensor,
        "labels": labels,
        "n_cell_types": n_cell_types,
        "real_bulk_log": real_bulk_log,
        "mapping_dict": mapping_dict,
        "common_genes": all_genes,
        "single_cell_matrix": single_cell_matrix,
        "cell_number_target_num": cell_number_target_num,
        "signature_array": signature_array,
        "breed_2_list": breed_2_list,
        "color_map": color_map,
        "bulk_sample_names": bulk_sample_names,
        "signatures": final_signatures,
        "sig_indices": sig_indices,
        "core_genes": core_genes,
        "core_indices_tensor": core_indices_tensor,
    }

    saved = save_preprocess_bundle(sc_metadata, output_dir=output_dir)
    save_config(cfg, output_dir / "config_used.yaml")

    print("[DeconvSC] Preprocessing finished.")
    print(f"[DeconvSC] Output directory: {output_dir}")
    print(f"[DeconvSC] Saved files: {saved}")


if __name__ == "__main__":
    main()