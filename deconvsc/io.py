# deconvsc/io.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import pickle

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if not exists and return Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_cpu(obj: Any) -> Any:
    """
    Recursively move torch.Tensor objects to CPU before saving.
    This prevents CUDA-device-specific pickle loading errors.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)

    return obj


def _json_safe(obj: Any) -> Any:
    """
    Convert common Python/numpy objects to JSON-serializable objects.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if torch.is_tensor(obj):
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    return str(obj)


def save_preprocess_bundle(
    bundle: Mapping[str, Any],
    output_dir: str | Path,
    pickle_name: str = "sc_metadata.pkl",
    tensor_name: str = "processed_tensors.pt",
    summary_name: str = "preprocess_summary.json",
) -> dict:
    """
    Save preprocessing outputs.

    Main output:
        sc_metadata.pkl

    Extra outputs:
        processed_tensors.pt
        preprocess_summary.json
        common_genes.txt
        core_genes.txt
        signature_genes.txt
        bulk_sample_names.txt

    Expected keys in bundle:
        X_tensor
        labels
        n_cell_types
        real_bulk_log
        mapping_dict
        common_genes
        cell_number_target_num
        signature_array
        breed_2_list
        color_map
        bulk_sample_names
        signatures
        sig_indices
        core_genes
        core_indices_tensor
    """
    output_dir = ensure_dir(output_dir)

    bundle_cpu = _to_cpu(dict(bundle))

    pickle_path = output_dir / pickle_name
    with pickle_path.open("wb") as f:
        pickle.dump(bundle_cpu, f)

    tensor_keys = [
        "X_tensor",
        "labels",
        "real_bulk_log",
        "core_indices_tensor",
    ]
    tensor_payload = {
        k: bundle_cpu[k]
        for k in tensor_keys
        if k in bundle_cpu and torch.is_tensor(bundle_cpu[k])
    }

    tensor_path = output_dir / tensor_name
    if tensor_payload:
        torch.save(tensor_payload, tensor_path)

    def write_list(key: str, filename: str) -> None:
        if key not in bundle_cpu or bundle_cpu[key] is None:
            return
        values = bundle_cpu[key]
        if torch.is_tensor(values):
            values = values.detach().cpu().tolist()
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            return

        with (output_dir / filename).open("w", encoding="utf-8") as f:
            for item in values:
                f.write(str(item) + "\n")

    write_list("common_genes", "common_genes.txt")
    write_list("core_genes", "core_genes.txt")
    write_list("signatures", "signature_genes.txt")
    write_list("bulk_sample_names", "bulk_sample_names.txt")

    summary = {
        "pickle_file": str(pickle_path),
        "tensor_file": str(tensor_path) if tensor_payload else None,
        "keys": sorted(bundle_cpu.keys()),
        "n_cells": int(bundle_cpu["X_tensor"].shape[0]) if "X_tensor" in bundle_cpu else None,
        "n_genes": int(bundle_cpu["X_tensor"].shape[1]) if "X_tensor" in bundle_cpu else None,
        "n_cell_types": int(bundle_cpu["n_cell_types"]) if "n_cell_types" in bundle_cpu else None,
        "n_common_genes": len(bundle_cpu.get("common_genes", [])),
        "n_core_genes": len(bundle_cpu.get("core_genes", [])),
        "n_signature_genes": len(bundle_cpu.get("signatures", [])),
        "mapping_dict": _json_safe(bundle_cpu.get("mapping_dict")),
        "cell_number_target_num": _json_safe(bundle_cpu.get("cell_number_target_num")),
    }

    summary_path = output_dir / summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "pickle_path": str(pickle_path),
        "tensor_path": str(tensor_path) if tensor_payload else None,
        "summary_path": str(summary_path),
    }


def load_preprocess_bundle(
    processed_dir: str | Path,
    pickle_name: str = "sc_metadata.pkl",
    device: str | torch.device | None = None,
) -> dict:
    """
    Load preprocessing bundle saved by save_preprocess_bundle().

    Args:
        processed_dir: directory containing sc_metadata.pkl
        device: optional device to move tensor values to, e.g. "cuda:0" or "cpu"

    Returns:
        dict
    """
    processed_dir = Path(processed_dir)
    pickle_path = processed_dir / pickle_name

    if not pickle_path.exists():
        raise FileNotFoundError(f"Preprocess bundle not found: {pickle_path}")

    with pickle_path.open("rb") as f:
        bundle = pickle.load(f)

    if device is not None:
        device = torch.device(device)
        for k, v in list(bundle.items()):
            if torch.is_tensor(v):
                bundle[k] = v.to(device)

    return bundle


def save_generated_anndata(
    adata,
    output_dir: str | Path,
    filename: str = "generated_data.h5ad",
) -> str:
    """
    Save generated AnnData object.
    """
    output_dir = ensure_dir(output_dir)
    save_path = output_dir / filename

    if adata is None:
        raise ValueError("adata is None. Nothing to save.")

    adata.write_h5ad(save_path)
    return str(save_path)


def save_checkpoint(
    model: torch.nn.Module,
    save_path: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        save_path: full checkpoint path, e.g. outputs/train/scvae_best.pth
        extra: optional metadata, optimizer state, config, epoch, etc.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    payload = {
        "model_state_dict": state_dict,
    }

    if extra is not None:
        payload["extra"] = _to_cpu(dict(extra))

    torch.save(payload, save_path)
    return str(save_path)


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict:
    """
    Remove 'module.' prefix from DataParallel checkpoints if needed.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load checkpoint into model.

    Supports two formats:
        1. plain state_dict
        2. {"model_state_dict": state_dict, "extra": ...}
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model


def load_decode_params(model, checkpoint_path, device="cpu"):
    """Load ONLY the decoder params (decoder_layers / final_layer / label_embedding), strict=False.
    Prior-anchored inference uses only decode(prior_mu) + the residual, so the encoder
    (gene_embedding / transformer / fc_mu|var) is irrelevant and its input_size need NOT match the
    checkpoint (mirrors deconv_20260610's load_vae_decode). This lets the published checkpoint be
    used with any consistent core-gene set for the residual/prophead nets."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device(device)
    sd = torch.load(checkpoint_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = _strip_module_prefix(sd)
    keep = {k: v for k, v in sd.items()
            if k.startswith(("decoder_layers", "final_layer", "label_embedding"))}
    own = model.state_dict()
    bad = [k for k in keep if k not in own or own[k].shape != keep[k].shape]
    assert not bad, f"decode-param shape mismatch vs model: {bad}"
    model.load_state_dict(keep, strict=False)
    model.to(device).eval()
    return model