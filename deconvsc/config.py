# deconvsc/config.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import yaml


def load_config(config_path: str | Path) -> dict:
    """
    Load YAML config file.

    Example:
        cfg = load_config("configs/train_gse141115.yaml")
        processed_dir = cfg["input"]["processed_dir"]
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return cfg


def save_config(config: Mapping[str, Any], save_path: str | Path) -> None:
    """
    Save config dictionary to YAML.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            dict(config),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def get_config_value(config: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot path.

    Example:
        get_config_value(cfg, "input.sc_path")
        get_config_value(cfg, "training.epochs", 100)
    """
    cur: Any = config
    for key in key_path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def require_config_value(config: Mapping[str, Any], key_path: str) -> Any:
    """
    Get nested config value. Raise error if missing.
    """
    value = get_config_value(config, key_path, default=None)
    if value is None:
        raise KeyError(f"Required config field is missing: {key_path}")
    return value


def update_nested(config: dict, key_path: str, value: Any) -> dict:
    """
    Update nested config value using dot path.

    Example:
        update_nested(cfg, "output.output_dir", "outputs/test")
    """
    keys = key_path.split(".")
    cur = config
    for key in keys[:-1]:
        if key not in cur or cur[key] is None:
            cur[key] = {}
        if not isinstance(cur[key], dict):
            raise TypeError(f"Cannot update nested key under non-dict field: {key}")
        cur = cur[key]
    cur[keys[-1]] = value
    return config