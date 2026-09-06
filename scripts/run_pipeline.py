# scripts/run_pipeline.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deconvsc.config import load_config, get_config_value


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeconvSC full pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline YAML config.")
    return parser.parse_args()


def run_cmd(cmd: list[str]):
    print("[DeconvSC] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    python_exec = sys.executable

    if bool(get_config_value(cfg, "steps.preprocess", False)):
        preprocess_cfg = get_config_value(cfg, "configs.preprocess")
        run_cmd([python_exec, "scripts/preprocess_data.py", "--config", preprocess_cfg])

    if bool(get_config_value(cfg, "steps.train", False)):
        train_cfg = get_config_value(cfg, "configs.train")
        run_cmd([python_exec, "scripts/train_model.py", "--config", train_cfg])

    if bool(get_config_value(cfg, "steps.predict", False)):
        predict_cfg = get_config_value(cfg, "configs.predict")
        run_cmd([python_exec, "scripts/predict_bulk.py", "--config", predict_cfg])

    if bool(get_config_value(cfg, "steps.evaluate", False)):
        evaluate_cfg = get_config_value(cfg, "configs.evaluate")
        run_cmd([python_exec, "scripts/evaluate_prediction.py", "--config", evaluate_cfg])

    print("[DeconvSC] Full pipeline finished.")


if __name__ == "__main__":
    main()