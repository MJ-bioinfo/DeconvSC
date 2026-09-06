#!/usr/bin/env python3
"""Compare two isolated executions of the unmodified legacy within-type script."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = np.load(args.run1 / "module_l2_withinType_dists.npz")
    second = np.load(args.run2 / "module_l2_withinType_dists.npz")
    rows = []
    for key in ("Route2", "BaseVAE", "p", "kept"):
        a = np.asarray(first[key])
        b = np.asarray(second[key])
        rows.append({
            "artifact": "module_l2_withinType_dists.npz",
            "field": key,
            "shape_run1": str(a.shape),
            "shape_run2": str(b.shape),
            "max_abs_difference": float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0,
            "exact_match": bool(np.array_equal(a, b)),
        })

    summary1 = pd.read_csv(args.run1 / "module_l2_withinType_summary.csv")
    summary2 = pd.read_csv(args.run2 / "module_l2_withinType_summary.csv")
    rows.append({
        "artifact": "module_l2_withinType_summary.csv",
        "field": "full_table",
        "shape_run1": str(summary1.shape),
        "shape_run2": str(summary2.shape),
        "max_abs_difference": 0.0,
        "exact_match": bool(summary1.equals(summary2)),
    })
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    if not bool(output["exact_match"].all()):
        raise SystemExit("Legacy bridge reruns differ")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
