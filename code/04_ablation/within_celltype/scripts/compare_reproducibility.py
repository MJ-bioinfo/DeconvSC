#!/usr/bin/env python3
"""Compare deterministic numeric artifacts from two within-cell-type runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILES = [
    "input_manifest.csv",
    "input_schema_audit.csv",
    "provenance_checks.csv",
    "architecture_equivalence_audit.csv",
    "protocol_deviations.csv",
    "eligibility_matrix.csv",
    "cell_count_audit.csv",
    "setting_eligibility.csv",
    "within_type_gene_panel.csv",
    "within_type_modules.csv",
    "within_type_metrics_by_seed.csv",
    "within_type_metrics_by_celltype.csv",
    "module_l2_by_celltype_module_seed.csv",
    "hub_jaccard_by_celltype_hub_seed.csv",
    "statistical_tests.csv",
    "sensitivity_analysis.csv",
    "figure4_inputs/topology_metrics_long.csv",
    "figure4_inputs/paired_statistics.csv",
]

NPZ_FILES = [
    "raw_distributions.npz",
    "figure4_inputs/module_l2_withinType_dists.npz",
    "figure4_inputs/within_celltype_panel_data.npz",
]


def compare_csv(reference: Path, rerun: Path, relative: str) -> dict:
    first = pd.read_csv(reference / relative)
    second = pd.read_csv(rerun / relative)
    structure = first.shape == second.shape and list(first.columns) == list(second.columns)
    numeric_difference = np.inf
    nonnumeric_match = False
    if structure:
        numeric_columns = [column for column in first.columns if pd.api.types.is_numeric_dtype(first[column])]
        nonnumeric_columns = [column for column in first.columns if column not in numeric_columns]
        if numeric_columns:
            a = first[numeric_columns].to_numpy(dtype=float)
            b = second[numeric_columns].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            same_missing = np.array_equal(np.isnan(a), np.isnan(b))
            numeric_difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            numeric_match = same_missing and numeric_difference == 0.0
        else:
            numeric_difference = 0.0
            numeric_match = True
        nonnumeric_match = first[nonnumeric_columns].fillna("<NA>").equals(second[nonnumeric_columns].fillna("<NA>"))
    else:
        numeric_match = False
    return {
        "artifact": relative,
        "kind": "csv",
        "reference_shape": str(first.shape),
        "rerun_shape": str(second.shape),
        "structure_match": structure,
        "max_abs_numeric_difference": numeric_difference,
        "nonnumeric_match": nonnumeric_match,
        "passed": bool(structure and numeric_match and nonnumeric_match),
    }


def compare_npz(reference: Path, rerun: Path, relative: str) -> dict:
    first = np.load(reference / relative)
    second = np.load(rerun / relative)
    keys_match = set(first.files) == set(second.files)
    maximum = 0.0
    content_match = keys_match
    if keys_match:
        for key in first.files:
            a, b = np.asarray(first[key]), np.asarray(second[key])
            if a.shape != b.shape:
                content_match = False
                maximum = np.inf
                continue
            if np.issubdtype(a.dtype, np.number):
                finite = np.isfinite(a.astype(float)) & np.isfinite(b.astype(float))
                difference = float(np.max(np.abs(a.astype(float)[finite] - b.astype(float)[finite]))) if finite.any() else 0.0
                maximum = max(maximum, difference)
                if not np.array_equal(a, b, equal_nan=True):
                    content_match = False
            elif not np.array_equal(a, b):
                content_match = False
    return {
        "artifact": relative,
        "kind": "npz",
        "reference_shape": f"{len(first.files)} keys",
        "rerun_shape": f"{len(second.files)} keys",
        "structure_match": keys_match,
        "max_abs_numeric_difference": maximum,
        "nonnumeric_match": content_match,
        "passed": bool(keys_match and content_match),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [compare_csv(args.reference, args.rerun, path) for path in CSV_FILES]
    rows.extend(compare_npz(args.reference, args.rerun, path) for path in NPZ_FILES)
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    if not bool(result["passed"].all()):
        raise SystemExit("Deterministic reproducibility comparison failed")


if __name__ == "__main__":
    main()
