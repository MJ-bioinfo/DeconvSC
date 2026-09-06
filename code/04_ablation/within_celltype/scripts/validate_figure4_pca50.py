#!/usr/bin/env python3
"""Validate Figure 4 after the no-real-split symmetric PCA<=50 update."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_D_METRIC = "coexpression_profile_pcc_pca50"
EXPECTED_DECONV = 0.6784316843
EXPECTED_BASE = 0.3795862929
EXPECTED_SETTING = "pca50_symmetric_lw_no_real_split"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    figures = root / "figures"
    source_path = root / "figure4_inputs/figure4d_pca50_symmetric_profile_pcc.csv"
    audit_path = figures / "Figure4_within_celltype_ABCD_plot_audit.csv"
    caption_path = figures / "caption.md"

    source = pd.read_csv(source_path)
    audit = pd.read_csv(audit_path)
    caption = caption_path.read_text(encoding="utf-8")
    d = audit[audit["panel"] == "d"]
    required_files = []
    for stem in (
        "Figure4_within_celltype_ABCD",
        "panel_d_within_celltype_coexpression_profile_pcc_pca50",
    ):
        for extension in ("png", "pdf", "svg", "emf"):
            required_files.append(figures / f"{stem}.{extension}")

    checks = {
        "source_has_12_rows": len(source) == 12,
        "source_is_no_split_symmetric_pca50": set(source["setting"]) == {EXPECTED_SETTING},
        "source_has_200_genes": set(source["n_genes_mean_over_seeds"]) == {200.0},
        "source_has_ten_seeds": set(source["n_seeds"]) == {10},
        "source_has_no_real_split": set(source["real_cell_split"]) == {False},
        "source_has_no_real_real_comparison": not source["comparison"].str.contains("RealA|RealB").any(),
        "audit_has_one_panel_d": len(d) == 1,
        "panel_d_metric_matches": len(d) == 1 and d.iloc[0]["plot_metric"] == EXPECTED_D_METRIC,
        "panel_d_deconv_mean_matches": len(d) == 1 and close(d.iloc[0]["deconvsc_mean"], EXPECTED_DECONV),
        "panel_d_base_mean_matches": len(d) == 1 and close(d.iloc[0]["basevae_mean"], EXPECTED_BASE),
        "panel_d_deconv_wins_six": len(d) == 1 and int(d.iloc[0]["deconvsc_wins"]) == 6,
        "no_inferential_annotations": bool((~audit["statistical_annotation_displayed"]).all()),
        "all_cell_type_points_displayed": bool(audit["paired_points_displayed"].all()),
        "no_cell_type_connecting_lines": bool((~audit["paired_lines_displayed"]).all()),
        "caption_states_symmetric_processing": "same PCA projection and Ledoit–Wolf estimator were applied" in caption,
        "caption_states_pca_cap": "PCA ≤ 50 rather than PCA80" in caption,
        "caption_states_no_real_split": "real cells were not divided into reference subsets" in caption,
        "caption_states_points_unconnected": "points are left unconnected" in caption,
        "caption_omits_real_real_reference": "real–real reference" not in caption and "RealA" not in caption and "RealB" not in caption,
        "all_required_figure_files_exist": all(path.is_file() and path.stat().st_size > 0 for path in required_files),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Figure 4 PCA<=50 validation failed: {checks}")

    report = {
        "checks": checks,
        "panel_d": {
            "metric": EXPECTED_D_METRIC,
            "panel_size": 200,
            "pca_setting_internal_label": EXPECTED_SETTING,
            "display_label": "PCA <= 50 applied to real and generated",
            "deconvsc_mean": float(d.iloc[0]["deconvsc_mean"]),
            "ablated_vae_mean": float(d.iloc[0]["basevae_mean"]),
            "real_cell_split": False,
            "deconvsc_wins": int(d.iloc[0]["deconvsc_wins"]),
            "n_cell_types": int(d.iloc[0]["n_cell_types"]),
        },
    }
    (figures / "Figure4_pca50_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_rows = [
        {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in required_files + [source_path, audit_path, caption_path]
    ]
    pd.DataFrame(manifest_rows).to_csv(
        figures / "Figure4_pca50_output_manifest.csv", index=False
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
