#!/usr/bin/env python3
"""Validate the portable DeconvSC Figshare release and locked endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "VERSION",
    "CITATION.cff",
    "LICENSE.txt",
    "environment/conda_environment.yml",
    "environment/requirements.txt",
    "environment/session_info.txt",
    "processed_data/real_data/mouse_kidney/train_data_canonical20.h5ad",
    "processed_data/real_data/mouse_kidney/test_data.h5ad",
    "processed_data/real_data/human_lung/train_data.h5ad",
    "processed_data/simulation_data/HCA/reference.h5ad",
    "model_weights/GSE141115/deconvsc/scvae_best.pth",
    "model_weights/GSE141115/deconvsc/cell_type_mu_logvar_best.pt",
    "model_outputs/GSE141115/prophead_generated.h5ad",
    "model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad",
    "model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad",
    "model_outputs/GSE159585/prophead_generated.h5ad",
    "model_outputs/HCA/prophead_generated.h5ad",
    "model_outputs/GSE226077/prophead_generated.h5ad",
    "benchmark_tables/expression_prediction/03_result_tables/summary_all_methods.csv",
    "benchmark_tables/expression_prediction/03_result_tables/benchmark_tables.xlsx",
    "results/ablation/within_celltype/figure4_inputs/topology_metrics_long.csv",
    "results/ablation/within_celltype/figure4_inputs/figure4d_pca50_symmetric_profile_pcc.csv",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD.png",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD.pdf",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD.svg",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD.emf",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD_plot_audit.csv",
    "results/supplementary_figure5/data/metric_summary.csv",
    "results/downstream/GSE159585/cellchat/comparison_df.csv",
    "results/downstream/GSE226077/BEAM_res.csv",
    "provenance/DATA_DICTIONARY.tsv",
    "provenance/FIGURE_SOURCE_MAP.tsv",
    "provenance/ARTIFACT_SOURCE_MAP.tsv",
]

EXPECTED_SHA256 = {
    "model_outputs/GSE141115/prophead_generated.h5ad": "cac25d4f0fd3a4e9df2a80e36f9182b40aaedd50d0d839bd0255b441eaa7694c",
    "model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad": "a78885bd8f53c1fbfee611f62844033f669af77bd245273e986d001a1817af41",
    "model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad": "78eb9a7b675195ade922f5bb122e8760ffebbca44ccd596d1dbddd45e6f08429",
    "model_outputs/GSE159585/prophead_generated.h5ad": "bd78d0ea785b64912ce48b911d0e4f78f7a1e07e9d21702ac3115ff355ea6d9f",
    "model_outputs/HCA/prophead_generated.h5ad": "617c28bc0e4fd52636905fde726d2e0ad175a53b292094ac79aaa6bf23c68f22",
    "model_outputs/GSE226077/prophead_generated.h5ad": "fdf4f1708105c79d46c413025b07b3b55c450b096938517d3a85061e86313841",
    "model_weights/GSE141115/deconvsc/scvae_best.pth": "296248e289267fb44aeddb49bf7a255c410d33babad28cffbbe6d040ea9d01c7",
    "model_weights/GSE159585/benchmark_reference/scvae_best.pth": "0b2c50608194b49409fc931afef24afbfd62faf574fd8e227748b6f6d41836dc",
    "model_weights/GSE159585/normal_application/scvae_best.pth": "30e38f796630640a75dcf1d0e1f4dee4dd016b0a6a5b36dd0a3fc679794620ba",
    "model_weights/GSE159585/covid_application/scvae_best.pth": "5423e4f1af73d0da49d0a2c92c2b4d10d3431a7d8371369e74979cd64562fae4",
    "model_weights/HCA/fold2/scvae_best.pth": "6cf71c87c5ebfd4c88f39607224955ae3eb8b496b65d2a31763aaf392b5f1efe",
    "model_weights/GSE226077/application/scvae_best.pth": "581cc81e5af4c8b8bf0efec34962b0f58d60942ad258ea94e1e43b2257859f38",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD.png": "d4f0d271086d8796e4953e03a414d2b1b401baf135840dce98d51c2438d34467",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD.pdf": "3cf82e9106cd288907f2616340a855ae7f1a079cee6803bb5091d17458e77b60",
    "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD.svg": "e261361d68231b35643f286424879cce978dcc2b312440aa8e19b21985b3aea1",
}

EXECUTABLE_SUFFIXES = {".py", ".R", ".r", ".sh", ".yaml", ".yml"}
ABSOLUTE_PATTERNS = [
    re.compile("/" + "disk1/maijl"),
    re.compile(r"[A-Za-z]:[/\\\\]Research"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks, name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})


def check_h5ad(checks, relative, expected_shape, required_obs=()):
    path = ROOT / relative
    try:
        obj = ad.read_h5ad(path, backed="r")
        missing = [column for column in required_obs if column not in obj.obs.columns]
        passed = tuple(obj.shape) == tuple(expected_shape) and not missing
        add(checks, f"h5ad:{relative}", passed,
            f"shape={obj.shape}; expected={expected_shape}; missing_obs={missing}")
        obj.file.close()
    except Exception as exc:
        add(checks, f"h5ad:{relative}", False, repr(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--skip-locked-hashes", action="store_true")
    args = parser.parse_args()
    checks = []

    for relative in REQUIRED:
        path = ROOT / relative
        add(checks, f"required:{relative}", path.is_file() and path.stat().st_size > 0,
            f"exists={path.is_file()}; size={path.stat().st_size if path.exists() else 0}")

    for path in sorted((ROOT / "configs").glob("*.yaml")):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            add(checks, f"yaml:{path.name}", isinstance(value, dict), "parsed")
        except Exception as exc:
            add(checks, f"yaml:{path.name}", False, repr(exc))

    offenders = []
    for base in (ROOT / "code", ROOT / "deconvsc", ROOT / "scripts", ROOT / "configs"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in EXECUTABLE_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in ABSOLUTE_PATTERNS):
                offenders.append(str(path.relative_to(ROOT)))
    add(checks, "portable_executable_paths", not offenders,
        offenders or "no workstation-specific executable paths")

    check_h5ad(checks, "processed_data/real_data/mouse_kidney/test_data.h5ad",
               (16947, 17328), ("sample", "cell type"))
    check_h5ad(checks, "model_outputs/GSE141115/prophead_generated.h5ad",
               (9598, 16801), ("Sample", "Cell_type"))
    check_h5ad(checks, "model_outputs/GSE141115/figure4_suppfigure5_deconvsc_generated.h5ad",
               (9934, 16801), ("Sample", "Cell_type"))
    check_h5ad(checks, "model_outputs/GSE141115/figure4_ablated_vae_generated.h5ad",
               (19936, 16792), ("Sample", "Cell_type"))
    check_h5ad(checks, "model_outputs/GSE159585/prophead_generated.h5ad",
               (13986, 12220), ("Sample", "Cell_type"))
    check_h5ad(checks, "model_outputs/HCA/prophead_generated.h5ad",
               (20943, 25366), ("pseudo_bulk_id", "Cell_type"))
    check_h5ad(checks, "model_outputs/GSE226077/prophead_generated.h5ad",
               (4800, 16129), ("Sample", "Cell_type", "development_stage"))

    benchmark = pd.read_csv(ROOT / "benchmark_tables/expression_prediction/03_result_tables/summary_all_methods.csv")
    methods = set(benchmark["method"].astype(str))
    expected_methods = {"DeconvSC", "BayesPrism", "CIBERSORTx", "DISSECT", "TAPE"}
    add(checks, "benchmark_methods", methods == expected_methods,
        f"rows={len(benchmark)}; methods={sorted(methods)}")
    add(checks, "benchmark_datasets",
        set(benchmark["dataset"].astype(str)) == {"GSE141115", "GSE159585", "HCA_fold2"},
        sorted(benchmark["dataset"].astype(str).unique()))

    fig4 = pd.read_csv(ROOT / "results/ablation/within_celltype/figure4_inputs/topology_metrics_long.csv")
    fig4_primary = fig4[fig4["analysis"].astype(str) == "within_celltype_P0"]
    add(checks, "figure4_methods", set(fig4_primary["method"].astype(str)) == {"Route2", "BaseVAE"},
        sorted(fig4_primary["method"].astype(str).unique()))
    add(checks, "figure4_cell_types", fig4_primary["cell_type"].nunique() == 6,
        sorted(fig4_primary["cell_type"].astype(str).unique()))
    audit = pd.read_csv(ROOT / "results/ablation/within_celltype/figures/Figure4_within_celltype_ABCD_plot_audit.csv")
    no_lines = (~audit["paired_lines_displayed"].astype(bool)).all()
    no_density = (~audit["density_layer_displayed"].astype(bool)).all()
    no_stats = (~audit["statistical_annotation_displayed"].astype(bool)).all()
    add(checks, "figure4_current_display", no_lines and no_density and no_stats,
        f"no_lines={no_lines}; no_density={no_density}; no_stats={no_stats}")
    legacy_figures = sorted(p.name for p in (ROOT / "results/ablation/within_celltype/figures").glob("*violin*"))
    add(checks, "figure4_no_superseded_violin", not legacy_figures, legacy_figures or "none")

    if not args.skip_locked_hashes:
        for relative, expected in EXPECTED_SHA256.items():
            observed = sha256_file(ROOT / relative)
            add(checks, f"sha256:{relative}", observed == expected,
                f"observed={observed}; expected={expected}")

    report = {
        "release_root": str(ROOT),
        "n_checks": len(checks),
        "n_passed": sum(item["passed"] for item in checks),
        "n_failed": sum(not item["passed"] for item in checks),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write_report:
        out = ROOT / "provenance/validation_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
