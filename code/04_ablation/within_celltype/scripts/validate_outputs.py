#!/usr/bin/env python3
"""Final integrity/statistical-unit validator for within-cell-type outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_TYPES = {"CD_PC", "CNT", "DCT", "MC", "PT", "aLOH"}
EXPECTED_SETTINGS = {
    "P0_primary", "S1_current", "S2a_threshold50", "S2b_threshold200",
    "S3_no_donor_centering", "S4_shared_global_panel", "S5_legacy_ward",
    "S6_donor_fisher", "S7_native_decoder_scale", "LODO_without_LDK1",
    "LODO_without_LDK2", "LODO_without_LDK3",
}

REQUIRED = [
    "plan.md", "analysis_report_zh.md", "experiment_record.md", "validation_report.md",
    "analysis_config.json", "run_metadata.json", "input_manifest.csv", "input_schema_audit.csv",
    "provenance_checks.csv", "architecture_equivalence_audit.csv", "protocol_deviations.csv",
    "eligibility_matrix.csv", "cell_count_audit.csv", "setting_eligibility.csv",
    "within_type_gene_panel.csv", "within_type_modules.csv", "within_type_metrics_by_seed.csv",
    "within_type_metrics_by_celltype.csv", "module_l2_by_celltype_module_seed.csv",
    "hub_jaccard_by_celltype_hub_seed.csv", "statistical_tests.csv", "sensitivity_analysis.csv",
    "raw_distributions.npz", "legacy_bridge_reproduction.csv", "reproducibility_comparison.csv",
    "scripts/run_within_celltype_topology.py", "scripts/compare_legacy_bridge.py",
    "scripts/compare_reproducibility.py", "scripts/validate_outputs.py",
    "scripts/plot_figure4_within_celltype.py",
    "logs/main_analysis.log", "logs/main_analysis_native_initial.log", "logs/repro_run.log",
    "logs/plot_figure4_within_celltype.log",
    "figures/primary_topology_paired.png", "figures/primary_topology_paired.pdf",
    "figures/primary_topology_paired.svg", "figures/primary_module_l2_paired.png",
    "figures/primary_module_l2_paired.pdf", "figures/primary_module_l2_paired.svg",
    "figure4_inputs/README.md", "figure4_inputs/figure4_input_manifest.csv",
    "figure4_inputs/topology_metrics_long.csv", "figure4_inputs/paired_statistics.csv",
    "figure4_inputs/module_l2_withinType_dists.npz",
    "figure4_inputs/within_celltype_panel_data.npz",
    "figures/Figure4_within_celltype_ABCD.png", "figures/Figure4_within_celltype_ABCD.pdf",
    "figures/Figure4_within_celltype_ABCD.svg", "figures/Figure4_within_celltype_ABCD.emf",
    "figures/Figure4_within_celltype_ABCD_plot_audit.csv",
    "figures/Figure4_within_celltype_ABCD_caption.md",
    "figures/panel_a_within_celltype_hub_jaccard.png",
    "figures/panel_a_within_celltype_hub_jaccard.pdf",
    "figures/panel_a_within_celltype_hub_jaccard.svg",
    "figures/panel_a_within_celltype_hub_jaccard.emf",
    "figures/panel_b_within_celltype_mi_fidelity.png",
    "figures/panel_b_within_celltype_mi_fidelity.pdf",
    "figures/panel_b_within_celltype_mi_fidelity.svg",
    "figures/panel_b_within_celltype_mi_fidelity.emf",
    "figures/panel_c_within_celltype_module_l2.png",
    "figures/panel_c_within_celltype_module_l2.pdf",
    "figures/panel_c_within_celltype_module_l2.svg",
    "figures/panel_c_within_celltype_module_l2.emf",
    "figures/panel_d_within_celltype_coexpression_edge_mae.png",
    "figures/panel_d_within_celltype_coexpression_edge_mae.pdf",
    "figures/panel_d_within_celltype_coexpression_edge_mae.svg",
    "figures/panel_d_within_celltype_coexpression_edge_mae.emf",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.outdir.resolve()
    checks: list[dict] = []

    def add(check: str, passed: bool, observed, expected) -> None:
        checks.append({
            "check": check,
            "passed": bool(passed),
            "observed": str(observed),
            "expected": str(expected),
        })

    missing = [path for path in REQUIRED if not (root / path).exists()]
    empty = [path for path in REQUIRED if (root / path).exists() and (root / path).stat().st_size == 0]
    add("all_required_files_exist", not missing, ";".join(missing) if missing else "none", "none missing")
    add("all_required_files_nonempty", not empty, ";".join(empty) if empty else "none", "none empty")

    metadata = json.loads((root / "run_metadata.json").read_text())
    config = json.loads((root / "analysis_config.json").read_text())
    add("run_completed", metadata.get("status") == "completed", metadata.get("status"), "completed")
    add("common_gene_count", config.get("common_gene_count") == 16791, config.get("common_gene_count"), 16791)
    add("inferential_unit", config.get("inferential_unit") == "cell_type", config.get("inferential_unit"), "cell_type")

    provenance = pd.read_csv(root / "provenance_checks.csv")
    add("figure3_provenance_passed", bool(provenance["passed"].all()), provenance["max_abs_difference"].max(), "<1e-5")
    add("figure3_provenance_profiles", int(provenance["n_profiles"].iloc[0]) == 48, provenance["n_profiles"].iloc[0], 48)

    manifest = pd.read_csv(root / "input_manifest.csv")
    hash_mismatches = []
    for _, row in manifest.iterrows():
        path = Path(row["path"])
        if not path.exists() or sha256_file(path) != row["sha256"]:
            hash_mismatches.append(row["name"])
    add("input_hashes_unchanged", not hash_mismatches, ";".join(hash_mismatches) if hash_mismatches else "none", "none")

    schema = pd.read_csv(root / "input_schema_audit.csv")
    generated = schema[schema["method"].isin(["DeconvSC", "BaseVAE", "DeconvSC_current"])]
    post = generated["post_transform_expm1_library_median"].to_numpy(dtype=float)
    add("generated_cp10k_transform", bool(np.all(np.abs(post - 10000.0) < 1e-2)), post.tolist(), "all within 0.01 of 10000")
    native = generated.set_index("method")["expm1_library_median_common_panel"]
    add("native_scale_assumption_falsified", bool((native < 5000).all()), native.to_dict(), "all generated native libraries <5000")

    architecture = pd.read_csv(root / "architecture_equivalence_audit.csv")
    attribution_limited = architecture["status"].isin(["UNVERIFIED", "NOT_EQUIVALENT", "NOT_CP10K"]).any()
    add("attention_attribution_limited", attribution_limited, sorted(architecture["status"].unique()), "contains unverified/not-equivalent status")
    deviation = pd.read_csv(root / "protocol_deviations.csv")
    scale_deviation = deviation[deviation["gate"] == "Gate 0 expression-scale audit"]
    panel_deviation = deviation[deviation["gate"] == "Figure 4d display-endpoint revision"]
    add("scale_protocol_deviation_recorded", len(scale_deviation) == 1 and bool(scale_deviation["result_seen_before_resolution"].iloc[0]), len(scale_deviation), "one transparent scale record")
    add("panel_d_post_result_revision_recorded", len(panel_deviation) == 1 and bool(panel_deviation["result_seen_before_resolution"].iloc[0]), len(panel_deviation), "one transparent Figure 4d record")

    eligible_methods = pd.read_csv(root / "eligibility_matrix.csv")
    allowed = set(eligible_methods[eligible_methods["within_celltype_eligible"]]["method"])
    add("method_eligibility", allowed == {"GT", "DeconvSC Figure3", "BaseVAE", "DeconvSC current"}, sorted(allowed), "four single-cell sources")

    eligibility = pd.read_csv(root / "setting_eligibility.csv")
    primary_eligible = set(eligibility[(eligibility["setting"] == "P0_primary") & eligibility["eligible"]]["cell_type"])
    add("primary_cell_types_locked", primary_eligible == PRIMARY_TYPES, sorted(primary_eligible), sorted(PRIMARY_TYPES))
    p0_audit = eligibility[(eligibility["setting"] == "P0_primary") & eligibility["eligible"]]
    add("primary_three_valid_donors", bool((p0_audit["n_valid_donors"] == 3).all()), p0_audit["n_valid_donors"].tolist(), "all 3")
    add("threshold50_type_count", int(eligibility[(eligibility["setting"] == "S2a_threshold50") & eligibility["eligible"]].shape[0]) == 10, eligibility[(eligibility["setting"] == "S2a_threshold50") & eligibility["eligible"]].shape[0], 10)
    add("threshold200_type_count", int(eligibility[(eligibility["setting"] == "S2b_threshold200") & eligibility["eligible"]].shape[0]) == 4, eligibility[(eligibility["setting"] == "S2b_threshold200") & eligibility["eligible"]].shape[0], 4)

    panels = pd.read_csv(root / "within_type_gene_panel.csv")
    cell_panels = panels[panels["panel"] == "celltype_specific"].groupby("cell_type").size()
    add("celltype_panels_have_500_genes", bool((cell_panels == 500).all()), cell_panels.to_dict(), "500 each")
    modules = pd.read_csv(root / "within_type_modules.csv")
    add("modules_gt_defined_nonempty", bool((modules["module_size"] >= 6).all() and len(modules) > 0), f"n={len(modules)}, min={modules['module_size'].min()}", "nonempty; size>=6")

    by_seed = pd.read_csv(root / "within_type_metrics_by_seed.csv")
    p0_seed = by_seed[by_seed["setting"] == "P0_primary"]
    seed_counts = p0_seed.groupby(["cell_type", "method"])["seed"].nunique()
    add("primary_ten_seeds_per_pair", bool((seed_counts == 10).all()), seed_counts.to_dict(), "10 each")
    add("primary_methods_only", set(p0_seed["method"]) == {"DeconvSC", "BaseVAE"}, sorted(p0_seed["method"].unique()), "DeconvSC;BaseVAE")
    finite_columns = [
        "module_l2_mean", "module_rmse_mean", "module_edge_mae_mean", "hub_jaccard_mean",
        "mi_fidelity_error_median", "mi_rank10_error_median", "network_pcc",
        "per_gene_network_fidelity_median", "per_gene_coexpression_edge_mae_median",
    ]
    add("primary_metrics_finite", bool(np.isfinite(p0_seed[finite_columns].to_numpy(dtype=float)).all()), int(np.isfinite(p0_seed[finite_columns].to_numpy(dtype=float)).sum()), p0_seed[finite_columns].size)

    by_cell = pd.read_csv(root / "within_type_metrics_by_celltype.csv")
    primary_cell = by_cell[by_cell["setting"] == "P0_primary"]
    add("primary_cell_rows", len(primary_cell) == 12, len(primary_cell), 12)
    add("no_aggregate_methods_in_metrics", not set(by_cell["method"]) & {"DISSECT", "BayesPrism", "TAPE", "CIBERSORTx"}, sorted(set(by_cell["method"])), "only DeconvSC/BaseVAE")

    tests = pd.read_csv(root / "statistical_tests.csv")
    module_test = tests[tests["metric"] == "module_l2_mean"].iloc[0]
    add("primary_test_celltype_n", int(module_test["n_cell_types"]) == 6, module_test["n_cell_types"], 6)
    add("primary_all_celltypes_favor_deconv", float(module_test["deconvsc_win_fraction"]) == 1.0, module_test["deconvsc_win_fraction"], 1.0)
    add("primary_exact_one_sided_p", abs(float(module_test["p_one_sided"]) - 0.015625) < 1e-15, module_test["p_one_sided"], 0.015625)
    add("primary_two_sided_p", abs(float(module_test["p_two_sided"]) - 0.03125) < 1e-15, module_test["p_two_sided"], 0.03125)
    add("primary_effect_ci_positive", float(module_test["advantage_bootstrap_ci_low"]) > 0, module_test["advantage_bootstrap_ci_low"], ">0")
    scale_tests = tests[tests["metric"].isin(["module_rmse_mean", "module_edge_mae_mean"])]
    add("module_scale_sensitivities_agree", bool((scale_tests["deconvsc_win_fraction"] == 1.0).all() and (scale_tests["p_one_sided"] == 0.015625).all()), scale_tests[["metric", "deconvsc_win_fraction", "p_one_sided"]].to_dict("records"), "both 6/6 and p=0.015625")
    figure4_tests = tests[tests["figure4_metric"].astype(str).str.lower().eq("true")]
    add("figure4_holm_family_complete", len(figure4_tests) == 4 and figure4_tests["p_one_sided_holm"].notna().all(), len(figure4_tests), 4)
    add("holm_not_overclaimed", bool((figure4_tests["p_one_sided_holm"] >= 0.05).all()), figure4_tests["p_one_sided_holm"].tolist(), "all >=0.05")
    edge_test = tests[tests["metric"] == "per_gene_coexpression_edge_mae_median"].iloc[0]
    add("per_gene_edge_mae_all_celltypes_favor_deconv", float(edge_test["deconvsc_win_fraction"]) == 1.0, edge_test["deconvsc_win_fraction"], 1.0)
    rank_mi = tests[tests["metric"] == "mi_rank10_error_median"].iloc[0]
    add("mi_rank_sensitivity_agrees", float(rank_mi["deconvsc_win_fraction"]) == 1.0, rank_mi["deconvsc_win_fraction"], 1.0)

    sensitivity = pd.read_csv(root / "sensitivity_analysis.csv")
    add("sensitivity_settings_complete", set(sensitivity["setting"]) == EXPECTED_SETTINGS, sorted(sensitivity["setting"]), sorted(EXPECTED_SETTINGS))
    add("all_module_sensitivities_directionally_agree", bool((sensitivity["deconvsc_win_fraction"] == 1.0).all()), sensitivity.set_index("setting")["deconvsc_win_fraction"].to_dict(), "all 1.0")
    lodo = sensitivity[sensitivity["setting"].str.startswith("LODO_")]
    add("leave_one_donor_out_complete", len(lodo) == 3 and (lodo["deconvsc_win_fraction"] == 1.0).all(), len(lodo), "3 stable runs")
    add("native_scale_sensitivity_present", "S7_native_decoder_scale" in set(sensitivity["setting"]), sorted(sensitivity["setting"]), "contains S7")

    legacy = pd.read_csv(root / "legacy_bridge_reproduction.csv")
    add("legacy_bridge_exact", len(legacy) == 5 and bool(legacy["exact_match"].all()), f"{legacy['exact_match'].sum()}/{len(legacy)}", "5/5")
    old_npz = np.load(root / "cache/legacy_bridge_run1/module_l2_withinType_dists.npz")
    add("legacy_bridge_scope", int(old_npz["kept"]) == 11 and len(old_npz["Route2"]) == 50, f"kept={int(old_npz['kept'])}, n={len(old_npz['Route2'])}", "kept=11, module×seed=50")

    reproduction = pd.read_csv(root / "reproducibility_comparison.csv")
    add("deterministic_reproduction", len(reproduction) == 21 and bool(reproduction["passed"].all()), f"{reproduction['passed'].sum()}/{len(reproduction)}", "21/21")
    add("reproduction_numeric_exact", float(reproduction["max_abs_numeric_difference"].max()) == 0.0, reproduction["max_abs_numeric_difference"].max(), 0.0)

    figure_long = pd.read_csv(root / "figure4_inputs/topology_metrics_long.csv")
    add("figure4_long_rows", len(figure_long) == 72, len(figure_long), "6 types × 2 methods × 6 metrics = 72")
    add("figure4_edge_mae_present", "coexpression_edge_mae" in set(figure_long["metric"]), sorted(figure_long["metric"].unique()), "contains coexpression_edge_mae")
    add("figure4_unit_celltype", set(figure_long["unit"]) == {"cell_type"}, sorted(figure_long["unit"].unique()), "cell_type")
    add("figure4_methods", set(figure_long["method"]) == {"Route2", "BaseVAE"}, sorted(figure_long["method"].unique()), "Route2;BaseVAE")
    add("figure4_no_dissect", "DISSECT" not in set(figure_long["method"]), sorted(figure_long["method"].unique()), "DISSECT absent")
    panel = np.load(root / "figure4_inputs/module_l2_withinType_dists.npz")
    expected_keys = {"Route2", "BaseVAE", "cell_types", "p", "p_two_sided", "unit", "value_definition"}
    add("figure4_panel_c_keys", set(panel.files) == expected_keys, sorted(panel.files), sorted(expected_keys))
    add("figure4_panel_c_n", len(panel["Route2"]) == 6 and len(panel["BaseVAE"]) == 6, (len(panel["Route2"]), len(panel["BaseVAE"])), (6, 6))
    add("figure4_panel_c_p_matches", abs(float(panel["p"]) - float(module_test["p_one_sided"])) < 1e-15, float(panel["p"]), module_test["p_one_sided"])
    primary_pivot = primary_cell.pivot(index="cell_type", columns="method", values="module_l2_mean").sort_index()
    panel_order = panel["cell_types"].astype(str)
    route_diff = float(np.max(np.abs(panel["Route2"] - primary_pivot["DeconvSC"].to_numpy())))
    base_diff = float(np.max(np.abs(panel["BaseVAE"] - primary_pivot["BaseVAE"].to_numpy())))
    arrays_match = np.array_equal(panel_order, primary_pivot.index.to_numpy(dtype=str)) and route_diff < 1e-15 and base_diff < 1e-15
    add("figure4_panel_c_values_match", arrays_match, f"max_diff={max(route_diff, base_diff):.3e}; order={panel_order.tolist()}", "cross-format max diff <1e-15 and identical order")

    plot_audit = pd.read_csv(root / "figures/Figure4_within_celltype_ABCD_plot_audit.csv")
    layer_columns = [
        "density_layer_displayed", "boxplot_displayed",
        "paired_points_displayed", "paired_lines_displayed",
    ]
    layers_present = all(column in plot_audit.columns for column in layer_columns)
    layers_valid = layers_present and (
        plot_audit["density_layer_displayed"].astype(str).str.lower().eq("false").all()
        and plot_audit["boxplot_displayed"].astype(str).str.lower().eq("true").all()
        and plot_audit["paired_points_displayed"].astype(str).str.lower().eq("true").all()
        and plot_audit["paired_lines_displayed"].astype(str).str.lower().eq("true").all()
    )
    panels_valid = (
        plot_audit["panel"].tolist() == ["a", "b", "c", "d"]
        and bool((plot_audit["n_cell_types"] == 6).all())
        and bool(layers_valid)
    )
    add(
        "figure4_abcd_panels_complete",
        panels_valid,
        plot_audit[["panel", "n_cell_types"] + layer_columns].to_dict("records") if layers_present else plot_audit.columns.tolist(),
        "a-d; n=6 each; box+points+paired lines; no density layer",
    )
    expected_means = figure_long.groupby(["metric", "method"])["value"].mean()
    mean_match = True
    for _, row in plot_audit.iterrows():
        mean_match &= abs(float(row["deconvsc_mean"]) - float(expected_means.loc[(row["plot_metric"], "Route2")])) < 1e-15
        mean_match &= abs(float(row["basevae_mean"]) - float(expected_means.loc[(row["plot_metric"], "BaseVAE")])) < 1e-15
    add("figure4_abcd_values_match_source", mean_match, "max cross-table tolerance checked", "all means differ <1e-15")
    add("figure4_panel_d_is_edge_mae", plot_audit.loc[plot_audit["panel"] == "d", "plot_metric"].tolist() == ["coexpression_edge_mae"], plot_audit.loc[plot_audit["panel"] == "d", "plot_metric"].tolist(), "coexpression_edge_mae")
    svg_text = (root / "figures/Figure4_within_celltype_ABCD.svg").read_text(encoding="utf-8")
    annotation_flags = plot_audit.get("statistical_annotation_displayed", pd.Series(dtype=object))
    audit_has_no_annotations = len(annotation_flags) == 4 and annotation_flags.astype(str).str.lower().eq("false").all()
    forbidden_svg_labels = [label for label in ("P$_", "P = ", ">ns</text>", ">*</text>") if label in svg_text]
    add(
        "figure4_abcd_no_stat_annotations",
        audit_has_no_annotations and not forbidden_svg_labels,
        {"audit_all_false": bool(audit_has_no_annotations), "forbidden_svg_labels": forbidden_svg_labels},
        "all panels false; no bracket-test text (P, ns, or stars) in combined SVG",
    )
    add("figure4_abcd_arial_vector_text", "font-family: 'Arial'" in svg_text and "DejaVu Sans" not in svg_text and "Liberation Sans" not in svg_text, "Arial" if "font-family: 'Arial'" in svg_text else "missing", "editable Arial text")

    analysis_text = (root / "analysis_report_zh.md").read_text(encoding="utf-8")
    validation_text = (root / "validation_report.md").read_text(encoding="utf-8")
    caption_text = (root / "figures/Figure4_within_celltype_ABCD_caption.md").read_text(encoding="utf-8")
    add("report_claim_boundary", "不能单独归因为 attention" in analysis_text and "GRN" in analysis_text, "present" if "GRN" in analysis_text else "missing", "attention and GRN boundaries present")
    add("figure4_caption_edge_mae_and_complementarity", "Per-gene co-expression edge mean absolute error" in caption_text and "complementary views of topology" in caption_text, "present" if "complementary views of topology" in caption_text else "missing", "edge MAE definition and complementary roles present")
    add("fallacy_scan_11_of_11", "11/11" in validation_text, "11/11" if "11/11" in validation_text else "missing", "11/11")

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(root / "validation_checks.csv", index=False)

    output_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "cache" in path.relative_to(root).parts or path.name == "output_manifest.csv":
            continue
        output_rows.append({
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "suffix": path.suffix,
        })
    pd.DataFrame(output_rows).to_csv(root / "output_manifest.csv", index=False)

    print(checks_df.to_string(index=False))
    passed = int(checks_df["passed"].sum())
    print(f"Validation passed: {passed}/{len(checks_df)} checks")
    if passed != len(checks_df):
        raise SystemExit("Validation failed")


if __name__ == "__main__":
    main()
