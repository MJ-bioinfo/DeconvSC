#!/usr/bin/env python3
"""Render the Figure-4-style 2x2 within-cell-type topology figure.

The visual grammar follows the existing Figure 4 (Arial, warm-pink DeconvSC
and cool-blue ablation). The script writes box plots with all six cell-type
observations shown as unconnected points; no density layer is generated.
Inferential annotations are intentionally not drawn; validated statistical
results remain in separate tables. Panels a-c use the established within-cell-
type benchmark table. Panel d uses the independently computed, training-defined
200-gene edge-profile analysis in which the same capped PCA basis (PCA <= 50)
and Ledoit-Wolf estimator are applied to real and generated expression. Real
cells are not split; the donor-stratified cell-count-balancing rule is the same
as that used for panels a-c.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RELEASE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RELEASE_ROOT / "code/03_model_evaluation/support"))
import plot_style  # noqa: E402  applies the same Arial export style as original Figure 4


PALETTE = {"Route2": "#D8B2AE", "BaseVAE": "#7187A7"}
DISPLAY = {"Route2": "DeconvSC", "BaseVAE": "Ablated VAE"}
ORDER = ["Route2", "BaseVAE"]
EXPECTED_TYPES = ["CD_PC", "CNT", "DCT", "MC", "PT", "aLOH"]
PLOT_STYLES = ("boxplot",)
EDGE_PROFILE_SETTING = "pca50_symmetric_lw_no_real_split"
EDGE_PROFILE_PANEL_SIZE = 200
EDGE_PROFILE_SOURCE_METRIC = "mean_over_seeds"
EDGE_PROFILE_PLOT_METRIC = "coexpression_profile_pcc_pca50"
EDGE_PROFILE_METHOD_TO_PLOT_METHOD = {"DeconvSC": "Route2", "BaseVAE": "BaseVAE"}

PANEL_SPECS = [
    {
        "letter": "a",
        "plot_metric": "hub_jaccard",
        "ylabel": "Hub-neighbour Jaccard",
        "higher_better": True,
        "filename": "panel_a_within_celltype_hub_jaccard",
    },
    {
        "letter": "b",
        "plot_metric": "mi_fidelity_error",
        "ylabel": "MI fidelity error",
        "higher_better": False,
        "filename": "panel_b_within_celltype_mi_fidelity",
    },
    {
        "letter": "c",
        "plot_metric": "module_l2",
        "ylabel": "Module L2 error",
        "higher_better": False,
        "filename": "panel_c_within_celltype_module_l2",
    },
    {
        "letter": "d",
        "plot_metric": EDGE_PROFILE_PLOT_METRIC,
        "ylabel": "Per-gene co-expression\nprofile PCC",
        "higher_better": True,
        "filename": "panel_d_within_celltype_coexpression_profile_pcc_pca50",
    },
]


plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", ha="left")


def prepare_panel(long_data: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = long_data[long_data["metric"] == spec["plot_metric"]].copy()
    data["Model"] = data["method"].map(DISPLAY)
    if set(data["cell_type"]) != set(EXPECTED_TYPES):
        raise RuntimeError(f"Unexpected cell types for {spec['plot_metric']}: {sorted(data['cell_type'].unique())}")
    if set(data["method"]) != set(ORDER) or len(data) != 12:
        raise RuntimeError(f"Expected 12 paired rows for {spec['plot_metric']}, found {len(data)}")
    pivot = data.pivot(index="cell_type", columns="method", values="value").loc[EXPECTED_TYPES]
    if pivot.isna().any().any():
        raise RuntimeError(f"Missing paired value for {spec['plot_metric']}")
    return data, pivot


def draw_panel(
    ax: plt.Axes,
    long_data: pd.DataFrame,
    spec: dict,
    include_letter: bool,
    plot_style_name: str,
) -> dict:
    if plot_style_name not in PLOT_STYLES:
        raise ValueError(f"Unsupported plot style: {plot_style_name}")
    data, pivot = prepare_panel(long_data, spec)
    categories = [DISPLAY[key] for key in ORDER]
    palette = {DISPLAY[key]: PALETTE[key] for key in ORDER}

    if plot_style_name == "violin":
        sns.violinplot(
            data=data, x="Model", y="value", order=categories, hue="Model",
            palette=palette, legend=False, inner=None, cut=0, width=0.72,
            linewidth=0.9, saturation=1.0, ax=ax,
        )
        for collection in ax.collections:
            collection.set_alpha(0.34)
    else:
        sns.boxplot(
            data=data, x="Model", y="value", order=categories, hue="Model",
            palette=palette, legend=False, width=0.38, showfliers=False, whis=1.5,
            linewidth=1.0,
            boxprops={"alpha": 0.58, "edgecolor": "#4A4A4A"},
            whiskerprops={"color": "#4A4A4A", "linewidth": 1.0},
            capprops={"color": "#4A4A4A", "linewidth": 1.0},
            medianprops={"color": "#303030", "linewidth": 1.2},
            ax=ax,
        )

    offsets = np.linspace(-0.075, 0.075, len(EXPECTED_TYPES))
    for offset, cell_type in zip(offsets, EXPECTED_TYPES):
        y0 = float(pivot.loc[cell_type, "Route2"])
        y1 = float(pivot.loc[cell_type, "BaseVAE"])
        ax.scatter(0 + offset, y0, s=32, color=PALETTE["Route2"],
                   edgecolor="#333333", linewidth=0.45, zorder=4)
        ax.scatter(1 + offset, y1, s=32, color=PALETTE["BaseVAE"],
                   edgecolor="#333333", linewidth=0.45, zorder=4)

    values = data["value"].to_numpy(dtype=float)
    data_min, data_max = float(values.min()), float(values.max())
    data_range = max(data_max - data_min, data_max * 0.25, 1e-6)
    ax.set_ylim(bottom=0.0, top=data_max + data_range * 0.18)
    ax.set_ylabel(spec["ylabel"], fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks([0, 1], categories)
    plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(False)
    if include_letter:
        panel_letter(ax, spec["letter"])

    if spec["higher_better"]:
        advantage = pivot["Route2"] - pivot["BaseVAE"]
    else:
        advantage = pivot["BaseVAE"] - pivot["Route2"]
    return {
        "panel": spec["letter"],
        "plot_metric": spec["plot_metric"],
        "n_cell_types": len(pivot),
        "deconvsc_mean": float(pivot["Route2"].mean()),
        "basevae_mean": float(pivot["BaseVAE"].mean()),
        "deconvsc_wins": int((advantage > 0).sum()),
        "plot_style": plot_style_name,
        "density_layer_displayed": plot_style_name == "violin",
        "boxplot_displayed": plot_style_name == "boxplot",
        "paired_points_displayed": True,
        "paired_lines_displayed": False,
        "statistical_annotation_displayed": False,
    }


def output_base(outdir: Path, base_name: str, plot_style_name: str, compatibility_name: bool) -> Path:
    if compatibility_name:
        return outdir / base_name
    return outdir / f"{base_name}_{plot_style_name}"


def save_panel(
    long_data: pd.DataFrame,
    spec: dict,
    outdir: Path,
    plot_style_name: str,
    compatibility_name: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=600)
    draw_panel(ax, long_data, spec, include_letter=False, plot_style_name=plot_style_name)
    fig.tight_layout()
    plot_style.save(
        fig,
        str(output_base(outdir, spec["filename"], plot_style_name, compatibility_name)),
        dpi=600,
    )
    plt.close(fig)


def render_figure(
    long_data: pd.DataFrame,
    outdir: Path,
    plot_style_name: str,
    compatibility_name: bool = False,
) -> pd.DataFrame:
    fig = plt.figure(figsize=(12.0, 10.4), dpi=600)
    grid = fig.add_gridspec(
        2, 2, hspace=0.38, wspace=0.30,
        left=0.085, right=0.965, top=0.965, bottom=0.075,
    )
    audit_rows = []
    for index, spec in enumerate(PANEL_SPECS):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        audit_rows.append(
            draw_panel(
                ax,
                long_data,
                spec,
                include_letter=True,
                plot_style_name=plot_style_name,
            )
        )
        save_panel(long_data, spec, outdir, plot_style_name, compatibility_name)

    combined_base = output_base(
        outdir, "Figure4_within_celltype_ABCD", plot_style_name, compatibility_name
    )
    plot_style.save(fig, str(combined_base), dpi=600)
    plt.close(fig)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        combined_base.with_name(f"{combined_base.name}_plot_audit.csv"),
        index=False,
    )
    combined_base.with_name(f"{combined_base.name}_caption.md").write_text(
        build_caption(plot_style_name, long_data), encoding="utf-8"
    )
    print(audit.to_string(index=False))
    print(f"saved: {combined_base}.{{png,pdf,svg,emf}}")
    return audit


def build_caption(
    plot_style_name: str,
    long_data: pd.DataFrame,
) -> str:
    if plot_style_name == "violin":
        distribution_text = (
            "For each metric and model, the violin layer shows a kernel-density summary "
            "of six cell-type-level values after averaging each value over ten fixed "
            "subsampling seeds. Each overlaid point is the observed value for one cell "
            "type; points are left unconnected to avoid visual clutter."
        )
    else:
        distribution_text = (
            "For each metric and model, the box plot was calculated from six cell-type-"
            "level values after averaging each value over ten fixed subsampling seeds. "
            "The centre line denotes the median, the box spans the interquartile range, "
            "and the whiskers extend to the most extreme observations within 1.5 times "
            "the interquartile range. Each overlaid point is the observed value for one "
            "cell type, including any point beyond the whiskers; points are left "
            "unconnected to avoid visual clutter."
        )
    panel_values = {}
    for spec in PANEL_SPECS:
        _, pivot = prepare_panel(long_data, spec)
        if spec["higher_better"]:
            advantage = pivot["Route2"] - pivot["BaseVAE"]
        else:
            advantage = pivot["BaseVAE"] - pivot["Route2"]
        panel_values[spec["letter"]] = {
            "deconv": float(pivot["Route2"].mean()),
            "base": float(pivot["BaseVAE"].mean()),
            "wins": int((advantage > 0).sum()),
        }

    def wins_phrase(value: int) -> str:
        if value == 6:
            return "all six"
        if value == 5:
            return "five of six"
        return f"{value} of six"

    return f"""# Figure 4 caption — revised within-cell-type topology

**Figure 4 | Comparison of within-cell-type expression-topology fidelity between DeconvSC and an attention-ablated VAE.** Within-cell-type topology was assessed in six mouse kidney cell types from GSE141115 (CD_PC, CNT, DCT, MC, PT and aLOH). Within each metric, equal numbers of cells were sampled independently from the held-out real data, DeconvSC output and ablated-VAE output for each cell type and donor, with a maximum of 200 cells per donor; real cells were not divided into reference subsets. **a,** Hub-neighbour Jaccard similarity quantifies overlap between generated and ground-truth co-expression neighbourhoods. For each cell type, 50 hub genes were defined exclusively from the ground-truth correlation matrix; for each hub, the top 20 non-self genes ranked by absolute Pearson correlation formed its neighbourhood. The mean cell-type-level Jaccard similarity was {panel_values['a']['deconv']:.3f} for DeconvSC and {panel_values['a']['base']:.3f} for the ablated VAE, with higher values for DeconvSC in {wins_phrase(panel_values['a']['wins'])} cell types. **b,** Mutual-information (MI) fidelity error is the median absolute difference between corresponding generated and ground-truth pairwise MI values; lower values indicate better preservation of non-linear expression dependence. The mean cell-type-level error was {panel_values['b']['deconv']:.3f} for DeconvSC and {panel_values['b']['base']:.3f} for the ablated VAE, with lower values for DeconvSC in {wins_phrase(panel_values['b']['wins'])} cell types. **c,** Module L2 error is the Frobenius distance between generated and ground-truth within-module correlation matrices divided by the square of module size; lower values indicate better preservation of module correlation structure. The mean error was {panel_values['c']['deconv']:.4f} for DeconvSC and {panel_values['c']['base']:.4f} for the ablated VAE, with lower values for DeconvSC in {wins_phrase(panel_values['c']['wins'])} cell types. **d,** Per-gene co-expression profile PCC compares, for each of 200 training-stable genes, its signed correlations with the other 199 genes between generated and held-out real correlation matrices. Cell-type-specific PCA bases were fitted only to donor-centred training-reference expression; the retained rank was the smaller of the rank needed to explain 80% of training variance and 50 components. Every 200-gene panel reached the 50-component cap (39.1–50.0% variance explained), so the analysis is denoted PCA ≤ 50 rather than PCA80. The same PCA projection and Ledoit–Wolf estimator were applied to real and generated expression. Gene-level PCCs were summarized by their median within each seed and averaged over ten subsampling seeds. The mean cell-type-level PCC was {panel_values['d']['deconv']:.3f} for DeconvSC and {panel_values['d']['base']:.3f} for the ablated VAE, with higher values for DeconvSC in {wins_phrase(panel_values['d']['wins'])} cell types. The four panels provide complementary views of topology: panel a tests sparse, rank-based local-neighbour membership around ground-truth-defined hubs; panel b tests non-linear pairwise dependence; panel c tests correlation blocks within ground-truth-defined modules; and panel d tests dense, gene-centred agreement in signed linear co-expression patterns without clustering. {distribution_text} The ten seeds averaged subsampling variation and were not treated as independent observations. The displayed distributions therefore summarize cell types, not cells, genes, gene pairs, modules or seeds. Cell type was the inferential unit, and no inferential annotations are displayed. BayesPrism, TAPE, CIBERSORTx and DISSECT were not included because their available outputs are sample-by-cell-type aggregate profiles and cannot estimate within-cell-type cell-level covariance. These metrics quantify undirected statistical co-expression or dependency structure and do not establish directed or causal gene-regulatory networks.
"""


def load_symmetric_pca50_panel(edge_profile_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(edge_profile_path)
    required = {
        "cell_type", "method", "n_cells_per_source", "mean_over_seeds",
        "sd_over_seeds",
    }
    exported = {
        "cell_type", "comparison", "n_cells_per_source_mean_over_seeds",
        "plotted_value", "profile_pcc_raw_median_sd_over_seeds",
    }
    if not required.issubset(source.columns) and exported.issubset(source.columns):
        # Accept the frozen Figure 4D plotting table as well as its upstream
        # per-cell-type summary, so a downloaded release can redraw the figure
        # directly from `figure4_inputs/`.
        source = source.assign(
            method=source["comparison"].astype(str).str.replace("_vs_Real", "", regex=False),
            n_cells_per_source=source["n_cells_per_source_mean_over_seeds"],
            mean_over_seeds=source["plotted_value"],
            sd_over_seeds=source["profile_pcc_raw_median_sd_over_seeds"],
        )
    if not required.issubset(source.columns):
        raise RuntimeError(f"Missing no-split PCA<=50 columns: {sorted(required - set(source.columns))}")
    selected = source[source["method"].isin(EDGE_PROFILE_METHOD_TO_PLOT_METHOD)].copy()
    if set(selected["method"]) != set(EDGE_PROFILE_METHOD_TO_PLOT_METHOD):
        raise RuntimeError(f"Unexpected methods: {sorted(selected['method'].unique())}")
    if set(selected["cell_type"]) != set(EXPECTED_TYPES) or len(selected) != 12:
        raise RuntimeError(
            f"Expected 12 rows (six cell types x two models), found {len(selected)}"
        )
    export = pd.DataFrame({
        "setting": EDGE_PROFILE_SETTING,
        "panel_size": EDGE_PROFILE_PANEL_SIZE,
        "cell_type": selected["cell_type"],
        "comparison": selected["method"] + "_vs_Real",
        "n_seeds": 10,
        "n_cells_per_source_mean_over_seeds": selected["n_cells_per_source"].astype(float),
        "n_genes_mean_over_seeds": float(EDGE_PROFILE_PANEL_SIZE),
        "plotted_value": selected[EDGE_PROFILE_SOURCE_METRIC].astype(float),
        "profile_pcc_raw_median_sd_over_seeds": selected["sd_over_seeds"].astype(float),
        "real_cell_split": False,
    }).sort_values(["cell_type", "comparison"])
    selected["plot_method"] = selected["method"].map(EDGE_PROFILE_METHOD_TO_PLOT_METHOD)
    panel_long = pd.DataFrame({
        "analysis": "within_celltype_P0",
        "unit": "cell_type",
        "cell_type": selected["cell_type"],
        "method": selected["plot_method"],
        "display_method": selected["plot_method"].map(DISPLAY),
        "metric": EDGE_PROFILE_PLOT_METRIC,
        "value": selected[EDGE_PROFILE_SOURCE_METRIC].astype(float),
        "better": "higher",
        "n_seeds_averaged": 10,
    })
    return panel_long, export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--edge-profile-data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    long_data = pd.read_csv(args.data)
    long_data = long_data[long_data["analysis"] == "within_celltype_P0"].copy()
    long_data = long_data[long_data["metric"] != "coexpression_edge_mae"].copy()
    edge_panel, edge_export = load_symmetric_pca50_panel(
        args.edge_profile_data
    )
    long_data = pd.concat([long_data, edge_panel], ignore_index=True, sort=False)

    input_dir = outdir.parent / "figure4_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    edge_export.to_csv(
        input_dir / "figure4d_pca50_symmetric_profile_pcc.csv", index=False
    )

    for plot_style_name in PLOT_STYLES:
        render_figure(
            long_data, outdir, plot_style_name,
            compatibility_name=False,
        )
    render_figure(
        long_data, outdir, "boxplot",
        compatibility_name=True,
    )
    (outdir / "caption.md").write_text(
        build_caption("boxplot", long_data), encoding="utf-8"
    )
    (outdir / "README.md").write_text(
        (
            "# Current Figure 4 files\n\n"
            "The authoritative combined figure is `Figure4_within_celltype_ABCD` "
            "in PNG, PDF, SVG and EMF formats. It contains box plots, six cell-type "
            "points without connecting lines, with no density layer or inferential "
            "annotation. "
            "Panel d reports per-gene co-expression profile PCC for 200 training-stable "
            "genes after the same training-derived PCA <= 50 projection and Ledoit-Wolf "
            "estimator were applied to real and generated expression. Ground-truth "
            "cells were not split, and panel d used the same donor-stratified, "
            "cell-count-balancing rule as panels a-c.\n\n"
            "The authoritative standalone panel-d files are named "
            "`panel_d_within_celltype_coexpression_profile_pcc_pca50.*`. The source rows "
            "used for panel d are in "
            "`../figure4_inputs/figure4d_pca50_symmetric_profile_pcc.csv`.\n\n"
            "Files whose names contain `violin`, `network_fidelity` or "
            "`coexpression_edge_mae` predate the current display-endpoint revision and "
            "are retained only as an audit trail; they should not be used as the current "
            "Figure 4.\n"
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
