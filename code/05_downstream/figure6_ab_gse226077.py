#!/usr/bin/env python
"""Figure 6 a/b for the GSE226077 iMGL Route2 application (new-model generated_data.h5ad).

  a  UMAP of the 3198 generated single cells coloured by the 5 microglial states
     (same palette as the Monocle2 cell-type trajectory, panel d -> colours match across panels).
  b  Curated marker dot plot: canonical microglia genes grouped by function
     (core / antigen presentation / complement / AD-risk / activation / immediate-early /
      cell-cycle / heat-shock), dot size = fraction of cells, colour = scaled mean expression.

Style matches the R panels: Arial (Liberation Sans is its metric substitute on this box; the
exported SVG is rewritten to a literal "Arial"), exported as PNG (preview) + SVG (editable) + PDF
(editable, fonts embedded). UMAP coordinates are cached so re-runs are fast and reproducible.
"""
import os, re, shutil, subprocess, numpy as np, scanpy as sc, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); import plot_style  # noqa: F401

RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUT = os.environ.get("IMGL_ROOT", str(RELEASE_ROOT / "work/downstream/GSE226077"))
os.makedirs(OUT, exist_ok=True)
SRC = os.environ.get("IMGL_H5AD", str(RELEASE_ROOT / "model_outputs/GSE226077/application_generated.h5ad"))
EMB = os.path.join(OUT, "umap_embedding.h5ad")
CT = "Cell_type"
sc.settings.verbosity = 0

# UMAP cell-type legend. Coordinates are in axes fraction; (0.02, 0.98) places the legend
# inside the main UMAP panel near the upper-left corner. Override from the shell if needed:
#   UMAP_LEGEND_FONTSIZE=7 UMAP_LEGEND_TITLE_FONTSIZE=8 UMAP_LEGEND_BBOX=0.03,0.97 ...
UMAP_LEGEND_FONTSIZE = float(os.environ.get("UMAP_LEGEND_FONTSIZE", "8"))
UMAP_LEGEND_TITLE_FONTSIZE = float(os.environ.get("UMAP_LEGEND_TITLE_FONTSIZE", "9"))
UMAP_LEGEND_MARKERSIZE = float(os.environ.get("UMAP_LEGEND_MARKERSIZE", "5"))
UMAP_LEGEND_LOC = os.environ.get("UMAP_LEGEND_LOC", "upper left")
UMAP_LEGEND_BBOX = tuple(float(x) for x in os.environ.get("UMAP_LEGEND_BBOX", "0.02,0.98").split(","))
UMAP_LEGEND_FRAME_ALPHA = float(os.environ.get("UMAP_LEGEND_FRAME_ALPHA", "1.0"))
UMAP_EXTRA_TOP_FRAC = float(os.environ.get("UMAP_EXTRA_TOP_FRAC", "0.16"))

# ---- typography: render in Liberation Sans (Arial metric twin); SVG relabelled to "Arial" ----
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none",   # editable text in SVG
    "pdf.fonttype": 42,       # editable embedded TrueType in PDF
    "font.size": 7, "axes.linewidth": 0.6, "axes.unicode_minus": False,
})

# 5 states (alphabetical order) -> identical colours to monocle2 panel d. Coloured by lineage family:
# Activated = red family, Homeostatic = blue family, Myeloid progenitor = green.
PAL = ["#C0504D",   # Activated, non-proliferative cells   (red, dark)
       "#E8A598",   # Activated, proliferative cells       (red, light)
       "#9CC0DC",   # Homeostatic, immediate-early cells   (blue, light)
       "#4F7CAC",   # Homeostatic, non-proliferative cells (blue, dark)
       "#8BAE8E"]   # Myeloid progenitor cells             (green)
# dot-plot colour map = the manuscript heatmap palette (blue -> cream -> salmon), matching every
# other Fig5/Fig6 heatmap (the original Figure-6b scheme; RdBu was too garish).
from matplotlib.colors import LinearSegmentedColormap
MS_CMAP = LinearSegmentedColormap.from_list("ms_div", ["#7187A7", "#F7F3EF", "#D8B2AE"])


def svg_to_arial(path):
    with open(path) as fh:
        s = fh.read()
    s = s.replace("Liberation Sans", "Arial").replace("DejaVu Sans", "Arial")
    # markeredgecolor="none" -> matplotlib writes a black stroke at stroke-opacity:0 (invisible in PDF),
    # but inkscape's EMF export ignores the opacity and draws a black ring on the legend dots. Convert
    # those invisible strokes to a real no-stroke so the EMF is clean too.
    s = re.sub(r"stroke:\s*#000000;\s*stroke-opacity:\s*0", "stroke: none", s)
    with open(path, "w") as fh:
        fh.write(s)


def save3(fig, base):
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight"); svg_to_arial(base + ".svg")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    # EMF (Office vector) from the Arial SVG; outline PDF text so no "LiberationSans" font is referenced
    if shutil.which("inkscape"):
        subprocess.run(["inkscape", base + ".svg", "--export-type=emf", "--export-filename=" + base + ".emf"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if shutil.which("gs"):
        subprocess.run(["gs", "-q", "-o", base + ".pdf.ol", "-dNoOutputFonts", "-sDEVICE=pdfwrite", base + ".pdf"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(base + ".pdf.ol"):
            os.replace(base + ".pdf.ol", base + ".pdf")
    print("[fig]", os.path.basename(base), ".png/.svg/.pdf/.emf")


def get_embedding():
    if os.path.exists(EMB):
        return sc.read_h5ad(EMB)
    a = sc.read_h5ad(SRC)                       # X is already log1p
    a.obs[CT] = a.obs[CT].astype("category")
    sc.pp.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_neighbors=15, random_state=0)
    sc.tl.umap(a, random_state=0)
    a.write_h5ad(EMB)
    return a


def panel_a(a):
    cats = list(a.obs[CT].cat.categories)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    sc.pl.umap(a, color=CT, palette=PAL[:len(cats)], ax=ax, show=False, size=26,
               frameon=True, title="", legend_loc="right margin")
    if ax.get_legend() is not None:
        ax.get_legend().remove()    # drop scanpy's legend; rebuild HCA-style (small dots) below
    ax.set_box_aspect(1)            # square UMAP panel
    ax.margins(0.06)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * UMAP_EXTRA_TOP_FRAC)
    # HCA style: big bold UMAP1/UMAP2 labels, no (arbitrary) tick numbers, full box
    ax.set_xlabel("UMAP1", fontsize=18, fontweight="bold")
    ax.set_ylabel("UMAP2", fontsize=18, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8)
    # HCA-style cell-type legend inside the UMAP axes with small marker dots.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=UMAP_LEGEND_MARKERSIZE,
                      markerfacecolor=PAL[i], markeredgecolor="none", markeredgewidth=0) for i in range(len(cats))]
    leg = ax.legend(handles, cats, title="Cell type",
                    title_fontsize=UMAP_LEGEND_TITLE_FONTSIZE, fontsize=UMAP_LEGEND_FONTSIZE,
                    loc=UMAP_LEGEND_LOC, bbox_to_anchor=UMAP_LEGEND_BBOX,
                    bbox_transform=ax.transAxes, frameon=True, fancybox=False,
                    labelspacing=0.35, handletextpad=0.25, borderaxespad=0.0,
                    borderpad=0.35)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(UMAP_LEGEND_FRAME_ALPHA)
    for c in ax.collections:
        c.set_rasterized(True)            # keep the SVG/PDF small; text stays vector
    save3(fig, os.path.join(OUT, "umap_celltype"))


def panel_b(a):
    cand = {
        # TMEM119 (undetected, ~1%) and P2RY12 (very low, ~10%) dropped — iMGLs express these
        # homeostatic markers poorly; the robustly-expressed pan-microglia markers are shown instead.
        "Core markers":          ["AIF1", "CSF1R", "GPR34", "OLFML3", "CX3CR1"],
        "Antigen presentation":  ["HLA-DRA", "HLA-DRB1", "CD74", "B2M"],
        "Complement":            ["C1QA", "C1QB", "C1QC", "C3"],
        "AD risk genes":         ["APOE", "TREM2", "BIN1", "CD33"],
        "Activation":            ["CD68", "SPP1", "CD83"],
        "Immediate-early genes": ["FOS", "JUN", "EGR1", "JUNB"],
        "Cell cycle":            ["MKI67", "TOP2A", "CDK1"],
        "Heat shock":            ["HSPA1A", "HSPA1B", "DNAJB1", "HSPB1"],
    }
    present = set(a.var_names)
    markers = {k: [g for g in v if g in present] for k, v in cand.items()}
    markers = {k: v for k, v in markers.items() if v}
    ngene = sum(len(v) for v in markers.values())
    old_fs = matplotlib.rcParams["font.size"]
    matplotlib.rcParams["font.size"] = 11         # category-bracket + legend-title font
    dp = sc.pl.dotplot(a, markers, groupby=CT, standard_scale="var", cmap=MS_CMAP,
                       figsize=(0.34 * ngene + 3.2, 3.8), return_fig=True, dot_max=1.0,
                       colorbar_title="Scaled mean\nexpression", size_title="Fraction of cells in group (%)")
    dp.style(largest_dot=110, dot_edge_lw=0.3, dot_edge_color="white")   # moderate dots
    dp.make_figure()
    fig = getattr(dp, "fig", None) or plt.gcf()
    for ax in fig.axes:                            # larger tick text
        for t in ax.get_xticklabels():
            t.set_fontsize(11); t.set_rotation(90)
        for t in ax.get_yticklabels():
            t.set_fontsize(12)
    save3(fig, os.path.join(OUT, "marker_dotplot"))
    matplotlib.rcParams["font.size"] = old_fs
    print("  markers used:", ngene, "genes in", len(markers), "categories")


if __name__ == "__main__":
    a = get_embedding()
    panel_a(a)
    panel_b(a)
    print("DONE Figure-6 a/b ->", OUT)
