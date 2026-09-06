"""Shared figure style for DeconvSC matplotlib plotting: Arial font + EMF export.

Self-contained port of the deconv_20260610 ``lib/plot_style.py`` so the papercode2 release carries
no external dependency. Importing this module applies the Arial style to matplotlib globally (a bare
``from . import plot_style`` is enough).

"Arial" is typically not installed on Linux, but Liberation Sans is its metric-identical open clone
(``fc-match Arial`` -> Liberation Sans); we render with Liberation Sans and relabel the exported
SVG/EMF text to a literal "Arial" so downstream editors (Illustrator/Office) see Arial.

  from . import plot_style                 # applies Arial to matplotlib on import
  plot_style.save(fig, "<dir>/name")       # writes name.{png,pdf,svg,emf} (svg relabelled to Arial)
"""
import os
import glob
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm

# Liberation Sans (Arial metric twin) lives here on most RHEL/Fedora hosts; harmless if absent.
_LIBSANS_DIRS = [
    "/usr/share/fonts/liberation-sans",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/liberation",
]


def apply_arial():
    """Register Liberation Sans (Arial metric twin) AND alias it to the literal name 'Arial', so
    ``findfont('Arial')`` returns Liberation Sans glyphs (not a DejaVu fallback) for every script —
    even ones that set ``font.family=['Arial']`` themselves. Then point matplotlib at Arial."""
    import copy
    import dataclasses
    for d in _LIBSANS_DIRS:
        for ttf in glob.glob(os.path.join(d, "LiberationSans-*.ttf")):
            try:
                _fm.fontManager.addfont(ttf)
            except Exception:
                pass
    # alias: clone each Liberation Sans entry under the name "Arial" (FontEntry is a frozen
    # dataclass on newer matplotlib, so use replace(); fall back to copy+setattr on older builds)
    if not any(getattr(fe, "name", "") == "Arial" for fe in _fm.fontManager.ttflist):
        for fe in list(_fm.fontManager.ttflist):
            if getattr(fe, "name", "") == "Liberation Sans":
                try:
                    alias = dataclasses.replace(fe, name="Arial")
                except Exception:
                    alias = copy.copy(fe)
                    try:
                        alias.name = "Arial"
                    except Exception:
                        object.__setattr__(alias, "name", "Arial")
                _fm.fontManager.ttflist.append(alias)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none",   # keep text as editable <text> (so relabel + EMF keep real text)
        "pdf.fonttype": 42,        # TrueType-embedded editable text (not outlines)
        "ps.fonttype": 42,
        # math text must render in Arial too, else matplotlib's default 'dejavusans' math fontset
        # leaks DejaVu Sans Oblique into the PDF.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial:italic",
        "mathtext.sf": "Arial",
        "mathtext.tt": "Arial",
    })


# applied on import so callers only need `from . import plot_style`
apply_arial()


def relabel_svg(path):
    """Rewrite a Liberation-Sans-labelled SVG so its font-family literally reads 'Arial'."""
    if not path.endswith(".svg") or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            s = fh.read()
        for old in ("Liberation Sans", "LiberationSans", "DejaVu Sans", "DejaVuSans"):
            s = s.replace(old, "Arial")   # uniform "Arial" font-family in the exported vector
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(s)
    except Exception:
        pass


def svg_to_emf(svg):
    """SVG -> EMF via inkscape (editable vector for MS Office). No-op if inkscape absent."""
    if not os.path.exists(svg) or not shutil.which("inkscape"):
        return
    emf = svg[:-4] + ".emf"
    try:
        subprocess.run(["inkscape", svg, "--export-type=emf", f"--export-filename={emf}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
    except Exception:
        pass


def save(fig, base, formats=("png", "pdf", "svg"), dpi=600, emf=True, bbox_inches="tight"):
    """Save ``fig`` to ``<base>.<ext>`` for each format, relabel the SVG to Arial, and (optionally)
    emit ``<base>.emf``. ``dpi`` applies to ALL formats: for vector outputs (pdf/svg) it sets the
    resolution of any ``set_rasterized(True)`` artists while text/lines stay vector."""
    if base.lower().endswith((".png", ".pdf", ".svg", ".emf")):
        base = base[:-4]
    for ext in formats:
        fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches=bbox_inches)
    svg = f"{base}.svg"
    if os.path.exists(svg):
        relabel_svg(svg)
        if emf:
            svg_to_emf(svg)
    return base


def emf_dir(outdir, recursive=False):
    """Relabel + convert every .svg under ``outdir`` to .emf (covers figures saved elsewhere)."""
    pat = os.path.join(outdir, "**", "*.svg") if recursive else os.path.join(outdir, "*.svg")
    n = 0
    for svg in glob.glob(pat, recursive=recursive):
        relabel_svg(svg)
        svg_to_emf(svg)
        n += 1
    return n
