#!/usr/bin/env python3
"""Compose a 'raster-body + vector-text' SVG from a rasterized heatmap body PNG
plus the original svglite <text> elements, so the resulting EMF is small
(one embedded bitmap) yet all labels stay crisp vector.

Usage: _build_raster_svg.py <body.png> <texts.txt> <out.svg> <viewbox_w> <viewbox_h>

The document width/height are set in *px* equal to the PNG's pixel size so that
Inkscape rasterizes the embedded <image> 1:1 (no resample) when exporting EMF;
the viewBox keeps the original pt coordinate system so the <text> lines, copied
verbatim from the source SVG, land exactly where svglite drew them.
"""
import sys, base64, struct

png_path, texts_path, out_svg, vb_w, vb_h = sys.argv[1:6]
png = open(png_path, "rb").read()
# PNG IHDR: width/height are big-endian uint32 at bytes 16:24
w, h = struct.unpack(">II", png[16:24])
b64 = base64.b64encode(png).decode("ascii")
texts = open(texts_path).read().rstrip("\n")

svg = (
    "<?xml version='1.0' encoding='UTF-8' ?>\n"
    "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' "
    f"width='{w}px' height='{h}px' viewBox='0 0 {vb_w} {vb_h}' preserveAspectRatio='none'>\n"
    f"<image x='0' y='0' width='{vb_w}' height='{vb_h}' preserveAspectRatio='none' "
    f"xlink:href='data:image/png;base64,{b64}'/>\n"
    f"{texts}\n"
    "</svg>\n"
)
open(out_svg, "w").write(svg)
print(f"wrote {out_svg}: image {w}x{h}px + {texts.count('<text')} vector text elems")
