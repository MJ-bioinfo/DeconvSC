#!/usr/bin/env Rscript
# Linux-adapted Monocle2.R for the GSE226077 iMGL Route2 application.
# Builds a Monocle2 DDRTree pseudotime trajectory from the Route2 generated single cells,
# roots it at the Myeloid-progenitor state (biological start of iMGL differentiation), and
# exports the pseudotime trajectory + cell-type trajectory as SVG (+PNG).
# Ordering genes come from Python (top markers per Cell_type) so no Seurat is needed.
# The interactive `trace(...,edit=TRUE)` igraph patches from the original are applied
# non-interactively below.
suppressMessages({
  library(monocle); library(reticulate); library(Matrix)
  library(dplyr); library(ggplot2); library(stringr); library(svglite); library(systemfonts)
  library(grid)
})
set.seed(18)
## ---- Arial everywhere; literal "Arial" in exported SVG ----
FF <- "Arial"
try(register_font(name = FF, plain = match_fonts(FF)$path, bold = match_fonts(FF, weight = "bold")$path,
  italic = match_fonts(FF, italic = TRUE)$path, bolditalic = match_fonts(FF, weight = "bold", italic = TRUE)$path), silent = TRUE)  # skip if a real "Arial" is already installed
svg_to_arial <- function(p) { x <- readLines(p, warn = FALSE)
  x <- gsub("Liberation Sans", "Arial", x, fixed = TRUE); x <- gsub("DejaVu Sans", "Arial", x, fixed = TRUE)
  x <- gsub(" textLength='[^']*'", "", x); x <- gsub(" lengthAdjust='[^']*'", "", x)   # avoid viewer clipping
  writeLines(x, p) }
outline_pdf <- function(f) {   # PDF text -> outlines: no "LiberationSans" font ref for Illustrator
  if (nchar(Sys.which("gs")) == 0) return(invisible())
  tmp <- paste0(f, ".ol")
  st <- tryCatch(system2("gs", c("-q", "-o", tmp, "-dNoOutputFonts", "-sDEVICE=pdfwrite", f), stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  if (identical(st, 0L) && file.exists(tmp)) file.rename(tmp, f)
}
svg_to_emf <- function(svg) {  # EMF (Office vector) from the final Arial SVG
  if (nchar(Sys.which("inkscape")) == 0) return(invisible())
  try(system2("inkscape", c(svg, "--export-type=emf", paste0("--export-filename=", sub("\\.svg$", ".emf", svg))), stdout = FALSE, stderr = FALSE), silent = TRUE)
}
svg_to_pdf <- function(svg) {
  if (nchar(Sys.which("inkscape")) == 0) return(invisible(FALSE))
  out <- sub("\\.svg$", ".pdf", svg)
  st <- tryCatch(system2("inkscape", c(svg, "--export-type=pdf", paste0("--export-filename=", out)),
                         stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  invisible(identical(st, 0L) && file.exists(out))
}
fix_branched_heatmap_legend_title_svg <- function(svg) {
  x <- readLines(svg, warn = FALSE)
  shift_legend_y <- function(line, x0, dy) {
    pat <- paste0("^(.*<(?:rect|text) x='", x0, "' y=')([0-9.]+)('.*)$")
    hit <- regmatches(line, regexec(pat, line, perl = TRUE))[[1]]
    if (length(hit) == 0) return(line)
    paste0(hit[2], sprintf("%.2f", as.numeric(hit[3]) + dy), hit[4])
  }
  # Put the heatmap colour legend below the annotation legends.
  x <- vapply(x, shift_legend_y, character(1), x0 = "786.88", dy = 95)
  x <- vapply(x, shift_legend_y, character(1), x0 = "800.88", dy = 95)
  x <- sub("(<text x='[0-9.]+' )y='[0-9.]+'( style='font-size: 8[.]00px; font-weight: bold; font-family: \"Arial\";'>Z-score</text>)",
           "\\1y='275.00'\\2", x)
  writeLines(x, svg)
}
matrix_to_colors <- function(mat, cols) {
  bks <- seq(range(mat, finite = TRUE)[1] - 0.1, range(mat, finite = TRUE)[2] + 0.1, by = 0.1)
  ids <- findInterval(as.numeric(mat), bks, all.inside = TRUE)
  ids <- pmin(pmax(ids, 1), length(cols))
  fill <- cols[ids]
  fill[is.na(fill)] <- "#FFFFFF"
  matrix(fill, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
}
insert_gap_cols <- function(col_mat, gaps, gap_width = 2) {
  gaps <- sort(unique(as.integer(gaps)), decreasing = TRUE)
  gaps <- gaps[is.finite(gaps) & gaps > 0 & gaps < ncol(col_mat)]
  for (g in gaps) {
    col_mat <- cbind(col_mat[, seq_len(g), drop = FALSE],
                     matrix("#FFFFFF", nrow = nrow(col_mat), ncol = gap_width),
                     col_mat[, (g + 1):ncol(col_mat), drop = FALSE])
  }
  col_mat
}
downsample_color_matrix <- function(col_mat, max_rows = 2400) {
  if (nrow(col_mat) <= max_rows) return(col_mat)
  col_mat[unique(round(seq(1, nrow(col_mat), length.out = max_rows))), , drop = FALSE]
}
prepare_branched_heatmap_gtable <- function(ph_beam, legend_title = "Z-score") {
  gt <- ph_beam$ph_res$gtable

  # pheatmap(useRaster=TRUE) still emits one SVG rect per heatmap cell in this environment.
  # Replace only the dense heatmap body with an embedded raster; text and legends remain vector.
  mat <- ph_beam$heatmap_matrix
  if (!is.null(ph_beam$ph_res$tree_row)) mat <- mat[ph_beam$ph_res$tree_row$order, , drop = FALSE]
  col_mat <- downsample_color_matrix(insert_gap_cols(matrix_to_colors(mat, ph_beam$hmcols), ph_beam$col_gap_ind))
  matrix_idx <- which(gt$layout$name == "matrix")
  if (length(matrix_idx) == 1) {
    gt$grobs[[matrix_idx]] <- rasterGrob(as.raster(col_mat), width = unit(1, "npc"), height = unit(1, "npc"),
                                         interpolate = FALSE)
  }

  legend_idx <- which(gt$layout$name == "legend")
  if (length(legend_idx) == 1) {
    title_grob <- textGrob(legend_title, x = unit(0, "bigpts"), y = unit(0, "npc") - unit(3, "mm"),
                           hjust = 0, vjust = 1, gp = gpar(fontfamily = FF, fontface = "bold", fontsize = 8))
    gt$grobs[[legend_idx]] <- addGrob(gt$grobs[[legend_idx]], title_grob)
  }
  gt
}
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])
RELEASE_ROOT <- normalizePath(file.path(dirname(script_file), "..", ".."), mustWork = TRUE)
OUT <- Sys.getenv("IMGL_ROOT", file.path(RELEASE_ROOT, "work", "downstream", "GSE226077"))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
use_python(Sys.getenv("RETICULATE_PYTHON", Sys.which("python")), required = TRUE)
ad <- import("anndata")

## ---- read generated cells ----
adata <- ad$read_h5ad(Sys.getenv("IMGL_H5AD", file.path(RELEASE_ROOT, "model_outputs", "GSE226077", "application_generated.h5ad")))
X <- t(as.matrix(py_to_r(adata$X)))                       # genes x cells (log1p space)
genes <- unlist(py_to_r(adata$var_names$tolist())); rownames(X) <- genes
obs <- py_to_r(adata$obs)
cells <- paste0("Cell_", seq_len(ncol(X))); colnames(X) <- cells; rownames(obs) <- cells
obs$Cell_type <- as.factor(as.character(obs$Cell_type))
cat("cells:", ncol(X), " genes:", nrow(X), " types:", nlevels(obs$Cell_type), "\n")

pd <- new("AnnotatedDataFrame", data = obs)
fd <- new("AnnotatedDataFrame", data = data.frame(gene_short_name = genes, row.names = genes))
HSMM <- newCellDataSet(as.matrix(X), phenoData = pd, featureData = fd, expressionFamily = uninormal())

## ---- ordering genes (top markers per Cell_type, computed in Python -> no Seurat) ----
ordering_file <- Sys.getenv("ORDERING_GENES", file.path(RELEASE_ROOT, "results", "downstream", "GSE226077", "ordering_genes.csv"))
ordering_genes <- read.csv(ordering_file)$ordering_genes
ordering_genes <- intersect(as.character(ordering_genes), rownames(HSMM))
cat("ordering genes:", length(ordering_genes), "\n")
## ---- non-interactive igraph compat patch (monocle 2.34 + igraph >= 1.3) ----
patch_ns <- function(fn, ns, subs) {
  f <- getFromNamespace(fn, ns); src <- deparse(f)
  for (s in subs) src <- gsub(s[1], s[2], src, fixed = TRUE)
  f2 <- eval(parse(text = paste(src, collapse = "\n"))); environment(f2) <- asNamespace(ns)
  assignInNamespace(fn, f2, ns = ns); cat("[patched]", fn, "\n")
}

## ---- DDRTree + order cells (cache-load if already computed) ----
rds <- file.path(OUT, "monocle2_HSMM_order.rds")
if (file.exists(rds)) {
  HSMM <- readRDS(rds); cat("[load] cached ordered HSMM\n")
} else {
  HSMM <- setOrderingFilter(HSMM, ordering_genes)
  # extract_ddrtree_ordering: graph.dfs(neimode=) -> mode= ; project2MST: nei( -> .nei(
  try(patch_ns("extract_ddrtree_ordering", "monocle", list(c("neimode", "mode"))), silent = TRUE)
  try(patch_ns("project2MST", "monocle", list(c("nei(", ".nei("))), silent = TRUE)
  HSMM <- reduceDimension(HSMM, max_components = 2, method = "DDRTree", norm_method = "none")
  HSMM <- orderCells(HSMM)
  ct_table <- table(pData(HSMM)$State, pData(HSMM)$Cell_type)
  root_state <- rownames(ct_table)[which.max(ct_table[, "Myeloid progenitor cells"])]
  cat("root State (most Myeloid progenitor):", root_state, "\n")
  HSMM <- orderCells(HSMM, root_state = root_state)
  saveRDS(HSMM, rds)
}
print(table(pData(HSMM)$development_stage, pData(HSMM)$State))

## ---- export trajectories: plot directly from the DDRTree coordinates.
##      (monocle's own plot_cell_trajectory calls dplyr::select_(), defunct in dplyr 1.x.)
S <- monocle::reducedDimS(HSMM); if (ncol(S) == ncol(HSMM)) S <- t(S)   # cells x 2
K <- monocle::reducedDimK(HSMM); if (nrow(K) == 2) K <- t(K)            # nodes x 2
el <- igraph::as_edgelist(monocle::minSpanningTree(HSMM), names = FALSE)
edge_df <- data.frame(x = K[el[, 1], 1], y = K[el[, 1], 2], xend = K[el[, 2], 1], yend = K[el[, 2], 2])
pdat <- pData(HSMM)
cell_df <- data.frame(C1 = S[, 1], C2 = S[, 2], Pseudotime = pdat$Pseudotime,
                      Cell_type = factor(pdat$Cell_type), Day = factor(pdat$development_stage))

save_gg <- function(p, base, w_mm, h_mm) {     # PNG + editable SVG + editable PDF (all Arial)
  ggsave(file.path(OUT, paste0(base, ".png")), p, width = w_mm, height = h_mm, units = "mm", dpi = 600)
  ggsave(file.path(OUT, paste0(base, ".svg")), p, width = w_mm, height = h_mm, units = "mm", device = svglite::svglite)
  svg_to_arial(file.path(OUT, paste0(base, ".svg")))
  ggsave(file.path(OUT, paste0(base, ".pdf")), p, width = w_mm, height = h_mm, units = "mm", device = grDevices::cairo_pdf)
  cat("[fig]", base, ".png/.svg/.pdf\n")
}
traj_base <- list(
  geom_segment(data = edge_df, aes(x, y, xend = xend, yend = yend), color = "grey35", linewidth = 0.45),
  theme_classic(base_size = 7, base_family = FF), labs(x = "Component 1", y = "Component 2"),
  theme(axis.text = element_text(colour = "black"), axis.line = element_line(linewidth = 0.3),
        axis.ticks = element_line(linewidth = 0.3),
        legend.title = element_text(size = 6.5), legend.text = element_text(size = 6),
        legend.key.size = unit(3.5, "mm")))

# Panel c — pseudotime trajectory (manuscript blue gradient)
p1 <- ggplot(cell_df, aes(C1, C2)) + traj_base +
  geom_point(aes(color = Pseudotime), size = 0.5, alpha = 0.9) +
  scale_color_gradient(low = "#E3ECF5", high = "#2F5D8A") +
  guides(color = guide_colorbar(barwidth = unit(20, "mm"), barheight = unit(2.5, "mm"))) +
  theme(legend.position = "top")
save_gg(p1, "trajectory_pseudotime", 88, 82)

# Panel d — cell-type trajectory. Colour by lineage family (matches Figure-6a UMAP):
# Activated = reds, Homeostatic = blues, Myeloid progenitor = green; same-state groups share a hue family.
pal_ct <- c("Activated, non-proliferative cells"   = "#A4322A",   # Activated   (dark red)
            "Activated, proliferative cells"       = "#E0907F",   # Activated   (light red)
            "Homeostatic, immediate-early cells"   = "#9CC0DC",   # Homeostatic (light blue)
            "Homeostatic, non-proliferative cells" = "#33608C",   # Homeostatic (dark blue)
            "Myeloid progenitor cells"             = "#6FA368")   # Myeloid progenitor (green)
p2 <- ggplot(cell_df, aes(C1, C2)) + traj_base +
  geom_point(aes(color = Cell_type), size = 0.5, alpha = 0.9) +
  scale_color_manual(name = "Cell type", values = pal_ct) +
  guides(color = guide_legend(nrow = 3, override.aes = list(size = 2))) + theme(legend.position = "top")
save_gg(p2, "trajectory_celltype", 95, 92)

## ---- EMF (Office vector) + outline PDF text for the trajectory panels ----
for (s in c("trajectory_pseudotime", "trajectory_celltype")) svg_to_emf(file.path(OUT, paste0(s, ".svg")))
for (s in c("trajectory_pseudotime", "trajectory_celltype")) outline_pdf(file.path(OUT, paste0(s, ".pdf")))

## ---- Branch Analysis (BEAM): branch-dependent genes at branch point 1 ----
## Mirrors script/Monocle2.R. The interactive trace(buildBranchCellDataSet, edit=TRUE) "nei -> .nei"
## fix is applied non-interactively here (same mechanism as the project2MST/extract_ddrtree patches).
try(patch_ns("buildBranchCellDataSet", "monocle", list(c("nei(", ".nei("), c("neimode", "mode"))), silent = TRUE)
beam_csv <- file.path(OUT, "BEAM_res.csv")
if (file.exists(beam_csv)) {
  BEAM_res <- read.csv(beam_csv, stringsAsFactors = FALSE, check.names = FALSE); cat("[load] cached", beam_csv, "\n")
} else {
  cat("[BEAM] running BEAM(branch_point = 1) on", ncol(HSMM), "cells ...\n")
  BEAM_res <- BEAM(HSMM, branch_point = 1, cores = as.integer(Sys.getenv("BEAM_CORES", "8")),
                   progenitor_method = "duplicate")
  write.csv(BEAM_res, beam_csv, row.names = FALSE); cat("[save]", beam_csv, "\n")
}
BEAM_res <- BEAM_res[order(BEAM_res$qval), ]
BEAM_sig <- BEAM_res[!is.na(BEAM_res$qval) & BEAM_res$qval < 1e-4, ]
rn <- intersect(as.character(BEAM_sig$gene_short_name), rownames(HSMM))
cat("[BEAM] branch-significant genes (qval < 1e-4):", length(rn), "\n")

## branched-expression heatmap, N modules (manuscript palette) -> svg / emf / pdf
## num_clusters via BEAM_NCLUST (default 3, matching the three-module Results narration).
## (set SKIP_BEAM_HEATMAP=1 to skip this slow 8k-gene render when only the trajectory recolours)
if (Sys.getenv("SKIP_BEAM_HEATMAP", "") == "") {
  NCL <- as.integer(Sys.getenv("BEAM_NCLUST", "3"))
  beam_cols <- colorRampPalette(c("#D8B2AE", "#F7F3EF", "#7187A7"))(62)
  heat_base <- file.path(OUT, paste0("plot_genes_branched_heatmap_cluster", NCL, "_1e4"))
  W_in <- 300 / 25.4; H_in <- 180 / 25.4    # = 300 mm x 180 mm
  pdf(NULL)
  ph_beam <- suppressWarnings(plot_genes_branched_heatmap(
    HSMM[rn, ], branch_point = 1, num_clusters = NCL,
    cores = as.integer(Sys.getenv("BEAM_HEATMAP_CORES", "1")),
    use_gene_short_name = TRUE, show_rownames = FALSE, hmcols = beam_cols, return_heatmap = TRUE))
  invisible(dev.off())
  heat_gtable <- prepare_branched_heatmap_gtable(ph_beam, legend_title = "Z-score")
  draw_branched <- function() { grid.newpage(); grid.draw(heat_gtable) }
  heat_svg <- paste0(heat_base, ".svg")
  svglite::svglite(heat_svg, width = W_in, height = H_in); draw_branched(); invisible(dev.off())
  svg_to_arial(heat_svg); fix_branched_heatmap_legend_title_svg(heat_svg)
  svg_to_emf(heat_svg); svg_to_pdf(heat_svg)    # -> Arial .svg + EMF/PDF from the fixed SVG
  if (!file.exists(paste0(heat_base, ".pdf"))) {
    grDevices::cairo_pdf(paste0(heat_base, ".pdf"), width = W_in, height = H_in, family = FF); draw_branched(); invisible(dev.off())
  }
  if (!is.null(ph_beam$annotation_row))                                             # gene -> module membership
    write.csv(data.frame(gene = rownames(ph_beam$annotation_row), cluster = as.integer(ph_beam$annotation_row$Cluster)),
              file.path(OUT, paste0("beam_cluster", NCL, "_membership.csv")), row.names = FALSE)
  cat("[fig] plot_genes_branched_heatmap_cluster", NCL, "_1e4 .svg/.emf/.pdf (+ membership csv) ->", OUT, "\n")
} else cat("[skip] BEAM branched heatmap (SKIP_BEAM_HEATMAP set)\n")

cat("DONE Monocle2 ->", OUT, "\n")
