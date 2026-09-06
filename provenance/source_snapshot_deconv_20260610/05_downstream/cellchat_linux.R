#!/usr/bin/env Rscript
# Linux-adapted cellchat.R for the Route2 GSE159585 COVID-vs-Normal downstream analysis.
# Builds CellChat objects for the COVID and Normal_18_24 generated predictions (re-using
# *_cellchat.rds if present), then exports ONLY the Figure-5 panels: the per-celltype
# interaction-difference barplot (b-right), the top-10 signalling-pathway counts (d), and the
# VEGF / MHC-II / COLLAGEN chord diagrams (e/f/g). All text Arial; every netVisual call guarded.
suppressMessages({
  library(CellChat); library(reticulate); library(Matrix)
  library(dplyr); library(tidyr); library(stringr); library(ggplot2); library(patchwork)
  library(svglite); library(systemfonts)
})
options(stringsAsFactors = FALSE)

## ---- typography: Arial everywhere (Liberation Sans is its metric-compatible Linux substitute);
## rewrite the resolved family to a literal "Arial" in every exported SVG. See ssGSEA_linux.R. ----
FF <- "Arial"
try(register_font(name = FF, plain = match_fonts(FF)$path, bold = match_fonts(FF, weight = "bold")$path,
  italic = match_fonts(FF, italic = TRUE)$path, bolditalic = match_fonts(FF, weight = "bold", italic = TRUE)$path), silent = TRUE)  # skip if a real "Arial" is already installed
pdfFonts(Arial = pdfFonts()$Helvetica); postscriptFonts(Arial = postscriptFonts()$Helvetica)
theme_fig5 <- ggplot2::theme_classic(base_size = 7, base_family = FF) +
  ggplot2::theme(axis.text = element_text(colour = "black"),
                 axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3))
svg_to_arial <- function(p) { x <- readLines(p, warn = FALSE)
  x <- gsub("Liberation Sans", "Arial", x, fixed = TRUE); x <- gsub("DejaVu Sans", "Arial", x, fixed = TRUE)
  # Strip svglite's textLength/lengthAdjust: Illustrator/Office mis-handle them and clip text.
  x <- gsub(" textLength='[^']*'", "", x); x <- gsub(" lengthAdjust='[^']*'", "", x)
  writeLines(x, p) }
# Convert PDF text to vector outlines so the file carries NO font name (Liberation Sans is the
# Linux Arial substitute; Illustrator would otherwise report "missing LiberationSans"). Glyphs are
# metric-identical to Arial, so this is visually Arial; use the .svg if editable text is needed.
outline_pdf <- function(f) {
  if (nchar(Sys.which("gs")) == 0) return(invisible())
  tmp <- paste0(f, ".ol")
  st <- tryCatch(system2("gs", c("-q", "-o", tmp, "-dNoOutputFonts", "-sDEVICE=pdfwrite", f),
                         stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  if (identical(st, 0L) && file.exists(tmp)) file.rename(tmp, f)
}
svg_to_emf <- function(svg) {  # EMF (editable vector for MS Office) from the final Arial SVG
  if (nchar(Sys.which("inkscape")) == 0) return(invisible())
  try(system2("inkscape", c(svg, "--export-type=emf", paste0("--export-filename=", sub("\\.svg$", ".emf", svg))),
              stdout = FALSE, stderr = FALSE), silent = TRUE)
}
# Sequential by default: CellChat's computeCommunProb permutation ships the whole expression
# matrix to each future worker, so multisession is serialization-bound here (16 workers ran
# SLOWER than 4). Running in-process (sequential) keeps the master at 100% CPU with zero IPC.
# Parallelism across conditions is achieved by launching the two prebuilds as separate processes.
.ccw <- as.integer(Sys.getenv("CC_WORKERS", "1"))
if (.ccw > 1) future::plan("multisession", workers = .ccw) else future::plan("sequential")
options(future.globals.maxSize = 8 * 1024^3)

DOWN <- Sys.getenv("DOWN_ROOT", "/disk1/maijl/deconv/deconv_20260605/downstream")
OUT  <- file.path(DOWN, "cellchat"); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
FILES <- list(
  Normal  = file.path(DOWN, "normal_18_24/generated_data_normal_18_24.h5ad"),
  COVID19 = file.path(DOWN, "covid/generated_data_covid.h5ad"))
COL_STATUS <- c(Normal = "#2F5D8A", COVID19 = "#E1895A")     # original palette

# Save every figure as BOTH .png and .svg. gg2: ggplot/patchwork object; dev2: base-graphics /
# ComplexHeatmap drawn by re-running fn() once per device. fname = basename (no extension).
gg2 <- function(p, fname, w, h) {     # exports PNG + editable SVG + editable PDF (all Arial)
  ggsave(file.path(OUT, paste0(fname, ".png")), p, width = w, height = h, units = "mm", dpi = 600)
  tryCatch({ ggsave(file.path(OUT, paste0(fname, ".svg")), p, width = w, height = h, units = "mm",
                    device = svglite::svglite); svg_to_arial(file.path(OUT, paste0(fname, ".svg"))) },
           error = function(e) cat("  [svg skip]", fname, ":", conditionMessage(e), "\n"))
  tryCatch(ggsave(file.path(OUT, paste0(fname, ".pdf")), p, width = w, height = h, units = "mm", device = grDevices::cairo_pdf),
           error = function(e) cat("  [pdf skip]", fname, ":", conditionMessage(e), "\n"))
}
dev2 <- function(fname, w, h, fn) {
  png(file.path(OUT, paste0(fname, ".png")), width = w, height = h, units = "mm", res = 600, family = FF); fn(); dev.off()
  tryCatch({ svglite::svglite(file.path(OUT, paste0(fname, ".svg")), width = w / 25.4, height = h / 25.4); fn(); dev.off()
             svg_to_arial(file.path(OUT, paste0(fname, ".svg"))) },
           error = function(e) cat("  [svg skip]", fname, ":", conditionMessage(e), "\n"))
  tryCatch({ grDevices::cairo_pdf(file.path(OUT, paste0(fname, ".pdf")), width = w / 25.4, height = h / 25.4, family = FF); fn(); dev.off() },
           error = function(e) cat("  [pdf skip]", fname, ":", conditionMessage(e), "\n"))
}
ok <- function(expr) tryCatch({ expr; TRUE }, error = function(e) { cat("  [skip]", conditionMessage(e), "\n"); FALSE })

use_python("/disk1/maijl/software/miniforge3/envs/pytorch/bin/python", required = TRUE)
ad <- import("anndata"); np <- import("numpy")

## ---------- build a CellChat object from a generated .h5ad ----------
build_cellchat <- function(h5ad, group = "Cell_type") {
  a <- ad$read_h5ad(h5ad)
  X <- t(as.matrix(py_to_r(a$X)))                      # genes x cells (X is log1p space)
  genes <- unlist(py_to_r(a$var_names$tolist())); rownames(X) <- genes
  if (max(X) < 20) X <- expm1(X)                       # -> linear
  libs <- pmax(Matrix::colSums(X), 1e-8)
  data.input <- log1p(sweep(X, 2, libs, "/") * 1e4)    # proper log-CP10K (original block-1 norm)
  meta <- py_to_r(a$obs); meta[[group]] <- as.character(meta[[group]])
  cells <- paste0("C", seq_len(ncol(data.input)))
  colnames(data.input) <- cells; rownames(meta) <- cells
  cc <- createCellChat(object = data.input, meta = meta, group.by = group)
  cc@DB <- CellChatDB.human
  cc <- subsetData(cc)
  cc <- identifyOverExpressedGenes(cc)
  cc <- identifyOverExpressedInteractions(cc)
  # nboot=20: 5x faster permutation null than the default 100; ample for the
  # group-level (triMean) communication-prob significance used in the downstream figures.
  cc <- computeCommunProb(cc, type = "triMean", nboot = 20)
  cc <- filterCommunication(cc, min.cells = 2)
  cc <- computeCommunProbPathway(cc)
  cc <- aggregateNet(cc)
  cc
}

get_or_build <- function(name) {
  rds <- file.path(OUT, paste0(tolower(gsub("19", "", name)), "_cellchat.rds"))
  if (file.exists(rds)) { cat("[load]", rds, "\n"); return(readRDS(rds)) }
  cat("[build]", name, "from", FILES[[name]], "\n")
  cc <- build_cellchat(FILES[[name]]); saveRDS(cc, rds)
  cat("[saved]", rds, " | cells:", ncol(cc@data), " pathways:", length(cc@netP$pathways), "\n")
  cc
}

ARGS <- commandArgs(trailingOnly = TRUE)
if ("prebuild_covid" %in% ARGS) {           # build COVID only (parallel prebuild), then exit
  covid_cc <- get_or_build("COVID19"); cat("[prebuild] COVID CellChat cached; exiting.\n"); quit(save = "no")
}
normal_cc <- get_or_build("Normal")
if ("prebuild_normal" %in% ARGS) { cat("[prebuild] Normal CellChat cached; exiting.\n"); quit(save = "no") }
covid_cc  <- get_or_build("COVID19")
cat("Normal pathways:", length(normal_cc@netP$pathways),
    "| COVID pathways:", length(covid_cc@netP$pathways), "\n")

## =====================================================================
## Fig 1 — per-cell-type interaction-strength difference (COVID - Normal)
## =====================================================================
ic <- function(cc) { net <- cc@net$weight; rowSums(net) + colSums(net) }
hc <- ic(normal_cc); cc_ <- ic(covid_cc)
cts <- union(names(hc), names(cc_))
## Conditions can have DIFFERENT cell-type sets (composition differs, e.g. prophead): hc[cts]/cc_[cts]
## then carry NA names for the cells absent in one condition, which data.frame rejects as row names
## ("row names contain missing values"). unname() the value vectors so CellType stays a plain column.
cmp <- data.frame(CellType = cts,
                  Healthy = unname(ifelse(is.na(hc[cts]), 0, hc[cts])),
                  COVID   = unname(ifelse(is.na(cc_[cts]), 0, cc_[cts])))
cmp$Difference <- cmp$COVID - cmp$Healthy
write.csv(cmp, file.path(OUT, "comparison_df.csv"), row.names = FALSE)
# Figure-5 panel (right of b): diverging per-celltype interaction difference, sorted, Arial.
p_bar <- ggplot(cmp, aes(reorder(CellType, Difference), Difference, fill = Difference >= 0)) +
  geom_col(width = 0.72) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey45") +
  scale_fill_manual(values = c(`FALSE` = "#2F5D8A", `TRUE` = "#E1895A"), guide = "none") +
  coord_flip() + labs(x = NULL, y = "Interaction Count Difference") +
  ## widen the value-axis range so the right-most tick label (e.g. 400) is not clipped at the panel edge
  scale_y_continuous(expand = expansion(mult = c(0.06, 0.10))) +
  theme_fig5 + theme(axis.text.y = element_text(size = 6), axis.text.x = element_text(size = 6.5),
                     axis.title.x = element_text(size = 7.5),
                     plot.margin = margin(t = 2, r = 6, b = 2, l = 2, unit = "pt"))
gg2(p_bar, "interaction_difference_barplot", 78, 168)
cat("[fig] interaction_difference_barplot.png/.svg\n")

## =====================================================================
## Fig 2 — top-10 signalling pathways by LR count (COVID)
## =====================================================================
if (!is.null(covid_cc@LR$LRsig) && nrow(covid_cc@LR$LRsig) > 0) {
  pc <- sort(table(covid_cc@LR$LRsig$pathway_name), decreasing = TRUE)
  pcd <- data.frame(pathway = names(head(pc, 10)), count = as.numeric(head(pc, 10)))
  write.csv(pcd, file.path(OUT, "covid_pathway_counts.csv"), row.names = FALSE)
  # Figure-5 panel d: vertical bars, ascending left->right, Arial.
  pcd <- pcd[order(pcd$count), ]; pcd$pathway <- factor(pcd$pathway, levels = pcd$pathway)
  p_top <- ggplot(pcd, aes(pathway, count)) +
    geom_col(fill = "#2F5D8A", width = 0.7) + labs(x = "Pathway", y = "Count") +
    theme_fig5 + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5),
                       axis.text.y = element_text(size = 6.5), axis.title = element_text(size = 7.5))
  gg2(p_top, "top10_signalingpathway", 92, 66)
  cat("[fig] top10_signalingpathway.png/.svg\n")
}

## =====================================================================
## Figure-5 panels e/f/g — COVID chord_gene for key pathways (single object, guarded).
## Tall canvas (100x128 mm) keeps the round chord high so the "Cell State" legend sits in the
## bottom margin without overlapping the arcs; small dots/gaps + Arial labels match the reference.
## =====================================================================
cov_ct <- levels(covid_cc@idents); cov_pw <- covid_cc@netP$pathways
# Original-script colours: the default scPalette(nlevels) here gives 62 interpolated hues that drift
# from the published figure (which had 61), so the per-celltype chord colours are pinned to the
# original palette (sampled from the reference Figure-5 legends).
# Per-chord palettes. VEGF + COLLAGEN use a blue-purple / light-yellow Morandi scheme (muted, low
# saturation, harmonious); MHC-II is restored to the original vivid Figure-5 palette (its antigen-
# presentation arcs read better in saturated colour). macro_alveolar / monocyte_classical appear in
# both VEGF and MHC-II, so the two palettes are kept separate and passed to chord() per diagram.
CHORD_COLORS <- c(            # VEGF + COLLAGEN: blue-purple / light-yellow Morandi
  artery_1 = "#7C93B8", vein_1 = "#9892BE", cap_arterial = "#D6C9A0", cap_venous = "#BBA6C4",
  macro_alveolar = "#8AA0C8", monocyte_classical = "#A98FC0", proliferating_epithelial = "#CFCB9A",
  fibroblast_matrix = "#9088BE", myofibroblast = "#D6CCA2")
MHCII_PAL <- c(               # MHC-II: restored original vivid palette (pre-Morandi)
  monocyte_classical = "#7E72B0", macro_alveolar = "#52C6D6", cDC_type_1 = "#2C6E80",
  CD4_Th1 = "#5BA84E", CD4_naive = "#B85AA0", CD4_EM = "#3E9AA8")
# Keep the chord canvas identical across the three panels so the central circles are the same size.
# The large original-style canvas leaves enough room for labels and legends without the 18 cm crop.
CHORD_W_MM <- 150
CHORD_H_MM <- 170
CHORD_SMALL_GAP <- 1.5
CHORD_BIG_GAP <- 6

chord <- function(src, tgt, sig, fname, w = CHORD_W_MM, h = CHORD_H_MM,
                  labcex = 0.5, smallgap = CHORD_SMALL_GAP, biggap = CHORD_BIG_GAP,
                  reduce = -1, pal = CHORD_COLORS, lpx = 8, lpy = 8) {
  src <- intersect(src, cov_ct); tgt <- intersect(tgt, cov_ct); sig <- intersect(sig, cov_pw)
  if (length(src) == 0 || length(tgt) == 0 || length(sig) == 0) { cat("[skip chord]", fname, "\n"); return(invisible()) }
  cu <- pal[unique(c(src, tgt))]                                # per-chord palette; fall back if any missing
  if (anyNA(cu)) cu <- NULL
  dev2(fname, w, h, function() {
    par(family = FF)
    # reduce>0 drops negligible-contribution gene sectors so dense chords (COLLAGEN) don't collide.
    ok(netVisual_chord_gene(covid_cc, sources.use = src, targets.use = tgt, signaling = sig,
                            color.use = cu, legend.pos.x = lpx, legend.pos.y = lpy, small.gap = smallgap,
                            big.gap = biggap, reduce = reduce, lab.cex = labcex, annotationTrackHeight = c(0.06)))
  })
  cat("[fig]", fname, ".png/.svg\n")
}
# e: VEGF (vascular permeability) | f: MHC-II (antigen presentation) | g: COLLAGEN/FN1/THBS/IGF (ECM/fibrosis)
# The `reduce` value is the LR-pair filtering step: it trims low-contribution sectors so dense pathways
# stay readable without overlapping labels. The three plots use slightly different label sizes and legend
# placements, but the canvas is fixed so the central circle stays visually consistent across panels.
# The dense MHC-II label fan would collide with the default bottom-right legend, so its legend is moved
# to the bottom (lpx = 24, lpy = 2). All four formats (png/svg/pdf/emf) are re-emitted per run.
# (thresholds documented in DeconvSC_0518_polished_v2.md, Methods "Representative pathways".)
chord(c("macro_alveolar", "proliferating_epithelial", "monocyte_classical"),
      c("artery_1", "vein_1", "cap_arterial", "cap_venous"), c("VEGF"), "chord_VEGF_vascular",
      w = CHORD_W_MM, h = CHORD_H_MM, biggap = CHORD_BIG_GAP, reduce = 0.025, labcex = 0.66)
chord(c("monocyte_classical", "macro_alveolar", "cDC_type_1"), c("CD4_Th1", "CD4_naive", "CD4_EM"),
      c("MHC-II"), "chord_MHCII_immune", pal = MHCII_PAL, labcex = 0.64, reduce = 0.025,
      w = CHORD_W_MM, h = CHORD_H_MM, lpx = 24, lpy = 2)
chord(c("fibroblast_matrix", "myofibroblast"), c("fibroblast_matrix", "myofibroblast"),
      c("COLLAGEN", "FN1", "THBS", "IGF"), "chord_COLLAGEN_fibroblast",
      w = CHORD_W_MM, h = CHORD_H_MM, labcex = 0.64, reduce = 0.03)

## ---- also emit EMF (Office vector) + outline PDF text (clean Illustrator import) ----
for (s in list.files(OUT, pattern = "\\.svg$", full.names = TRUE)) svg_to_emf(s)
for (f in list.files(OUT, pattern = "\\.pdf$", full.names = TRUE)) outline_pdf(f)
cat("DONE CellChat ->", OUT, "\n")
