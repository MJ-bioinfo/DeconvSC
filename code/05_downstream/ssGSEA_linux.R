#!/usr/bin/env Rscript
# Linux-adapted ssGSEA.R for the Route2 GSE159585 COVID-vs-Normal downstream analysis.
# Reads the precomputed per-(Sample,cell type) averaged log1p-CP10K matrices (prep_ssgsea_avg.py),
# runs GSVA over the curated MSigDB immune/fibrosis pathways, limma COVID19 - Normal, and exports the
# same three ComplexHeatmaps (all-celltype, specificity logFC, sample) with the original palettes.
suppressMessages({
  library(GSVA); library(limma); library(Biobase); library(msigdbr)
  library(dplyr); library(tidyr); library(tibble); library(purrr); library(stringr)
  library(ComplexHeatmap); library(circlize); library(RColorBrewer); library(svglite); library(systemfonts)
})

## ---- typography: Arial everywhere; export SVGs at 18 cm width ----
## "Arial" is not installed on Linux but is metric-compatible with Liberation Sans, which systemfonts
## resolves it to for cairo/svglite rendering. We (1) register the Liberation Sans variants under the
## name Arial, (2) alias Arial -> Helvetica metrics in the PostScript/PDF font DB so grid text
## measurement never warns, (3) set Arial as the global ComplexHeatmap default, and (4) rewrite the
## resolved family name back to a literal "Arial" in each exported SVG (layout is locked to the shared
## Arial/Liberation metrics via svglite's textLength, so real Arial renders identically downstream).
FF <- "Arial"
try(register_font(name = FF,
  plain      = match_fonts(FF)$path,
  bold       = match_fonts(FF, weight = "bold")$path,
  italic     = match_fonts(FF, italic = TRUE)$path,
  bolditalic = match_fonts(FF, weight = "bold", italic = TRUE)$path), silent = TRUE)  # skip if a real "Arial" is already installed
pdfFonts(Arial = pdfFonts()$Helvetica)
postscriptFonts(Arial = postscriptFonts()$Helvetica)
ht_opt(
  heatmap_row_names_gp    = gpar(fontfamily = FF),
  heatmap_column_names_gp = gpar(fontfamily = FF),
  heatmap_row_title_gp    = gpar(fontfamily = FF),
  heatmap_column_title_gp = gpar(fontfamily = FF),
  legend_title_gp         = gpar(fontfamily = FF, fontface = "bold", fontsize = 9),
  legend_labels_gp        = gpar(fontfamily = FF, fontsize = 8))
FIG_W_MM <- 180; FIG_W_IN <- FIG_W_MM / 25.4      # 18 cm
mm2in <- function(mm) mm / 25.4
svg_to_arial <- function(path) {                  # force the literal family name into the exported SVG
  x <- readLines(path, warn = FALSE)
  x <- gsub("Liberation Sans", "Arial", x, fixed = TRUE)
  x <- gsub("DejaVu Sans",     "Arial", x, fixed = TRUE)
  # Strip svglite's textLength/lengthAdjust (Illustrator/Office mis-handle them and clip text).
  x <- gsub(" textLength='[^']*'", "", x); x <- gsub(" lengthAdjust='[^']*'", "", x)
  writeLines(x, path)
}
# PDF text -> vector outlines (no embedded "LiberationSans" -> Illustrator imports cleanly; glyphs
# are metric-identical to Arial). Use the .svg for editable Arial text.
outline_pdf <- function(f) {
  if (nchar(Sys.which("gs")) == 0) return(invisible())
  tmp <- paste0(f, ".ol")
  st <- tryCatch(system2("gs", c("-q", "-o", tmp, "-dNoOutputFonts", "-sDEVICE=pdfwrite", f),
                         stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  if (identical(st, 0L) && file.exists(tmp)) file.rename(tmp, f)
}
# EMF (editable vector for MS Office) from the final Arial SVG.
svg_to_emf <- function(svg) {
  if (nchar(Sys.which("inkscape")) == 0) return(invisible())
  try(system2("inkscape", c(svg, "--export-type=emf", paste0("--export-filename=", sub("\\.svg$", ".emf", svg))),
              stdout = FALSE, stderr = FALSE), silent = TRUE)
}

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])
RELEASE_ROOT <- normalizePath(file.path(dirname(script_file), "..", ".."), mustWork = TRUE)
OUT <- file.path(Sys.getenv("DOWN_ROOT", file.path(RELEASE_ROOT, "work", "downstream", "GSE159585")), "ssGSEA")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
expr <- as.matrix(read.csv(file.path(OUT, "ssgsea_avg_expr.csv"), row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(OUT, "ssgsea_avg_meta.csv"), check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(all(colnames(expr) == meta$combined_id))
meta$status_sample <- paste(meta$disease_status, meta$Sample, sep = "_")
cat(sprintf("expr %d genes x %d (sample,celltype); %d celltypes; %d samples\n",
            nrow(expr), ncol(expr), length(unique(meta$CellType)), length(unique(meta$Sample))))

## ---- pathway universe: the FULL HALLMARK collection (50 sets) — balanced, spanning both the
##      COVID-up axes (inflammation, IFN, EMT, coagulation, hypoxia, complement...) AND
##      Normal/homeostatic axes (oxidative phosphorylation, fatty-acid metabolism, adipogenesis,
##      peroxisome, bile-acid/xenobiotic metabolism, myogenesis...). Plus curated non-HALLMARK
##      fibrosis/IFN/immune sets for depth. This removes the prior one-directional (COVID-only) bias. ----
extra <- c(
  "REACTOME_INTERFERON_SIGNALING", "GOBP_RESPONSE_TO_TYPE_I_INTERFERON",
  "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM", "KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION",
  "REACTOME_NEUTROPHIL_DEGRANULATION", "GOBP_REGULATION_OF_ENDOTHELIAL_CELL_MIGRATION",
  "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION", "GOBP_COLLAGEN_FIBRIL_ORGANIZATION",
  "REACTOME_ANTIGEN_PROCESSING_CROSS_PRESENTATION", "GOBP_REGULATION_OF_LEUKOCYTE_ACTIVATION",
  "KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY", "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY")
## MSigDB source: pinned to the saved snapshot (MSigDB 2025.1.Hs, the release used for the paper)
## for reproducibility, replacing the runtime msigdbr() call which would pull whatever release is
## currently installed (was drifting to 2026.1). Selection below is by gs_name -> version robust.
MSIG_SNAPSHOT <- Sys.getenv("MSIGDB_SNAPSHOT", "")
if (!nzchar(MSIG_SNAPSHOT) || !file.exists(MSIG_SNAPSHOT))
  stop("Set MSIGDB_SNAPSHOT to an authorized local MSigDB 2025.1.Hs RDS snapshot")
msig <- readRDS(MSIG_SNAPSHOT)
if ("db_version" %in% colnames(msig)) cat("MSigDB snapshot db_version:", paste(unique(msig$db_version), collapse = ","), "\n")
gene_sets <- msig %>% filter(grepl("^HALLMARK_", gs_name) | gs_name %in% extra) %>%
  select(gs_name, gene_symbol) %>% distinct() %>% split(.$gs_name) %>%
  lapply(function(x) unique(x$gene_symbol))
cat("pathway universe:", length(gene_sets), "sets (full HALLMARK + curated immune/fibrosis)\n")

common_genes <- intersect(rownames(expr), unique(unlist(gene_sets)))
gene_sets <- lapply(gene_sets, function(x) intersect(x, common_genes))
gene_sets <- gene_sets[sapply(gene_sets, length) >= 10]
cat("Selected", length(gene_sets), "gene sets for GSVA analysis\n")

## GSVA is the slow step; reuse the cached scores when they already match the current columns.
gsva_cache <- file.path(OUT, "gsva_scores.rds")
gsva_scores <- if (file.exists(gsva_cache) &&
                   identical(colnames(readRDS(gsva_cache)), colnames(expr))) {
  cat("Reusing cached GSVA scores (", gsva_cache, ")\n"); readRDS(gsva_cache)
} else {
  param <- gsvaParam(exprData = expr, geneSets = gene_sets, kcdf = "Gaussian", minSize = 10, maxSize = 500)
  gs <- gsva(param); saveRDS(gs, gsva_cache); gs
}
## ---- relevance check: drop biologically-irrelevant, non-respiratory tissue-identity HALLMARK
## sets. The pathway universe is the FULL HALLMARK collection (for a balanced background), so a few
## off-tissue programs can surface in the top pathways purely by spurious marker overlap (e.g. a
## goblet/secretory marker overlapping HALLMARK_PANCREAS_BETA_CELLS) yet are uninterpretable for a
## COVID-19 lung study. They are removed from EVERY ssGSEA result/figure here. Extend the vector to
## drop further off-tissue pathways; this is the single source of truth for the relevance filter.
PATHWAY_BLOCKLIST <- c("HALLMARK_PANCREAS_BETA_CELLS", "HALLMARK_SPERMATOGENESIS",
                       "HALLMARK_MYOGENESIS", "HALLMARK_ADIPOGENESIS", "HALLMARK_BILE_ACID_METABOLISM")
pathways <- rownames(gsva_scores)
.drop_pw <- intersect(pathways, PATHWAY_BLOCKLIST)
if (length(.drop_pw)) cat("relevance blocklist: dropping", length(.drop_pw),
                          "off-tissue pathway(s):", paste(.drop_pw, collapse = ", "), "\n")
pathways <- setdiff(pathways, PATHWAY_BLOCKLIST)

## ---- reshape to (CellType-Pathway) x (status_Sample), tracked via explicit columns ----
col_lut <- meta %>% select(combined_id, Sample, CellType, disease_status, status_sample)
long <- expand.grid(combined_id = colnames(gsva_scores), Pathway = pathways,
                    stringsAsFactors = FALSE) %>%
  left_join(col_lut, by = "combined_id") %>%
  mutate(score = mapply(function(p, cid) gsva_scores[p, cid], Pathway, combined_id),
         CP = paste(CellType, Pathway, sep = "@@"))
new_long <- long %>% select(CP, status_sample, score)
new_gsva_scores <- new_long %>%
  pivot_wider(names_from = status_sample, values_from = score) %>%
  column_to_rownames("CP") %>% as.matrix()
row_info <- tibble(CP = rownames(new_gsva_scores)) %>%
  separate(CP, into = c("CellType", "Pathway"), sep = "@@", remove = FALSE)
sample_meta <- tibble(status_sample = colnames(new_gsva_scores)) %>%
  separate(status_sample, into = c("disease_status", "SampleID"), sep = "_",
           remove = FALSE, extra = "merge")
cat("new_gsva_scores:", dim(new_gsva_scores)[1], "celltype-pathway x", dim(new_gsva_scores)[2], "samples\n")
saveRDS(new_gsva_scores, file.path(OUT, "combined_gsva_scores.rds"))

## ===== limma COVID19 - Normal (positive logFC = COVID-elevated; more intuitive) =====
design <- model.matrix(~0 + disease_status, data = sample_meta)
colnames(design) <- levels(as.factor(sample_meta$disease_status))   # COVID19, Normal
fit <- lmFit(new_gsva_scores, design)
contrast.matrix <- makeContrasts(contrasts = paste(colnames(design)[1], "-", colnames(design)[2]),
                                 levels = colnames(design))          # COVID19 - Normal
fit2 <- eBayes(contrasts.fit(fit, contrast.matrix))
results <- topTable(fit2, coef = 1, number = Inf, adjust.method = "fdr")
saveRDS(results, file.path(OUT, "limma_result.rds"))
res_df <- results %>% rownames_to_column("CP") %>%
  left_join(row_info, by = "CP") %>%
  mutate(direction = ifelse(logFC > 0, "COVID19_high", "Normal_high"),
         rank_score = -log10(adj.P.Val + 1e-300) * abs(logFC),
         sig = adj.P.Val < 0.05 & abs(logFC) > 0.2)
cat("limma: significant celltype-pathway (FDR<0.05 & |logFC|>0.2):", sum(res_df$sig, na.rm = TRUE), "\n")

## ===== Panel a: ht_main_sdlogfc_specific — WIDE top strip for Figure 5 =====
## Top-20 cell-type-specific pathways (both directions, by sd*max|logFC|) x 61 celltypes.
## Layout matches Figure 5a: wide strip, 45-deg celltype columns, pathway names at right,
## small significance dots (sized by -log10 FDR) proportional to the cells, and a horizontal
## COVID19-Normal score legend pinned to the bottom-right corner.
pathway_specificity <- res_df %>% group_by(Pathway) %>%
  summarise(n_sig = sum(adj.P.Val < 0.05 & abs(logFC) > 0.2, na.rm = TRUE),
            max_abs_logFC = max(abs(logFC), na.rm = TRUE),
            sd_logFC = sd(logFC, na.rm = TRUE),
            specificity_score = sd_logFC * max_abs_logFC, .groups = "drop") %>%
  filter(n_sig >= 1) %>% arrange(desc(specificity_score))
show_pathways <- pathway_specificity %>% slice_head(n = 20) %>% pull(Pathway)
if (length(show_pathways) == 0) show_pathways <- unique(res_df$Pathway)   # fallback
cat("Panel-a pathways:", length(show_pathways), "(top-20 by specificity)\n")
plot_main_df <- res_df %>% filter(Pathway %in% show_pathways)
logFC_mat <- plot_main_df %>% select(Pathway, CellType, logFC) %>% distinct() %>%
  pivot_wider(names_from = CellType, values_from = logFC, values_fill = 0) %>%
  column_to_rownames("Pathway") %>% as.matrix()
fdr_mat <- plot_main_df %>% select(Pathway, CellType, adj.P.Val) %>% distinct() %>%
  pivot_wider(names_from = CellType, values_from = adj.P.Val, values_fill = 1) %>%
  column_to_rownames("Pathway") %>% as.matrix()
fdr_mat <- fdr_mat[rownames(logFC_mat), colnames(logFC_mat), drop = FALSE]
## limma can return NA logFC for celltype-pathway pairs (the "Partial NA coefficients" warning;
## happens on some proportion schemes, e.g. prophead). pivot_wider's values_fill only fills MISSING
## combinations, not present-but-NA ones, so those NAs survive and break Heatmap column clustering
## (hclust: "NA/NaN/Inf in foreign function call"). Treat a non-finite difference as "no change" (0).
logFC_mat[!is.finite(logFC_mat)] <- 0
fdr_mat[!is.finite(fdr_mat)] <- 1
## Drop cell-type columns whose score is entirely empty (all logFC == 0 after the NA->0 fill above —
## i.e. limma returned NA for every shown pathway, typically single-condition cell types such as
## CD8-RM / CD8-naive). They render as blank columns, so remove them from both matrices.
.empty_ct <- apply(logFC_mat, 2, function(x) all(x == 0))
if (any(.empty_ct)) {
  cat("[fig a] dropping", sum(.empty_ct), "all-empty celltype column(s):",
      paste(colnames(logFC_mat)[.empty_ct], collapse = ", "), "\n")
  logFC_mat <- logFC_mat[, !.empty_ct, drop = FALSE]
  fdr_mat   <- fdr_mat[, colnames(logFC_mat), drop = FALSE]
}
lim <- max(quantile(abs(logFC_mat), 0.98, na.rm = TRUE), 0.2)
sig_score <- -log10(fdr_mat); sig_score[is.infinite(sig_score)] <- max(sig_score[is.finite(sig_score)], na.rm = TRUE)
sig_score[fdr_mat >= 0.05] <- NA; sig_cap <- 5; sig_score[sig_score > sig_cap] <- sig_cap
col_fun_a <- colorRamp2(c(-lim, 0, lim), c("#7187A7", "#F7F3EF", "#D8B2AE"))
ht2 <- Heatmap(logFC_mat, name = "logFC", col = col_fun_a,
  cluster_rows = TRUE, cluster_columns = TRUE, show_heatmap_legend = FALSE,
  row_names_gp = gpar(fontsize = 6.5, fontfamily = FF), column_names_gp = gpar(fontsize = 5.5, fontfamily = FF),
  row_names_max_width = unit(72, "mm"), column_names_max_height = unit(30, "mm"), column_names_rot = 45,
  rect_gp = gpar(col = "white", lwd = 0.3),
  width = unit(126, "mm"), height = unit(56, "mm"),
  cell_fun = function(j, i, x, y, w, h, fill) {
    s <- sig_score[i, j]
    if (!is.na(s)) grid.points(x, y, pch = 16, size = unit(0.25 + 0.6 * s / sig_cap, "mm"))
  })
lgd_a <- Legend(col_fun = col_fun_a, title = "COVID19 - Normal\nGSVA score", direction = "horizontal",
  legend_width = unit(34, "mm"), at = c(-round(lim, 1), 0, round(lim, 1)),
  title_gp = gpar(fontsize = 8, fontfamily = FF, fontface = "bold"), labels_gp = gpar(fontsize = 7, fontfamily = FF))
WA <- 214; HA <- 92
draw_a <- function() {
  draw(ht2, padding = unit(c(2, 2, 2, 2), "mm"))
  draw(lgd_a, x = unit(1, "npc") - unit(4, "mm"), y = unit(4, "mm"), just = c("right", "bottom"))
}
png(file.path(OUT, "ht_main_sdlogfc_specific.png"), width = WA, height = HA, units = "mm", res = 600); draw_a(); dev.off()
tryCatch({ svglite(file.path(OUT, "ht_main_sdlogfc_specific.svg"), width = mm2in(WA), height = mm2in(HA)); draw_a(); dev.off()
           svg_to_arial(file.path(OUT, "ht_main_sdlogfc_specific.svg")) },
         error = function(e) { cat("[svg skip] ht_main:", conditionMessage(e), "\n"); try(dev.off(), silent = TRUE) })
tryCatch({ cairo_pdf(file.path(OUT, "ht_main_sdlogfc_specific.pdf"), width = mm2in(WA), height = mm2in(HA), family = FF); draw_a(); dev.off() },
         error = function(e) { cat("[pdf skip] ht_main:", conditionMessage(e), "\n"); try(dev.off(), silent = TRUE) })
cat("[fig a] ht_main_sdlogfc_specific (", nrow(logFC_mat), "pathways x", ncol(logFC_mat), "celltypes ),", WA, "x", HA, "mm\n")

## ===== Figure 3: sample_heatmap.png (top per celltype/direction, z-score across samples) =====
celltype_rank <- res_df %>% filter(sig) %>% group_by(CellType) %>%
  summarise(total_score = sum(rank_score, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(total_score))
show_celltypes <- celltype_rank %>% slice_head(n = 12) %>% pull(CellType)
within_var <- apply(new_gsva_scores, 1, function(x) mean(tapply(x, sample_meta$disease_status, var, na.rm = TRUE), na.rm = TRUE))
res_df$within_var <- within_var[res_df$CP]
show_sub_df <- res_df %>% mutate(display_score = rank_score * (within_var + 1e-4)) %>%
  filter(sig, CellType %in% show_celltypes) %>%
  group_by(CellType, direction) %>% arrange(desc(display_score), .by_group = TRUE) %>%
  slice_head(n = 1) %>% ungroup()
sample_meta <- sample_meta %>% mutate(disease_status = factor(disease_status, levels = c("Normal", "COVID19"))) %>%
  arrange(disease_status)
sel <- unique(show_sub_df$CP); sel <- sel[sel %in% rownames(new_gsva_scores)]
mat <- as.matrix(new_gsva_scores[sel, sample_meta$status_sample, drop = FALSE])
mat <- mat[apply(mat, 1, sd, na.rm = TRUE) > 0.03, , drop = FALSE]
top_ha <- HeatmapAnnotation(Status = sample_meta$disease_status,
  col = list(Status = c("COVID19" = "#E8D2B3", "Normal" = "#9B8E95")),
  annotation_name_gp = gpar(fontfamily = FF, fontsize = 8))
mat_z <- t(scale(t(mat))); mat_z[!is.finite(mat_z)] <- 0   # zero-variance/NA rows -> 0 (keep clustering finite)
cap <- max(0.8, min(quantile(abs(mat_z), 0.90, na.rm = TRUE), 1.2))
mat_z[mat_z > cap] <- cap; mat_z[mat_z < -cap] <- -cap
rownames(mat_z) <- gsub("@@", " - ", rownames(mat_z))
## row names are long "celltype - PATHWAY" strings; 7 pt + a wide name column keeps the longest
## (~64 chars) inside the 18 cm width without clipping.
ht3 <- Heatmap(mat_z, name = "Row z-score",
  col = colorRamp2(c(-cap, 0, cap), c("#7187A7", "#F7F3EF", "#D8B2AE")),
  top_annotation = top_ha, column_split = sample_meta$disease_status,
  column_title_gp = gpar(fontfamily = FF, fontsize = 10),
  cluster_columns = TRUE, cluster_column_slices = FALSE, cluster_rows = TRUE,
  row_names_gp = gpar(fontsize = 7.5, fontfamily = FF), column_names_gp = gpar(fontsize = 7, fontfamily = FF),
  rect_gp = gpar(col = "white", lwd = 0.4),
  width = unit(58, "mm"), height = unit(100, "mm"), row_names_max_width = unit(110, "mm"),
  heatmap_legend_param = list(direction = "horizontal", legend_width = unit(30, "mm")))
WB <- 182; HB <- 210
draw_b <- function() draw(ht3, heatmap_legend_side = "bottom", annotation_legend_side = "bottom",
                          padding = unit(c(8, 4, 8, 4), "mm"), merge_legend = TRUE)
png(file.path(OUT, "sample_heatmap.png"), width = WB, height = HB, units = "mm", res = 600); draw_b(); dev.off()
tryCatch({ svglite(file.path(OUT, "sample_heatmap.svg"), width = mm2in(WB), height = mm2in(HB)); draw_b(); dev.off()
           svg_to_arial(file.path(OUT, "sample_heatmap.svg")) },
         error = function(e) { cat("[svg skip] sample_heatmap:", conditionMessage(e), "\n"); try(dev.off(), silent = TRUE) })
tryCatch({ cairo_pdf(file.path(OUT, "sample_heatmap.pdf"), width = mm2in(WB), height = mm2in(HB), family = FF); draw_b(); dev.off() },
         error = function(e) { cat("[pdf skip] sample_heatmap:", conditionMessage(e), "\n"); try(dev.off(), silent = TRUE) })
cat("[fig b] sample_heatmap (", nrow(mat_z), "rows ),", WB, "x", HB, "mm\n")

## ---- also emit EMF (Office vector) + outline PDF text (clean Illustrator import) ----
for (s in list.files(OUT, pattern = "\\.svg$", full.names = TRUE)) svg_to_emf(s)
for (f in list.files(OUT, pattern = "\\.pdf$", full.names = TRUE)) outline_pdf(f)
cat("DONE ssGSEA ->", OUT, "\n")
