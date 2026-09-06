#!/usr/bin/env Rscript
# Linux-adapted ssGSEA_GSE226077.R for the iMGL (induced microglia-like) Route2 application.
# Reads the precomputed per-(Sample,Cell_type) averaged expression (prep_ssgsea_gse226077.py),
# GSVA over the curated microglia identity/activation/stress/cycle/myeloid/lipid pathway list,
# limma D4-vs-D0, then exports ht_main_sample (top cell-type-specific pathways x sample,
# z-scored, split by day) as SVG (+PNG) with the original palette.
suppressMessages({
  library(GSVA); library(limma); library(msigdbr)
  library(dplyr); library(tidyr); library(tibble); library(stringr)
  library(ComplexHeatmap); library(circlize); library(svglite); library(systemfonts)
})
## ---- Arial everywhere (Liberation Sans metric substitute); literal "Arial" in exported SVG ----
FF <- "Arial"
try(register_font(name = FF, plain = match_fonts(FF)$path, bold = match_fonts(FF, weight = "bold")$path,
  italic = match_fonts(FF, italic = TRUE)$path, bolditalic = match_fonts(FF, weight = "bold", italic = TRUE)$path), silent = TRUE)  # skip if a real "Arial" is already installed
pdfFonts(Arial = pdfFonts()$Helvetica); postscriptFonts(Arial = postscriptFonts()$Helvetica)
ht_opt(heatmap_row_names_gp = gpar(fontfamily = FF), heatmap_column_names_gp = gpar(fontfamily = FF),
  heatmap_row_title_gp = gpar(fontfamily = FF), heatmap_column_title_gp = gpar(fontfamily = FF),
  legend_title_gp = gpar(fontfamily = FF, fontface = "bold", fontsize = 8), legend_labels_gp = gpar(fontfamily = FF, fontsize = 7))
mm2in <- function(mm) mm / 25.4
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
OUT <- file.path(Sys.getenv("IMGL_ROOT", "/disk1/maijl/deconv/deconv_20260610/prophead/GSE226077"), "ssGSEA")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
expr <- as.matrix(read.csv(file.path(OUT, "ssgsea_avg_expr.csv"), row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(OUT, "ssgsea_avg_meta.csv"), check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(all(colnames(expr) == meta$col_key))
cat(sprintf("expr %d genes x %d (sample,celltype); days: %s\n", nrow(expr), ncol(expr),
            paste(sort(unique(meta$Status)), collapse = ",")))

## ---- curated microglia pathway universe (filter msigdbr by gs_name -> version robust) ----
wanted <- c(
  "HALLMARK_COMPLEMENT","REACTOME_COMPLEMENT_CASCADE","KEGG_COMPLEMENT_AND_COAGULATION_CASCADES",
  "GOBP_COMPLEMENT_ACTIVATION","GOBP_CLASSICAL_PATHWAY_OF_COMPLEMENT_ACTIVATION",
  "KEGG_PHAGOSOME","GOBP_PHAGOCYTOSIS","GOBP_REGULATION_OF_PHAGOCYTOSIS","GOBP_PHAGOCYTOSIS_RECOGNITION",
  "GOBP_PHAGOCYTOSIS_ENGULFMENT","GOBP_SYNAPSE_PRUNING","GOBP_REGULATION_OF_SYNAPSE_ORGANIZATION",
  "KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION","REACTOME_ANTIGEN_PROCESSING_CROSS_PRESENTATION",
  "REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_AND_PRESENTATION","REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION",
  "GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION","GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION_OF_PEPTIDE_ANTIGEN",
  "GOBP_MICROGLIAL_CELL_ACTIVATION","GOBP_REGULATION_OF_MICROGLIAL_CELL_ACTIVATION","GOBP_MYELOID_LEUKOCYTE_ACTIVATION",
  "HALLMARK_INFLAMMATORY_RESPONSE","HALLMARK_TNFA_SIGNALING_VIA_NFKB","HALLMARK_IL6_JAK_STAT3_SIGNALING",
  "HALLMARK_INTERFERON_ALPHA_RESPONSE","HALLMARK_INTERFERON_GAMMA_RESPONSE","HALLMARK_ALLOGRAFT_REJECTION",
  "KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY","KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY",
  "KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION","KEGG_CHEMOKINE_SIGNALING_PATHWAY","KEGG_JAK_STAT_SIGNALING_PATHWAY",
  "REACTOME_INNATE_IMMUNE_SYSTEM","REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM","REACTOME_INTERFERON_SIGNALING",
  "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING","REACTOME_INTERFERON_GAMMA_SIGNALING","REACTOME_TOLL_LIKE_RECEPTOR_CASCADES",
  "GOBP_INFLAMMATORY_RESPONSE","GOBP_RESPONSE_TO_INTERFERON_GAMMA","GOBP_RESPONSE_TO_TYPE_I_INTERFERON",
  "GOBP_RESPONSE_TO_TUMOR_NECROSIS_FACTOR","GOBP_RESPONSE_TO_INTERLEUKIN_1","GOBP_RESPONSE_TO_LIPOPOLYSACCHARIDE",
  "GOBP_POSITIVE_REGULATION_OF_CYTOKINE_PRODUCTION","GOBP_CHEMOKINE_MEDIATED_SIGNALING_PATHWAY",
  "HALLMARK_APOPTOSIS","HALLMARK_P53_PATHWAY","HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
  "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY","HALLMARK_HYPOXIA",
  "REACTOME_CELLULAR_RESPONSE_TO_STRESS","REACTOME_UNFOLDED_PROTEIN_RESPONSE_UPR","REACTOME_APOPTOSIS",
  "REACTOME_PROGRAMMED_CELL_DEATH","GOBP_RESPONSE_TO_OXIDATIVE_STRESS",
  "GOBP_RESPONSE_TO_ENDOPLASMIC_RETICULUM_STRESS","GOBP_CELLULAR_RESPONSE_TO_STRESS",
  "GOBP_APOPTOTIC_PROCESS","GOBP_REGULATION_OF_APOPTOTIC_PROCESS",
  "HALLMARK_E2F_TARGETS","HALLMARK_G2M_CHECKPOINT","HALLMARK_MITOTIC_SPINDLE",
  "KEGG_CELL_CYCLE","KEGG_DNA_REPLICATION","REACTOME_CELL_CYCLE","REACTOME_DNA_REPLICATION",
  "REACTOME_CELL_CYCLE_CHECKPOINTS","REACTOME_G1_S_TRANSITION","GOBP_MITOTIC_CELL_CYCLE",
  "GOBP_DNA_REPLICATION","GOBP_CHROMOSOME_SEGREGATION",
  "KEGG_HEMATOPOIETIC_CELL_LINEAGE","GOBP_MYELOID_CELL_DIFFERENTIATION","GOBP_MACROPHAGE_DIFFERENTIATION",
  "GOBP_LEUKOCYTE_DIFFERENTIATION","GOBP_HEMATOPOIETIC_PROGENITOR_CELL_DIFFERENTIATION",
  "GOBP_HEMATOPOIETIC_STEM_CELL_DIFFERENTIATION","GOBP_ERYTHROCYTE_DIFFERENTIATION",
  "HALLMARK_CHOLESTEROL_HOMEOSTASIS","HALLMARK_FATTY_ACID_METABOLISM","HALLMARK_ADIPOGENESIS",
  "HALLMARK_OXIDATIVE_PHOSPHORYLATION","KEGG_PPAR_SIGNALING_PATHWAY","KEGG_FATTY_ACID_METABOLISM",
  "KEGG_PEROXISOME","KEGG_LYSOSOME","REACTOME_METABOLISM_OF_LIPIDS","REACTOME_CHOLESTEROL_BIOSYNTHESIS",
  "REACTOME_REGULATION_OF_CHOLESTEROL_BIOSYNTHESIS_BY_SREBP_SREBF","REACTOME_PPARA_ACTIVATES_GENE_EXPRESSION",
  "REACTOME_TRANSPORT_OF_FATTY_ACIDS","GOBP_CHOLESTEROL_EFFLUX","GOBP_CHOLESTEROL_TRANSPORT",
  "GOBP_REGULATION_OF_CHOLESTEROL_TRANSPORT","GOBP_LIPID_TRANSPORT","GOBP_LIPID_LOCALIZATION",
  "GOBP_FATTY_ACID_METABOLIC_PROCESS")
## MSigDB source: pinned to the saved snapshot (MSigDB 2025.1.Hs, the release used for the paper)
## for reproducibility, replacing the runtime msigdbr() call. Selection by gs_name -> version robust.
MSIG_SNAPSHOT <- "/disk1/maijl/deconv/script/msigdb_homosapiens.rds"
msig <- readRDS(MSIG_SNAPSHOT)
if ("db_version" %in% colnames(msig)) cat("MSigDB snapshot db_version:", paste(unique(msig$db_version), collapse = ","), "\n")
gene_sets <- msig %>% filter(gs_name %in% wanted) %>% select(gs_name, gene_symbol) %>%
  distinct() %>% split(.$gs_name) %>% lapply(function(x) unique(x$gene_symbol))
common_genes <- intersect(rownames(expr), unique(unlist(gene_sets)))
gene_sets <- lapply(gene_sets, function(x) intersect(x, common_genes))
gene_sets <- gene_sets[sapply(gene_sets, length) >= 10]
cat("Selected", length(gene_sets), "gene sets for GSVA (of", length(wanted), "requested)\n")

gsva_cache <- file.path(OUT, "gsva_scores.rds")     # GSVA is the slow step; reuse if columns match
gsva_scores <- if (file.exists(gsva_cache) && identical(colnames(readRDS(gsva_cache)), colnames(expr))) {
  cat("Reusing cached GSVA scores\n"); readRDS(gsva_cache)
} else { gs <- gsva(gsvaParam(exprData = expr, geneSets = gene_sets, kcdf = "Gaussian", minSize = 10, maxSize = 500))
  saveRDS(gs, gsva_cache); gs }
pathways <- rownames(gsva_scores)

## ---- reshape to (CellType-Pathway) x (Status-Sample) ----
key2 <- setNames(meta[, c("Sample", "Cell_type", "Status")], NULL)
samp <- meta$Sample[match(colnames(gsva_scores), meta$col_key)]
ct   <- meta$Cell_type[match(colnames(gsva_scores), meta$col_key)]
st   <- meta$Status[match(colnames(gsva_scores), meta$col_key)]
col_id <- paste(st, samp, sep = "-")
uc <- unique(col_id); uct <- unique(ct)
new_rows <- expand.grid(CellType = uct, Pathway = pathways, stringsAsFactors = FALSE)
new_rows$rn <- paste(new_rows$CellType, new_rows$Pathway, sep = "-")
M <- matrix(NA_real_, nrow = nrow(new_rows), ncol = length(uc), dimnames = list(new_rows$rn, uc))
for (j in seq_len(ncol(gsva_scores))) {
  rn <- paste(ct[j], pathways, sep = "-")
  M[rn, col_id[j]] <- gsva_scores[, j]
}
new_gsva_scores <- M[apply(M, 1, function(x) all(is.finite(x))), , drop = FALSE]
saveRDS(new_gsva_scores, file.path(OUT, "combined_gsva_scores.rds"))
sample_meta <- tibble(SampleID = colnames(new_gsva_scores)) %>%
  mutate(Status = sub("-.*", "", SampleID), Sample = sub("^[^-]*-", "", SampleID))
cat("new_gsva_scores:", nrow(new_gsva_scores), "celltype-pathway x", ncol(new_gsva_scores), "samples\n")

## ---- limma D4 - D0 ----
sample_meta$Status <- factor(sample_meta$Status)
design <- model.matrix(~0 + Status, data = sample_meta); colnames(design) <- levels(sample_meta$Status)
fit <- lmFit(new_gsva_scores, design)
fit2 <- eBayes(contrasts.fit(fit, makeContrasts(D4_vs_D0 = D4 - D0, levels = design)))
limma_result <- topTable(fit2, coef = 1, number = Inf, adjust.method = "fdr")
saveRDS(limma_result, file.path(OUT, "limma_result.rds"))
res_df <- limma_result %>% rownames_to_column("CellType_Pathway") %>%
  mutate(CellType = sub("-.*", "", CellType_Pathway), Pathway = sub("^[^-]*-", "", CellType_Pathway),
         direction = ifelse(logFC > 0, "D4_high", "D0_high"),
         rank_score = -log10(adj.P.Val + 1e-300) * abs(logFC), sig = adj.P.Val < 0.05)
cat("limma D4-D0 significant (FDR<0.05):", sum(res_df$sig, na.rm = TRUE), "\n")

## ===== ht_main_sample: top-3 per (CellType,direction) by rank_score, z-scored, split by day =====
sel <- res_df %>% group_by(CellType, direction) %>% arrange(desc(rank_score), .by_group = TRUE) %>%
  slice_head(n = 3) %>% ungroup() %>% pull(CellType_Pathway)
mat <- as.matrix(new_gsva_scores[sel, , drop = FALSE]); mat <- mat[apply(mat, 1, sd, na.rm = TRUE) > 0, , drop = FALSE]
mat_z <- t(scale(t(mat))); cap <- 2; mat_z[mat_z > cap] <- cap; mat_z[mat_z < -cap] <- -cap
ord <- order(sample_meta$Status); mat_z <- mat_z[, ord, drop = FALSE]; sm <- sample_meta[ord, ]
colnames(mat_z) <- sapply(strsplit(colnames(mat_z), "_"), function(x) paste(tail(x, 2), collapse = "_"))
day_levels <- sort(unique(sm$Status))
day_cols <- setNames(colorRampPalette(c("#f5eee9", "#dfcece", "#746377"))(length(day_levels)), day_levels)
## Figure 6e: the "celltype-PATHWAY" row labels are very long; to make them legible (7.5 pt) the row
## dendrogram and all legends are HIDDEN to free the horizontal name column (the binding constraint is
## body 82mm + longest label + padding <= 220mm canvas). D0/D3/D4 column split, z-score, Arial throughout.
ht_sample <- Heatmap(mat_z, name = "Row z-score",
  col = colorRamp2(c(-cap, 0, cap), c("#7187A7", "#F7F3EF", "#D8B2AE")),
  column_split = sm$Status, cluster_columns = FALSE, cluster_column_slices = FALSE,
  column_title_gp = gpar(fontfamily = FF, fontsize = 7, fontface = "bold"),
  row_names_gp = gpar(fontfamily = FF, fontsize = 7.5), column_names_gp = gpar(fontfamily = FF, fontsize = 7),
  column_names_rot = 45, rect_gp = gpar(col = "white", lwd = 0.3), row_names_max_width = unit(128, "mm"),
  show_row_dend = TRUE, row_dend_width = unit(3, "mm"), show_heatmap_legend = FALSE,
  width = unit(82, "mm"), height = unit(100, "mm"),
  top_annotation = HeatmapAnnotation(Status = sm$Status, col = list(Status = day_cols), show_legend = FALSE,
                                     annotation_name_gp = gpar(fontfamily = FF, fontsize = 6)),
  heatmap_legend_param = list(direction = "horizontal", legend_width = unit(30, "mm")))
WS <- 220; HS <- 150
png(file.path(OUT, "ht_main_sample.png"), width = WS, height = HS, units = "mm", res = 600)
draw(ht_sample, heatmap_legend_side = "bottom", annotation_legend_side = "bottom", padding = unit(c(3, 3, 3, 3), "mm"), merge_legend = TRUE); dev.off()
svglite(file.path(OUT, "ht_main_sample.svg"), width = mm2in(WS), height = mm2in(HS))
draw(ht_sample, heatmap_legend_side = "bottom", annotation_legend_side = "bottom", padding = unit(c(3, 3, 3, 3), "mm"), merge_legend = TRUE); dev.off()
svg_to_arial(file.path(OUT, "ht_main_sample.svg"))
cairo_pdf(file.path(OUT, "ht_main_sample.pdf"), width = mm2in(WS), height = mm2in(HS), family = FF)
draw(ht_sample, heatmap_legend_side = "bottom", annotation_legend_side = "bottom", padding = unit(c(3, 3, 3, 3), "mm"), merge_legend = TRUE); dev.off()
cat("[fig e] ht_main_sample (", nrow(mat_z), "rows x", ncol(mat_z), "samples ),", WS, "x", HS, "mm\n")
for (s in list.files(OUT, pattern = "\\.svg$", full.names = TRUE)) svg_to_emf(s)   # EMF for Office
for (f in list.files(OUT, pattern = "\\.pdf$", full.names = TRUE)) outline_pdf(f)  # outline PDF text
cat("DONE ssGSEA ->", OUT, "\n")
