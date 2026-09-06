## Plot profile-wise and pseudo-bulk PCC per dataset.
##
## Per total_visualization.R strategy, fine cell types and major cell types are
## plotted on SEPARATE figures:
##   - Fine plot:  BayesPrism / DISSECT / TAPE / DeconvSC  (CIBERSORTx excluded;
##                 it only emits major-level estimates)
##   - Major plot: same four methods aggregated fine -> major using the
##                 dataset-specific major_map, PLUS CIBERSORTx's native
##                 major-level values.
##
## HCA_fold1 special-case: the CIBERSORTxHiRes output for Job26 is constant
## (every sample-gene cell is 1, so per-sample Pearson is undefined and the
## unified pipeline wrote NaNs).  Fall back to the externally pre-computed
## `HCA_pearson_corr_results.csv` for the major bar.

library(dplyr)
library(tidyr)
library(ggplot2)
library(data.table)
## Typography: render in Arial. "Arial" is not installed on this Linux host but Liberation Sans is
## its metric-identical clone (fontconfig resolves Arial -> Liberation Sans); we register it under
## the literal name "Arial", and the exported SVG text is relabelled to "Arial" + also written as EMF
## (inkscape) for MS Office. (Replaces the earlier Windows extrafont path / the "sans" fallback.)
suppressMessages({ library(systemfonts); library(svglite) })
FF <- "Arial"
try(register_font(name = FF, plain = match_fonts(FF)$path, bold = match_fonts(FF, weight = "bold")$path,
              italic = match_fonts(FF, italic = TRUE)$path,
              bolditalic = match_fonts(FF, weight = "bold", italic = TRUE)$path), silent = TRUE)  # skip if a real "Arial" is already installed
try(pdfFonts(Arial = pdfFonts()$Helvetica), silent = TRUE)
BASE_FAM <- FF
svg_to_arial <- function(path) {       # force the literal family name into the exported SVG
  if (!file.exists(path)) return(invisible())
  x <- readLines(path, warn = FALSE)
  x <- gsub("Liberation Sans", "Arial", x, fixed = TRUE); x <- gsub("DejaVu Sans", "Arial", x, fixed = TRUE)
  x <- gsub(" textLength='[^']*'", "", x); x <- gsub(" lengthAdjust='[^']*'", "", x)
  writeLines(x, path)
}
svg_to_emf <- function(svg) {          # SVG -> EMF (editable vector for MS Office)
  if (!file.exists(svg) || nchar(Sys.which("inkscape")) == 0) return(invisible())
  try(system2("inkscape", c(svg, "--export-type=emf", paste0("--export-filename=", sub("\\.svg$", ".emf", svg))),
              stdout = FALSE, stderr = FALSE), silent = TRUE)
}

## Paths resolve from this release root unless BENCHMARK_TABLES or FIGURE_OUT is set.
.args <- commandArgs(FALSE); .sp <- sub("^--file=", "", .args[grep("^--file=", .args)])
script_dir   <- if (length(.sp)) normalizePath(dirname(.sp)) else normalizePath(".")
release_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
results_root <- Sys.getenv("BENCHMARK_TABLES", file.path(release_root, "benchmark_tables", "expression_prediction", "03_result_tables"))
out_dir      <- Sys.getenv("FIGURE_OUT", file.path(release_root, "work", "expression_benchmark_figures"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

datasets     <- c("HCA_fold2", "GSE141115", "GSE159585")
methods      <- c("BayesPrism", "CIBERSORTx", "DISSECT", "TAPE", "cVAE")
fine_methods <- c("BayesPrism", "DISSECT", "TAPE", "DeconvSC")
method_order <- c("BayesPrism", "CIBERSORTx", "DISSECT", "TAPE", "DeconvSC")

palette <- c("BayesPrism" = "#86ADC0",
             "CIBERSORTx" = "#7187A7",
             "DISSECT"    = "#AA96CA",
             "TAPE"       = "#989EB0",
             "DeconvSC"   = "#DEBBB1")

cell_type_label_maps <- list(
  HCA_fold1 = data.frame(
    old = c("StromalFibroblast", "Myeloid", "Lymphoid",
            "AlveolarEpithelial", "CapillaryVenousEndothelial", "Others"),
    new = c("Stromal cells and fibroblast", "Myeloid cells", "Lymphoid cells",
            "Alveolar epithelial cells", "Endothelial cells",
            "Others"),
    stringsAsFactors = FALSE
  ),
  GSE141115 = data.frame(
    old = c("B", "CNT", "DCT", "Endo", "Fib", "MC", "MPH", "NK", "Neut",
            "PT", "Podo", "T", "aLOH", "CD_PC","CD_IC","CD_Trans"),
    new = c("B cells", "Connecting tubule cells",
            "Distal convoluted tubule cell", "Endothelial cells",
            "Fibroblasts", "Macrophages", "Mesangium cells", "NK cells",
            "Neutrophils", "Proximal tubule cells", "Podocytes", "T cells",
            "Ascending loop of Henle", "Pricipal CD cells", "Intercalated CD cells","Transitional CD cells"),
    stringsAsFactors = FALSE
  )
)
cell_type_label_maps$HCA_fold2 <- cell_type_label_maps$HCA_fold1

## --- major_map (mirrors extended_multi_method_evaluation.py) ----------------

major_maps <- list(
  HCA_fold1 = list(
    AlveolarEpithelial = c("AT0","AT1","AT2","AT2 proliferating","pre-TB secretory",
                           "Basal resting","Suprabasal","Club (nasal)","Club (non-nasal)",
                           "Goblet (bronchial)","Goblet (nasal)","Goblet (subsegmental)",
                           "Multiciliated (non-nasal)","Deuterosomal","Ionocyte",
                           "Neuroendocrine","Mesothelium","Club","Goblet",
                           "Transitional AT2"),
    CapillaryVenousEndothelial = c("EC aerocyte capillary","EC general capillary",
                                   "EC venous pulmonary","EC venous systemic",
                                   "EC arterial","Lymphatic EC mature",
                                   "Lymphatic EC differentiating"),
    Lymphoid = c("B cells","CD4 T cells","CD8 T cells","NK cells","Plasma cells",
                 "T cells proliferating","Hematopoietic stem cells"),
    Myeloid = c("Alveolar macrophages","Alveolar Mph CCL3+","Alveolar Mph MT-positive",
                "Alveolar Mph proliferating","Monocyte-derived Mph",
                "Interstitial Mph perivascular","Classical monocytes",
                "Non-classical monocytes","DC1","DC2","Plasmacytoid DCs",
                "Migratory DCs","Mast cells","Macrophages MARCO-",
                "Macrophages SPP1 high","DC monocyte"),
    StromalFibroblast = c("Adventitial fibroblasts","Alveolar fibroblasts",
                          "Peribronchial fibroblasts","Subpleural fibroblasts",
                          "Myofibroblasts","Smooth muscle","Smooth muscle FAM83D+",
                          "SM activated stress response","Pericytes",
                          "Activated myofibroblasts","Fibroblasts PLIN2+")
  ),
  GSE141115 = list(
    `Collecting duct cells` = c("CD_IC","CD_PC","CD_Trans","CD_IC_A","CD_IC_B"),
    `Glomerular cells` = c("MC","Podo"),
    `Immune cells` = c("B","MPH","NK","Neut","T"),
    `Tubular cells` = c("PT","DCT","CNT","aLOH"),
    `Stromal cells` = c("Endo","Fib")
  ),
  GSE159585 = list(
    `B cells` = c("B_cell","plasma"),
    `Epithelial cells` = c("AT1","AT2","ciliated","club","goblet","basal",
                     "proliferating_epithelial","neuroendocrine","AT0",
                     "rare_epithelial","alveolar_AT0","epithelial_proliferating",
                     "epithelial_airway_ciliated","epithelial_airway_secretory",
                     "mesothelial"),
    Fibroblast   = c("fibroblast","adventitial_fibroblast","alveolar_fibroblast",
                     "myofibroblast","matrix_fibroblast","peribronchial_fibroblast",
                     "fibroblast_intermediate","fibroblast_matrix","fibromyocyte"),
    LEC          = c("LEC","lymphatic_EC"),
    `Myeloid cells` = c("macrophage_alv","macrophage_interstitial","monocyte_classical",
                     "monocyte_nonclassical","dendritic_cell_plasmacytoid",
                     "dendritic_cell_conventional","mast_cell","neutrophil",
                     "macrophage","DC","monocyte","macro_alveolar",
                     "macro_alveolar_SPP1","macro_HS3ST2","macrophage_interstitial",
                     "monocyte_IL1B","monocyte_non_classical","cDC_type_1",
                     "cDC_type_2","migDC","pDC","granulocyte","proliferating_myeloid"),
    `NK cells` = c("NK","NK_KLRC1"),
    `SMC and Pericyte` = c("SMC","pericyte","vascular_SMC","airway_SMC","smooth_muscle"),
    `T cells` = c("CD4_EM","CD4_naive","CD8_EMRA","CD4_Treg","CD8_EM","CD4_Th17",
                     "CD8_RM","CD8_naive","proliferating_lymphoid","CD4_Th1",
                     "CD8_MAIT","CD8_gamma_delta"),
    `Vascular cells` = c("cap_1","cap_2","cap_CX3CL1","cap_PB","vein_1","vein_2",
                     "vein_PB","arterial","venous","artery_1","artery_2",
                     "cap_arterial","cap_venous","large_vessel",
                     "proliferating_endothelial")
  )
)

## The current result table directory is HCA_fold2; it uses the same HLCA major
## groups as the older HCA_fold1 outputs referenced by the original script.
major_maps$HCA_fold2 <- major_maps$HCA_fold1

## CIBERSORTx emits its OWN major-level cell-type names (e.g. "Collecting_Duct"); map them onto the
## display names (major_maps keys) used by the aggregated methods so both land on the SAME x position
## (needed after the major_maps keys were renamed to spaced display names).
cibersortx_major_aliases <- list(
  GSE141115 = c(Collecting_Duct = "Collecting duct cells", Glomerular = "Glomerular cells",
                Immune = "Immune cells", Tubular_Epithelial = "Tubular cells",
                Vascular_Stromal = "Stromal cells"),
  GSE159585 = c(Bcell = "B cells", Epithelial = "Epithelial cells", Fibroblast = "Fibroblast",
                LEC = "LEC", Myeloid = "Myeloid cells", NKcell = "NK cells",
                SMC_Pericyte = "SMC and Pericyte", T_cell = "T cells", Vascular = "Vascular cells")
)

## CIBERSORTx fallback for HCA (HiRes output was a constant matrix on fold1). On this tree the
## unified pipeline's HCA_fold2/CIBERSORTx pearson CSVs are finite, so the fallback is not reached;
## we still point it at the local profile/pseudobulk CSVs (not the old Windows path) for safety.
cibersortx_fallback <- list(
  HCA_fold2 = list(
    profile    = file.path(results_root, "HCA_fold2", "CIBERSORTx", "pearson_profile_all_genes.csv"),
    pseudobulk = file.path(results_root, "HCA_fold2", "CIBERSORTx", "pearson_pseudobulk_all_genes.csv")
  )
)
cibersortx_fallback$HCA_fold1 <- cibersortx_fallback$HCA_fold2

## --- helpers ----------------------------------------------------------------
## ============================================================================
## Cross-dataset marker-gene boxplots (port of Python main_boxes)
## 每个 axis 按数据集分别出图；读 <root>/<ds>/<method>/pearson_<axis>_marker_genes.csv
## ============================================================================
main_methods <- c(BayesPrism = "BayesPrism",   # names = 磁盘文件夹名
                  DISSECT    = "DISSECT",       # values = 显示名
                  TAPE       = "TAPE",
                  cVAE       = "DeconvSC",
                  CIBERSORTx = "CIBERSORTx"
                  )
# Marker-boxplot palette == bar-plot `palette` (single source of truth), so every method's colour
# in the profile-wise/gene-wise/sample-wise boxplots matches its colour in the per-cell-type bars.
main_palette <- palette
main_methods <- main_methods[c("BayesPrism", "CIBERSORTx", "DISSECT", "TAPE", "cVAE")]
ds_label <- c(HCA_fold2 = "HLCA simulation",
              GSE141115 = "GSE141115 (mouse kidney)",
              GSE159585 = "GSE159585 (human lung)")

load_vec <- function(dataset, folder, axis, geneset = "marker_genes") {
  f <- file.path(results_root, dataset, folder,
                 sprintf("pearson_%s_%s.csv", axis, geneset))
  if (!file.exists(f)) return(NULL)
  raw <- tryCatch(fread(f), error = function(e) NULL)
  if (is.null(raw) || nrow(raw) == 0) return(NULL)
  v <- suppressWarnings(as.numeric(raw[[ncol(raw)]]))   # 最后一列 = 数值
  v <- v[is.finite(v)]
  if (length(v) < 3) return(NULL)
  v
}

build_marker_long <- function(axes, geneset = "marker_genes") {
  rows <- list()
  for (ds in datasets) for (ax in axes) {
    for (i in seq_along(main_methods)) {
      folder <- names(main_methods)[i]; disp <- main_methods[[i]]
      v <- load_vec(ds, folder, ax, geneset)
      if (is.null(v)) next
      rows[[length(rows) + 1L]] <-
        data.table(dataset = ds, axis = ax, method = disp, r = v)
    }
  }
  if (length(rows) == 0) return(NULL)
  rbindlist(rows)
}

## All-gene gene-wise on the cross-method COMMON gene set (apples-to-apples). `load_vec` drops gene
## names, so here we re-read WITH names, intersect the reported genes across all methods per dataset,
## and keep each method's per-gene PCC on the shared genes — mirrors lib/common_gene_set.py /
## apply_common_genewise.py so this figure matches the "(common set)" all-gene gene-wise in the tables.
## (Each method's OWN all-gene set differs 10x+ in size, e.g. CIBERSORTx HiRes ~1675 vs DeconvSC ~22804,
##  making the per-method all-gene box non-comparable; marker gene-wise is unaffected.)
build_allgene_common_long <- function() {
  rows <- list()
  for (ds in datasets) {
    perm <- list()
    for (i in seq_along(main_methods)) {
      folder <- names(main_methods)[i]; disp <- main_methods[[i]]
      f <- file.path(results_root, ds, folder, "pearson_gene_all_genes.csv")
      if (!file.exists(f)) next
      raw <- tryCatch(fread(f), error = function(e) NULL)
      if (is.null(raw) || nrow(raw) == 0) next
      g <- as.character(raw[[1]]); v <- suppressWarnings(as.numeric(raw[[ncol(raw)]]))
      keep <- is.finite(v) & nzchar(g); g <- g[keep]; v <- v[keep]
      dup <- duplicated(g); g <- g[!dup]; v <- v[!dup]
      perm[[disp]] <- setNames(v, g)
    }
    if (length(perm) == 0) next
    common <- Reduce(intersect, lapply(perm, names))
    if (length(common) < 3) next
    for (disp in names(perm)) {
      vv <- perm[[disp]][common]; vv <- vv[is.finite(vv)]
      if (length(vv) < 3) next
      rows[[length(rows) + 1L]] <- data.table(dataset = ds, axis = "gene", method = disp, r = as.numeric(vv))
    }
  }
  if (length(rows) == 0) return(NULL)
  rbindlist(rows)
}

## Combined methods box: all 4 axes (sample/profile/gene/pseudo-bulk) x methods, ONE figure per
## (dataset, geneset). Rebuilt here from the landed pearson_*.csv with the SAME pastel `palette` as
## the per-cell-type bars (replaces run_eval's Dark2 <ds>_methods_box_*; recolour without re-eval).
axis_label_box <- c(sample = "Sample-wise", profile = "Profile-wise",
                    gene = "Gene-wise", pseudobulk = "Pseudo-bulk")
make_methods_box <- function(geneset, gs_tag) {
  long <- build_marker_long(names(axis_label_box), geneset)
  if (is.null(long) || nrow(long) == 0) { message("Skip methods_box ", gs_tag, ": no data"); return(invisible()) }
  long[, axis := factor(axis_label_box[axis], levels = unname(axis_label_box))]
  for (ds in intersect(datasets, unique(long$dataset))) {
    sub <- long[dataset == ds]
    if (nrow(sub) == 0) next
    ms <- intersect(method_order, unique(as.character(sub$method)))
    sub[, method := factor(as.character(method), levels = ms)]
    y_lo <- min(-0.15, min(sub$r, na.rm = TRUE)); y_hi <- max(1.05, max(sub$r, na.rm = TRUE))
    p <- ggplot(sub, aes(x = axis, y = r, fill = method)) +
      geom_boxplot(width = 0.72, linewidth = 0.4, colour = "black", outlier.size = 0.6,
                   position = position_dodge(width = 0.8)) +
      scale_fill_manual(values = palette[ms], drop = TRUE) +
      guides(fill = guide_legend(nrow = if (length(ms) >= 5) 2 else 1, byrow = TRUE)) +  # wrap 5-method legend into 2 rows so DeconvSC isn't clipped at the right edge
      scale_y_continuous(breaks = seq(-1, 1, 0.2)) +
      coord_cartesian(ylim = c(y_lo, y_hi)) +
      geom_hline(yintercept = 0, colour = "black", linewidth = 0.3) +
      labs(x = NULL, y = "Pearson correlation coefficient", fill = "Method",
           title = sprintf("%s — %s", unname(ds_label[ds]), gs_tag)) +
      theme_minimal(base_family = BASE_FAM) +
      theme(axis.text.x = element_text(face = "bold", size = 9),
            axis.text.y = element_text(size = 9), legend.position = "top",
            legend.title = element_text(size = 9, face = "bold"), legend.text = element_text(size = 8),
            plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black", linewidth = 0.4),
            axis.ticks = element_line(colour = "black", linewidth = 0.3))
    base <- sprintf("%s_methods_box_%s", ds, geneset)
    fwrite(sub, file.path(out_dir, paste0(base, ".csv")))
    for (ext in c("png", "svg", "pdf")) save_plot_file(file.path(out_dir, paste0(base, ".", ext)), p, 150, 110)
    message("Saved ", file.path(out_dir, base))
  }
}

save_plot_file <- function(filename, plot, width, height, units = "mm", dpi = 600) {
  ext <- tolower(tools::file_ext(filename))
  if (ext == "pdf" && capabilities("cairo")) {
    ggsave(filename, plot = plot, device = grDevices::cairo_pdf,
           width = width, height = height, units = units, dpi = dpi)
  } else if (ext == "svg") {
    ggsave(filename, plot = plot, device = svglite::svglite,
           width = width, height = height, units = units, dpi = dpi)
    svg_to_arial(filename)            # relabel Liberation Sans -> literal "Arial"
    svg_to_emf(filename)              # also emit a matching .emf next to the .svg
  } else {
    ggsave(filename, plot = plot,
           width = width, height = height, units = units, dpi = dpi)
  }
}

make_marker_boxes <- function(long_df, which_axis, axis_label, fname, lw_mult = 1, center = "mean") {
  # lw_mult scales box/axis line weights. Gene-wise PCC spans into the negatives (down to ~-0.8),
  # so its boxes are vertically compressed vs the profile panel and the lines LOOK thinner at the same
  # 0.4 width; pass lw_mult>1 for the gene-wise figures to restore a matching visual weight.
  if (is.null(long_df) || nrow(long_df) == 0) {
    message("Skip ", fname, ": no marker data"); return(invisible(NULL))
  }
  sub <- long_df[axis == which_axis]
  if (nrow(sub) == 0) {
    message("Skip ", fname, ": no rows for axis '", which_axis,
            "' (该 axis 的 marker_genes.csv 可能不存在)")
    return(invisible(NULL))
  }
  present_methods <- intersect(method_order, unique(as.character(sub$method)))
  present_ds      <- intersect(datasets,             unique(sub$dataset))
  sub[, method  := factor(method,  levels = present_methods)]

  fwrite(sub, file.path(out_dir, paste0(fname, ".csv")))
  unlink(file.path(out_dir, paste0(fname, ".", c("png", "svg", "pdf"))))

  for (ds in present_ds) {
    sub_ds <- copy(sub[dataset == ds])
    if (nrow(sub_ds) == 0) next
    ds_methods <- intersect(method_order, unique(as.character(sub_ds$method)))
    sub_ds[, method := factor(as.character(method), levels = ds_methods)]

    cfun <- if (center == "median") median else mean             # statistic for the annotated number + white diamond
    med <- sub_ds[, .(r = cfun(r, na.rm = TRUE)), by = method]   # default mean (matches summary_all_methods.csv); center="median" -> annotate the median
    med_ours  <- med[method == "DeconvSC"]
    med_other <- med[method != "DeconvSC"]
    y_lower <- min(-0.05, min(sub_ds$r, na.rm = TRUE))
    y_upper <- max(1.12, max(sub_ds$r, na.rm = TRUE))
    ds_title <- unname(ds_label[ds])
    if (is.na(ds_title) || !nzchar(ds_title)) ds_title <- ds

    p <- ggplot(sub_ds, aes(x = method, y = r, fill = method)) +
      geom_boxplot(width = 0.62, linewidth = 0.4 * lw_mult, colour = "black",
                   outlier.size = 1) +
      stat_summary(fun = cfun, geom = "point", shape = 23, size = 1.8,
                   fill = "white", colour = "black", stroke = 0.4 * lw_mult) +   # white diamond = the annotated centre stat (mean by default; median when center="median")
      geom_text(data = med_other,
                aes(x = method, y = 1.02, label = sprintf("%.2f", r)),
                vjust = 0, size = 9 / .pt, colour = "black",
                fontface = "bold", inherit.aes = FALSE) +
      geom_text(data = med_ours,
                aes(x = method, y = 1.02, label = sprintf("%.2f", r)),
                vjust = 0, size = 9 / .pt, colour = "#B5651D",
                fontface = "bold", inherit.aes = FALSE) +
      scale_fill_manual(values = main_palette[ds_methods], drop = TRUE) +
      scale_y_continuous(breaks = seq(-1, 1, 0.2),
                         expand = expansion(mult = 0)) +
      coord_cartesian(ylim = c(y_lower, y_upper)) +
      labs(x = "Method",
           y = sprintf("%s Pearson correlation coefficient", axis_label),
           title = ds_title) +
      theme_minimal(base_family = BASE_FAM) +
      theme(
        legend.position   = "none",
        axis.text.x       = element_text(angle = 30, hjust = 0.92, vjust = 1,
                                         face = "bold", size = 9),
        axis.text.y       = element_text(face = "bold", size = 9),
        axis.title.x      = element_text(size = 10,
                                         margin = margin(t = 3, unit = "pt")),
        axis.title.y      = element_text(size = 9,
                                         margin = margin(r = 6, unit = "pt")),
        plot.title        = element_text(size = 11, face = "bold", hjust = 0.5,
                                         margin = margin(b = 6, unit = "pt")),
        panel.grid.major  = element_blank(),
        panel.grid.minor  = element_blank(),
        panel.background  = element_blank(),
        axis.line         = element_line(colour = "black", linewidth = 0.4 * lw_mult),
        axis.ticks        = element_line(colour = "black", linewidth = 0.3 * lw_mult),
        axis.ticks.length = unit(2, "pt"),
        plot.margin       = unit(c(0.2, 0.2, 0.2, 0.2), "cm")
      )

    out_base <- paste(fname, ds, sep = "_")
    fwrite(sub_ds, file.path(out_dir, paste0(out_base, ".csv")))
    for (ext in c("png", "svg", "pdf")) {
      out_f <- file.path(out_dir, paste0(out_base, ".", ext))
      save_plot_file(out_f, plot = p, width = 120, height = 120)
      message("Saved ", out_f)
    }
  }
  invisible(NULL)
}


read_metric <- function(path) {
  if (!file.exists(path)) return(NULL)
  df <- tryCatch(fread(path), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) return(NULL)
  if (ncol(df) == 2) {
    setnames(df, c("CellType", "Value"))
  } else if (ncol(df) >= 3) {
    setnames(df, c("Sample", "CellType", "Value"))
  } else {
    return(NULL)
  }
  df <- df[, .(CellType = as.character(CellType),
               Value    = suppressWarnings(as.numeric(Value)))]
  df <- df[is.finite(Value) & nzchar(CellType)]
  # "None" is a real HLCA ground-truth group (4653 GT cells) but an ugly placeholder name; rename it
  # to "Others" wherever it appears (it's present for DeconvSC in profile and for cVAE/BayesPrism/
  # DISSECT/TAPE in pseudobulk; CIBERSORTx has none -> its bar is simply absent, which is fine).
  df[tolower(CellType) %in% c("none", "na", "nan"), CellType := "Others"]
  if (nrow(df) == 0) return(NULL)
  df[, .(Value = mean(Value, na.rm = TRUE)), by = CellType]
}

## Aggregate fine -> major via per-major mean of fine PCC values.
aggregate_to_major <- function(fine_df, mmap) {
  if (is.null(fine_df) || nrow(fine_df) == 0) return(NULL)
  lookup <- data.table(
    CellType  = unlist(mmap, use.names = FALSE),
    CellGroup = rep(names(mmap), lengths(mmap))
  )
  lookup_lower <- copy(lookup)[, CellType := tolower(CellType)]
  fine_df <- copy(fine_df)
  fine_df[, key := tolower(CellType)]
  out <- merge(fine_df, lookup_lower, by.x = "key", by.y = "CellType",
               all.x = FALSE)
  if (nrow(out) == 0) return(NULL)
  out[, .(Value = mean(Value, na.rm = TRUE)), by = CellGroup
      ][, .(CellType = CellGroup, Value)]
}

## Load CIBERSORTx values at the major level for a given dataset / metric.
load_cibersortx_major <- function(dataset, metric) {
  ## first try the unified pipeline output
  f <- file.path(results_root, dataset, "CIBERSORTx",
                 sprintf("pearson_%s_all_genes.csv", metric))
  df <- read_metric(f)
  mmap <- major_maps[[dataset]]
  aliases <- cibersortx_major_aliases[[dataset]]                 # CIBERSORTx native major name -> display name
  if (!is.null(df) && nrow(df) > 0 && !is.null(aliases))
    df[CellType %in% names(aliases), CellType := unname(aliases[CellType])]
  if (!is.null(df) && nrow(df) > 0 && any(is.finite(df$Value))) {
    if (!is.null(mmap) && length(mmap) > 0) {
      major_names <- names(mmap)
      major_lookup <- setNames(major_names, tolower(major_names))
      is_major <- tolower(df$CellType) %in% names(major_lookup)
      if (all(is_major)) {
        df[, CellType := unname(major_lookup[tolower(CellType)])]
        return(df)
      }
      df_major <- aggregate_to_major(df, mmap)
      if (!is.null(df_major) && nrow(df_major) > 0) return(df_major)
    }
    return(df)
  }
  ## fallback (HCA only, HiRes was constant)
  fb <- cibersortx_fallback[[dataset]][[metric]]
  if (is.null(fb) || !file.exists(fb)) return(NULL)
  raw <- tryCatch(fread(fb), error = function(e) NULL)
  if (is.null(raw)) return(NULL)
  setnames(raw, c("CellType", "Value"))
  raw[, CellType := as.character(CellType)]
  raw[, Value   := suppressWarnings(as.numeric(Value))]
  raw <- raw[is.finite(Value) & CellType != "Others"]
  if (!is.null(mmap) && length(mmap) > 0) {
    raw_major <- aggregate_to_major(raw, mmap)
    if (!is.null(raw_major) && nrow(raw_major) > 0) return(raw_major)
  }
  raw
}

## --- build per-method per-method data --------------------------------------
build_fine_df <- function(dataset, metric) {
  per <- lapply(setdiff(methods, "CIBERSORTx"), function(m) {
    f  <- file.path(results_root, dataset, m,
                    sprintf("pearson_%s_all_genes.csv", metric))
    df <- read_metric(f)
    if (is.null(df)) return(NULL)
    df[, Method := ifelse(m == "cVAE", "DeconvSC", m)]
    df
  })
  per <- Filter(Negate(is.null), per)
  if (length(per) == 0) return(NULL)
  rbindlist(per)
}

build_major_df <- function(dataset, metric) {
  mmap <- major_maps[[dataset]]
  if (is.null(mmap) || length(mmap) == 0) {
    message("Skip ", dataset, " / ", metric, " / major: no major_map configured")
    return(NULL)
  }
  per <- list()
  for (m in setdiff(methods, "CIBERSORTx")) {
    f  <- file.path(results_root, dataset, m,
                    sprintf("pearson_%s_all_genes.csv", metric))
    df <- read_metric(f)
    if (is.null(df)) next
    df_major <- aggregate_to_major(df, mmap)
    if (is.null(df_major)) next
    df_major[, Method := ifelse(m == "cVAE", "DeconvSC", m)]
    per[[m]] <- df_major
  }
  cibx <- load_cibersortx_major(dataset, metric)
  if (!is.null(cibx)) {
    cibx[, Method := "CIBERSORTx"]
    per[["CIBERSORTx"]] <- cibx[, .(CellType, Value, Method)]
  }
  per <- Filter(Negate(is.null), per)
  if (length(per) == 0) return(NULL)
  out <- rbindlist(per, use.names = TRUE)
  out <- out[CellType != "Others"]
  out
}

## --- plotting ---------------------------------------------------------------
rename_cell_types_for_plot <- function(plot_df, dataset) {
  ct_names <- cell_type_label_maps[[dataset]]
  if (is.null(ct_names) || nrow(ct_names) == 0) return(plot_df)
  lookup <- setNames(ct_names$new, ct_names$old)
  plot_df <- copy(plot_df)
  plot_df[, CellType := as.character(CellType)]
  plot_df[CellType %in% names(lookup), CellType := lookup[CellType]]
  plot_df
}

make_plot <- function(plot_df, dataset, metric, level) {
  if (is.null(plot_df) || nrow(plot_df) == 0) {
    message("Skip ", dataset, " / ", metric, " / ", level, ": empty")
    return(invisible(NULL))
  }
  plot_df <- rename_cell_types_for_plot(plot_df, dataset)
  ## Drop cell types that DeconvSC could not reconstruct (no generated cells under the realistic
  ## prophead composition -> no DeconvSC profile). Otherwise such a type renders as a lone competitor
  ## bar (the Supplementary-Fig-2 "Neutrophils = single DISSECT bar" artefact). Applied only when a
  ## DeconvSC series exists; major-level panels are unaffected (every major group keeps a DeconvSC bar).
  if ("DeconvSC" %in% unique(as.character(plot_df$Method))) {
    keep_ct <- unique(as.character(plot_df[Method == "DeconvSC", CellType]))
    dropped <- setdiff(unique(as.character(plot_df$CellType)), keep_ct)
    if (length(dropped) > 0) {
      message("  [drop] ", dataset, " ", metric, "/", level, ": ", length(dropped),
              " cell type(s) absent in DeconvSC removed: ", paste(sort(dropped), collapse = ", "))
      plot_df <- plot_df[CellType %in% keep_ct]
    }
  }
  if (nrow(plot_df) == 0) { message("Skip ", dataset, "/", metric, "/", level, ": no DeconvSC cell types"); return(invisible(NULL)) }
  metric_label <- switch(metric,
                         profile    = "Profile-wise",
                         pseudobulk = "Pseudo-bulk",
                         sample     = "Sample-wise",
                         metric)
  level_label  <- if (level  == "fine")    "fine cell type" else "major cell type"

  ## order cell types by overall mean (descending)
  ct_order <- plot_df[, .(m = mean(Value, na.rm = TRUE)), by = CellType
                      ][order(-m), CellType]
  plot_df[, CellType := factor(CellType, levels = ct_order)]
  present_methods <- intersect(method_order, unique(as.character(plot_df$Method)))
  plot_df[, Method := factor(Method, levels = present_methods)]

  y_min <- min(0, min(plot_df$Value, na.rm = TRUE))
  y_max <- max(1, max(plot_df$Value, na.rm = TRUE))
  ## Profile panels: fixed 0.0,0.2,...,1.0 y ticks (one decimal), axis starting at 0 — consistent with
  ## the gene-wise main boxplots. Pseudo-bulk panels keep auto breaks.
  if (metric == "profile") { y_min <- 0; y_breaks <- seq(0, 1, 0.2); y_labs <- scales::label_number(accuracy = 0.1); y_tick_face <- "bold" }
  else                     { y_breaks <- waiver();   y_labs <- waiver(); y_tick_face <- "plain" }

  ## Adaptive LEFT margin. The leftmost 30°-rotated x label (the highest-PCC cell type) is anchored
  ## flush against the y-axis (scale_x_discrete expand keeps the first bar tight) and overhangs to the
  ## lower-left; ggsave's fixed canvas would clip its head (e.g. "Alveolar macrophages" -> "lar
  ## macrophages"). Size the left margin to that first label's measured width so it always fits.
  first_w_pt <- tryCatch(
    systemfonts::shape_string(as.character(ct_order[1]), family = BASE_FAM,
                              size = 9, weight = "bold")$metrics$width,
    error = function(e) nchar(as.character(ct_order[1])) * 5)   # ~5pt/char fallback
  overhang_mm <- as.numeric(first_w_pt) / 72 * 25.4 * cos(pi / 6) * 0.92  # left of tick: 30°, hjust 0.92
  left_cm     <- max(0.2, (overhang_mm - 11) / 10 + 0.25)        # y-axis furniture already covers ~11mm

  p <- ggplot(plot_df, aes(x = CellType, y = Value, fill = Method)) +
    geom_bar(stat = "identity",
             width = 0.8,
             position = position_dodge(width = 0.85),
             color = "white",
             linewidth = 0.1) +
    scale_fill_manual(values = palette[present_methods], drop = TRUE) +
    guides(fill = guide_legend(nrow = if (length(present_methods) >= 5) 2 else 1, byrow = TRUE)) +  # wrap 5-method legend into 2 rows so DeconvSC isn't clipped at the right edge
    scale_x_discrete(expand = expansion(add = 0.5)) +   # X轴：第一个柱体贴紧Y轴
    scale_y_continuous(expand = expansion(mult = c(0, 0.02)),
                       limits = c(y_min, y_max),
                       breaks = y_breaks, labels = y_labs,
                       ) +           # Y轴：柱体底部贴紧X轴
    # coord_cartesian(ylim = c(y_min, y_max)) +
    theme_set(theme_minimal(base_family = BASE_FAM)) +
    theme(
      axis.text.x  = element_text(angle = 30, face = "bold", hjust = 0.92,
                                  vjust = 1, size = 9),
      axis.text.y  = element_text(face = y_tick_face, size = 9),   # bold for profile -> matches main_genewise y ticks
      axis.title.x = element_text(size = 10, margin = margin(t = 3, unit = "pt")),
      axis.title.y = element_text(size = 9, margin = margin(r = 6, unit = "pt")),
      plot.title   = element_text(size = 11, face = "bold", hjust = 0.5,
                                  margin = margin(b = 6, unit = "pt")),
      legend.title       = element_text(size = 9, face = "bold"),
      legend.text        = element_text(size = 9),
      legend.position    = "top",
      legend.direction   = "horizontal",
      legend.key.size    = unit(0.3, "cm"),
      legend.spacing.y   = unit(0.05, "cm"),
      plot.margin        = unit(c(0.2, 0.2, 0.2, left_cm), "cm"),
      axis.line = element_line(color = "black", linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      panel.grid.major   = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.background   = element_blank()
    ) +
    # labs(x = sprintf("Cell type (%s)", level_label),
    labs(x = "Cell type",
         # y = "Pearson correlation coefficient",
         y = sprintf("%s Pearson correlation coefficient",
                     metric_label))
         # title = sprintf("%s PCC - %s (%s)",
         #                 metric_label, dataset, level_label))

  n_ct <- length(ct_order)
  # Width SCALES with the number of cell types so the 30°-rotated x-axis labels never overlap.
  # Fixed 180mm crammed GSE159585's ~62 fine types (~2.9mm each); ~7.5mm/type keeps them clear.
  # HCA (~21) / GSE141115 (~19) stay at the 180mm floor; major (5-9 groups) keeps 120mm.
  if (level == "fine") {
    width_mm <- max(180, round(45 + n_ct * 7.5))
    height_mm <- 120
  } else {
    width_mm <- max(120, round(35 + n_ct * 10))
    height_mm <- 120
  }

  base <- sprintf("%s_%s_%s_pcc", dataset, metric, level)
  out_csv <- file.path(out_dir, paste0(base, ".csv"))
  fwrite(plot_df, out_csv)
  for (ext in c("png", "svg", "pdf")) {
    out_f <- file.path(out_dir, paste0(base, ".", ext))
    save_plot_file(out_f, plot = p, width = width_mm, height = height_mm)
    message("Saved ", out_f)
  }
  invisible(p)
}

## --- main loop --------------------------------------------------------------

marker_long <- build_marker_long(c("profile", "gene", "sample"))
make_marker_boxes(marker_long, "profile", "Profile-wise", "main_profile_pcc")
make_marker_boxes(marker_long, "gene",   "Gene-wise",   "main_genewise_pcc", lw_mult = 1.4, center = "median")
make_marker_boxes(marker_long, "sample", "Sample-wise", "main_samplewise_pcc")

## gene-wise box, ALL genes, per dataset (complements the marker-gene main_genewise_pcc_<ds>)
allgene_long <- build_allgene_common_long()   # COMMON gene set (matches tables), not each method's own set
make_marker_boxes(allgene_long, "gene", "Gene-wise (all genes, common set)", "main_genewise_allgenes_pcc")

## combined methods box (4 axes x methods), pastel palette, per (dataset, geneset) —
## recoloured replacement for run_eval's Dark2 <ds>_methods_box_*
make_methods_box("marker_genes", "marker genes")
make_methods_box("all_genes", "all genes")

stale_sample_bases <- as.vector(outer(datasets,
                                      c("sample_fine_pcc", "sample_major_pcc"),
                                      paste, sep = "_"))
stale_sample_files <- file.path(
  out_dir,
  paste0(rep(stale_sample_bases, each = 4),
         ".",
         rep(c("csv", "png", "svg", "pdf"), times = length(stale_sample_bases)))
)
unlink(stale_sample_files)

for (ds in datasets) {
  for (mt in c("profile", "pseudobulk")) {
    fine_df  <- build_fine_df(ds, mt)
    make_plot(fine_df, ds, mt, "fine")
    major_df <- build_major_df(ds, mt)
    make_plot(major_df, ds, mt, "major")
  }
}
