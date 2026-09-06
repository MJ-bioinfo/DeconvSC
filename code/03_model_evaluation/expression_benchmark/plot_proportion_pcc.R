## Re-plot the HCA_fold2_proportion_* benchmark figures in the SAME style as the profile-series
## per-cell-type bars (04_plot_pcc_per_dataset.R::make_plot): theme_minimal + Arial, 30-deg bold
## x labels, y in [0,1] (0.2 breaks), pastel per-method palette, no grid, axis lines, legend on top,
## white-diamond NOT used (these are point estimates, not distributions). PNG/PDF/SVG + EMF.
## Reads $WORK_ROOT/03_result_tables/PROPORTION_*.csv -> $WORK_ROOT/03_result_figures/figures/.
suppressMessages({ library(data.table); library(ggplot2); library(systemfonts); library(svglite) })

FF <- "Arial"
try(register_font(name = FF, plain = match_fonts(FF)$path, bold = match_fonts(FF, weight = "bold")$path,
              italic = match_fonts(FF, italic = TRUE)$path,
              bolditalic = match_fonts(FF, weight = "bold", italic = TRUE)$path), silent = TRUE)  # skip if a real "Arial" is already installed
try(pdfFonts(Arial = pdfFonts()$Helvetica), silent = TRUE)
BASE_FAM <- FF
svg_to_arial <- function(path) {
  if (!file.exists(path)) return(invisible())
  x <- readLines(path, warn = FALSE)
  x <- gsub("Liberation Sans", "Arial", x, fixed = TRUE); x <- gsub("DejaVu Sans", "Arial", x, fixed = TRUE)
  x <- gsub(" textLength='[^']*'", "", x); x <- gsub(" lengthAdjust='[^']*'", "", x)
  writeLines(x, path)
}
svg_to_emf <- function(svg) {
  if (!file.exists(svg) || nchar(Sys.which("inkscape")) == 0) return(invisible())
  try(system2("inkscape", c(svg, "--export-type=emf", paste0("--export-filename=", sub("\\.svg$", ".emf", svg))),
              stdout = FALSE, stderr = FALSE), silent = TRUE)
}
save_plot_file <- function(filename, plot, width, height, units = "mm", dpi = 600) {
  ext <- tolower(tools::file_ext(filename))
  if (ext == "pdf" && capabilities("cairo")) {
    ggsave(filename, plot = plot, device = grDevices::cairo_pdf, width = width, height = height, units = units, dpi = dpi)
  } else if (ext == "svg") {
    ggsave(filename, plot = plot, device = svglite::svglite, width = width, height = height, units = units, dpi = dpi)
    svg_to_arial(filename); svg_to_emf(filename)
  } else {
    ggsave(filename, plot = plot, width = width, height = height, units = units, dpi = dpi)
  }
}

palette <- c("BayesPrism" = "#86ADC0", "CIBERSORTx" = "#7187A7", "DISSECT" = "#AA96CA",
             "TAPE" = "#989EB0", "DeconvSC" = "#DEBBB1")
method_order <- c("BayesPrism", "CIBERSORTx", "DISSECT", "TAPE", "DeconvSC")

.args <- commandArgs(FALSE); .sp <- sub("^--file=", "", .args[grep("^--file=", .args)])
script_dir <- if (length(.sp)) normalizePath(dirname(.sp)) else normalizePath(".")
release_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
TB  <- Sys.getenv("BENCHMARK_TABLES", file.path(release_root, "benchmark_tables", "expression_prediction", "03_result_tables"))
OUT <- Sys.getenv("FIGURE_OUT", file.path(release_root, "work", "expression_benchmark_figures")); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

plot_prop <- function(csv, fig_base) {
  path <- file.path(TB, csv)
  if (!file.exists(path)) { message("skip (missing): ", path); return(invisible()) }
  d <- fread(path)
  setnames(d, 1, "CellClass")
  d <- d[!grepl("MEAN", CellClass)]                       # drop the "— MEAN —" summary row
  if ("mean_true_prop" %in% names(d)) d[, mean_true_prop := NULL]
  long <- melt(d, id.vars = "CellClass", variable.name = "Method", value.name = "PCC")
  long <- long[is.finite(PCC)]                            # NaN (e.g. CIBERSORTx 'Others') -> bar absent (gap)
  ms <- intersect(method_order, unique(as.character(long$Method)))
  long[, Method := factor(as.character(Method), levels = ms)]
  ord <- long[, .(m = mean(PCC, na.rm = TRUE)), by = CellClass][order(-m), CellClass]   # order by mean PCC desc
  long[, CellClass := factor(CellClass, levels = ord)]
  y_min <- min(0, min(long$PCC, na.rm = TRUE)); y_max <- max(1, max(long$PCC, na.rm = TRUE))
  n_ct <- length(ord)

  p <- ggplot(long, aes(x = CellClass, y = PCC, fill = Method)) +
    geom_bar(stat = "identity", width = 0.8, position = position_dodge(width = 0.85),
             color = "white", linewidth = 0.1) +
    scale_fill_manual(values = palette[ms], drop = TRUE) +
    scale_x_discrete(expand = expansion(add = 0.5)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.02)), limits = c(y_min, y_max),
                       breaks = seq(-1, 1, 0.2)) +
    theme_minimal(base_family = BASE_FAM) +
    theme(
      axis.text.x  = element_text(angle = 30, face = "bold", hjust = 0.92, vjust = 1, size = 9),
      axis.text.y  = element_text(size = 9),
      axis.title.x = element_text(size = 10, margin = margin(t = 3, unit = "pt")),
      axis.title.y = element_text(size = 9, margin = margin(r = 6, unit = "pt")),
      plot.title   = element_text(size = 11, face = "bold", hjust = 0.5, margin = margin(b = 6, unit = "pt")),
      legend.title     = element_text(size = 9, face = "bold"),
      legend.text      = element_text(size = 9),
      legend.position  = "top",
      legend.direction = "horizontal",
      legend.key.size  = unit(0.3, "cm"),
      legend.spacing.y = unit(0.05, "cm"),
      plot.margin      = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),
      axis.line  = element_line(color = "black", linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
      panel.background = element_blank()
    ) +
    labs(x = "Cell class", y = "Proportion Pearson correlation coefficient", fill = "Method")

  width_mm <- max(170, round(45 + n_ct * 13)); height_mm <- 120     # >=170mm so the 5-method top legend fits (cf. profile-fine 180mm)
  fwrite(long, file.path(OUT, paste0(fig_base, ".csv")))
  for (ext in c("png", "svg", "pdf")) save_plot_file(file.path(OUT, paste0(fig_base, ".", ext)), p, width_mm, height_mm)
  message("Saved ", file.path(OUT, fig_base), "  (", n_ct, " classes, ", length(ms), " methods)")
}

plot_prop("PROPORTION_lineage5_HCA_prophead_PCC.csv",      "HCA_fold2_proportion_lineage5_prophead_pcc")
plot_prop("PROPORTION_group7_HCA_prophead_PCC.csv",        "HCA_fold2_proportion_group7_prophead_pcc")
plot_prop("PROPORTION_merged14_by_celltype_PCC_HCA_fold2.csv", "HCA_fold2_proportion_S4_pcc")
