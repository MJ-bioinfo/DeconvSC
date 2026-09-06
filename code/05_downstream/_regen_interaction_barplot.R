#!/usr/bin/env Rscript
# Standalone re-render of ONLY the interaction-difference barplot from the cached comparison_df.csv,
# applying the right-side axis expansion + plot margin fix (so the 400 tick label is not clipped).
# Helpers (FF / theme_fig5 / gg2 / svg_to_arial / outline_pdf / svg_to_emf) are copied verbatim from
# cellchat_linux.R so the output is byte-for-byte consistent with the full pipeline's exporter.
suppressMessages({ library(ggplot2); library(svglite); library(systemfonts) })
options(stringsAsFactors = FALSE)

FF <- "Arial"
try(register_font(name = FF, plain = match_fonts(FF)$path, bold = match_fonts(FF, weight = "bold")$path,
  italic = match_fonts(FF, italic = TRUE)$path, bolditalic = match_fonts(FF, weight = "bold", italic = TRUE)$path), silent = TRUE)
pdfFonts(Arial = pdfFonts()$Helvetica); postscriptFonts(Arial = postscriptFonts()$Helvetica)
theme_fig5 <- ggplot2::theme_classic(base_size = 7, base_family = FF) +
  ggplot2::theme(axis.text = element_text(colour = "black"),
                 axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3))
svg_to_arial <- function(p) { x <- readLines(p, warn = FALSE)
  x <- gsub("Liberation Sans", "Arial", x, fixed = TRUE); x <- gsub("DejaVu Sans", "Arial", x, fixed = TRUE)
  x <- gsub(" textLength='[^']*'", "", x); x <- gsub(" lengthAdjust='[^']*'", "", x)
  writeLines(x, p) }
outline_pdf <- function(f) {
  if (nchar(Sys.which("gs")) == 0) return(invisible())
  tmp <- paste0(f, ".ol")
  st <- tryCatch(system2("gs", c("-q", "-o", tmp, "-dNoOutputFonts", "-sDEVICE=pdfwrite", f),
                         stdout = FALSE, stderr = FALSE), error = function(e) 1L)
  if (identical(st, 0L) && file.exists(tmp)) file.rename(tmp, f)
}
svg_to_emf <- function(svg) {
  if (nchar(Sys.which("inkscape")) == 0) return(invisible())
  try(system2("inkscape", c(svg, "--export-type=emf", paste0("--export-filename=", sub("\\.svg$", ".emf", svg))),
              stdout = FALSE, stderr = FALSE), silent = TRUE)
}
gg2 <- function(p, fname, w, h) {
  ggsave(file.path(OUT, paste0(fname, ".png")), p, width = w, height = h, units = "mm", dpi = 600)
  tryCatch({ ggsave(file.path(OUT, paste0(fname, ".svg")), p, width = w, height = h, units = "mm",
                    device = svglite::svglite); svg_to_arial(file.path(OUT, paste0(fname, ".svg"))) },
           error = function(e) cat("  [svg skip]", fname, ":", conditionMessage(e), "\n"))
  tryCatch({ ggsave(file.path(OUT, paste0(fname, ".pdf")), p, width = w, height = h, units = "mm", device = grDevices::cairo_pdf)
             outline_pdf(file.path(OUT, paste0(fname, ".pdf"))) },
           error = function(e) cat("  [pdf skip]", fname, ":", conditionMessage(e), "\n"))
  svg_to_emf(file.path(OUT, paste0(fname, ".svg")))
}

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])
RELEASE_ROOT <- normalizePath(file.path(dirname(script_file), "..", ".."), mustWork = TRUE)
DOWN <- Sys.getenv("DOWN_ROOT", file.path(RELEASE_ROOT, "results", "downstream", "GSE159585"))
OUT  <- file.path(DOWN, "cellchat")
cmp <- read.csv(file.path(OUT, "comparison_df.csv"))

p_bar <- ggplot(cmp, aes(reorder(CellType, Difference), Difference, fill = Difference >= 0)) +
  geom_col(width = 0.72) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey45") +
  scale_fill_manual(values = c(`FALSE` = "#2F5D8A", `TRUE` = "#E1895A"), guide = "none") +
  coord_flip() + labs(x = NULL, y = "Interaction Count Difference") +
  scale_y_continuous(expand = expansion(mult = c(0.06, 0.10))) +
  theme_fig5 + theme(axis.text.y = element_text(size = 6), axis.text.x = element_text(size = 6.5),
                     axis.title.x = element_text(size = 7.5),
                     plot.margin = margin(t = 2, r = 6, b = 2, l = 2, unit = "pt"))
gg2(p_bar, "interaction_difference_barplot", 78, 168)
cat("[fig] interaction_difference_barplot regenerated in", OUT, "\n")
