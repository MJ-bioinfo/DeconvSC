#!/usr/bin/env Rscript

cran_packages <- c(
  "circlize", "reticulate", "dplyr", "tidyr", "stringr", "ggplot2",
  "patchwork", "svglite", "systemfonts", "msigdbr", "tibble", "purrr",
  "RColorBrewer", "future", "remotes"
)

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos = "https://cloud.r-project.org")

missing_cran <- cran_packages[!vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_cran))
  install.packages(missing_cran, repos = "https://cloud.r-project.org")

bioc_packages <- c("GSVA", "limma", "monocle", "ComplexHeatmap", "BiocParallel", "Biobase")
missing_bioc <- bioc_packages[!vapply(bioc_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc))
  BiocManager::install(missing_bioc, version = "3.20", ask = FALSE, update = FALSE)

if (!requireNamespace("CellChat", quietly = TRUE))
  remotes::install_github("jinworks/CellChat@75253cd0c9e68410e6e721a6d3a0419a1d7e358f")

message("R dependencies installed. Compare versions with R_packages.txt and session_info.txt.")
