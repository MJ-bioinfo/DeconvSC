#!/usr/bin/env Rscript
# Ablation pathway-recovery (GSE159585 COVID).
# Question: how faithfully does each model's DeconvSC-generated single-cell data recover the
# cell-cell-communication signalling pathways that CellChat detects in REAL COVID scRNA-seq?
#   Ground truth = pathways inferred from real COVID data (real_cellchat).
#   Ours = full DeconvSC model (covid_cellchat);  WO = attention-ablated model (ablation_cellchat).
# Metrics (set overlap vs ground truth): Recall, Precision, F1, Jaccard.
suppressMessages(library(CellChat))
P_real <- "/disk1/maijl/deconv/data/GSE159585/real_cellchat.rds"
P_ours <- "/disk1/maijl/deconv/deconv_20260605/downstream/cellchat/covid_cellchat.rds"
P_wo   <- "/disk1/maijl/deconv/cVAE/GSE159585/ablation_basicVAE/ablation_cellchat.rds"
OUT    <- "/disk1/maijl/deconv/deconv_20260605/downstream/cellchat"

get_paths <- function(p) {
  o <- readRDS(p); pw <- unique(o@netP$pathways)
  cond <- if ("condition" %in% colnames(o@meta)) paste(unique(as.character(o@meta$condition)), collapse = "/") else "NA"
  cat(sprintf("  %-22s cells=%6d celltypes=%2d pathways=%3d cond=%s\n",
              basename(p), ncol(o@data), nlevels(o@idents), length(pw), cond))
  pw
}
cat("Loading objects + extracting CellChat pathway sets (netP$pathways):\n")
real <- get_paths(P_real); ours <- get_paths(P_ours); wo <- get_paths(P_wo)

metrics <- function(pred, truth) {
  tp <- length(intersect(pred, truth))
  recall    <- tp / length(truth)
  precision <- if (length(pred)) tp / length(pred) else 0
  jaccard   <- tp / length(union(pred, truth))
  f1        <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  c(n_pathways = length(pred), TP = tp, Recall = recall, Precision = precision, F1 = f1, Jaccard = jaccard)
}
res <- round(rbind("Ours (full)" = metrics(ours, real), "WO (ablation)" = metrics(wo, real)), 4)
cat(sprintf("\nGround-truth (real COVID) pathways: %d\n\n", length(real)))
print(res)
df <- data.frame(model = rownames(res), res, check.names = FALSE, row.names = NULL)
write.csv(df, file.path(OUT, "ablation_pathway_recovery.csv"), row.names = FALSE)
cat("\n[saved]", file.path(OUT, "ablation_pathway_recovery.csv"), "\n")
cat("\nReal pathways missed by Ours:", paste(setdiff(real, ours), collapse = ", "), "\n")
cat("Real pathways missed by WO  :", paste(setdiff(real, wo),   collapse = ", "), "\n")
