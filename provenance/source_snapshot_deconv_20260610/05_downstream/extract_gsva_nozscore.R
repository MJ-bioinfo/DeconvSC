## Extract the Figure-5b sample matrix (celltype@@pathway x 14 samples) WITHOUT z-scoring.
## Writes raw GSVA + centroid-removed (row-mean subtracted) matrices for Python plotting.
suppressMessages({library(dplyr); library(tibble); library(tidyr)})
SS <- "/disk1/maijl/deconv/deconv_20260605/downstream/ssGSEA"
OUT <- "/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid"
G <- readRDS(file.path(SS, "combined_gsva_scores.rds"))          # celltype@@pathway x sample
lim <- readRDS(file.path(SS, "limma_result.rds")) %>% rownames_to_column("CP")
cat("gsva:", nrow(G), "celltype-pathway x", ncol(G), "samples\n")
## select top differential rows (same spirit as Fig5b: significant, strongest), cap ~45 for readability
sig <- lim %>% filter(adj.P.Val < 0.05, abs(logFC) > 0.2) %>% arrange(desc(abs(logFC)))
sel <- head(sig$CP[sig$CP %in% rownames(G)], 45)
M <- G[sel, , drop = FALSE]
## order columns: Normal first then COVID, parsed from "<status>_<sample>"
cond <- sub("_.*$", "", colnames(M)); samp <- sub("^[^_]*_", "", colnames(M))
ord <- order(cond != "Normal", samp)
M <- M[, ord]; cond <- cond[ord]; samp <- samp[ord]
## centroid-removed = subtract each row's mean across all 14 samples (keep real units, no scaling)
Mc <- M - rowMeans(M, na.rm = TRUE)
write.csv(M,  file.path(OUT, "gsva_sample_raw_nozscore.csv"))
write.csv(Mc, file.path(OUT, "gsva_sample_centroidremoved_nozscore.csv"))
write.csv(data.frame(sample = samp, condition = cond), file.path(OUT, "gsva_sample_meta.csv"), row.names = FALSE)
## also attach logFC for row ordering
write.csv(lim %>% filter(CP %in% sel) %>% select(CP, logFC, adj.P.Val), file.path(OUT, "gsva_sample_rows_logfc.csv"), row.names = FALSE)
cat("[saved] gsva_sample_{raw,centroidremoved}_nozscore.csv +", length(sel), "rows\n")
