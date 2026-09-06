#!/usr/bin/env Rscript
# CellChat edge-level impact of prophead-no-floor vs softmax(full) vs nnls(floor=20).
# Reads the COVID CellChat object per scheme ONE AT A TIME (objects are ~0.7-1.1 GB).
suppressMessages(library(CellChat))
files <- list(
  "softmax(605,nofloor)"  = "/disk1/maijl/deconv/deconv_20260605/downstream/cellchat/covid_cellchat.rds",
  "prophead(610,NOfloor)" = "/disk1/maijl/deconv/deconv_20260610/prophead/downstream/cellchat/covid_cellchat.rds",
  "nnls(610,floor=20)"    = "/disk1/maijl/deconv/deconv_20260610/nnls/downstream/cellchat/covid_cellchat.rds"
)
involve <- function(cnt, ct) if (ct %in% rownames(cnt)) sum(cnt[ct, ]) + sum(cnt[, ct]) else -1
for (nm in names(files)) {
  f <- files[[nm]]
  if (!file.exists(f)) { cat(sprintf("%-22s [missing]\n", nm)); next }
  cc <- readRDS(f)
  cnt <- cc@net$count                      # directed sender x receiver L-R interaction counts
  nodes <- nrow(cnt); edges <- sum(cnt > 0); tot <- sum(cnt)
  npath <- length(cc@netP$pathways)
  cat(sprintf("%-22s COVID  nodes=%d  edges(>0)=%d  total_interactions=%.0f  pathways=%d  | migDC_int=%d  pDC_int=%d\n",
              nm, nodes, edges, tot, npath, involve(cnt, "migDC"), involve(cnt, "pDC")))
  rm(cc, cnt); invisible(gc())
}
cat("[done] (migDC_int / pDC_int = total directed interactions involving that node; -1 = node ABSENT)\n")
