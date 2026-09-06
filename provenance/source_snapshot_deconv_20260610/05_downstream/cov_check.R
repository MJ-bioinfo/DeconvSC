suppressMessages({library(dplyr)})
msig <- readRDS("/disk1/maijl/deconv/script/msigdb_homosapiens.rds")
extra <- c("REACTOME_INTERFERON_SIGNALING","GOBP_RESPONSE_TO_TYPE_I_INTERFERON",
"REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM","KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION",
"REACTOME_NEUTROPHIL_DEGRANULATION","GOBP_REGULATION_OF_ENDOTHELIAL_CELL_MIGRATION",
"REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION","GOBP_COLLAGEN_FIBRIL_ORGANIZATION",
"REACTOME_ANTIGEN_PROCESSING_CROSS_PRESENTATION","GOBP_REGULATION_OF_LEUKOCYTE_ACTIVATION",
"KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY","KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY")
gs <- msig %>% filter(grepl("^HALLMARK_", gs_name) | gs_name %in% extra) %>%
  select(gs_name, gene_symbol) %>% distinct() %>% split(.$gs_name) %>% lapply(function(x) unique(x$gene_symbol))
cat("n sets:", length(gs), "| union genes:", length(unique(unlist(gs))), "\n\n")
nm <- c("SFTPC","SFTPB","ZBTB16","SRRM2","ARGLU1","TXNIP","FKBP5","NEDD9","CCNL1","MT-ATP6")
cv <- c("COL3A1","COL1A2","NEAT1","TSHZ2","COL6A3","TIMP1","HSPA1A","BNC2","IGKC","TCF4",
        "CCDC102B","COL1A1","BICC1","PVT1","COL5A2","RUNX1","NRP1","HSPA1B","TPST1","FAM118A")
ncount <- function(g) sum(sapply(gs, function(s) g %in% s))
which_sets <- function(g){ n<-names(gs)[sapply(gs, function(s) g %in% s)]; if(length(n)==0) "-" else paste(substr(n,1,28),collapse=", ")}
cat("=== Normal-high genes (real COVID<Normal): # sets ===\n")
for(g in nm) cat(sprintf("  %-9s %d sets | %s\n", g, ncount(g), which_sets(g)))
cat("\n=== COVID-high genes (real COVID>Normal): # sets ===\n")
for(g in cv) cat(sprintf("  %-9s %d sets | %s\n", g, ncount(g), which_sets(g)))
cat(sprintf("\nCOVERAGE: Normal-high %d/10 covered; COVID-high %d/20 covered\n",
            sum(sapply(nm,ncount)>0), sum(sapply(cv,ncount)>0)))
# dilution illustration: HALLMARK_EMT size
emt <- gs[["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"]]
cat(sprintf("\nDILUTION e.g. HALLMARK_EMT has %d genes; of our 30 top-DE genes only %d are members (%s)\n",
            length(emt), sum(c(nm,cv) %in% emt), paste(c(nm,cv)[c(nm,cv) %in% emt], collapse=",")))
