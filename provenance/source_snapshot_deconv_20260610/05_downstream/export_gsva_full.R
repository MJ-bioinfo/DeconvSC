G <- readRDS("/disk1/maijl/deconv/deconv_20260605/downstream/ssGSEA/combined_gsva_scores.rds")
write.csv(G, "/disk1/maijl/deconv/deconv_20260605/downstream/beyond_centroid/combined_gsva_full.csv")
cat("exported", nrow(G), "x", ncol(G), "\n")
