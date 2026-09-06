# External resources and licensing notes

- GSE141115, GSE159585 and GSE226077 are public accessioned datasets. Processed
  inputs used by this release are bundled under `processed_data/`; consult the
  source repositories for the original raw files and their terms.
- MSigDB 2025.1.Hs was used for the frozen ssGSEA/GSVA analyses. The licensed
  MSigDB snapshot is not redistributed. Set `MSIGDB_SNAPSHOT` to an authorized
  local RDS snapshot to recompute the analyses.
- CIBERSORTx is an external service/software product and is not redistributed.
  The release contains the available method output tables required to reproduce
  the manuscript comparison.
- BayesPrism, TAPE and DISSECT retain their own software licenses. This package
  contains derived benchmark intermediates/results, not vendored installations.
