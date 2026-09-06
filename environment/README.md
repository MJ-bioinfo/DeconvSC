# Environment setup

Create the Python environment from the portable Conda specification:

```bash
conda env create -f environment/conda_environment.yml
conda activate deconvsc
```

`requirements.txt` records the corresponding pip-level versions without any
machine-local `file://` URLs. It is provided for auditing or pip-based setup.
GPU users may need to install the PyTorch wheel appropriate for their CUDA driver.

Downstream R analyses were run in R 4.4.3/Bioconductor 3.20. Install the R
dependencies with:

```bash
Rscript environment/install_R_packages.R
```

Compare the resulting installation with `R_packages.txt` and
`session_info.txt`. CellChat is pinned to the Git commit used in the analysis.

The ssGSEA scripts require an authorized local MSigDB 2025.1.Hs snapshot. It is
not redistributed in this archive. Set its path before running:

```bash
export MSIGDB_SNAPSHOT=/path/to/msigdb_homosapiens_2025.1.Hs.rds
```
