# Module 3: model evaluation

The main `evaluate_prediction.py` entry point calculates generated-versus-real
expression accuracy and associated visualizations. `expression_benchmark/`
rebuilds cross-method figures from the frozen `prophead/03_result_tables`
snapshot included in this release.
`supplementary_figure5/` reconstructs the normalized-expression and gene-gene
correlation analyses for Supplementary Figure 5.

BayesPrism, TAPE, CIBERSORTx and DISSECT are represented by their final
method-specific metric tables. Their external executables and commercial web
service are not bundled.
