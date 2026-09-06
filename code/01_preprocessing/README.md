# Module 1: preprocessing

`preprocess_data.py` reads a processed single-cell reference and bulk expression
matrix, identifies compatible genes/signatures and writes the tensor/metadata
bundle used by training and prediction. Dataset-specific paths and annotation
columns are defined in `../../configs/preprocess_*.yaml`.

Example:

```bash
python code/01_preprocessing/preprocess_data.py --config configs/preprocess_gse141115.yaml
```
