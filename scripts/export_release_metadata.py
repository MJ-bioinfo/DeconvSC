#!/usr/bin/env python3
"""Export exact observation metadata for every locked generated H5AD."""

from pathlib import Path

import anndata as ad


ROOT = Path(__file__).resolve().parents[1]

for source in sorted((ROOT / "model_outputs").rglob("*.h5ad")):
    dataset = source.parent.name
    destination = ROOT / "metadata" / dataset / f"{source.stem}_obs.csv.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    obj = ad.read_h5ad(source, backed="r")
    obs = obj.obs.copy()
    obj.file.close()
    obs.index.name = "cell_id"
    obs.to_csv(destination, compression="gzip")
    print(f"{source.relative_to(ROOT)} -> {destination.relative_to(ROOT)} ({len(obs)} rows)")
