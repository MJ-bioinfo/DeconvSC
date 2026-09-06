#!/usr/bin/env python3
"""Build the Figshare checksum manifest and a compact artifact inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_OUT = ROOT / "MANIFEST.sha256"
TABLE_OUT = ROOT / "provenance/ARTIFACT_MANIFEST.tsv"
EXCLUDED = {
    CHECKSUM_OUT.relative_to(ROOT).as_posix(),
    TABLE_OUT.relative_to(ROOT).as_posix(),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def h5ad_shape(path: Path) -> str:
    if path.suffix.lower() != ".h5ad":
        return ""
    obj = ad.read_h5ad(path, backed="r")
    shape = f"{obj.n_obs}x{obj.n_vars}"
    obj.file.close()
    return shape


def main() -> None:
    rows = []
    checksum_lines = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED or relative.startswith("work/") or "__pycache__" in path.parts:
            continue
        digest = sha256_file(path)
        checksum_lines.append(f"{digest}  {relative}")
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "h5ad_shape": h5ad_shape(path),
        })
    CHECKSUM_OUT.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(TABLE_OUT, sep="\t", index=False)
    print(f"wrote {CHECKSUM_OUT.relative_to(ROOT)} ({len(rows)} files)")
    print(f"wrote {TABLE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
