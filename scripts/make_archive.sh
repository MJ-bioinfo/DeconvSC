#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
OUT="${1:-$PARENT/DeconvSC_Figshare_v1.0.0.tar.gz}"

tar --exclude='work' --exclude='__pycache__' --exclude='*.pyc' \
  -czf "$OUT" -C "$PARENT" "$NAME"
echo "$OUT"
