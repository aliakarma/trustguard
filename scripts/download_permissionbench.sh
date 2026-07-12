#!/usr/bin/env bash
# scripts/download_permissionbench.sh
# ─────────────────────────────────────────────────────────────────────────────
# Downloads the pre-processed PermissionBench release (feature vectors,
# per-permission labels, SHA-256 hashes, datasheet) to data/permissionbench/.
#
# Per the AndroZoo access agreement and Google Play terms, raw APKs and Play
# store metadata are NOT redistributed. The release contains everything
# needed to reproduce the paper's tables except the raw-text modality; to
# rebuild the full corpus (including retrieving APKs under your own AndroZoo
# agreement) use scripts/build_dataset.py instead.
#
# Usage:
#   bash scripts/download_permissionbench.sh [TARGET_DIR]
#
# The release URL can be overridden with the PERMISSIONBENCH_URL environment
# variable (e.g. to pin a specific version or use an institutional mirror).

set -euo pipefail

TARGET_DIR=${1:-"data/permissionbench"}
PB_URL=${PERMISSIONBENCH_URL:-"https://github.com/aliakarma/PermissionBench/releases/latest/download/permissionbench.tar.gz"}
ARCHIVE="$(mktemp -d)/permissionbench.tar.gz"

echo "PermissionBench download"
echo "  URL:    $PB_URL"
echo "  Target: $TARGET_DIR"

mkdir -p "$TARGET_DIR"

if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$ARCHIVE" "$PB_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$ARCHIVE" "$PB_URL"
else
    echo "ERROR: neither curl nor wget found." >&2
    exit 1
fi

tar -xzf "$ARCHIVE" -C "$TARGET_DIR"
rm -f "$ARCHIVE"

echo ""
echo "Verifying release contents ..."
for f in train.parquet val.parquet test.parquet datasheet.md sha256_hashes.txt; do
    if [ -e "$TARGET_DIR/$f" ]; then
        echo "  [ok] $f"
    else
        echo "  [MISSING] $f — the release may be incomplete or a different version." >&2
    fi
done

echo ""
echo "Done. Expected splits (paper §PermissionBench):"
echo "  train 43,288 benign / 10,158 malicious"
echo "  val    6,184 benign /  1,451 malicious"
echo "  test  12,368 benign /  2,903 malicious"
echo "  total 76,352 records"
