#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-${LEKIWI_SOURCE_ROOT:-/home/lyang116/Desktop/APP/lehome}}"
DST_ROOT="$(git rev-parse --show-toplevel)"

REQUIRED_DIRS=(
  "Assets/lekiwi"
  "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_002"
  "Assets/Material/Garment"
)

echo "[sync] source: ${SRC_ROOT}"
echo "[sync] target: ${DST_ROOT}"

for rel in "${REQUIRED_DIRS[@]}"; do
  src_dir="${SRC_ROOT}/${rel}"
  dst_dir="${DST_ROOT}/${rel}"

  if [[ ! -d "${src_dir}" ]]; then
    echo "[sync][error] missing source directory: ${src_dir}" >&2
    exit 1
  fi

  mkdir -p "${dst_dir}"
  rsync -av "${src_dir}/" "${dst_dir}/"
  echo "[sync] copied: ${rel}"
done

echo "[sync] lekiwi assets copied to local target workspace."
echo "[sync] note: Assets/ is git-ignored in this repository."
