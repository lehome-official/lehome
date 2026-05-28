#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-${XLEROBOT_SOURCE_ROOT:-/home/lyang116/Desktop/APP/lehome}}"
DST_ROOT="$(git rev-parse --show-toplevel)"

REQUIRED_DIRS=(
  "Assets/robots/xlerobot"
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
  rsync -av --ignore-existing "${src_dir}/" "${dst_dir}/"
  echo "[sync] synced missing-only: ${rel}"
done

echo "[sync] done (assets remain git-ignored in this repository)."
