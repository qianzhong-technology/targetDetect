#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/maixcdk_project"
OUTPUT_LIB="${SCRIPT_DIR}/libtargetdetect_accel.so"
OUTPUT_DL_LIB_DIR="${SCRIPT_DIR}/dl_lib"

if ! command -v maixcdk >/dev/null 2>&1; then
  echo "error: maixcdk command not found. Install MaixCDK tools first." >&2
  exit 1
fi

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "error: missing MaixCDK project directory: ${PROJECT_DIR}" >&2
  exit 1
fi

pushd "${PROJECT_DIR}" >/dev/null
maixcdk -p maixcam build

LIB_CANDIDATE="$(find build dist -type f -name 'libtargetdetect_accel.so' 2>/dev/null | head -n 1 || true)"
if [ -z "${LIB_CANDIDATE}" ]; then
  echo "error: build finished but libtargetdetect_accel.so not found in build/ or dist/." >&2
  exit 1
fi

cp -f "${LIB_CANDIDATE}" "${OUTPUT_LIB}"
echo "synced: ${OUTPUT_LIB}"

if [ -d "build/dl_lib" ]; then
  rm -rf "${OUTPUT_DL_LIB_DIR}"
  mkdir -p "${OUTPUT_DL_LIB_DIR}"
  cp -a build/dl_lib/. "${OUTPUT_DL_LIB_DIR}/"
  echo "synced: ${OUTPUT_DL_LIB_DIR}"
else
  echo "warn: no build/dl_lib produced; relying on system runtime libraries." >&2
fi

popd >/dev/null
