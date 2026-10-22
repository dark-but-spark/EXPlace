#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DREAMPLACE_ROOT="${DREAMPLACE_ROOT:-${PROJECT_ROOT}/DREAMPlace}"
PATCH_FILE="${PROJECT_ROOT}/patches/dreamplace/macro_legalize_local_grid_repair.patch"
TARGET_FILE="${DREAMPLACE_ROOT}/dreamplace/ops/macro_legalize/src/macro_legalize.cpp"

if [[ ! -f "${TARGET_FILE}" ]]; then
    echo "DREAMPlace source not found: ${TARGET_FILE}" >&2
    exit 2
fi
if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "DREAMPlace patch not found: ${PATCH_FILE}" >&2
    exit 2
fi
command -v patch >/dev/null 2>&1 || {
    echo "The patch utility is required to build DREAMPlace" >&2
    exit 2
}

if grep -Fq 'nearest grid-aligned side of an' "${TARGET_FILE}"; then
    echo "DREAMPlace iterative grid repair patch already applied"
    exit 0
fi

if grep -Fq 'local, grid-induced failure' "${TARGET_FILE}"; then
    echo "A legacy grid repair patch is already applied. Re-copy a clean DREAMPlace source tree before rebuilding." >&2
    exit 2
fi

patch --dry-run --directory "${DREAMPLACE_ROOT}" --forward --batch \
    --strip=1 < "${PATCH_FILE}" >/dev/null
patch --directory "${DREAMPLACE_ROOT}" --forward --batch \
    --strip=1 < "${PATCH_FILE}"
echo "Applied DREAMPlace local grid repair patch"
