#!/usr/bin/env bash

# Create a CUDA 12 experiment source tree while preserving DREAMPlace/ unchanged.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REFERENCE_SOURCE="${REFERENCE_SOURCE:-${PROJECT_ROOT}/DREAMPlace}"
CUDA12_SOURCE="${CUDA12_SOURCE:-${PROJECT_ROOT}/DREAMPlace-cuda12-src}"
PATCH_DIR="${PROJECT_ROOT}/cuda12_compat"
EXPECTED_TORCH_EXTENSION_SHA256="39f41961710905588c809373b57e4009659d3457a445fa4dadd7b4e218046557"
EXPECTED_UTILS_CUB_SHA256="ea6df158b1e4b63b4e3f916b372aae83ab426be72e2a27b4688fe998c8c067d0"

if [[ ! -f "${REFERENCE_SOURCE}/cmake/TorchExtension.cmake" ]]; then
    echo "Reference DREAMPlace source is missing: ${REFERENCE_SOURCE}" >&2
    exit 2
fi
if [[ "${CUDA12_SOURCE}" != /* || "${REFERENCE_SOURCE}" != /* ]]; then
    echo "REFERENCE_SOURCE and CUDA12_SOURCE must be absolute" >&2
    exit 2
fi
if [[ "${CUDA12_SOURCE}" == "${REFERENCE_SOURCE}" ]]; then
    echo "Refusing to patch the reference DREAMPlace source in place" >&2
    exit 2
fi
if [[ -e "${CUDA12_SOURCE}" ]]; then
    if rg -q 'DREAMPLACE_FORCE_CUDA' "${CUDA12_SOURCE}/cmake/TorchExtension.cmake"; then
        echo "CUDA 12 source already prepared: ${CUDA12_SOURCE}"
        exit 0
    fi
    echo "Destination already exists but is not the prepared source: ${CUDA12_SOURCE}" >&2
    exit 2
fi

actual_hash="$(sha256sum "${REFERENCE_SOURCE}/cmake/TorchExtension.cmake" | awk '{print $1}')"
if [[ "${actual_hash}" != "${EXPECTED_TORCH_EXTENSION_SHA256}" ]]; then
    echo "Reference TorchExtension.cmake changed; review the compatibility patch first." >&2
    echo "expected=${EXPECTED_TORCH_EXTENSION_SHA256}" >&2
    echo "actual=${actual_hash}" >&2
    exit 2
fi
actual_utils_cub_hash="$(sha256sum "${REFERENCE_SOURCE}/dreamplace/ops/utility/src/utils_cub.cuh" | awk '{print $1}')"
if [[ "${actual_utils_cub_hash}" != "${EXPECTED_UTILS_CUB_SHA256}" ]]; then
    echo "Reference utils_cub.cuh changed; review the compatibility patch first." >&2
    echo "expected=${EXPECTED_UTILS_CUB_SHA256}" >&2
    echo "actual=${actual_utils_cub_hash}" >&2
    exit 2
fi

mkdir -p "${CUDA12_SOURCE}"
rsync -a \
    --exclude '/build*/' \
    --exclude '/install*/' \
    --exclude '__pycache__/' \
    "${REFERENCE_SOURCE}/" "${CUDA12_SOURCE}/"
for patch_path in "${PATCH_DIR}"/*.patch; do
    patch --directory="${CUDA12_SOURCE}" --strip=1 --forward --batch < "${patch_path}"
done

{
    echo "prepared_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "reference_source=${REFERENCE_SOURCE}"
    echo "reference_torch_extension_sha256=${actual_hash}"
    echo "reference_utils_cub_sha256=${actual_utils_cub_hash}"
    for patch_path in "${PATCH_DIR}"/*.patch; do
        echo "compatibility_patch=${patch_path}"
    done
} > "${CUDA12_SOURCE}/CUDA12_SOURCE_PROVENANCE.txt"

echo "Prepared CUDA 12 experiment source: ${CUDA12_SOURCE}"
