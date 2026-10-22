#!/usr/bin/env bash

# Capture reproducibility metadata without requiring a CUDA-capable host.
set -u

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-${PROJECT_ROOT}/run_metadata.txt}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    METADATA_PYTHON="${PYTHON_BIN}"
elif [[ -x /usr/bin/python3 ]]; then
    METADATA_PYTHON=/usr/bin/python3
else
    METADATA_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${METADATA_PYTHON}" || ! -x "${METADATA_PYTHON}" ]]; then
    echo "No executable Python found for metadata capture" >&2
    exit 2
fi
if [[ -n "${NVCC_BIN:-}" ]]; then
    METADATA_NVCC="${NVCC_BIN}"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    METADATA_NVCC="${CUDA_HOME}/bin/nvcc"
else
    METADATA_NVCC="$(command -v nvcc || true)"
fi
mkdir -p "$(dirname -- "${OUTPUT_PATH}")"

search_text() {
    if command -v rg >/dev/null 2>&1; then
        rg "$@"
    else
        grep -E "$@"
    fi
}

{
    echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "project_root=${PROJECT_ROOT}"
    echo "pwd=$(pwd)"
    echo "python_path=${METADATA_PYTHON}"
    "${METADATA_PYTHON}" --version 2>&1 | sed 's/^/python_version=/' || true
    echo "cmake_path=$(command -v cmake || echo unavailable)"
    cmake --version 2>&1 | head -n 1 | sed 's/^/cmake_version=/' || true
    echo "cxx_path=$(command -v c++ || echo unavailable)"
    c++ --version 2>&1 | head -n 1 | sed 's/^/cxx_version=/' || true
    echo "git_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_branch=$(git -C "${PROJECT_ROOT}" branch --show-current 2>/dev/null || echo unavailable)"
    echo "git_status_begin"
    git -C "${PROJECT_ROOT}" status --short 2>/dev/null || true
    echo "git_status_end"

    echo "python_probe_begin"
    "${METADATA_PYTHON}" - <<'PY' 2>&1 || true
import sys
print("executable=" + sys.executable)
try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda=" + str(torch.version.cuda))
    print("cuda_available=" + str(torch.cuda.is_available()))
    print("cuda_device_count=" + str(torch.cuda.device_count()))
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print("cuda_device_%d=%s" % (index, torch.cuda.get_device_name(index)))
            print("cuda_capability_%d=%s" % (index, torch.cuda.get_device_capability(index)))
except Exception as error:
    print("torch_probe_error=%r" % (error,))
PY
    echo "python_probe_end"

    echo "nvcc_begin"
    if [[ -n "${METADATA_NVCC}" && -x "${METADATA_NVCC}" ]]; then
        echo "nvcc_path=${METADATA_NVCC}"
        "${METADATA_NVCC}" --version 2>&1 || true
    else
        echo "nvcc=unavailable"
    fi
    echo "nvcc_end"

    echo "driver_begin"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>&1 || true
    else
        echo "nvidia-smi=unavailable"
    fi
    echo "driver_end"

    echo "gpu_pci_begin"
    if command -v lspci >/dev/null 2>&1; then
        lspci -nn 2>&1 | search_text -i 'nvidia|vga compatible controller|3d controller' || true
    else
        echo "lspci=unavailable"
    fi
    echo "gpu_pci_end"

    echo "dreamplace_build_begin"
    for build_name in build build-cuda12; do
        cache_path="${PROJECT_ROOT}/DREAMPlace/${build_name}/CMakeCache.txt"
        echo "${build_name}_cache_begin"
        if [[ -f "${cache_path}" ]]; then
            search_text '^(CUDA_VERSION|CUDA_TOOLKIT_ROOT_DIR|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CXX_ABI|CMAKE_BUILD_TYPE|CMAKE_INSTALL_PREFIX|PYTHON_EXECUTABLE):' \
                "${cache_path}" || true
        else
            echo "cmake_cache=unavailable"
        fi
        echo "${build_name}_cache_end"
    done
    for install_name in install install-cuda12; do
        configure_path="${PROJECT_ROOT}/DREAMPlace/${install_name}/dreamplace/configure.py"
        echo "${install_name}_configure_begin"
        if [[ -f "${configure_path}" ]]; then
            search_text '"(CUDA_FOUND|CUDA_TOOLKIT_ROOT_DIR|CMAKE_CXX_ABI|CMAKE_CXX_COMPILER|CMAKE_BUILD_TYPE)"' \
                "${configure_path}" || true
        else
            echo "dreamplace_configure=unavailable"
        fi
        echo "${install_name}_configure_end"
    done
    echo "dreamplace_build_end"

    echo "tracked_generated_files_begin"
    for generated_file in hcg.lp vcg.lp; do
        if git -C "${PROJECT_ROOT}" ls-files --error-unmatch "${generated_file}" >/dev/null 2>&1; then
            echo "${generated_file}=tracked"
        else
            echo "${generated_file}=not_tracked"
        fi
        if [[ -e "${PROJECT_ROOT}/${generated_file}" ]]; then
            sha256sum "${PROJECT_ROOT}/${generated_file}" || true
        else
            echo "${generated_file}=missing"
        fi
    done
    echo "tracked_generated_files_end"
} > "${OUTPUT_PATH}"

printf '%s\n' "Wrote ${OUTPUT_PATH}"
