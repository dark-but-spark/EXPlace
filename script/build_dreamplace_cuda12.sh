#!/usr/bin/env bash

# Configure and build a CUDA 12 DREAMPlace tree without touching the cu118 install.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REFERENCE_DREAMPLACE="${PROJECT_ROOT}/DREAMPlace"
DREAMPLACE_SOURCE="${DREAMPLACE_SOURCE:-${PROJECT_ROOT}/DREAMPlace-cuda12-src}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
CUDA_ARCH="${CUDA_ARCH:-}"
BUILD_DIR="${BUILD_DIR:-${REFERENCE_DREAMPLACE}/build-cuda12}"
INSTALL_DIR="${INSTALL_DIR:-${REFERENCE_DREAMPLACE}/install-cuda12}"
BUILD_JOBS="${BUILD_JOBS:-8}"
ACTION="${ACTION:-all}"
ALLOW_TORCH_CUDA_MISMATCH="${ALLOW_TORCH_CUDA_MISMATCH:-0}"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${CUDA12_RUN_DIR:-${PROJECT_ROOT}/cuda12_runs/${timestamp}}"

case "${ACTION}" in
    configure|build|install|all) ;;
    *) echo "ACTION must be configure, build, install, or all" >&2; exit 2 ;;
esac

for path in "${BUILD_DIR}" "${INSTALL_DIR}" "${RUN_DIR}"; do
    if [[ "${path}" != /* ]]; then
        echo "BUILD_DIR, INSTALL_DIR, and CUDA12_RUN_DIR must be absolute: ${path}" >&2
        exit 2
    fi
done
if [[ "${BUILD_DIR}" == "${DREAMPLACE_SOURCE}/build" ]]; then
    echo "Refusing to use the reference build directory: ${BUILD_DIR}" >&2
    exit 2
fi
if [[ "${BUILD_DIR}" == "${REFERENCE_DREAMPLACE}/build" ]]; then
    echo "Refusing to use the reference build directory: ${BUILD_DIR}" >&2
    exit 2
fi
if [[ "${INSTALL_DIR}" == "${DREAMPLACE_SOURCE}/install" \
    || "${INSTALL_DIR}" == "${REFERENCE_DREAMPLACE}/install" ]]; then
    echo "Refusing to overwrite the reference install: ${INSTALL_DIR}" >&2
    exit 2
fi
if [[ ! -f "${DREAMPLACE_SOURCE}/CMakeLists.txt" ]]; then
    echo "CUDA 12 source is not prepared: ${DREAMPLACE_SOURCE}" >&2
    echo "Run bash script/prepare_dreamplace_cuda12_source.sh first." >&2
    exit 2
fi
if ! rg -q 'DREAMPLACE_FORCE_CUDA' "${DREAMPLACE_SOURCE}/cmake/TorchExtension.cmake"; then
    echo "CUDA 12 source lacks the force-CUDA compatibility patch: ${DREAMPLACE_SOURCE}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA 12 nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
    exit 2
fi
cuda_release="$(${CUDA_HOME}/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
if [[ "${cuda_release}" != "12" ]]; then
    echo "Expected CUDA 12, found release ${cuda_release:-unknown} under ${CUDA_HOME}" >&2
    exit 2
fi
if [[ -z "${CUDA_ARCH}" ]]; then
    CUDA_ARCH="$(${PYTHON_BIN} - <<'PY' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
)"
fi
if [[ ! "${CUDA_ARCH}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Set CUDA_ARCH to the target compute capability, for example CUDA_ARCH=8.6" >&2
    exit 2
fi

readarray -t torch_info < <("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda or "none")
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
print(torch.utils.cmake_prefix_path)
PY
)
torch_version="${torch_info[0]}"
torch_cuda="${torch_info[1]}"
torch_abi="${torch_info[2]}"
torch_cmake_prefix="${torch_info[3]}"
if [[ "${torch_cuda}" != 12.* && "${ALLOW_TORCH_CUDA_MISMATCH}" != "1" ]]; then
    echo "PyTorch ${torch_version} uses CUDA ${torch_cuda}, not CUDA 12." >&2
    echo "Use a CUDA 12 PyTorch environment, or set ALLOW_TORCH_CUDA_MISMATCH=1 only for compile diagnosis." >&2
    exit 2
fi

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}" "${RUN_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_HOME
export PYTHON_BIN

metadata_path="${RUN_DIR}/metadata.txt"
NVCC_BIN="${CUDA_HOME}/bin/nvcc" \
    "${PROJECT_ROOT}/script/capture_run_metadata.sh" "${metadata_path}" >/dev/null
{
    echo "action=${ACTION}"
    echo "cuda_home=${CUDA_HOME}"
    echo "cuda_arch=${CUDA_ARCH}"
    echo "build_dir=${BUILD_DIR}"
    echo "install_dir=${INSTALL_DIR}"
    echo "torch_version=${torch_version}"
    echo "torch_cuda=${torch_cuda}"
    echo "torch_abi=${torch_abi}"
    echo "allow_torch_cuda_mismatch=${ALLOW_TORCH_CUDA_MISMATCH}"
} >> "${metadata_path}"

configure_command=(
    cmake -S "${DREAMPLACE_SOURCE}" -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    -DCMAKE_CXX_ABI="${torch_abi}"
    -DCMAKE_PREFIX_PATH="${torch_cmake_prefix}"
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"
    -DDREAMPLACE_FORCE_CUDA=ON
    -DPython_EXECUTABLE="${PYTHON_BIN}"
    -DPython_ROOT_DIR="$(dirname -- "$(dirname -- "${PYTHON_BIN}")")"
    -DPYTHON_EXECUTABLE="${PYTHON_BIN}"
)
build_command=(cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}")
install_command=(cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}")

printf '%q ' "${configure_command[@]}" > "${RUN_DIR}/configure.command"
printf '\n' >> "${RUN_DIR}/configure.command"
printf '%q ' "${build_command[@]}" > "${RUN_DIR}/build.command"
printf '\n' >> "${RUN_DIR}/build.command"
printf '%q ' "${install_command[@]}" > "${RUN_DIR}/install.command"
printf '\n' >> "${RUN_DIR}/install.command"

record_failure() {
    local phase="$1"
    local exit_code="$2"
    local log_path="${RUN_DIR}/${phase}.log"
    printf 'phase=%s\nexit_code=%s\n' "${phase}" "${exit_code}" > "${RUN_DIR}/FAILED"
    if ! rg -m 1 -n '(^|[^[:alpha:]])(fatal error:|error:|CMake Error)' \
        "${log_path}" > "${RUN_DIR}/first_error.txt"; then
        tail -n 40 "${log_path}" > "${RUN_DIR}/first_error.txt"
    fi
    echo "${phase} failed; see ${RUN_DIR}/first_error.txt" >&2
}

run_phase() {
    local phase="$1"
    shift
    set +e
    "$@" 2>&1 | tee "${RUN_DIR}/${phase}.log"
    local exit_code=${PIPESTATUS[0]}
    set -e
    if (( exit_code != 0 )); then
        record_failure "${phase}" "${exit_code}"
        return "${exit_code}"
    fi
}

if [[ "${ACTION}" == "configure" || "${ACTION}" == "all" ]]; then
    run_phase configure "${configure_command[@]}"
fi
if [[ "${ACTION}" == "build" || "${ACTION}" == "all" ]]; then
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        echo "No configured build at ${BUILD_DIR}; run ACTION=configure first" >&2
        exit 2
    fi
    run_phase build "${build_command[@]}"
fi
if [[ "${ACTION}" == "install" || "${ACTION}" == "all" ]]; then
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        echo "No configured build at ${BUILD_DIR}; run ACTION=configure first" >&2
        exit 2
    fi
    run_phase install "${install_command[@]}"
fi

printf 'action=%s\nfinished_utc=%s\n' "${ACTION}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${RUN_DIR}/SUCCESS"
echo "CUDA 12 ${ACTION} completed; evidence: ${RUN_DIR}"
