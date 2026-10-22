SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CUDA_HOME="${CUDA_HOME:-/data0/cjy_cxsj/cuda}"
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${NVCC_BIN}" ]]; then
    echo "Missing Python or CUDA compiler: PYTHON_BIN=${PYTHON_BIN}, NVCC_BIN=${NVCC_BIN}" >&2
    exit 2
fi

if ! CUDA_HOME="${CUDA_HOME}" "${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is not available"' 2>/dev/null; then
    echo "CUDA-enabled PyTorch cannot access a GPU; refusing to build CPU-only DREAMPlace" >&2
    exit 2
fi

export CUDA_HOME NVCC_BIN
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

bash "${SCRIPT_DIR}/apply_dreamplace_patch.sh"
cd "${PROJECT_ROOT}/DREAMPlace/build"
cmake .. \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    -DCUDA_NVCC_EXECUTABLE="${NVCC_BIN}" \
    -DPython_EXECUTABLE="${PYTHON_BIN}"
make -j 8
make -j 8 install
cd ../install
cd "${PROJECT_ROOT}"
