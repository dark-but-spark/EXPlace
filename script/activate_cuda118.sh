#!/usr/bin/env bash

# Source this file before running EXPlace GPU/DREAMPlace experiments.
# The reference host toolchain is CUDA 11.8 under /data0/cjy_cxsj/cuda.
set -u

CUDA_HOME="${CUDA_HOME:-/data0/cjy_cxsj/cuda}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"

if [[ ! -x "${NVCC_BIN}" ]]; then
    echo "CUDA nvcc not found: ${NVCC_BIN}" >&2
    return 2 2>/dev/null || exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found: ${PYTHON_BIN}" >&2
    return 2 2>/dev/null || exit 2
fi

export CUDA_HOME PYTHON_BIN NVCC_BIN
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "CUDA_HOME=${CUDA_HOME}"
echo "NVCC_BIN=${NVCC_BIN}"
echo "PYTHON_BIN=${PYTHON_BIN}"
"${NVCC_BIN}" --version | sed -n '1,4p'
"${PYTHON_BIN}" -c 'import torch; print("torch=" + torch.__version__); print("torch_cuda=" + str(torch.version.cuda)); print("cuda_available=" + str(torch.cuda.is_available()))'
