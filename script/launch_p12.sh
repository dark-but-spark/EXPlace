#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data0/cjy_cxsj/EXPlace-test/EXPlace
cd "${PROJECT_ROOT}"

export CONFIG=adaptive_global
export BENCHMARKS="superblue1 superblue3"
export SEEDS="3 7 11 17 23 29"
export VARIANTS="fixed_global adaptive_global"
export GPU_IDS="0 1 2 3 4 5 6 7"
export PYTHON_BIN=/usr/bin/python3
export EXECUTION_MODE=legacy
export RESUME=1
export RUN_ROOT="${PROJECT_ROOT}/long_runs/p12_multidesign_20261109"

echo "[launch] CONFIG=${CONFIG} BENCHMARKS=${BENCHMARKS}"
echo "[launch] SEEDS=${SEEDS} VARIANTS=${VARIANTS} GPU_IDS=${GPU_IDS}"
echo "[launch] DRY_RUN=${DRY_RUN:-0} RESUME=${RESUME} RUN_ROOT=${RUN_ROOT}"

bash script/run_experiment_matrix.sh
