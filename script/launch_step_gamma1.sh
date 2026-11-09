#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data0/cjy_cxsj/EXPlace-test/EXPlace
cd "${PROJECT_ROOT}"

export CONFIG=step_gamma1
export BENCHMARKS="superblue18"
export SEEDS="3 7 11"
export VARIANTS="fixed_step adaptive_step"
export GPU_IDS="3 5"
export PYTHON_BIN=/usr/bin/python3
export EXECUTION_MODE=legacy
export RESUME=1

if [[ -n "${RUN_ROOT:-}" ]]; then
    export RUN_ROOT="${PROJECT_ROOT}/long_runs/${RUN_ROOT}"
else
    export RUN_ROOT="${PROJECT_ROOT}/long_runs/$(date -u +%Y%m%d_%H%M%S)_step_gamma1"
fi

echo "[launch] CONFIG=${CONFIG} BENCHMARKS=${BENCHMARKS} SEEDS=${SEEDS} VARIANTS=${VARIANTS} GPU_IDS=${GPU_IDS}"
echo "[launch] EXECUTION_MODE=${EXECUTION_MODE} DRY_RUN=${DRY_RUN:-0} RESUME=${RESUME} RUN_ROOT=${RUN_ROOT}"

bash script/run_experiment_matrix.sh
