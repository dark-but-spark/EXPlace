#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/data0/cjy_cxsj/EXPlace-test/EXPlace
STAMP=$(date -u +%Y%m%d_%H%M%S)
RUN_ROOT="${PROJECT_ROOT}/long_runs/repro_segv_${STAMP}"
mkdir -p "${RUN_ROOT}/dumps_fixed3" "${RUN_ROOT}/dumps_adapt17"

cd "${PROJECT_ROOT}"
source script/activate_cuda118.sh >/dev/null 2>&1

run_one() {
    local variant=$1 seed=$2 gpu=$3 dumpdir=$4 logfile=$5
    (
        ulimit -c unlimited
        cd "${PROJECT_ROOT}"
        export EXPLACE_DUMP_PRE_TIMING="${dumpdir}"
        export MPLCONFIGDIR=/tmp
        export PYTHONUNBUFFERED=1
        exec /usr/bin/python3 -m src.main \
            --benchmark superblue3 \
            --config adaptive_global \
            --seed "${seed}" \
            --reward-weight-mode "$( [ "${variant}" = fixed_global ] && echo fixed || echo ppa_correlation )" \
            --reward-granularity global \
            --log-dir "${RUN_ROOT}/${variant}/superblue3/seed_${seed}/output" \
            --no-compact-transition --no-worker-rollout --no-mask-cache \
            --no-vectorized-occupancy --no-batch-device-copy --no-mixed-precision \
            --no-async-evaluation --no-policy-mask-fp32 \
            --gpu "${gpu}"
    ) > "${logfile}" 2>&1
    echo "exit=$? $variant seed=$seed" >> "${RUN_ROOT}/exitcodes.txt"
}

echo "[repro] run root ${RUN_ROOT}"
run_one fixed_global 3 3 "${RUN_ROOT}/dumps_fixed3" "${RUN_ROOT}/fixed_sb3_seed3.log" &
PID_A=$!
run_one adaptive_global 17 5 "${RUN_ROOT}/dumps_adapt17" "${RUN_ROOT}/adapt_sb3_seed17.log" &
PID_B=$!
wait "${PID_A}" "${PID_B}"
echo "[repro] done; exit codes:"
cat "${RUN_ROOT}/exitcodes.txt" 2>/dev/null
echo "[repro] dumps:"
ls -1 "${RUN_ROOT}/dumps_fixed3" "${RUN_ROOT}/dumps_adapt17" 2>/dev/null | tail -6
