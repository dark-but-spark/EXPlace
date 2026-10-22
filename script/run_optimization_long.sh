#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
CUDA_HOME="${CUDA_HOME:-/data0/cjy_cxsj/cuda}"
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"
CONFIG="${CONFIG:-optimization_long}"
BENCHMARKS_TEXT="${BENCHMARKS:-superblue18}"
SEEDS_TEXT="${SEEDS:-3 4 5}"
# GPU_IDS is supplied by the scheduler/user. Auto mode deliberately leaves
# GPU selection to CUDA_VISIBLE_DEVICES and runs one task at a time.
GPU_IDS_TEXT="${GPU_IDS:-}"
VARIANTS_TEXT="${VARIANTS:-baseline compact worker masks ppo policy_decode optimized amp_fp16 amp_bf16}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/optimization_runs/$(date -u +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-1}"
ALLOW_CPU="${ALLOW_CPU:-0}"
REQUIRE_PREPROCESSED="${REQUIRE_PREPROCESSED:-1}"
DRY_RUN="${DRY_RUN:-0}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-4096}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-80}"
TASK_ORDER="${TASK_ORDER:-variant}"

if [[ ! -x "${NVCC_BIN}" ]]; then
    echo "CUDA 11.8 nvcc not found: ${NVCC_BIN}" >&2
    exit 2
fi
export CUDA_HOME NVCC_BIN
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

select_idle_gpu() {
    local selected
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "GPU_IDS=auto requires nvidia-smi" >&2
        return 2
    }
    selected="$(nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', *' -v min_free="${GPU_MIN_FREE_MIB}" \
            -v max_util="${GPU_MAX_UTILIZATION}" '
            $2 >= min_free && $4 <= max_util {
                if ($2 > best_free) { best_free=$2; best=$1 }
            }
            END { if (best != "") print best }
        ' | head -n 1)"
    if [[ -z "${selected}" ]]; then
        echo "No idle GPU meets GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB}, GPU_MAX_UTILIZATION=${GPU_MAX_UTILIZATION}" >&2
        return 1
    fi
    printf '%s\n' "${selected}"
}

if [[ "${RUN_ROOT}" != /* ]]; then
    RUN_ROOT="${PROJECT_ROOT}/${RUN_ROOT}"
fi
read -r -a BENCHMARKS_ARRAY <<< "${BENCHMARKS_TEXT}"
read -r -a SEEDS_ARRAY <<< "${SEEDS_TEXT}"
if [[ -n "${GPU_IDS_TEXT}" ]]; then
    read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
else
    GPU_IDS_ARRAY=(auto)
    GPU_IDS_TEXT=auto
fi
read -r -a VARIANTS_ARRAY <<< "${VARIANTS_TEXT}"

TASKS=()
case "${TASK_ORDER}" in
    variant)
        for variant in "${VARIANTS_ARRAY[@]}"; do
            for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
                for seed in "${SEEDS_ARRAY[@]}"; do
                    TASKS+=("${variant} ${benchmark} ${seed}")
                done
            done
        done
        ;;
    seed_interleaved)
        for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
            for seed in "${SEEDS_ARRAY[@]}"; do
                for variant in "${VARIANTS_ARRAY[@]}"; do
                    TASKS+=("${variant} ${benchmark} ${seed}")
                done
            done
        done
        ;;
    pair_alternating)
        if (( ${#VARIANTS_ARRAY[@]} != 2 )); then
            echo "TASK_ORDER=pair_alternating requires exactly two VARIANTS" >&2
            exit 2
        fi
        for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
            for seed_index in "${!SEEDS_ARRAY[@]}"; do
                seed="${SEEDS_ARRAY[${seed_index}]}"
                if (( seed_index % 2 == 0 )); then
                    TASKS+=("${VARIANTS_ARRAY[0]} ${benchmark} ${seed}")
                    TASKS+=("${VARIANTS_ARRAY[1]} ${benchmark} ${seed}")
                else
                    TASKS+=("${VARIANTS_ARRAY[1]} ${benchmark} ${seed}")
                    TASKS+=("${VARIANTS_ARRAY[0]} ${benchmark} ${seed}")
                fi
            done
        done
        ;;
    *)
        echo "TASK_ORDER must be variant, seed_interleaved, or pair_alternating" >&2
        exit 2
        ;;
esac

if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in python3 /usr/bin/python3; do
        if command -v "${candidate}" >/dev/null 2>&1 \
            && "${candidate}" -c 'import torch, ray, yaml' >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "${candidate}")"
            break
        fi
    done
fi
if [[ -z "${PYTHON_BIN}" ]] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Set PYTHON_BIN to an environment containing torch, ray and yaml." >&2
    exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/config/${CONFIG}.yaml" ]]; then
    echo "Config not found: config/${CONFIG}.yaml" >&2
    exit 2
fi

case_args() {
    case "$1" in
        baseline)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation --no-policy-mask-fp32 ;;
        compact)
            printf '%s\n' --compact-transition --no-worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation --no-policy-mask-fp32 ;;
        worker)
            printf '%s\n' --no-compact-transition --worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation --no-policy-mask-fp32 ;;
        masks)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --mask-cache --vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation --no-policy-mask-fp32 ;;
        ppo)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --batch-device-copy \
                --no-mixed-precision --no-async-evaluation --no-policy-mask-fp32 ;;
        policy_decode)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation --policy-mask-fp32 ;;
        optimized)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --no-mixed-precision \
                --no-async-evaluation --policy-mask-fp32 ;;
        amp_bf16)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --mixed-precision \
                --no-async-evaluation --policy-mask-fp32 ;;
        masks_ppo)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --mask-cache --vectorized-occupancy --batch-device-copy \
                --no-mixed-precision --no-async-evaluation --no-policy-mask-fp32 ;;
        masks_amp_bf16)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --mask-cache --vectorized-occupancy --no-batch-device-copy \
                --mixed-precision --no-async-evaluation --policy-mask-fp32 ;;
        ppo_amp_bf16)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --batch-device-copy \
                --mixed-precision --no-async-evaluation --policy-mask-fp32 ;;
        masks_ppo_amp_bf16)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --mask-cache --vectorized-occupancy --batch-device-copy \
                --mixed-precision --no-async-evaluation --policy-mask-fp32 ;;
        amp_fp16)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --mixed-precision \
                --amp-dtype float16 --no-async-evaluation --policy-mask-fp32 ;;
        async_off)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --no-mixed-precision \
                --no-async-evaluation --no-policy-mask-fp32 ;;
        async_on)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --no-mixed-precision \
                --async-evaluation --no-policy-mask-fp32 ;;
        *) echo "Unknown variant: $1" >&2; return 2 ;;
    esac
}

for variant in "${VARIANTS_ARRAY[@]}"; do case_args "${variant}" >/dev/null; done
for value in "${BENCHMARKS_ARRAY[@]}" "${SEEDS_ARRAY[@]}" "${GPU_IDS_ARRAY[@]}"; do
    [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe matrix value: ${value}" >&2; exit 2; }
done
for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
    [[ "${gpu_id}" == auto || "${gpu_id}" =~ ^[0-9]+$ ]] || {
        echo "GPU_IDS must contain non-negative integer IDs or auto: ${gpu_id}" >&2
        exit 2
    }
done

if [[ "${REQUIRE_PREPROCESSED}" == 1 ]]; then
    for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
        [[ -f "${PROJECT_ROOT}/preprocessed_data/${benchmark}.pt" ]] || {
            echo "Missing preprocessed_data/${benchmark}.pt" >&2; exit 2;
        }
    done
fi
if [[ "${ALLOW_CPU}" != 1 ]]; then
    cuda_count="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
    if (( cuda_count == 0 )); then
        echo "No CUDA devices visible. Use the GPU container or ALLOW_CPU=1." >&2
        exit 2
    fi
    for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
        [[ "${gpu_id}" == auto ]] && continue
        if (( gpu_id >= cuda_count )); then
            echo "GPU ${gpu_id} requested, but PyTorch sees ${cuda_count} CUDA device(s)." >&2
            exit 2
        fi
    done
fi

mkdir -p "${RUN_ROOT}"
RESULTS_CSV="${RUN_ROOT}/results.csv"
if [[ ! -f "${RESULTS_CSV}" ]]; then
    printf 'variant,benchmark,seed,gpu,status,duration_seconds,mean_rollout_seconds,mean_ppo_seconds,log_dir\n' > "${RESULTS_CSV}"
fi

for task in "${TASKS[@]}"; do
    read -r variant benchmark seed <<< "${task}"
    seed_index=0
    for candidate_index in "${!SEEDS_ARRAY[@]}"; do
        if [[ "${SEEDS_ARRAY[${candidate_index}]}" == "${seed}" ]]; then
            seed_index="${candidate_index}"
            break
        fi
    done
    mapfile -t VARIANT_ARGS < <(case_args "${variant}")
    gpu="${GPU_IDS_ARRAY[$((seed_index % ${#GPU_IDS_ARRAY[@]}))]}"
    task_dir="${RUN_ROOT}/${variant}/${benchmark}/seed_${seed}"
    mkdir -p "${task_dir}"
    if [[ "${RESUME}" == 1 && -f "${task_dir}/SUCCESS" ]]; then
        echo "RESUME skip ${variant} ${benchmark} seed=${seed}"
        continue
    fi
    command=("${PYTHON_BIN}" -m src.main --benchmark "${benchmark}"
                     --config "${CONFIG}" --seed "${seed}"
                     --log-dir "${task_dir}/output" "${VARIANT_ARGS[@]}")
    if [[ "${gpu}" != auto ]]; then
        command+=(--gpu "${gpu}")
    fi
    printf '%q ' "${command[@]}" > "${task_dir}/command.sh"
    printf '\n' >> "${task_dir}/command.sh"
    if [[ "${DRY_RUN}" == 1 ]]; then
        echo "DRY-RUN ${variant} ${benchmark} seed=${seed} gpu=${gpu}"
        continue
    fi
    start="$(date +%s)"
    status=SUCCESS
    task_gpu="${gpu}"
    if [[ "${task_gpu}" == auto ]]; then
        task_gpu="$(select_idle_gpu)" || {
            rc=$?
            status=FAILED
            printf '%s\n' "exit_code=${rc}" "duration_seconds=0" > "${task_dir}/${status}"
            echo "FAILED: no suitable idle GPU for ${variant} ${benchmark} seed=${seed}" >&2
            continue
        }
    fi
    if [[ "${gpu}" == auto ]]; then
        echo "Selected idle GPU ${task_gpu} for ${variant} ${benchmark} seed=${seed}"
    fi
    if (cd "${PROJECT_ROOT}" && \
        CUDA_VISIBLE_DEVICES="${task_gpu}" \
        MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" \
        PYTHONUNBUFFERED=1 "${command[@]}") >"${task_dir}/stdout.log" 2>&1; then
        rc=0
    else
        rc=$?
        status=FAILED
    fi
    end="$(date +%s)"
    duration=$((end - start))
    # main.py may append benchmark/timestamp directories below the
    # requested log root; locate the single runtime file recursively.
    runtime_file="$(find "${task_dir}" -type f -name runtime.csv -print -quit)"
    means="$(${PYTHON_BIN} - "${runtime_file}" <<'PY'
import csv, sys
rollout, ppo = [], []
try:
    with open(sys.argv[1], newline='') as handle:
        for row in csv.DictReader(handle):
            rollout.append(float(row['state_transition_time']))
            ppo.append(float(row['training_time']))
except FileNotFoundError:
    pass
def mean(values):
    return sum(values) / len(values) if values else 0.0
print(f'{mean(rollout):.6f},{mean(ppo):.6f}')
PY
    )"
    IFS=, read -r mean_rollout mean_ppo <<< "${means}"
    printf '%s\n' "exit_code=${rc}" "duration_seconds=${duration}" "selected_gpu=${task_gpu}" > "${task_dir}/${status}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${variant}" "${benchmark}" "${seed}" "${task_gpu}" "${status}" \
        "${duration}" "${mean_rollout}" "${mean_ppo}" "${task_dir}" >> "${RESULTS_CSV}"
    [[ "${status}" == SUCCESS ]] || echo "FAILED: ${task_dir}" >&2
done

"${PYTHON_BIN}" - "${RESULTS_CSV}" "${RUN_ROOT}/speedups.csv" <<'PY'
import csv, sys
from pathlib import Path
results_path, output_path = map(Path, sys.argv[1:])
rows = list(csv.DictReader(results_path.open()))
baselines = {(r['benchmark'], r['seed']): r for r in rows if r['variant'] == 'baseline'}
if not baselines:
    # Physical async comparisons use async_off/async_on instead of a generic
    # baseline; the first listed variant is the explicit reference there.
    for row in rows:
        baselines.setdefault((row['benchmark'], row['seed']), row)
fields = list(rows[0]) + ['wall_speedup', 'rollout_speedup', 'ppo_speedup'] if rows else []
with output_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    if fields:
        writer.writeheader()
    for row in rows:
        base = baselines.get((row['benchmark'], row['seed']))
        if base and row['status'] == 'SUCCESS' and base['status'] == 'SUCCESS':
            def ratio(key):
                value = float(row[key]); reference = float(base[key])
                return f'{reference / value:.4f}' if value > 0 else ''
            row['wall_speedup'] = ratio('duration_seconds')
            row['rollout_speedup'] = ratio('mean_rollout_seconds')
            row['ppo_speedup'] = ratio('mean_ppo_seconds')
        else:
            row['wall_speedup'] = row['rollout_speedup'] = row['ppo_speedup'] = ''
        writer.writerow(row)
print(f'wrote {output_path}')
PY
echo "Results: ${RESULTS_CSV}"
echo "Speedups: ${RUN_ROOT}/speedups.csv"
