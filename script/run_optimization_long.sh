#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
CONFIG="${CONFIG:-optimization_long}"
BENCHMARKS_TEXT="${BENCHMARKS:-superblue18}"
SEEDS_TEXT="${SEEDS:-3 4 5}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2}"
VARIANTS_TEXT="${VARIANTS:-baseline compact worker masks ppo optimized amp_bf16}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/optimization_runs/$(date -u +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-1}"
ALLOW_CPU="${ALLOW_CPU:-0}"
REQUIRE_PREPROCESSED="${REQUIRE_PREPROCESSED:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${RUN_ROOT}" != /* ]]; then
    RUN_ROOT="${PROJECT_ROOT}/${RUN_ROOT}"
fi
read -r -a BENCHMARKS_ARRAY <<< "${BENCHMARKS_TEXT}"
read -r -a SEEDS_ARRAY <<< "${SEEDS_TEXT}"
read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
read -r -a VARIANTS_ARRAY <<< "${VARIANTS_TEXT}"

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
                --no-mixed-precision --no-async-evaluation ;;
        compact)
            printf '%s\n' --compact-transition --no-worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation ;;
        worker)
            printf '%s\n' --no-compact-transition --worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation ;;
        masks)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --mask-cache --vectorized-occupancy --no-batch-device-copy \
                --no-mixed-precision --no-async-evaluation ;;
        ppo)
            printf '%s\n' --no-compact-transition --no-worker-rollout \
                --no-mask-cache --no-vectorized-occupancy --batch-device-copy \
                --no-mixed-precision --no-async-evaluation ;;
        optimized)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --no-mixed-precision \
                --no-async-evaluation ;;
        amp_bf16)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --mixed-precision \
                --no-async-evaluation ;;
        async_off)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --no-mixed-precision \
                --no-async-evaluation ;;
        async_on)
            printf '%s\n' --compact-transition --worker-rollout --mask-cache \
                --vectorized-occupancy --batch-device-copy --no-mixed-precision \
                --async-evaluation ;;
        *) echo "Unknown variant: $1" >&2; return 2 ;;
    esac
}

for variant in "${VARIANTS_ARRAY[@]}"; do case_args "${variant}" >/dev/null; done
for value in "${BENCHMARKS_ARRAY[@]}" "${SEEDS_ARRAY[@]}" "${GPU_IDS_ARRAY[@]}"; do
    [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe matrix value: ${value}" >&2; exit 2; }
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
fi

mkdir -p "${RUN_ROOT}"
RESULTS_CSV="${RUN_ROOT}/results.csv"
if [[ ! -f "${RESULTS_CSV}" ]]; then
    printf 'variant,benchmark,seed,gpu,status,duration_seconds,mean_rollout_seconds,mean_ppo_seconds,log_dir\n' > "${RESULTS_CSV}"
fi

for variant in "${VARIANTS_ARRAY[@]}"; do
    mapfile -t VARIANT_ARGS < <(case_args "${variant}")
    for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
        for seed_index in "${!SEEDS_ARRAY[@]}"; do
            seed="${SEEDS_ARRAY[${seed_index}]}"
            gpu="${GPU_IDS_ARRAY[$((seed_index % ${#GPU_IDS_ARRAY[@]}))]}"
            task_dir="${RUN_ROOT}/${variant}/${benchmark}/seed_${seed}"
            mkdir -p "${task_dir}"
            if [[ "${RESUME}" == 1 && -f "${task_dir}/SUCCESS" ]]; then
                echo "RESUME skip ${variant} ${benchmark} seed=${seed}"
                continue
            fi
            command=("${PYTHON_BIN}" -m src.main --benchmark "${benchmark}"
                     --config "${CONFIG}" --seed "${seed}" --gpu "${gpu}"
                     --log-dir "${task_dir}/output" "${VARIANT_ARGS[@]}")
            printf '%q ' "${command[@]}" > "${task_dir}/command.sh"
            printf '\n' >> "${task_dir}/command.sh"
            if [[ "${DRY_RUN}" == 1 ]]; then
                echo "DRY-RUN ${variant} ${benchmark} seed=${seed} gpu=${gpu}"
                continue
            fi
            start="$(date +%s)"
            status=SUCCESS
            if (cd "${PROJECT_ROOT}" && MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" \
                PYTHONUNBUFFERED=1 "${command[@]}") >"${task_dir}/stdout.log" 2>&1; then
                rc=0
            else
                rc=$?
                status=FAILED
            fi
            end="$(date +%s)"
            duration=$((end - start))
            runtime_file="${task_dir}/output/runtime.csv"
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
            printf '%s\n' "exit_code=${rc}" "duration_seconds=${duration}" > "${task_dir}/${status}"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${variant}" "${benchmark}" "${seed}" "${gpu}" "${status}" \
                "${duration}" "${mean_rollout}" "${mean_ppo}" "${task_dir}" >> "${RESULTS_CSV}"
            [[ "${status}" == SUCCESS ]] || echo "FAILED: ${task_dir}" >&2
        done
    done
done

"${PYTHON_BIN}" - "${RESULTS_CSV}" "${RUN_ROOT}/speedups.csv" <<'PY'
import csv, sys
from pathlib import Path
results_path, output_path = map(Path, sys.argv[1:])
rows = list(csv.DictReader(results_path.open()))
baselines = {(r['benchmark'], r['seed']): r for r in rows if r['variant'] == 'baseline'}
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
