#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
CUDA_HOME="${CUDA_HOME:-/data0/cjy_cxsj/cuda}"
NVCC_BIN="${NVCC_BIN:-${CUDA_HOME}/bin/nvcc}"
CONFIG="${CONFIG:-adaptive_global}"
BENCHMARKS_TEXT="${BENCHMARKS:-superblue18}"
SEEDS_TEXT="${SEEDS:-3}"
# GPU_IDS is supplied by the scheduler/user. Auto mode inherits the current
# CUDA allocation and keeps the matrix single-worker to avoid GPU contention.
GPU_IDS_TEXT="${GPU_IDS:-}"
VARIANTS_TEXT="${VARIANTS:-fixed_step fixed_global adaptive_step adaptive_global}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
ALLOW_CPU="${ALLOW_CPU:-0}"
REQUIRE_PREPROCESSED="${REQUIRE_PREPROCESSED:-1}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-4096}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-80}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
TASK_ORDER="${TASK_ORDER:-variant}"
# Legacy mode is the reproducible reward-ablation default. Optimized mode keeps
# the config's performance paths but still disables async evaluation by default.
EXECUTION_MODE="${EXECUTION_MODE:-legacy}"
ASYNC_EVALUATION="${ASYNC_EVALUATION:-0}"

case "${EXECUTION_MODE}" in
    legacy|optimized) ;;
    *) echo "EXECUTION_MODE must be legacy or optimized" >&2; exit 2 ;;
esac
case "${ASYNC_EVALUATION}" in
    0|1) ;;
    *) echo "ASYNC_EVALUATION must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${EXECUTION_MODE}" == "legacy" && "${ASYNC_EVALUATION}" == "1" ]]; then
    echo "ASYNC_EVALUATION=1 is incompatible with EXECUTION_MODE=legacy" >&2
    exit 2
fi

if [[ ! -x "${NVCC_BIN}" ]]; then
    echo "CUDA 11.8 nvcc not found: ${NVCC_BIN}" >&2
    exit 2
fi
export CUDA_HOME NVCC_BIN
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

select_idle_gpu() {
    local selected elapsed=0
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "GPU_IDS=auto requires nvidia-smi" >&2
        return 2
    }
    while :; do
        selected="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F', *' -v min_free="${GPU_MIN_FREE_MIB}" \
                -v max_util="${GPU_MAX_UTILIZATION}" \
                '$2 >= min_free && $3 <= max_util {
                    if ($2 > best_free) { best_free=$2; best=$1 }
                }
                END { if (best != "") print best }')"
        if [[ -n "${selected}" ]]; then
            printf '%s\n' "${selected}"
            return 0
        fi
        if (( elapsed >= GPU_WAIT_SECONDS )); then
            echo "No idle GPU meets GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB}, GPU_MAX_UTILIZATION=${GPU_MAX_UTILIZATION}" >&2
            return 1
        fi
        echo "Waiting for an idle GPU (${elapsed}/${GPU_WAIT_SECONDS}s)" >&2
        sleep "${GPU_POLL_SECONDS}"
        elapsed=$((elapsed + GPU_POLL_SECONDS))
    done
}

timestamp="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/long_runs/${timestamp}}"
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

if (( ${#BENCHMARKS_ARRAY[@]} == 0 || ${#SEEDS_ARRAY[@]} == 0 || ${#GPU_IDS_ARRAY[@]} == 0 || ${#VARIANTS_ARRAY[@]} == 0 )); then
    echo "BENCHMARKS, SEEDS, GPU_IDS, and VARIANTS must not be empty" >&2
    exit 2
fi

for value in "${BENCHMARKS_ARRAY[@]}" "${SEEDS_ARRAY[@]}" "${GPU_IDS_ARRAY[@]}"; do
    if [[ ! "${value}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "Unsafe matrix value: ${value}" >&2
        exit 2
    fi
done

declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
    if [[ "${gpu_id}" == auto ]]; then
        continue
    fi
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "GPU_IDS must contain non-negative integer IDs or auto: ${gpu_id}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "Duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu_id}"]=1
done

variant_args() {
    case "$1" in
        fixed_step) echo "fixed step" ;;
        fixed_global) echo "fixed global" ;;
        adaptive_step) echo "ppa_correlation step" ;;
        adaptive_global) echo "ppa_correlation global" ;;
        *)
            echo "Unknown variant '$1'. Valid variants: fixed_step fixed_global adaptive_step adaptive_global" >&2
            return 2
            ;;
    esac
}

for variant in "${VARIANTS_ARRAY[@]}"; do
    variant_args "${variant}" >/dev/null
done

if [[ -z "${PYTHON_BIN}" ]]; then
    for python_candidate in python3 /usr/bin/python3; do
        if command -v "${python_candidate}" >/dev/null 2>&1 \
            && "${python_candidate}" -c 'import torch, ray, yaml' >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "${python_candidate}")"
            break
        fi
    done
fi
if [[ -z "${PYTHON_BIN}" ]] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "No Python executable with torch, ray, and yaml was found." >&2
    echo "Set PYTHON_BIN=/absolute/path/to/python for the EXPlace environment." >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import torch, ray, yaml' >/dev/null 2>&1; then
    echo "Python is missing torch, ray, or yaml: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/config/${CONFIG}.yaml" ]]; then
    echo "Config not found: ${PROJECT_ROOT}/config/${CONFIG}.yaml" >&2
    exit 2
fi
for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
    if [[ "${REQUIRE_PREPROCESSED}" == "1" && ! -f "${PROJECT_ROOT}/preprocessed_data/${benchmark}.pt" ]]; then
        echo "Preprocessed data not found: preprocessed_data/${benchmark}.pt" >&2
        echo "Set REQUIRE_PREPROCESSED=0 only if preprocessing inputs are complete." >&2
        exit 2
    fi
    if [[ "${benchmark}" == superblue* ]]; then
        benchmark_json="${PROJECT_ROOT}/DREAMPlace/test/iccad2015.ot/${benchmark}.json"
    else
        benchmark_json="${PROJECT_ROOT}/DREAMPlace/test/or_cases/${benchmark}.json"
    fi
    if [[ ! -f "${benchmark_json}" ]]; then
        echo "DREAMPlace benchmark config not found: ${benchmark_json}" >&2
        exit 2
    fi
done

if [[ "${DRY_RUN}" != "1" && "${ALLOW_CPU}" != "1" ]]; then
    cuda_count="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
    for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
        [[ "${gpu_id}" == auto ]] && continue
        if (( gpu_id >= cuda_count )); then
            echo "GPU ${gpu_id} requested, but PyTorch sees ${cuda_count} CUDA device(s)." >&2
            echo "Run inside the GPU container or set ALLOW_CPU=1 intentionally." >&2
            exit 2
        fi
    done
fi

mkdir -p "${RUN_ROOT}"
MASTER_LOG="${RUN_ROOT}/master.log"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
PLAN_TSV="${RUN_ROOT}/plan.tsv"
SUMMARY_LOCK="${RUN_ROOT}/.summary.lock"

# Keep toolchain and worktree evidence beside every matrix, including dry-runs.
# A run without this file is not a reproducible formal result.
if [[ ! -x "${PROJECT_ROOT}/script/capture_run_metadata.sh" ]]; then
    echo "Metadata capture script is missing or not executable" >&2
    exit 2
fi
PYTHON_BIN="${PYTHON_BIN}" \
    "${PROJECT_ROOT}/script/capture_run_metadata.sh" "${RUN_ROOT}/run_metadata.txt" >/dev/null

if [[ ! -f "${SUMMARY_CSV}" ]]; then
    echo "variant,benchmark,seed,gpu,status,exit_code,duration_seconds,task_dir" > "${SUMMARY_CSV}"
fi
echo -e "task_id\tvariant\tbenchmark\tseed\tgpu\ttask_dir" > "${PLAN_TSV}"

log_message() {
    local message
    message="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "${message}"
    echo "${message}" >> "${MASTER_LOG}"
}

append_summary() {
    local row="$1"
    (
        flock -x 9
        echo "${row}" >> "${SUMMARY_CSV}"
    ) 9>"${SUMMARY_LOCK}"
}

TASKS=()
task_id=0
case "${TASK_ORDER}" in
    variant)
        for variant in "${VARIANTS_ARRAY[@]}"; do
            for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
                for seed in "${SEEDS_ARRAY[@]}"; do
                    gpu_index=$((task_id % ${#GPU_IDS_ARRAY[@]}))
                    gpu_id="${GPU_IDS_ARRAY[${gpu_index}]}"
                    task_dir="${RUN_ROOT}/${variant}/${benchmark}/seed_${seed}"
                    TASKS+=("${variant}|${benchmark}|${seed}|${gpu_id}|${task_dir}")
                    echo -e "${task_id}\t${variant}\t${benchmark}\t${seed}\t${gpu_id}\t${task_dir}" >> "${PLAN_TSV}"
                    task_id=$((task_id + 1))
                done
            done
        done
        ;;
    seed_interleaved)
        for benchmark in "${BENCHMARKS_ARRAY[@]}"; do
            for seed in "${SEEDS_ARRAY[@]}"; do
                for variant in "${VARIANTS_ARRAY[@]}"; do
                    gpu_index=$((task_id % ${#GPU_IDS_ARRAY[@]}))
                    gpu_id="${GPU_IDS_ARRAY[${gpu_index}]}"
                    task_dir="${RUN_ROOT}/${variant}/${benchmark}/seed_${seed}"
                    TASKS+=("${variant}|${benchmark}|${seed}|${gpu_id}|${task_dir}")
                    echo -e "${task_id}\t${variant}\t${benchmark}\t${seed}\t${gpu_id}\t${task_dir}" >> "${PLAN_TSV}"
                    task_id=$((task_id + 1))
                done
            done
        done
        ;;
    *)
        echo "TASK_ORDER must be variant or seed_interleaved" >&2
        exit 2
        ;;
esac

run_task() {
    local task="$1"
    local variant benchmark seed gpu_id task_dir
    local task_gpu
    local reward_weight_mode reward_granularity
    local start_epoch end_epoch duration rc status
    IFS='|' read -r variant benchmark seed gpu_id task_dir <<< "${task}"
    read -r reward_weight_mode reward_granularity <<< "$(variant_args "${variant}")"

    mkdir -p "${task_dir}"
    if [[ "${RESUME}" == "1" && -f "${task_dir}/SUCCESS" ]]; then
        log_message "SKIP ${variant} ${benchmark} seed=${seed}: SUCCESS already exists"
        append_summary "${variant},${benchmark},${seed},${gpu_id},SKIPPED,0,0,${task_dir}"
        return 0
    fi

    task_gpu="${gpu_id}"
    if [[ "${gpu_id}" == auto && "${DRY_RUN}" != "1" ]]; then
        if task_gpu="$(select_idle_gpu)"; then
            :
        else
            rc=$?
            status="FAILED"
            printf 'exit_code=%s\nreason=no_idle_gpu\n' "${rc}" > "${task_dir}/${status}"
            append_summary "${variant},${benchmark},${seed},auto,${status},${rc},0,${task_dir}"
            log_message "FAILED ${variant} ${benchmark} seed=${seed}: no idle GPU"
            return 1
        fi
        log_message "Selected idle GPU ${task_gpu} for ${variant} ${benchmark} seed=${seed}"
    fi

    local -a command=(
        "${PYTHON_BIN}" -m src.main
        --benchmark "${benchmark}"
        --config "${CONFIG}"
        --seed "${seed}"
        --reward-weight-mode "${reward_weight_mode}"
        --reward-granularity "${reward_granularity}"
        --log-dir "${task_dir}/output"
    )
    if [[ "${EXECUTION_MODE}" == "legacy" ]]; then
        command+=(
            --no-compact-transition
            --no-worker-rollout
            --no-mask-cache
            --no-vectorized-occupancy
            --no-batch-device-copy
            --no-mixed-precision
            --no-async-evaluation
            --no-policy-mask-fp32
        )
    else
        if [[ "${ASYNC_EVALUATION}" == "1" ]]; then
            command+=(--async-evaluation)
        else
            command+=(--no-async-evaluation)
        fi
    fi
    if [[ "${task_gpu}" != auto ]]; then
        command+=(--gpu "${task_gpu}")
    fi

    {
        printf '#!/usr/bin/env bash\n'
        printf 'cd %q\n' "${PROJECT_ROOT}"
        printf 'export MPLCONFIGDIR=%q\n' "${MPLCONFIGDIR:-/tmp}"
        printf 'export PYTHONUNBUFFERED=1\n'
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "${task_dir}/command.sh"
    chmod +x "${task_dir}/command.sh"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log_message "DRY-RUN gpu=${gpu_id} ${variant} ${benchmark} seed=${seed}"
        return 0
    fi

    rm -f "${task_dir}/SUCCESS" "${task_dir}/FAILED"
    start_epoch="$(date +%s)"
    printf '%s\n' "${start_epoch}" > "${task_dir}/RUNNING"
    log_message "START gpu=${gpu_id} ${variant} ${benchmark} seed=${seed}"

    if (
        cd "${PROJECT_ROOT}"
        if [[ "${gpu_id}" == auto ]]; then
            CUDA_VISIBLE_DEVICES="${task_gpu}" \
                MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" PYTHONUNBUFFERED=1 "${command[@]}"
        else
            MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}" PYTHONUNBUFFERED=1 "${command[@]}"
        fi
    ) > "${task_dir}/stdout.log" 2>&1; then
        rc=0
        status="SUCCESS"
    else
        rc=$?
        status="FAILED"
    fi

    end_epoch="$(date +%s)"
    duration=$((end_epoch - start_epoch))
    rm -f "${task_dir}/RUNNING"
    printf 'exit_code=%s\nduration_seconds=%s\nfinished_utc=%s\n' \
        "${rc}" "${duration}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${task_dir}/${status}"
    append_summary "${variant},${benchmark},${seed},${task_gpu},${status},${rc},${duration},${task_dir}"
    log_message "${status} gpu=${task_gpu} ${variant} ${benchmark} seed=${seed} duration=${duration}s rc=${rc}"
    return "${rc}"
}

worker() {
    local worker_index="$1"
    local task_index
    local worker_failed=0
    for ((task_index=worker_index; task_index<${#TASKS[@]}; task_index+=${#GPU_IDS_ARRAY[@]})); do
        if ! run_task "${TASKS[${task_index}]}"; then
            worker_failed=1
            if [[ "${FAIL_FAST}" == "1" ]]; then
                break
            fi
        fi
    done
    return "${worker_failed}"
}

log_message "Matrix root: ${RUN_ROOT}"
log_message "Tasks: ${#TASKS[@]}, order: ${TASK_ORDER}, execution_mode: ${EXECUTION_MODE}, async_evaluation: ${ASYNC_EVALUATION}, GPUs: ${GPU_IDS_TEXT}, config: ${CONFIG}, python: ${PYTHON_BIN}, dry_run: ${DRY_RUN}, allow_cpu: ${ALLOW_CPU}"

if [[ "${DRY_RUN}" == "1" ]]; then
    for task in "${TASKS[@]}"; do
        run_task "${task}"
    done
    log_message "Dry run complete. Plan: ${PLAN_TSV}"
    exit 0
fi

PIDS=()
for ((worker_index=0; worker_index<${#GPU_IDS_ARRAY[@]}; worker_index++)); do
    worker "${worker_index}" &
    PIDS+=("$!")
done

overall_rc=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        overall_rc=1
    fi
done

if [[ "${overall_rc}" == "0" ]]; then
    log_message "All matrix tasks completed successfully. Summary: ${SUMMARY_CSV}"
else
    log_message "Matrix completed with failures. Inspect ${SUMMARY_CSV} and task stdout.log files."
fi
exit "${overall_rc}"
