#!/usr/bin/env bash
# Route A: get a *legal* EXPlace placement on an OpenROAD-case design.
# or.yaml is the paper config for these designs (corner_flag: True,
# core_area_block: 2, halo: 1) and fails DREAMPlace's legality check on
# ariane133.  Relax the legality-hostile knobs to the values that worked in
# config/adaptive_sanity.yaml (halo 0, core_area_block 0, corner_flag False)
# and shorten the run so we can iterate.
set -uo pipefail
cd /data0/cjy_cxsj/EXPlace-test/EXPlace || exit 1

DESIGN=${1:-bp_fe}
SEED=${2:-3}
GPU=${3:-3}
EPISODES=${4:-60}
INTERVAL=${5:-30}
TAG=${6:-r1}

CFG=config/or_legal_${TAG}.yaml
sed -e "s/^name : test/name : or_legal_${TAG}/" \
    -e "s/^episode : 1500/episode : ${EPISODES}/" \
    -e "s/^PO_episode : 500/PO_episode : ${EPISODES}/" \
    -e "s/^halo: 1/halo: 0/" \
    -e "s/^core_area_block: 2/core_area_block: 0/" \
    -e "s/^corner_flag : True/corner_flag : False/" \
    -e "s/^dataflow_cutoff : 0.80/dataflow_cutoff : 0.00/" \
    -e "s/^soft_coefficient : 0.2/soft_coefficient : 1.0/" \
    config/or.yaml > "$CFG"
printf '\nevaluation_interval : %s\nvisualization_interval : 0\nvisualize_prototype : False\nuse_reward_scaling : True\n' "$INTERVAL" >> "$CFG"

echo "[legal] config $CFG:"
grep -nE "^(name|episode|PO_episode|halo|core_area_block|corner_flag|dataflow_cutoff|soft_coefficient|evaluation_interval|use_timer_for_evaluation)" "$CFG"

RUN=long_runs/or_legal_${DESIGN}_${TAG}
mkdir -p "$RUN"
LOG=$RUN/run.log
echo "[legal] start $(date -u +%Y-%m-%dT%H:%M:%SZ) design=$DESIGN seed=$SEED gpu=$GPU" | tee "$LOG"

MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1 /usr/bin/python3 -m src.main \
    --benchmark "$DESIGN" \
    --config "or_legal_${TAG}" \
    --seed "$SEED" \
    --log-dir "$RUN/output" \
    --no-compact-transition --no-worker-rollout --no-mask-cache \
    --no-vectorized-occupancy --no-batch-device-copy --no-mixed-precision \
    --no-async-evaluation --no-policy-mask-fp32 \
    --gpu "$GPU" >> "$LOG" 2>&1
rc=$?
echo "[legal] rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

echo "[legal] physical evaluations:" | tee -a "$LOG"
find "$RUN" -name physical_evaluations.csv -exec cat {} \; 2>/dev/null | tee -a "$LOG"
echo "[legal] DEF produced:" | tee -a "$LOG"
find "$RUN" -name '*.def' -printf '%s\t%p\n' 2>/dev/null | tee -a "$LOG"
