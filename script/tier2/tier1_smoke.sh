#!/usr/bin/env bash
# P1-1 needs Tier-1 *timing* on the same placement, so create an or_cases config
# that is identical to or_legal.yaml but leaves the OpenTimer stage enabled.
# Quick 20-episode check first: do TNS/WNS come back non-zero?
set -uo pipefail
cd /data0/cjy_cxsj/EXPlace-test/EXPlace || exit 1

sed -e 's/^use_timer_for_evaluation : False/use_timer_for_evaluation : True/' \
    -e 's/^name :.*/name : or_tier1/' \
    config/or_legal.yaml > config/or_tier1.yaml
printf '\nevaluation_interval : 20\n' >> config/or_tier1.yaml

echo "[tier1] config:"
grep -nE "^(name|episode|use_timer_for_evaluation|evaluation_interval|use_dmp)" config/or_tier1.yaml

RUN=long_runs/p11_smoke_bp_fe
rm -rf "$RUN"; mkdir -p "$RUN"
LOG=$RUN/run.log
echo "[tier1] smoke start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG"

MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1 /usr/bin/python3 -m src.main \
    --benchmark bp_fe --config or_tier1 --seed 3 \
    --log-dir "$RUN/output" \
    --no-compact-transition --no-worker-rollout --no-mask-cache \
    --no-vectorized-occupancy --no-batch-device-copy --no-mixed-precision \
    --no-async-evaluation --no-policy-mask-fp32 --gpu 3 >> "$LOG" 2>&1
rc=$?
echo "[tier1] rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

echo "[tier1] physical evaluations (tns/wns must be non-zero):" | tee -a "$LOG"
find "$RUN" -name physical_evaluations.csv -exec cat {} \; | tee -a "$LOG"
echo "[tier1] DEF:" | tee -a "$LOG"
find "$RUN" -name '*.def' -printf '%s\t%p\n' | tee -a "$LOG"
