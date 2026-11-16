#!/usr/bin/env bash
# After the CORE_AREA inset, GRT-0041 is gone and global_route succeeds; the
# failure moved to `estimate_parasitics -global_routing` (EST-0005) right after
# "Repair antennas...". Test whether antenna repair is what invalidates the
# routing database.
set -uo pipefail
FLOW=/data0/cjy_cxsj/EXPlace-test/EXPlace/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH
cd "$FLOW" || exit 1

VAR=${1:-norepairant}
LOG=$FLOW/tier2_bp_fe_${VAR}.log
CFG=designs/nangate45/bp_fe_top/config_EXPlace.mk

echo "[probe] start $(date -u +%Y-%m-%dT%H:%M:%SZ) variant=$VAR" | tee "$LOG"
make run_wo_synth DESIGN_CONFIG="$CFG" SKIP_ANTENNA_REPAIR=1 >> "$LOG" 2>&1
rc=$?
echo "[probe] rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
grep -nE "Running .*stage|GRT-0115|EST-0005|Error:" "$LOG" | tail -12 | tee -a "$LOG"
ls -1 results/nangate45/bp_fe/EXPlace | grep -E "5_|finish|final" | tee -a "$LOG"
