#!/usr/bin/env bash
# Probe the GRT stage directly: does it fail because RESULTS_DIR is missing in
# the stage environment, or because global_route itself fails?
set -uo pipefail
FLOW=/data0/cjy_cxsj/EXPlace-test/EXPlace/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH
cd "$FLOW" || exit 1

LOG=$FLOW/tier2_bp_fe_grt_probe.log
echo "[probe] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG"

make grt \
    DESIGN_CONFIG=designs/nangate45/bp_fe_top/config_EXPlace.mk \
    >> "$LOG" 2>&1
rc=$?
echo "[probe] rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
tail -25 "$LOG" | tee -a "$LOG"
