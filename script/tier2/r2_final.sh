#!/usr/bin/env bash
# GRT now completes and produces timing metrics, but the stage dies writing its
# outputs because $::env(RESULTS_DIR) is missing inside the stage. Export it
# explicitly (make runs the stage in a sub-shell) and re-run just route+finish.
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH
cd "$FLOW" || exit 1

DESIGN=bp_fe
CFGDIR=bp_fe_top
CFG=designs/nangate45/${CFGDIR}/config_EXPlace.mk
RES=$FLOW/results/nangate45/${DESIGN}/EXPlace
export RESULTS_DIR="$RES"
export REPORTS_DIR="$FLOW/reports/nangate45/${DESIGN}/EXPlace"
export LOG_DIR="$FLOW/logs/nangate45/${DESIGN}/EXPlace"
export OBJECTS_DIR="$FLOW/objects/nangate45/${DESIGN}/EXPlace"
mkdir -p "$RESULTS_DIR" "$REPORTS_DIR" "$LOG_DIR" "$OBJECTS_DIR"

LOG=$FLOW/tier2_bp_fe_r2_final.log
echo "[final] RESULTS_DIR=$RESULTS_DIR" | tee "$LOG"
echo "[final] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

make run_wo_synth DESIGN_CONFIG="$CFG" \
     ROUTING_LAYER_ADJUSTMENT=0.13 \
     GLOBAL_ROUTE_ARGS="-congestion_iterations 30 -congestion_report_iter_step 5 -verbose" \
     >> "$LOG" 2>&1
rc=$?
echo "[final] rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

echo "[final] stages:" | tee -a "$LOG"
grep -E "Running .*stage|Error:|GRT-0115" "$LOG" | tail -10 | tee -a "$LOG"
echo "[final] timing lines:" | tee -a "$LOG"
grep -E "FLW-000[789]|slack|wns|tns" "$LOG" | tail -10 | tee -a "$LOG"
echo "[final] artifacts:" | tee -a "$LOG"
ls -1 "$RES" | grep -E "5_|6_|final|route.guide" | tee -a "$LOG"
