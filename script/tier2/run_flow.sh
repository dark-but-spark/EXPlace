#!/usr/bin/env bash
# P1-1 Tier-2: feed a real EXPlace placement DEF into the OpenROAD flow.
#   usage: tier2_run.sh <design_nickname> <config_design_dir> <source_def>
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH          # flow scripts need PyYAML; conda python3 lacks it
cd "$FLOW" || exit 1

NICK=${1:?nickname, e.g. bp_fe}
CFGDIR=${2:?config dir under designs/nangate45, e.g. bp_fe_top}
SRCDEF=${3:?source DEF}

LOG=$FLOW/tier2_${NICK}.log
mkdir -p EXPlace_defs
cp -v "$SRCDEF" "EXPlace_defs/${NICK}.gp.def"
echo "[tier2] staged ${NICK}.gp.def ($(stat -c%s "EXPlace_defs/${NICK}.gp.def") bytes)" | tee "$LOG"

echo "[tier2] step A: run_EXPlace $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
make run_EXPlace DESIGN_CONFIG="designs/nangate45/${CFGDIR}/config_EXPlace.mk" >> "$LOG" 2>&1
rc=$?
echo "[tier2] run_EXPlace rc=$rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "[tier2] step A failed"; tail -15 "$LOG"; exit $rc; }

echo "[tier2] macro_out: $(wc -l < "results/nangate45/${NICK}/EXPlace/macro_out" 2>/dev/null) lines" | tee -a "$LOG"

echo "[tier2] step B: run_wo_synth $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
make run_wo_synth DESIGN_CONFIG="designs/nangate45/${CFGDIR}/config_EXPlace.mk" >> "$LOG" 2>&1
rc=$?
echo "[tier2] run_wo_synth rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

echo "[tier2] stage results:" | tee -a "$LOG"
grep -E "Running .*stage|PDN-0179|PDN-0113|Error:" "$LOG" | tail -10 | tee -a "$LOG"
echo "[tier2] artifacts:" | tee -a "$LOG"
ls -la "results/nangate45/${NICK}/EXPlace/" 2>/dev/null | tail -10 | tee -a "$LOG"
