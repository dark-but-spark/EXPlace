#!/usr/bin/env bash
# Solution 1 for the GRT-0041 failure: inset CORE_AREA for bp_fe_top so PDN
# rails/straps cannot land outside the die (ariane133 already has a 5um inset).
#
#   DIE_AREA  = 0 0 800 600
#   CORE_AREA = 0 0 790 590   ->   5 5 795 595
#
# Then rebuild the flow from floorplan (CORE_AREA feeds 2_1_floorplan), keeping
# only the pre-existing synth outputs.
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH
cd "$FLOW" || exit 1

CFG=designs/nangate45/bp_fe_top/config_EXPlace.mk
echo "[fix] before:"; grep -nE "^export (DIE|CORE)_AREA" "$CFG"
sed -i 's|^export CORE_AREA   = 0 0 790 590 *$|export CORE_AREA   = 5 5 795 595|' "$CFG"
echo "[fix] after:";  grep -nE "^export (DIE|CORE)_AREA" "$CFG"

RES=results/nangate45/bp_fe/EXPlace
echo
echo "[fix] clearing derived flow products (keeping synth outputs)"
mkdir -p "$RES.keep"
for f in 1_synth.v 1_synth.sdc; do
  [ -f "$RES/$f" ] && cp -f "$RES/$f" "$RES.keep/$f"
done
rm -rf "$RES"
mkdir -p "$RES"
for f in 1_synth.v 1_synth.sdc; do
  [ -f "$RES.keep/$f" ] && cp -f "$RES.keep/$f" "$RES/$f"
done
rm -rf "$RES.keep"
ls -la "$RES"

LOG=$FLOW/tier2_bp_fe_inset.log
echo
echo "[fix] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG"

echo "[fix] step A: run_EXPlace" | tee -a "$LOG"
make run_EXPlace DESIGN_CONFIG="$CFG" >> "$LOG" 2>&1
rc=$?
echo "[fix] run_EXPlace rc=$rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { tail -20 "$LOG"; exit $rc; }

echo "[fix] step B: run_wo_synth $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
make run_wo_synth DESIGN_CONFIG="$CFG" >> "$LOG" 2>&1
rc=$?
echo "[fix] run_wo_synth rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

echo "[fix] stages:" | tee -a "$LOG"
grep -E "Running .*stage|GRT-0041|PDN-01|Error:" "$LOG" | tail -14 | tee -a "$LOG"
echo "[fix] artifacts:" | tee -a "$LOG"
ls -1 "$RES" | tail -12 | tee -a "$LOG"
