#!/usr/bin/env bash
# Feed the 600-episode bp_fe placement and relax the layer derating that GRT
# itself suggested (25% -> 13%), with more congestion iterations.
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH
cd "$FLOW" || exit 1

DESIGN=bp_fe
CFGDIR=bp_fe_top
DEF=$(find "$PROJECT/long_runs/or_legal_bp_fe_r2" -name 'best_gp_hpwl.def' | head -1)
if [ -z "$DEF" ]; then
  echo "[run] no r2 DEF yet; aborting"
  exit 1
fi
echo "[run] using DEF: $DEF ($(stat -c%s "$DEF") bytes)"

mkdir -p EXPlace_defs
cp -f "$DEF" "EXPlace_defs/${DESIGN}.gp.def"

CFG=designs/nangate45/${CFGDIR}/config_EXPlace.mk
LOG=$FLOW/tier2_bp_fe_r2_grt.log
RES=results/nangate45/${DESIGN}/EXPlace

# rebuild from floorplan with the new macro placement
for f in 1_synth.v 1_synth.sdc; do
  [ -f "$RES/$f" ] && cp -f "$RES/$f" "/tmp/${f}.keep"
done
rm -rf "$RES"; mkdir -p "$RES"
for f in 1_synth.v 1_synth.sdc; do
  [ -f "/tmp/${f}.keep" ] && cp -f "/tmp/${f}.keep" "$RES/$f"
done

echo "[run] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG"
make run_EXPlace DESIGN_CONFIG="$CFG" >> "$LOG" 2>&1 || { echo "[run] run_EXPlace failed"; tail -15 "$LOG"; exit 1; }

make run_wo_synth DESIGN_CONFIG="$CFG" \
     ROUTING_LAYER_ADJUSTMENT=0.13 \
     GLOBAL_ROUTE_ARGS="-congestion_iterations 30 -congestion_report_iter_step 5 -verbose" \
     >> "$LOG" 2>&1
rc=$?
echo "[run] run_wo_synth rc=$rc end $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
grep -nE "Running .*stage|GRT-0115|EST-0005|Error:" "$LOG" | tail -12 | tee -a "$LOG"
echo "[run] artifacts:" | tee -a "$LOG"
ls -1 "$RES" | grep -E "5_|6_|final" | tee -a "$LOG"
