#!/usr/bin/env bash
# Stratified P1-1 sampler: one EXPlace run per seed, but *every* periodic
# evaluation is dumped as a DEF, so the Tier-2 labels cover the whole quality
# range of the run (not just its best checkpoint). This is the fix for the
# 1.6 % gp_hpwl span observed in the pilot batch.
#
#   usage: batch_stratified.sh [seed] [indices]
#     indices: space separated eval indices to label (default: first, middle, last)
#
# Output: experiment_records/20261117_p1_1_stratified/labels.csv
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH

DESIGN=${DESIGN:-bp_fe}
NICK=${NICK:-bp_fe}
CFGDIR=${CFGDIR:-bp_fe_top}
GPU=${GPU:-3}
EPISODES=${EPISODES:-600}
INTERVAL=${INTERVAL:-100}
SEED=${1:-3}
INDICES=${INDICES:-}

OUT=$PROJECT/experiment_records/20261117_p1_1_stratified
mkdir -p "$OUT"
CSV=$OUT/labels.csv
LOG=$OUT/run.log
if [ ! -f "$CSV" ]; then
  echo "design,seed,eval_index,tier1_gp_hpwl,tier2_grt_tns,tier2_grt_wns,tier2_drt_tns,tier2_drt_wns,rwl_um,hotspots,drc_violations,power_w,flow_rc" > "$CSV"
fi
say() { echo "[strat] $*" | tee -a "$LOG"; }

RUN=$PROJECT/long_runs/strat_${DESIGN}_s${SEED}
DUMP=$RUN/eval_defs
say "start $(date -u +%Y-%m-%dT%H:%M:%SZ) design=$DESIGN seed=$SEED episodes=$EPISODES interval=$INTERVAL"

# ---------- phase 1: one training run, DEF per evaluation ----------
if [ ! -f "$DUMP/manifest.csv" ]; then
  rm -rf "$RUN"; mkdir -p "$RUN"
  cd "$PROJECT" || exit 1
  sed -e "s/^name :.*/name : or_strat/" \
      -e "s/^episode :.*/episode : ${EPISODES}/" \
      -e "s/^PO_episode :.*/PO_episode : ${EPISODES}/" \
      config/or_legal.yaml > config/or_strat.yaml
  printf '\nevaluation_interval : %s\nvisualization_interval : 0\nvisualize_prototype : False\n' "$INTERVAL" >> config/or_strat.yaml
  EXPLACE_DUMP_EVAL_DEF="$DUMP" MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1 \
    /usr/bin/python3 -m src.main \
      --benchmark "$DESIGN" --config or_strat --seed "$SEED" \
      --log-dir "$RUN/output" \
      --no-compact-transition --no-worker-rollout --no-mask-cache \
      --no-vectorized-occupancy --no-batch-device-copy --no-mixed-precision \
      --no-async-evaluation --no-policy-mask-fp32 --gpu "$GPU" \
      > "$RUN/run.log" 2>&1
  say "phase 1 rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  say "phase 1 already present, reusing $DUMP"
fi

say "manifest:"
cat "$DUMP/manifest.csv" | tee -a "$LOG"

# ---------- pick indices ----------
if [ -z "$INDICES" ]; then
  n=$(grep -vc '^index' "$DUMP/manifest.csv")
  # quartile sweep by default: cheap and spans early/mid/late checkpoints
  INDICES=$(/usr/bin/python3 - "$n" <<'PYEOF'
import sys
n = int(sys.argv[1])
print(' '.join(str(i) for i in sorted({0, n // 3, (2 * n) // 3, n - 1})))
PYEOF
)
fi
say "labelling eval indices: $INDICES"

# ---------- phase 2: flow per dumped DEF ----------
cd "$FLOW" || exit 1
CFG=designs/nangate45/${CFGDIR}/config_EXPlace.mk
RES=$FLOW/results/nangate45/${NICK}/EXPlace
REP=$FLOW/reports/nangate45/${NICK}/EXPlace

for idx in $INDICES; do
  if grep -q "^${DESIGN},${SEED},${idx}," "$CSV"; then
    say "idx=$idx: already labelled, skipping"
    continue
  fi
  def=$(printf '%s/eval_%03d.def' "$DUMP" "$idx")
  [ -f "$def" ] || { say "idx=$idx: missing $def"; continue; }
  gp=$(awk -F, -v i="$idx" 'NR>1 && $1==i {print $3}' "$DUMP/manifest.csv")
  valid=$(awk -F, -v i="$idx" 'NR>1 && $1==i {print $2}' "$DUMP/manifest.csv")
  if [ "$valid" != "1" ]; then say "idx=$idx: invalid placement, skipping"; continue; fi

  cp -f "$def" "EXPlace_defs/${NICK}.gp.def"
  for f in 1_synth.v 1_synth.sdc; do [ -f "$RES/$f" ] && cp -f "$RES/$f" "/tmp/k_$f"; done
  rm -rf "$RES"; mkdir -p "$RES"
  for f in 1_synth.v 1_synth.sdc; do [ -f "/tmp/k_$f" ] && cp -f "/tmp/k_$f" "$RES/$f"; done

  slog=$OUT/flow_s${SEED}_i${idx}.log
  say "idx=$idx: flow start $(date -u +%Y-%m-%dT%H:%M:%SZ) (gp_hpwl=$gp)"
  make run_EXPlace DESIGN_CONFIG="$CFG" >> "$slog" 2>&1 || { say "idx=$idx stage A failed"; continue; }
  make run_wo_synth DESIGN_CONFIG="$CFG" \
       ROUTING_LAYER_ADJUSTMENT=0.13 \
       GLOBAL_ROUTE_ARGS="-congestion_iterations 30 -congestion_report_iter_step 5 -verbose" \
       >> "$slog" 2>&1
  frc=$?
  row=$(/usr/bin/python3 "$PROJECT/script/tier2/extract_labels.py" \
          --results "$RES" --reports "$REP" 2>>"$LOG")
  echo "${DESIGN},${SEED},${idx},${gp},${row},${frc}" >> "$CSV"
  say "idx=$idx: rc=$frc labels=${row}"
done

say "done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "$CSV" | tee -a "$LOG"
