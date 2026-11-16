#!/usr/bin/env bash
# P1-1 batch: paired labels for several placements of one design.
#
# Tier-1 here is the **DREAMPlace proxy** (gp_hpwl), not OpenTimer timing:
# enabling the timing stage on the or_cases designs aborts with
#   timing_cpp.cpp:162 Assertion `net_iter != timer.nets().end()` failed
# because DREAMPlace's net names do not match OpenTimer's netlist.  That is a
# hard limitation of this setup, documented in TODO.md.
#
# Tier-2 is the OpenROAD flow: post-GRT and post-DRT TNS/WNS.
#
# Output: experiment_records/20261117_p1_1_labels/labels.csv
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH

DESIGN=${DESIGN:-bp_fe}
NICK=${NICK:-bp_fe}
CFGDIR=${CFGDIR:-bp_fe_top}
SEEDS=${SEEDS:-"3 7 11 17 23"}
GPUS=${GPUS:-"3 4 5 6 7"}
EPISODES=${EPISODES:-600}
INTERVAL=${INTERVAL:-100}

OUT=$PROJECT/experiment_records/20261117_p1_1_labels
mkdir -p "$OUT"
CSV=$OUT/labels.csv
LOG=$OUT/run.log
if [ ! -f "$CSV" ]; then
  echo "design,seed,episodes,def_bytes,tier1_proxy_gp_hpwl,tier2_grt_tns,tier2_grt_wns,tier2_drt_tns,tier2_drt_wns,flow_rc,notes" > "$CSV"
fi
say() { echo "[p11] $*" | tee -a "$LOG"; }

say "start $(date -u +%Y-%m-%dT%H:%M:%SZ) design=$DESIGN seeds='$SEEDS' episodes=$EPISODES"

cd "$PROJECT" || exit 1
# timer stays OFF (enabling it aborts on or_cases designs)
sed -e "s/^name :.*/name : or_batch/" \
    -e "s/^episode :.*/episode : ${EPISODES}/" \
    -e "s/^PO_episode :.*/PO_episode : ${EPISODES}/" \
    config/or_legal.yaml > config/or_batch.yaml
printf '\nevaluation_interval : %s\nvisualization_interval : 0\nvisualize_prototype : False\n' "$INTERVAL" >> config/or_batch.yaml

# ---------------- Phase 1: placements (one GPU each, parallel) ----------------
run_explace() {
  local seed=$1 gpu=$2 run=$PROJECT/long_runs/p11_${DESIGN}_s${seed}
  rm -rf "$run"; mkdir -p "$run"
  MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1 /usr/bin/python3 -m src.main \
      --benchmark "$DESIGN" --config or_batch --seed "$seed" \
      --log-dir "$run/output" \
      --no-compact-transition --no-worker-rollout --no-mask-cache \
      --no-vectorized-occupancy --no-batch-device-copy --no-mixed-precision \
      --no-async-evaluation --no-policy-mask-fp32 --gpu "$gpu" \
      > "$run/run.log" 2>&1
  echo "[p11] train seed=$seed gpu=$gpu rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
}

i=0
PIDS=()
for seed in $SEEDS; do
  gpu=$(echo "$GPUS" | cut -d' ' -f$((i % $(echo "$GPUS" | wc -w) + 1)))
  run_explace "$seed" "$gpu" &
  PIDS+=($!)
  i=$((i+1))
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
say "phase 1 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------- Phase 2: OpenROAD flow (sequential: shared staging dirs) ----
cd "$FLOW" || exit 1
CFG=designs/nangate45/${CFGDIR}/config_EXPlace.mk
RES=$FLOW/results/nangate45/${NICK}/EXPlace
REP=$FLOW/reports/nangate45/${NICK}/EXPlace

for seed in $SEEDS; do
  run=$PROJECT/long_runs/p11_${DESIGN}_s${seed}
  def=$(find "$run" -name 'best_gp_hpwl.def' | head -1)
  if [ -z "$def" ]; then say "seed=$seed: no DEF, skipping"; continue; fi

  gp=$(/usr/bin/python3 - "$run" <<'PY'
import csv, glob, os, sys
rows = []
for f in glob.glob(os.path.join(sys.argv[1], 'output/rl_logs/*/*/physical_evaluations.csv')):
    rows += [r for r in csv.DictReader(open(f)) if r['valid'] == '1']
print(min(float(r['gp_hpwl']) for r in rows) if rows else '')
PY
)
  [ -z "$gp" ] && { say "seed=$seed: no valid Tier-1 rows"; continue; }

  cp -f "$def" "EXPlace_defs/${NICK}.gp.def"
  for f in 1_synth.v 1_synth.sdc; do [ -f "$RES/$f" ] && cp -f "$RES/$f" "/tmp/k_$f"; done
  rm -rf "$RES"; mkdir -p "$RES"
  for f in 1_synth.v 1_synth.sdc; do [ -f "/tmp/k_$f" ] && cp -f "/tmp/k_$f" "$RES/$f"; done

  slog=$OUT/flow_seed_${seed}.log
  say "seed=$seed: flow start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  make run_EXPlace DESIGN_CONFIG="$CFG" >> "$slog" 2>&1 || { say "seed=$seed run_EXPlace failed"; continue; }
  make run_wo_synth DESIGN_CONFIG="$CFG" \
       ROUTING_LAYER_ADJUSTMENT=0.13 \
       GLOBAL_ROUTE_ARGS="-congestion_iterations 30 -congestion_report_iter_step 5 -verbose" \
       >> "$slog" 2>&1
  frc=$?
  g_tns=$(grep -m1 "^tns max" "$REP/5_global_route.rpt" 2>/dev/null | awk '{print $3}')
  g_wns=$(grep -m1 "^wns max" "$REP/5_global_route.rpt" 2>/dev/null | awk '{print $3}')
  d_tns=$(grep -m1 "^tns max" "$REP/6_finish.rpt" 2>/dev/null | awk '{print $3}')
  d_wns=$(grep -m1 "^wns max" "$REP/6_finish.rpt" 2>/dev/null | awk '{print $3}')
  echo "${DESIGN},${seed},${EPISODES},$(stat -c%s "$def"),${gp},${g_tns:-},${g_wns:-},${d_tns:-},${d_wns:-},${frc}," >> "$CSV"
  say "seed=$seed: flow rc=$frc grt(tns=${g_tns:-NA} wns=${g_wns:-NA}) drt(tns=${d_tns:-NA} wns=${d_wns:-NA})"
done

say "batch done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------- summary: Spearman (manual, scipy may be absent) -------------
/usr/bin/python3 - "$CSV" "$OUT/summary.txt" <<'PY'
import csv, sys

def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r

def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 3:
        return float('nan')
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float('nan')

rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r.get('tier2_grt_tns') and r.get('tier1_proxy_gp_hpwl')]
out = []
out.append(f"samples: {len(rows)}")
for col in ('tier2_grt_tns', 'tier2_grt_wns', 'tier2_drt_tns', 'tier2_drt_wns'):
    try:
        a = [float(r['tier1_proxy_gp_hpwl']) for r in rows]
        b = [float(r[col]) for r in rows]
    except (KeyError, ValueError):
        continue
    out.append(f"spearman(tier1_gp_hpwl, {col}) = {spearman(a, b):+.3f}")
out.append("n<10: treat as pipeline validation only, not a P1-1 decision")
text = "\n".join(out)
open(sys.argv[2], 'w').write(text + "\n")
print(text)
PY
tail -30 "$LOG"
