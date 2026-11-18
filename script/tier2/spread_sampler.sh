#!/usr/bin/env bash
# P1-1 spread sampler: build a *deliberately wide* quality range on one design
# so label fidelity (Tier-1 proxy vs Tier-2 post-GRT) can actually be measured.
#
# Why: checkpoint sampling alone gives only 2.9 % spread in gp_hpwl (RL training
# converges into a narrow band), which attenuates any correlation toward zero.
# This script varies three independent knobs instead:
#   - halo            (macro spacing pressure)
#   - core_area_block (placement region restriction)
#   - episodes        (untrained -> converged)
#
# Phase 0 probes each config with a 20-episode run and keeps only the configs
# whose placements pass DREAMPlace's legality check (invalid placements are
# useless downstream).
# Phase 1 trains the surviving configs x lengths x seeds in parallel waves.
# Phase 2 runs the OpenROAD flow once per run (last valid evaluation) and
# extracts the Tier-2 labels.
# Phase 3 prints per-group and overall Spearman.
#
# Output: experiment_records/20261117_p1_1_spread/{labels.csv,run.log,summary.txt}
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH

DESIGN=${DESIGN:-bp_fe}
NICK=${NICK:-bp_fe}
CFGDIR=${CFGDIR:-bp_fe_top}
GPUS=${GPUS:-"3 4 5 6 7"}
SEEDS=${SEEDS:-"3 7"}
LENGTHS=${LENGTHS:-"20 150 600"}
CONFIGS=${CONFIGS:-"halo0_cab0 halo2_cab0 halo0_cab2 halo2_cab2"}

OUT=${OUT:-$PROJECT/experiment_records/20261117_p1_1_spread}
mkdir -p "$OUT"
CSV=$OUT/labels.csv
LOG=$OUT/run.log
if [ ! -f "$CSV" ]; then
  echo "design,seed,halo,core_area_block,episodes,eval_index,tier1_gp_hpwl,tier2_grt_tns,tier2_grt_wns,tier2_drt_tns,tier2_drt_wns,rwl_um,hotspots,drc_violations,power_w,flow_rc,note" > "$CSV"
fi
say() { echo "[spread] $*" | tee -a "$LOG"; }

parse_cfg() {  # halo0_cab2 -> "0 2"
  echo "${1#halo}" | tr '_' ' ' | sed 's/cab//'
}
interval_for() {  # episodes -> evaluation interval giving 1-3 evaluations
  case "$1" in
    20|25|30) echo 20 ;;
    100|150|200) echo 50 ;;
    *) echo 200 ;;
  esac
}

make_cfg() {  # name halo cab episodes
  local name=$1 halo=$2 cab=$3 ep=$4
  sed -e "s/^name :.*/name : ${name}/" \
      -e "s/^episode :.*/episode : ${ep}/" \
      -e "s/^PO_episode :.*/PO_episode : ${ep}/" \
      -e "s/^halo: .*/halo: ${halo}/" \
      -e "s/^core_area_block: .*/core_area_block: ${cab}/" \
      "$PROJECT/config/or_legal.yaml" > "$PROJECT/config/${name}.yaml"
  printf '\nevaluation_interval : %s\nvisualization_interval : 0\nvisualize_prototype : False\n' \
      "$(interval_for "$ep")" >> "$PROJECT/config/${name}.yaml"
}

train_one() {  # config_name halo cab episodes seed gpu
  local name=$1 halo=$2 cab=$3 ep=$4 seed=$5 gpu=$6
  local run=$PROJECT/long_runs/spread_${name}_e${ep}_s${seed}
  # Config files must be unique per run: parallel jobs sharing one filename let a
  # later job's episode count leak into an earlier job (halo0_cab0 trained three
  # times with the same length).
  local cfgname=${name}_e${ep}_s${seed}
  rm -rf "$run"; mkdir -p "$run"
  make_cfg "$cfgname" "$halo" "$cab" "$ep"
  EXPLACE_DUMP_EVAL_DEF="$run/eval_defs" MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1 \
    /usr/bin/python3 -m src.main \
      --benchmark "$DESIGN" --config "$cfgname" --seed "$seed" \
      --log-dir "$run/output" \
      --no-compact-transition --no-worker-rollout --no-mask-cache \
      --no-vectorized-occupancy --no-batch-device-copy --no-mixed-precision \
      --no-async-evaluation --no-policy-mask-fp32 --gpu "$gpu" \
      > "$run/run.log" 2>&1
  say "train ${name} e${ep} s${seed} gpu${gpu} rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

cd "$PROJECT" || exit 1
say "start $(date -u +%Y-%m-%dT%H:%M:%SZ) design=$DESIGN configs='$CONFIGS' lengths='$LENGTHS' seeds='$SEEDS'"

# ---------------- Phase 0: legality probe per config ----------------
VALID=()
if [ "${SKIP_PROBE:-0}" = "1" ]; then
  VALID=($CONFIGS)
  say "skipping legality probe (SKIP_PROBE=1), using: ${VALID[*]}"
else
i=0
for cfg in $CONFIGS; do
  set -- $(parse_cfg "$cfg")
  halo=$1; cab=$2
  run=$PROJECT/long_runs/spread_probe_${cfg}
  rm -rf "$run"; mkdir -p "$run"
  make_cfg "probe_${cfg}" "$halo" "$cab" 20
  EXPLACE_DUMP_EVAL_DEF="$run/eval_defs" MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1 \
    /usr/bin/python3 -m src.main --benchmark "$DESIGN" --config "probe_${cfg}" --seed 3 \
      --log-dir "$run/output" \
      --no-compact-transition --no-worker-rollout --no-mask-cache \
      --no-vectorized-occupancy --no-batch-device-copy --no-mixed-precision \
      --no-async-evaluation --no-policy-mask-fp32 --gpu "$(echo $GPUS | cut -d' ' -f$((i % $(echo "$GPUS" | wc -w) + 1)))" \
      > "$run/run.log" 2>&1
  ok=$(awk -F, 'NR>1 && $2==1' "$run/eval_defs/manifest.csv" 2>/dev/null | wc -l)
  gp=$(awk -F, 'NR>1 && $2==1 {print $3}' "$run/eval_defs/manifest.csv" 2>/dev/null | head -1)
  say "probe ${cfg}: valid_evals=$ok first_gp_hpwl=${gp:-NA}"
  [ "${ok:-0}" -ge 1 ] && VALID+=("$cfg")
  i=$((i+1))
done
fi
[ ${#VALID[@]} -eq 0 ] && { say "no config passed legality; aborting"; exit 1; }
say "valid configs: ${VALID[*]}"

# ---------------- Phase 1: trainings (paced by GPU count) ----------------
NPROC=$(echo "$GPUS" | wc -w)
idx=0
for cfg in "${VALID[@]}"; do
  set -- $(parse_cfg "$cfg"); halo=$1; cab=$2
  for ep in $LENGTHS; do
    for seed in $SEEDS; do
      if grep -q "^${DESIGN},${seed},${halo},${cab},${ep}," "$CSV"; then
        say "train ${cfg} e${ep} s${seed}: already labelled, skipping"
        continue
      fi
      if [ -f "$PROJECT/long_runs/spread_${cfg}_e${ep}_s${seed}/eval_defs/manifest.csv" ]; then
        say "train ${cfg} e${ep} s${seed}: run dir exists, reusing its dumps"
        continue
      fi
      gpu=$(echo "$GPUS" | cut -d' ' -f$((idx % NPROC + 1)))
      train_one "$cfg" "$halo" "$cab" "$ep" "$seed" "$gpu" &
      idx=$((idx+1))
      while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 15; done
    done
  done
done
wait
say "phase 1 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------- Phase 2: one flow per run ----------------
cd "$FLOW" || exit 1
CFG_MK=designs/nangate45/${CFGDIR}/config_EXPlace.mk
RES=$FLOW/results/nangate45/${NICK}/EXPlace
REP=$FLOW/reports/nangate45/${NICK}/EXPlace

for cfg in "${VALID[@]}"; do
  set -- $(parse_cfg "$cfg"); halo=$1; cab=$2
  for ep in $LENGTHS; do
    for seed in $SEEDS; do
      run=$PROJECT/long_runs/spread_${cfg}_e${ep}_s${seed}
      man=$run/eval_defs/manifest.csv
      [ -f "$man" ] || { say "SKIP ${cfg} e${ep} s${seed}: no manifest"; continue; }
      # last valid evaluation of this run
      read -r idxn gp < <(awk -F, 'NR>1 && $2==1 {i=$1; g=$3} END {print i, g}' "$man")
      if [ -z "${idxn:-}" ]; then say "SKIP ${cfg} e${ep} s${seed}: no valid eval"; continue; fi
      if grep -q "^${DESIGN},${seed},${halo},${cab},${ep}," "$CSV"; then
        say "SKIP ${cfg} e${ep} s${seed}: already labelled"; continue
      fi
      def=$(printf '%s/eval_defs/eval_%03d.def' "$run" "$idxn")
      [ -f "$def" ] || { say "SKIP ${cfg} e${ep} s${seed}: missing $def"; continue; }

      # stale-report guard: a failed stage must not inherit the previous sample's numbers
      rm -f "$REP/5_global_route.rpt" "$REP/6_finish.rpt" "$REP/congestion.rpt" "$REP/5_route_drc.rpt" "$RES/6_final.def"
      cp -f "$def" "EXPlace_defs/${NICK}.gp.def"
      for f in 1_synth.v 1_synth.sdc; do [ -f "$RES/$f" ] && cp -f "$RES/$f" "/tmp/k_$f"; done
      rm -rf "$RES"; mkdir -p "$RES"
      for f in 1_synth.v 1_synth.sdc; do [ -f "/tmp/k_$f" ] && cp -f "/tmp/k_$f" "$RES/$f"; done

      slog=$OUT/flow_${cfg}_e${ep}_s${seed}.log
      say "flow ${cfg} e${ep} s${seed} eval=${idxn} gp_hpwl=${gp} start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      make run_EXPlace DESIGN_CONFIG="$CFG_MK" >> "$slog" 2>&1 || { say "stage A failed"; continue; }
      make run_wo_synth DESIGN_CONFIG="$CFG_MK" \
           ROUTING_LAYER_ADJUSTMENT=0.13 \
           GLOBAL_ROUTE_ARGS="-congestion_iterations 30 -congestion_report_iter_step 5 -verbose" \
           >> "$slog" 2>&1
      frc=$?
      row=$(/usr/bin/python3 "$PROJECT/script/tier2/extract_labels.py" \
              --results "$RES" --reports "$REP" 2>>"$LOG")
      grt=$(echo "$row" | cut -d, -f1)
      drt=$(echo "$row" | cut -d, -f3)
      if [ -z "$grt" ]; then
        note="grt_failed"
      elif [ -z "$drt" ] || [ "$frc" != "0" ]; then
        # the flow can die in 6_report (PDN/IR connectivity, PSM-0069) *after*
        # GRT; post-GRT labels stay valid, post-DRT ones are left empty
        note="grt_only_rc${frc}"
      else
        note="ok"
      fi
      echo "${DESIGN},${seed},${halo},${cab},${ep},${idxn},${gp},${row},${frc},${note}" >> "$CSV"
      say "flow ${cfg} e${ep} s${seed} rc=$frc note=${note} labels=${row}"
    done
  done
done
say "phase 2 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------- Phase 3: correlation summary ----------------
/usr/bin/python3 - "$CSV" "$OUT/summary.txt" <<'PY'
import csv, sys

def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals); i = 0
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
    n = len(a)
    if n < 3:
        return float('nan')
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float('nan')

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get('tier2_grt_tns')]
out = [f'samples: {len(rows)}']
if rows:
    xs = [float(r['tier1_gp_hpwl']) for r in rows]
    out.append(f"gp_hpwl span: {min(xs):.0f} .. {max(xs):.0f} "
               f"({(max(xs)-min(xs))/min(xs)*100:.1f}%)")
    for col in ('tier2_grt_tns', 'tier2_grt_wns', 'tier2_drt_tns', 'tier2_drt_wns',
                'rwl_um', 'drc_violations', 'hotspots', 'power_w'):
        try:
            ys = [float(r[col]) for r in rows]
        except (KeyError, ValueError):
            continue
        out.append(f"spearman(gp_hpwl, {col}) = {spearman(xs, ys):+.3f}")
    # per (halo, cab) group, if enough samples
    groups = {}
    for r in rows:
        groups.setdefault((r['halo'], r['core_area_block']), []).append(r)
    for key, grp in sorted(groups.items()):
        if len(grp) < 3:
            continue
        gx = [float(r['tier1_gp_hpwl']) for r in grp]
        gy = [float(r['tier2_grt_tns']) for r in grp]
        out.append(f"  group halo={key[0]} cab={key[1]} n={len(grp)}: "
                   f"spearman(gp_hpwl, grt_tns) = {spearman(gx, gy):+.3f}")
    out.append("reminder: report per-circuit rho; aggregate across circuits with Fisher z")
text = '\n'.join(out)
open(sys.argv[2], 'w').write(text + '\n')
print(text)
PY
