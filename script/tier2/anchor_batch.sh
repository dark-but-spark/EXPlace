#!/usr/bin/env bash
# Anchor batch: perturb one legal macro placement and run the flow per variant.
# No training involved; each sample costs only a flow run (~9 min).
#
# Uses a dedicated FLOW_VARIANT per anchor, so it never touches the results dir
# or the staged EXPlace_defs DEF of the sampling batches and can run alongside
# them (CPU permitting).
#
#   usage: anchor_batch.sh [base_def] [variants]
set -uo pipefail
PROJECT=/data0/cjy_cxsj/EXPlace-test/EXPlace
FLOW=$PROJECT/OpenROAD-flow
export OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad
export PATH=/usr/bin:$PATH

NICK=bp_fe
CFGDIR=bp_fe_top
BASE=${1:-}
if [ -z "${BASE}" ] || [ ! -f "${BASE}" ]; then
  # discover instead of hard-coding a timestamped run directory
  BASE=$(find "$PROJECT/long_runs/spread_halo0_cab0_e600_s3" -name best_gp_hpwl.def 2>/dev/null | head -1)
fi
[ -n "${BASE}" ] && [ -f "${BASE}" ] || { echo "[anchor] no base DEF found"; exit 1; }
VARIANTS=${2:-"jitter5 jitter20 shuffle"}

OUT=$PROJECT/experiment_records/20261118_p1_1_anchors
mkdir -p "$OUT"
CSV=$OUT/labels.csv
LOG=$OUT/run.log
if [ ! -f "$CSV" ]; then
  echo "anchor,flow_variant,tier1_gp_hpwl,tier2_grt_tns,tier2_grt_wns,tier2_drt_tns,tier2_drt_wns,rwl_um,hotspots,drc_violations,power_w,flow_rc,note" > "$CSV"
fi
say() { echo "[anchor] $*" | tee -a "$LOG"; }

cd "$FLOW" || exit 1
CFG=designs/nangate45/${CFGDIR}/config_EXPlace.mk

# 1) base macro_out from a known-good placement
BK=$OUT/base
mkdir -p "$BK"
if [ ! -f "$BK/macro_out" ]; then
  say "generating base macro_out from $(basename "$(dirname "$(dirname "$BASE")")")"
  # use a private DEF name and FLOW_VARIANT: the sampling batches own
  # EXPlace_defs/<nick>.gp.def and results/.../EXPlace/macro_out, and overwriting
  # either mid-run would corrupt their floorplan stage.
  cp -f $BASE "EXPlace_defs/anchor_base.gp.def"
  mkdir -p "results/nangate45/${NICK}/ANCHOR_base"
  make run_EXPlace DESIGN_CONFIG="$CFG" FLOW_VARIANT=ANCHOR_base        MACRO_DEF=./EXPlace_defs/anchor_base >> "$LOG" 2>&1 || { say "base run_EXPlace failed"; exit 1; }
  cp -f "results/nangate45/${NICK}/ANCHOR_base/macro_out" "$BK/macro_out"
fi
say "base macro_out: $(wc -l < "$BK/macro_out") lines"

# 2) perturbed TCLs
/usr/bin/python3 "$PROJECT/script/tier2/make_anchors.py" \
    --macro-out "$BK/macro_out" --out-dir "$BK/variants" --core 5,5,795,595 \
    --jitters 5,20,50 >> "$LOG" 2>&1 || { say "make_anchors failed"; exit 1; }
cat "$BK/variants/manifest.csv" | tee -a "$LOG"

# tier-1 for every anchor is the same base placement
gp=$(/usr/bin/python3 - "$PROJECT/long_runs/spread_halo0_cab0_e600_s3" <<'PY'
import csv, glob, os, sys
rows = []
for f in glob.glob(os.path.join(sys.argv[1], 'output/rl_logs/*/*/physical_evaluations.csv')):
    rows += [r for r in csv.DictReader(open(f)) if r['valid'] == '1']
print(min(float(r['gp_hpwl']) for r in rows) if rows else '')
PY
)
say "tier1 gp_hpwl of base placement: ${gp:-NA}"

# 3) one flow per anchor, isolated by FLOW_VARIANT
for v in $VARIANTS; do
  tcl=$BK/variants/macro_out_${v#jitter}.tcl
  [ "$v" = "jitter5" ] && tcl=$BK/variants/macro_out_j5.tcl
  [ "$v" = "jitter20" ] && tcl=$BK/variants/macro_out_j20.tcl
  [ "$v" = "jitter50" ] && tcl=$BK/variants/macro_out_j50.tcl
  [ "$v" = "shuffle" ] && tcl=$BK/variants/macro_out_shuffle.tcl
  [ -f "$tcl" ] || { say "SKIP $v: missing $tcl"; continue; }
  if grep -q "^${v}," "$CSV"; then say "SKIP $v: already labelled"; continue; fi

  FV="ANCHOR_$v"
  RES=$FLOW/results/nangate45/${NICK}/$FV
  REP=$FLOW/reports/nangate45/${NICK}/$FV
  mkdir -p "$RES"
  for f in 1_synth.v 1_synth.sdc; do
    [ -f "$FLOW/results/nangate45/${NICK}/EXPlace/$f" ] && cp -f "$FLOW/results/nangate45/${NICK}/EXPlace/$f" "$RES/$f"
  done
  rm -f "$REP/5_global_route.rpt" "$REP/6_finish.rpt" "$REP/congestion.rpt" "$REP/5_route_drc.rpt" "$RES/6_final.def"

  slog=$OUT/flow_${v}.log
  say "$v: flow start $(date -u +%Y-%m-%dT%H:%M:%SZ) variant=$FV"
  make run_wo_synth DESIGN_CONFIG="$CFG" FLOW_VARIANT="$FV" \
       MACRO_PLACEMENT_TCL="$tcl" \
       ROUTING_LAYER_ADJUSTMENT=0.13 \
       GLOBAL_ROUTE_ARGS="-congestion_iterations 30 -congestion_report_iter_step 5 -verbose" \
       >> "$slog" 2>&1
  rc=$?
  row=$(/usr/bin/python3 "$PROJECT/script/tier2/extract_labels.py" --results "$RES" --reports "$REP" 2>>"$LOG")
  grt=$(echo "$row" | cut -d, -f1); drt=$(echo "$row" | cut -d, -f3)
  if [ -z "$grt" ]; then note="grt_failed"
  elif [ -z "$drt" ] || [ "$rc" != "0" ]; then note="grt_only_rc${rc}"
  else note="ok"; fi
  echo "${v},${FV},${gp},${row},${rc},${note}" >> "$CSV"
  say "$v: rc=$rc note=$note labels=${row}"
done

say "done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat "$CSV" | tee -a "$LOG"
