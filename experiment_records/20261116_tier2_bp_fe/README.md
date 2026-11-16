# bp_fe Tier-2 label (first end-to-end post-GRT sample)

Status: **pipeline works**. One placement carries Tier-1 (DREAMPlace) and Tier-2
(post-GRT / post-DRT) numbers. Enough to prove the plumbing, far too little to
compute a correlation — P1-1 needs 20–30 of these.

## Placement

| field | value |
| --- | --- |
| design | `bp_fe` (OpenROAD `nangate45`, DIE 800×600 µm, 11 macros) |
| source run | `long_runs/or_legal_bp_fe_r2` (EXPlace, 600 episodes, seed 3) |
| config | `config/or_legal.yaml` (halo 0, core_area_block 0, corner_flag False, soft_coefficient 1.0, dataflow_cutoff 0.00) |
| DEF | `placement/best_gp_hpwl.def` (9 068 068 B), staged as `OpenROAD-flow/EXPlace_defs/bp_fe.gp.def` |
| legality | EXPlace `valid=1` for all 6 periodic evaluations |

## Tier-1 (DREAMPlace proxy, timer disabled)

`use_timer_for_evaluation: False` in this config, so no OpenTimer TNS/WNS yet;
only the placement proxy is available.

| episode | gp_hpwl |
| ---: | ---: |
| 100 | 10 975 406 |
| 200 | 11 170 036 |
| 300 | 11 168 992 |
| 400 | 10 885 164 |
| 500 | 11 196 430 |
| 600 | 11 142 746 |

## Tier-2 (OpenROAD flow, post-GRT and post-DRT)

| stage | TNS (ns) | WNS (ns) | source |
| --- | ---: | ---: | --- |
| post-GRT | **-75.59** | **-0.27** | `reports/nangate45/bp_fe/EXPlace/5_global_route.rpt` |
| post-DRT (finish) | **-103.89** | **-0.29** | `reports/nangate45/bp_fe/EXPlace/6_finish.rpt` |

Also from the finish report: clock period 1.400 → 1.590 ns, 614 setup and 38 hold
violations, 220 max-capacitance violations, 748 detailed-routing violations.
DRC is expected to be non-clean: this placement came from a 600-episode run and
the global-route layer adjustment was relaxed to 0.13 to clear congestion.

## Flow knobs used for this label (non-default)

| knob | value | why |
| --- | --- | --- |
| `CORE_AREA` (bp_fe_top) | `5 5 795 595` (was `0 0 790 590`) | PDN straps landed outside the die → `GRT-0041 Net VSS has wires/vias outside die area` |
| `ROUTING_LAYER_ADJUSTMENT` | `0.13` (was 0.25) | GRT-0704/GRT-0115: routing finished with congestion at 0.25 |
| `GLOBAL_ROUTE_ARGS` | `-congestion_iterations 30 -congestion_report_iter_step 5 -verbose` | default in this config used 5 iterations |

## Reproduce

```bash
# 1. placement (fast: 600 episodes ~ 22 min on one GPU)
bash script/tier2/legal_probe.sh bp_fe 3 6 600 100 r2

# 2. flow (floorplan -> place -> cts -> grt -> drt -> finish, ~40 min)
bash script/tier2/r2_final.sh

# 3. labels
grep -A2 "report_tns\|report_wns" \
  OpenROAD-flow/reports/nangate45/bp_fe/EXPlace/6_finish.rpt
```

## Next

- Tier-1 timing is still missing for this placement (the or_cases config disables
  the timer). For P1-1 the same placement must be re-timed with
  `use_timer_for_evaluation: True` (or replayed through
  `script/repro_timer_pos.py`) so Tier-1 TNS/WNS can be paired with Tier-2.
- Then repeat for 20–30 stratified placements (best / median / worst per circuit)
  and compute per-circuit Spearman + Fisher-z aggregate.
