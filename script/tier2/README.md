# P1-1 Tier-2 pipeline (OpenROAD post-GRT labels)

Goal: label a set of placements with **post-GRT timing** so Tier-1
(post-placement DREAMPlace + OpenTimer) can be checked against something closer
to the real PPA. See `HANDOVER` §3 P1-1 for the decision gate (ρ > 0.7 keep /
0.4–0.7 downgrade / < 0.4 stop learning weights).

## Data flow

```
EXPlace run on an or_cases design          (config/or_legal.yaml)
  └─ placement/best_gp_hpwl.def  ──────────┐
                                           ▼
                     OpenROAD-flow/EXPlace_defs/<nick>.gp.def
                                           │
        run_flow.sh  ── make run_EXPlace ──►  macro_out   (place_macro lines)
                     ── make run_wo_synth ─► floorplan → place → cts → grt → finish
                                           │
                                    5_1_grt.odb + STA report
                                      └─ Tier-2 labels (WNS/TNS/rWL)
```

The ICCAD2015 `superblue*` designs have no OpenROAD flow here, so Tier-2 work
must use the `benchmarks/or_cases` family (`ariane133`, `ariane136`,
`black_parrot`, `bp_be_top`, `bp_fe_top`, `swerv_wrapper`). That is also the
family PPAPlace reports on, so their published `ρ̄ = 0.86` doubles as a sanity
check for our numbers.

## Scripts

| Script | Purpose |
| --- | --- |
| `legal_probe.sh` | Short EXPlace run with legality-hostile knobs relaxed; prints `physical_evaluations.csv` (`valid=1` means DREAMPlace's legality check passed) and the produced DEF |
| `run_flow.sh` | Stages a DEF as `<nick>.gp.def` and runs `run_EXPlace` + `run_wo_synth` |
| `grt_probe.sh` | Tries the `grt` make target directly (currently blocked: the target pulls synthesis and `oss-cad-suite/bin/yosys` is absent) |

Both flow scripts must run with `/usr/bin` first on `PATH`: the flow uses
`#!/usr/bin/env python3` and needs PyYAML, while the conda `python3` on PATH has
no yaml. They also need `OPENROAD_EXE=/data0/cjy_cxsj/OpenROAD/install/bin/openroad`.

## Flow defects found and fixed in this copy

All are inside `OpenROAD-flow/` and are required for the flow to run at all:

1. **`scripts/*.py|sh` were not executable** — `floorplan.tcl` invokes them
   directly and died with `permission denied`. Fixed with `chmod +x` (32 files,
   mode-only change).
2. **`PDN_TCL` missing in five of six `config_EXPlace.mk`** — only
   `bp_be_top` declared it, so the other designs fell back to the platform
   default PDN and failed with `PDN-0179`. Added a literal path per design
   (`$(DESIGN_NICKNAME)` is wrong for `bp_fe_top`, whose nickname is `bp_fe`).
3. **`own_pdn.tcl`: metal4 strap commented out in all six designs** while the
   same file still calls `add_pdn_connect -layers {metal1 metal4}` and
   `{metal4 metal7}`. The grid therefore owned no metal4 shapes →
   `PDN-0113 Cannot find shapes to connect to on metal4`. Uncommented.
4. The flow is written for a different OpenROAD version than the installed one
   (`bazel-nostamp`): `rtl_macro_placer -signature_net_threshold` is rejected
   (`STA-0562`), so the **RTLMP fallback cannot be used** — macro placement must
   come from our DEF.

## Resolved blockers on the way to post-GRT labels

1. **`GRT-0041 Net VSS has wires/vias outside die area`** — `bp_fe_top` had
   `CORE_AREA = 0 0 790 590`, i.e. the core origin equalled the die origin, so
   PDN rails/straps straddled the die boundary. Inset to `5 5 795 595`
   (ariane133 already uses a 5 um inset). GRT-0041 count: 15 -> 0.
2. **`GRT-0115 Global routing finished with congestion`** — the first DEF came
   from a 60-episode run (effectively untrained). With a 600-episode placement
   plus `ROUTING_LAYER_ADJUSTMENT=0.13` (the value GRT-0704 itself suggests) and
   `GLOBAL_ROUTE_ARGS="-congestion_iterations 30 -congestion_report_iter_step 5
   -verbose"` (the config quietly used 5), congestion disappears.
3. **`can't read "env(RESULTS_DIR)"`** — the misleading error behind both of the
   above. Real cause: the EXPlace fork moved `source .../write_ref_sdc.tcl`
   inside `proc global_route_helper`, and that script used the unqualified
   `$env(RESULTS_DIR)`; inside a proc the global env array is not visible. Fixed
   to `$::env(RESULTS_DIR)`. This error appears whenever `global_route` or its
   post-processing fails, so always look at the warnings *above* it.

## Verified end-to-end (bp_fe, seed 3, 600 episodes)

```
floorplan -> PDN -> place -> CTS -> GRT -> DRT -> fill -> 6_report      ~40 min
post-GRT : TNS -75.59 ns, WNS -0.27 ns
post-DRT : TNS -103.89 ns, WNS -0.29 ns, 748 DRT violations (not DRC-clean)
```

Record: `experiment_records/20261116_tier2_bp_fe/`.

## Tier-1 timing is not available on or_cases designs

Leaving `use_timer_for_evaluation: True` on `bp_fe` aborts:

```
timing_cpp.cpp:162 Assertion `net_iter != timer.nets().end()' failed:
could not find net name icache_1/lce/_07002_ in timer
```

DREAMPlace's net names do not match the timer's netlist for these designs, so
P1-1 pairs **Tier-1 = DREAMPlace proxy (`gp_hpwl`)** with Tier-2 post-GRT timing,
which is also what PPAPlace's HPWL-vs-PPA comparison does. Not fixable from our
side; see TODO.md.

## Batch runner

`batch_p1_1.sh` produces paired labels for several seeds of one design:

- Phase 1: N EXPlace runs (600 episodes) in parallel, one GPU each, timer off.
- Phase 2: OpenROAD flow per placement, **sequential** (the flow stages a single
  `EXPlace_defs/<nick>.gp.def` and a single results directory), with the three
  knobs from the fixes above.
- Writes `experiment_records/20261117_p1_1_labels/{labels.csv,run.log,flow_seed_*.log}`
  and a `summary.txt` with manual Spearman (scipy is not installed).

```bash
DESIGN=bp_fe NICK=bp_fe CFGDIR=bp_fe_top SEEDS="3 7 11 17 23" GPUS="3 4 5 6 7" \
  bash script/tier2/batch_p1_1.sh
```

Reminder: with n < 10 the correlation is a pipeline check, not a P1-1 decision.
