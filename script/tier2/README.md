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

## Current state (2026-11-16)

Working:
- `bp_fe` produces legal placements with `config/or_legal.yaml`
  (`valid=1` at ep30 and ep60, `best_gp_hpwl.def` written, 60 episodes ≈ 64 s).
- `run_flow.sh` gets `bp_fe` through floorplan → PDN → place → CTS.

Blocked:
- **`5_1_grt`**: `global_route` logs `[WARNING GRT-0041] Net VSS has wires/vias
  outside die area` repeatedly and then the stage dies reporting
  `can't read "env(RESULTS_DIR)"` — i.e. `global_route` fails and the failure
  branch of `global_route.tcl` trips over the environment. Root cause not yet
  isolated (suspicion: the PDN straps extend outside the die because
  `CORE_AREA`/`DIE_AREA` for this design are 790×590 inside 800×600).
- Running `make grt` directly is not a workaround: the target depends on
  `1_synth.rtlil`, which needs `oss-cad-suite/bin/yosys` (not installed).

Next steps:
1. Check the `GRT-0041` VSS-outside-die warnings: compare `own_pdn.tcl`
   strap offsets against `DIE_AREA`/`CORE_AREA`, or add `-pitch`/`-offset`
   clamping.
2. If GRT keeps failing, try `SKIP_INCREMENTAL_REPAIR=1` and
   `SKIP_ANTENNA_REPAIR=1` to see whether the failure is in the repair flow
   rather than the router itself.
3. Any flow change must be listed in the experiment record so Tier-2 labels are
   reproducible.
