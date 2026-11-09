# superblue4 / superblue18 Legalization Matrix (50 episodes)

Derived summary of `long_runs/20261027_105819_legalization_finish`. Regenerate with:

```bash
cd /data0/cjy_cxsj/EXPlace-test/EXPlace
/usr/bin/python3 script/summarize_experiment.py \
  long_runs/20261027_105819_legalization_finish \
  --output-dir experiment_records/20261027_legalization_finish
```

## Purpose

A **correctness** matrix, not a PPA experiment. It checks that the DREAMPlace
legality repair (`patches/dreamplace/macro_legalize_local_grid_repair.patch`)
holds across 4 variants, 2 benchmarks and 3 seeds at 50 episodes per task.

## Scope

- Benchmarks: `superblue4`, `superblue18`
- Seeds: `3`, `7`, `11`
- Variants: `fixed_step`, `fixed_global`, `adaptive_step`, `adaptive_global`
- 50 episodes per task, `evaluation_interval` default; no convergence claimed
- 24/24 tasks successful

## Legality outcome

| benchmark | legal evaluations | rate |
| --- | ---: | ---: |
| `superblue18` | 60/60 | 100% |
| `superblue4` | 47/60 | 78.3% |

All 13 rejected evaluations carry `dreamplace_legality_check_failed`. Failures
are thin strips of 2–9 units where standard cells overlap **fixed macros**
(observed widths `577×9`, `90×9`, `1940×9`), concentrated on seed 7 (8/13);
the four variants differ by at most 1 failure each, so the failure is driven by
seed/placement rather than by reward weights.

## Decision (2026-11-08)

The remaining thin-strip failures are **deferred, not fixed**. The legality gate
rejects them, so they reduce the effective sample count but never become valid
evaluations or bias metrics; `superblue18` (the main line) is 100% legal. See
the `P0: DREAMPlace 合法性` section of `TODO.md` for the reasoning and the
conditions for reopening this work.

Do **not** summarize this as "DREAMPlace tolerates small errors": DREAMPlace's
Hannan/Abacus legalization resolves standard-cell-to-standard-cell overlap, but
does not guarantee moving a cell out from under a *fixed* macro.

## Provenance limits

- 50 episodes only: enough to exercise legalization, far too few for PPA.
- `best_*` values here must not be compared against 1500-episode runs.
