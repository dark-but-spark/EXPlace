# Multi-design adaptive reward rerun

This record preserves compact evidence from two local run roots:

- `long_runs/global_multidesign_legacy_20261008_111740`: 48/48 successful
  tasks, four variants, four benchmarks, and seeds 3/7/11. The synchronous
  path did not feed physical results to the adaptive learner. Consequently,
  `adaptive_step` is byte-equivalent to `fixed_step`, and `adaptive_global`
  is byte-equivalent to `fixed_global`; only its fixed arms are valid controls.
- `long_runs/global_multidesign_adaptive_fixed_20261012_095649`: 24/24
  successful reruns of the two adaptive arms after fixing the synchronous
  update path. Every reward-weight file has episode 0 plus 15 evaluations;
  updates begin at episode 400 after four observations.

The episode 100/200/300 physical rows match the old fixed controls exactly,
which is a useful cross-run determinism check before adaptation begins. Paired
final-episode results against the old fixed arms are descriptive, not a claim
of statistical significance:

| Adaptive arm | GP-HPWL favorable pairs | median GP gain | TNS favorable pairs | median TNS gain |
| --- | ---: | ---: | ---: | ---: |
| step | 8/12 | 1.639% | 9/12 | 12.601% |
| global | 11/12 | 3.895% | 7/12 | 9.898% |

WNS is mixed: step is favorable in 8/12 pairs (median 4.453%), while global is
favorable in 4/12 (median -8.995%). Historical best values often occur before
adaptation, so the strongest observation is reduced final-episode degradation,
especially final GP-HPWL for adaptive global. Three seeds and noisy timing
metrics are insufficient for a general or statistically significant claim.

`legacy/` and `adaptive_fixed/` contain the original task summary, plan,
metadata snapshot, and regenerated per-seed/aggregate physical metrics. These
runs predate the strict final-legality flag, so their CSV files are PPA records,
not proof that every placement passes the newer legality gate.
