# superblue18 step+gamma=1 Experiment Record (2026-11-08)

This record is a tracked, derived summary of the raw run at
`long_runs/step_gamma1_20261108`. Regenerate it with:

```bash
cd /data0/cjy_cxsj/EXPlace-test/EXPlace
/usr/bin/python3 script/summarize_experiment.py \
  long_runs/step_gamma1_20261108 \
  --output-dir experiment_records/20261108_superblue18_step_gamma1
```

## Why this run exists

The original 2x2 ablation confounded **reward granularity** with the
**discount factor**: `step` ran with `gamma = 0.95` while `global` ran with
`global_reward_gamma = 1.0`. `step + gamma=1` is the control group that
separates the two.

`gamma` in this codebase is also the PLM community resolution
(`problem_instance.py`), so the discount could not be changed by editing
`gamma`. A separate `reward_discount_gamma` was added in `src/agent.py`
(`resolve_return_gamma`), and `config/step_gamma1.yaml` sets it to `1.0` while
leaving `gamma: 0.95` untouched.

## Scope

- Benchmark: `superblue18`
- Seeds: `3`, `7`, `11`
- Variants: `fixed_step`, `adaptive_step`
- Config: `config/step_gamma1.yaml` (`reward_granularity: step`,
  `reward_discount_gamma: 1.0`, `gamma: 0.95`)
- Training: 1500 episodes per task, 6/6 tasks successful
- Execution mode: `legacy` (comparable to the 20260920 matrix)
- Logging: 150 training rows per task, one every 10 episodes
- Physical evaluation: 15 DREAMPlace + OpenTimer evaluations per task, one
  every 100 episodes
- Evaluation stage: post-placement OpenTimer, not post-route signoff
- GPU: L40S ids `3` and `5`
- Commit: `39e82e1` (`cjy-test`), recorded in `run_metadata.txt`

`best_*` are independent retrospective checkpoint selections and may refer to
different episodes. `final` is the physical evaluation at episode 1500.
Positive `*_best_to_final_percent` means the final value regressed away from
that metric's independently selected best value.

## Result

superblue18, mean over seeds 3/7/11:

| variant | discount | mean best GP-HPWL | mean best TNS | mean final GP-HPWL | mean final TNS | GP regression | TNS regression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_step | 0.95 | 275.68M | -159.48 | 284.75M | -241.56 | 3.29% | 56.78% |
| fixed_global | 1.0 | 277.50M | -184.13 | 282.47M | -229.69 | 1.80% | 25.23% |
| adaptive_step | 0.95 | 274.60M | -133.61 | 283.52M | -167.40 | 3.28% | 25.17% |
| adaptive_global | 1.0 | **272.11M** | -141.61 | **272.33M** | **-148.02** | **0.08%** | **5.26%** |
| **fixed_step** | **1.0** | 278.35M | -167.12 | 279.53M | -233.31 | 0.43% | 41.87% |
| **adaptive_step** | **1.0** | 273.70M | **-123.88** | 278.33M | -190.56 | 1.70% | 55.40% |

The first four rows are the 20260920 baseline record
(`experiment_records/20260920_superblue18_4x3`), reproduced here for a
same-benchmark, same-seed comparison. The last two are this run.

## Conclusions

1. **The discount factor matters within `step`.** Moving the step discount
   from 0.95 to 1.0 improves `adaptive_step` best TNS from -133.61 to
   **-123.88** (7.3% better) and best GP-HPWL from 274.60M to 273.70M.
2. **`step+gamma=1` is not equivalent to `global+gamma=1`.** The two differ in
   a consistent direction: `step` reaches the strongest peak TNS (-123.88 vs
   -141.61) while `global` reaches the best peak GP-HPWL (272.11M vs 273.70M)
   and the best final TNS (-148.02 vs -190.56).
3. **Stability favors `global`.** `adaptive_global` regresses 0.08% on GP-HPWL
   and 5.26% on TNS from best to final, whereas `adaptive_step+gamma=1`
   regresses 1.70% and 55.40%.
4. Therefore the original H1 framing ("granularity vs discount, pick one") is
   incomplete. The supported reading is: **the discount is a real confound
   that was corrected, but granularity still has an independent effect** --
   `step` is the peak-seeking setting, `global` is the stable-convergence
   setting.

## Legality

`adaptive_step` seed 3 had 2 of 15 physical evaluations rejected by the
legality gate (`invalid_evaluations=2`). All other tasks had 0 rejects. Rejected
layouts are never converted into valid evaluations, so they reduce effective
sample counts rather than biasing metrics.

## Provenance limits

- Single benchmark, three seeds: **no statistical significance is claimed.**
  Per-seed spread is large (e.g. `adaptive_global` seed 7 reaches -185.41 while
  seeds 3/11 reach -121.96/-117.46).
- post-placement OpenTimer surrogate, not post-GRT/post-DRT signoff.
- The 20260920 baseline predates `script/capture_run_metadata.sh`; this run has
  a contemporaneous `run_metadata.txt`.

Machine-readable files:

- `per_seed_metrics.csv`: checkpoint episodes, companion metrics, final values,
  and best-to-final regressions for all 6 tasks.
- `aggregate_metrics.csv`: means across the three seeds for each variant.
