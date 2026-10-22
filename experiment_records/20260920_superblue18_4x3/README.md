# superblue18 4x3 Experiment Record

This record is a tracked, derived summary of the local raw run at
`long_runs/20260920_140834`. Regenerate it with:

```bash
cd /data0/cjy_cxsj/EXPlace-test/EXPlace
/usr/bin/python3 script/summarize_experiment.py \
  long_runs/20260920_140834 \
  --output-dir experiment_records/20260920_superblue18_4x3
```

## Scope

- Benchmark: `superblue18`
- Seeds: `3`, `7`, `11`
- Variants: `fixed_step`, `fixed_global`, `adaptive_step`, `adaptive_global`
- Training: 1500 episodes per task, 12/12 tasks successful
- Logging: 150 training rows per task, one every 10 episodes
- Physical evaluation: 15 DREAMPlace + OpenTimer evaluations per task, one
  every 100 episodes
- Evaluation stage: post-placement OpenTimer, not post-route signoff

`best_gp_hpwl`, `best_tns`, and `best_wns` are independent retrospective
selections and may refer to different episodes. `final` is the physical
evaluation at episode 1500. Positive `*_best_to_final_percent` means the final
value regressed away from that metric's independently selected best value.

## Result

| variant | mean best GP-HPWL | mean best TNS | mean final GP-HPWL | mean final TNS | mean GP regression | mean TNS regression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_step | 275.68M | -159.48 | 284.75M | -241.56 | 3.29% | 56.78% |
| fixed_global | 277.50M | -184.13 | 282.47M | -229.69 | 1.80% | 25.23% |
| adaptive_step | 274.60M | **-133.61** | 283.52M | -167.40 | 3.28% | 25.17% |
| adaptive_global | **272.11M** | -141.61 | **272.33M** | **-148.02** | **0.08%** | **5.26%** |

The stable conclusion is that adaptive weighting improves timing and that
`adaptive_global` has the smallest late-training regression in this run.
Global reward alone is not beneficial. This single-benchmark, three-seed result
does not establish statistical significance, routing quality, or full PPA.

## Provenance Limits

The run predates `script/capture_run_metadata.sh`; therefore it has no
contemporaneous `run_metadata.txt`. The command log proves `/usr/bin/python3`
and GPU id 0 were selected, and the retained DREAMPlace configure file records
the cu118-era CUDA 11.8 build, but current host probes cannot prove the exact
driver/GPU state at run time. Do not add a retrospectively captured metadata
file to the old run directory as if it were contemporaneous evidence.

Machine-readable files:

- `per_seed_metrics.csv`: checkpoint episodes, same-checkpoint companion
  metrics, final values, and best-to-final regressions for all 12 tasks.
- `aggregate_metrics.csv`: means across the three seeds for each variant.
