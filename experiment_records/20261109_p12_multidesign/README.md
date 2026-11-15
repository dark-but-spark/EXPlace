# P1-2 multi-design matrix: `superblue1` / `superblue3`, 6 seeds (2026-11-09)

Derived, tracked record of `long_runs/p12_multidesign_20261109`.
Verified numbers regenerate with:

```bash
cd /data0/cjy_cxsj/EXPlace-test/EXPlace
/usr/bin/python3 script/summarize_experiment.py \
  long_runs/p12_multidesign_20261109 \
  --output-dir experiment_records/20261109_p12_multidesign
/usr/bin/python3 experiment_records/20261109_p12_multidesign/analyze_post_adaptation.py \
  long_runs/p12_multidesign_20261109 \
  --output-dir experiment_records/20261109_p12_multidesign
```

## Purpose

P1-2: does the `adaptive_global` advantage seen on `superblue18` transfer to
other circuits and more seeds? This run adds `superblue1` and `superblue3` and
doubles the seed count (3 → 6).

`fixed_global` vs `adaptive_global` was chosen deliberately: both run with
granularity `global` and `global_reward_gamma = 1.0`, so the only difference is
whether the PPA-correlation learner updates `reg/hier/df/wire`. (The historical
`fixed_step` arm would additionally confound the discount factor, which
`step+gamma=1` already showed to be an independent effect.)

## Scope

- Benchmarks: `superblue1`, `superblue3`
- Seeds: `3`, `7`, `11`, `17`, `23`, `29`
- Variants: `fixed_global`, `adaptive_global`
- Config: `config/adaptive_global.yaml` (same config as the 20260920 matrix, so
  the arms stay comparable to `experiment_records/20260920_superblue18_4x3`)
- Training: 1500 episodes per task, 24 tasks
- Execution mode: `legacy`, `async_evaluation=0`
- Logging: 150 training rows per task; 15 periodic DREAMPlace + OpenTimer
  evaluations per task (one every 100 episodes)
- Evaluation stage: post-placement OpenTimer, not post-route signoff
- GPUs: physical `0`–`7` (6 × RTX 6000 Ada, 2 × L40S), 8 workers in parallel
- Wall clock: 2026-11-09T11:28Z → 2026-11-10T02:33Z (~15 h); per-task
  13 890 s – 22 726 s
- Commit at launch: `6fbd468` on `cjy-test` (`run_metadata.txt` in the run root)

## Run outcome

| status | count |
| --- | ---: |
| SUCCESS | 22 |
| FAILED (`exit 139`, SIGSEGV) | 2 |
| physical evaluations | 342 |
| invalid (legality gate) | 4 (1.17%) |

Both failures crash inside OpenTimer **after** legalization passed:

```
[INFO   ] Legality check takes 134.234 ms
[INFO   ] launch rc tree construction...
Segmentation fault (core dumped)
```

- `fixed_global/superblue3/seed_3` — stopped at episode 800 (8 evaluations)
- `adaptive_global/superblue3/seed_17` — stopped at episode 800 (8 evaluations)

So `superblue3` has only 5 usable seeds for the `ep1500` comparison; the
`episode > 400` windowed comparison still has 6 per arm because both crashed
runs contributed evaluations before the crash. This is downstream of the
legality gate, not a legality failure, and it must not be read as a PPA result.

## Method note: the shared prefix inflates `best_*` comparisons

`reward_weight_min_observations: 4` means the learner does not change the reward
until four real evaluations exist. The first update therefore happens at
episode 400, and **every evaluation up to and including episode 400 is computed
with identical weights in both arms**. Those rows are bit-identical:

```
adaptive_global/sb3/seed_11        fixed_global/sb3/seed_11
ep100  -1302.54896   ==            -1302.54896
ep200  -2074.40336   ==            -2074.40336
ep300  -1962.87104   ==            -1962.87104
ep400  -1731.51552   ==            -1731.51552     <- last shared row
ep500  -2423.80880   !=            -2369.28496     <- first treated row
```

Because early episodes usually carry the mildest (least negative) TNS, a global
`best_tns` often lands on the shared prefix, which makes a paired difference
collapse to exactly `0`. Any comparison of `adaptive_*` against `fixed_*` must
therefore use **episode > 400** (strictly after the first update). `final`
(ep1500) is unaffected and stays directly comparable.

`post_adaptation_per_pair.csv` and `post_adaptation_summary.csv` in this
directory implement that rule, deriving the first-update episode from
`reward_weights.csv` rather than hard-coding it.

## Result: post-adaptation window (episode > 400)

Paired deltas are `adaptive_global - fixed_global`. For TNS/WNS higher is better;
for GP-HPWL lower is better. `sign_test_p` is an exact two-sided binomial sign
test (ties excluded).

| scope | metric | mean Δ | median Δ | mean rel. | adaptive wins | fixed wins | sign p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL (n=12) | best TNS | -3.32 | -19.68 | -1.26% | 5 | 7 | 0.774 |
| ALL (n=12) | best GP-HPWL | +1.46M | +3.35M | +0.12% | 5 | 7 | 0.774 |
| ALL (n=12) | best WNS | **+5.09** | +2.77 | **+5.05%** | **10** | 2 | **0.039** |
| ALL (n=9) | final TNS | +70.02 | +428.70 | +0.41% | 6 | 3 | 0.508 |
| ALL (n=9) | final GP-HPWL | -5.73M | +6.41M | -1.18% | 4 | 5 | 1.000 |
| ALL (n=9) | final WNS | +7.44 | +5.72 | +4.30% | 6 | 3 | 0.508 |
| sb1 (n=6) | best TNS | -71.96 | -90.75 | -6.46% | 2 | 4 | 0.688 |
| sb1 (n=6) | best GP-HPWL | -4.41M | -2.48M | -0.81% | 4 | 2 | 0.688 |
| sb3 (n=6) | best TNS | +65.32 | +75.06 | +3.93% | 3 | 3 | 1.000 |
| sb3 (n=6) | best GP-HPWL | +7.32M | +7.88M | +1.05% | 1 | 5 | 0.219 |

(`n=9` for `final` because two runs never reached episode 1500.)

### Per-seed detail

| circuit | seed | fixed best TNS | adaptive best TNS | Δ | fixed best GP | adaptive best GP | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| superblue1 | 3 | -979.01 | -825.72 | +153.29 | 475.7M | 449.9M | -25.8M |
| superblue1 | 7 | -758.23 | -958.13 | -199.89 | 480.0M | 482.3M | +2.3M |
| superblue1 | 11 | -1196.44 | -1196.49 | -0.05 | 538.2M | 538.1M | -0.2M |
| superblue1 | 17 | -1646.59 | -1929.49 | -282.90 | 705.4M | 687.1M | -18.2M |
| superblue1 | 23 | -1653.19 | -1573.96 | +79.23 | 548.3M | 543.5M | -4.8M |
| superblue1 | 29 | -1157.22 | -1338.67 | -181.45 | 569.5M | 589.7M | +20.3M |
| superblue3 | 3 | -1836.98 | -1562.94 | +274.04 | 661.4M | 672.7M | +11.3M |
| superblue3 | 7 | -1335.83 | -1146.41 | +189.42 | 686.0M | 705.1M | +19.1M |
| superblue3 | 11 | -1871.50 | -1910.81 | -39.31 | 721.1M | 714.5M | -6.6M |
| superblue3 | 17 | -1533.08 | -1676.69 | -143.61 | 791.9M | 796.3M | +4.4M |
| superblue3 | 23 | -1804.83 | -1581.21 | +223.62 | 711.7M | 721.2M | +9.5M |
| superblue3 | 29 | -1748.01 | -1860.25 | -112.24 | 730.1M | 736.4M | +6.3M |

## Weights do not converge to a stable target

Final learnable weights after 11–12 updates (first → last), `reg` / `df` / `wire`:

| circuit | seed | reg | df | wire |
| --- | ---: | --- | --- | --- |
| superblue1 | 3 | 0.381 → 0.230 | 0.154 → **0.341** | 0.188 → 0.099 |
| superblue1 | 17 | 0.378 → 0.178 | 0.173 → **0.489** | 0.128 → 0.166 |
| superblue1 | 23 | 0.371 → 0.162 | 0.129 → 0.184 | 0.271 → **0.371** |
| superblue3 | 11 | 0.386 → 0.206 | 0.144 → 0.131 | 0.166 → **0.382** |
| superblue3 | 29 | 0.379 → 0.222 | 0.177 → 0.179 | 0.177 → **0.388** |

For the same circuit, `df` ends anywhere between 0.13 and 0.49 and `wire`
between 0.10 and 0.39 depending on the seed. The learner has **no stable fixed
point across seeds**, which is what you expect if the rank-correlation signal is
partly fitting noise. That is the P1-1 (label fidelity) question, now with
direct supporting evidence.

## Conclusions

1. **The `superblue18` advantage does not replicate.** On `superblue18`,
   `adaptive_global` was clearly better than `fixed_global` on both best GP-HPWL
   (272.11M vs 277.50M) and final TNS (-148.02 vs -229.69). On `superblue1` and
   `superblue3` the same comparison is a wash or slightly negative for TNS and
   best GP-HPWL.
2. **The one reproducible cross-circuit effect is WNS**, where adaptive wins
   10/12 pairs (+5.05% mean, sign test p=0.039). WNS is not the primary metric
   and this is one test among many, so treat it as a hypothesis, not a result.
3. **Weights do not converge**, so "adaptive weighting is better" cannot be
   stated as a general claim; at most it is design- and seed-dependent.
4. **Do not report `best_*` over the full episode range for adaptive arms**
   without the episode > 400 window; the shared prefix manufactures ties.

## Provenance limits

- `n = 6` per circuit, and within-design seed spread dominates: `superblue1`
  best GP-HPWL ranges 475.7M–705.4M across seeds. **No statistical significance
  is claimed** for TNS or GP-HPWL (p = 0.77).
- `superblue3` has only 5 seeds at episode 1500 (two runs crashed at episode
  800).
- post-placement OpenTimer surrogate; not post-GRT/post-DRT, not signoff.
- `best_*` values are independent retrospective selections and may come from
  different episodes than `final`.
- 8 tasks ran concurrently on one host (load average ≈ 69 on 192 threads), so
  wall-clock per task is not a single-task measurement. Numerics are unaffected.

Machine-readable files:

- `per_seed_metrics.csv`, `aggregate_metrics.csv` — naive whole-run summary from
  `script/summarize_experiment.py`; **these contain the shared-prefix artifact**,
  kept for traceability and for comparison with the other records.
- `post_adaptation_per_pair.csv` — per-design/seed paired values on the
  `episode > 400` window, with first-update episode recorded.
- `post_adaptation_summary.csv` — aggregates, win counts, sign-test p-values.
- `analyze_post_adaptation.py` — regenerates both post-adaptation CSVs.
