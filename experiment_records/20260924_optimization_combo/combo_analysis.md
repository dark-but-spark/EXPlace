# Combination Experiment Analysis

Source runs:

- Performance: `optimization_runs/combo_long_20260924_061555`
- Quality: `optimization_runs/combo_quality_20260924_144146`
- Samples: `superblue18`, seeds 3, 4, 5

All 15 performance tasks and all 15 physical quality evaluations completed successfully.

## Performance

Speedup is baseline time divided by variant time; values above 1.0 are faster.

| Variant | Wall speedup | Rollout speedup | PPO speedup |
| --- | ---: | ---: | ---: |
| `masks_ppo` | 1.0304x | 1.0208x | 1.2468x |
| `masks_amp_bf16` | 1.0304x | 1.0213x | 1.2087x |
| `ppo_amp_bf16` | 1.0174x | 0.9892x | 1.5752x |
| `masks_ppo_amp_bf16` | 1.0414x | 1.0164x | 1.5699x |

`masks_ppo_amp_bf16` is the fastest combination in this run, but its advantage is mostly in the PPO stage. `ppo_amp_bf16` makes PPO fastest while slightly slowing rollout. The observed wall improvements are modest (1.7% to 4.1%), so they should not be treated as a large system-level gain yet.

## Quality versus same-seed baseline

Negative GP percentage is better. For TNS and WNS, positive deltas are better because the values are negative and closer to zero.

| Variant | Mean GP delta | Mean TNS delta | Mean WNS delta |
| --- | ---: | ---: | ---: |
| `masks_ppo` | +0.286% | -9.912 | -1.301 |
| `masks_amp_bf16` | +1.344% | -28.195 | -0.089 |
| `ppo_amp_bf16` | +1.008% | -29.754 | +0.279 |
| `masks_ppo_amp_bf16` | +0.765% | -31.600 | -1.614 |

The combinations improve neither all metrics nor all seeds consistently. `masks_ppo` has the smallest mean GP_HPWL regression and the smallest timing regression, so it is the most conservative candidate. `masks_ppo_amp_bf16` gives the best speed but has a larger quality tradeoff. `masks_amp_bf16` and `ppo_amp_bf16` show larger GP_HPWL regressions.

## Decision

No combination is a strict Pareto winner under the gate `wall_speedup > 1`, GP_HPWL no worse than baseline, and TNS/WNS no worse than baseline. The current ranking is:

1. `masks_ppo` for a quality-preserving follow-up.
2. `masks_ppo_amp_bf16` for a speed-oriented follow-up.
3. `ppo_amp_bf16` only if PPO throughput is the primary objective.
4. `masks_amp_bf16` is not preferred because its GP regression is larger without a wall-time advantage over `masks_ppo`.

The sample is only three seeds and the quality deltas vary substantially by seed. Before selecting a final method, repeat the two leading candidates (`masks_ppo` and `masks_ppo_amp_bf16`) on additional seeds with the same paired baseline protocol. Do not claim a definitive quality improvement from the current three-seed result.
