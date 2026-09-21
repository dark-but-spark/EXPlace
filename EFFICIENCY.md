# EXPlace Efficiency Analysis

## Measured quick comparison

Environment: CPU-only, `superblue18`, seed 3, 2 Ray workers, 4 episodes,
1 PPO epoch, no DREAMPlace/OpenTimer evaluation, no visualization.

| Reward mode | Total training | Mean rollout/batch | Mean PPO/batch |
| --- | ---: | ---: | ---: |
| step | 23.34 s | 8.93 s | 2.63 s |
| global | 23.22 s | 8.91 s | 2.58 s |

The terminal global reward adds no material runtime overhead. Four episodes are
only a pipeline smoke test; they are not sufficient for a placement-quality or
PPA conclusion. Both runs sampled the same best trajectories, while their actor
losses differed because their return targets differ.

The earlier repository log for `superblue18` reported 44.17 s for one batch,
of which PPO took only 0.83 s. That run performed physical evaluation and
visualization in the batch path, so it is diagnostic evidence rather than a
direct speedup comparison with the CPU-only run above.

After changing observation construction from repeated float64 concatenation to
a single float32 concatenation, the same step run took 22.69 s, with mean
rollout time reduced from 8.93 s to 8.68 s. This is a 2.8% end-to-end reduction
on the small 57-macro smoke test, with identical reported reward components.
The more important effect is the 50% reduction in observation/buffer memory.

## Implemented efficiency fixes

- Physical evaluation is initialized only when a configured code path uses it.
- Best-reward DREAMPlace evaluation and visualization can be disabled for RL
  ablations while their historical defaults remain enabled.
- Reward-scaling warm-up trajectories no longer occupy the PPO training buffer.
- The critic receives only `place_idx`, the only state feature used by its
  current forward pass, and its output is reused for the value loss.
- Environment observations are assembled with one concatenate directly as
  `float32`. Reward masks and reward calculations themselves are unchanged.
- CPU-only experiments no longer reserve Ray GPU resources.
- Each run writes `resolved_config.yaml`, including command-line overrides.

## Remaining bottlenecks

Each observation contains one canvas and current/next copies of six masks at
224 x 224. This is 652,291 values: 4.98 MiB as the previous float64 state and
2.49 MiB as float32. For `superblue4` (143 macros), a 10-environment rollout
buffer falls from about 6.95 GiB to 3.48 GiB, before temporary training copies.

Recommended next work, in priority order:

1. Store compact channel tensors or reconstruct look-ahead masks on demand,
   instead of retaining every flattened full state in Python transitions.
2. Profile individual mask generators. Cache masks that depend only on macro
   geometry and vectorize the remaining placed-macro updates.
3. Replace per-step Ray transfer of full observations with worker-side rollout
   fragments, returning a batch once per episode.
4. Transfer each PPO minibatch to the accelerator once and consider mixed
   precision for CNN branches after numerical comparison.
5. Run DREAMPlace/OpenTimer asynchronously at a lower cadence; do not place it
   on every new proxy-reward best event.

Use `config/compare_quick.yaml` with `--reward-granularity step` or `global` to
repeat the smoke comparison. Enable physical evaluation separately for final
quality comparisons.
