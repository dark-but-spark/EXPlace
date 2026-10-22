# Optimization combination experiment

This directory is the compact archive originally produced under
`optimization_runs/combo_quality_20260924_144146/archive`.

All 15 performance tasks and 15 quality evaluations completed on superblue18,
seeds 3/4/5. The fastest combination, `masks_ppo_amp_bf16`, improved mean wall
time by about 4.1% and PPO time by about 57%, but regressed mean GP-HPWL by
0.765% and mean TNS/WNS. `masks_ppo` was the conservative option: about 3.0%
wall speedup with 0.286% mean GP-HPWL regression. No tested combination was a
strict speed-and-quality Pareto winner.

See `combo_analysis.md` for interpretation and the CSV files for per-seed
measurements. Raw checkpoints and logs remain in ignored local directories.
