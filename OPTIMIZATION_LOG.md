# Runtime Optimization Log

## Phase 0: expensive side effects off by default

- Scope: new configs that do not explicitly choose visualization or repeated
  best-reward physical evaluation.
- Change: `main.process_args` defaults `evaluate_best_reward_with_dmp` and
  `visualize_prototype` to false, and `visualization_interval` to zero.
- Compatibility: existing YAML values remain authoritative; `--visualize` still
  enables step visualization in test mode.
- Verification: `tests/test_performance_defaults.py` asserts the CPU smoke config
  has all three expensive paths disabled. Full unittest results are recorded in
  the commit message/run notes.
- Measurement: a full training timing comparison requires benchmark data and
  DREAMPlace; this host-level change is validated by the no-DREAMPlace smoke
  configuration before measuring end-to-end speed.

## Phase 1: compact host-side PPO observations

- Scope: rollout transition storage only; actor inference remains float32.
- Change: observations are copied as `float16` into the PPO buffer and widened
  once while assembling an update tensor. This avoids repeated float32 host
  copies while retaining the existing model input layout.
- Verification: the reward-granularity test checks a 50% byte reduction and
  `float32` reconstruction within `1e-3` tolerance; full unittest output is
  recorded with the phase commit.
- Limitation: this is the compatible first step toward structural state
  reconstruction. It does not yet remove the multi-channel representation from
  Ray messages; that requires a worker-side state cache and a separate rollout
  protocol.
