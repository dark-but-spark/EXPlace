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

## Phase 2: PPO batch device staging

- Change: state, action, reward, old log-probability, done flags, and return
  targets are moved to the configured device once per PPO update. Minibatch
  iterations now index already-resident tensors instead of issuing repeated
  `.to(device)` calls.
- Compatibility: model inputs and loss calculations are unchanged.
- Verification: the full unit suite and a CPU update smoke test must pass before
  this phase is considered complete; GPU transfer timing requires a CUDA node.

## Phase 3: static mask caching

- Change: masks depending only on macro geometry and static floorplan metadata
  (`reg`, `port`, `displacement`) are cached per macro and reused across reset
  and step look-ahead construction. Cached arrays are copied on access to avoid
  accidental mutation by normalization.
- Verification: the environment test checks cache reuse and no aliasing.
- Scope: dynamic `wire`, `df`, `hier`, and `pos` masks remain placement-dependent
  and are not cached.
