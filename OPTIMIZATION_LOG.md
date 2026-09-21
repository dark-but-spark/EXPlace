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

## Phase 4: batched worker stepping foundation

- Change: Ray workers now expose `step_many(actions)`, executing a contiguous
  action chunk locally and returning one result list. It stops at episode end.
- This is the transport primitive for worker-side rollout aggregation and is
  useful for evaluation/precomputed action sequences without changing step
  semantics. Training still uses policy-dependent single steps until model
  synchronization is introduced and validated.

## Phase 5: gated CNN mixed precision

- Change: add opt-in AMP around actor and critic forward passes, supporting
  `float16` or `bfloat16`. Defaults remain disabled and CPU execution never
  enables AMP.
- Gate: numerical comparison and GPU throughput measurement are required before
  enabling this in production configs; no speedup claim is made on this host.

## Phase 6: vectorized occupancy updates

- Change: maintain a boolean occupied grid and derive the position mask with an
  integral image. This removes the Python loop over every placed macro while
  preserving exact rectangle-overlap semantics and boundary blocking.
- Verification: the environment test confirms overlap and free-position results;
  full suite result for this phase is 14/14 passing.
- Measurement note: benchmark-level timing still requires a Ray run; the local
  Ray GCS startup is unavailable in this container.

## Async physical evaluation audit

- A first thread-based draft was rejected because sharing `ProblemInstance`,
  OpenTimer, or the training environment across threads is unsafe and it lost
  existing best-metric handling.
- The safe implementation target is a single Ray evaluator actor owning an
  isolated `ProblemInstance`, with at most one in-flight deep-copied placement;
  results must be consumed without changing the training episode counter.

## Phase 7: compact environment snapshot API

- Change: `PlaceEnv.compact_snapshot()` stores placement metadata, macro order,
  and reward-scaling bounds; `rebuild_state()` reconstructs the full policy
  observation on demand.
- Verification: a two-macro environment test compares the rebuilt observation
  against the live state element-wise.
- Integration note: PPO/Ray transition wiring remains separate until the
  snapshot path is validated on dynamic wire/dataflow cases.

## Phase 8: compact transition integration

- Change: compact transitions are enabled by default. Ray reset/step helpers
  return a snapshot alongside the policy observation; PPO stores snapshots and
  materializes the full state through the driver environment only at update
  time.
- Compatibility: `compact_transition: false` retains the previous FP16 state
  buffer path.
- Verification: snapshot reconstruction is element-wise equivalent in the
  environment test; full Ray training still requires a functioning Ray runtime.
