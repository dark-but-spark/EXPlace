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
- Verification: a CPU BF16 local-CNN comparison test has max absolute error
  below `1e-2`. This does not substitute for the full actor/GPU FP16/BF16 gate;
  `mixed_precision` remains explicitly false in `compare_quick`.

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
- Measurement: on the 32x32 two-macro fixture, the float32 observation is
  20,492 bytes while the compact snapshot serializes to 166 bytes. Scaling the
  observation layout to 224x224 gives about 1,004,108 bytes for this reduced
  mask fixture; the full six-mask production state is larger as documented in
  `EFFICIENCY.md`.

## Phase 9: isolated asynchronous physical evaluation

- Change: add a single Ray `AsyncPhysicalEvaluator` actor that owns an isolated
  CPU `ProblemInstance`/OpenTimer state. The training loop submits at most one
  deep-copied placement and polls it without blocking while later rollout/PPO
  work proceeds.
- Result handling preserves periodic best-TNS/best-GP-HPWL artifacts and
  adaptive reward updates. Optional best-reward evaluation is also submitted
  to the same bounded actor queue.
- Compatibility: `async_evaluation: false` retains the synchronous evaluation
  path. The default is true when physical evaluation is enabled.
- Verification: syntax and unit tests run locally; end-to-end timing remains
  pending because the container's Ray GCS startup is unavailable.

## Phase 10: worker-side complete rollouts

- Change: Ray workers now synchronize the actor weights once and execute a full
  policy-controlled episode locally. They return one compact rollout object,
  so the driver no longer transfers observations or invokes `step.remote()` per
  macro.
- Compatibility: `worker_rollout: false` uses the previous driver-controlled
  stepping path. The CPU comparison config enables worker rollouts explicitly.
- Verification: syntax and unit tests cover configuration/defaults; a live
  Ray benchmark is still required for transport and wall-clock measurements.

- Protocol verification: the fake-worker test confirms one `rollout.remote`
  call per worker and correct transition aggregation without requiring a Ray
  cluster.

### Smoke evidence (2026-09-21)

- `compare_quick` loaded the cached `superblue18` environment successfully but
  failed before worker creation with Ray's `Timed out waiting for file
  /tmp/ray/.../gcs_server_port_*`. This is an infrastructure failure; no
  rollout-path timing is claimed from this run.

## Phase 11: positive A/B switches for optimized paths

- Change: add `mask_cache`, `vectorized_occupancy`, and `batch_device_copy`
  configuration switches. All default to `true`; setting one to `false`
  restores the corresponding pre-optimization implementation for controlled
  measurements.
- Verification: the cache-disabled path, vectorized/reference position-mask
  equivalence, and both PPO device-staging paths are covered by unit tests.
  Full suite result: 21/21 passing; Python syntax checks pass.
- Measurement: no new runtime multiplier is claimed in this container. These
  switches make the required paired benchmark possible on a functioning Ray
  and/or CUDA host. The expected speedup must be computed as
  `baseline_time / optimized_time` using identical seeds and workloads.
- Compatibility: existing configs receive the optimized defaults through
  `process_args`; `compare_quick.yaml` records the three values explicitly.

## Phase 12: reproducible long-run speedup harness

- Change: add `config/optimization_long.yaml` (600 episodes, 10 parallel
  environments) and `script/run_optimization_long.sh`. The script runs
  baseline, one-factor ablations, all optimizations, and BF16 AMP with the
  same benchmark/seed matrix.
- Measurement: each task stores the exact command, stdout, wall duration, and
  `runtime.csv` means. `speedups.csv` reports `baseline_time / variant_time`
  for wall, rollout, and PPO time. No multiplier is recorded until the run
  succeeds; missing runtime data is written as zero and should be treated as
  an invalid measurement.
- Physical evaluation: `config/optimization_physical_long.yaml` enables
  DREAMPlace/OpenTimer. Run only `async_off` and `async_on` with that config
  to isolate asynchronous evaluation; the default long run keeps physical
  evaluation off to avoid mixing design-signoff cost into RL throughput.
- Verification: shell syntax, dry-run command generation, Python syntax, and
  the full unit suite pass locally. Actual long-run timings require a host
  where CUDA devices and Ray GCS socket binding are available.

### First GPU long-run result (2026-09-21)

- The 600-episode matrix completed 15/21 tasks. Seeds using GPU 1 failed with
  CUDA out-of-memory, and all `amp_bf16` tasks failed because this PyTorch/CUDA
  build does not implement `smooth_l1_backward_cuda` for BF16.
- The first report incorrectly showed zero rollout/PPO means because the
  script looked only at `output/runtime.csv`; `main.py` writes below a nested
  `output/rl_logs/.../<timestamp>/` directory. The harness now searches
  recursively. Wall durations remain valid; component means from that report
  must be recomputed from the nested files.
- Successful wall-time evidence is inconclusive for a full optimized matrix:
  worker-only was faster on seeds 3/5, while the cumulative optimized path was
  approximately equal to or slower than baseline. Mask/vectorized and PPO
  component groups were slower in this workload. These results do not justify
  claiming a universal speedup.
