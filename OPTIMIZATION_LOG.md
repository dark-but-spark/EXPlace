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
