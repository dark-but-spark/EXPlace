# OpenTimer SIGSEGV diagnostics

Tools kept from the 2026-11-15/16 investigation of

```
exit 139 (SIGSEGV), last log line: [INFO] launch rc tree construction...
```

## Root cause (confirmed with a stack trace)

`ot::Rct::_update_load()` recurses without a cycle guard:

```cpp
// DREAMPlace/thirdparty/OpenTimer/ot/timer/net.cpp:122
void Rct::insert_segment(node1, node2, res) {
  insert_edge(node1, node2, res);   // both directions -> undirected graph
  insert_edge(node2, node1, res);
}

// net.cpp:154  the only guard is "do not return to the immediate parent"
void Rct::_update_load(RctNode* parent, RctNode* from) {
  for (auto e : from->_fanout) {
    if (auto& to = e->_to; &to != parent) {
      _update_load(from, &to);      // net.cpp:158  <-- stack overflow on a cycle
    }
  }
}
```

`_update_delay`, `_update_ldelay` and `_update_response` use the same pattern and
would fail the same way.

The RC graph is built from the SPEF `*RES` section in `Net::_make_rct()`
(`net.cpp:295`), whose node names come from pin geometry. When a placement makes
a net's SPEF degenerate (duplicate node names collapse in `_nodes[name]`), the
segment graph closes into a loop and the traversal never terminates.

### Evidence

- Reproduced with the production `.so` untouched: `superblue3`, 2 runs, both
  `exit 139`, both with 9 dumped positions (ep100..900).
- Replaying those 9 positions in order reproduces the crash on the 9th.
- Backtrace: `#0..#44` all `_update_load` at `net.cpp:158`.
- Crash-frame node: `_name = "A1_B20_o611715:ck"` (a pin node with one fanin and
  one fanout edge) while its parent had three fanout edges.

### Ruled out

- **Not cumulative corruption**: calling `update_timing()` 12 times on the same
  position never crashes; the same position replays fine in isolation.
- **Not legality**: `[DEBUG] Macro legality check PASSED` precedes the crash.
- **Not the debug rebuild**: production `.so` sha256 `8c855e17...` was verified
  before the clean reproduction.

## Tools

| File | Purpose |
| --- | --- |
| `repro_segv_run.sh` | Runs one seed with `EXPLACE_DUMP_PRE_TIMING` set, so every evaluation's position is dumped before timing |
| `repro_timer_pos.py` | Replays one dumped position through the timer (exit 139 reproduces) |
| `repro_timer_seq.py` | Replays a directory of dumps in order; needed because the crash requires several updates in one process |
| `repro_timer_repeat.py` | Calls `update_timing()` N times on one position; used to rule out cumulative corruption |
| `build_debug_timing2.sh` | Builds `DREAMPlace/debug_timing/timing_cpp_debug.so` (non-LTO, `-g`) because the installed `.so` is LTO-stripped and has no symbols |
| `gdb_seq.sh` | Runs `repro_timer_seq.py` under gdb against the debug build via `PYTHONPATH`, without touching the installed `.so` |
| `gdb_replay_segv.sh`, `gdb_validate.sh`, `build_debug_timing.sh` | Earlier, superseded variants kept for history |

The dump hook lives in `src/problem_instance.py` and is inert unless
`EXPLACE_DUMP_PRE_TIMING` is set.

### Warning from this investigation

Do **not** replace `DREAMPlace/install/.../timing_cpp*.so` while training
processes are alive. Two runs crashed at the exact minute of a `.so` swap and the
crash was initially misattributed to the OpenTimer bug. Use the `PYTHONPATH`
approach in `gdb_seq.sh` instead.

## Decision (2026-11-16): no OpenTimer patch

Adding a visited-set guard would fix it in ~20 lines and would be zero-diff on
valid trees, but it modifies third-party numerics-critical code and creates an
upstream-sync burden. Current choice:

- **Keep the known issue documented** (this file + `TODO.md`).
- Rerun failed tasks through the matrix (`RESUME=1`); note that there is no
  checkpoint resume, so a rerun starts from episode 0 (~4 h per task).
- Failure rate so far: `superblue18` 0/24, `superblue1` 0/12, `superblue3` 2/12.
- Build process isolation for evaluations **only if** large `superblue3`
  matrices come back (see `TODO.md` for the three options and their costs).
