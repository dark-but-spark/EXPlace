# CUDA 12 Migration Archive

This file records the CUDA 12 migration work separately from the working
EXPlace/DREAMPlace baseline. The purpose is to allow operator rewrites and
performance experiments without overwriting the reproducible environment used
for the existing placement results.

## Baseline To Preserve

- Python runtime used by the long runs: `/usr/bin/python3`.
- PyTorch observed in the host environment: `2.0.0+cu118`.
- DREAMPlace is built through `script/make.sh` and loaded from
  `DREAMPlace/install`.
- The current result archive is
  `long_runs/20260920_140834/RESULT_ANALYSIS.md` and its raw data is under
  `long_runs/20260920_140834/`.
- Do not replace `DREAMPlace/install` in place while the CUDA 11.8 baseline is
  still being used for comparisons.

## Current Status

CUDA 12 compilation is a separate engineering task. Some DREAMPlace CUDA
operators do not compile unchanged with the CUDA 12 toolchain and need source
rewrites. No CUDA 12 result is currently valid for comparison with the 12-run
EXPlace matrix. Until the rewritten operators pass the checks below, the
cu118 build remains the reference implementation.

Host preflight on 2026-09-21 found:

- CUDA 12.2 toolkit at `/usr/local/cuda-12.2`;
- the default `/usr/local/cuda` symlink points to a missing CUDA 12.4 tree, so
  it must not be used implicitly;
- `/usr/bin/python3` loads PyTorch `2.0.0+cu118` with CUDA runtime 11.8 and ABI 0;
- `nvidia-smi` cannot communicate with the driver and PyTorch sees zero CUDA
  devices;
- PCI devices are visible, but without a working driver their CUDA index and
  compute capability are not authoritative.

The current evidence is archived in
`experiment_records/cuda12_20260921/host_preflight.txt`. This permits a
compile-only diagnostic with an explicitly provisional architecture, not a
correctness or performance claim.

The problem is not expected to be solved by changing only `nvcc`. The build
must keep the PyTorch CUDA runtime, CUDA toolkit, C++ ABI, compiler, and GPU
architecture compatible. The current CMake defaults also target older
architectures; the target `sm_XX` must be selected from the actual GPU.

## Recommended Work Order

1. Use `script/build_dreamplace_cuda12.sh`; it creates isolated
   `DREAMPlace/build-cuda12` and `DREAMPlace/install-cuda12` trees and refuses
   the reference build/install paths.
2. Record the toolchain tuple: Python, PyTorch version, `torch.version.cuda`,
   `nvcc --version`, NVIDIA driver, GPU name, compute capability, GCC, and
   `_GLIBCXX_USE_CXX11_ABI`.
3. Build one failing operator at a time. Record the exact compiler error,
   source file, CUDA API or C++ construct involved, and the minimal patch.
4. Rebuild the complete operator set after each compatibility patch. Do not
   declare success when only the first extension compiles.
5. Run numerical smoke tests on the same placement input with CPU, cu118, and
   CUDA 12 builds. Compare operator outputs and gradients before timing.
6. Run one fixed `superblue18` evaluation with identical macro coordinates,
   seed, DREAMPlace parameters, and iteration count.
7. Only after correctness matches, benchmark 100, 300, and 1000 global
   placement iterations and the full OpenTimer evaluation.

## Operator Rewrite Guidance

The likely compatibility work belongs in the CUDA/C++ extensions under
`DREAMPlace/dreamplace/ops/` and their CMake files. Keep rewrites narrow:

- Prefer current CUDA/PyTorch-compatible APIs over broad refactors.
- Preserve tensor shapes, dtype, device, boundary behavior, and reduction
  order where possible.
- Treat atomic operations, reductions, warp primitives, and deprecated CUDA
  headers as high-risk areas.
- Compile and test each operator independently before changing placement logic.
- Keep the original implementation available for an A/B comparison until the
  rewritten operator passes numerical tolerances.

## Correctness Gates

A CUDA 12 build is usable only if all of the following pass:

- DREAMPlace imports all compiled extensions without warnings that affect
  execution.
- CPU and GPU operator outputs agree within an explicitly recorded tolerance.
- One deterministic placement run produces finite HPWL, overflow, and
  coordinates.
- The same macro input produces comparable GP-HPWL/TNS/WNS to the baseline;
  small numerical drift must be documented rather than silently ignored.
- No illegal placement, NaN, CUDA launch failure, or intermittent crash is
  observed across at least three repeated runs.

## Performance Gates

Measure the following separately so CUDA compilation is not confused with
pipeline changes:

| Measurement | Required comparison |
| --- | --- |
| DREAMPlace initialization | cu118 vs CUDA 12 |
| 100/300/1000 global iterations | same input and parameters |
| legalization and OpenTimer | separately from global placement |
| peak GPU memory and utilization | evaluator process only |
| end-to-end EXPlace batch | with and without async evaluation |

Changing CUDA versions alone is not expected to fix Ray/state construction,
PPO transfer, OpenTimer CPU work, or repeated placement initialization. Warm
start, short screening runs, and an isolated evaluator process should be
benchmarked separately.

## Safe Build Template

Use the guarded build script and select the architecture for the actual GPU.
It refuses to run without an explicit or driver-detected compute capability:

```bash
cd /data0/cjy_cxsj/EXPlace-test/EXPlace

PYTHON_BIN=/path/to/cuda12-python \
CUDA_HOME=/usr/local/cuda-12.2 \
CUDA_ARCH=<actual_compute_capability> \
bash script/build_dreamplace_cuda12.sh
```

For compile diagnosis only, before a CUDA 12 PyTorch environment is available:

```bash
PYTHON_BIN=/usr/bin/python3 \
CUDA_HOME=/usr/local/cuda-12.2 \
CUDA_ARCH=<provisional_compute_capability> \
ALLOW_TORCH_CUDA_MISMATCH=1 \
ACTION=configure \
bash script/build_dreamplace_cuda12.sh
```

Every attempt is archived under `cuda12_runs/<timestamp>/` with exact commands,
metadata, logs, `SUCCESS`/`FAILED`, and `first_error.txt`. The install command
also repeats `--prefix DREAMPlace/install-cuda12`, so it cannot silently inherit
the source project's default `DREAMPlace/install` prefix.

Do not install this build over the reference `DREAMPlace/install` until the
correctness gates pass. In practice, a complete CUDA 12 migration should use a
PyTorch build whose CUDA version is compatible with the selected toolkit;
mixing a cu118 PyTorch binary with an unverified CUDA 12 extension build is
not an acceptable performance baseline.

## Next Experiment Record

When the work resumes, append one entry per attempt:

```text
Date:
Toolchain:
GPU/SM:
Operators changed:
First compiler error:
Correctness result:
100/300/1000 iteration timings:
Decision: keep / revise / revert
Artifacts:
```
