# CUDA 12 Host Preflight: 2026-09-21

`host_preflight.txt` is a current-state snapshot, not metadata for the
2026-09-20 training run. It establishes:

- CUDA 12.2 compiler toolkit is installed at `/usr/local/cuda-12.2`.
- The host Python environment uses PyTorch `2.0.0+cu118`, not a CUDA 12 build.
- NVIDIA PCI devices are present, but `nvidia-smi` fails and PyTorch sees zero
  CUDA devices.
- The preserved DREAMPlace reference build uses CUDA 11.8, ABI 0, and installs
  to `DREAMPlace/install`.
- At capture time no `DREAMPlace/build-cuda12` or `install-cuda12` result had
  passed any gate.

Regenerate the snapshot with:

```bash
cd /data0/cjy_cxsj/EXPlace-test/EXPlace
NVCC_BIN=/usr/local/cuda-12.2/bin/nvcc \
PYTHON_BIN=/usr/bin/python3 \
bash script/capture_run_metadata.sh \
  experiment_records/cuda12_20260921/host_preflight.txt
```

The PCI IDs are useful for diagnosis but do not prove which physical device is
CUDA index 0. A native architecture must be confirmed after the driver is
restored before correctness or performance measurements are accepted.
