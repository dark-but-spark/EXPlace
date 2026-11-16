#!/usr/bin/env bash
# Validate the dump -> replay -> gdb path on a known-good dump.
set -uo pipefail
cd /data0/cjy_cxsj/EXPlace-test/EXPlace || exit 1
source script/activate_cuda118.sh >/dev/null 2>&1
export MPLCONFIGDIR=/tmp
ulimit -c unlimited

POS=${1:-long_runs/repro_validate/dumps/pre_timing_pos_000.npy}
GPU=${2:-6}
OUT=${3:-long_runs/repro_validate/gdb_validate.txt}

gdb --batch \
    -ex 'set pagination off' \
    -ex run \
    -ex 'echo \n===== BACKTRACE =====\n' \
    -ex 'bt 25' \
    --args /usr/bin/python3 script/repro_timer_pos.py \
        --benchmark superblue3 --config adaptive_global --pos "${POS}" --gpu "${GPU}" \
    > "${OUT}" 2>&1
rc=$?
echo "gdb exit=${rc}"
grep -E "\[repro\]|Segmentation|SIGSEGV|Program received|===== BACKTRACE" "${OUT}" | head -20
