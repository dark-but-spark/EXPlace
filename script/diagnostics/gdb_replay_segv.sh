#!/usr/bin/env bash
# Swap in the symbol-rich timing extension, run one replay under gdb, restore.
# Usage: gdb_replay.sh <pos.npy> <gpu> <out.txt>
set -uo pipefail
cd /data0/cjy_cxsj/EXPlace-test/EXPlace || exit 1
source script/activate_cuda118.sh >/dev/null 2>&1
export MPLCONFIGDIR=/tmp
ulimit -c unlimited

POS=${1:?pos npy}
GPU=${2:-7}
OUT=${3:-long_runs/repro_segv_gdb.txt}

SO=dreamplace/ops/timing/timing_cpp.cpython-310-x86_64-linux-gnu.so
INSTALL_DIR=DREAMPlace/install
DBG=DREAMPlace/debug_timing/timing_cpp_debug.so
BACKUP=DREAMPlace/debug_timing/timing_cpp_prod_backup.so

if [ ! -f "$BACKUP" ]; then
  cp -v "$INSTALL_DIR/$SO" "$BACKUP" || exit 1
fi

restore() {
  cp -f "$BACKUP" "$INSTALL_DIR/$SO"
  echo "[gdb] restored production .so"
}
trap restore EXIT

cp -f "$DBG" "$INSTALL_DIR/$SO" || exit 1
echo "[gdb] swapped in debug build ($(stat -c%s "$INSTALL_DIR/$SO") bytes)"

echo "[gdb] replaying $POS on gpu $GPU"
gdb --batch \
    -ex 'set pagination off' \
    -ex 'set print frame-arguments none' \
    -ex run \
    -ex 'echo \n===== SIGNAL =====\n' \
    -ex 'info program' \
    -ex 'echo \n===== BACKTRACE =====\n' \
    -ex 'bt 40' \
    -ex 'echo \n===== FRAME 0 =====\n' \
    -ex 'frame 0' \
    -ex 'info locals' \
    --args /usr/bin/python3 script/repro_timer_pos.py \
        --benchmark superblue3 --config adaptive_global --pos "$POS" --gpu "$GPU" \
    > "$OUT" 2>&1
echo "[gdb] exit=$? output=$OUT"
