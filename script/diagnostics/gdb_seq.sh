#!/usr/bin/env bash
# Replay a dump (single or sequence) with the debug timing build, under gdb.
# Uses DREAMPlace/install_dbg so the production install is never touched.
#   gdb_seq.sh <dump-dir|pos.npy> <gpu> <out.txt> [seq]
set -uo pipefail
cd /data0/cjy_cxsj/EXPlace-test/EXPlace || exit 1
source script/activate_cuda118.sh >/dev/null 2>&1
export MPLCONFIGDIR=/tmp
export PYTHONPATH="${PWD}/DREAMPlace/install_dbg:${PWD}/DREAMPlace/install_dbg/dreamplace${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c unlimited

TARGET=${1:?dump dir or npy}
GPU=${2:-7}
OUT=${3:-long_runs/gdb_replay.txt}
MODE=${4:-seq}

if [ "$MODE" = "seq" ]; then
  ARGS=(--benchmark superblue3 --config adaptive_global --dump-dir "$TARGET" --gpu "$GPU")
  SCRIPT=script/repro_timer_seq.py
else
  ARGS=(--benchmark superblue3 --config adaptive_global --pos "$TARGET" --gpu "$GPU")
  SCRIPT=script/repro_timer_pos.py
fi

echo "[gdb] PYTHONPATH head: ${PYTHONPATH%%:*}"
echo "[gdb] mode=$MODE target=$TARGET gpu=$GPU"
gdb --batch \
    -ex 'set pagination off' \
    -ex 'set print frame-arguments none' \
    -ex run \
    -ex 'echo \n===== SIGNAL =====\n' \
    -ex 'info program' \
    -ex 'echo \n===== BACKTRACE =====\n' \
    -ex 'bt 45' \
    -ex 'echo \n===== FRAME 0-3 =====\n' \
    -ex 'frame 0' -ex 'info args' -ex 'info locals' \
    -ex 'frame 1' -ex 'info args' -ex 'info locals' \
    -ex 'frame 2' -ex 'info args' \
    -ex 'frame 3' -ex 'info args' \
    --args /usr/bin/python3 "$SCRIPT" "${ARGS[@]}" \
    > "$OUT" 2>&1
echo "[gdb] exit=$? out=$OUT"
