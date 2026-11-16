#!/usr/bin/env bash
# Build a debug-info copy of the timing extension (OpenTimer + timing_cpp).
# Does NOT install: the installed .so stays untouched.
set -uo pipefail
cd /data0/cjy_cxsj/EXPlace-test/EXPlace/DREAMPlace/build || exit 1

BACKUP=/data0/cjy_cxsj/EXPlace-test/EXPlace/DREAMPlace/timing_cpp_orig.so
SO=dreamplace/ops/timing/timing_cpp.cpython-310-x86_64-linux-gnu.so

if [ ! -f "$BACKUP" ]; then
  cp -v "$SO" "$BACKUP" || exit 1
fi

echo "[build] reconfiguring with -g (Release + debug info)"
cmake -DCMAKE_CXX_FLAGS="-g" . > /tmp/cmake_g.log 2>&1 || { echo "cmake failed"; tail -20 /tmp/cmake_g.log; exit 1; }

echo "[build] rebuilding timing_cpp (this pulls OpenTimer)"
START=$(date +%s)
make -j24 timing_cpp > /tmp/build_timing_g.log 2>&1
RC=$?
END=$(date +%s)
echo "[build] make rc=$RC elapsed=$((END-START))s"

if [ $RC -ne 0 ]; then
  echo "[build] tail of log:"
  tail -30 /tmp/build_timing_g.log
  exit $RC
fi

echo "[build] resulting .so:"
ls -la "$SO"
file "$SO" | sed 's/.*: //'
echo -n "[build] symbol count: "; nm "$SO" 2>/dev/null | wc -l
echo -n "[build] debug sections: "; readelf -S "$SO" 2>/dev/null | grep -c debug_info
echo -n "[build] opentimer symbols: "; nm -C "$SO" 2>/dev/null | grep -cE "ot::"
echo "[build] sample symbols:"
nm -C "$SO" 2>/dev/null | grep -E "ot::Timer::(update_timing|report_wns)" | head -5
