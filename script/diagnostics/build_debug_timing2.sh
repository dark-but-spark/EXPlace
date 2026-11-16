#!/usr/bin/env bash
# Build a symbol-rich (non-LTO) copy of the timing extension for gdb.
# The installed .so is built with -flto -fno-fat-lto-objects, which strips all
# symbols and debug info.  Here the three timing translation units are
# recompiled with -g -O2 and linked without LTO against the already-built
# libOpenTimer.a (which retains ~27k symbols).
set -uo pipefail
BASE=/data0/cjy_cxsj/EXPlace-test/EXPlace/DREAMPlace
BUILD=$BASE/build/dreamplace/ops/timing
SRCDIR=$BASE/dreamplace/ops/timing/src
OUT=$BASE/debug_timing
mkdir -p "$OUT"

cd "$BUILD" || exit 1
FLAGSMAKE=CMakeFiles/timing_cpp.dir/flags.make
CXX_DEFINES=$(grep -E '^CXX_DEFINES' "$FLAGSMAKE" | sed -e 's/^CXX_DEFINES *= *//')
CXX_INCLUDES=$(grep -E '^CXX_INCLUDES' "$FLAGSMAKE" | sed -e 's/^CXX_INCLUDES *= *//')
RAW_FLAGS=$(grep -E '^CXX_FLAGS' "$FLAGSMAKE" | sed -e 's/^CXX_FLAGS *= *//')
CLEAN_FLAGS=$(echo "$RAW_FLAGS" | sed -e 's/-flto//g' -e 's/-fno-fat-lto-objects//g')

echo "[dbg] defines: ${CXX_DEFINES:0:80}..."
echo "[dbg] flags: $CLEAN_FLAGS"

for src in timing_cpp timing_io_cpp timing_pybind; do
    echo "  - compiling $src.cpp"
    /usr/bin/c++ $CXX_DEFINES $CXX_INCLUDES $CLEAN_FLAGS -g -O2 \
        -c "$SRCDIR/$src.cpp" -o "$OUT/$src.o" || { echo "[dbg] compile failed: $src"; exit 1; }
done

echo "[dbg] linking"
LINK=$(cat CMakeFiles/timing_cpp.dir/link.txt)
LINK=${LINK//-flto/}
for src in timing_cpp timing_io_cpp timing_pybind; do
    LINK=${LINK//CMakeFiles\/timing_cpp.dir\/src\/$src.cpp.o/$OUT\/$src.o}
done
LINK=${LINK//-o timing_cpp.cpython-310-x86_64-linux-gnu.so/-o $OUT\/timing_cpp_debug.so}
eval "$LINK" || { echo "[dbg] link failed"; exit 1; }

echo "[dbg] result:"
ls -la "$OUT/timing_cpp_debug.so"
echo -n "[dbg] debug sections: "; readelf -S "$OUT/timing_cpp_debug.so" 2>/dev/null | grep -c debug_info
echo -n "[dbg] ot:: symbols: "; nm -C "$OUT/timing_cpp_debug.so" 2>/dev/null | grep -c "ot::"
echo "[dbg] key symbols:"
nm -C "$OUT/timing_cpp_debug.so" 2>/dev/null | grep -E "T ot::(Timer::update_timing|RCTree|Timer::report_wns)" | head -6
