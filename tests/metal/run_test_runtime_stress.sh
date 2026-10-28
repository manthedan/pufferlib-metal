#!/bin/bash
# Build and run the Metal runtime multi-thread stress test. Run from the repo root:
#   tests/metal/run_test_runtime_stress.sh [seconds=20] [rollout_threads=3] [stall_seconds=120]
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_runtime_stress
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_runtime_stress.mm src/metal/runtime.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_runtime_stress"
"$OUT/test_runtime_stress" "$@"
