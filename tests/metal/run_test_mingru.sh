#!/bin/bash
# Build and run the Metal MinGRU scan / policy-backward test. Run from the repo root:
#   tests/metal/run_test_mingru.sh [GOLDEN_RUN_DIR ...] [--bench]
# e.g. tests/metal/run_test_mingru.sh ../golden/run_a --bench
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_mingru
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_mingru.mm src/metal/runtime.mm src/metal/launch.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation -framework Accelerate \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_mingru"
"$OUT/test_mingru" "$@"
