#!/bin/bash
# Build and run the Metal GEMM test. Run from the repo root:
#   tests/metal/run_test_gemm.sh [GOLDEN_RUN_DIR ...]
# e.g. tests/metal/run_test_gemm.sh ../golden/run_a
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_gemm
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_gemm.mm src/metal/runtime.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation -framework Accelerate \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_gemm"
"$OUT/test_gemm" "$@"

bash tests/metal/run_test_bwd_overlap.sh
