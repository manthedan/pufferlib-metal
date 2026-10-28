#!/bin/bash
# Build and run the Metal rollout-kernel test. Run from the repo root:
#   tests/metal/run_test_kernels.sh [GOLDEN_RUN_DIR ...]
# e.g. tests/metal/run_test_kernels.sh ../golden/run_a
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_kernels
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_kernels.mm src/metal/runtime.mm src/metal/launch.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_kernels"
"$OUT/test_kernels" "$@"
