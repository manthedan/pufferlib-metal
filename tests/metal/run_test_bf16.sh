#!/bin/bash
# Build and run the BF16 self-consistency test (bf16 metallib). Run from the
# repo root: tests/metal/run_test_bf16.sh
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_bf16
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib"
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat \
    tests/metal/test_bf16.mm src/metal/runtime.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_bf16"
"$OUT/test_bf16" "$@"

# Independent exact norm-tree oracle in FP32 and BF16, including wide stages.
tests/metal/run_test_muon_tree.sh
