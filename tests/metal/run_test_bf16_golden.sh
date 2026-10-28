#!/bin/bash
# BF16 learner golden gate: Metal bf16 kernels vs CUDA BF16 goldens, plus the
# bf16 self-consistency test (exact GEMM/cast/Muon/advantage identities).
# Run from the repo root:
#   tests/metal/run_test_bf16_golden.sh GOLDEN_BF16_RUN_DIR ...
# e.g. tests/metal/run_test_bf16_golden.sh ../golden/run_bf16
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_bf16_golden
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib"
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat \
    tests/metal/test_bf16_golden.mm src/metal/runtime.mm src/metal/launch.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_bf16_golden"
"$OUT/test_bf16_golden" "$@"
