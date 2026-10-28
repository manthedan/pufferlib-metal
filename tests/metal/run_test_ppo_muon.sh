#!/bin/bash
# Run from the worktree root. Raw CUDA FP32 golden inputs are read-only.
set -euo pipefail
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_ppo_muon
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_ppo_muon.mm src/metal/runtime.mm src/metal/launch.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation -framework Accelerate \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_ppo_muon"
"$OUT/test_ppo_muon" "$@"
