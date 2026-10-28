#!/bin/bash
# Build and run the rollout-step microbenchmark. Run from the repo root:
#   tests/metal/run_bench_rollout_step.sh [iters]          # BF16
#   PRECISION=float tests/metal/run_bench_rollout_step.sh  # FP32
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
if [ "${PRECISION:-bf16}" = float ]; then
    OUT=build/metal/bench_rollout_step_f32
    DEFS=-DPRECISION_FLOAT
else
    OUT=build/metal/bench_rollout_step
    DEFS=
fi
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" $DEFS
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat $DEFS \
    tests/metal/bench_rollout_step.mm src/metal/runtime.mm src/metal/launch.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/bench_rollout_step"
"$OUT/bench_rollout_step" "$@"
