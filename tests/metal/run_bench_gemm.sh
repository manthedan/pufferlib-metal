#!/bin/bash
# Build and run the GEMM microbenchmark. Run from the repo root:
#   tests/metal/run_bench_gemm.sh M N K ta tb [iters] [bf16]
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/bench_gemm
mkdir -p "$OUT"
if [ ! -x "$OUT/bench_gemm" ] || [ -n "$(find src/metal tests/metal/bench_gemm.mm -newer "$OUT/bench_gemm" 2>/dev/null)" ]; then
    bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
    clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
        -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
        tests/metal/bench_gemm.mm src/metal/runtime.mm src/metal/gemm.mm \
        -framework Metal -framework Foundation \
        -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
        -o "$OUT/bench_gemm"
fi
"$OUT/bench_gemm" "$@"
