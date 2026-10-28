#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/test_wide_gemm
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_wide_gemm.mm src/metal/runtime.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" -o "$OUT/test_wide_gemm"
"$OUT/test_wide_gemm" "$@"
PUF_METAL_WIDE_LIB=/nonexistent/wide.metallib PUF_METAL_GEMM_WIDE_TN=0 PUF_METAL_GEMM_RAGGED=0 "$OUT/test_wide_gemm"
