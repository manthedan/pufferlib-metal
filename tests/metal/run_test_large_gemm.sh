#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/test_large_gemm
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_large_gemm.mm src/metal/runtime.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" -o "$OUT/test_large_gemm"
"$OUT/test_large_gemm" "$@"
if [ $# = 0 ]; then
 PUF_METAL_WIDE_LIB=/nonexistent/wide.metallib PUF_METAL_GEMM_LARGE_TN=0 "$OUT/test_large_gemm"
fi
