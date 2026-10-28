#!/bin/bash
# BF16 fused learner vs generic BF16, exact except the H=32 dW reduction order.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/test_mingru_bf16
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib"
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat \
    tests/metal/test_mingru_bf16.mm src/metal/runtime.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_mingru_bf16"
PUF_METAL_FUSED_MINGRU=1 "$OUT/test_mingru_bf16" "$@"
PUF_METAL_FUSED_MINGRU=1 PUF_METAL_MINGRU_FWD32=0 "$OUT/test_mingru_bf16" "$@"
PUF_METAL_FUSED_MINGRU=1 PUF_METAL_FUSED_MINGRU64=0 "$OUT/test_mingru_bf16" --no64
PUF_METAL_FUSED_MINGRU=0 "$OUT/test_mingru_bf16" --disabled
