#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
for p in f32 bf16; do
    OUT=build/metal/test_wide_muon_$p
    mkdir -p "$OUT"
    flag=; test "$p" = f32 && flag=-DPRECISION_FLOAT
    bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" $flag
    clang++ -std=c++17 -O2 -fobjc-arc -Wall $flag -Isrc/metal -Isrc/metal/compat \
        tests/metal/test_wide_muon.mm src/metal/runtime.mm src/metal/gemm.mm \
        -framework Metal -framework Foundation \
        -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" -o "$OUT/test_wide_muon"
    "$OUT/test_wide_muon"
    python3 tests/metal/check_wide_muon_dispatch.py "$OUT/test_wide_muon"
done
