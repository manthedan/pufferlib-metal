#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
for p in f32 bf16; do
    OUT=build/metal/test_wide_$p
    mkdir -p "$OUT"
    flag=; test "$p" = f32 && flag=-DPRECISION_FLOAT
    bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" $flag
    clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function $flag \
        -Isrc/metal -Isrc/metal/compat \
        tests/metal/test_wide_mingru.mm src/metal/runtime.mm src/metal/gemm.mm \
        -framework Metal -framework Foundation \
        -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
        -o "$OUT/test_wide_mingru"
    PUF_METAL_WIDE_LIB=/nonexistent/wide.metallib PUF_METAL_FUSED_MINGRU_WIDE=0 "$OUT/test_wide_mingru" --disabled
    "$OUT/test_wide_mingru" "$@"
done

# The auxiliary library has an independent precision marker.
if PUF_METAL_WIDE_LIB="$PWD/build/metal/test_wide_bf16/puffer.wide.metallib" \
    build/metal/test_wide_f32/test_wide_mingru --check 128 > build/metal/test_wide_f32/wrong_precision.log 2>&1; then
    echo "FAIL: wrong-precision wide library accepted"; exit 1
fi
grep -q 'wide learner library was not compiled for this binary' build/metal/test_wide_f32/wrong_precision.log
