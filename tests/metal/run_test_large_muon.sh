#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
for p in f32 bf16; do
 OUT=build/metal/test_large_muon_$p; mkdir -p "$OUT"
 flag=; [ "$p" = f32 ] && flag=-DPRECISION_FLOAT
 bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" $flag
 clang++ -std=c++17 -O2 -fobjc-arc -Wall $flag -Isrc/metal -Isrc/metal/compat \
  tests/metal/test_large_muon.mm src/metal/runtime.mm src/metal/gemm.mm \
  -framework Metal -framework Foundation \
  -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" -o "$OUT/test_large_muon"
 PUF_METAL_MUON_LARGE=0 PUF_METAL_WIDE_LIB=/nonexistent/wide.metallib "$OUT/test_large_muon" --disabled
 python3 tests/metal/check_large_muon_dispatch.py "$OUT/test_large_muon" --log "$OUT/dispatch.log"
done
