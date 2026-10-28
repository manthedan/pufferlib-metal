#!/bin/bash
# Exact 256-lane norm trees in both storage precisions.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
for P in f32 bf16; do
    OUT=build/metal/test_muon_tree_$P
    mkdir -p "$OUT"
    FLAGS=()
    if [ "$P" = f32 ]; then FLAGS+=(-DPRECISION_FLOAT); fi
    bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" "${FLAGS[@]}"
    clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
        -Isrc/metal -Isrc/metal/compat "${FLAGS[@]}" \
        tests/metal/test_muon_tree.mm src/metal/runtime.mm src/metal/gemm.mm \
        -framework Metal -framework Foundation \
        -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
        -o "$OUT/test_muon_tree"
    "$OUT/test_muon_tree" "$@"
done
