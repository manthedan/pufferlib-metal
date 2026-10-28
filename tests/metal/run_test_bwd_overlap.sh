#!/bin/bash
# Real algo.cu backward + pufferl.cu policy, with test-only GEMM observation.
# Keep both precisions in the existing GEMM gate.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
unset PUF_METAL_BWD_OVERLAP PUF_METAL_DW_SIDE
unset PUF_METAL_FUSED_MINGRU PUF_METAL_FUSED_MINGRU64 PUF_METAL_TOPS_TN_SPLIT
OMP_PREFIX=$(brew --prefix libomp)
fail=0
for precision in f32 bf16; do
    OUT=build/metal/test_bwd_overlap_$precision
    mkdir -p "$OUT"
    PRECISION=()
    if [ "$precision" = f32 ]; then PRECISION=(-DPRECISION_FLOAT); fi
    bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" "${PRECISION[@]}"
    clang++ -std=c++17 -O2 -Wno-narrowing -Wno-c99-designator \
        -Wno-writable-strings -Wno-missing-braces -Wno-logical-op-parentheses \
        -Wno-vla-cxx-extension \
        -Isrc/metal/compat -Isrc/metal -I. -Isrc -Ivendor \
        -Iraylib-5.5_macos/include -I"$OMP_PREFIX/include" -Xclang -fopenmp \
        -DENV_HEADER='"ocean/breakout/breakout.h"' -DENV_NAME=breakout \
        -DPUFFER_ENV_NAME='"breakout"' -DPUFFER_BREAKOUT -DPLATFORM_DESKTOP \
        -DPUFFERLIB_BUILD_MAIN "${PRECISION[@]}" \
        -c tests/metal/test_bwd_overlap.cpp -o "$OUT/test.o"
    for source in runtime gemm launch; do
        clang++ -std=c++17 -O2 -fobjc-arc -Isrc/metal/compat -Isrc/metal \
            "${PRECISION[@]}" -c "src/metal/$source.mm" -o "$OUT/$source.o"
    done
    clang++ "$OUT/test.o" "$OUT/runtime.o" "$OUT/gemm.o" "$OUT/launch.o" \
        raylib-5.5_macos/lib/libraylib.a -L"$OMP_PREFIX/lib" -lomp -lm -lpthread \
        -framework Metal -framework Foundation -framework Accelerate -framework Cocoa -framework IOKit \
        -framework CoreVideo -framework OpenGL \
        -Wl,-sectcreate,__PUF_METAL,__metallib,"$OUT/puffer.metallib" \
        -Wl,-sectcreate,__PUF_METAL,__rollgroup,"$OUT/puffer.rollout.metallib" \
        -o "$OUT/test_bwd_overlap"
    echo "== $precision"
    if "$OUT/test_bwd_overlap"; then :; else fail=1; fi
done
exit "$fail"
