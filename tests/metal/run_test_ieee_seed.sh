#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
mkdir -p build/metal
OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/ieee_seed.XXXXXX")
export PUFFER_REAL_XCRUN=$(command -v xcrun)
export PUFFER_VARIANT_BUILD_LOG="$OUT/variant-shaders.log"
mkdir "$OUT/toolchain"
ln -s "$PWD/tests/metal/shader_variant_xcrun.py" "$OUT/toolchain/xcrun"
for precision in bf16 f32; do
    flag=(); [ "$precision" = bf16 ] || flag=(--float)
    ./build.sh cartpole "$OUT/default-$precision" --metal ${flag[@]+"${flag[@]}"} > "$OUT/default-$precision-build.log" 2>&1
    PATH="$OUT/toolchain:$PATH" NVCC_EXTRA=-DPUF_SIG_IEEE_DIV \
        ./build.sh cartpole "$OUT/ieee-$precision" --metal ${flag[@]+"${flag[@]}"} > "$OUT/ieee-$precision-build.log" 2>&1
    python3 tests/metal/test_alternate_paths.py ieee "$OUT" "$OUT/default-$precision" "$precision" "$OUT/ieee-$precision"
    rm "$OUT/default-$precision" "$OUT/ieee-$precision"
done
# The wrapper must have modified every sigmoid-bearing shader build.
for shader in mingru rollout rollout_forward rollout_group wide; do
    count=$(grep -c "/$shader.metal .* -DPUF_SIG_IEEE_DIV$" "$PUFFER_VARIANT_BUILD_LOG" || true)
    if [ "$count" != 2 ]; then
        echo "FAIL: expected two IEEE shader compiles for $shader, got $count" >&2
        exit 1
    fi
done
