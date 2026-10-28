#!/bin/bash
# Compile the primary library and a separate *.rollout.metallib for grouped
# rollout entry points. Keep the primary input list identical to the original
# backend: unused grouped kernels must not perturb single-buffer pipelines.
# Usage: tools/metal/build_metallib.sh OUT.metallib [-DPRECISION_FLOAT ...]
set -e
OUT=$1
shift
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
AIR_DIR="$(dirname "$OUT")/air"
mkdir -p "$AIR_DIR"
AIRS=()
for src in src/metal/shaders/*.metal; do
    case "$(basename "$src")" in rollout_group.metal|wide.metal) continue ;; esac
    air="$AIR_DIR/$(basename "${src%.metal}").air"
    # Safe math: kernels rely on NaN/Inf semantics (isnan/isinf guards), like
    # nvcc without --use_fast_math.
    xcrun -sdk macosx metal -std=metal4.0 -O2 \
        -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
        -Isrc/metal/shaders "$@" -c "$src" -o "$air"
    AIRS+=("$air")
done
xcrun -sdk macosx metallib "${AIRS[@]}" -o "$OUT"

xcrun -sdk macosx metal -std=metal4.0 -O2 \
    -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
    -Isrc/metal/shaders "$@" -c src/metal/shaders/rollout_group.metal \
    -o "$AIR_DIR/rollout_group.air"
xcrun -sdk macosx metallib "$AIR_DIR/rollout_group.air" \
    -o "${OUT%.metallib}.rollout.metallib"

# Wide learner kernels are loaded only by wide learner paths.
xcrun -sdk macosx metal -std=metal4.0 -O2 \
    -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
    -Isrc/metal/shaders "$@" -c src/metal/shaders/wide.metal \
    -o "$AIR_DIR/wide.air"
xcrun -sdk macosx metallib "$AIR_DIR/wide.air" -o "${OUT%.metallib}.wide.metallib"
