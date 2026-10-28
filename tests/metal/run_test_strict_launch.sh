#!/bin/bash
# Compile/link the standard trainer; never launch training in this gate.
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
mkdir -p build/metal
OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/strict_launch.XXXXXX")
for precision in bf16 f32; do
    flag=(); [ "$precision" = bf16 ] || flag=(--float)
    for env in breakout pong nethack drone nmmo3; do
        NVCC_EXTRA=-DPUF_METAL_STRICT_LAUNCH ./build.sh "$env" "$OUT/trainer" --metal ${flag[@]+"${flag[@]}"} > "$OUT/$env-$precision.log" 2>&1
        test -x "$OUT/trainer"
        rm "$OUT/trainer"
        echo "PASS strict launch compile/link: $env $precision"
    done
done
