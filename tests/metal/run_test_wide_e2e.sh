#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
cd "$(dirname "$0")/../.."
for e in freeway breakout; do
 args=(); steps=2097152
 if [ "$e" = freeway ]; then steps=4194304; fi
 # Always exercise the tracked dense Breakout policy at H256.
 if [ "$e" = breakout ]; then
    args=(--extra=--policy.hidden-size=256 --extra=--policy.num-layers=2 --extra=--train.horizon=64 --extra=--train.minibatch-size=8192)
 fi
 for p in bf16 f32; do
    d=build/metal/wide_e2e/${e}_${p}; mkdir -p "$d"
    flag=; [ "$p" = f32 ] && flag=--float
    ./build.sh "$e" "$d/puffer" --metal $flag > "$d/build.log" 2>&1
    python3 tests/metal/wide_e2e.py "$d/puffer" "$e" --steps="$steps" "${args[@]}"
 done
done
