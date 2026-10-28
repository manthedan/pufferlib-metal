#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
for e in tetris wef; do
 for p in bf16 f32; do
  d=build/metal/large_e2e/${e}_${p}; mkdir -p "$d"
  flag=; [ "$p" = f32 ] && flag=--float
  ./build.sh "$e" "$d/puffer" --metal $flag > "$d/build.log" 2>&1
  python3 tests/metal/large_e2e.py "$d/puffer" "$e"
 done
done
