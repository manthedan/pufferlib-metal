#!/bin/bash
# Poisoned custom-encoder BF16 primitives against exact/fp64 math.
set -e
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/test_env_bf16
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib"
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat \
    tests/metal/test_env_bf16.mm src/metal/runtime.mm src/metal/launch.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" -o "$OUT/test_env_bf16"
"$OUT/test_env_bf16"
