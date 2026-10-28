#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
unset PUF_PROTEIN_SOBOL_TABLE
OUT=build/metal/protein/public
mkdir -p "$OUT"
clang++ -std=c++17 -O2 -Wall -Wno-unused-function -Wno-unused-variable \
    -Wno-vla-cxx-extension -Isrc tests/metal/test_protein_public.cpp \
    -framework Accelerate -o "$OUT/test"
python3 tests/metal/test_protein_public.py "$OUT/test"
./build.sh cartpole "$OUT/puffer" --metal
python3 tests/metal/test_sweep_cli.py "$OUT/puffer" "$OUT/sweep" 2
