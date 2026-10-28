#!/bin/bash
# Limits are forced only in test builds; no assumption about host RAM size.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/memory_preflight
mkdir -p "$OUT"
NVCC_EXTRA=-DPUF_METAL_MEMORY_TEST \
    ./build.sh g2048 "$OUT/puffer" --metal --float > "$OUT/build.log" 2>&1
NVCC_EXTRA='-DPUF_METAL_MEMORY_TEST -DPUF_METAL_MEMORY_PREFLIGHT=0' \
    ./build.sh g2048 "$OUT/puffer_no_preflight" --metal --float \
    > "$OUT/mutant-build.log" 2>&1
python3 tests/metal/test_pack_eligibility.py
python3 tests/metal/test_memory_preflight.py "$OUT"
