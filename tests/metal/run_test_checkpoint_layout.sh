#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/checkpoint_layout
mkdir -p "$OUT"
for precision in bf16 f32; do
    flag=; [ "$precision" = f32 ] && flag=--float
    NVCC_EXTRA=-DPUF_WARM_START_TEST ./build.sh drone "$OUT/$precision" --metal $flag > "$OUT/$precision-build.log" 2>&1
done
NVCC_EXTRA='-DPUF_WARM_START_TEST -DPUF_TEST_SKIP_CHECKPOINT_LAYOUT' \
    ./build.sh drone "$OUT/mutant" --metal > "$OUT/mutant-build.log" 2>&1
python3 tests/metal/test_checkpoint_layout.py "$OUT"
python3 tests/metal/test_frozen_cache.py
bash tests/metal/run_test_frozen_nethack.sh
