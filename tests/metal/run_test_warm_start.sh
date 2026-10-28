#!/bin/bash
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/warm_start
mkdir -p "$OUT"

NVCC_EXTRA=-DPUF_WARM_START_TEST \
    ./build.sh cartpole "$OUT/puffer" --metal > "$OUT/build.log" 2>&1
NVCC_EXTRA='-DPUF_WARM_START_TEST -DPUF_TEST_SKIP_TRAIN_WARM_START' \
    ./build.sh cartpole "$OUT/puffer_skip_load" --metal \
    > "$OUT/mutant-build.log" 2>&1
python3 tests/metal/test_warm_start.py \
    "$OUT/puffer" "$OUT/puffer_skip_load" "$OUT"
