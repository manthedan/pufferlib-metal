#!/bin/bash
# Real warm-start/cache smoke; the full runner already builds fast-nle goldens.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/frozen_nethack
mkdir -p "$OUT"
NVCC_EXTRA='-DPUF_WARM_START_TEST -DNH_MSG_FROZEN' \
    ./build.sh nethack "$OUT/puffer" --metal --float > "$OUT/build.log" 2>&1
python3 tests/metal/test_frozen_nethack.py "$OUT"
