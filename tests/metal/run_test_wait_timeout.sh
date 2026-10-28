#!/bin/bash
# Build and run the MTL4 queue-wait timeout probe. Run from the repo root:
#   tests/metal/run_test_wait_timeout.sh [cpu [limit_s=120] | gpu [total_s=10]]
# Bounded: the CPU releases the withheld producer events after the limit.
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_wait_timeout
mkdir -p "$OUT"
clang++ -std=c++17 -O2 -fobjc-arc -Wall tests/metal/test_wait_timeout.mm \
    -framework Metal -framework Foundation -o "$OUT/test_wait_timeout"
"$OUT/test_wait_timeout" "$@"
