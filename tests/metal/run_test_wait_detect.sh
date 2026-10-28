#!/bin/bash
# Build and run the Metal runtime queue-wait timeout detection test. Run from the repo root:
#   tests/metal/run_test_wait_detect.sh [control|sync|device|late|chain ...]
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
OUT=build/metal/test_wait_detect
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" -DPRECISION_FLOAT
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat -DPRECISION_FLOAT \
    tests/metal/test_wait_detect.mm src/metal/runtime.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_wait_detect"
CASES="$*"
[ -n "$CASES" ] || CASES="control sync device late chain lazy"
rc=0
for c in $CASES; do
    echo "== $c"
    "$OUT/test_wait_detect" "$c" || rc=1
done
[ $rc -eq 0 ] && echo "wait_detect: ALL CASES PASS" || echo "wait_detect: FAILED"
exit $rc
