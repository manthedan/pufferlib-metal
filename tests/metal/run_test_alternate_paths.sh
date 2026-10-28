#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
suite=${1:?math or schedule}
case "$suite" in math|schedule) ;; *) exit 2 ;; esac
mkdir -p build/metal
OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/alternate-$suite.XXXXXX")
for precision in bf16 f32; do
    flag=(); [ "$precision" = bf16 ] || flag=(--float)
    ./build.sh cartpole "$OUT/trainer" --metal ${flag[@]+"${flag[@]}"} > "$OUT/$precision-build.log" 2>&1
    python3 tests/metal/test_alternate_paths.py "$suite" "$OUT" "$OUT/trainer" "$precision"
    rm "$OUT/trainer"
done
if [ "$suite" = schedule ]; then
    # Compile an isolated test-only mutation that leaves strict-mode gating
    # disabled but skips the strict-feedback join itself. The positive counter
    # assertion above must reject it after both training processes succeed.
    CXX="clang++ -DPUF_METAL_MUTATE_SKIP_STRICT_FEEDBACK" \
        ./build.sh cartpole "$OUT/strict-mutation" --metal > "$OUT/strict-mutation-build.log" 2>&1
    status=0
    python3 tests/metal/test_alternate_paths.py strict "$OUT/mutation" \
        "$OUT/strict-mutation" bf16 > "$OUT/strict-mutation-check.log" 2>&1 || status=$?
    rm "$OUT/strict-mutation"
    if [ "$status" != 1 ] || ! grep -q 'strict feedback join counter stayed zero' \
            "$OUT/strict-mutation-check.log"; then
        echo "strict feedback mutation was not rejected; see $OUT/strict-mutation-check.log" >&2
        exit 1
    fi
    echo "PASS strict-feedback skipped-join mutation rejected"
fi
