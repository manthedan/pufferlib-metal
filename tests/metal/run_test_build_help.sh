#!/bin/bash
# Check the user-visible precision contract and prove the assertion rejects the
# former "Metal is FP32" wording.
set -e
OUT=build/metal/build_help
mkdir -p "$OUT"

check_help() {
    file=$1
    grep -- '--metal: native Apple Metal trainer (macOS; BF16 by default)' "$file" >/dev/null
    grep -- '--float: use FP32 instead of the default BF16 trainer precision' "$file" >/dev/null
    if grep -E -- '(--metal.*FP32 only|--metal.*macOS, FP32)' "$file" >/dev/null; then
        echo "stale Metal precision help" >&2
        return 1
    fi
}

if ./build.sh > "$OUT/help.txt" 2>&1; then
    echo "FAIL: build.sh without ENV unexpectedly succeeded" >&2
    exit 1
fi
check_help "$OUT/help.txt"

sed 's/BF16 by default/FP32 only/' "$OUT/help.txt" > "$OUT/mutant-help.txt"
if check_help "$OUT/mutant-help.txt" 2> "$OUT/mutant-check.log"; then
    echo "FAIL: stale-help mutant escaped detection" >&2
    exit 1
fi

check_help "$OUT/help.txt"
echo "build_help: production PASS; stale-help mutant CAUGHT; restored PASS"

# all must not silently drop Metal mode or precision and invoke CUDA.
for precision in default float; do
    args=(all --metal)
    [ "$precision" = default ] || args+=(--float)
    if ./build.sh "${args[@]}" > "$OUT/all-$precision.txt" 2>&1; then
        echo 'FAIL: unsupported all --metal unexpectedly succeeded' >&2
        exit 1
    fi
    grep 'all --metal is unsupported' "$OUT/all-$precision.txt" >/dev/null
done
echo 'build_help: all --metal rejects both precision requests before any build PASS'
