#!/bin/bash
# A Metal headless evaluation must identify its backend. The second build
# supplies the legacy CUDA token as an isolated mutant; the same check must
# reject it without modifying production source.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/eval_label
mkdir -p "$OUT"

check_label() {
    binary=$1
    log=$2
    "$binary" eval --headless --base.eval-episodes=1 > "$log" 2>&1
    grep '^METAL_EVAL env=cartpole ' "$log" >/dev/null
    if grep '^CUDA_EVAL ' "$log" >/dev/null; then
        echo "FAIL: Metal evaluation used the CUDA label" >&2
        return 1
    fi
}

./build.sh cartpole "$OUT/puffer" --metal > "$OUT/build.log" 2>&1
check_label "$OUT/puffer" "$OUT/eval.log"

NVCC_EXTRA=-DPUF_EVAL_BACKEND=CUDA \
    ./build.sh cartpole "$OUT/puffer_legacy_mutant" --metal \
    > "$OUT/mutant-build.log" 2>&1
if check_label "$OUT/puffer_legacy_mutant" "$OUT/mutant-eval.log" \
        2> "$OUT/mutant-check.log"; then
    echo "FAIL: legacy-label mutant escaped detection" >&2
    exit 1
fi

check_label "$OUT/puffer" "$OUT/restored-eval.log"
echo "eval_label: production PASS; legacy-label mutant CAUGHT; restored PASS"
