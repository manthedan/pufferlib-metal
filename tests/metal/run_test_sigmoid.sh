#!/bin/bash
# Exhaustive corrected-sigmoid check (puf_sigmoid_exact, common.h): all 2^32 fp32
# inputs, each shader translation unit's shipped sigmoid vs the original
# IEEE-division formula (<= 1 ULP, or bit equality with PUF_SIG_REQUIRE_EXACT=1),
# plus a rounding/FTZ-aware CPU oracle on a strided sample.
# See tests/metal/test_sigmoid.mm. --quick checks 5 of the
# 16 batches. Exact-mode mutations: --mutate (residual from a rounded product
# instead of an fma), --mutate2 (no correction step). --mutate3 injects exactly
# 2 ULP at the two signed-zero bit patterns and must fail with either tolerance.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/test_sigmoid
mkdir -p "$OUT"
MUT=() ARGS=()
for a in "$@"; do
    case "$a" in
        --mutate) MUT=(-DPUF_SIG_MUTATE=1) ;;
        --mutate2) MUT=(-DPUF_SIG_MUTATE=2) ;;
        --mutate3) MUT=(-DPUF_SIG_MUTATE=3) ;;
        -D*) MUT+=("$a") ;;
        *) ARGS+=("$a") ;;
    esac
done
bash tools/metal/build_metallib.sh "$OUT/test.metallib" -DPUF_TEST_SIGMOID "${MUT[@]}"
clang++ -std=c++17 -O2 -fobjc-arc -Wall tests/metal/test_sigmoid.mm \
    -framework Metal -framework Foundation -o "$OUT/test_sigmoid"
# Every translation unit that defines a sigmoid: MinGRU learner, rollout
# (primary and grouped-rollout libraries), rollout forward, wide learner.
"$OUT/test_sigmoid" "$OUT/test.metallib:mingru" "$OUT/test.metallib:rollout" \
    "$OUT/test.metallib:rollout_fwd" "$OUT/test.rollout.metallib:rollout" \
    "$OUT/test.wide.metallib:wide" "${ARGS[@]}"
