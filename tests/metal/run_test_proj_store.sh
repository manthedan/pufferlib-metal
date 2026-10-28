#!/bin/bash
# Exhaustive BF16 fused-projection store check (every fp32 accumulator
# pattern; see tests/metal/test_proj_store.mm). --mutate builds the reference
# with a truncating store and must FAIL. --quick samples batches, retaining
# both signs' FTZ cases, signed zero, and the full output-column probe.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
OUT=build/metal/test_proj_store
mkdir -p "$OUT"
MUT=() ARGS=()
for a in "$@"; do
    case "$a" in --mutate) MUT=(-DPST_MUTATE_REF) ;; *) ARGS+=("$a") ;; esac
done
bash tools/metal/build_metallib.sh "$OUT/test.metallib" -DPUF_TEST_PROJ_STORE "${MUT[@]}"
clang++ -std=c++17 -O2 -fobjc-arc -Wall tests/metal/test_proj_store.mm \
    -framework Metal -framework Foundation -o "$OUT/test_proj_store"
"$OUT/test_proj_store" "$OUT/test.metallib" "${ARGS[@]}"
