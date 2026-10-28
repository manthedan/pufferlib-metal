#!/bin/bash
# Expected numerical failures must reach the test, not just fail compilation.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p build/metal
OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/mutation-controls.XXXXXX")
reject() {
    local name=$1 pattern=$2 status=0
    shift 2
    "$@" > "$OUT/$name.log" 2>&1 || status=$?
    if [ "$status" != 1 ] || ! grep -Eq "$pattern" "$OUT/$name.log"; then
        echo "FAIL $name: expected numerical rejection, got exit $status; see $OUT/$name.log" >&2
        return 1
    fi
    echo "PASS rejected $name (numerical diagnostic; exit 1)"
}
case "${1:-}" in
    sigmoid)
        PUF_SIG_REQUIRE_EXACT=0 tests/metal/run_test_sigmoid.sh --quick > "$OUT/baseline.log" 2>&1
        for mutant in --mutate --mutate2; do
            reject "sigmoid$mutant" '^FAIL$' env PUF_SIG_REQUIRE_EXACT=1 \
                tests/metal/run_test_sigmoid.sh --quick "$mutant"
        done
        reject sigmoid-2ulp '^FAIL$' env PUF_SIG_REQUIRE_EXACT=0 \
            tests/metal/run_test_sigmoid.sh --quick --mutate3
        ;;
    proj_store)
        tests/metal/run_test_proj_store.sh --quick > "$OUT/baseline.log" 2>&1
        reject proj-store '^  FAIL: bit mismatches:' tests/metal/run_test_proj_store.sh --quick --mutate
        ;;
    goldwide)
        PUFFER_GOLDWIDE_OUT="$OUT/goldwide" tests/metal/run_test_goldwide.sh --build-only > "$OUT/build.log" 2>&1
        python3 tests/metal/mutate_goldwide.py --build-dir "$OUT/goldwide"
        ;;
    *) echo 'Usage: run_test_mutation_controls.sh sigmoid|proj_store|goldwide' >&2; exit 2 ;;
esac
