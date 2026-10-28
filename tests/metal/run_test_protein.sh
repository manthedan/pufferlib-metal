#!/bin/bash
# Standalone CPU Protein tests; no GPU/runtime or CUDA install required.
set -e
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
G=${PUFFER_GOLDEN_DIR:-$PWD/golden}
status=0
python3 tests/metal/check_fixtures.py "$G" protein || status=$?
if [ "$status" = 1 ]; then
    echo 'protein SKIP(missing fixture)'
    exit 0
elif [ "$status" != 0 ]; then
    exit "$status"
fi
export PUF_PROTEIN_SOBOL_TABLE="$G/protein/curand_sobol32.bin"
OUT=build/metal/protein
mkdir -p "$OUT"
RNG=0
if [ $# -eq 0 ]; then
    set -- test_protein test_protein_numerics test_protein_sweep
    RNG=1
fi
for test in "$@"; do
    case "$test" in
        test_protein|test_protein_numerics|test_protein_sweep) ;;
        *)
            echo "Unknown Protein test: $test" >&2
            exit 1
            ;;
    esac
    clang++ -x c++ -std=c++17 -O2 -Wall -Wno-unused-function -Wno-unused-variable \
        -Wno-vla-cxx-extension -DPUF_METAL -Isrc "tests/$test.cu" \
        -framework Accelerate -o "$OUT/$test"
    "$OUT/$test"
done
if [ "$RNG" = 1 ]; then
    clang++ -std=c++17 -O2 -Wall -Wno-unused-function -Wno-unused-variable \
        -Wno-vla-cxx-extension -Isrc tests/metal/test_protein_rng.cpp \
        -framework Accelerate -o "$OUT/test_protein_rng"
    "$OUT/test_protein_rng"
fi
