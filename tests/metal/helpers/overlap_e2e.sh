#!/bin/bash
# Backward-overlap end-to-end gate: the backward dW overlap (concurrent windows) in
# its default (per-epoch auto policy), forced-on (PUF_METAL_BWD_OVERLAP=1)
# and off (=0) modes must all reproduce the seed-1 score and checkpoint
# sha256 prefix. Cartpole (H=32, 2 layers, fused dW
# with deferred reductions) and Breakout (H=32, 4 layers, side-stream
# decoder dW policy), BF16 and FP32. Pong/Flappy (H=64, deferred split-K
# dW GEMMs) run with the overlap on in tests/metal/helpers/h64_e2e.sh.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
O=${PUFFER_GATE_RUN_DIR:-$ROOT/build/metal}/overlap_e2e
fail=0
check() {  # ENV PREC SCORE SHA
    D=$O/$1_$2; mkdir -p "$D"
    F=""; [ "$2" = f32 ] && F=--float
    ./build.sh "$1" "$D/puffer" --metal $F > "$D/build.log" 2>&1
    [ -e "$D/config" ] || ln -s "$ROOT/config" "$D/config"
    [ -e "$D/resources" ] || ln -s "$ROOT/resources" "$D/resources"
    for mode in default forced optout; do
        if [ $mode = optout ]; then
            R=$(PUF_METAL_BWD_OVERLAP=0 tests/metal/helpers/seedrun.sh "$D" $1 1)
        elif [ $mode = forced ]; then
            R=$(PUF_METAL_BWD_OVERLAP=1 tests/metal/helpers/seedrun.sh "$D" $1 1)
        else
            R=$(tests/metal/helpers/seedrun.sh "$D" $1 1)
        fi
        if echo "$R" | grep -q "score=$3 sha=$4"; then echo "PASS $mode $R"; else echo "FAIL $mode $R (want $3 $4)"; fail=1; fi
    done
}
check cartpole bf16 199.96110534667969 b10ccdb69f45c996
check cartpole f32 200 bf0330663b50b01e
check breakout bf16 852.43658447265625 f5dc197b8d7fde55
check breakout f32 844.95672607421875 2ee4da82d356a44c
exit $fail
