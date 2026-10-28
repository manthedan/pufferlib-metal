#!/bin/bash
# Leg 13 end-to-end gate: Pong (H=64, 1 buffer) and Flappy (H=64, 4 buffers)
# seed 1, BF16 and FP32, must reproduce the 0e34f2f9 score and checkpoint
# sha256 prefix with the fused H=64 learner / rollout step on (default) and
# with both opted out (generic path).
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
O=${PUFFER_GATE_RUN_DIR:-$ROOT/build/metal}/h64_e2e
fail=0
check() {  # ENV PREC SCORE SHA
    D=$O/$1_$2; mkdir -p "$D"
    F=""; [ "$2" = f32 ] && F=--float
    ./build.sh "$1" "$D/puffer" --metal $F > "$D/build.log" 2>&1
    [ -e "$D/config" ] || ln -s "$ROOT/config" "$D/config"
    [ -e "$D/resources" ] || ln -s "$ROOT/resources" "$D/resources"
    for mode in default optout; do
        if [ $mode = optout ]; then
            R=$(PUF_METAL_FUSED_MINGRU64=0 PUF_METAL_ROLLOUT_STEP_H64=0 tests/metal/helpers/seedrun.sh "$D" $1 1)
        else
            R=$(tests/metal/helpers/seedrun.sh "$D" $1 1)
        fi
        if echo "$R" | grep -q "score=$3 sha=$4"; then echo "PASS $mode $R"; else echo "FAIL $mode $R (want $3 $4)"; fail=1; fi
    done
}
check pong bf16 20.996614456176758 b1425713d9e71433
check pong f32 20.99445915222168 5ee574ef2f33c6dd
check flappy bf16 50.560173034667969 91886134a50625e1
check flappy f32 51.408348083496094 4eafd59e547e43e1
exit $fail
