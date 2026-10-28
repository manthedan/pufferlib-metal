#!/bin/bash
# Compare small-policy fused inference with the original GEMM/gate/copy path.
# PRECISION=bf16 builds the bf16 variant (tests/metal/run_test_rollout_bf16.sh).
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
if [ "${PRECISION:-float}" = bf16 ]; then
    OUT=build/metal/test_rollout_bf16
    DEFS=
else
    OUT=build/metal/test_rollout
    DEFS=-DPRECISION_FLOAT
fi
mkdir -p "$OUT"
bash tools/metal/build_metallib.sh "$OUT/puffer.metallib" $DEFS
python3 - "$OUT/puffer.metallib" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
primary = p.read_bytes()
aux = p.with_suffix('.rollout.metallib').read_bytes()
assert b'rollout_steps_packed_' not in primary
for h, split in [(32, 1), (32, 2), (64, 1), (64, 2), (64, 4)]:
    assert f'rollout_steps_packed_h{h}_p{split}'.encode() in aux
print('primary / auxiliary entry points isolated PASS')
PY
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat $DEFS \
    tests/metal/test_rollout.mm src/metal/runtime.mm src/metal/launch.mm src/metal/gemm.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_rollout"
"$OUT/test_rollout" "$@"

# The grouped entry points must keep every buffer's arguments, RNG and state.
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
    -Isrc/metal -Isrc/metal/compat $DEFS \
    tests/metal/test_rollout_group.mm src/metal/runtime.mm \
    -framework Metal -framework Foundation \
    -Wl,-sectcreate,__DATA,__metallib,"$OUT/puffer.metallib" \
    -o "$OUT/test_rollout_group"
"$OUT/test_rollout_group"
PUF_METAL_ROLLOUT_LIB="$OUT/must_not_load.metallib" "$OUT/test_rollout_group" primary-only

# The auxiliary library has its own precision guard, before any dispatch.
WRONG_DEFS=
if [ -z "$DEFS" ]; then WRONG_DEFS=-DPRECISION_FLOAT; fi
xcrun -sdk macosx metal -std=metal4.0 -O2 \
    -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
    -Isrc/metal/shaders $WRONG_DEFS -c src/metal/shaders/runtime.metal \
    -o "$OUT/wrong_aux.air"
xcrun -sdk macosx metallib "$OUT/wrong_aux.air" -o "$OUT/wrong_aux.metallib"
PUF_METAL_ROLLOUT_LIB="$OUT/wrong_aux.metallib" "$OUT/test_rollout_group" reject-aux
