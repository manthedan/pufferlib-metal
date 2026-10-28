#!/bin/bash
# Replay the captured CUDA train() call on Metal and compare against the CUDA
# replay golden. Run from the repo root:
#   tests/metal/run_train_replay.sh [GOLDEN_ROOT]     (default ../golden)
# Needs numpy (e.g. the project venv) for the comparison step.
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
GOLDEN=${1:-../golden}
OUT=build/metal/train_replay
rm -rf "$OUT"
mkdir -p "$OUT"
NVCC_EXTRA="-DPUF_GOLDEN_DUMP -DPUF_METAL_TRAIN_REPLAY" ./build.sh breakout build/metal/puffer_replay --metal --float
build/metal/puffer_replay replay "$GOLDEN/run_a/train_epoch3" "$OUT"
# Noise floor: the same replay with inputs nudged by +-1 ulp, two seeds.
for seed in 1 2; do
    rm -rf "$OUT.noise$seed"
    PUF_REPLAY_PERTURB_SEED=$seed build/metal/puffer_replay replay \
        "$GOLDEN/run_a/train_epoch3" "$OUT.noise$seed" > /dev/null
done
python3 tests/metal/compare_train_replay.py "$GOLDEN/replay_a/train_epoch3" "$OUT" \
    "$OUT.noise1" "$OUT.noise2"
