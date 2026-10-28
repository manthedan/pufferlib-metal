#!/bin/bash
# BF16 counterpart of run_train_replay.sh: replay the captured CUDA BF16
# train() call on the default (bf16) Metal build and compare against the CUDA
# BF16 replay golden. Run from the repo root:
#   tests/metal/run_train_replay_bf16.sh [GOLDEN_ROOT]     (default ../golden)
# Uses GOLDEN_ROOT/run_bf16/train_epoch3 (input) and
# GOLDEN_ROOT/replay_bf16/train_epoch3 (CUDA BF16 replay output).
# Needs numpy (e.g. the project venv) for the comparison step.
set -e
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
fi
GOLDEN=${1:-../golden}
OUT=build/metal/train_replay_bf16
rm -rf "$OUT"
mkdir -p "$OUT"
NVCC_EXTRA="-DPUF_GOLDEN_DUMP -DPUF_METAL_TRAIN_REPLAY" ./build.sh breakout build/metal/puffer_replay_bf16 --metal
build/metal/puffer_replay_bf16 replay "$GOLDEN/run_bf16/train_epoch3" "$OUT"
# Noise floor: the same replay with the bf16 parameters nudged by +-1 bf16 ulp.
for seed in 1 2; do
    rm -rf "$OUT.noise$seed"
    PUF_REPLAY_PERTURB_SEED=$seed build/metal/puffer_replay_bf16 replay \
        "$GOLDEN/run_bf16/train_epoch3" "$OUT.noise$seed" > /dev/null
done
python3 tests/metal/compare_train_replay.py "$GOLDEN/replay_bf16/train_epoch3" "$OUT" \
    "$OUT.noise1" "$OUT.noise2"
