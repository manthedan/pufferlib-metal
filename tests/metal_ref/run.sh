#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

out="${1:-${PUFFER_GOLDEN_DIR:-$PWD/golden}}"
mkdir -p "$out"
out="$(realpath "$out")"
for tag in a b; do
 ./tests/metal_ref/generate "$out/run_$tag" > "$out/ops-$tag.log" 2>&1
 PUF_GOLDEN_TRAIN="$out/run_$tag/train_epoch3" PUF_GOLDEN_STOP=1 \
  ./tests/metal_ref/puffer_golden train --base.run_id="golden_$tag" \
  --base.log_dir="$out/native_logs" --base.checkpoint_dir="$out/native_checkpoints" \
  > "$out/train-$tag.log" 2>&1
 ./tests/metal_ref/generate replay "$out/run_a/train_epoch3" "$out/replay_$tag/train_epoch3" \
  > "$out/replay-$tag.log" 2>&1
done
python tests/metal_ref/golden.py "$out/run_a" "$out/run_b" --output "$out/determinism_live.json"
python tests/metal_ref/golden.py "$out/replay_a" "$out/replay_b" --output "$out/determinism_replay.json"
OPENBLAS_NUM_THREADS=1 python tests/metal_ref/validate.py "$out/run_a"
python tests/metal_ref/golden.py "$out" --output "$out/inventory.json"
cp tests/metal_ref/golden.py "$out/golden.py"
cp tests/metal_ref/README.md "$out/README.md"
cp config/default.ini config/breakout.ini "$out/"
python tests/metal_ref/summarize.py "$out"
