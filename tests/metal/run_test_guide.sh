#!/bin/bash
# Execute guide build/train/load/eval/match/sweep examples with bounded work.
set -e
unset PUF_PROTEIN_SOBOL_TABLE
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
cd "$(dirname "$0")/../.."
python3 tests/metal/test_guide_contract.py
OUT=build/metal/guide
mkdir -p "$OUT"
# Record fully expanded commands and preserve all command output.
exec > "$OUT/commands.log" 2>&1
set -x
xcrun -sdk macosx -f metal
./build.sh breakout "$OUT/puffer" --metal
SMALL=(--base.eval-episodes=0 --vec.total-agents=128 --vec.num-buffers=1
       --vec.num-threads=1 --train.horizon=16 --train.minibatch-size=2048
       --train.replay-ratio=1 --train.total-timesteps=2048
       --base.log-dir="$OUT/logs" --base.checkpoint-dir="$OUT/checkpoints")
"$OUT/puffer" train "${SMALL[@]}" --base.run-id=bf16
./build.sh breakout "$OUT/puffer_breakout_f32" --metal --float
"$OUT/puffer_breakout_f32" train "${SMALL[@]}" --base.run-id=f32
"$OUT/puffer" train --base.seed=1 "${SMALL[@]}" --base.run-id=seed1
SOURCE="$OUT/checkpoints/breakout/bf16/0000000000002048.bin"
"$OUT/puffer" train --base.load-model-path="$SOURCE" "${SMALL[@]}" --base.run-id=warm
EVAL=(--base.eval-episodes=1 --base.eval-agents=128 --vec.num-buffers=1
      --vec.num-threads=1 --train.horizon=16 --train.minibatch-size=2048)
"$OUT/puffer" eval latest --headless --base.checkpoint-dir="$OUT/checkpoints" "${EVAL[@]}"
"$OUT/puffer" eval "$SOURCE" --headless "${EVAL[@]}"
# Compatible discrete layouts still load across build precisions.
"$OUT/puffer_breakout_f32" eval "$SOURCE" --headless "${EVAL[@]}"
./build.sh slimevolley "$OUT/puffer" --metal
"$OUT/puffer" train --selfplay.enabled=0 --base.eval-episodes=0 --vec.total-agents=64 \
    --vec.num-buffers=1 --vec.num-threads=1 --train.horizon=16 \
    --train.minibatch-size=1024 --train.replay-ratio=1 --train.total-timesteps=1024 \
    --base.run-id=match_source --base.log-dir="$OUT/logs" --base.checkpoint-dir="$OUT/checkpoints"
SOURCE="$OUT/checkpoints/slimevolley/match_source/0000000000001024.bin"
"$OUT/puffer" match --headless --base.load-model-path="$SOURCE" --base.load-enemy-model-path="$SOURCE" \
    --base.eval-episodes=1 --base.eval-agents=64 --vec.num-buffers=1 --vec.num-threads=1 \
    --train.horizon=16 --train.minibatch-size=1024 --env.max-ticks=32
./build.sh cartpole "$OUT/puffer" --metal
"$OUT/puffer" sweep --sweep.max-runs=2 --sweep.gpus=-1 \
    --base.eval-episodes=0 --train.total-timesteps=131072 \
    --sweep.train.total-timesteps.min=65536 --sweep.train.total-timesteps.max=262144 \
    --base.log-dir="$OUT/sweep_logs" --base.checkpoint-dir="$OUT/sweep_checkpoints" \
    --vec.total-agents=256 --vec.num-buffers=1 --vec.num-threads=1 \
    --policy.hidden-size=32 --policy.num-layers=2 --train.horizon=32 --train.minibatch-size=4096 \
    --sweep.policy.hidden-size.min=32 --sweep.policy.hidden-size.max=32.01 \
    --sweep.policy.num-layers.min=2 --sweep.policy.num-layers.max=2.01 \
    --sweep.vec.total-agents.min=256 --sweep.vec.total-agents.max=256.01 \
    --sweep.train.horizon.min=32 --sweep.train.horizon.max=32.01 \
    --sweep.train.minibatch-size.min=4096 --sweep.train.minibatch-size.max=4096.01
set +x
echo 'guide: BF16/FP32 builds, bounded train/warm start, latest/path/cross-precision eval, SlimeVolley match, tiny sweep PASS'
