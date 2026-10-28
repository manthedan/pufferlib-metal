#!/bin/bash
set -eu
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
cd "$(dirname "$0")/../.."
mkdir -p build/robot_arm
out=$(mktemp -d build/robot_arm/gate.XXXXXX)
echo "Robot-arm gate artifacts: $out"
for precision in bf16 f32; do
    flag=; define=
    if [ "$precision" = f32 ]; then flag=--float; define=-DPRECISION_FLOAT; fi
    bash tests/metal/build_robot_arm_cpu.sh tests/metal/robot_arm_trajectory.cpp \
        "$out/capture_$precision" $define > "$out/capture_build_$precision.log" 2>&1
    "$out/capture_$precision" "$out/cpu_$precision.bin" 4
    python3 tests/metal/compare_robot_arm.py \
        "tests/metal/fixtures/robot_arm/cuda_$precision.bin" "$out/cpu_$precision.bin" --gate
    if [ "$precision" = bf16 ]; then
        "$out/capture_$precision" "$out/cpu_reset.bin" 4 reset
        python3 tests/metal/compare_robot_arm.py \
            tests/metal/fixtures/robot_arm/cuda_reset_bf16.bin "$out/cpu_reset.bin" --gate
    fi
    if [ "${ROBOT_ARM_TRAJECTORY_ONLY:-0}" = 1 ]; then continue; fi
    bash build.sh robot_arm "$out/train_$precision" --metal $flag > "$out/train_build_$precision.log" 2>&1
    PUF_ENV_THREADS=10 "$out/train_$precision" train \
        --vec.total_agents=256 --train.total_timesteps=262144 \
        --base.eval_episodes=0 --base.profile=1 --base.checkpoint_interval=1000 \
        --base.run_id=smoke --base.log_dir="$out/$precision/logs" \
        --base.checkpoint_dir="$out/$precision/checkpoints" > "$out/train_$precision.log" 2>&1
    python3 tests/metal/check_robot_arm_smoke.py "$out" "$precision"
done
if [ "${ROBOT_ARM_TRAJECTORY_ONLY:-0}" != 1 ]; then
    python3 tests/metal/mutate_robot_arm.py
fi
echo "robot_arm PASS"
