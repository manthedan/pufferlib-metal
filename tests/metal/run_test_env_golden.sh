#!/bin/bash
# Replay the CUDA custom-encoder goldens (OSRS entity, craftax, nmmo3 conv, NetHack policy) on
# Metal and compare. Goldens come from CUDA-reference branch golden-envs:
#   tests/metal_ref/env_golden.sh  ->  rsync to ../golden/envs
# Usage: tests/metal/run_test_env_golden.sh [--bf16] [GOLDEN_ROOT] [ENV...]
# BF16 goldens: CUDA-reference golden-envs-bf16, including minimal and asteroids.
# Needs numpy for the comparison.
set -e
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
PREC=--float
TAG=env_golden
if [ "${1:-}" = --bf16 ]; then
    PREC=
    TAG=env_golden_bf16
    shift
fi
DEFAULT_GOLDEN=../golden/envs
if [ -z "$PREC" ]; then DEFAULT_GOLDEN=../golden/envs_bf16; fi
GOLDEN=${1:-$DEFAULT_GOLDEN}
shift || true
ENVS=${*:-"osrs_zulrah osrs_pvp osrs_colosseum osrs_inferno craftax nmmo3 nethack"}
if [ -z "$PREC" ] && [ $# -eq 0 ]; then
    ENVS="$ENVS minimal asteroids"
fi
OUT=build/metal/$TAG
rm -rf "$OUT/out"
mkdir -p "$OUT/out"
for env in $ENVS; do
    NVCC_EXTRA="-DPUF_ENV_GOLDEN" ./build.sh "$env" "$OUT/bin_$env" --metal $PREC > "$OUT/$env.build.log" 2>&1 \
        || { tail -20 "$OUT/$env.build.log"; exit 1; }
    "$OUT/bin_$env" replay "$GOLDEN/${PUF_ENV_GOLDEN_RUN:-run_a}" "$OUT/out"
    for size in small default; do
        test -s "$OUT/out/${env}_${size}/manifest.json" || {
            echo "missing replay case: ${env}_${size}" >&2
            exit 1
        }
    done
done
COMPARE=compare_env_golden.py
if [ -z "$PREC" ]; then COMPARE=compare_env_bf16.py; fi
OPENBLAS_NUM_THREADS=1 python3 "tests/metal/$COMPARE" "$GOLDEN/${PUF_ENV_GOLDEN_RUN:-run_a}" "$OUT/out" --json "$OUT/report.json"
