#!/bin/bash
# CUDA wide-policy gates. --build-only never initializes Metal or runs a binary.
set -euo pipefail
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
cd "$(dirname "$0")/../.."
build_only=0
if [ "${1:-}" = --build-only ]; then build_only=1; shift; fi
GOLDEN=${1:-../golden}
mkdir -p build/metal
OUT=${PUFFER_GOLDWIDE_OUT:-$(mktemp -d "$PWD/build/metal/goldwide.XXXXXX")}
mkdir -p "$OUT"
nice -n 19 clang++ -std=c++17 -O2 tests/metal/test_policy_kink.mm \
 -framework Accelerate -o "$OUT/test_policy_kink"
for precision in fp32 bf16; do
 dir="$OUT/$precision"; mkdir -p "$dir"
 flags=(); float_arg=()
 if [ "$precision" = fp32 ]; then flags=(-DPRECISION_FLOAT); float_arg=(--float); fi
 nice -n 19 bash tools/metal/build_metallib.sh "$dir/puffer.metallib" ${flags[@]+"${flags[@]}"}
 if [ "$precision" = fp32 ]; then readers=(kernels gemm mingru ppo_muon); else readers=(bf16_golden); fi
 readers+=("goldwide_${precision}_controls")
 for reader in "${readers[@]}"; do
  nice -n 19 clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wno-unused-function \
   -Isrc/metal -Isrc/metal/compat ${flags[@]+"${flags[@]}"} \
   "tests/metal/test_$reader.mm" src/metal/runtime.mm src/metal/launch.mm src/metal/gemm.mm \
   -framework Metal -framework Foundation -framework Accelerate \
   -Wl,-sectcreate,__DATA,__metallib,"$dir/puffer.metallib" -o "$dir/test_$reader"
 done
 for env in freeway tetris; do
  NVCC_EXTRA="-DPUF_GOLDEN_DUMP -DPUF_METAL_TRAIN_REPLAY" nice -n 19 ./build.sh \
   "$env" "$dir/replay_$env" --metal ${float_arg[@]+"${float_arg[@]}"}
 done
done
if [ "$build_only" = 1 ]; then echo "Wide gate builds complete; no GPU work run."; exit 0; fi
"$OUT/test_policy_kink"
bash tests/metal/check_goldwide_controls.sh "$OUT"
python3 tests/metal/check_goldwide.py "$GOLDEN"
# Existing numerical tolerances are retained; report both paths against CUDA.
mkdir -p build/metal/test_ppo_muon
for precision in fp32 bf16; do
 dir="$OUT/$precision"
 if [ "$precision" = fp32 ]; then tags=(wide_a wide_b); else tags=(wide_bf16 wide_bf16_b); fi
 for tag in "${tags[@]}"; do
  if [ "$precision" = fp32 ]; then
   "$dir/test_kernels" "$GOLDEN/$tag"
   "$dir/test_gemm" "$GOLDEN/$tag"
   "$dir/test_ppo_muon" "$GOLDEN/$tag"
  fi
  for fused in 0 1; do
   if [ "$precision" = fp32 ]; then reader=mingru; else reader=bf16_golden; fi
   PUF_GOLDWIDE_FUSED=$fused "$dir/test_$reader" "$GOLDEN/$tag"
  done
 done
 for env in freeway tetris; do
  input="$GOLDEN/${tags[0]}/train_$env/train_epoch3"
  reference="$GOLDEN/${tags[0]}/replay_$env/train_epoch3"
  for fused in 0 1; do
   result="$dir/train_${env}_fused$fused"
   if [ -e "$result" ]; then echo "Refusing existing replay output: $result" >&2; exit 2; fi
   PUF_METAL_FUSED_MINGRU=$fused "$dir/replay_$env" replay-wide "$input" "$result"
   for seed in 1 2; do
    PUF_METAL_FUSED_MINGRU=$fused PUF_REPLAY_PERTURB_SEED=$seed \
     "$dir/replay_$env" replay-wide "$input" "$result.noise$seed"
   done
   python3 tests/metal/compare_train_replay.py "$reference" "$result" "$result.noise1" "$result.noise2"
  done
 done
done
