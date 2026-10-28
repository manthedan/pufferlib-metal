#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA toolkit root}"
: "${NCCL_HOME:?Set NCCL_HOME to the NCCL installation root}"
export CUDA_HOME NCCL_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$CUDA_HOME/bin/nvcc" -O2 -arch=native -std=c++17 -DPRECISION_FLOAT -DPUF_GOLDEN_DUMP \
 -DENV_HEADER=\"ocean/breakout/breakout.h\" -DENV_NAME=breakout -DPUFFER_BREAKOUT \
 -DPUFFER_ENV_NAME=\"breakout\" -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
 -Xcompiler=-Wno-narrowing --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 \
 -I. -Isrc -Ivendor -Iocean/breakout -Iraylib-5.5_linux_amd64/include \
 -I"$CUDA_HOME/include/cccl" -I"$NCCL_HOME/include" \
 tests/metal_ref/generate.cu raylib-5.5_linux_amd64/lib/libraylib.a \
 -L"$CUDA_HOME/lib64" -L"$NCCL_HOME/lib" -lcudart -lnccl -lnvidia-ml -lcublas \
 -lcusolver -lcurand -lGL -lm -lpthread -lomp5 -o tests/metal_ref/generate
NVCC_EXTRA=-DPUF_GOLDEN_DUMP ./build.sh breakout tests/metal_ref/puffer_golden --float
