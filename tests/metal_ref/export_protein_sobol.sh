#!/bin/bash
# CUDA host exporter. Added on macOS; this wrapper has NOT been run on CUDA.
# Requires a CUDA 12.8 toolkit and a little-endian host. No GPU work is launched.
set -e
cd "$(dirname "$0")/../.."
: "${PUFFER_GOLDEN_DIR:?Set PUFFER_GOLDEN_DIR to your external fixture directory}"
mkdir -p build/metal_ref "$PUFFER_GOLDEN_DIR/protein"
nvcc -O2 tests/metal/export_protein_rng.cu -lcurand -o build/metal_ref/export_protein_sobol
build/metal_ref/export_protein_sobol "$PUFFER_GOLDEN_DIR/protein/curand_sobol32.bin"
python3 tests/metal/check_fixtures.py "$PUFFER_GOLDEN_DIR" protein
