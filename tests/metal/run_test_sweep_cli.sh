#!/bin/bash
set -e
unset PUF_PROTEIN_SOBOL_TABLE
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
mkdir -p build/metal/protein
./build.sh cartpole build/metal/protein/puffer --metal
python3 tests/metal/test_sweep_cli.py build/metal/protein/puffer
