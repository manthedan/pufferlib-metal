#!/bin/bash
# Additional H512/H1024 gates, also registered in tests/metal/run_all.sh.
set -e
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
bash tests/metal/run_test_large_gemm.sh
bash tests/metal/run_test_large_muon.sh
bash tests/metal/run_test_large_linear.sh
python3 tests/metal/mutate_large.py
bash tests/metal/run_test_large_e2e.sh
