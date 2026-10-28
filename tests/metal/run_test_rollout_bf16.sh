#!/bin/bash
# BF16 fused rollout (rollout_forward / rollout_step_h32) vs the generic bf16
# GEMM + gate + copy path, bit-exact. Run from the repo root.
PRECISION=bf16 exec bash "$(dirname "$0")/run_test_rollout.sh" "$@"
