#!/bin/bash
# Use already-built readers; no Metal initialization or golden-file writes.
set -euo pipefail
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
cd "$(dirname "$0")/../.."
OUT=${1:-build/metal/goldwide}
LOG="$OUT/reference_controls"
mkdir -p "$LOG"
"$OUT/fp32/test_goldwide_fp32_controls" good "$LOG/good"
"$OUT/bf16/test_goldwide_bf16_controls" rounding
"$OUT/bf16/test_goldwide_bf16_controls" good
reject() {
 local name=$1 expected=$2 status=0
 shift 2
 "$@" > "$LOG/$name.log" 2>&1 || status=$?
 if [ "$status" != 1 ] || ! grep -Fq "$expected" "$LOG/$name.log"; then
  cat "$LOG/$name.log"
  echo "FAIL: $name must fail numerically (exit 1), got exit $status" >&2
  exit 1
 fi
 echo "negative control $name: rejected (exit 1)"
}
reject sign_bound 'hidden sign change exceeds FP32 dot-product bound' \
 "$OUT/fp32/test_goldwide_fp32_controls" sign_bound "$LOG/sign_bound"
for mode in scan dw dx; do
 case "$mode" in scan) label='f64 scan dC (branch)' ;; dw) label='f64 same-input dW' ;; dx) label='f64 same-input dX' ;; esac
 reject "$mode" "$label" "$OUT/fp32/test_goldwide_fp32_controls" "$mode" "$LOG/$mode"
 # A report line alone is insufficient: require the validator's failure.
 grep -F "$label" "$LOG/$mode.log" | grep -q 'FAIL'
done
for mode in metal_ulp cuda_ulp; do
 reject "$mode" 'FAIL' "$OUT/bf16/test_goldwide_bf16_controls" "$mode"
done
echo 'goldwide reference controls: ALL PASS (6 rejected)'
