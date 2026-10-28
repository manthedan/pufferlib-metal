#!/bin/bash
# Same model bytes across buffer grouping, with held GPU steps to catch a
# missing action-completion wait. Optional trainer binary; default Flappy BF16.
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
mkdir -p build/metal
if [ $# -eq 0 ]; then
    TRAIN_OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/rollout_group_order.XXXXXX")
    trap 'rm -f "$TRAIN_OUT/puffer"' EXIT
    ./build.sh flappy "$TRAIN_OUT/puffer" --metal
    set -- "$TRAIN_OUT/puffer"
fi
OUT=$(mktemp -d build/metal/test_rollout_group_order.XXXXXX)
python3 tests/metal/check_rollout_group.py "$1" "$OUT"
