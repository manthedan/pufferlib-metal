#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR-/Applications/Xcode.app/Contents/Developer}"
python3 tests/metal/test_log_history.py
mkdir -p build/metal/log_history
./build.sh freeway build/metal/log_history/freeway --metal --float
python3 tests/metal/test_log_history_train.py build/metal/log_history/freeway
