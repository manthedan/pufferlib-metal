#!/bin/bash
# Bash 3.2 compatible. Run from anywhere; all artifacts stay in this worktree.
set -e
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
exec python3 tools/metal/env_sweep.py "$@"
