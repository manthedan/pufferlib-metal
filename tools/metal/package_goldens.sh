#!/bin/bash
# Bash 3.2; Python >=3.9 standard library supplies portable tar/gzip/SHA256.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec python3 "$ROOT/tools/metal/golden_bundle.py" package "$@"
