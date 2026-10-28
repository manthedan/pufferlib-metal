#!/bin/bash
# Bash 3.2; curl is needed only for URL downloads.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec python3 "$ROOT/tools/metal/golden_bundle.py" fetch "$@"
