#!/bin/bash
# Compare complete training results with copied vs shared CPU env IO.
# Run from the repo root: tests/metal/run_test_shared_env.sh [TRAINER_BINARY]
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
mkdir -p build/metal
if [ $# -eq 0 ]; then
    TRAIN_OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/shared_env.XXXXXX")
    trap 'rm -f "$TRAIN_OUT/puffer"' EXIT
    ./build.sh breakout "$TRAIN_OUT/puffer" --metal --float
    set -- "$TRAIN_OUT/puffer"
fi
python3 - "$1" <<'PY'
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
out = Path(tempfile.mkdtemp(prefix="test_shared_env.", dir="build/metal")).resolve()
cases = [
    ("async", []),
    ("sync", ["--base.async=0"]),
    ("buffers2", ["--vec.num-buffers=2"]),
    ("reset", ["--base.reset-every-horizon=1"]),
]
for name, args in cases:
    weights = []
    for shared in ("0", "1"):
        root = out / (name + "_" + shared)
        env = dict(os.environ, PUF_METAL_SHARED_ENV=shared)
        cmd = [str(binary), "train", "--base.seed=1",
               "--train.total-timesteps=2000000", "--base.eval-episodes=0",
               "--base.checkpoint-interval=100000",
               "--base.log-dir=" + str(root / "logs"),
               "--base.checkpoint-dir=" + str(root / "checkpoints")] + args
        with (out / (name + "_" + shared + ".log")).open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        paths = list((root / "checkpoints").rglob("*.bin"))
        assert len(paths) == 1, paths
        weights.append(paths[0].read_bytes())
    assert weights[0] == weights[1], name + ": copied/shared final weights differ; logs in " + str(out)
    print(name + ": final model bytes exact " + hashlib.sha256(weights[1]).hexdigest(), flush=True)
print("ALL PASS (logs: " + str(out) + ")")
PY
