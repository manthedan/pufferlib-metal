#!/bin/bash
# The packed rollout step reads weights packed on the default stream at
# rollout start (pufferl.cu metal_rollout_pack). PUF_METAL_TEST_PACK_HOLD=1
# holds that stream ~3 ms before every pack; training must still be bit-exact,
# i.e. no worker step may run before the pack it reads. Covers async (4
# buffers) and sync rollouts. Run from the repo root:
#   tests/metal/run_test_pack_order.sh [TRAINER_BINARY]   (default: Flappy BF16)
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
mkdir -p build/metal
if [ $# -eq 0 ]; then
    TRAIN_OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/pack_order.XXXXXX")
    trap 'rm -f "$TRAIN_OUT/puffer"' EXIT
    ./build.sh flappy "$TRAIN_OUT/puffer" --metal
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
out = Path(tempfile.mkdtemp(prefix="test_pack_order.", dir="build/metal")).resolve()
cases = [
    ("async", []),
    ("sync", ["--base.async=0"]),
]
for name, args in cases:
    weights = []
    for hold in ("0", "1"):
        root = out / (name + "_" + hold)
        env = dict(os.environ, PUF_METAL_TEST_PACK_HOLD=hold)
        env.pop("PUF_METAL_ROLLOUT_PACKED", None)
        cmd = [str(binary), "train", "--base.seed=1",
               "--train.total-timesteps=2000000", "--base.eval-episodes=0",
               "--base.checkpoint-interval=100000",
               "--base.log-dir=" + str(root / "logs"),
               "--base.checkpoint-dir=" + str(root / "checkpoints")] + args
        with (out / (name + "_" + hold + ".log")).open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        paths = list((root / "checkpoints").rglob("*.bin"))
        assert len(paths) == 1, paths
        weights.append(paths[0].read_bytes())
    assert weights[0] == weights[1], name + ": held pack changed the final weights; logs in " + str(out)
    print(name + ": final model bytes exact " + hashlib.sha256(weights[1]).hexdigest(), flush=True)
print("ALL PASS (logs: " + str(out) + ")")
PY

bash tests/metal/run_test_rollout_group_order.sh "$1"
