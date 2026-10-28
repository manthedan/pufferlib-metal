#!/bin/bash
# Compare complete models with direct (agent,time) observation storage enabled
# and disabled, including async slots, agent chunks and unsupported fallbacks.
# PUF_METAL_DW_SIDE=0 pins the decoder dW placement: the default policy
# chooses it per epoch from timing, which would make train_dispatches vary.
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
mkdir -p build/metal
if [ $# -eq 0 ]; then
    TRAIN_OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/obs_layout.XXXXXX")
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
out = Path(tempfile.mkdtemp(prefix='test_obs_layout.', dir='build/metal')).resolve()
cases = [
    ('async', [], {}),
    ('sync', ['--base.async=0'], {}),
    ('buffers2', ['--vec.num-buffers=2'], {}),
    ('reset', ['--base.reset-every-horizon=1'], {}),
    ('fallback_step', [], {'PUF_METAL_ROLLOUT_STEP': '0'}),
    ('h64_step', ['--policy.hidden-size=64'], {}),
    ('fallback_hidden', ['--policy.hidden-size=64'], {'PUF_METAL_ROLLOUT_STEP_H64': '0'}),
]
for name, args, extra in cases:
    models = []
    dispatches = []
    for bt in ('0', '1'):
        root = out / (name + '_' + bt)
        env = dict(os.environ, PUF_METAL_OBS_BT=bt, PUF_METAL_DW_SIDE='0', **extra)
        cmd = [str(binary), 'train', '--base.seed=1',
               '--train.total-timesteps=2000000', '--base.eval-episodes=0',
               '--base.checkpoint-interval=100000',
               '--base.log-dir=' + str(root / 'logs'),
               '--base.checkpoint-dir=' + str(root / 'checkpoints')] + args
        with (out / (name + '_' + bt + '.log')).open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (out / (name + '_' + bt + '.log')).read_text()
        summary = next(line for line in text.splitlines() if line.startswith('metal_summary'))
        counters = dict(token.split('=', 1) for token in summary.split()[1:])
        dispatches.append(float(counters['train_dispatches']))
        files = list((root / 'checkpoints').rglob('*.bin'))
        assert len(files) == 1, files
        models.append(files[0].read_bytes())
    assert dispatches[0] - dispatches[1] == (0 if name.startswith('fallback') else 1), (name, dispatches)
    assert models[0] == models[1], name + ': observation layouts differ; logs in ' + str(out)
    print(name + ': final model bytes exact ' + hashlib.sha256(models[1]).hexdigest(), flush=True)
print('ALL PASS (logs: ' + str(out) + ')')
PY
