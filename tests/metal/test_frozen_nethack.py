"""Real Metal frozen NetHack must replace a conflicting NH_MSG_INIT cache."""
from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile
from host_mutant import build_mutant

out = Path(sys.argv[1]).resolve()
root = Path(tempfile.mkdtemp(prefix='run-', dir=out))
common = ['--base.eval-episodes=0', '--base.async=0', '--base.seed=11',
          '--vec.total-agents=8', '--vec.num-buffers=1', '--vec.num-threads=1',
          '--policy.hidden-size=32', '--policy.num-layers=1', '--train.horizon=8',
          '--train.minibatch-size=64', '--train.replay-ratio=1', '--train.total-timesteps=64',
          f'--base.log-dir={root}/logs', f'--base.checkpoint-dir={root}/checkpoints']

def train(binary, name, checkpoint=None, init=None, async_mode=0):
    table, cache = root / f'{name}-table.bin', root / f'{name}-cache.bin'
    env = dict(os.environ, PUF_WARM_START_TEST_NH_TABLE=str(table), PUF_WARM_START_TEST_NH_CACHE=str(cache))
    actor_cache = root / f'{name}-actor-cache.bin'
    if async_mode:
        env['PUF_WARM_START_TEST_NH_ACTOR_CACHE'] = str(actor_cache)
    env.pop('NH_MSG_INIT', None)
    if init:
        env['NH_MSG_INIT'] = str(init)
    args = [f'--base.load-model-path={checkpoint}'] if checkpoint else []
    with (root / f'{name}.log').open('w') as log:
        subprocess.run([str(binary), 'train', *common, f'--base.async={async_mode}', f'--base.run-id={name}', *args],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if async_mode:
        assert actor_cache.read_bytes() == cache.read_bytes(), 'async actor cache differs from policy cache'
    return table.read_bytes(), cache.read_bytes()

source_table, source_cache = train(out / 'puffer', 'source')
assert source_table == source_cache
checkpoint = next((root / 'checkpoints/nethack/source').glob('*.bin'))
init = root / 'message-init.bin'
init.write_bytes(struct.pack('<f', 0.125) * (len(source_table) // 4))
table, cache = train(out / 'puffer', 'warm', checkpoint, init)
assert table == cache and table != init.read_bytes(), 'loaded NetHack table/cache disagree'
assert 'NH_MSG_INIT: loaded' in (root / 'warm.log').read_text()
async_table, async_cache = train(out / 'puffer', 'warm_async', checkpoint, init, async_mode=1)
assert async_table == async_cache == table
source = Path('src/pufferl.cu').read_text()
old = 'nethack_refresh_weight_cache(pol->weights.encoder, pufferl->default_stream);'
assert source.count(old) == 1
mutant = build_mutant('nethack', out / 'mutant', source.replace(old, '(void)0;'),
                      flags='-DPUF_WARM_START_TEST -DNH_MSG_FROZEN')
mutant_table, mutant_cache = train(mutant, 'mutant', checkpoint, init)
assert mutant_table != mutant_cache and mutant_cache == init.read_bytes(), 'stale-cache mutant escaped'
table, cache = train(out / 'puffer', 'restored', checkpoint, init)
assert table == cache and table != init.read_bytes()
print('frozen_nethack: real FP32 sync/async warm starts override conflicting NH_MSG_INIT PASS')
print('frozen_nethack: omitted-refresh mutant CAUGHT; restored PASS; logs=' + str(root))
