"""Drone's shipped H64/L2 continuous policy has equal counts, different offsets."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

out = Path(sys.argv[1]).resolve()
root = Path(tempfile.mkdtemp(prefix='run-', dir=out))
common = ['--base.eval-episodes=0', '--base.async=0', '--vec.total-agents=64',
          '--vec.num-buffers=1', '--vec.num-threads=1', '--train.horizon=16',
          '--train.minibatch-size=1024', '--train.replay-ratio=1',
          '--train.total-timesteps=1024', '--policy.hidden-size=64', '--policy.num-layers=2',
          f'--base.checkpoint-dir={root}/checkpoints', f'--base.log-dir={root}/logs']

def run(binary, name, source=None, ok=True):
    extra = [f'--base.load-model-path={source}'] if source else []
    dump = root / f'{name}-loaded.bin'
    r = subprocess.run([str(out / binary), 'train', *common, f'--base.run-id={name}', *extra],
                       env=dict(os.environ, PUF_WARM_START_TEST_WEIGHTS=str(dump)), capture_output=True, text=True)
    text = r.stdout + r.stderr
    (root / f'{name}.log').write_text(text)
    assert (r.returncode == 0) == ok, text
    return text, dump

sources = {}
for precision in ['bf16', 'f32']:
    run(precision, precision)
    sources[precision] = next((root / f'checkpoints/drone/{precision}').glob('*.bin'))
assert sources['bf16'].stat().st_size == sources['f32'].stat().st_size
for src, dst in [('bf16', 'f32'), ('f32', 'bf16')]:
    source = sources[src]
    assert Path(str(source) + '.layout').is_file()
    _, same = run(src, f'{src}_same', source)
    assert same.read_bytes() == source.read_bytes(), 'same-precision load changed master weights'
    text, _ = run(dst, f'{src}_to_{dst}', source, ok=False)
    assert 'checkpoint parameter layout mismatch' in text and 'Cross-precision' in text

text, _ = run('bf16', 'missing', root / 'missing.bin', ok=False)
assert 'checkpoint load failed:' in text
truncated = root / 'truncated.bin'
truncated.write_bytes(sources['bf16'].read_bytes()[:4])
text, _ = run('bf16', 'truncated', truncated, ok=False)
assert 'checkpoint shape/precision mismatch' in text

legacy = root / 'legacy.bin'
legacy.write_bytes(sources['bf16'].read_bytes())
text, legacy_loaded = run('bf16', 'legacy', legacy)
assert 'no .layout sidecar' in text, text
assert legacy_loaded.read_bytes() == legacy.read_bytes(), 'legacy same-precision load changed master weights'
bad = root / 'bad.bin'
bad.write_bytes(sources['bf16'].read_bytes())
Path(str(bad) + '.layout').write_text('garbage\n')
text, _ = run('bf16', 'bad_sidecar', bad, ok=False)
assert 'checkpoint parameter layout mismatch' in text
# The original flat-load mutant silently accepts the same-size incompatible file.
run('mutant', 'mutant', sources['f32'])
text, _ = run('bf16', 'restored', sources['f32'], ok=False)
assert 'checkpoint parameter layout mismatch' in text
print('checkpoint_layout: Drone H64/L2 same-precision exact; both cross-precision directions rejected PASS')
print('checkpoint_layout: legacy (no sidecar) loads with warning; bad sidecar rejected; unchecked-flat-load mutant CAUGHT; restored PASS')
print('checkpoint_layout: logs=' + str(root))
