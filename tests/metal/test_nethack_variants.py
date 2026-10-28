#!/usr/bin/env python3
"""Default Metal shader preprocessing and clear unsupported-build diagnostics."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
os.chdir(root)
os.environ.setdefault('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer')
(root / 'build/metal').mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix='nethack-variants.',
           dir=os.environ.get('PUFFER_GATE_RUN_DIR', root / 'build/metal')))
shader = ['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.0', '-E', '-P',
          '-Isrc/metal/shaders', 'src/metal/shaders/envs_nethack.metal']
for precision, flags in [('bf16', []), ('f32', ['-DPRECISION_FLOAT'])]:
    default = subprocess.check_output(shader + flags)
    explicit = subprocess.check_output(shader + flags + ['-DNETHACK_V3_K=16'])
    assert default == explicit, precision
    print('PASS default == explicit K=16 preprocessing', precision,
          hashlib.sha256(default).hexdigest(), flush=True)
for flag, diagnostic in [
    ('NH_MSG_V4096', 'Metal NetHack does not support NH_MSG_V4096'),
    ('NH_MSG128', 'Metal NetHack does not support NH_MSG128'),
    ('NH_NO_APANEL', 'Metal NetHack does not support NH_NO_APANEL'),
    ('NETHACK_V3_K=32', 'Metal NetHack requires NETHACK_V3_K=16'),
    ('NETHACK_V3_K=0', 'Metal NetHack requires NETHACK_V3_K=16'),
]:
    # Exercise the real build entrypoint as well as standalone shader users.
    for target in ('host', 'shader'):
        env = dict(os.environ, NVCC_EXTRA='-D' + flag)
        cmd = (['./build.sh', 'nethack', str(out / 'unsupported'), '--metal']
               if target == 'host' else shader + ['-D' + flag])
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        text = run.stdout + run.stderr
        (out / (flag + '-' + target + '.log')).write_text(text)
        assert run.returncode != 0 and diagnostic in text, (flag, target, out)
        assert not (out / 'unsupported').exists(), (flag, 'unexpected binary')
        print('PASS rejected', target, flag, flush=True)
