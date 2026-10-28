import hashlib, os, re, subprocess, sys
from pathlib import Path
binary = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
cases = [('async', []), ('sync', ['--base.async=0']),
         ('copied', []), ('reset', ['--base.reset-every-horizon=1']),
         ('time_obs', [])]
for name, args in cases:
    ref = None
    ref_dispatches = None
    for group, hold in [(1, 0), (2, 0), (4, 0), (2, 1)]:
        if hold and name not in ('async', 'copied'): continue
        tag = f'{name}_g{group}_hold{hold}'
        root = out / tag
        env = dict(os.environ, PUF_METAL_ROLLOUT_GROUP=str(group), PUF_METAL_TEST_STEP_HOLD=str(hold),
                   PUF_METAL_SHARED_ENV='0' if name == 'copied' else '1',
                   PUF_METAL_OBS_BT='0' if name == 'time_obs' else '1')
        for toggle in ('PUF_METAL_ROLLOUT_PACKED', 'PUF_METAL_ROLLOUT_STEP', 'PUF_METAL_FUSED_RESET'):
            env.pop(toggle, None)
        if group == 1:
            # Ungrouped paths must not load the auxiliary shader library.
            env['PUF_METAL_ROLLOUT_LIB'] = str(out / 'must_not_load.metallib')
        else:
            env.pop('PUF_METAL_ROLLOUT_LIB', None)
        cmd = [str(binary), 'train', '--base.seed=1', '--train.total-timesteps=2000000',
               '--base.eval-episodes=0', '--base.checkpoint-interval=100000',
               f'--base.log-dir={root}/logs', f'--base.checkpoint-dir={root}/checkpoints'] + args
        logfile = out / (tag + '.log')
        with logfile.open('w') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        summary = re.findall(r'^metal_summary .*', logfile.read_text(), re.M)[-1]
        metrics = dict(t.split('=', 1) for t in summary.split()[1:])
        dispatches = float(metrics['dispatches'])
        if ref_dispatches is None: ref_dispatches = dispatches
        if group > 1:
            # For four original buffers, grouping must eliminate at least
            # (4 - 4/group) calls per step. Catch an eligibility fallback
            # that would otherwise make the model-byte comparison pass.
            saved = (4 - 4 / group) * float(metrics['rollout_dispatches'])
            assert ref_dispatches - dispatches > 0.9 * saved, (tag, metrics)
        paths = list((root / 'checkpoints').rglob('*.bin'))
        assert len(paths) == 1, paths
        data = paths[0].read_bytes()
        if ref is None: ref = data
        assert data == ref, f'{tag} differs; see {out}'
        print(tag, hashlib.sha256(data).hexdigest()[:16], 'PASS', flush=True)
print('group_order ALL PASS', flush=True)
