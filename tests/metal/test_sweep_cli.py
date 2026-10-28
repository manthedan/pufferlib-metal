"""A bounded real cartpole sweep through child train processes and result pipes."""
import configparser
import pathlib
import re
import subprocess
import sys
import tempfile
import time

binary = pathlib.Path(sys.argv[1]).resolve()
out_root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else 'build/metal/protein/demo')
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
assert runs in (2, 12)
out_root.mkdir(parents=True, exist_ok=True)
# Retain each run's artifacts without accepting logs from an earlier invocation.
out = pathlib.Path(tempfile.mkdtemp(prefix='run-', dir=out_root))
print(f'Sweep artifacts: {out}', flush=True)
args = {
    'base.seed': 1, 'base.eval_episodes': 0, 'base.checkpoint_interval': 1000,
    'base.log_dir': out / 'logs', 'base.checkpoint_dir': out / 'checkpoints',
    'vec.total_agents': 256, 'vec.num_buffers': 1, 'vec.num_threads': 1,
    'policy.hidden_size': 32, 'policy.num_layers': 2,
    'train.total_timesteps': 131072, 'train.horizon': 32, 'train.minibatch_size': 4096,
    'sweep.max_runs': runs, 'sweep.gpus': -1, 'sweep.downsample': 3,
    'sweep.train.total_timesteps.min': 65536, 'sweep.train.total_timesteps.max': 262144,
}
# Keep architecture/batch axes within the trainer's legal discrete shapes.
# Preserve the real continuous optimization axes, including cost and LR.
for key, value in [('policy.hidden_size', 32), ('policy.num_layers', 2),
                   ('vec.total_agents', 256), ('train.horizon', 32),
                   ('train.minibatch_size', 4096)]:
    args[f'sweep.{key}.min'] = value
    args[f'sweep.{key}.max'] = value + 0.01
command = [str(binary), 'sweep'] + [f'--{k}={v}' for k, v in args.items()]
(out / 'command.txt').write_text(' '.join(command) + '\n')
start = time.monotonic()
run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     timeout=600)
(out / 'sweep.log').write_text(run.stdout)
print(run.stdout, end='')
assert run.returncode == 0, run.returncode
rows = re.findall(r'sweep run=(\d+) score=([^ ]+) cost=([^ ]+) steps=([^ ]+) '
                  r'random=(\d+) gp_obs=(\d+) pareto=(\d+)', run.stdout)
assert len(rows) == runs and [int(r[0]) for r in rows] == list(range(runs))
assert all(0 < float(r[3]) <= 262144 for r in rows)
assert sum(int(r[4]) for r in rows) == min(runs - 1, 10)
if runs > 10:
    assert int(rows[-1][5]) > 0 and int(rows[-1][6]) > 0
# Every worker persists its actual unnormalized suggestion and learning curve.
ini_paths = sorted((out / 'logs/cartpole').glob('*.ini'))
assert len(ini_paths) == runs, f'expected {runs} worker logs in {out}, got {len(ini_paths)}'
for ini_path in ini_paths:
    ini = configparser.ConfigParser(strict=False)
    ini.read(ini_path)
    steps = float(ini['train']['total_timesteps'])
    assert 65536 <= steps <= 262144
    assert 'metrics' in ini and len(ini['metrics']['uptime'].split(',')) == 3
    print(f"suggestion {ini_path.stem}: timesteps={steps:g} "
          f"lr={ini['train']['learning_rate']} perf={ini['metrics']['env/perf']} "
          f"costs={ini['metrics']['uptime']}")
print(f'PASS real Metal sweep: {runs} trials ({time.monotonic() - start:.2f}s)')
