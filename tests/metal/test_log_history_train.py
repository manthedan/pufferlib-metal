"""Real FP32 Freeway: final state, no episodes, and the Protein result wire."""
import configparser
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

binary = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(tempfile.mkdtemp(prefix='train-', dir='build/metal/log_history')).resolve()
print('Training evidence:', out, flush=True)
for name, epochs, metric in [('episodes', 60, 'score'), ('no-episodes', 2, 'score'),
                              ('trainer-objective', 60, 'metal/epoch_ms')]:
    dest = out / name
    dest.mkdir()
    batch = 512 * 64
    args = {
        'base.seed': 1, 'base.eval_episodes': 0, 'base.checkpoint_interval': 100000,
        'base.log_dir': dest / 'logs', 'base.checkpoint_dir': dest / 'checkpoints',
        'vec.total_agents': 512, 'vec.num_buffers': 1, 'vec.num_threads': 2,
        'policy.hidden_size': 128, 'policy.num_layers': 4,
        'train.horizon': 64, 'train.minibatch_size': 8192,
        'train.total_timesteps': batch * epochs,
        'sweep.downsample': 32, 'sweep.metric': metric,
    }
    with (dest / 'result.bin').open('wb') as wire:
        args['base.result_fd'] = wire.fileno()
        command = [str(binary), 'train'] + [f'--{k}={v}' for k, v in args.items()]
        (dest / 'command.txt').write_text(' '.join(command) + '\n')
        env = os.environ.copy()
        if name == 'episodes':
            env['PUF_TEST_NO_INTERMEDIATE_LOG'] = '1'
        run = subprocess.run(command, pass_fds=(wire.fileno(),),
                             capture_output=True, text=True, env=env)
    (dest / 'train.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, (name, run.returncode, dest)
    logs = list((dest / 'logs').rglob('*.ini'))
    checkpoints = list((dest / 'checkpoints').rglob('*.bin'))
    assert len(logs) == len(checkpoints) == 1
    ini = configparser.ConfigParser(strict=False)
    ini.read(logs[0])
    data = {k: [float(x) for x in v.split(',')] for k, v in ini['metrics'].items()}
    final_steps = int(checkpoints[0].stem)
    assert final_steps == batch * epochs
    for key in ('epoch', 'agent_steps', 'uptime'):
        values = data[key]
        assert len(values) == 32 and all(math.isfinite(x) for x in values)
        assert values == sorted(values), (key, values)
        if name != 'no-episodes':
            assert values[-1] > values[-2], (key, values)
    assert data['epoch'][-1] == epochs
    assert data['agent_steps'][-1] == final_steps
    assert all(s == e * batch for s, e in zip(data['agent_steps'], data['epoch']))
    # TrainResult is a native C struct of 3 floats, int, and three float[64] arrays.
    result = struct.unpack('=3fi192f', (dest / 'result.bin').read_bytes())
    score, cost, steps, points = result[:4]
    scores, costs, step_points = result[4:68], result[68:132], result[132:196]
    assert steps == final_steps and cost > 0
    if name == 'no-episodes':
        assert 'env/score' not in data and data['env/n'][-1] == 0
        assert points == 1 and score == scores[0] == 0
        assert costs[0] == cost and step_points[0] == steps
    else:
        assert points == 32
        assert 'env/score' in data  # Also appears after trainer-objective history starts.
        key = 'env/score' if name == 'episodes' else metric
        for actual, expected in [(scores, data[key]), (costs, data['uptime']),
                                 (step_points, data['agent_steps'])]:
            assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                       for a, b in zip(actual[:points], expected)), (name, key)
        assert scores[points - 1] == score
        assert costs[points - 1] == cost and step_points[points - 1] == steps
    if name == 'episodes':
        assert max(data['env/n']) > 0 and data['env/n'][-1] == 0, data['env/n']
        assert data['env/score'][-1] == data['env/score'][-2]
        # The test schedule logs until the first episode-bearing snapshot, then
        # suppresses every ordinary log until the forced final no-episode log.
        assert len(set(data['epoch'])) == 2, data['epoch']
    print(f'PASS {name}: epochs={data["epoch"][-4:]} '
          f'final_steps={final_steps} final_env/n={data["env/n"][-1]} '
          f'Protein points={points}', flush=True)
