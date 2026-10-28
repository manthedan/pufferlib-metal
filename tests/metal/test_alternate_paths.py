#!/usr/bin/env python3
"""Fresh-process seeded alternate-path checks; dispatch evidence is mandatory."""
import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('suite', choices=['math', 'schedule', 'strict', 'ieee'])
p.add_argument('out', type=Path)
p.add_argument('binary', type=Path)
p.add_argument('precision', choices=['bf16', 'f32'])
p.add_argument('variant', nargs='?', type=Path)
a = p.parse_args()
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=True)
results = {}


def run(tag, overrides=None, hidden=32, profile=False, binary=None, steps=524288):
    tag = a.precision + '-' + tag
    root = out / tag
    # Inherited internal overrides must not silently defeat path assertions.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PUF_METAL_', 'PUF_WARM_', 'PUF_TEST_', 'PUF_SIG_'))
           and k not in ('PUF_METALLIB', 'PUF_ENV_THREADS')}
    if profile:
        env.update(PUF_METAL_KPROF='1', PUF_METAL_GEMM_LOG='1')
    # Keep loss averaging independent of wall-clock log cadence.
    env['PUF_TEST_NO_INTERMEDIATE_LOG'] = '1'
    env.update(overrides or {})
    cmd = [str((binary or a.binary).resolve()), 'train', '--base.seed=1',
           f'--train.total-timesteps={steps}', '--vec.total-agents=1024',
           '--vec.num-buffers=4', '--vec.num-threads=4', '--base.async=1',
           '--train.horizon=32', '--train.minibatch-size=8192',
           '--policy.num-layers=2', f'--policy.hidden-size={hidden}',
           '--base.eval-episodes=128', '--base.eval-agents=128',
           '--base.checkpoint-interval=100000',
           f'--base.log-dir={root}/logs', f'--base.checkpoint-dir={root}/checkpoints']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    text = r.stdout + r.stderr
    (out / (tag + '.log')).write_text(text)
    assert r.returncode == 0, (tag, r.returncode, out)
    ck = list((root / 'checkpoints').rglob('*.bin'))
    logs = list((root / 'logs').rglob('*.ini'))
    assert len(ck) == len(logs) == 1, (tag, ck, logs)
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(logs[0])
    score = float(ini['metrics']['env/score'].split(',')[-1])
    assert math.isfinite(score), (tag, score)
    summary = re.search(r'^metal_summary .*', text, re.M)
    assert summary, (tag, 'missing runtime counters')
    counters = {k: float(v) for k, v in
                (field.split('=', 1) for field in summary[0].split()[1:])}
    kernels = {name: int(count) for name, count in
               re.findall(r'^(\w+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+$', text, re.M)}
    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    losses = re.findall(r'\bLosses\s+([-+0-9.eE]+)', clean)
    assert losses and math.isfinite(float(losses[-1])), (tag, 'missing finite displayed loss')
    result = dict(loss=float(losses[-1]), sha256=hashlib.sha256(ck[0].read_bytes()).hexdigest(),
                  score=score, counters=counters, kernels=kernels,
                  muon_modes=re.findall(r'^muon_ns .*', text, re.M))
    results[tag] = result
    (out / (a.precision + '-results.json')).write_text(json.dumps(results, indent=2) + '\n')
    # Keep small logs/results, discard generated model payloads after hashing.
    shutil.rmtree(root / 'checkpoints')
    print(tag, 'score=', score, 'sha256=', result['sha256'], flush=True)
    return result


def exact(ref, alt):
    assert alt['sha256'] == ref['sha256'], ('checkpoint differs', ref['sha256'], alt['sha256'], out)
    assert alt['score'] == ref['score'], ('score differs', ref['score'], alt['score'], out)


def count(result, name):
    return result['kernels'].get(name, 0)


if a.suite == 'math':
    ref = run('h32-default', profile=True)
    assert count(ref, 'muon_ns5_batch') > 0
    assert count(ref, 'mingru_fused_backward_dw') > 0
    for flag in ('PUF_METAL_FUSED_NS', 'PUF_METAL_MUON_NS_MMA',
                 'PUF_METAL_MUON_BATCH'):
        alt = run(flag, {flag: '0'}, profile=True)
        if flag == 'PUF_METAL_FUSED_NS':
            assert count(alt, 'muon_ns5_batch') == count(alt, 'muon_ns5') == 0
            assert count(alt, 'muon_wide_sumsq') > 0
        elif flag == 'PUF_METAL_MUON_NS_MMA':
            # Scalar/MMA use the same pipeline; inspect actual dispatch args.
            assert any('mma=1' in line for line in ref['muon_modes'])
            assert alt['muon_modes'] and all('mma=0' in line for line in alt['muon_modes'])
            assert count(alt, 'muon_ns5_batch') > 0
        elif flag == 'PUF_METAL_MUON_BATCH':
            assert count(alt, 'muon_ns5_batch') == 0 and count(alt, 'muon_ns5') > 0
        exact(ref, alt)
        print('PASS', flag, 'alternate dispatch and exact checkpoint/score', flush=True)
    # H32 fused dW sums 64-row partials, unlike the materialized GEMM.
    # The existing local FP32 norm gate is 2e-5, not a bitwise contract.
    # Check early training (four horizons) with tight score/loss tolerances;
    # longer nonlinear training amplifies the different rounding order.
    ref = run('dw-short-default', profile=True, steps=131072)
    alt = run('PUF_METAL_FUSED_DW', {'PUF_METAL_FUSED_DW': '0'}, profile=True, steps=131072)
    assert count(ref, 'mingru_fused_backward_dw') > 0
    assert count(alt, 'mingru_fused_backward_dw') == 0
    assert count(alt, 'mingru_fused_backward') > 0
    score_delta = abs(ref['score'] - alt['score'])
    loss_delta = abs(ref['loss'] - alt['loss'])
    # The trainer displays loss to three decimal places; allow that resolution
    # in addition to 0.5% relative error. No production logging change needed.
    loss_budget = 0.001 + 0.005 * abs(ref['loss'])
    assert score_delta <= 0.25 and loss_delta <= loss_budget, (
        'fused dW tolerance', score_delta, loss_delta, loss_budget, out)
    print('PASS PUF_METAL_FUSED_DW alternate dispatch; score delta=', score_delta,
          'tolerance=0.25; displayed loss delta=', loss_delta, 'budget=', loss_budget,
          'checkpoint exact=', ref['sha256'] == alt['sha256'], flush=True)
    ref = run('h64-default', hidden=64, profile=True)
    alt = run('PUF_METAL_MUON_WIDE', {'PUF_METAL_MUON_WIDE': '0'}, hidden=64, profile=True)
    assert count(ref, 'muon_wide_sumsq') > 0
    assert not any(k.startswith('muon_wide_') for k in alt['kernels'])
    assert count(alt, 'puf_gemm_f32') + count(alt, 'puf_gemm_bf16') > 0
    exact(ref, alt)
    print('PASS PUF_METAL_MUON_WIDE alternate dispatch and exact checkpoint/score', flush=True)
elif a.suite in ('schedule', 'strict'):
    # KPROF disables gating: these processes deliberately use runtime counters.
    ref = run('schedule-default')
    assert ref['counters']['gated_commits'] > 0
    assert ref['counters']['spin_hits'] + ref['counters']['spin_misses'] == 0
    assert ref['counters']['strict_feedback_joins'] == 0
    for flag, value in [('PUF_METAL_STRICT_FEEDBACK', '1'),
                        ('PUF_METAL_LAZY_PROGRESS', '0'),
                        ('PUF_METAL_ROLLOUT_SPIN_US', '50')]:
        alt = run(flag, {flag: value})
        if flag == 'PUF_METAL_STRICT_FEEDBACK':
            assert alt['counters']['strict_feedback_joins'] > 0, (
                'strict feedback join counter stayed zero: strict joins were skipped', out)
        else:
            assert alt['counters']['strict_feedback_joins'] == 0
        if flag == 'PUF_METAL_ROLLOUT_SPIN_US':
            assert alt['counters']['spin_hits'] + alt['counters']['spin_misses'] > 0
        else:
            assert alt['counters']['gated_commits'] == 0
        exact(ref, alt)
        print('PASS', flag, 'alternate counters and exact checkpoint/score', flush=True)
        if a.suite == 'strict':
            break
else:
    assert a.variant is not None
    ref = run('sigmoid-default')
    alt = run('sigmoid-ieee', binary=a.variant)
    exact(ref, alt)
    print('PASS IEEE sigmoid seeded checkpoint/score exact', flush=True)
