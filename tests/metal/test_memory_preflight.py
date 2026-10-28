"""Safe real training, host-independent hard limits, and allocation accounting."""
from pathlib import Path
import os
import re
import subprocess
import sys

out = Path(sys.argv[1]).resolve()
base = ['--base.async=0', '--base.eval-episodes=0', '--vec.total-agents=64',
        '--vec.num-buffers=8', '--vec.num-threads=1', '--policy.hidden-size=64',
        '--policy.num-layers=1', '--train.horizon=16', '--train.minibatch-size=1024',
        '--train.replay-ratio=1', '--train.total-timesteps=1024',
        f'--base.log-dir={out}/logs', f'--base.checkpoint-dir={out}/checkpoints']

def run(name, args=base, env=None, binary='puffer'):
    result = subprocess.run([str(out / binary), 'train', *args, f'--base.run-id={name}'],
                            env=dict(os.environ, **(env or {})), capture_output=True, text=True)
    (out / f'{name}.log').write_text(result.stdout + result.stderr)
    return result.returncode, result.stdout + result.stderr

def plan(text):
    line = re.search(r'metal memory plan: (.*)', text).group(1)
    return {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}

code, fit = run('fit')
assert code == 0, fit
p = plan(fit)
actual = int(re.search(r'metal memory actual: created=(\d+)', fit).group(1))
rounded = lambda n: ((n + 16383) // 16384) * 16384
# Independent accounting against the real runtime after all rollout workers
# have registered their workspaces. Packed weights are lazy, scratch is freed.
assert p['stream_buffers'] == 4 * 1024**2 + 16384, p
assert p['workspace'] == 4 * 1024**2, p
assert p['peak'] == actual + p['init_scratch'] + rounded(p['packed']), (p, actual)
assert p['init_scratch'] == rounded(3 * 64 * 64 * 4), p
assert p['packed'] > 0, p

# Just one page above a forced recommendation still performs a full epoch.
code, advisory = run('advisory', env={'PUF_METAL_TEST_MEMORY_BUDGET_BYTES': str(p['peak'] - 16384)})
assert code == 0 and 'WARNING:' in advisory and 'continuing' in advisory, advisory
assert 'requested' in advisory and 'recommended available' in advisory
assert 'vec.total_agents, train.minibatch_size, train.horizon, or policy.hidden_size' in advisory

# The shipped large geometry is rejected deterministically BEFORE allocation,
# even on hosts where it would fit the real device limits.
code, large = run('large_forced', ['--base.eval-episodes=0', '--train.total-timesteps=262144'],
                  {'PUF_METAL_TEST_MAX_BUFFER_BYTES': str(1024**3)})
assert code != 0 and 'hard buffer limit' in large, large
# Record the real plan/device limits safely, without a host-capacity assertion.
code, real = run('shipped_plan', ['--base.eval-episodes=0', '--train.total-timesteps=262144'],
                 {'PUF_METAL_TEST_PLAN_ONLY': '1'})
assert (code != 0) == (plan(real)['largest'] > plan(real)['max_buffer']), real

# Page rounding matters: one byte under the rounded arena must fail.
cap = str(p['largest'] - 1)
code, hard = run('hard', env={'PUF_METAL_TEST_MAX_BUFFER_BYTES': cap})
assert code != 0 and 'largest page-rounded allocation' in hard, hard
code, mutant = run('mutant', env={'PUF_METAL_TEST_MAX_BUFFER_BYTES': cap}, binary='puffer_no_preflight')
assert code == 0 and 'hard buffer limit' not in mutant, mutant
code, restored = run('restored', env={'PUF_METAL_TEST_MAX_BUFFER_BYTES': cap})
assert code != 0 and 'hard buffer limit' in restored, restored

# Each setting really prevents the pack allocation; the estimate must agree.
for name, env, extra in [
    ('pack_off', {'PUF_METAL_ROLLOUT_PACKED': '0'}, []),
    ('step_off', {'PUF_METAL_ROLLOUT_STEP': '0'}, []),
    ('fused_off', {'PUF_METAL_FUSED_ROLLOUT': '0'}, []),
    ('reset_off', {'PUF_METAL_FUSED_RESET': '0'}, []),
    ('h64_off', {'PUF_METAL_ROLLOUT_STEP_H64': '0'}, []),
    ('bad_split', {'PUF_METAL_ROLLOUT_PACKED': '3'}, []),
    ('layers5', {}, ['--policy.num-layers=5']),
]:
    code, text = run(name, [*base, *extra], env)
    assert code == 0 and plan(text)['packed'] == 0, text
    q = plan(text)
    created = int(re.search(r'metal memory actual: created=(\d+)', text).group(1))
    assert q['peak'] == created + q['init_scratch'], (q, created)
print('memory_preflight: 8-stream accounting/init scratch exact; packed eligibility PASS')
print('memory_preflight: advisory trains; forced hard limit/page rounding PASS; disabled mutant CAUGHT; restored PASS')
print('memory_preflight: shipped g2048 FP32 plan:', plan(real))

# Isolated host mutants prove each accounting assertion is sensitive to the
# reviewed omissions. Production source is never edited during these builds.
from host_mutant import build_mutant
source = Path('src/pufferl.cu').read_text()
cases = [
    ('rings', 'future += puf_mtl::stream_buffer_bytes();', '(void)0;'),
    ('workspace', 'if (PUF_BACKEND != PUF_GPU || i == 0) planned(kCublasWsBytes);', '(void)0;'),
    ('scratch', 'init_scratch = (std::max)(init_scratch, rounded(((n + 1) & ~size_t(1)) * sizeof(float)));', 'init_scratch = 0;'),
    ('eligibility', 'if (metal_rollout_pack_split(pufferl)) {', 'if (hidden_size == 32 || hidden_size == 64) {'),
    ('rounding', 'if (bytes > largest) largest = bytes;', 'if (n > largest) largest = n;'),
]
for name, old, new in cases:
    assert source.count(old) == 1, name
    binary = build_mutant('g2048', out / f'mutate_{name}', source.replace(old, new),
                          flags='-DPUF_METAL_MEMORY_TEST')
    env = {'PUF_METAL_ROLLOUT_PACKED': '0'} if name == 'eligibility' else {}
    args = base
    if name == 'rounding':
        # The default tiny case is workspace-dominated (already page-aligned).
        # Use an arena with a non-page-aligned tail to distinguish raw bytes.
        args = [*base, '--vec.total-agents=480', '--vec.num-buffers=1',
                '--train.minibatch-size=7680', '--train.total-timesteps=7680']
        code, page_plan = run('rounding_plan', args, {'PUF_METAL_TEST_PLAN_ONLY': '1'})
        assert code == 0, page_plan
        env['PUF_METAL_TEST_MAX_BUFFER_BYTES'] = str(plan(page_plan)['largest'] - 1)
        code, rejection = run('rounding_production', args, env)
        assert code != 0 and 'hard buffer limit' in rejection, rejection
    code, text = run(f'mutate_{name}', args, env=env, binary=binary)
    if name == 'rounding':
        caught = code == 0  # raw size wrongly passes the page-rounded cap
    else:
        assert code == 0, text
        q = plan(text)
        created = int(re.search(r'metal memory actual: created=(\d+)', text).group(1))
        if name == 'scratch':
            caught = q['init_scratch'] != rounded(3 * 64 * 64 * 4)
        else:
            caught = q['peak'] != created + q['init_scratch'] + (rounded(q['packed']) if name != 'eligibility' else 0)
    assert caught, f'{name} mutant escaped'
    print(f'memory_preflight: {name} mutant CAUGHT', flush=True)
code, text = run('accounting_restored')
assert code == 0 and plan(text)['peak'] == p['peak']
print('memory_preflight: accounting restored PASS')
