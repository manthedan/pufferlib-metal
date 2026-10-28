"""Keep the executed smoke commands tied to the reviewed guide corrections."""
from pathlib import Path
import re


def check(text):
    blocks = re.findall(r'```sh\n(.*?)```', text, re.S)
    match = next(b for b in blocks if './puffer match ' in b)
    before = match[:match.index('./puffer match ')]
    assert before[before.rfind('./build.sh '):].strip() == './build.sh slimevolley --metal', \
        'match example must build SlimeVolley first'
    sweep = next(b for b in blocks if './puffer sweep ' in b)
    initial = re.search(r'--train.total-timesteps=(\d+)', sweep)
    assert initial, 'sweep needs an explicit initial timestep budget'
    low = re.search(r'--sweep.train.total-timesteps.min=(\d+)', sweep)
    high = re.search(r'--sweep.train.total-timesteps.max=(\d+)', sweep)
    assert int(low[1]) <= int(initial[1]) <= int(high[1]), 'initial sweep budget is out of range'
    assert 'compatible allocator layouts' in text, 'cross-precision portability needs a layout qualification'


guide = Path('docs/metal-guide.md').read_text()
check(guide)
a = guide.index('Checkpoint files remain FP32.')
b = guide.index('\nThe [README', a)
for name, mutant in [
    ('missing_match_build', guide.replace('./build.sh slimevolley --metal\n', '')),
    ('missing_sweep_initial', guide.replace('--train.total-timesteps=131072', '')),
    ('out_of_range_sweep', guide.replace('--train.total-timesteps=131072', '--train.total-timesteps=15000000')),
    ('unqualified_portability', guide[:a] + 'Checkpoint files are portable when model shapes match.\n' + guide[b:]),
]:
    try:
        check(mutant)
    except AssertionError:
        print('guide_contract:', name, 'mutant CAUGHT')
    else:
        raise AssertionError(name + ' mutant escaped')
check(guide)
print('guide_contract: production/restored PASS')
