#!/usr/bin/env python3
"""Guard must notice creation, deletion, writes and identical-byte replacement."""
from pathlib import Path
import tempfile
from root_output_guard import snapshot

scratch = Path(__file__).resolve().parents[2] / 'build/metal'
scratch.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory(dir=scratch) as directory:
    root = Path(directory)
    binary = root / 'puffer'
    absent = snapshot(binary)
    assert absent is None
    binary.write_bytes(b'first')
    first = snapshot(binary)
    assert first != absent and first == snapshot(binary)
    binary.write_bytes(b'other')
    assert snapshot(binary) != first
    first = snapshot(binary)
    replacement = root / 'replacement'
    replacement.write_bytes(binary.read_bytes())
    replacement.replace(binary)
    assert snapshot(binary) != first
    first = snapshot(binary)
    binary.unlink()
    assert snapshot(binary) != first
    binary.symlink_to(replacement)
    dangling = snapshot(binary)
    replacement.write_bytes(b'target')
    assert snapshot(binary) != dangling
    first = snapshot(binary)
    replacement.write_bytes(b'change')
    assert snapshot(binary) != first
print('PASS root output guard: absent/created/deleted/modified/replaced/symlink target')

# Exercise the actual runner function with disposable root paths. A numerical
# gate that exits zero must still fail if it writes/replaces the root trainer.
import shlex
import subprocess

source = Path(__file__).with_name('run_all.sh').read_text()
start = source.index('run() {')
function = source[start:source.index('\n}\n', start) + 3]
with tempfile.TemporaryDirectory(dir=scratch) as directory:
    root = Path(directory)
    (root / 'logs').mkdir()
    binary = root / 'puffer'
    for action in ('create', 'modify', 'replace', 'unchanged', 'delete'):
        if action == 'create':
            code = f'from pathlib import Path; Path({str(binary)!r}).write_bytes(b"first")'
        elif action == 'modify':
            code = f'from pathlib import Path; Path({str(binary)!r}).write_bytes(b"other")'
        elif action == 'replace':
            code = (f'from pathlib import Path; p=Path({str(binary)!r}); q=p.with_name("replacement"); '
                    'q.write_bytes(p.read_bytes()); q.replace(p)')
        elif action == 'delete':
            code = f'from pathlib import Path; Path({str(binary)!r}).unlink()'
        else:
            code = 'pass'
        command = ' '.join(shlex.quote(s) for s in ['python3', '-c', code])
        script = (f'ROOT={shlex.quote(str(root))}\n'
                  f'PUFFER_GATE_RUN_DIR={shlex.quote(str(root / "logs"))}\n'
                  'mode=run; total=0; pass=0; failed=0; fail=0; skip=0\n'
                  + function + '\nrun probe "" ' + command + '\n'
                  + 'echo "result:$pass:$failed:$fail"\n')
        run = subprocess.run(['/bin/bash', '-c', script], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[2])
        expected = 'result:1:0:0' if action == 'unchanged' else 'result:0:1:1'
        assert run.returncode == 0 and expected in run.stdout, (action, run.stdout, run.stderr)
print('PASS runner rejects a successful gate that creates/modifies/replaces/deletes root ./puffer')
