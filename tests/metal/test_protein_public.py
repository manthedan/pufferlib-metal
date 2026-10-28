"""Offline generation/runtime equivalence, repeatability and bad-file controls."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sobol', ROOT / 'tools/metal/gen_protein_sobol.py')
sobol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sobol)
binary = str(Path(sys.argv[1]).resolve())
env = dict(os.environ)
env.pop('PUF_PROTEIN_SOBOL_TABLE', None)
compact = sobol.ARTIFACT.read_bytes()
# Regenerate from the committed publisher-source excerpt. The gate deliberately
# does not use unpack_rows: parser or packing drift must change these bytes.
assert hashlib.sha256(compact).hexdigest() == 'adca16e3a020a0ee099fd130b6cb5f46628c14fe629909b7fa850eb585d07210'
excerpt = sobol.EXCERPT.read_bytes()
assert hashlib.sha256(excerpt).hexdigest() == 'af7492ff6eb0c4e1d968a15710d6496e61709d151821859841b0c16893d7b905'
rows = sobol.source_rows(excerpt)
assert sobol.pack_rows(rows) == compact
expanded = sobol.expand_rows(rows)
assert hashlib.sha256(expanded).hexdigest() == 'e6c1be592b7b514b41adb06d9f2e364471e8e4670825b1428e8974e2a2072749'
assert subprocess.check_output([binary, 'table'], env=env) == expanded
subprocess.run([binary], check=True, env=env)
a = subprocess.check_output([binary, 'suggest'], env=env)
b = subprocess.check_output([binary, 'suggest'], env=env)
assert a == b and len(a) == 16 * 21 * 4
print('PASS public table matches Python generator byte-for-byte; seeded suggestions repeat')
with tempfile.TemporaryDirectory(dir=ROOT / 'build/metal/protein') as tmp:
    tmp = Path(tmp)
    generated = tmp / 'generated.bin'
    subprocess.run([sys.executable, str(ROOT / 'tools/metal/gen_protein_sobol.py'),
                    '--excerpt', str(sobol.EXCERPT), '--output', str(generated)],
                   check=True, cwd=ROOT)
    assert generated.read_bytes() == compact
    mutated = tmp / 'mutated-joe-kuo'
    needle = b'3       2       1       1 3 \n'
    assert excerpt.count(needle) == 1
    mutated.write_bytes(excerpt.replace(needle, b'3       2       1       1 1 \n'))
    rejected = subprocess.run(
        [sys.executable, str(ROOT / 'tools/metal/gen_protein_sobol.py'),
         '--excerpt', str(mutated), '--output', str(generated)],
        capture_output=True, text=True, cwd=ROOT)
    assert rejected.returncode != 0 and 'excerpt SHA256 mismatch' in rejected.stderr, rejected
    print('PASS offline source excerpt regeneration and mutated-direction rejection')
    path = Path(tmp) / 'invalid.bin'
    for contents, message in [(None, 'cannot open'), (b'', '2640000 bytes'),
                              (bytes(2640001), '2640000 bytes'),
                              (bytes(2640000), 'SHA256 mismatch')]:
        if contents is not None:
            path.write_bytes(contents)
        bad_env = dict(env, PUF_PROTEIN_SOBOL_TABLE=str(path))
        result = subprocess.run([binary, 'probe'], env=bad_env, capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr, result
    # Check the inventory's missing/invalid distinction as well as the loader.
    fixture = Path(tmp) / 'protein/curand_sobol32.bin'
    fixture.parent.mkdir()
    for contents, code in [(None, 1), (b'', 2), (bytes(2640000), 2)]:
        if contents is not None:
            fixture.write_bytes(contents)
        result = subprocess.run([sys.executable, str(ROOT / 'tests/metal/check_fixtures.py'),
                                 tmp, 'protein'], capture_output=True, text=True)
        assert result.returncode == code, result
print('PASS missing/short/oversized/wrong-hash external table rejection and fixture classification')
