#!/usr/bin/env python3
"""Host-only negative controls: malformed captures must fail, never skip/pass."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fixture_validation import DTYPE_BYTES, validate_hashes, validate_manifest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAME = 'run_a/gate_tiny/manifest.json'
VALID = json.loads((HERE / 'fixtures/validation/gate_tiny.json').read_text())


class FixtureValidationTests(unittest.TestCase):
    def test_valid_metadata_and_inventory(self):
        self.assertEqual(len(validate_manifest(VALID, NAME)), 5)
        validate_hashes({'a.bin': 'a' * 64}, {'a.bin'})

    def test_bad_manifests_fail_cli(self):
        mutations = {}
        mutations['empty'] = {}
        for label in ('missing_scalar', 'missing_schema', 'empty_tensors', 'zero_size',
                      'empty_shape', 'missing_tensor', 'zero_scalar', 'wrong_precision'):
            m = copy.deepcopy(VALID)
            if label == 'missing_scalar': del m['scalars']['B']
            elif label == 'missing_schema': del m['schema_version']
            elif label == 'empty_tensors': m['tensors'] = {}
            elif label == 'zero_size': m['tensors']['out']['shape'][0] = 0
            elif label == 'empty_shape': m['tensors']['out']['shape'] = []
            elif label == 'missing_tensor': del m['tensors']['out']
            elif label == 'zero_scalar': m['scalars']['B'] = 0
            elif label == 'wrong_precision': m['precision'] = 'BF16'
            mutations[label] = m
        for label, m in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
                path = Path(tmp) / NAME
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(m))
                result = subprocess.run([sys.executable, str(HERE / 'check_fixtures.py'),
                                         tmp, 'run_a'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('Invalid fixture:', result.stderr)

    def test_tensor_file_size(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
            path = Path(tmp) / NAME
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(VALID))
            for t in VALID['tensors'].values():
                size = math.prod(t['shape']) * DTYPE_BYTES[t['dtype']]
                (path.parent / t['file']).write_bytes(bytes(size))
            (path.parent / 'out.bin').write_bytes(b'')
            result = subprocess.run([sys.executable, str(HERE / 'check_fixtures.py'), tmp, 'run_a'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('incorrect tensor byte size', result.stderr)

    def test_bad_hash_inventory_both_entrypoints(self):
        bad_path = {f'file_{i}.bin': 'a' * 64 for i in range(3644)}
        bad_path['.'] = 'a' * 64
        for hashes in ({}, {'a.bin': 'a' * 64}, {'a.bin': 'invalid'}, bad_path):
            for script, args in [('check_fixtures.py', ['wide_reports']), ('check_goldwide.py', [])]:
                with self.subTest(entries=len(hashes), script=script), tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
                    reports = Path(tmp) / 'wide_reports'
                    reports.mkdir()
                    (reports / 'SHA256.json').write_text(json.dumps(hashes))
                    (reports / 'inventory.json').write_text('{}')
                    result = subprocess.run([sys.executable, str(HERE / script), tmp, *args],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    if script == 'check_fixtures.py':
                        self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertTrue(result.stderr)
                    self.assertNotIn('"passed": true', result.stdout)

    def test_hash_coverage_and_path(self):
        for hashes in ({'b.bin': 'a' * 64}, {'../a.bin': 'a' * 64}, {'.': 'a' * 64}):
            with self.assertRaises(ValueError):
                validate_hashes(hashes, {'a.bin'})


if __name__ == '__main__':
    (ROOT / 'build').mkdir(exist_ok=True)
    unittest.main()
