#!/usr/bin/env python3
"""Host-only transport tests in an isolated miniature repository."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


class GoldenBundleTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'build').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'build', prefix='bundle-test-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / 'repo'
        self.spec = self.repo / 'tests/metal'
        self.spec.mkdir(parents=True)
        self.tools = self.repo / 'tools/metal'
        self.tools.mkdir(parents=True)
        for name in ('golden_bundle.py', 'package_goldens.sh', 'fetch_goldens.sh'):
            shutil.copy2(ROOT / 'tools/metal' / name, self.tools / name)
        for name in ('fixture_validation.py', 'check_fixtures.py'):
            shutil.copy2(ROOT / 'tests/metal' / name, self.spec / name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.dest = self.base / 'installed'
        self.archive = self.base / 'bundle.tar.gz'
        self.metadata = Path(str(self.archive) + '.json')
        self.manifest = dict(schema_version=1, byte_order='little', precision='PRECISION_FLOAT',
                             scalars={'B': 1}, tensors={'x': dict(file='x.bin', shape=[1], dtype='f32', role='input')})
        self.write_manifest()
        (self.source / 'run_a/case/x.bin').write_bytes(bytes(4))
        (self.source / 'unused.txt').write_text('do not ship')
        (self.source / 'protein').mkdir()
        (self.source / 'protein/curand_sobol32.bin').write_bytes(b'private')
        self.cases = {'run_a': ['run_a/case/manifest.json'], 'protein': ['protein/curand_sobol32.bin']}
        self.contracts = dict(manifest_metadata_sha256={}, wide_report_files=[], wide_sha256_file_count=0,
                              binary_files={'protein/curand_sobol32.bin': dict(size=7, sha256=digest(b'private'))})
        self.write_spec()

    def write_manifest(self):
        path = self.source / 'run_a/case/manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest))

    def write_spec(self):
        self.contracts['manifest_metadata_sha256']['run_a/case/manifest.json'] = digest(
            json.dumps(self.manifest, sort_keys=True, separators=(',', ':')).encode())
        (self.spec / 'fixture_contracts.json').write_text(json.dumps(self.contracts))
        (self.spec / 'fixture_cases.json').write_text(json.dumps(self.cases))
        (self.spec / 'run_all.sh').write_text(''.join(f'run {i} "{g}" true\n' for i, g in enumerate(self.cases)))

    def call(self, script, *args, error=None, env=None):
        command_env = os.environ.copy()
        command_env.pop('PUF_BUNDLE_PRIVATE_TOKENS', None)
        if env:
            command_env.update(env)
        result = subprocess.run(['/bin/bash', str(self.tools / script), *map(str, args)],
                                capture_output=True, text=True, env=command_env)
        output = result.stdout + result.stderr
        if error is None:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(error, output)
        return output

    def package(self, *args, **kwargs):
        return self.call('package_goldens.sh', '--source', self.source, '--output', self.archive,
                         *args, **kwargs)

    def fetch(self, *args, **kwargs):
        return self.call('fetch_goldens.sh', '--file', self.archive, '--metadata', self.metadata,
                         '--destination', self.dest, *args, **kwargs)

    def repin_archive(self):
        meta = json.loads(self.metadata.read_text())
        meta.update(archive_size=self.archive.stat().st_size, archive_sha256=digest(self.archive.read_bytes()))
        self.metadata.write_text(json.dumps(meta))

    def test_success_determinism_and_force_preserves(self):
        output = self.package()
        self.assertIn('WARNING: NO PRIVATE TOKENS SUPPLIED', output)
        original = self.archive.read_bytes()
        with tarfile.open(self.archive) as archive:
            members = archive.getmembers()
            self.assertEqual([m.name for m in members], ['run_a/case/manifest.json', 'run_a/case/x.bin'])
            self.assertTrue(all(m.mtime == m.uid == m.gid == 0 and m.mode == 0o644 for m in members))
        for path in self.source.rglob('*'):
            os.utime(path, (12345678, 12345678))
        second = self.base / 'second.tar.gz'
        self.call('package_goldens.sh', '--source', self.source, '--output', second)
        self.assertEqual(original, second.read_bytes())
        self.fetch()
        self.assertEqual((self.dest / 'run_a/case/x.bin').read_bytes(), bytes(4))
        self.assertFalse((self.dest / 'protein').exists())
        self.fetch(error='destination is non-empty')
        (self.dest / 'keep.txt').write_text('preserve me')
        self.fetch('--force')
        backups = list(self.base.glob('installed.backup-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'keep.txt').read_text(), 'preserve me')
        self.assertFalse((self.dest / 'keep.txt').exists())

    def test_checksum_and_partial_download(self):
        self.package()
        original = self.archive.read_bytes()
        corrupt = bytearray(original)
        corrupt[25] ^= 1
        self.archive.write_bytes(corrupt)
        self.fetch(error='archive SHA256 mismatch')
        self.assertFalse(self.dest.exists())
        self.archive.write_bytes(original[:len(original)//2])
        self.fetch(error='archive size mismatch')
        for partial in (original[:len(original)//2], original[:-8]):
            self.archive.write_bytes(partial)
            self.repin_archive()
            self.fetch(error='golden bundle:')
            self.assertFalse(self.dest.exists())
            self.assertFalse(list(self.base.glob('.installed.staging-*')))

    def test_missing_contract_file_and_force_failure_preserve_old(self):
        self.package()
        self.dest.mkdir()
        (self.dest / 'keep').write_text('original')
        # A complete, hash-verified archive still must satisfy fixture contracts.
        manifest = (self.source / 'run_a/case/manifest.json').read_bytes()
        with tarfile.open(self.archive, 'w:gz') as archive:
            member = tarfile.TarInfo('run_a/case/manifest.json')
            member.size = len(manifest)
            archive.addfile(member, io.BytesIO(manifest))
        self.repin_archive()
        meta = json.loads(self.metadata.read_text())
        meta.update(file_count=1, uncompressed_size=len(manifest))
        self.metadata.write_text(json.dumps(meta))
        self.fetch('--force', error='missing required fixtures')
        self.assertEqual((self.dest / 'keep').read_text(), 'original')
        self.assertFalse(list(self.base.glob('installed.backup-*')))

    def test_contract_digest_mismatch(self):
        self.package()
        self.contracts['extra'] = True
        self.write_spec()
        self.fetch(error='fixture_contracts.json SHA256 mismatch')

    def test_traversal_and_links_rejected(self):
        self.package()
        for name, kind in [('../outside', tarfile.REGTYPE), ('link', tarfile.SYMTYPE),
                           ('protein/curand_sobol32.bin', tarfile.REGTYPE)]:
            with self.subTest(name=name):
                with tarfile.open(self.archive, 'w:gz') as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = '../outside' if kind == tarfile.SYMTYPE else ''
                    archive.addfile(member)
                self.repin_archive()
                self.fetch(error='golden bundle:')
                self.assertFalse(self.dest.exists())
                self.assertFalse((self.base / 'outside').exists())

    def test_privacy_stop_and_source_unchanged(self):
        self.manifest['capture_host'] = '/home/alice/examplehost'
        self.write_manifest()
        self.write_spec()
        before = (self.source / 'run_a/case/manifest.json').read_bytes()
        self.package(error='personal data outside sanitization allow-list')
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.metadata.exists())
        report = json.loads(Path(str(self.archive) + '.scan.json').read_text())
        self.assertEqual(report['blocked_files'], ['run_a/case/manifest.json'])
        self.assertEqual(before, (self.source / 'run_a/case/manifest.json').read_bytes())

    def test_env_and_file_private_tokens_stop(self):
        self.manifest['capture_owner'] = 'alice'
        self.manifest['capture_host'] = 'examplehost'
        self.write_manifest()
        self.write_spec()
        token_file = self.base / 'private-tokens'
        token_file.write_text('examplehost\n')
        self.package('--private-tokens', token_file,
                     env={'PUF_BUNDLE_PRIVATE_TOKENS': 'alice'},
                     error='personal data outside sanitization allow-list')
        report = json.loads(Path(str(self.archive) + '.scan.json').read_text())
        matches = {hit['match'].lower() for hit in report['personal_data_hits']}
        self.assertEqual(matches, {'alice', 'examplehost'})
        self.assertFalse(self.archive.exists())

    def make_wide(self, content):
        hashes, inventory = {}, {}
        for tag in ('wide_a', 'wide_b', 'wide_bf16', 'wide_bf16_b'):
            case = tag + '/case'
            path = self.source / case
            path.mkdir(parents=True)
            shutil.copy2(self.source / 'run_a/case/manifest.json', path / 'manifest.json')
            (path / 'x.bin').write_bytes(bytes(4))
            self.cases[tag] = [case + '/manifest.json']
            self.contracts['manifest_metadata_sha256'][case + '/manifest.json'] = digest(
                json.dumps(self.manifest, sort_keys=True, separators=(',', ':')).encode())
            entry = {n: digest((path / n).read_bytes()) for n in ('manifest.json', 'x.bin')}
            hashes.update({case + '/' + n: h for n, h in entry.items()})
            inventory[case] = dict(tensor_count=1, bytes=4, nonfinite_float_tensors=[], sha256=entry)
        reports = self.source / 'wide_reports'
        reports.mkdir()
        (reports / 'audit.log').write_bytes(content)
        (reports / 'inventory.json').write_text(json.dumps(inventory))
        hashes.update({n: digest((reports / n).read_bytes()) for n in ('audit.log', 'inventory.json')})
        (reports / 'SHA256.json').write_text(json.dumps(hashes))
        self.cases['wide_reports'] = ['wide_reports/inventory.json', 'wide_reports/SHA256.json']
        self.contracts.update(wide_report_files=['audit.log', 'inventory.json'],
                              wide_sha256_file_count=len(hashes))
        self.write_spec()
        return reports

    def test_checksum_covered_wide_log_stops(self):
        reports = self.make_wide(b'wrote /home/alice/capture')
        self.package(error='personal data outside sanitization allow-list')
        report = json.loads(Path(str(self.archive) + '.scan.json').read_text())
        self.assertEqual(report['blocked_files'], ['wide_reports/audit.log'])
        self.assertFalse(self.archive.exists())
        self.assertEqual((reports / 'audit.log').read_text(), 'wrote /home/alice/capture')

    def test_unprotected_text_still_stops(self):
        self.cases['run_a'].append('run_a/note.txt')
        note = self.source / 'run_a/note.txt'
        hex_digest = 'a' * 30 + 'cafe' + 'b' * 30
        note.write_text('capture by alice on examplehost ' + hex_digest)
        self.write_spec()
        self.package(env={'PUF_BUNDLE_PRIVATE_TOKENS': 'alice,examplehost,cafe'},
                     error='personal data outside sanitization allow-list')
        self.assertFalse(self.archive.exists())
        self.assertEqual(note.read_text(), 'capture by alice on examplehost ' + hex_digest)

    def prepare_sanitized_wide(self):
        original = b'wrote /home/alice/metal_puffer_ref/golden/case\r\n'
        sanitized = b'wrote <cuda-ref>/golden/case\r\n'
        reports = self.make_wide(original)
        # Equal digest in an unrelated binary entry must not be rewritten.
        (reports / 'other.bin').write_bytes(original)
        hashes = json.loads((reports / 'SHA256.json').read_bytes())
        hashes['other.bin'] = digest(original)
        (reports / 'SHA256.json').write_text(json.dumps(hashes))
        self.contracts['wide_report_files'].append('other.bin')
        self.contracts['wide_sha256_file_count'] += 1
        old_sha = (reports / 'SHA256.json').read_bytes()
        new_sha = old_sha.replace(('"audit.log": "' + digest(original)).encode(),
                                  ('"audit.log": "' + digest(sanitized)).encode())
        records = {}
        self.contracts['file_sha256'] = {}
        for name, old, new in [('audit.log', original, sanitized), ('SHA256.json', old_sha, new_sha),
                               ('inventory.json', (reports / 'inventory.json').read_bytes(),
                                (reports / 'inventory.json').read_bytes())]:
            rel = 'wide_reports/' + name
            rule = ('cuda-reference-checkout-v1' if name.endswith('.log') else
                    'regenerate-log-sha256-entries-v1')
            records[rel] = dict(original_sha256=digest(old), sanitized_sha256=digest(new),
                                rule=rule)
            self.contracts['file_sha256'][rel] = digest(new)
        rules = {
            'cuda-reference-checkout-v1': {
                'scope': 'wide_reports/*.log',
                'replace_regex': '/home/[A-Za-z0-9._-]+/metal_puffer_ref/',
                'with_token': '<cuda-ref>/',
            },
            'regenerate-log-sha256-entries-v1': {
                'description': 'Replace only digest bytes for sanitized files.',
            },
        }
        (self.spec / 'golden_sanitization.json').write_text(json.dumps(dict(files=records, rules=rules)))
        self.write_spec()
        return reports, original, sanitized, old_sha, new_sha

    def test_allowlist_public_contract_migration_and_byte_preservation(self):
        reports, original, sanitized, old_sha, new_sha = self.prepare_sanitized_wide()
        self.package()
        # A private capture is rejected by the public contract and can be migrated.
        result = subprocess.run(['python3', str(self.spec / 'check_fixtures.py'),
                                 str(self.source), 'wide_reports'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('migrate with fetch_goldens.sh', result.stderr)
        shutil.copytree(self.source, self.dest)
        self.fetch('--force')
        backup, = self.base.glob('installed.backup-*')
        self.assertEqual((backup / 'wide_reports/audit.log').read_bytes(), original)
        self.assertEqual((reports / 'audit.log').read_bytes(), original)
        self.assertEqual((reports / 'SHA256.json').read_bytes(), old_sha)
        self.assertEqual((self.dest / 'wide_reports/audit.log').read_bytes(), sanitized)
        self.assertEqual((self.dest / 'wide_reports/SHA256.json').read_bytes(), new_sha)
        self.assertEqual((self.dest / 'wide_reports/inventory.json').read_bytes(),
                         (reports / 'inventory.json').read_bytes())
        second = self.base / 'public-again.tar.gz'
        self.call('package_goldens.sh', '--source', self.dest, '--output', second)
        self.assertEqual(self.archive.read_bytes(), second.read_bytes())

    def test_allowlist_remaining_hit_stops(self):
        self.prepare_sanitized_wide()
        with (self.source / 'wide_reports/audit.log').open('ab') as out:
            out.write(b'private@example.org')
        hashes_path = self.source / 'wide_reports/SHA256.json'
        hashes = json.loads(hashes_path.read_bytes())
        hashes['audit.log'] = digest((self.source / 'wide_reports/audit.log').read_bytes())
        hashes_path.write_text(json.dumps(hashes))
        self.package(error='personal data outside sanitization allow-list')
        self.assertFalse(self.archive.exists())

    def test_allowlist_wrong_scope_stops(self):
        self.cases['run_a'].append('run_a/audit.log')
        (self.source / 'run_a/audit.log').write_bytes(b'/home/alice/metal_puffer_ref/case')
        self.write_spec()
        self.package(error='personal data outside sanitization allow-list')

    def test_provenance_mismatch_stops(self):
        self.prepare_sanitized_wide()
        (self.spec / 'golden_sanitization.json').write_text('{"files": {}}')
        self.package(error='sanitization provenance mismatch')
        self.assertFalse(self.archive.exists())

    def test_tampered_provenance_rules_stop(self):
        self.prepare_sanitized_wide()
        path = self.spec / 'golden_sanitization.json'
        provenance = json.loads(path.read_text())
        provenance['files']['wide_reports/audit.log']['rule'] = 'unreviewed-rule'
        path.write_text(json.dumps(provenance))
        self.package(error='sanitization provenance mismatch')
        provenance['files']['wide_reports/audit.log']['rule'] = 'cuda-reference-checkout-v1'
        provenance['rules']['cuda-reference-checkout-v1']['scope'] = '**/*.log'
        path.write_text(json.dumps(provenance))
        self.package(error='sanitization provenance rule definition mismatch')
        self.assertFalse(self.archive.exists())

    def test_source_change_after_validation_stops(self):
        module_name = 'isolated_golden_bundle_for_change_test'
        module_spec = importlib.util.spec_from_file_location(module_name,
                                                             self.tools / 'golden_bundle.py')
        module = importlib.util.module_from_spec(module_spec)
        previous_fixture = sys.modules.pop('fixture_validation', None)
        sys.path.insert(0, str(self.spec))
        try:
            module_spec.loader.exec_module(module)
            target = self.source / 'run_a/case/x.bin'
            args = SimpleNamespace(source=self.source, output=self.archive,
                                   _before_archive=lambda: target.write_bytes(b'evil'))
            with self.assertRaisesRegex(ValueError, 'source changed after validation'):
                module.package(args)
        finally:
            sys.path.remove(str(self.spec))
            sys.modules.pop('fixture_validation', None)
            if previous_fixture is not None:
                sys.modules['fixture_validation'] = previous_fixture
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.metadata.exists())

    def test_sanitized_contract_mismatch_stops(self):
        self.prepare_sanitized_wide()
        self.contracts['file_sha256']['wide_reports/audit.log'] = '0' * 64
        self.write_spec()
        self.package(error='SHA256 differs from public fixture contract')
        self.assertFalse(self.archive.exists())

    def test_digest_substring_is_reported_without_redaction(self):
        self.manifest['capture_digest'] = 'a' * 30 + 'cafe' + 'b' * 30
        self.write_manifest()
        self.write_spec()
        self.package(env={'PUF_BUNDLE_PRIVATE_TOKENS': 'cafe'})
        self.fetch()
        report = json.loads(Path(str(self.archive) + '.scan.json').read_text())
        self.assertEqual(len(report['personal_data_hits']), 1)
        self.assertTrue(report['personal_data_hits'][0]['hex_digest_match'])
        self.assertEqual(report['blocked_files'], [])
        self.assertEqual((self.source / 'run_a/case/manifest.json').read_bytes(),
                         (self.dest / 'run_a/case/manifest.json').read_bytes())

    def test_file_url_download_and_default_metadata(self):
        self.package()
        meta = json.loads(self.metadata.read_text())
        meta['url'] = self.archive.as_uri()
        (self.spec / 'golden_bundle.json').write_text(json.dumps(meta))
        partial = self.base / ('.installed.' + meta['archive_sha256'] + '.part')
        partial.write_bytes(self.archive.read_bytes()[:32])
        self.call('fetch_goldens.sh', '--destination', self.dest)
        self.assertFalse(partial.exists())
        self.assertTrue((self.dest / 'run_a/case/x.bin').is_file())

    def test_symlinked_partial_download_is_rejected(self):
        self.package()
        meta = json.loads(self.metadata.read_text())
        meta['url'] = self.archive.as_uri()
        (self.spec / 'golden_bundle.json').write_text(json.dumps(meta))
        victim = self.base / 'partial-victim'
        victim.write_bytes(b'do not overwrite')
        partial = self.base / ('.installed.' + meta['archive_sha256'] + '.part')
        partial.symlink_to(victim)
        self.call('fetch_goldens.sh', '--destination', self.dest,
                  error='partial download path is not a regular file')
        self.assertEqual(victim.read_bytes(), b'do not overwrite')
        self.assertFalse(self.dest.exists())


if __name__ == '__main__':
    unittest.main()
