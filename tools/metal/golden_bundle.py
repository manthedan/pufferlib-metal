#!/usr/bin/env python3
"""Deterministic, contract-bound public fixture transport (Python >=3.9)."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zlib

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'tests/metal'
sys.path.insert(0, str(SPEC))
import fixture_validation as fv

EXCLUDED = 'protein/curand_sobol32.bin'
FORMAT = 'ustar+gzip'
VERSION = 1
GENERIC_PERSONAL = re.compile(r'/Users/|/home/|'
                              r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I)
HEX = re.compile(r'\b[0-9a-f]{64}\b', re.I)
# Owner-reviewed checkout shape, restricted to wide report logs by sanitize().
CUDA_PREFIX = re.compile(rb'/home/[A-Za-z0-9._-]+/metal_puffer_ref/')
CUDA_TOKEN = b'<cuda-ref>/'
SANITIZATION_RULE = dict(name='cuda-reference-checkout-v1', scope='wide_reports/*.log',
                         pattern=CUDA_PREFIX, replacement=CUDA_TOKEN)
SHA256_RULE = 'regenerate-log-sha256-entries-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def load_private_tokens(args):
    values = []
    env_value = os.environ.get('PUF_BUNDLE_PRIVATE_TOKENS', '')
    if env_value:
        values.extend(env_value.split(','))
    token_file = getattr(args, 'private_tokens', None)
    if token_file is not None:
        values.extend(re.split(r'[,\r\n]+', token_file.read_text()))
    tokens = []
    seen = set()
    for value in values:
        token = value.strip()
        folded = token.casefold()
        if token and folded not in seen:
            seen.add(folded)
            tokens.append(token)
    if not tokens:
        print('WARNING: NO PRIVATE TOKENS SUPPLIED; the privacy scan is limited to generic '
              'home paths and email addresses. Set PUF_BUNDLE_PRIVATE_TOKENS or pass '
              '--private-tokens FILE before packaging a release.', file=sys.stderr)
    return tokens


def groups():
    result = set()
    for line in (SPEC / 'run_all.sh').read_text().splitlines():
        m = re.match(r'^run \S+ "([^"]*)" ', line)
        if m:
            result.update(m[1].split())
    fv.require(result == set(fv.CASES), 'gate groups differ from fixture_cases.json')
    return sorted(result)


def validate(root, replacements=None):
    selected = groups()
    missing = fv.check_fixtures(root, selected, file_overrides=replacements)
    fv.require(not (set(missing) - {EXCLUDED}), f'missing required fixtures: {missing[:8]}')
    # Packaging checks the in-memory public bytes; installation also exercises
    # the unmodified entrypoint used by the runner, including the private skip.
    if replacements is not None:
        return
    result = subprocess.run([sys.executable, str(SPEC / 'check_fixtures.py'),
                             str(root), *selected], capture_output=True, text=True)
    fv.require(result.returncode == (1 if missing else 0),
               'check_fixtures.py failed: ' + result.stdout + result.stderr)


def inventory(root):
    names = {n for g in groups() for n in fv.CASES[g]} - {EXCLUDED}
    covered = set(fv.CONTRACTS['manifest_metadata_sha256'])
    covered.update(fv.CONTRACTS.get('binary_files', {}))
    covered.update(fv.CONTRACTS.get('file_sha256', {}))
    for name in sorted(names):
        fv.safe_relative(name)
        if Path(name).name == 'manifest.json':
            manifest = fv.read_json(root / name)
            files = fv.validate_manifest(manifest, name)
            names.update(str(Path(name).parent / f) for f in files)
    hashes = {}
    if 'wide_reports' in fv.CASES:
        for name, digest in fv.read_json(root / 'wide_reports/SHA256.json').items():
            fv.safe_relative(name)
            rel = name if Path(name).parts[0] in fv.TAGS else 'wide_reports/' + name
            names.add(rel)
            covered.add(rel)
            hashes[rel] = digest
    fv.require(EXCLUDED not in names, 'private cuRAND table referenced by public inventory')
    for name in names:
        fv.safe_relative(name)
        path = root / name
        parents = [root / Path(name).parts[0]]
        parents.extend(root / p for p in Path(name).parents if str(p) != '.')
        fv.require(path.is_file() and not any(p.is_symlink() for p in [path, *parents]),
                   f'not a regular fixture: {name}')
    return sorted(names), covered, hashes


def sanitize(root, names):
    replacements = {}
    for name in names:
        if Path(name).parent == Path('wide_reports') and name.endswith('.log'):
            original = (root / name).read_bytes()
            sanitized = SANITIZATION_RULE['pattern'].sub(SANITIZATION_RULE['replacement'],
                                                         original)
            if original != sanitized:
                replacements[name] = sanitized
    if not replacements:
        return replacements
    # Patch only digest bytes, preserving every other entry, whitespace and order.
    # inventory.json currently contains no logs, so it is deliberately unchanged.
    changes = {Path(n).name: (sha(root / n), hashlib.sha256(data).hexdigest())
               for n, data in replacements.items()}
    def replace_digest(match):
        key = json.loads(match[1])
        if key not in changes:
            return match[0]
        old, new = changes[key]
        fv.require(match[3].decode() == old, f'original inventory SHA256 mismatch: {key}')
        return match[1] + match[2] + new.encode() + match[4]

    for name in ('wide_reports/inventory.json', 'wide_reports/SHA256.json'):
        original = (root / name).read_bytes()
        sanitized = re.sub(rb'("(?:[^"\\]|\\.)*")(\s*:\s*")([0-9a-f]{64})(")',
                           replace_digest, original)
        if original != sanitized:
            replacements[name] = sanitized
            changes[Path(name).name] = (hashlib.sha256(original).hexdigest(),
                                        hashlib.sha256(sanitized).hexdigest())
    return replacements


def verify_provenance(root, replacements):
    provenance = fv.read_json(SPEC / 'golden_sanitization.json')
    records = provenance.get('files', {})
    rule_name = SANITIZATION_RULE['name']
    rule_definition = dict(scope=SANITIZATION_RULE['scope'],
                           replace_regex=SANITIZATION_RULE['pattern'].pattern.decode(),
                           with_token=SANITIZATION_RULE['replacement'].decode())
    for name, data in replacements.items():
        record = records.get(name, {})
        expected_rule = rule_name if name.endswith('.log') else SHA256_RULE
        fv.require(record.get('original_sha256') == sha(root / name) and
                   record.get('sanitized_sha256') == hashlib.sha256(data).hexdigest() and
                   record.get('rule') == expected_rule,
                   f'sanitization provenance mismatch: {name}')
    fv.require(provenance.get('rules', {}).get(rule_name) == rule_definition,
               'sanitization provenance rule definition mismatch')


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        data = self.source.read(size)
        self.hash.update(data)
        return data


def open_archive_source(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        fv.require(stat.S_ISREG(info.st_mode), f'archive source is not a regular file: {path}')
        return os.fdopen(fd, 'rb'), info.st_size
    except BaseException:
        os.close(fd)
        raise


def open_partial_download(path):
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    else:
        fv.require(stat.S_ISREG(before.st_mode),
                   f'partial download path is not a regular file: {path}')
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            os.close(fd)
            raise ValueError(f'partial download path changed while opening: {path}')
    try:
        info = os.fstat(fd)
        fv.require(stat.S_ISREG(info.st_mode),
                   f'partial download path is not a regular file: {path}')
        return os.fdopen(fd, 'ab'), info.st_size
    except BaseException:
        os.close(fd)
        raise


def scan(root, names, covered, private_tokens, replacements=None):
    hits = []
    replacements = replacements or {}
    patterns = [GENERIC_PERSONAL]
    if private_tokens:
        alternatives = sorted((re.escape(token) for token in private_tokens),
                              key=len, reverse=True)
        patterns.append(re.compile('|'.join(alternatives), re.I))
    for name in names:
        path = root / name
        # Raw tensors are binary; scan all other UTF-8 files, including logs/source.
        if path.suffix == '.bin':
            continue
        try:
            content = (replacements[name] if name in replacements else path.read_bytes()).decode('utf-8')
        except UnicodeError:
            continue
        for line_no, line in enumerate(content.splitlines(), 1):
            spans = [m.span() for m in HEX.finditer(line)]
            for pattern in patterns:
                for match in pattern.finditer(line):
                    false_positive = any(a <= match.start() and match.end() <= b for a, b in spans)
                    hits.append(dict(file=name, line=line_no, match=match[0],
                                     checksum_covered=name in covered,
                                     hex_digest_match=false_positive))
    return hits


def package(args):
    start = time.monotonic()
    private_tokens = load_private_tokens(args)
    root = args.source.resolve()
    output = args.output.resolve()
    fv.require(root.is_dir(), 'source is not a directory')
    fv.require(root not in output.parents, 'output must be outside the fixture source')
    sidecar = Path(str(output) + '.json')
    scan_path = Path(str(output) + '.scan.json')
    fv.require(not output.exists() and not sidecar.exists() and not sidecar.is_symlink(),
               'archive or sidecar already exists')
    fv.require(not scan_path.is_symlink(), 'scan report must not be a symlink')
    names, covered, hashes = inventory(root)
    validated_digests = {name: sha(root / name) for name in names}
    all_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    original_hits = scan(root, names, covered, private_tokens)
    replacements = sanitize(root, names)
    hits = scan(root, names, covered, private_tokens, replacements)
    omitted = sorted(all_files - set(names))
    blocked = sorted({h['file'] for h in hits if not h['hex_digest_match']})
    report = dict(included_files=names, excluded_files=omitted, personal_data_hits=hits,
                  original_personal_data_hits=original_hits,
                  blocked_files=blocked, rewritten_files=sorted(replacements),
                  source_bytes=sum((root / n).stat().st_size for n in names))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(scan_path, report)
    for name in omitted:
        print('EXCLUDE ' + name)
    for hit in hits:
        print('SCAN ' + json.dumps(hit, sort_keys=True))
    fv.require(not blocked, f'personal data outside sanitization allow-list in {len(blocked)} files; '
               f'stopping without archive; see {output}.scan.json')
    for name, digest in hashes.items():
        fv.require(validated_digests[name] == digest, f'wide SHA256 mismatch: {name}')
    if replacements:
        verify_provenance(root, replacements)
    validate(root, replacements)
    before_archive = getattr(args, '_before_archive', None)
    if before_archive is not None:
        before_archive()
    # Explicit headers avoid platform tar options, filesystem metadata, and gzip timestamps.
    temp = None
    archive_sizes = {}
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.golden-', delete=False) as raw:
            temp = Path(raw.name)
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as zipped:
                with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                    for name in names:
                        header = tarfile.TarInfo(name)
                        header.mode = 0o644
                        header.uid = header.gid = header.mtime = 0
                        header.uname = header.gname = ''
                        if name in replacements:
                            header.size = len(replacements[name])
                            archive.addfile(header, io.BytesIO(replacements[name]))
                        else:
                            src, header.size = open_archive_source(root / name)
                            with src:
                                hashed = HashingReader(src)
                                archive.addfile(header, hashed)
                                fv.require(hashed.hash.hexdigest() == validated_digests[name],
                                           f'source changed after validation: {name}')
                        archive_sizes[name] = header.size
        metadata = dict(format_version=VERSION, format=FORMAT, archive_sha256=sha(temp),
                        archive_size=temp.stat().st_size, file_count=len(names),
                        uncompressed_size=sum(archive_sizes.values()),
                        fixture_contracts_sha256=sha(SPEC / 'fixture_contracts.json'), url='')
        temp.rename(output)
        write_json(Path(str(output) + '.json'), metadata)
        print(json.dumps(metadata, sort_keys=True))
        print(f'Packaged in {time.monotonic() - start:.3f} seconds')
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def download_parts(meta, override_url):
    """Return [(url, size)] whose concatenation is the archive.

    GitHub release assets are capped at 2 GiB, so a published bundle may list
    `parts` instead of a single `url`. An explicit --url always means one file.
    """
    def url_ok(url):
        return isinstance(url, str) and url.startswith(('https://', 'http://', 'file://'))

    if override_url is not None:
        fv.require(url_ok(override_url), 'unsupported --url scheme')
        return [(override_url, meta['archive_size'])]
    parts = meta.get('parts')
    if parts is None:
        fv.require(url_ok(meta.get('url')), 'no published URL; supply --url or --file')
        return [(meta['url'], meta['archive_size'])]
    fv.require(not meta.get('url'), 'metadata sets both url and parts')
    fv.require(isinstance(parts, list) and parts, 'parts must be a non-empty list')
    result = []
    for part in parts:
        fv.require(isinstance(part, dict) and set(part) == {'url', 'size'} and url_ok(part['url']) and
                   type(part['size']) is int and part['size'] > 0, f'invalid part entry: {part!r}')
        result.append((part['url'], part['size']))
    fv.require(sum(size for _, size in result) == meta['archive_size'], 'part sizes do not sum to archive_size')
    return result


def split(args):
    archive, size = open_archive_source(args.archive)
    fv.require(args.part_size > 0, '--part-size must be positive')
    parts = []
    with archive:
        remaining = size
        while remaining:
            name = f'{args.archive.name}.part-{len(parts):02d}'
            part_size = min(args.part_size, remaining)
            # Copy in chunks; a part can be close to 2 GiB.
            with open(args.archive.with_name(name), 'xb') as out:
                left = part_size
                while left:
                    data = archive.read(min(left, 1 << 24))
                    fv.require(data, 'archive shrank while splitting')
                    out.write(data)
                    left -= len(data)
            parts.append({'url': args.base_url.rstrip('/') + '/' + name, 'size': part_size})
            remaining -= part_size
    fv.require(sum(p['size'] for p in parts) == size, 'split size mismatch')
    print(json.dumps({'parts': parts}, indent=2))


def destination_ok(dest, force):
    fv.require(not dest.is_symlink(), 'destination must not be a symlink')
    fv.require(not dest.exists() or dest.is_dir(), 'destination is not a directory')
    fv.require(force or not dest.exists() or not any(dest.iterdir()),
               'destination is non-empty; use --force to move it aside')


def fetch(args):
    start = time.monotonic()
    meta = fv.read_json(args.metadata)
    fv.require(meta.get('format_version') == VERSION and meta.get('format') == FORMAT,
               'unsupported bundle format/version')
    fv.require(meta.get('fixture_contracts_sha256') == sha(SPEC / 'fixture_contracts.json'),
               'fixture_contracts.json SHA256 mismatch')
    fv.require(isinstance(meta.get('archive_sha256'), str) and
               re.fullmatch('[a-f0-9]{64}', meta['archive_sha256']), 'archive SHA256 is unset/invalid')
    for key in ('archive_size', 'file_count', 'uncompressed_size'):
        fv.require(type(meta.get(key)) is int and meta[key] > 0, f'invalid {key}')
    dest = Path(os.path.abspath(args.destination))
    destination_ok(dest, args.force)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A sibling lock serializes installers; staging and backup renames stay on one filesystem.
    lock = dest.with_name('.' + dest.name + '.install-lock')
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f'installation lock exists: {lock}; check for another installer')
    try:
        destination_ok(dest, args.force)
        archive = args.file
        if archive is None:
            parts = download_parts(meta, args.url)
            archive = dest.with_name('.' + dest.name + '.' + meta['archive_sha256'] + '.part')
            partial, partial_size = open_partial_download(archive)
            with partial:
                fv.require(partial_size <= meta['archive_size'], 'partial download is larger than the archive')
                # Parts concatenate to the archive; resume inside whichever part the partial file ends in.
                start_offset = 0
                for url, size in parts:
                    end = start_offset + size
                    if partial_size < end:
                        command = ['curl', '-fL', '--retry', '5', '--retry-delay', '2']
                        if partial_size > start_offset:
                            command.extend(['-C', str(partial_size - start_offset)])
                        command.extend(['-o', '-', '--', url])
                        subprocess.run(command, stdout=partial, check=True)
                        partial.flush()
                        partial_size = os.fstat(partial.fileno()).st_size
                        fv.require(partial_size == end, f'download size mismatch after part {url}')
                    start_offset = end
        fv.require(archive.stat().st_size == meta['archive_size'], 'archive size mismatch (partial download?)')
        fv.require(sha(archive) == meta['archive_sha256'], 'archive SHA256 mismatch; extraction refused')
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.' + dest.name + '.staging-') as tmp:
            staging = Path(tmp) / 'golden'
            staging.mkdir()
            seen, total = set(), 0
            # Only canonical regular files are accepted; no links, devices or traversal.
            with gzip.open(archive, 'rb') as zipped:
                with tarfile.open(fileobj=zipped, mode='r|') as bundle:
                    for member in bundle:
                        fv.safe_relative(member.name)
                        fv.require(member.isfile() and member.name not in seen and member.name != EXCLUDED,
                                   f'forbidden/duplicate archive member: {member.name}')
                        seen.add(member.name)
                        total += member.size
                        fv.require(len(seen) <= meta['file_count'] and total <= meta['uncompressed_size'],
                                   'archive exceeds declared file count/size')
                        target = staging / member.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as src, target.open('xb') as dst:
                            shutil.copyfileobj(src, dst)
                        target.chmod(0o644)
                # Drain through the gzip CRC/trailer even after tar's end marker.
                while zipped.read(1024 * 1024):
                    pass
            fv.require(len(seen) == meta['file_count'] and total == meta['uncompressed_size'],
                       'archive file count/size mismatch')
            validate(staging)
            names, _, hashes = inventory(staging)
            fv.require(seen == set(names), 'archive differs from required fixture file list')
            for name, digest in hashes.items():
                fv.require(sha(staging / name) == digest, f'wide SHA256 mismatch: {name}')
            destination_ok(dest, args.force)
            backup = None
            if dest.exists():
                # Preserve even an empty existing directory. No deletion under --force.
                backup = dest.with_name(dest.name + '.backup-' + uuid.uuid4().hex)
                dest.rename(backup)
            try:
                staging.rename(dest)
            except BaseException:
                if backup is not None:
                    backup.rename(dest)
                raise
            if backup is not None:
                print(f'Previous destination preserved at {backup}')
        if args.file is None:
            archive.unlink()
        print(f'Installed {len(seen)} files in {time.monotonic() - start:.3f} seconds; '
              'exact-cuRAND Protein fixture intentionally absent')
    finally:
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('package')
    p.add_argument('--source', type=Path, default=Path(os.environ.get('PUFFER_GOLDEN_DIR', ROOT / 'golden')))
    p.add_argument('--output', type=Path, required=True, help='new .tar.gz archive; sidecars appended automatically')
    p.add_argument('--private-tokens', type=Path, metavar='FILE',
                   help='untracked comma- or newline-separated private tokens to scan for')
    p.set_defaults(action=package)
    p = commands.add_parser('split', help='cut an archive into release-sized parts and print the metadata')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--base-url', required=True, help='release download URL prefix for the parts')
    p.add_argument('--part-size', type=int, default=1900 * 1024 * 1024)
    p.set_defaults(action=split)
    p = commands.add_parser('fetch')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--url')
    source.add_argument('--file', type=Path)
    p.add_argument('--metadata', type=Path, default=SPEC / 'golden_bundle.json', help='trusted bundle metadata')
    p.add_argument('--destination', type=Path, default=Path(os.environ.get('PUFFER_GOLDEN_DIR', ROOT / 'golden')))
    p.add_argument('--force', action='store_true')
    p.set_defaults(action=fetch)
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, OSError, tarfile.TarError, EOFError, zlib.error, subprocess.CalledProcessError) as error:
        print(f'golden bundle: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
