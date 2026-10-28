"""Host-only validation of the pinned CUDA capture metadata (no numerical work)."""
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / 'fixture_cases.json').read_text())
CONTRACTS = json.loads((HERE / 'fixture_contracts.json').read_text())
TAGS = ('wide_a', 'wide_b', 'wide_bf16', 'wide_bf16_b')
DTYPE_BYTES = {'f32': 4, 'bf16': 2, 'i32': 4, 'u8': 1, 'u64': 8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def safe_relative(name):
    require(isinstance(name, str) and name and Path(name).parts and not Path(name).is_absolute()
            and '..' not in Path(name).parts and str(Path(name)) == name,
            f'invalid relative fixture path: {name!r}')
    return Path(name)


def validate_manifest(manifest, name):
    require(isinstance(manifest, dict), f'{name}: expected manifest object')
    require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
            and manifest.get('byte_order') == 'little', f'{name}: invalid schema/byte order')
    require(manifest.get('precision') in ('PRECISION_FLOAT', 'BF16'), f'{name}: invalid precision')
    tensors = manifest.get('tensors')
    require(isinstance(tensors, dict) and tensors, f'{name}: empty/invalid tensor map')
    scalars = manifest.get('scalars')
    require(isinstance(scalars, dict) and scalars, f'{name}: empty/missing scalars')
    files = {}
    for key, tensor in tensors.items():
        require(isinstance(tensor, dict), f'{name}/{key}: invalid tensor')
        shape = tensor.get('shape')
        require(isinstance(shape, list) and shape
                and all(type(d) is int and d > 0 for d in shape), f'{name}/{key}: nonpositive/invalid shape')
        dtype = tensor.get('dtype')
        require(dtype in DTYPE_BYTES, f'{name}/{key}: invalid dtype')
        require(tensor.get('role') in ('input', 'param', 'state', 'output'), f'{name}/{key}: invalid role')
        file = str(safe_relative(tensor.get('file')))
        require(file not in files, f'{name}: duplicate tensor file: {file}')
        files[file] = math.prod(shape) * DTYPE_BYTES[dtype]
    # Pin all scalar keys/values, shapes, tensor names and schema metadata. This
    # prevents omissions/default-to-zero dimensions even in permissive C++ readers.
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    require(digest == CONTRACTS['manifest_metadata_sha256'].get(name),
            f'{name}: metadata differs from pinned contract (required scalars, tensors or schema)')
    return files


def validate_hashes(hashes, expected=None):
    require(isinstance(hashes, dict) and hashes, 'empty/invalid SHA256 inventory')
    for name, digest in hashes.items():
        safe_relative(name)
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                f'invalid SHA256 digest: {name}')
    if expected is not None:
        require(set(hashes) == set(expected), 'SHA256 inventory differs from expected file list')


def check_fixtures(root, groups, file_overrides=None):
    """Return missing paths; malformed metadata always raises, even with missing files."""
    groups = list(groups)
    require(groups, 'at least one fixture group is required')
    if 'wide_reports' in groups:
        groups = list(dict.fromkeys(groups + list(TAGS)))
        for name, digest in CONTRACTS.get('file_sha256', {}).items():
            path = root / safe_relative(name)
            if path.is_file():
                data = file_overrides[name] if file_overrides and name in file_overrides else path.read_bytes()
                require(hashlib.sha256(data).hexdigest() == digest,
                        f'{path}: SHA256 differs from public fixture contract; '
                        'migrate with fetch_goldens.sh --file <bundle> --force')
    missing = []
    wide_files = set(CONTRACTS['wide_report_files'])
    wide_complete = True
    wide_cases = {}
    for group in groups:
        require(group in CASES and CASES[group], f'unknown/empty fixture group: {group}')
        for name in CASES[group]:
            path = root / name
            if not path.is_file():
                missing.append(name)
                if group in TAGS:
                    wide_complete = False
                continue
            if name in CONTRACTS.get('binary_files', {}):
                contract = CONTRACTS['binary_files'][name]
                require(path.stat().st_size == contract['size'], f'{path}: incorrect binary byte size')
                require(hashlib.sha256(path.read_bytes()).hexdigest() == contract['sha256'],
                        f'{path}: SHA256 mismatch')
            if path.name != 'manifest.json':
                continue
            files = validate_manifest(read_json(path), name)
            if group in TAGS:
                wide_cases[str(Path(name).parent)] = files
                wide_files.add(name)
                wide_files.update(str(Path(name).parent / f) for f in files)
            for file, size in files.items():
                data = path.parent / file
                if not data.is_file():
                    missing.append(str(data.relative_to(root)))
                else:
                    require(data.stat().st_size == size, f'{data}: incorrect tensor byte size')
    if 'wide_reports' in groups:
        path = root / 'wide_reports/SHA256.json'
        if path.is_file():
            hashes = read_json(path)
            # Even incomplete bundles must not accept an empty/truncated inventory.
            validate_hashes(hashes, wide_files if wide_complete else None)
            require(len(hashes) == CONTRACTS['wide_sha256_file_count'],
                    'SHA256 inventory must contain 3645 files')
            for name in hashes:
                rel = safe_relative(name)
                data = root / rel if rel.parts[0] in TAGS else path.parent / rel
                if not data.is_file():
                    missing.append(str(data.relative_to(root)))
        path = root / 'wide_reports/inventory.json'
        if path.is_file():
            inventory = read_json(path)
            expected_cases = {str(Path(name).parent) for tag in TAGS for name in CASES[tag]}
            require(isinstance(inventory, dict) and set(inventory) == expected_cases,
                    'wide inventory case coverage differs from expected inventory')
            for name, entry in inventory.items():
                require(isinstance(entry, dict) and entry.get('nonfinite_float_tensors') == [],
                        f'{name}: invalid inventory/nonfinite tensors')
                require(type(entry.get('tensor_count')) is int and entry['tensor_count'] > 0
                        and type(entry.get('bytes')) is int and entry['bytes'] > 0,
                        f'{name}: invalid inventory tensor count/bytes')
                files = wide_cases.get(name)
                validate_hashes(entry.get('sha256'), {'manifest.json', *files} if files else None)
                if files:
                    require(entry['tensor_count'] == len(files) and entry['bytes'] == sum(files.values()),
                            f'{name}: inventory tensor count/bytes differ from manifest')
    return missing
