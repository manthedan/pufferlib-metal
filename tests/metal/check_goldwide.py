#!/usr/bin/env python3
"""CPU-only shape, coverage, determinism and SHA-256 audit of the wide bundle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from fixture_validation import check_fixtures, require, read_json

TAGS = ('wide_a', 'wide_b', 'wide_bf16', 'wide_bf16_b')
DTYPE_BYTES = {'f32': 4, 'bf16': 2, 'i32': 4, 'u8': 1, 'u64': 8}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def audit(root):
    missing = check_fixtures(root, [*TAGS, 'wide_reports'])
    require(not missing, f'Missing fixtures: {missing[:1]}')
    reports = root / 'wide_reports'
    inventory = read_json(reports / 'inventory.json')
    hashes = json.loads((reports / 'SHA256.json').read_text())
    for name, expected in hashes.items():
        rel = Path(name)
        path = root / rel if rel.parts[0] in TAGS else reports / rel
        assert sha(path) == expected, f'SHA-256 mismatch: {path}'
    expected_cases = {'policy_freeway', 'policy_tetris'}
    expected_cases.update(f'policy_layer_h{h}' for h in (256, 512, 1024))
    expected_cases.update(f'{kind}_{env}/train_epoch3' for kind in ('train', 'replay') for env in ('freeway', 'tetris'))
    for h in (128, 256, 512, 1024):
        expected_cases.add(f'gate_h{h}')
        expected_cases.update(f'scan_{variant}_h{h}_t{t}' for variant in ('plain', 'row') for t in (32, 35, 64, 129))
        expected_cases.update(f'gemm_h{h}_{op}' for op in ('mingru_nt', 'mingru_tn', 'mingru_nn', 'ragged_n'))
        expected_cases.update(f'muon_h{h}_{kind}' for kind in ('matrix', 'wide'))
    tensor_files = raw_bytes = 0
    for tag in TAGS:
        cases = {str(p.parent.relative_to(root / tag)) for p in (root / tag).rglob('manifest.json')}
        assert cases == expected_cases, f'Wrong case coverage: {tag}: {cases ^ expected_cases}'
        for name in sorted(cases):
            rel = f'{tag}/{name}'
            path = root / rel
            manifest = json.loads((path / 'manifest.json').read_text())
            assert manifest['schema_version'] == 1 and manifest['byte_order'] == 'little', rel
            assert manifest['precision'] == ('BF16' if 'bf16' in tag else 'PRECISION_FLOAT'), rel
            assert rel in inventory, rel
            assert not inventory[rel]['nonfinite_float_tensors'], rel
            for tensor in manifest['tensors'].values():
                assert tensor['role'] in ('input', 'param', 'state', 'output'), rel
                size = math.prod(tensor['shape']) * DTYPE_BYTES[tensor['dtype']]
                assert (path / tensor['file']).stat().st_size == size, f'{rel}/{tensor["file"]}'
                raw_bytes += size
                tensor_files += 1
            entry = inventory[rel]
            expected_hashes = {f: hashes[f'{rel}/{f}'] for f in
                               ['manifest.json', *(t['file'] for t in manifest['tensors'].values())]}
            require(entry['sha256'] == expected_hashes, f'{rel}: inventory hash coverage')
            require(entry['tensor_count'] == len(manifest['tensors']), f'{rel}: tensor count')
            s = manifest['scalars']
            if name.startswith('scan_'):
                t = manifest['tensors']['terminals']
                raw = (path / t['file']).read_bytes()
                vals = ([v[0] for v in struct.iter_unpack('<f', raw)] if t['dtype'] == 'f32'
                        else [struct.unpack('<f', struct.pack('<I', v[0] << 16))[0] for v in struct.iter_unpack('<H', raw)])
                T = int(s['T'])
                assert vals[0] == vals[T//2] == vals[T-1] == 1 and .25 in vals, rel
            if name.startswith('train_'):
                env = name.split('/')[0][6:]
                assert s['total_agents'] == 128 and s['epoch'] == 3 and s['slot'] == 1, rel
                assert s['hidden'] == (128 if env == 'freeway' else 512), rel
                assert s['layers'] == (4 if env == 'freeway' else 2), rel
                assert s['horizon'] == (64 if env == 'freeway' else 128), rel
                assert s['minibatch_size'] == 128 * s['horizon'], rel
    assert len(inventory) == len(TAGS) * len(expected_cases), 'Inventory coverage'
    for a, b in ((TAGS[0], TAGS[1]), (TAGS[2], TAGS[3])):
        for name in expected_cases:
            assert inventory[f'{a}/{name}']['sha256'] == inventory[f'{b}/{name}']['sha256'], f'Independent runs differ: {a}/{name}'
        for env in ('freeway', 'tetris'):
            live = inventory[f'{a}/train_{env}/train_epoch3']['sha256']
            replay = inventory[f'{a}/replay_{env}/train_epoch3']['sha256']
            assert all(live[k] == v for k, v in replay.items() if k != 'manifest.json'), f'Replay/live difference: {a}/{env}'
    return {'passed': True, 'cases': len(inventory), 'tensor_files': tensor_files,
            'raw_tensor_bytes': raw_bytes, 'sha256_files_verified': len(hashes),
            'gpu_work_run': False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('golden_root', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit(args.golden_root.resolve())
    text = json.dumps(result, indent=2) + '\n'
    print(text, end='')
    if args.report:
        args.report.write_text(text)

if __name__ == '__main__':
    main()
