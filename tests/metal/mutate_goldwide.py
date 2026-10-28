#!/usr/bin/env python3
"""Require the committed reference-control gate to catch three reader mutations.

Run after run_test_goldwide.sh (or --build-only). Mutated sources and binaries
stay under build/metal; tracked files and golden fixtures are never changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build/metal/goldwide'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.environ.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    baseline = args.build_dir.resolve()
    subprocess.run(['/bin/bash', 'tests/metal/check_goldwide_controls.sh', str(baseline)], check=True)
    mutations = (
        ('remove_sign_bound', 'fp32', 'test_mingru.mm',
         'fabs(comb[k]-r.combined[k])>bound', 'false', 'sign_bound',
         'FAIL: sign_bound must fail numerically'),
        ('bypass_float64_validation', 'fp32', 'test_mingru.mm',
         'auto verify=[&](const char* label,const float* a,const D& b) {',
         'auto verify=[&](const char* label,const float* a,const D& b) { return;', 'scan',
         'FAIL: scan must fail numerically'),
        ('reintroduce_float_cast', 'bf16', 'test_bf16_golden.mm',
         'bf16 rounded=double_to_bf16_rne(exact);',
         'bf16 rounded=__float2bfloat16((float)exact);', 'metal_ulp',
         'reference control good: REJECTED'),
    )
    out = baseline / 'mutations'
    out.mkdir(exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix='sources-', dir=out) as temporary:
        temp = Path(temporary)
        for name, precision, reader, old, new, control, diagnostic in mutations:
            case = temp / name
            case.mkdir()
            source = (root / 'tests/metal' / reader).read_text()
            assert source.count(old) == 1, f'{name}: mutation target changed'
            (case / reader).write_text(source.replace(old, new))
            for p in ('fp32', 'bf16'):
                (case / p).mkdir()
                filename = f'test_goldwide_{p}_controls'
                if p != precision:
                    (case / p / filename).symlink_to(baseline / p / filename)
            filename = f'test_goldwide_{precision}_controls'
            # Quoted includes resolve to the mutated reader beside the control.
            (case / f'{filename}.mm').write_text((root / 'tests/metal' / f'{filename}.mm').read_text())
            command = ['clang++', '-std=c++17', '-O2', '-fobjc-arc', '-Wno-deprecated-declarations',
                       '-Itests/metal', '-Isrc/metal', '-Isrc/metal/compat']
            if precision == 'fp32':
                command.append('-DPRECISION_FLOAT')
            command += [str(case / f'{filename}.mm'), 'src/metal/runtime.mm', 'src/metal/launch.mm',
                        'src/metal/gemm.mm', '-framework', 'Metal', '-framework', 'Foundation',
                        '-framework', 'Accelerate', '-o', str(case / precision / filename)]
            subprocess.run(command, check=True)
            # Also prove that the individual bad fixture escapes the mutant.
            escaped_command = [str(case / precision / filename), control]
            if precision == 'fp32':
                escaped_command.append(str(case / 'escaped_fixture'))
            escaped = subprocess.run(escaped_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            gate = subprocess.run(['/bin/bash', 'tests/metal/check_goldwide_controls.sh', str(case)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (out / f'{name}.log').write_text(escaped.stdout + '\nGATE:\n' + gate.stdout)
            caught = escaped.returncode == 0 and gate.returncode == 1 and diagnostic in gate.stdout
            results[name] = {'bad_control_exit': escaped.returncode, 'gate_exit': gate.returncode,
                             'caught': caught}
            print(f'{name}: {results[name]}', flush=True)
            if not caught:
                raise RuntimeError(f'{name}: mutation escaped or failed for the wrong reason; see {out}')
    subprocess.run(['/bin/bash', 'tests/metal/check_goldwide_controls.sh', str(baseline)], check=True)
    results['restored'] = {'gate_exit': 0}
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
