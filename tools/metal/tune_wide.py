#!/usr/bin/env python3
"""Offline M4 tile search. Never changes source/defaults; all candidates must
pass bit-exact generic-reference tests before timing. Run from repository root.
The timing is diagnostic: validate selected tiles with clean end-to-end A/B.
Usage: python3 tools/metal/tune_wide.py [--hidden 128|256] [--precision f32|bf16]
       python3 tools/metal/tune_wide.py --ragged
Outputs: build/metal/tune_wide/{candidate}/{check,bench,build*}.log and results.json.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden', type=int, choices=[128, 256])
    parser.add_argument('--precision', choices=['f32', 'bf16'])
    parser.add_argument('--ragged', action='store_true')
    parser.add_argument('--output', default='build/metal/tune_wide')
    args = parser.parse_args()
    root = Path.cwd()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DEVELOPER_DIR=os.environ.get('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer'))
    for key in ['PUF_METAL_WIDE_LIB', 'PUF_METAL_FUSED_MINGRU',
                'PUF_METAL_FUSED_MINGRU_WIDE', 'PUF_METAL_GEMM_RAGGED']:
        env.pop(key, None)
    records = []
    start = time.monotonic()

    def run(cmd, log):
        with log.open('w') as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode == 0

    precisions = [args.precision] if args.precision else ['f32', 'bf16']
    if args.ragged:
        precisions = ['f32']  # this harness covers both storage types
    for precision in precisions:
        flags = ['-DPRECISION_FLOAT'] if precision == 'f32' else []
        base = out / precision
        base.mkdir(exist_ok=True)
        primary = base / 'puffer.metallib'
        if not run(['bash', 'tools/metal/build_metallib.sh', str(primary), *flags], base / 'primary.log'):
            raise SystemExit(f'Primary build failed: {base}/primary.log')
        if args.ragged:
            candidates = [(None, m, s, n) for m, n, s in itertools.product([16, 32, 64], [16, 32, 64], [64, 128])]
        else:
            candidates = []
            for h in ([args.hidden] if args.hidden else [128, 256]):
                choices = set(itertools.product([8, 16], [64, 128], [32, 64, 128]))
                choices.update((c, 256, 128) for c in [8, 16] if h == 128)
                choices.update((c, s, 256) for c in [8, 16] for s in [128, 256] if h == 256)
                candidates.extend((h, c, s, n) for c, s, n in sorted(choices)
                                  if c * 4 * h * (4 if precision == 'f32' else 2) <= 32768)
        for h, c, threads, n in candidates:
            tag = f'{precision}_h{h}_c{c}_s{threads}_n{n}'
            d = out / tag
            d.mkdir(exist_ok=True)
            if args.ragged:
                defs = [f'-DPUF_WIDE_TN_M_F32={c}', f'-DPUF_WIDE_TN_M_BF16={c}',
                        f'-DPUF_WIDE_TN_N={n}', f'-DPUF_WIDE_TN_THREADS={threads}']
                source = 'tests/metal/test_wide_gemm.mm'
                check_args, bench_args = [], ['--bench']
            else:
                defs = [f'-DPUF_WIDE_{h}_CHUNK={c}', f'-DPUF_WIDE_{h}_THREADS={threads}', f'-DPUF_WIDE_{h}_N={n}']
                source = 'tests/metal/test_wide_mingru.mm'
                check_args, bench_args = ['--check', str(h)], ['--bench', str(h)]
            commands = [
                ['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.0', '-O2',
                 '-fmetal-math-mode=safe', '-fmetal-math-fp32-functions=precise',
                 '-Isrc/metal/shaders', *flags, *defs, '-c', 'src/metal/shaders/wide.metal', '-o', str(d/'wide.air')],
                ['xcrun', '-sdk', 'macosx', 'metallib', str(d/'wide.air'), '-o', str(d/'puffer.wide.metallib')],
                ['clang++', '-std=c++17', '-O2', '-fobjc-arc', *flags, *defs,
                 '-Isrc/metal', '-Isrc/metal/compat', source, 'src/metal/runtime.mm', 'src/metal/gemm.mm',
                 '-framework', 'Metal', '-framework', 'Foundation',
                 f'-Wl,-sectcreate,__DATA,__metallib,{primary}', '-o', str(d/'test')]]
            ok = all(run(cmd, d/f'build{i}.log') for i, cmd in enumerate(commands))
            ok = ok and run([str(d/'test'), *check_args], d/'check.log')
            ok = ok and run([str(d/'test'), *bench_args], d/'bench.log')
            bench = (d/'bench.log').read_text() if ok else ''
            rec = dict(tag=tag, exact=ok, bench=bench, elapsed=time.monotonic()-start)
            if ok and not args.ragged:
                fwd = re.search(r'fused ([0-9.]+)', bench)
                bwd = re.search(r'no grad_input ([0-9.]+)', bench)
                rec['fwd_plus_bwd_ms'] = float(fwd[1]) + float(bwd[1])
            records.append(rec)
            (out/'results.json').write_text(json.dumps(records, indent=2))
            print(tag, 'PASS' if ok else 'FAIL', bench.strip(), flush=True)
    print(f'Search took {time.monotonic()-start:.1f} seconds. Results: {out}/results.json')
    if not args.ragged:
        for precision, h in itertools.product(precisions, [args.hidden] if args.hidden else [128, 256]):
            choices = [r for r in records if r['exact'] and r['tag'].startswith(f'{precision}_h{h}_')]
            if choices:
                winner = min(choices, key=lambda r: r['fwd_plus_bwd_ms'])
                print('Winner:', winner['tag'], winner['fwd_plus_bwd_ms'], 'ms')


if __name__ == '__main__':
    main()
