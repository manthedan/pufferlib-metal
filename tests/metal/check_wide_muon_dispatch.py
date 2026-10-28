#!/usr/bin/env python3
"""Assert actual production-wrapper dispatches using the existing kernel profiler.

Each process also runs one independent original scalar Muon reference (18 dispatches).
Fresh processes are required because the Muon environment flag is cached.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess


def check(binary, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    for enabled in (True, False):
        mode = 'enabled' if enabled else 'disabled'
        for width in (128, 256):
            env = dict(os.environ, PUF_METAL_MUON_WIDE='1',
                       PUF_METAL_MUON_WIDE_MMA=str(int(enabled)), PUF_METAL_KPROF='1')
            result = subprocess.run([str(binary), '--wrapper', str(width)], env=env,
                                    capture_output=True, text=True)
            log = result.stdout + result.stderr
            path = log_dir / f'{mode}_{width}.log'
            path.write_text(log)
            assert result.returncode == 0 and 'ALL PASS' in result.stdout, (path, log)
            counts = {name: int(count) for name, count in re.findall(
                r'^((?:wide_muon_|muon_wide_)\w+)\s+(\d+)\s', log, re.M)}
            expected = {
                'muon_wide_sumsq': 2, 'muon_wide_normalize': 2,
                'muon_wide_gram': 10, 'muon_wide_store': 2,
                'muon_wide_gram_buf': 5 if enabled else 10,
                'muon_wide_update': 5 if enabled else 10,
            }
            if enabled:
                expected.update(wide_muon_gram_buf=5, wide_muon_update=5)
            assert counts == expected, (
                f'wrapper kernel selection {mode} H={width}: got {counts}, '
                f'expected {expected}; log={path}')
            print(f'wrapper {mode} H={width}: exact results, guards and kernel counts PASS',
                  flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--log-dir', type=Path)
    args = parser.parse_args()
    check(args.binary.resolve(), args.log_dir or args.binary.parent / 'dispatch')
