"""Compile an isolated trainer mutation without editing the working source."""
from pathlib import Path
import os
import subprocess


def build_mutant(env_name, out, source, precision='--float', flags=''):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pufferl.cu').write_text(source)
    wrapper = Path('src/metal/pufferl_metal.cpp').read_text().replace(
        '#include "../pufferl.cu"', f'#include "{out}/pufferl.cu"')
    (out / 'wrapper.cpp').write_text(wrapper)
    cxx = out / 'cxx.py'
    cxx.write_text('#!/usr/bin/env python3\nimport subprocess, sys\n'
                   f'args = ["{out}/wrapper.cpp" if a == "src/metal/pufferl_metal.cpp" else a for a in sys.argv[1:]]\n'
                   'sys.exit(subprocess.call(["clang++", *args]))\n')
    cxx.chmod(0o755)
    binary = out / 'puffer'
    with (out / 'build.log').open('w') as log:
        subprocess.run(['./build.sh', env_name, str(binary), '--metal', *([precision] if precision else [])],
                       env=dict(os.environ, CXX=str(cxx), NVCC_EXTRA=flags),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return binary
